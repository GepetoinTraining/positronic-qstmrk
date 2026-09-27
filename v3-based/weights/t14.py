"""T14 — the weights through the atom kernel. Registered in ../prereg_T14.txt (sha256 073178c4...) before this file.

Every weight is a position: sign, odd address (odd part of the BF16 significand), 2-band (power of two).
value = address * 2^band, exactly. numpy holds integers only (int64 / int16 / object for big integers); the only floats
are in the layer's attention and norms (kept as they are, decision 2) and the one rounding that snaps a projection's
input onto the 2^-16 grid, identical on both paths.
Run from v3-based/weights:  python t14.py   (numpy)
"""
import os, sys, re, time, math, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tensor-atom", "data")
CUBES = os.environ.get("T14_CUBES", os.path.join(HERE, "cubes"))
OUT = []
def say(tag, ok, msg): OUT.append((tag, bool(ok))); print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}", flush=True)
def note(tag, msg): print(f"[{tag}] note  {msg}", flush=True)

L18 = ["self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_o_proj", "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj"]
NORMS = ["input_layernorm", "post_attention_layernorm", "self_attn_q_norm", "self_attn_k_norm"]
FILES = {f"L18_{n}": f"L18_{n}.npy" for n in L18 + NORMS}
FILES["L0_k_proj"] = "L0_k_proj.npy"
PROJ8 = [f"L18_{n}" for n in L18] + ["L0_k_proj"]

# ---------------- positions: BF16 bits <-> (address, band) ----------------
def bitlen_arr(v):
    out = np.zeros(v.shape, np.int32); v = v.copy()
    while (v > 0).any(): out += (v > 0); v >>= 1
    return out

def decompose(bits):
    """bits (uint16) -> address (int16: signed odd, 0 for zero), band (int16). value = address * 2^band."""
    b = bits.astype(np.int32)
    s, ef, f = b >> 15, (b >> 7) & 0xFF, b & 0x7F
    if (ef == 255).any(): raise ValueError("Inf/NaN pattern present")
    M = np.where(ef > 0, 128 + f, f)                       # integer significand
    e0 = np.where(ef > 0, ef - 134, -133)                  # value = M * 2^e0
    t = np.zeros_like(M); m = M.copy()
    for _ in range(8):
        k = (m > 0) & (m % 2 == 0); m[k] >>= 1; t[k] += 1
    addr = np.where(s == 1, -m, m).astype(np.int16)
    band = np.where(M > 0, e0 + t, s).astype(np.int16)     # a zero keeps its sign in the band field: +0 -> 0, -0 -> 1
    return addr, band

def rebuild(addr, band):
    a = addr.astype(np.int32); bd = band.astype(np.int32)
    neg = (a < 0) | ((a == 0) & (bd == 1)); odd = np.abs(a)
    sh = 8 - bitlen_arr(odd)                               # odd << sh lands in [128, 255]
    ef = bd - sh + 134
    normal = (odd > 0) & (ef > 0)
    sub = (odd > 0) & ~normal
    f = np.where(normal, (odd << np.where(normal, sh, 0)) - 128, 0)
    f = np.where(sub, odd << np.where(sub, np.maximum(bd + 133, 0), 0), f)
    efield = np.where(normal, ef, 0)
    return ((neg.astype(np.int32) << 15) | (efield << 7) | f).astype(np.uint16)

# ---------------- the seat: an ISIS3 cube, two bands ----------------
def write_cub(path, addr, band, label):
    L, S = addr.shape
    body = addr.astype("<i2").tobytes() + band.astype("<i2").tobytes()
    extra = "".join(f"    {k} = {v}\n" for k, v in label.items())
    def head(start):
        return (f"Object = IsisCube\n  Object = Core\n    StartByte = {start}\n    Format = BandSequential\n"
                f"    Group = Dimensions\n      Samples = {S}\n      Lines = {L}\n      Bands = 2\n    End_Group\n"
                f"    Group = Pixels\n      Type = SignedWord\n      ByteOrder = Lsb\n      Base = 0.0\n      Multiplier = 1.0\n    End_Group\n"
                f"  End_Object\n  Group = BandBin\n    Name = (Address, TwoBand)\n  End_Group\n"
                f"  Group = Atom\n    Rule = \"value = Address * 2^TwoBand; Address 0 is zero, its sign in TwoBand\"\n{extra}  End_Group\n"
                f"End_Object\nEnd\n")
    h = head(0); start = len(head(len(h) + 1).encode()) + 1
    with open(path, "wb") as fh: fh.write(head(start).encode().ljust(start - 1, b" ") + body)

def read_cub(path):
    raw = open(path, "rb").read(); txt = raw[:raw.index(b"\nEnd\n") + 5].decode()
    g = lambda k: int(re.search(rf"\b{k} = (\S+)", txt).group(1))
    start, S, L = g("StartByte"), g("Samples"), g("Lines")
    core = np.frombuffer(raw[start - 1:start - 1 + 4 * S * L], dtype="<i2")
    return core[:S * L].reshape(L, S).astype(np.int16), core[S * L:].reshape(L, S).astype(np.int16)

# ---------------- exact products ----------------
def join(parts):
    """parts: (int64 array, shift >= 0). An object array of exact Python integers: sum of part * 2^shift."""
    tot = None
    for arr, sh in parts:
        o = arr.astype(object) * (1 << sh)
        tot = o if tot is None else tot + o
    return tot

def headroom(xi, width):
    """Bits left in int64 for the weight side of one matmul: 62 - bits(max|x|) - bits(width)."""
    return 62 - int(np.abs(xi).max()).bit_length() - int(width).bit_length()

def collect(addr, band, xi):
    """The collector, from the seats alone: bands grouped in windows that fit int64; in each window, address shifted by
    its band offset, times x, summed in integers; windows joined by shifts. y = N * 2^emin exactly."""
    nz = addr != 0; emin, emax = int(band[nz].min()), int(band[nz].max())
    win = headroom(xi, addr.shape[1]) - 8
    assert win >= 1, "no room"
    parts = []; a64 = addr.astype(np.int64); b64 = band.astype(np.int64)
    for lo in range(emin, emax + 1, win):
        m = nz & (b64 >= lo) & (b64 < lo + win)
        if not m.any(): continue
        A = np.where(m, a64 << np.where(m, b64 - lo, 0), 0)
        parts.append((xi @ A.T, lo - emin))
    return join(parts), emin

def reference(bits, xi):
    """Independent exact reference, straight from the BF16 bits: every weight on one grid (significand shifted by its
    exponent above the smallest), cut into limbs that fit int64, each limb times x, limbs joined. y = N * 2^emin."""
    b = bits.astype(np.int64); s, ef, f = b >> 15, (b >> 7) & 0xFF, b & 0x7F
    M = np.where(ef > 0, 128 + f, f); e = np.where(ef > 0, ef, 1) - 134
    nz = M > 0; emin = int(e[nz].min())
    mag = np.where(nz, M, 0).astype(np.int64); sh = np.where(nz, e - emin, 0).astype(np.int64)
    limb = headroom(xi, bits.shape[1]) - 1
    assert limb >= 9
    width = 8 + int(sh.max())                                  # |q| = mag << sh has at most this many bits
    parts = []
    for lo in range(0, width, limb):                          # q is never formed: bits [lo, lo+limb) of |q|, one limb at a time
        d = sh - lo
        part = np.where(d >= 0, mag << np.clip(d, 0, limb), mag >> np.clip(-d, 0, 62))
        chunk = part & ((1 << limb) - 1)                       # d >= limb leaves zero low bits, so clipping d at limb is exact
        parts.append((xi @ np.where(s == 1, -chunk, chunk).T, lo))
    return join(parts), emin

def same(A, B):
    (NA, sA), (NB, sB) = A, B
    m = min(sA, sB)
    return bool(np.all(NA * (1 << (sA - m)) == NB * (1 << (sB - m))))

# ---------------- the layer (T6's hull, projections pluggable) ----------------
G, H, KV, D, EPS, THETA = 16, 32, 8, 128, 1e-6, 1e6
def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g
def spin(x, pos):
    inv = THETA ** (-np.arange(0, D, 2) / D); ang = pos[:, None] * inv[None]; c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
def snap(delta):
    q = np.round(delta * 2.0**G).astype(np.int64); return q, delta - q * 2.0**-G
def to_float(N, s):
    return np.array([math.ldexp(float(z), s) for z in N.ravel()]).reshape(N.shape)
def gridx(x): return np.round(x * 2.0**G).astype(np.int64)          # the one rounding into each projection

def layer(r, proj, norm):
    """proj(name, x) -> x W^T as float, from an exact integer product; norm(name) -> float gains."""
    x = rms(r * 2.0**-G, norm("input_layernorm")); T = x.shape[0]; pos = np.arange(T, dtype=np.float64)
    q = proj("self_attn_q_proj", x).reshape(T, H, D); k = proj("self_attn_k_proj", x).reshape(T, KV, D)
    v = proj("self_attn_v_proj", x).reshape(T, KV, D)
    q = spin(rms(q, norm("self_attn_q_norm")).transpose(1, 0, 2), pos); k = spin(rms(k, norm("self_attn_k_norm")).transpose(1, 0, 2), pos)
    k = np.repeat(k, H // KV, 0); v = np.repeat(v.transpose(1, 0, 2), H // KV, 0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(D); s = s + np.triu(np.full((T, T), -np.inf), 1)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    o = proj("self_attn_o_proj", (p @ v).transpose(1, 0, 2).reshape(T, H * D))
    da, ea = snap(o); r2 = r + da
    x2 = rms(r2 * 2.0**-G, norm("post_attention_layernorm")); g = proj("mlp_gate_proj", x2)
    dm, em = snap(proj("mlp_down_proj", (g / (1 + np.exp(-g))) * proj("mlp_up_proj", x2)))
    return da, ea, dm, em

def bits_to_f64(bits): return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

# ---------------- exact readings ----------------
def profiles(addr, band, axis):
    """Band, address and full energy profiles along `axis` (0: one value per row), exact Python integers."""
    nz = addr != 0; other = 1 - axis
    emin = int(band[nz].min()); bands = np.unique(band[nz])
    a2 = addr.astype(np.int64) ** 2
    Bp, Fp = [], []
    for e in bands:
        m = nz & (band == e)
        Bp.append((m.sum(axis=other).astype(np.int64), 2 * (int(e) - emin)))
        Fp.append((np.where(m, a2, 0).sum(axis=other), 2 * (int(e) - emin)))
    return join(Bp), a2.sum(axis=other).astype(object), join(Fp)

def excess(p, T):
    S = p.reshape(-1, T).sum(axis=0)
    return T * sum(v * v for v in S) - sum(p) ** 2

def line(p, T, rng, n=100):
    real = excess(p, T)
    beat = sum(real > excess(p[rng.permutation(len(p))], T) for _ in range(n))
    return beat == n, beat

# =================================================================================================
if __name__ == "__main__":
    os.makedirs(CUBES, exist_ok=True)
    raw = {k: np.load(os.path.join(DATA, f)) for k, f in FILES.items()}
    dup = np.array_equal(np.load(os.path.join(DATA, "L18_k_proj.npy")), raw["L18_self_attn_k_proj"])
    note("data", f"{len(raw)} tensors; L18_k_proj.npy equals L18_self_attn_k_proj.npy: {dup}")

    # ---- G1 ----
    g1 = True; negzero = 0
    for k, bits in raw.items():
        a, b = decompose(bits); g1 &= np.array_equal(rebuild(a, b), bits); negzero += int(((a == 0) & (b == 1)).sum())
    say("G1", g1, f"all {len(raw)} tensors ({sum(v.size for v in raw.values()):,} weights) decompose into sign, odd address and "
        f"2-band and rebuild their BF16 bits exactly; no Inf/NaN; -0 patterns: {negzero}")

    # ---- G2 ----
    g2 = True; seated_bytes = raw_bytes = 0
    for k, bits in raw.items():
        a, b = decompose(bits); a2 = a if a.ndim == 2 else a[None, :]; b2 = b if b.ndim == 2 else b[None, :]
        path = os.path.join(CUBES, f"{k}.cub")
        write_cub(path, a2, b2, {"Source": f"\"Qwen3-4B {k}\"", "Shape": f"({', '.join(map(str, bits.shape))})"})
        ra, rb = read_cub(path)
        g2 &= hashlib.sha256(rebuild(ra, rb).reshape(bits.shape).tobytes()).hexdigest() == hashlib.sha256(bits.tobytes()).hexdigest()
        seated_bytes += os.path.getsize(path); raw_bytes += bits.nbytes
    say("G2", g2, "every tensor seated as an ISIS3 cube (bands Address and TwoBand, SignedWord, the rule in the label); read back, "
        "its BF16 bytes rebuild with the same sha256")
    note("G2", f"seated {seated_bytes:,} bytes against raw {raw_bytes:,} ({seated_bytes / raw_bytes:.3f}x): two 16-bit bands per weight")

    cube = {}
    def seat(k):
        if k not in cube: cube[k] = read_cub(os.path.join(CUBES, f"{k}.cub"))
        return cube[k]

    # ---- G3 ----
    rng = np.random.default_rng(14); g3 = True; t = time.time()
    for n in L18:
        k = f"L18_{n}"; xi = np.round(rng.normal(0, 1.0, (16, raw[k].shape[1])) * 2.0**G).astype(np.int64)
        ok = same(collect(*seat(k), xi), reference(raw[k], xi)); g3 &= ok
        note("G3", f"{k:26s} {str(raw[k].shape):13s} collector == reference at all {16 * raw[k].shape[0]:,} cells: {ok}")
    say("G3", g3, f"y = W x for all 7 L18 projections, from the cubes by the collector, equals the independent exact reference "
        f"(from the BF16 bits on one grid) at every output cell ({time.time() - t:.0f}s)")

    # ---- G4 ----
    rng = np.random.default_rng(11)
    ctx = np.round(rng.normal(0, 1.0, (24, 2560)) * 2.0**G).astype(np.int64)
    prompt = np.round(rng.normal(0, 1.0, (8, 2560)) * 2.0**G).astype(np.int64)
    r = np.vstack([ctx, prompt])
    t = time.time()
    ref = layer(r.copy(), lambda n, x: (lambda N_s: to_float(N_s[0], N_s[1] - G))(reference(raw[f"L18_{n}"], gridx(x))),
                lambda n: bits_to_f64(raw[f"L18_{n}"]))
    t_ref = time.time() - t
    readings_seats = {k: (lambda ab: (ab[0], ab[1]) if raw[k].ndim == 2 else (ab[0][0], ab[1][0]))(seat(k)) for k in FILES}
    shapes = {k: raw[k].shape for k in FILES}
    del raw                                               # the weights leave reach
    real_load = np.load
    def refuse(*a, **kw): raise PermissionError("weights are out of reach")
    np.load = refuse
    t = time.time()
    seated = layer(r.copy(), lambda n, x: (lambda N_s: to_float(N_s[0], N_s[1] - G))(collect(*seat(f"L18_{n}"), gridx(x))),
                   lambda n: bits_to_f64(rebuild(*seat(f"L18_{n}")).reshape(shapes[f"L18_{n}"])))
    t_seat = time.time() - t
    np.load = real_load
    same4 = all(np.array_equal(a, b) for a, b in zip(ref, seated))
    say("G4", same4, f"one real layer (Qwen3-4B L18: attention and MLP, 32 tokens) run from the cubes alone, np.load refusing, "
        f"gives the same two deltas and the same two rounding receipts as the run from the raw weights, bit for bit "
        f"(ref {t_ref:.0f}s, seated {t_seat:.0f}s)")
    note("G4", f"attention delta max |{int(np.abs(ref[0]).max())}| and MLP delta max |{int(np.abs(ref[2]).max())}| grid steps; "
         f"largest rounding {max(float(np.abs(ref[1]).max()), float(np.abs(ref[3]).max())):.2e} (half step {2.0**-(G+1):.2e})")

    # ---- readings ----
    rng = np.random.default_rng(1414)
    res = {}
    t = time.time()
    for k in PROJ8:
        a, b = readings_seats[k]
        for axis in (0, 1):
            Bp, Ap, Fp = profiles(a, b, axis)
            N = len(Bp); Ts = [T for T in (2**i for i in range(1, 14)) if T < N and N % T == 0]
            for name, p in (("band", Bp), ("address", Ap), ("full", Fp)):
                for T in Ts: res[(k, axis, name, T)] = line(p, T, rng)
    note("P", f"period readings done ({time.time() - t:.0f}s)")

    p1_trip = ["L18_self_attn_q_proj", "L18_self_attn_k_proj", "L0_k_proj"]
    p1 = all(res[(k, 0, "band", 128)][0] for k in p1_trip) and not any(res[(k, 0, "address", 128)][0] for k in p1_trip)
    say("P1", p1, "rows, T = 128, band profile vs address profile (shuffles beaten of 100): " + "; ".join(
        f"{k}: band {res[(k, 0, 'band', 128)][1]}, address {res[(k, 0, 'address', 128)][1]}, full {res[(k, 0, 'full', 128)][1]}" for k in p1_trip))

    def xor_read(k, n_ctrl=200):
        a, b = readings_seats[k]; Bp, _, _ = profiles(a, b, 0); P = Bp.reshape(-1, 128)
        Dx = [sum(sum((P[h, d] - P[h, d ^ (1 << bb)]) ** 2 for d in range(128)) for h in range(P.shape[0])) for bb in range(7)]
        rr = np.random.default_rng(64); ctrl = []
        for _ in range(n_ctrl):
            tot = 0
            for h in range(P.shape[0]):
                pm = rr.permutation(128)
                tot += 2 * sum((P[h, pm[2 * i]] - P[h, pm[2 * i + 1]]) ** 2 for i in range(64))
            ctrl.append(tot)
        return Dx, ctrl
    p2 = True; lines = []
    for k in ("L18_self_attn_q_proj", "L18_self_attn_k_proj", "L0_k_proj"):
        Dx, ctrl = xor_read(k)
        order = sorted(range(7), key=lambda i: Dx[i])
        below = sum(Dx[6] < c for c in ctrl)
        ok = order[0] == 6 and below == len(ctrl)
        if k != "L0_k_proj": p2 &= ok
        lines.append(f"{k}: xor bits ranked most-shared first {[2**i for i in order]}, xor64 below {below}/200 random pairings")
    say("P2", p2, "inside each head, band profile, D_b = sum (B_d - B_(d xor 2^b))^2: " + "; ".join(lines) + " (L0 reported only)")

    print("\n[P3] exploratory: T with a line (band / address / full), rows (axis 0) and columns (axis 1)")
    for k in PROJ8:
        for axis in (0, 1):
            cells = []
            for name in ("band", "address", "full"):
                Ts = sorted(T for (kk, ax, nm, T) in res if kk == k and ax == axis and nm == name and res[(kk, ax, nm, T)][0])
                cells.append(f"{name}: {Ts if Ts else '-'}")
            print(f"  {k:26s} axis {axis}  " + " | ".join(cells))

    print("\n[P4] exploratory: addresses +-1 (the weight is an exact power of two) against 1 in 128 of normal values")
    for k in PROJ8:
        a, b = readings_seats[k]
        odd = np.abs(a.astype(np.int32))
        normal = (a != 0) & (b.astype(np.int32) - (8 - bitlen_arr(odd)) + 134 > 0)     # the exponent field is nonzero
        n1 = int(((a == 1) | (a == -1)).sum()); nn = int(normal.sum())
        print(f"  {k:26s} +-1: {n1:,} of {nn:,} normal  (1/128 would be {nn / 128:,.1f}; ratio {n1 * 128 / nn:.3f})")

    print(f"\n{sum(ok for _, ok in OUT)}/{len(OUT)} passed  (gates G1-G4; predictions P1, P2)")
