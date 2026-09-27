"""The packed cube: one UnsignedWord pixel per weight, the rule and the band floor in the label, the tensor in its own order.

  code = (band - floor) * 256 + sign * 128 + (odd - 1) / 2        high byte: the band (envelope); low byte: the address
  code 0 is ISIS's Null special pixel: the zero, the one value with no address. Codes stay inside ISIS's valid range
  for UnsignedWord (band offset 1..254, so 256 <= code <= 65279).
Doubling a value is +256 on its code; the address byte is untouched.
Run from v3-based/weights:  python pack.py   (numpy; uses t14's seat, reference and layer)
"""
import os, re, time, hashlib
import numpy as np
import t14

def pack(bits):
    a, b = t14.decompose(bits)
    nz = a != 0
    floor = int(b[nz].min()) - 1
    off = b.astype(np.int32) - floor
    assert off[nz].max() <= 254, "band span too wide for one byte"
    odd = np.abs(a.astype(np.int32)); sign = (a < 0).astype(np.int32)
    code = np.where(nz, off * 256 + sign * 128 + (odd - 1) // 2, 0).astype(np.uint16)
    return code, floor

def unpack(code, floor):
    c = code.astype(np.int32); nz = c != 0
    odd = 2 * (c & 127) + 1
    addr = np.where(nz, np.where((c >> 7) & 1, -odd, odd), 0).astype(np.int16)
    band = np.where(nz, (c >> 8) + floor, 0).astype(np.int16)
    return addr, band

def write_packed(path, code, floor, label):
    L, S = code.shape
    extra = "".join(f"    {k} = {v}\n" for k, v in label.items())
    def head(start):
        return (f"Object = IsisCube\n  Object = Core\n    StartByte = {start}\n    Format = BandSequential\n"
                f"    Group = Dimensions\n      Samples = {S}\n      Lines = {L}\n      Bands = 1\n    End_Group\n"
                f"    Group = Pixels\n      Type = UnsignedWord\n      ByteOrder = Lsb\n      Base = 0.0\n      Multiplier = 1.0\n    End_Group\n"
                f"  End_Object\n  Group = Atom\n    BandFloor = {floor}\n"
                f"    Rule = \"code = (band - BandFloor) * 256 + sign * 128 + (odd - 1) / 2; value = +-odd * 2^band; Null (0) is zero\"\n"
                f"{extra}  End_Group\nEnd_Object\nEnd\n")
    h = head(0); start = len(head(len(h) + 1).encode()) + 1
    with open(path, "wb") as fh: fh.write(head(start).encode().ljust(start - 1, b" ") + code.astype("<u2").tobytes())

def read_packed(path):
    raw = open(path, "rb").read(); txt = raw[:raw.index(b"\nEnd\n") + 5].decode()
    g = lambda k: int(re.search(rf"\b{k} = (-?\d+)", txt).group(1))
    start, S, L, floor = g("StartByte"), g("Samples"), g("Lines"), g("BandFloor")
    return np.frombuffer(raw[start - 1:start - 1 + 2 * S * L], dtype="<u2").reshape(L, S), floor

def collect_packed(code, floor, xi):
    """The collector straight on the codes: a band window is a code range (band is the high byte)."""
    nz = code != 0; c = code.astype(np.int64)
    hi_byte = c >> 8; odd = 2 * (c & 127) + 1; addr = np.where((c >> 7) & 1, -odd, odd)
    omin, omax = int(hi_byte[nz].min()), int(hi_byte[nz].max())
    win = t14.headroom(xi, code.shape[1]) - 8
    parts = []
    for lo in range(omin, omax + 1, win):
        m = nz & (c >= lo << 8) & (c < (lo + win) << 8)
        if not m.any(): continue
        A = np.where(m, addr << np.where(m, hi_byte - lo, 0), 0)
        parts.append((xi @ A.T, lo - omin))
    return t14.join(parts), omin + floor

if __name__ == "__main__":
    CUBES = os.environ.get("T14_CUBES", os.path.join(t14.HERE, "cubes"))
    PACKED = CUBES + "_packed"; os.makedirs(PACKED, exist_ok=True)
    raw = {k: np.load(os.path.join(t14.DATA, f)) for k, f in t14.FILES.items()}
    OUT = []
    def say(tag, ok, msg): OUT.append(bool(ok)); print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}", flush=True)

    ok = True; sizes = 0; rawb = 0; floors = {}; spans = {}
    for k, bits in raw.items():
        code, floor = pack(bits); c2 = code if code.ndim == 2 else code[None, :]
        path = os.path.join(PACKED, f"{k}.cub")
        write_packed(path, c2, floor, {"Source": f"\"Qwen3-4B {k}\"", "Shape": f"({', '.join(map(str, bits.shape))})"})
        rc, rf = read_packed(path)
        back = t14.rebuild(*unpack(rc, rf)).reshape(bits.shape)
        ok &= hashlib.sha256(back.tobytes()).hexdigest() == hashlib.sha256(bits.tobytes()).hexdigest()
        sizes += os.path.getsize(path); rawb += bits.nbytes; floors[k] = rf
        nzc = rc[rc != 0] >> 8; spans[k] = (int(nzc.min()), int(nzc.max()), len(np.unique(nzc)))
    say("K1", ok, f"all 12 tensors packed one UnsignedWord per weight in their own order, rule and floor in the label; read back "
        f"sha-identical. {sizes:,} bytes against raw {rawb:,} ({sizes / rawb:.4f}x; the rest is the label)")
    for k, (lo, hi, n) in spans.items(): print(f"      {k:30s} band floor {floors[k]:5d}  band offsets {lo}..{hi}  ({n} distinct)")

    # doubling is +256 on the code, address byte untouched
    code, floor = read_packed(os.path.join(PACKED, "L18_self_attn_q_proj.cub"))
    c = code.astype(np.int32); sel = (c != 0) & ((c >> 8) < 254)
    bits = raw["L18_self_attn_q_proj"].reshape(code.shape)
    doubled = t14.rebuild(*unpack((c + 256 * sel).astype(np.uint16), floor))
    v = t14.bits_to_f64(bits); v2 = t14.bits_to_f64(doubled)
    say("K2", np.array_equal(v2[sel], 2 * v[sel]) and np.array_equal(doubled[~sel], bits[~sel]),
        f"+256 on the code doubles the value exactly on all {int(sel.sum()):,} q_proj weights, the address byte untouched")

    rng = np.random.default_rng(14); g3 = True; t = time.time()
    for n in t14.L18:
        k = f"L18_{n}"; xi = np.round(rng.normal(0, 1.0, (16, raw[k].shape[1])) * 2.0**t14.G).astype(np.int64)
        g3 &= t14.same(collect_packed(*read_packed(os.path.join(PACKED, f"{k}.cub")), xi), t14.reference(raw[k], xi))
    say("K3", g3, f"all 7 projections from the packed cubes, the collector reading band windows as code ranges, equal the exact "
        f"reference at every cell ({time.time() - t:.0f}s)")

    rng = np.random.default_rng(11)
    r = np.vstack([np.round(rng.normal(0, 1.0, (24, 2560)) * 2.0**t14.G).astype(np.int64),
                   np.round(rng.normal(0, 1.0, (8, 2560)) * 2.0**t14.G).astype(np.int64)])
    ref = t14.layer(r.copy(), lambda n, x: (lambda N: t14.to_float(N[0], N[1] - t14.G))(t14.reference(raw[f"L18_{n}"], t14.gridx(x))),
                    lambda n: t14.bits_to_f64(raw[f"L18_{n}"]))
    shapes = {k: v.shape for k, v in raw.items()}; del raw
    real_load = np.load; np.load = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("weights out of reach"))
    cache = {}
    def pk(n):
        if n not in cache: cache[n] = read_packed(os.path.join(PACKED, f"L18_{n}.cub"))
        return cache[n]
    t = time.time()
    seated = t14.layer(r.copy(), lambda n, x: (lambda N: t14.to_float(N[0], N[1] - t14.G))(collect_packed(*pk(n), t14.gridx(x))),
                       lambda n: t14.bits_to_f64(t14.rebuild(*unpack(*pk(n))).reshape(shapes[f"L18_{n}"])))
    np.load = real_load
    say("K4", all(np.array_equal(a, b) for a, b in zip(ref, seated)),
        f"the real layer from the packed cubes alone (np.load refusing) equals the raw-weight run bit for bit ({time.time() - t:.0f}s)")
    print(f"\n{sum(OUT)}/{len(OUT)} passed")
