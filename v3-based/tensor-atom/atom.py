"""Tensor atom: object (exact weights) + PVL label (the rule), stored as an ISIS3 cube.

Floor, stated plainly:
  object   = the weight matrix on an exact integer grid (bf16 values are m * 2^e; one shared 2^E makes them all integers)
  harmonic = the pattern the data repeats with period T along one axis (discovered, never assumed)
  residue  = object - harmonic, exact integers
  label    = the rule: axis, T, scale (Base/Multiplier), the discovered line's strength. Everything else is rebuilt from it.
"""
import numpy as np, re

# ---------- exact grid ----------
def bf16_to_f64(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

def to_grid(bits):
    """bf16 bits -> (int64 integers, E) with value = integer * 2^-E, exactly."""
    v = bf16_to_f64(bits); nz = v[v != 0]
    E = int(-np.floor(np.log2(np.abs(nz))).min()) + 8          # enough bits for the smallest mantissa
    q = np.round(v * 2.0**E).astype(np.int64)
    assert np.array_equal(q.astype(np.float64) * 2.0**-E, v), "grid not exact"
    return q, E

def from_grid(q, E):
    v = (q.astype(np.float64) * 2.0**-E).astype(np.float32)
    return (v.view(np.uint32) >> 16).astype(np.uint16)

# ---------- discovering the rule ----------
def line_strength(profile, k, w=6):
    P = np.abs(np.fft.rfft(profile - profile.mean()))**2
    nb = np.r_[P[max(1, k - w):k], P[k + 1:k + w + 1]]
    return P[k] / np.median(nb)

def discover(q, controls=100, seed=0):
    """Blind search: the strongest spectral line on either axis, against position-shuffled controls."""
    rng = np.random.default_rng(seed); best = None
    for axis in (0, 1):
        prof = (q.astype(np.float64)**2).sum(axis=1 - axis); N = len(prof)
        scores = [(line_strength(prof, k), k) for k in range(3, N // 2 - 6)]
        s, k = max(scores)
        ctrl = max(max(line_strength(p, kk) for kk in range(3, N // 2 - 6)) for p in
                   (rng.permutation(prof) for _ in range(controls)))
        if s > ctrl and (best is None or s > best["line"]):
            best = dict(axis=axis, repeats=k, length=N, line=float(s), shuffle_max=float(ctrl))
    return best          # None means: no rule on this tensor; the atom is all residue

def split(q, axis, T):
    """harmonic = integer mean of the T-periodic slices along `axis`; residue = q - harmonic. Exact."""
    x = np.moveaxis(q, axis, 0); N = x.shape[0]; assert N % T == 0, "period must tile the axis"
    s = x.reshape(N // T, T, *x.shape[1:])
    h = s.sum(axis=0) // (N // T)                                   # integer floor mean
    H = np.broadcast_to(h, s.shape).reshape(x.shape)
    return np.moveaxis(H, 0, axis), np.moveaxis(x - H, 0, axis)

# ---------- ISIS3 cube ----------
def write_cub(path, bits, label, layout="BandSequential", tile=(128, 128), order=None):
    """bits: (lines, samples) uint16. layout BandSequential or Tile. order: optional row permutation (stored in label)."""
    L, S = bits.shape
    rows = bits if order is None else bits[order]
    if layout == "Tile":
        tl, ts = tile; body = b"".join(rows[i:i + tl, j:j + ts].astype("<u2").tobytes()
                                       for i in range(0, L, tl) for j in range(0, S, ts))
    else:
        body = rows.astype("<u2").tobytes()
    extra = "".join(f"    {k} = {v}\n" for k, v in label.items())
    perm = "" if order is None else "    RowOrder = (" + ", ".join(map(str, order)) + ")\n"
    def head(start):
        core = (f"Object = IsisCube\n  Object = Core\n    StartByte = {start}\n    Format = {layout}\n"
                + (f"    TileSamples = {tile[1]}\n    TileLines = {tile[0]}\n" if layout == "Tile" else "")
                + f"    Group = Dimensions\n      Samples = {S}\n      Lines = {L}\n      Bands = 1\n    End_Group\n"
                  f"    Group = Pixels\n      Type = UnsignedWord\n      ByteOrder = Lsb\n      Base = 0.0\n      Multiplier = 1.0\n    End_Group\n"
                  f"  End_Object\n  Group = TensorAtom\n{extra}{perm}  End_Group\nEnd_Object\nEnd\n")
        return core
    h = head(0); start = len(head(len(h) + 1).encode()) + 1; h = head(start)
    h = h.encode().ljust(start - 1, b" ")
    open(path, "wb").write(h + body)

def read_cub(path):
    raw = open(path, "rb").read(); txt = raw[:raw.index(b"\nEnd\n") + 5].decode()
    g = lambda k: re.search(rf"\b{k} = (\S+)", txt).group(1)
    start, S, L, fmt = int(g("StartByte")), int(g("Samples")), int(g("Lines")), g("Format")
    body = np.frombuffer(raw[start - 1:start - 1 + 2 * S * L], dtype="<u2")
    if fmt == "Tile":
        tl, ts = int(g("TileLines")), int(g("TileSamples")); out = np.empty((L, S), np.uint16); p = 0
        for i in range(0, L, tl):
            for j in range(0, S, ts):
                h, w = min(tl, L - i), min(ts, S - j); out[i:i + h, j:j + w] = body[p:p + h * w].reshape(h, w); p += h * w
    else:
        out = body.reshape(L, S).copy()
    m = re.search(r"RowOrder = \(([^)]*)\)", txt)
    if m:
        order = np.array([int(t) for t in m.group(1).split(",")]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
        out = out[inv]
    label = dict(re.findall(r"^\s{4}(\w+) = (.+)$", txt.split("Group = TensorAtom")[1], re.M))
    return out, label
