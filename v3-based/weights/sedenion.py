"""Anchors as bands: a weight pixel is a sedenion.

Cube: Bands = 16 (the anchors e0..e15), Lines = output rows, Samples = input / 16. Pixel (line i, sample p) is the sedenion
whose band a holds weight W[i, 16p + a], as its packed code (pack.py: band byte, address byte, Null = zero).
Inference reads band 0 of conj(w) * x, pixel by pixel, summed: the crossing of the weight and the activation.
The signs come from the kernel's XOR-sign table (legos/kernel.py), not from an assumption.
Run from v3-based/weights:  python sedenion.py   (numpy)
"""
import os, sys, re, time, hashlib
import numpy as np
import t14, pack
sys.path.insert(0, os.path.join(t14.HERE, "..", "legos"))
from kernel import SIGN, mul as smul

CONJ = [1] + [-1] * 15
# band 0 of conj(w) * x receives exactly the pairs a xor b = 0, i.e. b = a, each with sign CONJ[a] * SIGN[a][a]
DIAG = [CONJ[a] * SIGN[a][a] for a in range(16)]
ONLY_DIAG = all((a ^ b == 0) == (a == b) for a in range(16) for b in range(16))

def write_sed(path, code, floor, label):
    L, S16 = code.shape; S = S16 // 16
    planes = code.reshape(L, S, 16).transpose(2, 0, 1)            # band a = the a-th anchor of every pixel
    extra = "".join(f"    {k} = {v}\n" for k, v in label.items())
    names = ", ".join(f"e{a}" for a in range(16))
    def head(start):
        return (f"Object = IsisCube\n  Object = Core\n    StartByte = {start}\n    Format = BandSequential\n"
                f"    Group = Dimensions\n      Samples = {S}\n      Lines = {L}\n      Bands = 16\n    End_Group\n"
                f"    Group = Pixels\n      Type = UnsignedWord\n      ByteOrder = Lsb\n      Base = 0.0\n      Multiplier = 1.0\n    End_Group\n"
                f"  End_Object\n  Group = BandBin\n    Name = ({names})\n  End_Group\n"
                f"  Group = Atom\n    BandFloor = {floor}\n    Pixel = \"sedenion: band a holds W[line, 16 * sample + a]\"\n"
                f"    Rule = \"code = (band - BandFloor) * 256 + sign * 128 + (odd - 1) / 2; value = +-odd * 2^band; Null (0) is zero\"\n"
                f"    Read = \"y[line] = sum over samples of band 0 of conj(w) * x\"\n{extra}  End_Group\nEnd_Object\nEnd\n")
    h = head(0); start = len(head(len(h) + 1).encode()) + 1
    with open(path, "wb") as fh: fh.write(head(start).encode().ljust(start - 1, b" ") + np.ascontiguousarray(planes).astype("<u2").tobytes())

def read_sed(path):
    raw = open(path, "rb").read(); txt = raw[:raw.index(b"\nEnd\n") + 5].decode()
    g = lambda k: int(re.search(rf"\b{k} = (-?\d+)", txt).group(1))
    start, S, L, B, floor = g("StartByte"), g("Samples"), g("Lines"), g("Bands"), g("BandFloor")
    return np.frombuffer(raw[start - 1:start - 1 + 2 * S * L * B], dtype="<u2").reshape(B, L, S), floor

def add_exact(parts):
    """parts: (N object array, s) meaning N * 2^s. Their exact sum on the smallest s."""
    m = min(s for _, s in parts); tot = None
    for N, s in parts:
        o = N * (1 << (s - m)); tot = o if tot is None else tot + o
    return tot, m

def read_band0(planes, floor, xi):
    """y = sum over pixels of band 0 of conj(w) * x: for each anchor a, DIAG[a] * (plane_a . x_a), all exact."""
    parts = []
    for a in range(16):
        plane = np.ascontiguousarray(planes[a])
        if not (plane != 0).any(): continue
        N, s = pack.collect_packed(plane, floor, np.ascontiguousarray(xi[:, a::16]))
        parts.append((N * DIAG[a], s))
    return add_exact(parts)

if __name__ == "__main__":
    CUBES = os.environ.get("T14_CUBES", os.path.join(t14.HERE, "cubes"))
    SED = CUBES + "_sedenion"; os.makedirs(SED, exist_ok=True)
    raw = {k: np.load(os.path.join(t14.DATA, f)) for k, f in t14.FILES.items()}
    OUT = []
    def say(tag, ok, msg): OUT.append(bool(ok)); print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}", flush=True)

    say("S0", ONLY_DIAG and DIAG == [1] * 16, f"from the kernel's table: band 0 of conj(w) * x takes only a = b, with sign "
        f"CONJ[a] * SIGN[a][a] = {sorted(set(DIAG))} for all 16 anchors: band 0 is the crossing of w and x, anchor by anchor")

    ok = True; size = rawb = 0
    for k, bits in raw.items():
        code, floor = pack.pack(bits); c2 = code if code.ndim == 2 else code[None, :]
        path = os.path.join(SED, f"{k}.cub")
        write_sed(path, c2, floor, {"Source": f"\"Qwen3-4B {k}\"", "Shape": f"({', '.join(map(str, bits.shape))})"})
        planes, rf = read_sed(path)
        back_code = planes.transpose(1, 2, 0).reshape(c2.shape)
        back = t14.rebuild(*pack.unpack(back_code, rf)).reshape(bits.shape)
        ok &= hashlib.sha256(back.tobytes()).hexdigest() == hashlib.sha256(bits.tobytes()).hexdigest()
        size += os.path.getsize(path); rawb += bits.nbytes
    say("S1", ok, f"all 12 tensors as 16-band cubes (anchors e0..e15 as bands, each pixel a sedenion of 16 weights), read back "
        f"sha-identical; {size:,} bytes against raw {rawb:,} ({size / rawb:.4f}x)")

    # the whole product exists: on a sample, every band of conj(w) * x from the kernel's own multiplication
    planes, floor = read_sed(os.path.join(SED, "L18_self_attn_q_proj.cub"))
    rng = np.random.default_rng(16)
    xs = np.round(rng.normal(0, 1.0, 2560) * 2.0**t14.G).astype(np.int64)
    a, b = pack.unpack(planes.transpose(1, 2, 0).reshape(4096, 2560), floor)
    nz = a != 0; bmin = int(b[nz].min())
    Wint = [[int(a[i, j]) << (int(b[i, j]) - bmin) if a[i, j] else 0 for j in range(2560)] for i in range(64)]
    full_ok = True; other = 0
    for i in range(64):
        acc = [0] * 16
        for p in range(160):
            w = Wint[i][16 * p:16 * p + 16]; x = [int(v) for v in xs[16 * p:16 * p + 16]]
            prod = smul([CONJ[c] * w[c] for c in range(16)], x)
            acc = [s + t for s, t in zip(acc, prod)]
        full_ok &= acc[0] == sum(Wint[i][j] * int(xs[j]) for j in range(2560))
        other += sum(1 for v in acc[1:] if v != 0)
    say("S2", full_ok, f"on 64 q_proj rows, the full sedenion product conj(w) * x, pixel by pixel through the kernel's mul, "
        f"summed: band 0 equals the row's dot product exactly; the other 15 bands carry {other} nonzero sums of {64 * 15} "
        f"(what the crossing also makes, and inference does not read)")

    rng = np.random.default_rng(14); g3 = True; t = time.time()
    for n in t14.L18:
        k = f"L18_{n}"; xi = np.round(rng.normal(0, 1.0, (16, raw[k].shape[1])) * 2.0**t14.G).astype(np.int64)
        g3 &= t14.same(read_band0(*read_sed(os.path.join(SED, f"{k}.cub")), xi), t14.reference(raw[k], xi))
    say("S3", g3, f"all 7 projections read as band 0 of the sedenion products, from the 16-band cubes, equal the exact reference "
        f"at every cell ({time.time() - t:.0f}s)")

    rng = np.random.default_rng(11)
    r = np.vstack([np.round(rng.normal(0, 1.0, (24, 2560)) * 2.0**t14.G).astype(np.int64),
                   np.round(rng.normal(0, 1.0, (8, 2560)) * 2.0**t14.G).astype(np.int64)])
    ref = t14.layer(r.copy(), lambda n, x: (lambda N: t14.to_float(N[0], N[1] - t14.G))(t14.reference(raw[f"L18_{n}"], t14.gridx(x))),
                    lambda n: t14.bits_to_f64(raw[f"L18_{n}"]))
    shapes = {k: v.shape for k, v in raw.items()}; del raw
    real_load = np.load; np.load = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("weights out of reach"))
    cache = {}
    def sd(n):
        if n not in cache: cache[n] = read_sed(os.path.join(SED, f"L18_{n}.cub"))
        return cache[n]
    def norm(n):
        planes, fl = sd(n); L = planes.shape[1]
        return t14.bits_to_f64(t14.rebuild(*pack.unpack(planes.transpose(1, 2, 0).reshape(L, -1), fl))).reshape(shapes[f"L18_{n}"])
    t = time.time()
    seated = t14.layer(r.copy(), lambda n, x: (lambda N: t14.to_float(N[0], N[1] - t14.G))(read_band0(*sd(n), t14.gridx(x))), norm)
    np.load = real_load
    say("S4", all(np.array_equal(u, v) for u, v in zip(ref, seated)),
        f"the real layer, every projection read as band 0 of sedenion products from the 16-band cubes alone (np.load refusing), "
        f"equals the raw-weight run bit for bit ({time.time() - t:.0f}s)")
    print(f"\n{sum(OUT)}/{len(OUT)} passed")
