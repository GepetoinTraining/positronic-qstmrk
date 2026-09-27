"""The layer read from rules: the correct way, lean.

The rule of a projection is its seated code matrix (one UnsignedWord per weight, band byte and address byte, in the
model's own order, from the 16-band sedenion cube) plus the kernel's XOR-sign table. Nothing else is stored.
  band c of sum_p conj(w_p) x_p  =  W . x^(c),   x^(c)_j = s(c, j mod 16) * x_(j xor c),   s(c, a) = CONJ[a] SIGN[a][a xor c]
so inference (band 0) is W . x, and all 16 bands are one product of the rule with 16 table-moved copies of x.
Exact by bound: band windows of 18 bits and 14-bit activation limbs keep every operand and every partial sum an integer
below 2^53, so the multiply-add hardware (BLAS) adds no rounding (T6f: the release comes from staying on the grid,
whatever arithmetic carries it). Windows and limbs are joined by shifts. No division anywhere. Gate: bit for bit
against the pure-integer reference.
Run from v3-based/weights:  python rules.py   (numpy)
"""
import os, time
import numpy as np
import t14, pack, sedenion as sd
from sedenion import CONJ, SIGN

SED = os.environ.get("T14_CUBES", os.path.join(t14.HERE, "cubes")) + "_sedenion"
LIMB, XBITS, CHUNK = 14, 27, 1024

def seat(path):
    planes, floor = sd.read_sed(path); L, S = planes.shape[1], planes.shape[2]
    return np.ascontiguousarray(planes.transpose(1, 2, 0).reshape(L, 16 * S)), floor

def apply(rule, xi):
    """Exact x . W^T from the rule: N * 2^s."""
    codes, floor = rule
    width = codes.shape[1]
    wb = 53 - LIMB - int(width).bit_length() - 8 + 1            # band window: |R| < 2^(8 + wb - 1)
    limbs, rest, k = [], xi.copy(), 0                            # xi = sum limb_k 2^(14k); every limb |.| < 2^14
    while True:
        low = rest & ((1 << LIMB) - 1); rest = rest >> LIMB
        limbs.append((low.astype(np.float64), LIMB * k)); k += 1
        if (rest == 0).all(): break
        if ((rest == -1) | (rest == 0)).all(): limbs.append((rest.astype(np.float64), LIMB * k)); break
    hb_all = codes >> 8; nzall = codes != 0
    omin, omax = int(hb_all[nzall].min()), int(hb_all[nzall].max())
    out = []
    for r0 in range(0, codes.shape[0], CHUNK):
        c = codes[r0:r0 + CHUNK].astype(np.int64); nz = c != 0; hb = c >> 8
        odd = 2 * (c & 127) + 1; addr = np.where((c >> 7) & 1, -odd, odd)
        parts = []
        for lo in range(omin, omax + 1, wb):
            m = nz & (hb >= lo) & (hb < lo + wb)
            if not m.any(): continue
            R = np.where(m, addr << np.where(m, hb - lo, 0), 0).astype(np.float64).T
            for xl, sh in limbs: parts.append(((xl @ R).astype(np.int64), lo - omin + sh))
        out.append(t14.join(parts))
    return np.concatenate(out, axis=1), omin + floor

SGN_X = [np.array([CONJ[a] * SIGN[a][a ^ c] for a in range(16)], np.int64) for c in range(16)]
def moved(xi, c):
    return xi[:, np.arange(xi.shape[1]) ^ c] * np.tile(SGN_X[c], xi.shape[1] // 16)

def reference_chunked(bits, xi):
    parts = [t14.reference(bits[r0:r0 + CHUNK], xi) for r0 in range(0, bits.shape[0], CHUNK)]
    m = min(s for _, s in parts)
    return np.concatenate([N * (1 << (s - m)) for N, s in parts], axis=1), m

if __name__ == "__main__":
    OUT = []
    def say(tag, ok, msg): OUT.append(bool(ok)); print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}", flush=True)
    rng = np.random.default_rng(11)
    r = np.vstack([np.round(rng.normal(0, 1.0, (24, 2560)) * 2.0**t14.G).astype(np.int64),
                   np.round(rng.normal(0, 1.0, (8, 2560)) * 2.0**t14.G).astype(np.int64)])

    # the reference, from the raw weights, pure integers (one tensor in memory at a time)
    t = time.time()
    shapes = {k: np.load(os.path.join(t14.DATA, f), mmap_mode="r").shape for k, f in t14.FILES.items()}
    def ref_proj(n, x):
        bits = np.load(os.path.join(t14.DATA, f"L18_{n}.npy"))
        N, s = reference_chunked(bits, t14.gridx(x)); del bits
        return t14.to_float(N, s - t14.G)
    ref = t14.layer(r.copy(), ref_proj, lambda n: t14.bits_to_f64(np.load(os.path.join(t14.DATA, f"L18_{n}.npy"))))
    t_ref = time.time() - t

    # the rules, seated once from the cubes; the weights out of reach from here
    t = time.time()
    rules = {n: seat(os.path.join(SED, f"L18_{n}.cub")) for n in t14.L18}
    norms = {n: t14.bits_to_f64(t14.rebuild(*pack.unpack(*seat(os.path.join(SED, f"L18_{n}.cub"))))).reshape(shapes[f"L18_{n}"]) for n in t14.NORMS}
    t_seat = time.time() - t
    real_load = np.load; np.load = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("weights out of reach"))
    xin = {}
    def proj(n, x):
        xi = t14.gridx(x); xin[n] = xi
        N, s = apply(rules[n], xi); return t14.to_float(N, s - t14.G)
    t = time.time(); out = t14.layer(r.copy(), proj, lambda n: norms[n]); t_run = time.time() - t
    np.load = real_load
    say("R1", all(np.array_equal(a, b) for a, b in zip(ref, out)),
        f"the real layer read from the rules (np.load refusing) equals the pure-integer reference bit for bit. "
        f"Seat once {t_seat:.1f}s; layer {t_run:.1f}s (32 tokens); reference {t_ref:.0f}s")

    t = time.time(); ok = True; lines = []
    for n in t14.L18:
        xi = xin[n]; T = xi.shape[0]
        N, s = apply(rules[n], np.concatenate([moved(xi, c) for c in range(16)], axis=0))
        ok &= bool(np.all(N[:T] == apply(rules[n], xi)[0]))
        E = [sum(int(v) * int(v) for v in N[c * T:(c + 1) * T].ravel()) for c in range(16)]
        lines.append(f"{n.replace('self_attn_', '').replace('mlp_', '')} {sum(E[1:]) / E[0]:.2f}")
    say("R2", ok, f"all 16 bands of all 7 projections on the layer's own inputs: one product per projection (the rule with 16 "
        f"table-moved activations), band 0 equal to the inference read ({time.time() - t:.1f}s). Imaginary / band 0: " + ", ".join(lines))
    print(f"\n{sum(OUT)}/{len(OUT)} passed")
