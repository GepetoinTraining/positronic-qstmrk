"""Read the 15 bands: the imaginary part of sum_p conj(w_p) * x_p, across one real layer, exact.

For every projection of Qwen3-4B L18, on the inputs the real layer feeds it (the 8 prompt rows of the 32-token run):
  Y_c[row, t] = sum_a CONJ[a] * SIGN[a][a xor c] * (plane_a . x_(a xor c))     c = 0..15, from the 16-band cubes.
Band 0 is what inference reads. Readings (exact integers; controls declared here, before the run):
  R1  energy E_c = sum Y_c^2 per band, against E_0.
  R2  the pixel grouping: 10 random permutations of the input axis applied to W and x together (band 0 unchanged,
      checked exactly). Real imaginary energy against the 10: above all, below all, or inside. Per band too.
  R3  the comparator per pixel on a sample (64 rows, last token): sign of |w|^2 |x|^2 - |conj(w) x|^2.
Run from v3-based/weights:  python read15.py   (numpy; needs the sedenion cubes from sedenion.py)
"""
import os, sys, time
import numpy as np
import t14, pack, sedenion as sd
from sedenion import CONJ, SIGN, smul

SED = os.environ.get("T14_CUBES", os.path.join(t14.HERE, "cubes")) + "_sedenion"
SGN = [[CONJ[a] * SIGN[a][a ^ c] for a in range(16)] for c in range(16)]    # sign of anchor a's term in band c

def load(n): return sd.read_sed(os.path.join(SED, f"L18_{n}.cub"))

def bands(planes, floor, xi):
    """All 16 bands of sum_p conj(w_p) x_p, exact: returns (Y list of object arrays (T x out), scale s): value = Y * 2^s."""
    T = xi.shape[0]
    X = np.ascontiguousarray(np.concatenate([xi[:, b::16] for b in range(16)], axis=0))      # (16T, S): the 16 x-planes
    win = t14.headroom(X, planes.shape[2]) - 8
    per = []
    for a in range(16):
        c = planes[a].astype(np.int64); nz = c != 0
        if not nz.any(): per.append(None); continue
        hb = c >> 8; odd = 2 * (c & 127) + 1; addr = np.where((c >> 7) & 1, -odd, odd)
        omin, omax = int(hb[nz].min()), int(hb[nz].max()); parts = []
        for lo in range(omin, omax + 1, win):
            m = nz & (c >= lo << 8) & (c < (lo + win) << 8)
            if m.any(): parts.append((X @ np.where(m, addr << np.where(m, hb - lo, 0), 0).T, lo - omin))
        per.append((t14.join(parts), omin))
    smin = min(p[1] for p in per if p)
    Y = [None] * 16
    for a in range(16):
        if per[a] is None: continue
        N, om = per[a]; N = N * (1 << (om - smin))
        for cc in range(16):
            blk = N[(a ^ cc) * T:((a ^ cc) + 1) * T] * SGN[cc][a]
            Y[cc] = blk if Y[cc] is None else Y[cc] + blk
    return Y, smin + floor

def energy(Y): return [sum(int(v) * int(v) for v in y.ravel()) for y in Y]

def regroup(planes, perm):
    L, S = planes.shape[1], planes.shape[2]
    code = planes.transpose(1, 2, 0).reshape(L, 16 * S)[:, perm]
    return np.ascontiguousarray(code.reshape(L, S, 16).transpose(2, 0, 1))

if __name__ == "__main__":
    t0 = time.time()
    # the real layer's inputs to every projection, captured on the seated run (sedenion cubes, band 0)
    rng = np.random.default_rng(11)
    r = np.vstack([np.round(rng.normal(0, 1.0, (24, 2560)) * 2.0**t14.G).astype(np.int64),
                   np.round(rng.normal(0, 1.0, (8, 2560)) * 2.0**t14.G).astype(np.int64)])
    shapes = {k: v.shape for k, v in ((k, np.load(os.path.join(t14.DATA, f), mmap_mode="r")) for k, f in t14.FILES.items())}
    cache, xin = {}, {}
    def cub(n):
        if n not in cache: cache[n] = load(n)
        return cache[n]
    def proj(n, x):
        xi = t14.gridx(x); xin[n] = xi
        N, s = sd.read_band0(*cub(n), xi); return t14.to_float(N, s - t14.G)
    def norm(n):
        planes, fl = cub(n); L = planes.shape[1]
        return t14.bits_to_f64(t14.rebuild(*pack.unpack(planes.transpose(1, 2, 0).reshape(L, -1), fl))).reshape(shapes[f"L18_{n}"])
    t14.layer(r.copy(), proj, norm)
    print(f"[in] inputs captured from the real layer ({time.time() - t0:.0f}s); reading the 8 prompt rows", flush=True)

    print("\n[R1/R2] energy per band (E_c / E_0, shown to 4 places; verdicts exact); imaginary energy I = sum E_1..E_15")
    rr = np.random.default_rng(15)
    summary = {}
    for n in t14.L18:
        t = time.time(); planes, fl = cub(n); xi = xin[n][-8:]
        Y, s = bands(planes, fl, xi); E = energy(Y); I = sum(E[1:])
        ctrl_I, ctrl_E, band0_same = [], [], True
        for _ in range(10):
            perm = rr.permutation(xi.shape[1])
            Yc, sc = bands(regroup(planes, perm), fl, np.ascontiguousarray(xi[:, perm]))
            band0_same &= t14.same((Yc[0], sc), (Y[0], s))
            Ec = energy(Yc); ctrl_E.append(Ec); ctrl_I.append(sum(Ec[1:]))       # same scale s as the real read
        below = sum(I < ci for ci in ctrl_I)
        verdict = "below all 10" if below == 10 else "above all 10" if below == 0 else f"inside ({below} of 10 above it)"
        perband = [sum(E[c] < ce[c] for ce in ctrl_E) for c in range(16)]
        rank = sorted(range(1, 16), key=lambda c: -E[c])
        summary[n] = (verdict, band0_same)
        print(f"  {n:18s} I/E_0 = {I / E[0]:8.4f}   real I vs 10 regroupings: {verdict}; band 0 identical under regrouping: {band0_same}")
        print(f"  {'':18s} bands by energy: {rank}")
        print(f"  {'':18s} per band, regroupings with more energy (0 = real is highest, 10 = lowest): "
              + " ".join(f"{c}:{perband[c]}" for c in range(1, 16)) + f"   ({time.time() - t:.0f}s)", flush=True)

    print("\n[R3] comparator per pixel, sign of |w|^2 |x|^2 - |conj(w) x|^2 (64 rows, last prompt token)")
    for n in t14.L18:
        planes, fl = cub(n); L, S = planes.shape[1], planes.shape[2]
        code = planes.transpose(1, 2, 0).reshape(L, 16 * S)
        a, b = pack.unpack(code[:64], fl); nzm = a != 0; bmin = int(b[nzm].min())
        xrow = [int(v) for v in xin[n][-1]]
        pos = neg = zer = 0
        for i in range(64):
            w = [int(a[i, j]) << (int(b[i, j]) - bmin) if a[i, j] else 0 for j in range(16 * S)]
            for p in range(S):
                wp = w[16 * p:16 * p + 16]; xp = xrow[16 * p:16 * p + 16]
                d = sum(v * v for v in wp) * sum(v * v for v in xp) - sum(v * v for v in smul([CONJ[c] * wp[c] for c in range(16)], xp))
                pos += d > 0; neg += d < 0; zer += d == 0
        print(f"  {n:18s} pixels {64 * S:6,}: stretched (|w||x| < |w x|) {neg:6,}  shrunk {pos:6,}  exact (octonion-like) {zer:6,}", flush=True)
    print(f"\ndone ({time.time() - t0:.0f}s)")
