"""The value counts of the local Qwen3-1.7B weights (layers 0-13 and part of 14, u16 as positions).
Run from v3-based/weights:  python counts.py   (numpy)
Result 27 Sep 2026: 710,997,760 weights in 159 tensors; 7,099 distinct values, no zero; 3,894 magnitudes, 3,205 of them
present with both signs (6,410 values cancel in +- pairs, 689 carry one sign); the 3,894^2 = 15,163,236 magnitude
pairs land on 402,411 distinct product magnitudes (odd * odd, band + band).
"""
import os, numpy as np
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ARCHIVE-2026-09-25-mistaken", "weights")
seen = np.zeros(65536, bool); n = 0
for f in os.listdir(d):
    a = np.fromfile(os.path.join(d, f), dtype="<u2"); n += a.size; seen |= np.bincount(a, minlength=65536) > 0
v = np.nonzero(seen)[0]
mag = v & 0x7FFF; both = np.intersect1d(v[v < 0x8000], mag[v >= 0x8000])
print(f"{n:,} weights; {v.size:,} distinct values (zero present: {bool(seen[0] or seen[0x8000])}); "
      f"{np.unique(mag).size:,} magnitudes, {both.size:,} with both signs")
ef = (mag >> 7) & 0xFF; fr = mag & 0x7F
M = np.where(ef > 0, 128 + fr, fr).astype(np.int64); e = np.where(ef > 0, ef.astype(np.int64) - 134, -133)
tz = np.zeros_like(M); m = M.copy()
for _ in range(8): k = (m % 2 == 0) & (m > 0); m[k] //= 2; tz[k] += 1
um = np.unique(np.stack([m, e + tz], 1), axis=0); o, b = um[:, 0], um[:, 1]
key = (b[:, None] + b[None, :]).ravel() * (1 << 17) + (o[:, None] * o[None, :]).ravel()
print(f"{um.shape[0] ** 2:,} magnitude pairs -> {np.unique(key).size:,} distinct product magnitudes")
