"""The rule of a sedenion weight pixel, from the kernel's table (legos/kernel.py), exact.
For a pixel w and a symbolic activation x: Y_c(x) = band c of conj(w) x (16 linear forms; Y_0 = w . x), and the defect
delta_w(x) = |w|^2 |x|^2 - sum_c Y_c(x)^2 = x^T D x.
Result 27 Sep 2026, on 4 real Qwen3-4B q_proj pixels and 1,000 random integer sedenions: D is zero on the inner block
(x_0..x_7) and on the outer block (x_8..x_15); it lives only in the cross block, delta_w(x) = 2 x_inner^T C_w x_outer,
rank 8 every time (C_w rank 4), trace 0; identically zero on octonions; L_w has no kernel on the pixels read.
Run from v3-based/weights:  python defect.py
"""
import os, sys, random
from fractions import Fraction as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legos"))
from kernel import SIGN

CONJ = [1] + [-1] * 15
def L(w): return [[CONJ[b ^ c] * SIGN[b ^ c][b] * w[b ^ c] for b in range(16)] for c in range(16)]
def D(w):
    Lw = L(w); n2 = sum(v * v for v in w)
    return [[(n2 if i == j else 0) - sum(Lw[c][i] * Lw[c][j] for c in range(16)) for j in range(16)] for i in range(16)]
def rank(M):
    M = [[F(v) for v in r] for r in M]; rk = 0
    for c in range(len(M[0])):
        piv = next((r for r in range(rk, len(M)) if M[r][c] != 0), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]
        for r in range(len(M)):
            if r != rk and M[r][c] != 0:
                f = M[r][c] / M[rk][c]; M[r] = [x - f * y for x, y in zip(M[r], M[rk])]
        rk += 1
    return rk

rng = random.Random(5); blocks = True; ranks = {}
for _ in range(1000):
    w = [rng.randint(-50, 50) for _ in range(16)]; d = D(w)
    blocks &= all(d[i][j] == 0 for i in range(8) for j in range(8)) and all(d[i][j] == 0 for i in range(8, 16) for j in range(8, 16))
    rk = rank(d); ranks[rk] = ranks.get(rk, 0) + 1
octo = all(all(v == 0 for r in D(w) for v in r) for w in ([rng.randint(-50, 50) for _ in range(8)] + [0] * 8 for _ in range(200)))
print(f"1000 random sedenions: defect only in the inner x outer block: {blocks}; rank of the defect form: {ranks}")
print(f"200 random octonions: defect identically 0: {octo}")
