# Test: is the "stretch" a translation in the inverse-twin coordinate g = 1/f ?
# Closed form (inviscid CLM): f_t = f^2/2  ->  g_t = -1/2 exactly, at every point: a rigid shift, no stretch.
# Here we do NOT use the closed form: we step the coefficients (the kernel's scheme), then evaluate
# f at exact rational points on the unit circle (Gaussian rationals), invert, and measure g(t)-g(0)+t/2.
from fractions import Fraction as F

import sys
K, dt = 32, F(1, int(sys.argv[1]))
def step(b, nu):
    nl = [F(0)]*(K+1)
    for k in range(2, K+1):
        nl[k] = sum(b[j]*b[k-j] for j in range(1, k)) / 2
    return [b[k]*(1 - nu*dt*k*k) + dt*nl[k] for k in range(K+1)]

def cmul(x, y): return (x[0]*y[0]-x[1]*y[1], x[0]*y[1]+x[1]*y[0])
def cinv(x):
    d = x[0]*x[0]+x[1]*x[1]; return (x[0]/d, -x[1]/d)
IPOW = [(F(1),F(0)), (F(0),F(1)), (F(-1),F(0)), (F(0),F(-1))]   # i^k as a lookup, never computed

def f_at(b, w):            # f = sum i^k b_k w^k, w = e^{-ix} an exact rational point on the circle
    acc, wk = (F(0), F(0)), (F(1), F(0))
    for k in range(1, K+1):
        wk = cmul(wk, w)
        term = cmul(IPOW[k % 4], (b[k]*wk[0], b[k]*wk[1]))
        acc = (acc[0]+term[0], acc[1]+term[1])
    return acc

pts = []
for u in [F(0), F(1,7), F(1,3), F(1,2), F(2,3), F(1), F(3,2), F(3), F(-1,2), F(-2)]:
    d = 1+u*u; pts.append(((1-u*u)/d, 2*u/d))   # exact (cos, sin)

for nu in [F(sys.argv[2])]:
    b = [F(0)]*(K+1); b[1] = F(1)
    g0 = [cinv(f_at(b, w)) for w in pts]
    print(f"\nnu = {nu}")
    print("   t     max |g(t) - g(0) + t/2| over the points   (0 = pure translation, no stretch)")
    for n in range(1, int(sys.argv[1])+1):
        b = step(b, nu)
        if n == int(sys.argv[1]):
            t = n*dt
            dev = 0.0
            for w, g in zip(pts, g0):
                gt = cinv(f_at(b, w))
                r = (gt[0]-g[0]+t/2, gt[1]-g[1])
                dev = max(dev, float(abs(r[0])+abs(r[1])))   # print only
            print(f"  {float(t):4.2f}   {dev:.3e}")
