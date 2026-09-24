# Constantin-Lax-Majda vortex-stretching model, w_t = w * H(w), in exact rationals.
# f = Hw + i*w obeys f_t = f^2/2  ->  f(t) = 2 f0 / (2 - t f0)   (a Mobius map; no limit taken)
# w0 = cos x, Hw0 = sin x, points on the circle taken at rational parameter u:
#   cos = (1-u^2)/(1+u^2), sin = 2u/(1+u^2)   (exact rational points; u=1 is x = pi/2)
from fractions import Fraction as F

def point(u):
    d = 1 + u*u
    return (1 - u*u)/d, 2*u/d            # (cos, sin) exact

def f_at(u, t):                           # returns (Hw, w) exact, via complex Mobius map
    c, s = point(u)
    a, b = s, c                           # f0 = a + i b
    # 2 f0 / (2 - t f0) = 2(a+ib)(2 - t a + i t b) / ((2-ta)^2 + (tb)^2)
    re_d, im_d = 2 - t*a, t*b
    den = re_d*re_d + im_d*im_d
    num_re = 2*(a*re_d - b*im_d)
    num_im = 2*(a*im_d + b*re_d)
    return num_re/den, num_im/den, den

# 1) the ODE identity, checked exactly: d/dt[2f0/(2-t f0)] == f^2/2 at sample points
def check(u, t):
    c, s = point(u); f0 = complex  # not used; do it in Fractions
    a, b = s, c
    # derivative: 2 f0^2 / (2 - t f0)^2 ; f^2/2 with f = 2f0/(2-tf0) gives same expression symbolically
    # verify numerically-exact by computing both as Gaussian rationals
    def cmul(x, y): return (x[0]*y[0]-x[1]*y[1], x[0]*y[1]+x[1]*y[0])
    def cdiv(x, y):
        d = y[0]*y[0]+y[1]*y[1]; return ((x[0]*y[0]+x[1]*y[1])/d, (x[1]*y[0]-x[0]*y[1])/d)
    f0 = (a, b); den = (2 - t*a, -t*b)
    f = cdiv((2*a, 2*b), den)
    lhs = cdiv(cmul((2*a,2*b), f0), cmul(den, den))          # 2 f0^2/(2-tf0)^2
    rhs = cmul(f, f); rhs = (rhs[0]/2, rhs[1]/2)
    return lhs == rhs

ok = all(check(F(p, 97), t) for p in range(1, 200, 7) for t in [F(1,3), F(1), F(19,10)])
print("ODE identity exact at all sampled points:", ok)

# 2) blow-up: only where the denominator (2 - t f0) hits zero -> w0 = 0 and t = 2/Hw0
c, s = point(F(1)); print("at u=1: w0 =", c, " Hw0 =", s, " -> T* =", F(2)/s)

# 3) approach T* on a rational ladder t = 2 - 2^-k; scan rational points near u=1
print("\n   t            (T*-t)*max|w|    core width (u-units)")
for k in range(1, 13):
    t = 2 - F(1, 2**k)
    best, arg = F(0), None
    q = 2**(k+6)
    for p in range(q - 2**6, q + 2**6 + 1):      # window scales with the core
        u = F(p, q)
        _, w, _ = f_at(u, t)
        if abs(w) > best: best, arg = abs(w), u
    print(f"  2-2^-{k:<3}   {float((2-t)*best):.6f}         {float(abs(arg-1)):.3e}")
