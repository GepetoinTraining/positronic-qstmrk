"""
Viscous CLM (w_t = w*Hw + nu*w_xx) in exact rationals, built from the morning's pieces.
- Analytic signal f = Hw + i w = sum_k a_k e^{-ikx}   (only one-signed harmonics: the un-collapse)
- The i is factored out as an ADDRESS:  a_k = i^k * b_k,  b_k rational.  {1,i,-1,-i} is a lookup.
- Then:  b_k' = 1/2 * sum_{j+l=k} b_j b_l  -  nu k^2 b_k        (b_1(0) = 1)
- Viscosity as one rational per harmonic (heavy teeth damp harder):  divide by (1 + nu dt k^2)
- Rung / fixed point: self-similar <=> ratios b_{k+1}/b_k equal across k (one profile + one rung rho).
  Singularity reaches the real line <=> rho reaches 1 (under-half x over-half = 1). No log, no limit.
Inviscid exact answer for checking: b_k = (t/2)^(k-1), blow-up at T* = 2.
"""
from fractions import Fraction as F
import time, sys

def run(nu, K=32, dt=F(1,64), tmax=F(5,2), report_every=16, dyadic=False):
    b = [F(0)]*(K+1); b[1] = F(1)
    damp = [1 + nu*dt*k*k for k in range(K+1)]
    keep = [1 - nu*dt*k*k for k in range(K+1)]   # explicit form: stays dyadic if nu*dt is
    if dyadic: assert all(0 < x < 2 for x in keep) or True
    t, n = F(0), 0
    rows, crossed = [], None
    t0 = time.time()
    while t < tmax:
        nl = [F(0)]*(K+1)
        for k in range(2, K+1):
            s = F(0)
            for j in range(1, k):
                s += b[j]*b[k-j]
            nl[k] = s/2
        if dyadic: b = [b[k]*keep[k] + dt*nl[k] for k in range(K+1)]
        else:      b = [(b[k] + dt*nl[k]) / damp[k] for k in range(K+1)]
        t += dt; n += 1
        # rung: ratios over the mid band, where truncation hasn't bitten
        ks = range(K//4, 3*K//4)
        rat = [b[k+1]/b[k] for k in ks if b[k] != 0 and b[k+1] != 0]
        if len(rat) < len(ks):  # band not yet filled: harmonics propagate one index per step
            continue
        rho_hi, rho_lo = max(rat), min(rat)
        if crossed is None and rho_lo >= 1: crossed = t
        digits = max(x.numerator.bit_length()+x.denominator.bit_length() for x in b[1:])
        if n % report_every == 0 or (crossed == t):
            rows.append((t, rho_lo, rho_hi, digits, time.time()-t0))
        if time.time()-t0 > 80: rows.append(("TIMEOUT", t, digits)); break
    return rows, crossed, b

def show(nu, dyadic=False):
    print(f"\n=== nu = {nu}  ({'explicit dyadic' if dyadic else 'implicit 1/(1+x)'}) ===")
    print("   t       rho_min     rho_max     spread      max bits     secs")
    rows, crossed, b = run(nu, dyadic=dyadic)
    for r in rows:
        if r[0] == "TIMEOUT": print("  TIMEOUT at t =", r[1], "digits", r[2]); continue
        t, lo, hi, d, s = r
        print(f"  {float(t):5.3f}   {float(lo):.6f}   {float(hi):.6f}   {float(hi-lo):.2e}   {d:8d}   {s:6.1f}")
    print("  rung rho reaches 1 (singularity touches the real line) at t =", crossed, "" if crossed is None else f"= {float(crossed)}")
    return b

if __name__ == "__main__":
    import sys
    show(F(sys.argv[1]), dyadic=len(sys.argv)>2)
