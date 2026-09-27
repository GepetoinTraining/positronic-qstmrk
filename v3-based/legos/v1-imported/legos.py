"""Legos kernel: the capabilities of legos.tex (26 Sep 2026) as exact code.

Python standard library only. Integers and Fractions (the half is an ingredient); no float anywhere in the kernel.
Section marks (C1..C11, A.x, B.x) point at the paper. test_legos.py checks every claim against this file.
"""
from fractions import Fraction as F
from math import isqrt, gcd
from array import array

HALF = F(1, 2)

# ---------- ingredient 2: Z[r], r^2 = r + 1 (root of x^3 - 2x - 1 = (x+1)(x^2-x-1); phi is never written) ----------
class R:
    __slots__ = ("p", "q")
    def __init__(s, p, q=0): s.p, s.q = p, q
    def __add__(s, o): return R(s.p + o.p, s.q + o.q)
    def __sub__(s, o): return R(s.p - o.p, s.q - o.q)
    def __neg__(s): return R(-s.p, -s.q)
    def __mul__(s, o): return R(s.p * o.p + s.q * o.q, s.p * o.q + s.q * o.p + s.q * o.q)
    def __eq__(s, o): return s.p == o.p and s.q == o.q
    def __hash__(s): return hash((s.p, s.q))
    def __repr__(s): return f"({s.p}{'+' if s.q >= 0 else '-'}{abs(s.q)}r)"
    def conj(s): return R(s.p + s.q, -s.q)                       # Galois: r -> 1 - r = -1/r
    def norm(s): return s.p * s.p + s.p * s.q - s.q * s.q         # s * conj(s), a rational
    def sign(s):
        """Exact sign of p + q*phi = ((2p+q) + q*sqrt5)/2: compare squares, never take the root."""
        a, b = 2 * s.p + s.q, s.q
        if a >= 0 and b >= 0: return 0 if a == 0 and b == 0 else 1
        if a <= 0 and b <= 0: return -1
        d = a * a - 5 * b * b
        return (1 if d > 0 else -1) if a > 0 else (1 if d < 0 else -1)
    def __lt__(s, o): return (s - o).sign() < 0

ZERO, ONE, r = R(0), R(1), R(0, 1)
RINV, RINV2, R2, SQRT5 = R(-1, 1), R(2, -1), R(1, 1), R(-1, 2)   # r^-1, r^-2, r^2, sqrt5 (C5)

def rpow(n):
    """r^n for any integer n, by the relation only (r^-1 = r - 1)."""
    x = ONE
    for _ in range(abs(n)): x = x * (r if n > 0 else RINV)
    return x

# ---------- quaternions over Z[r] ----------
def qmul(x, y):
    a1, b1, c1, d1 = x; a2, b2, c2, d2 = y
    return (a1*a2 - b1*b2 - c1*c2 - d1*d2, a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2, a1*d2 + b1*c2 - c1*b2 + d1*a2)
def qconj(x): return (x[0], -x[1], -x[2], -x[3])
def qdot(x, y): return x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3]

h = R(HALF)
IDENT = (ONE, ZERO, ZERO, ZERO)
ICOSIAN_GENS = [(ZERO, ONE, ZERO, ZERO), (ZERO, ZERO, ONE, ZERO), (h, h, h, h), (r * h, h, RINV * h, ZERO)]   # i, j, half-diagonal, (r + i + r^-1 j)/2

# ---------- B.4 Legos: the auto-function (verbatim) ----------
def lego(pieces, snap, identity):
    items = {identity: 0}; order = [identity]; front = [identity]
    while front:                          # snap pieces on until nothing new appears
        nxt = []
        for a in front:
            for p in pieces:
                b = snap(a, p)
                if b not in items: items[b] = len(order); order.append(b); nxt.append(b)
        front = nxt
    table = [[items[snap(a, b)] for b in order] for a in order]   # once: the madness
    look = lambda a, b: order[table[items[a]][items[b]]]          # after: syntax
    return order, table, look

turn = lambda a, b: (a + b) % 1          # snapping two turns = one after the other

# ---------- C1 / B.1 exact circle ----------
def circle(n):
    """n exact turns by (4+3i)/5 from (1,0): yields (a, b, d) with a^2 + b^2 = d^2."""
    a, b, d = 1, 0, 1
    for _ in range(n):
        a, b, d = 4*a - 3*b, 3*a + 4*b, 5*d
        yield a, b, d

# ---------- C2 division as inversion, in homogeneous integers ----------
def invert(X, w, rho):
    """x -> rho * x/|x|^2 for the point X/w (rho = radius^2, a Fraction). Returns (X', w'). No division:
    rho * (X/w) / (|X|^2/w^2) = rho * w * X / |X|^2."""
    n2 = sum(t * t for t in X)
    return tuple(rho.numerator * w * t for t in X), rho.denominator * n2

def same_point(A, B):
    """Projective equality by cross-multiplication. (0,...,0; 0) is no point at all."""
    u, v = tuple(A[0]) + (A[1],), tuple(B[0]) + (B[1],)
    if not any(u) or not any(v): return False
    return all(u[i] * v[j] == u[j] * v[i] for i in range(len(u)) for j in range(i + 1, len(u)))

# ---------- C3 counting that keeps its marks ----------
def marks(N):
    """Count up once to N, marking every number with its smallest prime atom (Eratosthenes).
    Larger primes are laid first so each smaller prime overwrites them; unmarked numbers are atoms themselves."""
    s = isqrt(N); comp = bytearray(s + 1); small = []
    for p in range(2, s + 1):
        if not comp[p]:
            small.append(p); comp[p*p::p] = b"\x01" * len(range(p*p, s + 1, p))
    spf = array("I", range(N + 1))
    for p in reversed(small):
        spf[p*p::p] = array("I", [p]) * len(range(p*p, N + 1, p))
    return spf

def atoms(n, spf):
    out = []
    while n > 1:
        p = spf[n]; out.append(p); n //= p
    return out

# ---------- B.3 Cayley-Dickson (verbatim, on integers) and B.5 cubed indices ----------
def cd(x, y):
    n = len(x)
    if n == 1: return [x[0]*y[0]]
    hh = n//2; a, b, c, d = x[:hh], x[hh:], y[:hh], y[hh:]
    conj = lambda v: [v[0]] + [-t for t in v[1:]]
    ac, db = cd(a, c), cd(conj(d), b); da, bc = cd(d, a), cd(b, conj(c))
    return [p - q for p, q in zip(ac, db)] + [p + q for p, q in zip(da, bc)]

def e(i, n=16): return [1 if k == i else 0 for k in range(n)]
def sq(v): return sum(t * t for t in v)            # |v|^2: the comparator is read on squares, exactly

SIGN = [[sum(cd(e(a), e(b))) for b in range(16)] for a in range(16)]   # once
def mul(x, y):                                                         # after
    out = [0]*16
    for a in range(16):
        if x[a]:
            for b in range(16):
                if y[b]: out[a ^ b] += SIGN[a][b] * x[a] * y[b]
    return out

# ---------- C11 the three-sevens tree: exact rotations in Q(zeta_28) ----------
# Phi_28(x) = x^12 - x^10 + x^8 - x^6 + x^4 - x^2 + 1, so x^12 = x^10 - x^8 + x^6 - x^4 + x^2 - 1.
def _reduce(c):
    c = list(c) + [0] * max(0, 23 - len(c))
    for k in range(len(c) - 1, 11, -1):
        t = c[k]
        if t:
            c[k] = 0; c[k-2] += t; c[k-4] -= t; c[k-6] += t; c[k-8] -= t; c[k-10] += t; c[k-12] -= t
    return tuple(c[:12])
def _mono(n): return _reduce([0] * (n % 28) + [1])
def _padd(a, b): return tuple(x + y for x, y in zip(a, b))
def _psub(a, b): return tuple(x - y for x, y in zip(a, b))
def _pmul(a, b):
    c = [0] * 23
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y: c[i + j] += x * y
    return _reduce(c)
_Z12 = (0,) * 12
TWO_COS = _padd(_mono(2), _mono(26))            # 2cos(pi/7) = zeta^2 + zeta^-2, zeta = e^(i pi/14)
TWO_SIN = _psub(_mono(23), _mono(19))           # 2sin(pi/7) = (zeta^2 - zeta^-2) / i, 1/i = zeta^21

def seven(axis):
    """1/7 turn about x, y or z as a unit quaternion (numerators, exponent e): value = numerators / 2^e."""
    v = [TWO_COS, _Z12, _Z12, _Z12]; v[1 + axis] = TWO_SIN
    return (tuple(v), 1)

def _pq(x, y):
    a1, b1, c1, d1 = x; a2, b2, c2, d2 = y; m = _pmul
    s = lambda *t: tuple(sum(z) for z in zip(*t))
    n = lambda t: tuple(-z for z in t)
    return (s(m(a1,a2), n(m(b1,b2)), n(m(c1,c2)), n(m(d1,d2))), s(m(a1,b2), m(b1,a2), m(c1,d2), n(m(d1,c2))),
            s(m(a1,c2), n(m(b1,d2)), m(c1,a2), m(d1,b2)), s(m(a1,d2), m(b1,c2), n(m(c1,b2)), m(d1,a2)))

def rot_mul(A, B):
    """Compose two exact rotations and canonicalize: strip common factors of 2, and fold q ~ -q (one orientation)."""
    (x, ex), (y, ey) = A, B
    q, e2 = _pq(x, y), ex + ey
    while e2 > 0 and all(t % 2 == 0 for part in q for t in part):
        q = tuple(tuple(t // 2 for t in part) for part in q); e2 -= 1
    neg = tuple(tuple(-t for t in part) for part in q)
    return (max(q, neg), e2)

ROT_ID = ((((1,) + (0,) * 11), _Z12, _Z12, _Z12), 0)

# ---------- C11 the Collatz odd step with the halving kept as a 2-adic band ----------
def collatz_memo(limit):
    """Walk every odd start below `limit`: check the table; if absent, count. Returns (counted, total) odd steps
    and (counted, total) steps including the halvings (each halving is a shift, the band)."""
    memo = {1: (0, 0)}; counted = [0, 0]; total = [0, 0]
    for s in range(1, limit, 2):
        path = []; v = s
        while v not in memo:
            w = 3 * v + 1; tz = (w & -w).bit_length() - 1
            path.append((v, tz)); v = w >> tz
        o, hh = memo[v]
        for u, tz in reversed(path):
            o, hh = o + 1, hh + 1 + tz; memo[u] = (o, hh)
            counted[0] += 1; counted[1] += 1 + tz
        total[0] += memo[s][0]; total[1] += memo[s][1]
    return counted, total

# ---------- C9 pi: Archimedes' lift from the nonagon, as exact rational brackets ----------
def _sqrt_lo(x, k):  # largest m/2^k with (m/2^k)^2 <= x
    return F(isqrt((x.numerator << (2*k)) // x.denominator), 1 << k)
def _sqrt_hi(x, k):
    return _sqrt_lo(x, k) + F(1, 1 << k)

def cos40_bracket(bits=200):
    """cos 40 deg is the root of 8c^3 - 6c + 1 in (3/4, 4/5) (cos 3t = cos 120 = -1/2); bisect with exact signs."""
    f = lambda c: 8*c**3 - 6*c + 1
    lo, hi = F(3, 4), F(4, 5)
    assert f(lo) < 0 < f(hi)
    for _ in range(bits):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return lo, hi

def pi_lifts(levels=5, k=200):
    """Half-perimeters of the inscribed 9, 18, 36, ... -gons (unit radius), each as an exact bracket [lo, hi].
    chord_9^2 = 2 - 2cos40; each lift is one half-angle: chord_2n^2 = 2 - sqrt(4 - chord_n^2)."""
    clo, chi = cos40_bracket()
    L, U = 2 - 2*chi, 2 - 2*clo; n = 9; out = []
    for _ in range(levels):
        out.append((n, F(n, 2) * _sqrt_lo(L, k), F(n, 2) * _sqrt_hi(U, k)))
        L, U = 2 - _sqrt_hi(4 - L, k), 2 - _sqrt_lo(4 - U, k); n *= 2
    return out
