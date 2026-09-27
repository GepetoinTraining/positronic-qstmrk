"""The atom kernel, second build (26 Sep 2026), from the atom outward.

The chain [PG]:
  space is infinite, and its origin is the missing point of the first atom. Inversion through the atom's sphere
  gives every point of space a partner except that one point, whose partner is infinity. So the atom is 1, and
  counting starts there. Nothing below uses sin, cos, pi, roots or floats.

Scale is outside the map, as in an ISIS3 cube: a stored value is an integer address; its scale is a band in the label.
  r-band n : times r^n  (r^2 = r + 1, the relation's root; phi is never written)
  2-band m : times 2^-m (the half)
The atom is 16-dimensional and carries 16 anchors, the cubed indices: 3 bits name a corner, the 4th inner or outer.
In doubled coordinates (2-band 1) the inner anchors (the ninth cube) sit at (+-1,+-1,+-1), the outer at (+-2,+-2,+-2),
and the atom's sphere is |X|^2 = 3: an integer. The origin is the missing point.
"""
from fractions import Fraction as F
from math import isqrt, gcd
from array import array

# ---------- scale outside the map: address (p, q) in Z[r], bands (n, m) in the label ----------
def _up(p, q): return q, p + q               # address times r
def _down(p, q): return q - p, p             # address times r^-1 = r - 1 (integer: no division)

class Num:
    """value = r^n * 2^-m * (p + q r), p and q integers. Times r moves n; the half moves m; the address stays."""
    __slots__ = ("p", "q", "n", "m")
    def __init__(s, p, q=0, n=0, m=0): s.p, s.q, s.n, s.m = p, q, n, m
    def at(s, n, m):
        """The same value written at bands (n, m), m >= s.m: only the address changes."""
        p, q = s.p, s.q
        for _ in range(s.n - n): p, q = _up(p, q)
        for _ in range(n - s.n): p, q = _down(p, q)
        return (p << (m - s.m), q << (m - s.m))
    def key(s):
        """Canonical form for equality: r-band folded into the address, the 2-band stripped to its least."""
        p, q = s.at(0, max(s.m, 0)); m = max(s.m, 0)
        while m > 0 and p % 2 == 0 and q % 2 == 0: p, q, m = p >> 1, q >> 1, m - 1
        return (p, q, m)
    def __mul__(s, o): return Num(s.p*o.p + s.q*o.q, s.p*o.q + s.q*o.p + s.q*o.q, s.n + o.n, s.m + o.m)
    def __add__(s, o):
        n, m = min(s.n, o.n), max(s.m, o.m); a, b = s.at(n, m), o.at(n, m)
        return Num(a[0] + b[0], a[1] + b[1], n, m)
    def __neg__(s): return Num(-s.p, -s.q, s.n, s.m)
    def __sub__(s, o): return s + (-o)
    def __eq__(s, o): return s.key() == o.key()
    def __hash__(s): return hash(s.key())
    def __repr__(s): return f"[{s.p},{s.q}|r^{s.n} 2^-{s.m}]"
    def sign(s):
        """Exact sign: the bands are positive, so it is the address's sign, p + q*phi, read by comparing squares."""
        a, b = 2 * s.p + s.q, s.q
        if a >= 0 and b >= 0: return 0 if a == 0 and b == 0 else 1
        if a <= 0 and b <= 0: return -1
        d = a * a - 5 * b * b
        return (1 if d > 0 else -1) if a > 0 else (1 if d < 0 else -1)
    def __lt__(s, o): return (s - o).sign() < 0

ZERO, ONE = Num(0), Num(1)
R = Num(1, 0, 1)                 # r itself: address 1, one r-band up
HALF = Num(1, 0, 0, 1)           # the half: address 1, one 2-band up

# ---------- quaternions (the first 4 anchors: 1, i, j, k) ----------
def qmul(x, y):
    a1, b1, c1, d1 = x; a2, b2, c2, d2 = y
    return (a1*a2 - b1*b2 - c1*c2 - d1*d2, a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2, a1*d2 + b1*c2 - c1*b2 + d1*a2)
def qconj(x): return (x[0], -x[1], -x[2], -x[3])
def qdot(x, y): return x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3]

IDENT = (ONE, ZERO, ZERO, ZERO)
# i, j, the half-diagonal, and (r + i + r^-1 j)/2: every coordinate is address 1 or 0; all scale is in the bands
ICOSIAN_GENS = [(ZERO, ONE, ZERO, ZERO), (ZERO, ZERO, ONE, ZERO), (HALF, HALF, HALF, HALF),
                (Num(1, 0, 1, 1), HALF, Num(1, 0, -1, 1), ZERO)]

# ---------- B.4 Legos, the auto-function (verbatim) ----------
def lego(pieces, snap, identity):
    items = {identity: 0}; order = [identity]; front = [identity]
    while front:
        nxt = []
        for a in front:
            for p in pieces:
                b = snap(a, p)
                if b not in items: items[b] = len(order); order.append(b); nxt.append(b)
        front = nxt
    table = [[items[snap(a, b)] for b in order] for a in order]   # once: the madness
    look = lambda a, b: order[table[items[a]][items[b]]]          # after: syntax
    return order, table, look

turn = lambda a, b: (a + b) % 1

# ---------- the atom in space: 16 anchors, the origin as its missing point, inversion ----------
def anchor(a):
    """Anchor a (0..15): bits 0-2 a corner, bit 3 inner (0) or outer (1). Doubled coordinates."""
    s = tuple(1 - 2 * ((a >> i) & 1) for i in range(3))
    return tuple(c * (2 if a & 8 else 1) for c in s)

RHO = 3                                      # the atom's sphere through the inner anchors, doubled: |X|^2 = 3

def invert(X, w, rho=RHO):
    """x -> rho x / |x|^2 for the point X/w, homogeneous integers: rho * (X/w) / (|X|^2/w^2) = rho w X / |X|^2.
    The origin (X = 0) returns (0, 0): no point. Infinity (w = 0) returns the origin."""
    return tuple(rho * w * t for t in X), sum(t * t for t in X)

def same_point(A, B):
    u, v = tuple(A[0]) + (A[1],), tuple(B[0]) + (B[1],)
    if not any(u) or not any(v): return False
    return all(u[i] * v[j] == u[j] * v[i] for i in range(len(u)) for j in range(i + 1, len(u)))

# ---------- the 16 anchors as bands: doubling, and the XOR-with-sign table ----------
def cd(x, y):
    """Cayley-Dickson doubling: each doubling adds one bit, one more axis of anchors."""
    n = len(x)
    if n == 1: return [x[0]*y[0]]
    hh = n//2; a, b, c, d = x[:hh], x[hh:], y[:hh], y[hh:]
    conj = lambda v: [v[0]] + [-t for t in v[1:]]
    ac, db = cd(a, c), cd(conj(d), b); da, bc = cd(d, a), cd(b, conj(c))
    return [p - q for p, q in zip(ac, db)] + [p + q for p, q in zip(da, bc)]
def e(i, n=16): return [1 if k == i else 0 for k in range(n)]
def sq(v): return sum(t * t for t in v)

SIGN = [[sum(cd(e(a), e(b))) for b in range(16)] for a in range(16)]   # once
def mul(x, y):                                                         # after: band a times band b lands in band a^b
    out = [0]*16
    for a in range(16):
        if x[a]:
            for b in range(16):
                if y[b]: out[a ^ b] += SIGN[a][b] * x[a] * y[b]
    return out

# ---------- counting that keeps its marks ----------
def marks(N):
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

# ---------- the exact circle (A.1's rational repair: 3-4-5) ----------
def circle(n):
    a, b, d = 1, 0, 1
    for _ in range(n):
        a, b, d = 4*a - 3*b, 3*a + 4*b, 5*d
        yield a, b, d

# ---------- the Collatz odd step, halving kept as a 2-adic band ----------
def collatz_memo(limit):
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

# ---------- turns on a clock: integer polynomials in the brick x, reduced by the clock's relation ----------
def pmul(a, b):
    """Integer polynomials, lowest degree first."""
    c = [0] * (len(a) + len(b) - 1)
    for i, s in enumerate(a):
        if s:
            for j, t in enumerate(b):
                if t: c[i + j] += s * t
    return c

def pdivmod(a, b, p=None):
    """a = quot * b + rem, b monic (lowest degree first); over the integers, or mod p. b is monic, so nothing is divided."""
    a = list(a); db = len(b) - 1; quot = [0] * max(1, len(a) - db)
    for k in range(len(a) - 1, db - 1, -1):
        t = a[k] % p if p else a[k]
        if t:
            quot[k - db] = t
            for j in range(db + 1): a[k - db + j] -= t * b[j]
    rem = a[:db]
    if p: rem, quot = [v % p for v in rem], [v % p for v in quot]
    return quot, rem

def lifts(L, U, n, levels, k=240):
    """From a chord^2 held in [L, U] on the n-clock: half-perimeters n*chord/2 as exact brackets, halving the turn each level."""
    out = []
    for _ in range(levels):
        out.append((n, F(n, 2) * _sqrt_lo(L, k), F(n, 2) * _sqrt_hi(U, k)))
        L, U = 2 - _sqrt_hi(4 - L, k), 2 - _sqrt_lo(4 - U, k); n *= 2
    return out

def cubic_roots(bits=240):
    """All three roots of 8c^3 - 6c + 1, each held by sign counting (the waves of the turns 1/9, 2/9, 4/9)."""
    f = lambda c: 8 * c**3 - 6 * c + 1
    grid = [F(k, 20) for k in range(-20, 21)]
    out = []
    for lo, hi in zip(grid, grid[1:]):
        if f(lo) * f(hi) < 0:
            up = f(lo) < 0
            for _ in range(bits):
                mid = (lo + hi) / 2
                if (f(mid) < 0) == up: lo = mid
                else: hi = mid
            out.append((lo, hi))
    return out

class Clock:
    """Turns as integer polynomials in one brick x, reduced by a monic relation rel (lowest degree first)."""
    def __init__(s, rel): s.rel, s.d = rel, len(rel) - 1
    def red(s, c):
        c = list(c) + [0] * max(0, s.d - len(c))
        for k in range(len(c) - 1, s.d - 1, -1):
            t = c[k]
            if t:
                c[k] = 0
                for j in range(s.d): c[k - s.d + j] -= t * s.rel[j]
        return tuple(c[:s.d])
    def x(s, n, period): return s.red([0] * (n % period) + [1])
    def mul(s, a, b): return s.red(pmul(a, b))
    def add(s, *t): return tuple(sum(z) for z in zip(*t))
    def neg(s, a): return tuple(-z for z in a)
    def zero(s): return (0,) * s.d

# The 28-clock: bricks 1/4 (the oscillator) and 1/7 close on 28 turns. x is the 1/28 turn. The half-turn is -1,
# so x^14 + 1 = 0; and x^14 + 1 = (x^2 + 1) * PHI28, where x^2 + 1 = 0 would make x the quarter-turn itself.
PHI28 = [1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
C28 = Clock(PHI28)
_Z = C28.zero()
_z, _zb, _i = C28.x(2, 28), C28.x(26, 28), C28.x(7, 28)       # the 1/14 turn, its reverse, the quarter-turn i
WAVE_R = C28.add(_z, _zb)                                       # 2 x (real wave of the 1/14 turn)
WAVE_I = C28.mul(C28.add(_z, C28.neg(_zb)), C28.x(21, 28))     # 2 x (i-wave): (z - z^-1) / i, and 1/i = x^21

def seven(axis):
    """The 1/7 turn about x, y or z: the 1/14 turn's two waves, real on the scalar and i-wave on the axis
    (the double cover: a turn is carried by half of it). Numerators over one 2-band."""
    v = [WAVE_R, _Z, _Z, _Z]; v[1 + axis] = WAVE_I
    return (tuple(v), 1)

def _qc(x, y):
    a1, b1, c1, d1 = x; a2, b2, c2, d2 = y; m, ad, ng = C28.mul, C28.add, C28.neg
    return (ad(m(a1, a2), ng(m(b1, b2)), ng(m(c1, c2)), ng(m(d1, d2))), ad(m(a1, b2), m(b1, a2), m(c1, d2), ng(m(d1, c2))),
            ad(m(a1, c2), ng(m(b1, d2)), m(c1, a2), m(d1, b2)), ad(m(a1, d2), m(b1, c2), ng(m(c1, b2)), m(d1, a2)))

def turn_mul(A, B):
    """Compose two turns and write them canonically: strip the 2-band to its least, and fold q ~ -q (one orientation)."""
    (x, ex), (y, ey) = A, B
    q, e2 = _qc(x, y), ex + ey
    while e2 > 0 and all(t % 2 == 0 for part in q for t in part):
        q = tuple(tuple(t // 2 for t in part) for part in q); e2 -= 1
    return (max(q, tuple(tuple(-t for t in part) for part in q)), e2)

TURN_ID = (((1,) + (0,) * 11, _Z, _Z, _Z), 0)

# The 9-clock: omega = 1/9 turn, omega^3 is the third-turn t with t^2 + t + 1 = 0, so omega^6 + omega^3 + 1 = 0 (A.7).
PHI9 = [1, 0, 0, 1, 0, 0, 1]
C9 = Clock(PHI9)

def _sqrt_lo(v, k): return F(isqrt((v.numerator << (2 * k)) // v.denominator), 1 << k)   # counting squares: largest m/2^k with (m/2^k)^2 <= v
def _sqrt_hi(v, k): return _sqrt_lo(v, k) + F(1, 1 << k)

def nonagon_wave(bits=200):
    """The real wave c of the 1/9 turn: 2c = omega + omega^-1 satisfies (2c)^3 - 3(2c) + 1 = 0 on the 9-clock, i.e.
    8c^3 - 6c + 1 = 0. Its three roots are the waves of the turns 1/9, 2/9, 4/9; the smallest turn has the largest wave.
    Held between two exact rationals by sign counting."""
    f = lambda c: 8 * c**3 - 6 * c + 1
    lo, hi = F(3, 4), F(4, 5)
    for _ in range(bits):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return lo, hi

def pi_lifts(levels=5, k=200):
    """Half-perimeters of the 9, 18, 36, ... -gons, each as an exact bracket. chord^2 of the 1/9 turn = (1-omega)(1-omega^-1)
    = 2 - 2c. Halving a turn: if y^2 = z, then (y + y^-1)^2 = 2 + (z + z^-1), so chord_2n^2 = 2 - sqrt(4 - chord_n^2)."""
    clo, chi = nonagon_wave()
    L, U = 2 - 2 * chi, 2 - 2 * clo; n = 9; out = []
    for _ in range(levels):
        out.append((n, F(n, 2) * _sqrt_lo(L, k), F(n, 2) * _sqrt_hi(U, k)))
        L, U = 2 - _sqrt_hi(4 - L, k), 2 - _sqrt_lo(4 - U, k); n *= 2
    return out

# ---------- E8: the 120 and their band-down copy ----------
def euclid(x):
    """p + q on the address: the only form read on the address that gives a unit and its band-down copy the same norm
    (test K18). It is Conway and Sloane's Euclidean norm, a + b sqrt5 -> a + b."""
    p, q, m = x.key()
    return F(p + q, 1 << m)

# ---------- ISIS3 cube: integer core, scale and anchors in the PVL label ----------
TYPES = {"SignedWord": ("h", 2), "UnsignedByte": ("B", 1)}

def write_cub(path, core, ptype, groups):
    """core: list of bands, each a list of lines, each a list of ints. groups: {name: {key: value}} for the label."""
    B, L, S = len(core), len(core[0]), len(core[0][0])
    body = array(TYPES[ptype][0], [v for band in core for line in band for v in line]).tobytes()   # BandSequential, Lsb
    extra = "".join(f"  Group = {g}\n" + "".join(f"    {k} = {v}\n" for k, v in kv.items()) + "  End_Group\n" for g, kv in groups.items())
    def head(start):
        return (f"Object = IsisCube\n  Object = Core\n    StartByte = {start}\n    Format = BandSequential\n"
                f"    Group = Dimensions\n      Samples = {S}\n      Lines = {L}\n      Bands = {B}\n    End_Group\n"
                f"    Group = Pixels\n      Type = {ptype}\n      ByteOrder = Lsb\n      Base = 0.0\n      Multiplier = 1.0\n    End_Group\n"
                f"  End_Object\n{extra}End_Object\nEnd\n")
    h = head(0); start = len(head(len(h) + 1).encode()) + 1
    with open(path, "wb") as fh: fh.write(head(start).encode().ljust(start - 1, b" ") + body)

def read_cub(path):
    import re
    raw = open(path, "rb").read(); txt = raw[:raw.index(b"\nEnd\n") + 5].decode()
    g = lambda k: re.search(rf"\b{k} = (\S+)", txt).group(1)
    start, S, L, B, ptype = int(g("StartByte")), int(g("Samples")), int(g("Lines")), int(g("Bands")), g("Type")
    code, size = TYPES[ptype]
    flat = array(code); flat.frombytes(raw[start - 1:start - 1 + size * S * L * B])
    core = [[list(flat[(b * L + l) * S:(b * L + l + 1) * S]) for l in range(L)] for b in range(B)]
    groups = {}
    for name, body in re.findall(r"^  Group = (\w+)\n(.*?)^  End_Group", txt, re.M | re.S):
        groups[name] = dict(re.findall(r"^\s{4}(\w+) = (.+)$", body, re.M))
    return core, groups
