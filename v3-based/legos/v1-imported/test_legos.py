"""Tests for the Legos kernel: one check per claim of legos.tex. Run: python test_legos.py  (stdlib only, about a minute)

Kinds:  exact    checked in integers, Fractions or Z[r]; no float anywhere
        reading  the paper's sentence admits more than one construction; this one is Claude's, labeled as such
        bracket  an irrational value held between two exact rationals
Times are printed as notes and are never a pass condition.
"""
import time, random
from math import prod
from functools import cmp_to_key
from itertools import permutations, product
from legos import *
from legos import _padd, _pmul

RES = []
def say(tag, kind, ok, msg): RES.append((tag, kind, ok)); print(f"[{tag}] {'PASS' if ok else 'FAIL'} ({kind})  {msg}", flush=True)
def note(tag, msg): print(f"[{tag}] note  {msg}", flush=True)
rng = random.Random(26092026)
by_value = cmp_to_key(lambda a, b: (a - b).sign())

# ---------- ingredients ----------
def pmul(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): c[i + j] += x * y
    return c
cube = lambda x: x * x * x
f = lambda x: cube(x) - R(2) * x - ONE
roots = [r, ONE - r, R(-1)]
say("I1", "exact", pmul([1, 1], [-1, -1, 1]) == [-1, -2, 0, 1] and all(f(t) == ZERO for t in roots)
    and roots[0] + roots[1] + roots[2] == ZERO and roots[0] * roots[1] * roots[2] == ONE
    and r * RINV == ONE and r * r - ONE == RINV + ONE,
    "x^3-2x-1 = (x+1)(x^2-x-1); roots r, 1-r (= -1/r), -1 sum to 0 and multiply to 1; x^2-1 = 1/x+1 holds at r with 1/r = r-1")

# ---------- C1 exact rotation ----------
t = time.time(); pts = list(circle(3000))
on = all(a*a + b*b == d*d for a, b, d in pts)
lowest = all(a % 5 != 0 for a, b, d in pts)          # a/5^n in lowest terms, so no two steps can be the same point
distinct = len({F(a, d) for a, b, d in pts}) == 3000
say("C1", "exact", on and lowest and distinct, f"3000 turns by (4+3i)/5: every point has a^2+b^2 = d^2; a is never divisible by 5, "
    f"so each point sits in lowest terms over 5^n and none repeats ({time.time()-t:.2f}s)")

# ---------- C2 division as inversion ----------
rho = F(3, 4)                                         # the sphere through the ninth cube's corners (+-1/2, +-1/2, +-1/2)
rand_pts = [(tuple(rng.randint(-50, 50) for _ in range(3)), rng.randint(1, 50)) for _ in range(2000)]
rand_pts = [p for p in rand_pts if any(p[0])]
invol = all(same_point(invert(*invert(X, w, rho), rho), (X, w)) for X, w in rand_pts)
sphere = [(X, isqrt(4 * n2 // 3)) for X in product(range(-30, 31), repeat=3) for n2 in [sum(t*t for t in X)]
          if n2 and 4 * n2 % 3 == 0 and isqrt(4 * n2 // 3)**2 == 4 * n2 // 3]            # |X/w|^2 = 3/4 exactly
fixed = all(same_point(invert(X, w, rho), (X, w)) for X, w in sphere)
centre = invert((0, 0, 0), 1, rho); infinity = invert((1, 2, 3), 0, rho)
say("C2a", "exact", invol and fixed and not any(centre[0]) and centre[1] == 0 and same_point(infinity, ((0, 0, 0), 1)),
    f"inversion x -> (3/4) x/|x|^2 in homogeneous integers (no division): an involution on {len(rand_pts)} random points; "
    f"all {len(sphere)} rational sphere points found are fixed; the centre has no image and infinity maps to the centre")
diag_ok = True
for s in product((1, -1), repeat=3):
    for k in range(250, 501):                          # t = k/1000 from 1/4 (half-way to the corner) to 1/2 (the corner)
        X2, w2 = invert(tuple(k * c for c in s), 1000, rho)
        tt = F(X2[0] * s[0], w2)                       # image parameter along the same diagonal
        diag_ok &= all(X2[i] * s[0] == X2[0] * s[i] for i in range(3)) and F(k, 1000) * tt == F(1, 4) and F(1, 2) <= tt <= 1
    diag_ok &= same_point(invert(tuple(s), 4, rho), (tuple(s), 1)) and same_point(invert(tuple(s), 2, rho), (tuple(s), 2))
say("C2b", "exact", diag_ok, "on all 8 diagonals the segment from the midpoint (t=1/4) to the ninth-cube corner (t=1/2) goes onto "
    "the segment from that corner to the outer corner (t=1): t -> 1/(4t); the corner is fixed; the centre is reached by none")

# ---------- C3 counting that keeps its marks ----------
t = time.time(); spf = marks(10**7); t_sieve = time.time() - t
ex = sorted(set(atoms(12 + 26, spf))) == [2, 19] and sorted(set(atoms(23 + 54, spf))) == [7, 11]
sums = [(rng.randrange(1, 5 * 10**6), rng.randrange(1, 5 * 10**6)) for _ in range(200000)]
t = time.time(); res = [atoms(a + b, spf) for a, b in sums]; t_look = time.time() - t
good = True
for (a, b), fs in zip(sums, res):
    prod_ = 1
    for p in fs: prod_ *= p; good &= spf[p] == p
    good &= prod_ == a + b
say("C3", "exact", ex and good, "12+26 -> {2,19}, 23+54 -> {7,11}; 200,000 random sums inside the frontier resolve into prime "
    "atoms by lookup, each product checked back")
note("C3", f"frontier 10^7 built in {t_sieve:.2f}s, 200,000 sums resolved in {t_look:.2f}s (paper: 1.4s and 0.1s)")

# ---------- C4 / C10 self-composition, by the auto-function ----------
t = time.time(); order, table, look = lego(ICOSIAN_GENS, qmul, IDENT); t_build = time.time() - t
n = len(order); idx = {x: i for i, x in enumerate(order)}
unit = all(qdot(x, x) == ONE for x in order)
latin = all(sorted(row) == list(range(n)) for row in table) and all(sorted(col) == list(range(n)) for col in zip(*table))
inv = all(qconj(x) in idx for x in order)
shape = lambda x: sum(c != ZERO for c in x)
shapes = sorted((shape(x), sum(1 for y in order if shape(y) == shape(x))) for x in {shape(y): y for y in order}.values())
say("C4a", "exact", n == 120 and unit and latin and inv,
    f"i, j, the half-diagonal and (r + i + r^-1 j)/2 close under multiplication on exactly {n} unit quaternions in Z[r]; "
    f"the {n}x{n} table is a Latin square; every conjugate is present. Shapes (nonzero coords: count) {shapes}")
ips = {qdot(x, y) for x in order for y in order}
below1 = sorted([v for v in ips if v < ONE], key=by_value)
top = below1[-1]
nbrs = {sum(1 for y in order if qdot(x, y) == top) for x in order}
say("C4b", "exact", top == r * h and nbrs == {12}, f"largest inner product below 1 is {top} = phi/2, and every one of the 120 has "
    f"exactly {nbrs} neighbours there: the icosahedral vertex figure of the 600-cell")
cj = [idx[qconj(x)] for x in order]
t = time.time()
rot = {tuple(table[table[l][x]][m] for x in range(n)) for l in range(n) for m in range(n)}
mir = {tuple(table[table[l][cj[x]]][m] for x in range(n)) for l in range(n) for m in range(n)}
say("C4c", "exact", len(rot) == 7200 and len(mir) == 7200 and not (rot & mir) and n * n == 14400,
    f"x -> l x m over all 120^2 pairs gives {len(rot)} distinct maps (l,m and -l,-m coincide); with the mirror x -> l conj(x) m, "
    f"{len(rot | mir)} = 14,400, the order of the 120-cell's group. Read by table lookup only ({time.time()-t:.2f}s)")
pairs = [(rng.choice(order), rng.choice(order)) for _ in range(7200)]
t = time.time(); got = [look(a, b) for a, b in pairs]; t_look = time.time() - t
say("C10b", "exact", all(g == qmul(a, b) for g, (a, b) in zip(got, pairs)), "7,200 random compositions by lookup agree with multiplication")
note("C10b", f"table built once in {t_build:.2f}s, 7,200 lookups in {t_look:.4f}s (paper: 2.1s and 0.016s)")

# ---------- C5 addressability ----------
ident = r * RINV == ONE and RINV * RINV == RINV2 and r * r == r + ONE and SQRT5 * SQRT5 == R(5)
def even(p): return sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0
def signed(vals): return product(*[[v] if v == ZERO else [v, -v] for v in vals])
V = set()
for b, perms in [((ZERO, ZERO, R(2), R(2)), None), ((ONE, ONE, ONE, SQRT5), None), ((RINV2, r, r, r), None),
                 ((RINV, RINV, RINV, R2), None), ((ZERO, RINV2, ONE, R2), even), ((ZERO, RINV, r, SQRT5), even), ((RINV, ONE, r, R(2)), even)]:
    for p in permutations(range(4)):
        if perms is None or perms(p):
            for s in signed([b[i] for i in p]): V.add(tuple(s))
V = list(V)
ints = all(isinstance(c.p, int) and isinstance(c.q, int) for v in V for c in v)
norm8 = all(qdot(v, v) == R(8) for v in V)
t = time.time(); vals = {}
for i in range(len(V)):
    for j in range(i + 1, len(V)):
        k = qdot(V[i], V[j]); vals[k] = vals.get(k, 0) + 1
nearest = sorted([v for v in vals if v < R(8)], key=by_value)[-1]
say("C5a", "exact", ident and len(V) == 600 and ints and norm8 and vals[nearest] == 1200,
    f"r^-1 = r-1, r^-2 = 2-r, r^2 = r+1, sqrt5 = 2r-1; the 120-cell: {len(V)} vertices, each an address of 8 integers, all at |v|^2 = 8; "
    f"nearest pairs (inner product {nearest}) number {vals[nearest]}: its 1200 edges ({time.time()-t:.1f}s)")
box = 300
units = {(p, q) for p in range(-box, box + 1) for q in range(-box, box + 1) if abs(p*p + p*q - q*q) == 1}
pw = {(s * x.p, s * x.q) for m in range(-40, 41) for x in [rpow(m)] for s in (1, -1)}
pw = {u for u in pw if abs(u[0]) <= box and abs(u[1]) <= box}
steps = [rpow(-m) for m in range(0, 61)]
dense = all(x.sign() > 0 for x in steps) and all(steps[m + 1] < steps[m] for m in range(60))
say("C5b", "exact", units == pw and dense, f"units of Z[r] in the box |p|,|q| <= {box}: all {len(units)} are +-r^n; r^-n > 0 and strictly "
    f"falling for n <= 60, so there is no smallest positive step")
halves = sum(1 for x in order for c in x if not (F(c.p).denominator == 1 and F(c.q).denominator == 1))
note("C5", f"the 120-cell's coordinates are p+qr with p, q integers; the 120 icosians of C4 are not: {halves} of their 480 coordinates "
     f"carry a half (they become integer only doubled, as walk/kernel.rs does)")
note("C5", "'multiplying by r changes band without touching the address inside it': r*(p+qr) = q + (p+q)r changes (p,q); "
     "the band/address split this sentence means is not built here (meaning needed)")

# ---------- C6 one-pass readings ----------
crt = len({(k % 7, k % 11) for k in range(77)}) == 77 and len({(k % 2, k % 3, k % 7) for k in range(42)}) == 42
say("C6a", "exact", crt, "the 77-clock visits every (mod 7, mod 11) once; the 42-cycle every (mod 2, mod 3, mod 7) once")
ri = lambda n: [rng.randint(-9, 9) for _ in range(n)]
octo = all(sq(x) * sq(y) == sq(cd(x, y)) for x, y in ((ri(8), ri(8)) for _ in range(2000)))
dl = [sq(x) * sq(y) - sq(cd(x, y)) for x, y in ((ri(16), ri(16)) for _ in range(2000))]
pos, neg, zer = sum(d > 0 for d in dl), sum(d < 0 for d in dl), sum(d == 0 for d in dl)
say("C6b", "exact", octo and zer == 0, f"comparator read on squares (|x|^2|y|^2 - |xy|^2 has delta's sign): 0 on 2000 random octonion pairs; "
    f"nonzero on all 2000 sedenion pairs, {pos} positive and {neg} negative")
def zero_products(prod_fn):
    zs, fp = 0, set()
    for a in range(16):
        for b in range(a + 1, 16):
            for s1 in (1, -1):
                x = [p + s1 * q for p, q in zip(e(a), e(b))]
                for c in range(16):
                    for d in range(c + 1, 16):
                        for s2 in (1, -1):
                            if not any(prod_fn(x, [p + s2 * q for p, q in zip(e(c), e(d))])):
                                zs += 1; fp.add((a, b))
    return zs, fp
t = time.time(); zs, fp = zero_products(cd); t_ex = time.time() - t
xz, yz = [p + q for p, q in zip(e(1), e(10))], [p - q for p, q in zip(e(4), e(15))]
reading = {(o, s) for o in range(1, 8) for s in range(9, 16) if s != o + 8}
say("C6c", "exact", zs == 336 and len(fp) == 42 and not any(cd(xz, yz)) and sq(xz) * sq(yz) == 4,
    f"exhaustive (e_a +- e_b)(e_c +- e_d) over all 16 indices: {zs} zero products from {len(fp)} factor pairs; "
    f"(e1+e10)(e4-e15) = 0 with |x|^2|y|^2 = 4, so delta = 2 ({t_ex:.1f}s)")
say("C6d", "exact", reading == fp, "one reading, lower unit o with upper unit s skipping its mirror o+8: 7x6 = 42 pairs, "
    "identical to the exhaustive set")

# ---------- B.5 cubed indices ----------
basis = all(cd(e(a), e(b)) == [SIGN[a][b] if k == a ^ b else 0 for k in range(16)] for a in range(16) for b in range(16))
rand_ok = all(mul(x, y) == cd(x, y) for x, y in ((ri(16), ri(16)) for _ in range(500)))
t = time.time(); zs2, fp2 = zero_products(mul); t_lk = time.time() - t
octo_idx = all(cd(e(a, 8), e(b, 8)) == [(1 if k == a ^ b else 0) * sum(cd(e(a, 8), e(b, 8))) for k in range(8)] for a in range(8) for b in range(8))
say("B5", "exact", basis and rand_ok and zs2 == 336 and fp2 == fp and octo_idx,
    f"e_a e_b = SIGN[a][b] e_(a xor b) for all 256 slots (and on the octonions, A.11); 500 random products agree; "
    f"the 336 zero products and 42 pairs recovered by lookup alone ({t_lk:.1f}s vs {t_ex:.1f}s)")

# ---------- C7 the magic angle ----------
normals = [tuple(s if i == k else 0 for i in range(3)) for k in range(3) for s in (1, -1)]
centres = [tuple(1 - 2 * ((m >> i) & 1) for i in range(3)) for m in range(8)]    # corner m -> octant cube centre (x2)
def readings(sights):
    out = []
    for s in sights:
        for nn in normals:
            dot = sum(a * b for a, b in zip(nn, s)); c2 = F(dot * dot, sum(a*a for a in s) * sum(a*a for a in nn))
            out.append((F(3 * c2 - 1, 2), (dot > 0) - (dot < 0)))
    return out
A = readings([tuple(-c for c in centres[m]) for m in range(1, 8)])
say("C7", "reading", len(A) == 42 and all(p2 == 0 for p2, _ in A) and sum(sg > 0 for _, sg in A) == 21 and sum(sg < 0 for _, sg in A) == 21,
    "sight = each of the 7 other octant cubes looking at the empty middle (its body diagonal), x 6 oriented planes: "
    "42 readings, P2 = (3cos^2-1)/2 = 0 exactly on all, cosine +1/sqrt3 on 21 and -1/sqrt3 on 21")
B = readings([tuple(a - b for a, b in zip(centres[0], centres[m])) for m in range(1, 8)])
note("C7", f"other reading (sight from each of the 7 cubes to cube 000): P2 = 0 on {sum(p2 == 0 for p2, _ in B)} of 42. "
     f"The check holds on any 7 of the 8 body diagonals, and on all 8; why 7 is taken from A.11's octonion indexing")

# ---------- C8 the searching cone ----------
d3 = (1, 1, 1)
cone = [v for v in product(range(-12, 13), repeat=3) if any(v) and gcd(gcd(v[0], v[1]), v[2]) == 1
        and v[0]*v[1] + v[1]*v[2] + v[2]*v[0] == 0]
p2zero = all(3 * sum(a * b for a, b in zip(v, d3))**2 == 3 * sum(a*a for a in v) for v in cone)   # 3cos^2 = 1
family = all((lambda v: v[0]*v[1] + v[1]*v[2] + v[2]*v[0] == 0)((a*(a+b), b*(a+b), -a*b)) for a in range(-40, 41) for b in range(-40, 41))
P = lambda v: (v[2], v[0], v[1])                      # 1/3 turn about (1,1,1): exact, a permutation
axes_on = all(ax in cone for ax in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]) and P((1, 0, 0)) == (0, 1, 0) and P(P(P((5, 7, 11)))) == (5, 7, 11)
say("C8a", "exact", p2zero and family and axes_on and all(P(v) in cone for v in cone),
    f"the cone of half-angle arccos(1/sqrt3) about (1,1,1) is the rational conic ab+bc+ca = 0: {len(cone)} primitive integer "
    f"directions in [-12,12]^3, P2 = 0 on each; (a(a+b), b(a+b), -ab) lies on it for every a, b; the axes are on it; "
    f"the 1/3 turn about the diagonal is the permutation x->y->z and keeps the cone")
walk = [k * F(7, 9) % 1 for k in range(9)]
quarter = {F(0), F(1, 4), F(1, 2), F(3, 4)}
lifts = [len({F(k, m) for k in range(m)} & quarter) for m in (9, 18, 36)]
say("C8b", "exact", len(set(walk)) == 9 and [k for k, w in enumerate(walk) if w in {F(0), F(1, 3), F(2, 3)}] == [0, 3, 6]
    and F(7, 9) % 1 == F(-2, 9) % 1 and lifts == [1, 2, 4] and atoms(36, spf) == [2, 2, 3, 3] and atoms(360, spf) == [2, 2, 2, 3, 3, 5],
    "in turns: the step 7/9 (+280) equals -2/9 (-80), visits all 9 positions as the star {9/2}, and lands on a third-turn (an axis) "
    "exactly at steps 0, 3, 6; lifting 9 -> 18 -> 36 holds 1, then 2, then all 4 of the oscillator's quarter-turns")
note("C8", "the turn by 40 degrees about the diagonal has no rational matrix (cos 40 is cubic), so the walk is checked in turn "
     "arithmetic; the exact geometry checked is the 1/3 turn it lands on")

# ---------- C9 pi ----------
cands = [F(s * a, b) for a in (1,) for b in (1, 2, 4, 8) for s in (1, -1)]
irreducible = all(8 * c**3 - 6 * c + 1 != 0 for c in cands)
half = F(9, 2) * F(1, 9) == F(1, 2) and look(order[idx[(ZERO, ONE, ZERO, ZERO)]], (ZERO, ONE, ZERO, ZERO)) == (R(-1), ZERO, ZERO, ZERO)
paper = [F(30782, 10**4), F(31257, 10**4), F(31376, 10**4), F(31406, 10**4), F(31413, 10**4)]
L = pi_lifts()
rounds = all(v - F(1, 20000) <= lo and hi < v + F(1, 20000) for (nn, lo, hi), v in zip(L, paper))
rising = all(L[i][2] < L[i + 1][1] for i in range(len(L) - 1))
say("C9", "bracket", irreducible and half and rounds and rising,
    "omega^(9/2) is the half-turn, and i*i = -1 in the icosian table; 8c^3-6c+1 has no rational root (cos 40 is cubic, the "
    "nonagon is not constructible); the lifts 9, 18, 36, 72, 144 are held in exact brackets that round to 3.0782, 3.1257, "
    "3.1376, 3.1406, 3.1413 and rise strictly")
wid = L[-1][2] - L[-1][1]
note("C9", f"bracket width at 144 sides < 2^-{wid.denominator.bit_length() - wid.numerator.bit_length() - 1}. "
     "Each lift is one square root; the seed chord of the 9-gon is not (it needs cos 40, a cubic)")

# ---------- C10 legos on turns ----------
ring, T, lk = lego([F(1, 9), F(1, 4)], turn, F(0))
say("C10a", "exact", len(ring) == 36 and set(ring) == {F(k, 36) for k in range(36)} and min(x for x in ring if x) == F(1, 36)
    and lk(F(1, 4), F(1, 4)) == F(1, 2), "the nonagon brick (1/9) and the oscillator brick (1/4) close on exactly 36 turns, the "
    "smallest 1/36; look(1/4, 1/4) = 1/2, the -1")

# ---------- C11 the three-sevens tree ----------
unit7 = _padd(_pmul(TWO_COS, TWO_COS), _pmul(TWO_SIN, TWO_SIN)) == (4,) + (0,) * 11
gens = [seven(0), seven(1), seven(2)]
seen = {ROT_ID: ""}; front = [ROT_ID]; counts = [1]; repeats = {}
t = time.time()
for depth in range(1, 9):
    nxt = []
    for g in front:
        for gi, gen in enumerate(gens):
            k = rot_mul(g, gen); w = seen[g] + "xyz"[gi]
            if k in seen: repeats.setdefault(depth, []).append((w, seen[k])); continue
            seen[k] = w; nxt.append(k)
    front = nxt; counts.append(len(seen))
t_tree = time.time() - t
free = counts[:7] == [(3**k - 1) // 2 for k in range(1, 8)] and all(dd not in repeats for dd in range(1, 7))
at7 = sorted(repeats.get(7, [])) == [("xxxxxxx", ""), ("yyyyyyy", ""), ("zzzzzzz", "")]
say("C11a", "exact", unit7 and free and at7 and counts[7] == 3280 - 3,
    f"1/7 turns about x, y, z as exact quaternions over Q(zeta_28): orientations within 0..6 moves {counts[:7]} = (3^k-1)/2, "
    f"no repeat; at 7 moves the only repeats are x^7 = y^7 = z^7 = 1, together ({counts[7]} = 3280 - 3)")
note("C11", f"8 moves: {counts[8]} orientations, repeats at 8: {len(repeats.get(8, []))} ({t_tree:.1f}s). Never closing is the known "
     "theorem (no finite 3D rotation group has 7-fold turns about two axes); a finite walk cannot prove it")
c = lambda k: (3**k - 1) // 2
def first(mod):
    x, k = 0, 0
    while True:
        k += 1; x = (3 * x + 1) % mod
        if x == 0: return k
def ordp(p):
    k, x = 1, 3 % p
    while x != 1: x = x * 3 % p; k += 1
    return k
odd_primes = [p for p in range(5, 1000) if spf[p] == p]
say("C11b", "exact", all(c(k) == 3 * c(k - 1) + 1 for k in range(2, 400)) and all(first(p) == ordp(p) for p in odd_primes)
    and (first(7), first(49), first(343)) == (6, 42, 294) and all(c(k) % 3 == 1 for k in range(1, 1000))
    and c(6) == 364 == 2 * 2 * 7 * 13 == 7**3 + 3 * 7,
    f"each count is 3x the previous + 1; every odd prime 5..997 enters at the order of 3 mod p; 7 at step 6 (364 = 2^2*7*13 = 7^3+3*7), "
    f"7^2 at 42, 7^3 at 294; 3 never enters")
note("C11", f"the prime 2 enters at step {first(2)}, not at the order of 3 mod 2 (= 1): the /2 in (3^k-1)/2")
t = time.time(); (co, ch), (to, th) = collatz_memo(2**20); t_col = time.time() - t
note("C11", f"Collatz, all 2^19 odd starts below 2^20, 'check the table; if absent, count': counted {co:,} of {to:,} odd steps "
     f"({100*co/to:.2f}%), {ch:,} of {th:,} steps with halvings ({100*ch/th:.2f}%) (paper: 3.4%; which count it used is not stated) ({t_col:.1f}s)")
t = time.time(); spf20 = marks(2**20); t20 = time.time() - t
tree = [c(k) for k in range(1, 14)]
ok = tree[-1] == 797161 and all(all(spf20[p] == p for p in atoms(v, spf20)) and prod(atoms(v, spf20)) == v for v in tree)
say("C11c", "exact", ok, f"every tree count up to 797161 = (3^13-1)/2 resolves into atoms from the marks to 2^20")
note("C11c", f"marks to 2^20 in {t20:.2f}s (paper 0.33s)")

# ---------- Appendix A, the checkable steps ----------
lattice = [(x, y) for x in range(7) for y in range(7)]
d2 = lambda a, b: sum((p - q)**2 for p, q in zip(a, b))
eq2 = any(d2(a, b) == d2(b, c_) == d2(a, c_) > 0 for i, a in enumerate(lattice) for j, b in enumerate(lattice[i+1:], i+1) for c_ in lattice[j+1:])
evens = [v for v in product((0, 1), repeat=3) if sum(v) % 2 == 0]
say("A2", "exact", not eq2 and d2((1,0,0),(0,1,0)) == d2((0,1,0),(0,0,1)) == d2((1,0,0),(0,0,1)) == 2
    and {d2(a, b) for a in evens for b in evens if a != b} == {2}, "no equilateral triangle in the 7x7 integer grid; (1,0,0),(0,1,0),(0,0,1) "
    "is one in Z^3; the even-parity corners of {0,1}^3 are a regular tetrahedron with edge^2 = 2")
dod = set(product((ONE, -ONE), repeat=3))
for a, b in product((RINV, -RINV), (r, -r)):
    dod |= {(ZERO, a, b), (a, b, ZERO), (b, ZERO, a)}
dod = list(dod)
dd = lambda a, b: sum(((p - q) * (p - q) for p, q in zip(a, b)), ZERO)
dvals = sorted({dd(a, b) for a in dod for b in dod if a != b}, key=by_value)
edge = dvals[0]; deg = {sum(1 for b in dod if b != a and dd(a, b) == edge) for a in dod}
cubes = set()
for v in dod:
    nb = [u for u in dod if dd(u, v) == R(4)]
    for trip in [(a, b, c_) for i, a in enumerate(nb) for j, b in enumerate(nb[i+1:], i+1) for c_ in nb[j+1:]]:
        if all(dd(x, y) == R(8) for x, y in [(trip[0], trip[1]), (trip[1], trip[2]), (trip[0], trip[2])]):
            cset = frozenset([v, *trip] + [tuple(-t for t in u) for u in [v, *trip]])
            if all(x in dod for x in cset) and len(cset) == 8: cubes.add(cset)
share = {sum(1 for cb in cubes if x in cb) for x in dod}
say("A5", "exact", len(dod) == 20 and edge == R(4) * RINV2 and deg == {3} and r - ONE == RINV and len(cubes) == 5 and share == {2},
    f"roofs of height r - 1 = 1/r on the cube give 20 vertices, all edges^2 = 4r^-2, 3 at each vertex: the dodecahedron; "
    f"{len(cubes)} cubes share its corners, each corner in 2")
say("A9", "exact", 180 - 360 // 9 == 140 == 7 * 20 and 360 - 140 == 220 == 11 * 20 and 220 - 140 == 80 == 2 * 40,
    "nonagon angle 140 = 7*20, outer 220 = 11*20, clearance 80 = two turns of 40")
tess = list(product((0, 1), repeat=4))
say("A12", "exact", len(tess) == 16 and sum(1 for a in tess for b in tess if a < b and d2(a, b) == 1) == 32, "16 corners and 32 edges: the tesseract")
say("A14", "exact", 10 % 9 == 1 and 2 * 5 % 9 == 1 and 7 % 9 == -2 % 9 and 360 % 7 != 0, "on the nonagon's clock 10 = 1, 2 and 5 are inverses, 7 = -2; 7 does not divide 360")
note("A11", "the doubling of the 120 icosians into the 240 roots of E8 is listed as known (section 3) and is not built here")

print(f"\n{sum(ok for _, _, ok in RES)}/{len(RES)} passed  "
      f"(exact {sum(1 for _, k, _ in RES if k == 'exact')}, reading {sum(1 for _, k, _ in RES if k == 'reading')}, "
      f"bracket {sum(1 for _, k, _ in RES if k == 'bracket')})")
