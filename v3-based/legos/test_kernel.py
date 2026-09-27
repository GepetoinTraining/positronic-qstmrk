"""Checks for the atom kernel, second build. Run: python test_kernel.py   (stdlib only)

Every check is exact: integers, turn counts, and Z[r] addresses with bands. Nothing imported from outside the atom:
no sin, cos, pi, roots or floats. Claims that cannot yet be reached from the atom are printed as 'open', not tested.
Times are notes, never pass conditions.
"""
import os, time, random, tempfile
from math import prod
from functools import cmp_to_key
from itertools import permutations, product
from kernel import *
from kernel import _up, _qc

RES = []
def say(tag, ok, msg): ok = bool(ok); RES.append(ok); print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}", flush=True)
def note(tag, msg): print(f"[{tag}] note  {msg}", flush=True)
def open_(tag, msg): print(f"[{tag}] open  {msg}", flush=True)
rng = random.Random(26092026)
by_value = cmp_to_key(lambda a, b: (a - b).sign())
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- K0 the atom is 1 ----------
inner = [anchor(a) for a in range(8)]; outer = [anchor(a) for a in range(8, 16)]
inner_fixed = all(same_point(invert(X, 1), (X, 1)) for X in inner)
quarter = [(X, 2) for X in inner]                                 # the midpoints, half-way from the origin to each inner anchor
outer_pairs = all(same_point(invert(Y, 1), (X, 2)) and same_point(invert(X, 2), (Y, 1)) for X, Y in zip(inner, outer))
S = [((0, 0, 0), 1)] + [(X, 1) for X in inner + outer] + quarter
def partner_in(P): img = invert(*P); return any(same_point(img, Q) for Q in S)
unpaired = [P for P in S if not partner_in(P)]
origin_img = invert((0, 0, 0), 1)
say("K0a", inner_fixed and outer_pairs and unpaired == [((0, 0, 0), 1)] and origin_img == ((0, 0, 0), 0),
    "the 16 anchors with the origin as the missing point: the 8 inner anchors lie on the atom's sphere |X|^2 = 3 and stay; "
    "each outer anchor and the midpoint on its diagonal are partners; of the 25 points exactly 1 has no partner, the origin")
far = [(tuple(rng.randint(-60, 60) for _ in range(3)), rng.randint(1, 60)) for _ in range(3000)]
far = [P for P in far if any(P[0])]
space = all(same_point(invert(*invert(X, w)), (X, w)) for X, w in far)
inf_to_origin = all(same_point(invert(X, 0), ((0, 0, 0), 1)) for X, _ in far[:200])
say("K0b", space and inf_to_origin, f"all of space pairs through the first atom: {len(far)} random points of space are each other's partners "
    "twice over (an involution, in homogeneous integers, no division); every direction at infinity lands on the origin")

# ---------- K1 scale outside the map ----------
xs = [Num(rng.randint(-50, 50), rng.randint(-50, 50), rng.randint(-5, 5), rng.randint(0, 3)) for _ in range(500)]
times_r = all((lambda y: (y.p, y.q) == (x.p, x.q) and y.n == x.n + 1)(x * R) for x in xs)
half = all((lambda y: (y.p, y.q) == (x.p, x.q) and y.m == x.m + 1)(x * HALF) for x in xs)
value = all(x * R == Num(*_up(x.p, x.q), x.n, x.m) for x in xs)          # the value is the folded product
ident = R * Num(1, 0, -1) == ONE and R * R == R + ONE and Num(1, 0, -1) == R - ONE and Num(1, 0, -2) == Num(2, -1) \
    and Num(-1, 2) * Num(-1, 2) == Num(5)
say("K1", times_r and half and value and ident, "times r moves the r-band and leaves the address; the half moves the 2-band and "
    "leaves the address; the value equals the product folded into the address. r^-1 = r-1, r^-2 = 2-r, r^2 = r+1, "
    "(2r-1)^2 = 5, all as bands and integer addresses")

# ---------- K2 the exact circle ----------
pts = list(circle(3000))
say("K2", all(a*a + b*b == d*d for a, b, d in pts) and all(a % 5 for a, b, d in pts),
    "A.1's rational repair (3-4-5) turned 3000 times: a^2+b^2 = d^2 at every point; a never divisible by 5, so every point "
    "is in lowest terms over 5^n and none repeats")

# ---------- K3 counting keeps its marks ----------
t = time.time(); spf = marks(10**7); t_s = time.time() - t
sums = [(rng.randrange(1, 5 * 10**6), rng.randrange(1, 5 * 10**6)) for _ in range(200000)]
t = time.time(); res = [atoms(a + b, spf) for a, b in sums]; t_l = time.time() - t
say("K3", sorted(set(atoms(38, spf))) == [2, 19] and sorted(set(atoms(77, spf))) == [7, 11]
    and all(prod(fs) == a + b and all(spf[p] == p for p in fs) for (a, b), fs in zip(sums, res)),
    "12+26 -> {2,19}, 23+54 -> {7,11}; 200,000 sums inside the frontier resolve into prime atoms by lookup, checked back")
note("K3", f"frontier 10^7 marked in {t_s:.2f}s; 200,000 sums in {t_l:.2f}s (paper 1.4s, 0.1s)")

# ---------- K4 self-composition to 120 ----------
t = time.time(); order, table, look = lego(ICOSIAN_GENS, qmul, IDENT); t_b = time.time() - t
n = len(order); idx = {x: i for i, x in enumerate(order)}
unit = all(qdot(x, x) == ONE for x in order)
latin = all(sorted(row) == list(range(n)) for row in table) and all(sorted(c) == list(range(n)) for c in zip(*table))
def pure(c):          # is this coordinate 0, or address +-1 with all its scale in the bands?
    return c == ZERO or any(c == Num(s, 0, nn, mm) for s in (1, -1) for nn in range(-3, 4) for mm in (0, 1))
say("K4a", n == 120 and unit and latin and all(qconj(x) in idx for x in order) and all(pure(c) for x in order for c in x),
    "i, j, the half-diagonal and (r + i + r^-1 j)/2 close on exactly 120 unit quaternions; Latin table; conjugates present; "
    "every one of the 480 coordinates is 0 or address +-1: all scale lives in the bands, the address never carries it")
top = sorted([v for v in {qdot(x, y) for x in order for y in order} if v < ONE], key=by_value)[-1]
say("K4b", top == Num(1, 0, 1, 1) and {sum(qdot(x, y) == top for y in order) for x in order} == {12},
    "largest inner product below 1 is address 1 at bands (r^1, 2^-1), i.e. r/2; every element has exactly 12 neighbours "
    "there: the 600-cell")
note("K4", f"table built once in {t_b:.2f}s (paper 2.1s)")

# ---------- K5 the shadow as ISIS3 cubes ----------
tmp = tempfile.gettempdir()
def at_half(c):                                                   # the coordinate written at the common bands (r^0, 2^-1)
    p, q, m = c.key(); assert m <= 1; return (p << (1 - m), q << (1 - m))
common = [[at_half(c) for c in x] for x in order]
core = [[[common[i][k][pq] for i in range(n)] for pq in (0, 1)] for k in range(4)]   # 4 anchor bands x (p, q) lines x 120
lab = {"BandBin": {"Name": "(1, i, j, k)"}, "Atom": {"RBand": 0, "HalfBand": 1, "Relation": "r^2 = r + 1", "Lines": "(p, q)"}}
f_el, f_tb = os.path.join(HERE, "icosians.cub"), os.path.join(HERE, "icosian_table.cub")
write_cub(f_el, core, "SignedWord", lab)
write_cub(f_tb, [table], "UnsignedByte", {"Atom": {"Shadow": "i * j -> index", "Elements": "icosians.cub"}})
c2, g2 = read_cub(f_el); t2, _ = read_cub(f_tb)
back = [tuple(Num(c2[k][0][i], c2[k][1][i], int(g2["Atom"]["RBand"]), int(g2["Atom"]["HalfBand"])) for k in range(4)) for i in range(n)]
pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(7200)]
t = time.time(); got = [t2[0][a][b] for a, b in pairs]; t_lk = time.time() - t
say("K5a", c2 == core and t2[0] == table and back == order and all(back[g] == qmul(back[a], back[b]) for g, (a, b) in zip(got, pairs)),
    f"the 120 as a 4-band cube (bands = anchors 1, i, j, k; lines = the address p, q; SignedWord), scale only in the label "
    f"(RBand 0, HalfBand 1), and the table as a 120x120 cube: written, read back identical, and 7,200 lookups from the "
    f"stored cube agree with multiplication")
note("K5", f"7,200 lookups from the cube in {t_lk:.4f}s (paper 0.016s). Cubes: {os.path.basename(f_el)} "
     f"({os.path.getsize(f_el)} bytes), {os.path.basename(f_tb)} ({os.path.getsize(f_tb)} bytes)")
cj = [idx[qconj(x)] for x in order]
rot = {tuple(table[table[l][x]][m] for x in range(n)) for l in range(n) for m in range(n)}
mir = {tuple(table[table[l][cj[x]]][m] for x in range(n)) for l in range(n) for m in range(n)}
say("K5b", len(rot) == 7200 and len(mir) == 7200 and not rot & mir,
    "from the table alone: x -> l x m gives 7,200 distinct maps (l,m and -l,-m coincide), the mirror x -> l conj(x) m "
    "7,200 more; 14,400 in all, the order of the 120-cell's group")

# ---------- K6 the golden ring, as bands ----------
units = {(p, q) for p in range(-300, 301) for q in range(-300, 301) if abs(p*p + p*q - q*q) == 1}
as_band = all(any(Num(p, q) == Num(s, 0, nn) for s in (1, -1) for nn in range(-15, 16)) for p, q in units)
steps = [Num(1, 0, -k) for k in range(61)]
say("K6", as_band and all(x.sign() > 0 for x in steps) and all(steps[k + 1] < steps[k] for k in range(60)),
    f"every one of the {len(units)} units of Z[r] with |p|,|q| <= 300 is address +-1 at some r-band: a unit is a pure band move; "
    f"address 1 at bands 0, -1, ..., -60 is positive and strictly falling: no smallest step, and the address never changes")

# ---------- K7 the 120-cell ----------
def even(p): return sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0
def signed(vals): return product(*[[v] if v == ZERO else [v, -v] for v in vals])
T2, S5 = Num(2), Num(-1, 2)
V = set()
for b, ev in [((ZERO, ZERO, T2, T2), 0), ((ONE, ONE, ONE, S5), 0), ((Num(1, 0, -2), R, R, R), 0), ((Num(1, 0, -1),) * 3 + (Num(1, 0, 2),), 0),
              ((ZERO, Num(1, 0, -2), ONE, Num(1, 0, 2)), 1), ((ZERO, Num(1, 0, -1), R, S5), 1), ((Num(1, 0, -1), ONE, R, T2), 1)]:
    for p in permutations(range(4)):
        if not ev or even(p):
            for s in signed([b[i] for i in p]): V.add(tuple(s))
V = list(V)
t = time.time(); vals = {}
for i in range(len(V)):
    for j in range(i + 1, len(V)):
        k = qdot(V[i], V[j]); vals[k] = vals.get(k, 0) + 1
near = sorted([v for v in vals if v < Num(8)], key=by_value)[-1]
say("K7", len(V) == 600 and all(qdot(v, v) == Num(8) for v in V) and vals[near] == 1200,
    f"the 120-cell from its coordinates, each a band or an integer address (r^-2, r^-1, r, r^2, 2, 2r-1): 600 vertices at "
    f"|v|^2 = 8, and 1200 nearest pairs, its edges ({time.time()-t:.1f}s)")

# ---------- K8 the 16 anchors as bands ----------
basis = all(cd(e(a), e(b)) == [SIGN[a][b] if k == a ^ b else 0 for k in range(16)] for a in range(16) for b in range(16))
ri = lambda: [rng.randint(-9, 9) for _ in range(16)]
A_, B_ = [ri() for _ in range(300)], [ri() for _ in range(300)]
f_s = os.path.join(tmp, "sedenions.cub")
write_cub(f_s, [[[A_[i][b] for i in range(300)]] for b in range(16)], "SignedWord", {"BandBin": {"Name": "(e0 .. e15)"}})
cs, _ = read_cub(f_s)
pix = [[cs[b][0][i] for b in range(16)] for i in range(300)]
say("K8", basis and all(mul(pix[i], B_[i]) == cd(A_[i], B_[i]) for i in range(300)),
    "a sedenion is one pixel across 16 anchor bands: anchor a times anchor b lands in band a xor b with one stored sign, the "
    "address untouched; 300 pixels read from a 16-band cube multiply by that table exactly as by doubling")

# ---------- K9 the zero pairs, read from the anchors ----------
def zero_pairs(prod_fn):
    zs, fp = 0, set()
    for a in range(16):
        for b in range(a + 1, 16):
            for s1 in (1, -1):
                x = [p + s1 * q for p, q in zip(e(a), e(b))]
                for c in range(16):
                    for d in range(c + 1, 16):
                        for s2 in (1, -1):
                            if not any(prod_fn(x, [p + s2 * q for p, q in zip(e(c), e(d))])): zs += 1; fp.add((a, b))
    return zs, fp
t = time.time(); zs, fp = zero_pairs(mul); t_z = time.time() - t
geo = {(o, s) for o in range(16) for s in range(16) if not o & 8 and s & 8 and o & 7 and s & 7 and (o & 7) != (s & 7)}
same_diag = [(o, o + 8) for o in range(8)]
say("K9a", zs == 336 and fp == geo and not any(p in fp for p in same_diag),
    f"all (e_a +- e_b)(e_c +- e_d) by lookup: {zs} zero products from {len(fp)} factor pairs, and the 42 are exactly: an inner "
    f"imaginary anchor with an outer imaginary anchor on another diagonal. No anchor closes with its own diagonal's partner, "
    f"and the real diagonal (corner 000) never closes ({t_z:.1f}s)")
octo = all(sq(x) * sq(y) == sq(cd(x, y)) for x, y in (([rng.randint(-9, 9) for _ in range(8)], [rng.randint(-9, 9) for _ in range(8)]) for _ in range(2000)))
dl = [sq(x) * sq(y) - sq(mul(x, y)) for x, y in ((ri(), ri()) for _ in range(2000))]
say("K9b", octo and all(dl), f"the comparator on squares, |x|^2|y|^2 - |xy|^2: 0 on 2000 octonion pairs (the 8 inner anchors); "
    f"nonzero on all 2000 sedenion pairs, {sum(d > 0 for d in dl)} positive, {sum(d < 0 for d in dl)} negative")

# ---------- K10 reading the missing point ----------
normals = [tuple(s if i == k else 0 for i in range(3)) for k in range(3) for s in (1, -1)]
reads = []
for a in range(1, 8):                                   # the 7 imaginary inner anchors look at the origin
    sight = tuple(-c for c in anchor(a))
    for nn in normals:
        dot = sum(x * y for x, y in zip(nn, sight))
        reads.append((3 * dot * dot == sum(x*x for x in nn) * sum(x*x for x in sight), dot))
say("K10", len(reads) == 42 and all(ok for ok, _ in reads) and sum(d > 0 for _, d in reads) == 21 and sum(d < 0 for _, d in reads) == 21,
    "each of the 7 imaginary anchors reads the missing point through its 6 oriented planes: 42 readings, every one on the "
    "integer relation 3(n.s)^2 = |n|^2 |s|^2 (the magic angle, no arccos), facing on 21 and away on 21")

# ---------- K11 the cone, and the walk in turn counts ----------
cone = [v for v in product(range(-12, 13), repeat=3) if any(v) and gcd(gcd(v[0], v[1]), v[2]) == 1 and v[0]*v[1] + v[1]*v[2] + v[2]*v[0] == 0]
P = lambda v: (v[2], v[0], v[1])
walk = [k * F(7, 9) % 1 for k in range(9)]
say("K11", all((a*(a+b))*(b*(a+b)) + (b*(a+b))*(-a*b) + (-a*b)*(a*(a+b)) == 0 for a in range(-40, 41) for b in range(-40, 41))
    and all(P(v) in cone for v in cone) and (1, 0, 0) in cone and P((1, 0, 0)) == (0, 1, 0)
    and len(set(walk)) == 9 and [k for k, w in enumerate(walk) if w * 3 == int(w * 3)] == [0, 3, 6]
    and [len({F(k, m) for k in range(m)} & {F(0), F(1, 4), F(1, 2), F(3, 4)}) for m in (9, 18, 36)] == [1, 2, 4],
    f"the cone about the diagonal is the integer conic ab+bc+ca = 0 ({len(cone)} primitive directions in [-12,12]^3), with "
    f"(a(a+b), b(a+b), -ab) on it for every a, b; the third-turn about the diagonal is x->y->z and keeps it; the step 7/9 "
    f"visits all 9 positions and lands on an axis at 0, 3, 6; lifting 9 -> 18 -> 36 holds 1, 2, then all 4 quarter-turns")

# ---------- K12 legos on turns, and pi as the half-turn ----------
ring, _, lk = lego([F(1, 9), F(1, 4)], turn, F(0))
i_ = (ZERO, ONE, ZERO, ZERO)
say("K12", len(ring) == 36 and min(x for x in ring if x) == F(1, 36) and lk(F(1, 4), F(1, 4)) == F(1, 2)
    and look(i_, i_) == (Num(-1), ZERO, ZERO, ZERO) and F(9, 2) * F(1, 9) == F(1, 2),
    "the nonagon brick and the oscillator brick close on 36 turns, the smallest 1/36; two quarter-turns are the half-turn, "
    "and in the stored table i*i = -1: pi is determined as the half-turn, omega^(9/2), before any digit")

# ---------- K13 the three-sevens tree ----------
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
say("K13a", all(c(k) == 3 * c(k - 1) + 1 for k in range(2, 400)) and all(first(p) == ordp(p) for p in range(5, 1000) if spf[p] == p)
    and (first(7), first(49), first(343)) == (6, 42, 294) and all(c(k) % 3 == 1 for k in range(1, 1000)) and c(6) == 364 == 2*2*7*13 == 7**3 + 3*7,
    "counts (3^k-1)/2, each 3x the last + 1; every odd prime 5..997 enters at the order of 3 mod p; 7 at step 6 "
    "(364 = 2^2*7*13 = 7^3+3*7), 7^2 at 42, 7^3 at 294; 3 never enters")
note("K13", f"2 enters at step {first(2)}, not at the order of 3 mod 2: the /2 in (3^k-1)/2")
t = time.time(); (co, ch), (to, th) = collatz_memo(2**20)
note("K13", f"Collatz, 2^19 odd starts below 2^20, check the table, else count: {100*co/to:.2f}% of odd steps counted, "
     f"{100*ch/th:.2f}% with halvings (paper 3.4%) ({time.time()-t:.1f}s)")
spf20 = marks(2**20)
say("K13b", c(13) == 797161 and all(prod(atoms(c(k), spf20)) == c(k) and all(spf20[p] == p for p in atoms(c(k), spf20)) for k in range(1, 14)),
    "every tree count up to 797161 resolves into atoms from the marks to 2^20")
ring28, _, _ = lego([F(1, 4), F(1, 7)], turn, F(0))
x14p1 = [1] + [0] * 13 + [1]
clock = len(ring28) == 28 and pmul([1, 0, 1], PHI28) == x14p1 and C28.x(14, 28) == C28.neg(C28.x(0, 28))
waves = C28.add(C28.mul(WAVE_R, WAVE_R), C28.mul(WAVE_I, WAVE_I)) == (4,) + (0,) * 11
bricks = [seven(0), seven(1), seven(2)]
seen = {TURN_ID: ""}; front = [TURN_ID]; counts = [1]; repeats = {}
t = time.time()
for depth in range(1, 9):
    nxt = []
    for g in front:
        for gi, b in enumerate(bricks):
            k = turn_mul(g, b); w = seen[g] + "xyz"[gi]
            if k in seen: repeats.setdefault(depth, []).append((w, seen[k])); continue
            seen[k] = w; nxt.append(k)
    front = nxt; counts.append(len(seen))
free = counts[:7] == [c(k) for k in range(1, 8)] and all(dd not in repeats for dd in range(1, 7))
at7 = sorted(repeats.get(7, [])) == [("xxxxxxx", ""), ("yyyyyyy", ""), ("zzzzzzz", "")]
say("K13c", clock and waves and free and at7 and counts[7] == c(8) - 3,
    f"derived: the 1/4 and 1/7 bricks close on 28 turns; the half-turn is -1, x^14 + 1 = (x^2 + 1) PHI28, so the 1/28 brick "
    f"obeys PHI28; a 1/7 turn is the 1/14 turn's two waves (real, and i = x^7), whose squares sum to 1. Turned about x, y, z: "
    f"{counts[:7]} orientations within 0..6 moves, (3^k-1)/2, no repeat; at 7 the only repeats are x^7 = y^7 = z^7 = 1, together")
note("K13c", f"8 moves: {counts[8]} orientations, {len(repeats.get(8, []))} repeats, all y x^7 = y and the like ({time.time()-t:.1f}s). "
     "Reading different polynomials as different turns needs PHI28 not to split: shown by counting in K16. That they never "
     "close at any depth: K17")

# ---------- appendix steps, as counts and addresses ----------
grid = [(x, y) for x in range(7) for y in range(7)]
d2 = lambda a, b: sum((p - q)**2 for p, q in zip(a, b))
say("A2", not any(d2(a, b) == d2(b, cc) == d2(a, cc) > 0 for i, a in enumerate(grid) for j, b in enumerate(grid[i+1:], i+1) for cc in grid[j+1:])
    and {d2(a, b) for a in inner for b in inner if a != b and prod(a) == 1 and prod(b) == 1} == {8},
    "no equilateral triangle in the 7x7 integer grid; the inner anchors with an even number of minus signs form a regular "
    "tetrahedron (doubled edge^2 = 8)")
RI = Num(1, 0, -1)
dod = set(product((ONE, -ONE), repeat=3))
for a, b in product((RI, -RI), (R, -R)): dod |= {(ZERO, a, b), (a, b, ZERO), (b, ZERO, a)}
dod = list(dod)
dd = lambda a, b: sum(((p - q) * (p - q) for p, q in zip(a, b)), ZERO)
edge = sorted({dd(a, b) for a in dod for b in dod if a != b}, key=by_value)[0]
cubes = set()
for v in dod:
    nb = [u for u in dod if dd(u, v) == Num(4)]
    for x in nb:
        for y in nb:
            for z in nb:
                if len({x, y, z}) == 3 and dd(x, y) == dd(y, z) == dd(x, z) == Num(8):
                    cs_ = frozenset([v, x, y, z] + [tuple(-t for t in u) for u in (v, x, y, z)])
                    if all(u in dod for u in cs_): cubes.add(cs_)
say("A5", len(dod) == 20 and R - ONE == RI and edge == Num(1, 0, -2) * Num(4) and {sum(dd(a, b) == edge for b in dod) for a in dod} == {3}
    and len(cubes) == 5 and {sum(u in cb for cb in cubes) for u in dod} == {2},
    "roofs on the cube at height r - 1, which is address 1 one r-band down: 20 vertices, edges^2 = 4 at band -2, 3 per vertex: "
    "the dodecahedron; 5 cubes share its corners, each corner in 2")
say("A9", 9 - 2 == 7 and 18 - 7 == 11 and 11 - 7 == 2 * 2,
    "counted in the plane steps (18 to the turn, 9 to the half-turn, the nonagon's turn is 2): its angle is 9 - 2 = 7 steps, "
    "its outer angle 18 - 7 = 11, the clearance 11 - 7 = 4, two of its turns")
say("A12", sum(1 for a in range(16) for b in range(a + 1, 16) if bin(a ^ b).count("1") == 1) == 32,
    "the 16 anchor indices, joined where they differ in one bit: 32 edges, the tesseract")
say("A14", 10 % 9 == 1 and 2 * 5 % 9 == 1 and 7 % 9 == -2 % 9 and 360 % 7, "on the nonagon's clock 10 = 1, 2 and 5 are inverses, 7 = -2; 7 does not divide 360")
# ---------- K14 the digits of pi, from the nonagon's clock ----------
w9, w9b = C9.x(1, 9), C9.x(8, 9)
two_c = C9.add(w9, w9b)
rel9 = pmul([-1, 0, 0, 1], PHI9) == [-1] + [0] * 8 + [1]                          # (x^3 - 1)(x^6 + x^3 + 1) = x^9 - 1
cubic = C9.add(C9.mul(C9.mul(two_c, two_c), two_c), C9.neg(C9.add(two_c, two_c, two_c)), C9.x(0, 9)) == C9.zero()
chord = C9.mul(C9.add(C9.x(0, 9), C9.neg(w9)), C9.add(C9.x(0, 9), C9.neg(w9b))) == C9.add(C9.x(0, 9), C9.x(0, 9), C9.neg(two_c))
fc = lambda v: 8 * v**3 - 6 * v + 1
grid10 = [F(k, 20) for k in range(-20, 21)]
roots3 = sum(1 for a, b in zip(grid10, grid10[1:]) if fc(a) * fc(b) < 0) == 3 and fc(F(3, 4)) < 0 < fc(F(4, 5)) \
    and all(fc(v) > 0 for v in grid10 if v >= F(4, 5))            # a cubic has at most 3 roots; the largest is in (3/4, 4/5)
paper = [F(30782, 10**4), F(31257, 10**4), F(31376, 10**4), F(31406, 10**4), F(31413, 10**4)]
L = pi_lifts()
rounds = all(v - F(1, 20000) <= lo and hi < v + F(1, 20000) for (nn, lo, hi), v in zip(L, paper))
say("K14", rel9 and cubic and chord and roots3 and rounds and all(L[i][2] < L[i + 1][1] for i in range(4)),
    "derived: the 1/9 turn obeys omega^6 + omega^3 + 1 (its cube is the third-turn); its real wave 2c = omega + omega^-1 obeys "
    "(2c)^3 - 3(2c) + 1 = 0 on that clock, and its chord^2 = (1-omega)(1-omega^-1) = 2 - 2c; the cubic has three real roots, the "
    "smallest turn takes the largest, in (3/4, 4/5); each lift halves the turn, (y + y^-1)^2 = 2 + (z + z^-1). The 9, 18, 36, "
    "72, 144-gons' half-perimeters, held by counting squares, round to 3.0782, 3.1257, 3.1376, 3.1406, 3.1413 and rise strictly")

# ---------- K15 E8: the 120 and their image under the other golden root ----------
E8 = list(order) + [tuple(Num(1, 0, -1) * c for c in x) for x in order]        # times -1/r: one r-band down, sign absorbed
B = lambda a, b: euclid(qdot(a, b))
t = time.time()
norms = {B(a, a) for a in E8}; gram = {B(a, b) for a in E8 for b in E8}
E8set = set(E8)
def reflect(a, b):                                   # b - 2 B(a,b)/B(a,a) a, with B(a,a) = 1
    k = Num(int(2 * B(a, b))); return tuple(bb - k * aa for aa, bb in zip(a, b))
refl = all(reflect(a, b) in E8set for a in E8 for b in E8)
nb56 = {sum(B(a, b) == F(1, 2) for b in E8) for a in E8}
say("K15", len(E8set) == 240 and norms == {1} and gram <= {F(-1), F(-1, 2), F(0), F(1, 2), F(1)} and refl and nb56 == {56},
    f"the 120 icosians and the same 120 one r-band down (times -1/r, the other golden root; addresses untouched): 240 distinct, "
    f"all of norm 1, inner products in {{0, +-1/2, +-1}}, closed under every reflection, 56 neighbours at 1/2 each: the roots of E8 "
    f"({time.time()-t:.1f}s)")
# ---------- K16 the 28-clock's relation does not split: by counting ----------
from math import comb
def monic_divisors(f, p, d):
    return [c + [1] for c in map(list, product(range(p), repeat=d)) if not any(pdivmod(f, c + [1], p)[1])]
def pprod(ps, p=None):
    out = [1]
    for g in ps: out = pmul(out, g)
    return [v % p for v in out] if p else out
t = time.time()
fac = {}
for p, deg in ((3, 6), (5, 6), (13, 2)):
    small = [g for d in range(1, deg) for g in monic_divisors(PHI28, p, d)]
    fac[p] = monic_divisors(PHI28, p, deg)
    fac[p + 1000] = small
split_ok = (not fac[1003] and not fac[1005] and not fac[1013] and len(fac[3]) == 2 and len(fac[5]) == 2 and len(fac[13]) == 6
            and pprod(fac[3], 3) == [v % 3 for v in PHI28] and pprod(fac[5], 5) == [v % 5 for v in PHI28] and pprod(fac[13], 13) == [v % 13 for v in PHI28])
def crt(a3, a5, a13):
    v = next(x for x in range(195) if x % 3 == a3 and x % 5 == a5 and x % 13 == a13)
    return v - 195 if v > 97 else v
cands = []
for g3 in fac[3]:
    for g5 in fac[5]:
        for trio in [(a, b, c_) for i, a in enumerate(fac[13]) for j, b in enumerate(fac[13][i+1:], i+1) for c_ in fac[13][j+1:]]:
            g13 = pprod(trio, 13)
            cands.append([crt(g3[k], g5[k], g13[k]) for k in range(7)])
in_bound = [g for g in cands if all(abs(g[k]) <= comb(6, k) for k in range(7))]
divides = [g for g in cands if not any(pdivmod(PHI28, g)[1])]
say("K16", split_ok and len(cands) == 80 and not divides,
    f"PHI28 does not split, by counting: mod 3 and mod 5 it breaks only into two pieces of degree 6, mod 13 into six of degree 2, "
    f"so an integer piece (Gauss: a split over pairs of counts clears to one over counts) has degree 6 and is fixed by its "
    f"residues mod 3*5*13 = 195 > 2*20 (every root is a turn, of size 1, so |coefficient| <= C(6,k) <= 20). All 80 "
    f"candidates tried ({len(in_bound)} inside the bound): none divides ({time.time()-t:.1f}s). K13c's count stands on this")

# ---------- K17 the three sevens never close ----------
units28 = [a for a in range(1, 28) if gcd(a, 28) == 1]
xy = _qc(seven(0)[0], seven(1)[0])                      # numerators over the 2-band 2
w2 = C28.mul(WAVE_R, WAVE_R)                            # 2 * Re(xy) = w2 / 2
def conj28(c, a):                                       # the same turn read on the clock whose brick is x^a
    out = [0] * 28
    for k, v in enumerate(c): out[a * k % 28] += v
    return C28.red(out)
normK = (1,) + (0,) * 11
for a in units28: normK = C28.mul(normK, conj28(w2, a))
K = normK[0]
say("K17", xy[0] == w2 and all(v == 0 for v in normK[1:]) and K % 2**12 != 0,
    f"x then y (1/7 turns about two axes) has real part w/4 with w = (2 x real wave)^2; a turn that ever closes has 2 x real part "
    f"= a sum of two turns, whose norm is a whole count. Here 2 x real part = w/2, and its norm over the 12 conjugate clocks is "
    f"{K}/2^12, not whole: x then y never closes, so the three sevens lie in no finite set of turns")

# ---------- K18 the E8 form is forced by the band ----------
forms = [(s0, s1) for s0 in range(-10, 11) for s1 in range(-10, 11)
         if s0 * 1 == 1 and s0 * Num(1, 0, -2).key()[0] + s1 * Num(1, 0, -2).key()[1] == 1]
say("K18", forms == [(1, 1)] and all(euclid(qdot(a, a)) == 1 for a in E8),
    "a form read on the address, S(p + q r) = s0 p + s1 q, with the unit at norm 1 and its band-down copy (norm r^-2 = 2 - r) "
    "at the same norm: s0 = 1 and 2 s0 - s1 = 1, so S = p + q, the only one. It is the form K15 used")

# ---------- K19 which root is the nonagon: seen by winding ----------
four = C9.red([4])
sk = lambda k: C9.add(C9.red([2]), C9.neg(C9.add(C9.x(k, 9), C9.x(-k, 9))))
dbl = lambda s: C9.mul(s, C9.add(four, C9.neg(s)))
cycle = dbl(sk(1)) == sk(2) and dbl(sk(2)) == sk(4) and dbl(sk(4)) == sk(8) == sk(1)
rts = cubic_roots()
deep = [lifts(2 - 2 * hi, 2 - 2 * lo, 9, 30) for lo, hi in rts]
last = [(d[-1][1], d[-1][2]) for d in deep]
by_size = sorted(range(3), key=lambda i: last[i][0])
a1, a2, a4 = (last[i] for i in by_size)
tol = F(1, 10**9)
ratio = (2 - tol) * a1[1] < a2[0] and a2[1] < (2 + tol) * a1[0] and (4 - tol) * a1[1] < a4[0] and a4[1] < (4 + tol) * a1[0]
used = nonagon_wave()
say("K19", cycle and len(rts) == 3 and ratio and max(rts[by_size[0]][0], used[0]) <= min(rts[by_size[0]][1], used[1]),
    "doubling a turn doubles on the 9-clock, s -> s(4 - s), cycling the three chords 1/9 -> 2/9 -> 4/9 -> 8/9 = -1/9. Lifted 30 "
    "times, the three roots wind 1 : 2 : 4 (within 10^-9): one brick, two, four. The root winding once, the nonagon, is the "
    "largest wave, the one K14 used")

print(f"\n{sum(RES)}/{len(RES)} passed")
