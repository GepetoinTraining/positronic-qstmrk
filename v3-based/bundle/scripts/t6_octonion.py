# T6 run. Integers scaled by 2 (x2 = 2*x). No floats.
from itertools import permutations, product, combinations
LINES = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
MUL = {}                                   # (i,j) -> (sign, k) for basis e_i e_j
for a in range(8):
    MUL[(0,a)] = (1,a); MUL[(a,0)] = (1,a)
for a in range(1,8): MUL[(a,a)] = (-1,0)
for (a,b,c) in LINES:
    for (x,y,z) in [(a,b,c),(b,c,a),(c,a,b)]:
        MUL[(x,y)] = (1,z); MUL[(y,x)] = (-1,z)

def omul2(u, v):          # inputs scaled by 2 -> returns 4*(uv); caller rescales
    out = [0]*8
    for i in range(8):
        if u[i] == 0: continue
        for j in range(8):
            if v[j] == 0: continue
            s, k = MUL[(i,j)]; out[k] += s*u[i]*v[j]
    return out
def norm4(u): return sum(x*x for x in u)   # = 4*N for scaled-by-2 vectors

# sanity: the table is a real octonion algebra (alternative + norm multiplicative) on random ints
import random
rnd = random.Random(7)
for _ in range(300):
    a = [rnd.randint(-5,5) for _ in range(8)]; b = [rnd.randint(-5,5) for _ in range(8)]
    assert sum(x*x for x in omul2(a,b)) == norm4(a)*norm4(b), "table not a composition algebra"
    aa = omul2(a,a); assert omul2(aa,b) == omul2(a, omul2(a,b)), "not left-alternative"

BASE = [frozenset({0,*L}) for L in LINES]
BASE += [frozenset(set(range(8)) - b) for b in BASE]
systems = set()
for p in permutations(range(8)):
    systems.add(frozenset(frozenset(p[x] for x in b) for b in BASE))
systems = sorted(systems, key=lambda s: sorted(sorted(b) for b in s))
aligned = frozenset(BASE)
print("Steiner systems S(3,4,8):", len(systems), "| aligned among them:", aligned in systems)

def unit_set(S):
    U = set()
    for a in range(8):
        for s in (2,-2):
            v = [0]*8; v[a] = s; U.add(tuple(v))
    for b in S:
        b = sorted(b)
        for signs in product((1,-1), repeat=4):
            v = [0]*8
            for idx, sg in zip(b, signs): v[idx] = sg
            U.add(tuple(v))
    return U

def run(S):
    U = unit_set(S); Ul = list(U); outside = []
    for u in Ul:
        for v in Ul:
            w = omul2(u, v)                        # = 4*uv ; want 2*uv
            if any(x % 2 for x in w): outside.append((u, v, w, "quarter")); continue
            w2 = tuple(x//2 for x in w)
            if w2 not in U: outside.append((u, v, w2, "half"))
    return U, outside

results = []
for S in systems:
    U, out = run(S)
    results.append((S, U, out))
closed = [r for r in results if not r[2]]
print("H1 closed labelings:", len(closed), "of", len(systems))
al = [r for r in results if r[0] == aligned][0]
print("H2 aligned labeling closed?", not al[2], "| its outside products:", len(al[2]))
# H3 on each closed set
for S, U, _ in closed:
    ok = all({tuple(x//2 for x in omul2(u, v)) for v in U} == U for u in U)
    nm = all(norm4(tuple(x//2 for x in omul2(u, v))) * 4 == norm4(u)*norm4(v) for u in U for v in U)
    print("H3 closed set: every left multiplication permutes U:", ok, "| norms multiply exactly:", nm)
    break
print("H4 count == 7:", len(closed) == 7)
# E1 fallout census for non-closed sets
all_units = {S: U for S, U, _ in results}
from collections import Counter
agg = Counter()
for S, U, out in results:
    if not out: continue
    for u, v, w, kind in out:
        if kind == "quarter": agg["quarter-integer coords"] += 1
        elif any(w in all_units[T] for T in systems if T != S): agg["inside another labeling's set"] += 1
        else: agg["half-integer, in no labeling's set"] += 1
print("E1 fallout over all non-closed labelings:", dict(agg))
print("E1 aligned labeling fallout:", Counter(k for *_, k in al[2]))
