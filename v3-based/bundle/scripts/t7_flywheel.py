import sys; 
exec(open('t6_octonion.py').read().split("results = []")[0])
from itertools import permutations
from collections import Counter
auts = [p for p in permutations(range(1,8))
        if all(MUL[(p[i-1],p[j-1])] == (MUL[(i,j)][0], (p[MUL[(i,j)][1]-1] if MUL[(i,j)][1] else 0))
               for i in range(1,8) for j in range(1,8) if i != j)]
def img(S, p):
    m = {0:0, **{i: p[i-1] for i in range(1,8)}}
    return frozenset(frozenset(m[x] for x in b) for b in S)
def compose(p, q): return tuple(p[q[i]-1] for i in range(7))
def order(p):
    k, r = 1, p
    while r != tuple(range(1,8)): r = compose(p, r); k += 1
    return k
units = {S: unit_set(S) for S in systems}
def is_closed(S):
    U = units[S]
    for u in U:
        for v in U:
            w = omul2(u, v)
            if any(x % 2 for x in w) or tuple(x//2 for x in w) not in U: return False
    return True
closed = [S for S in systems if is_closed(S)]
print("closed sets:", len(closed))
ord3 = [p for p in auts if order(p) == 3]
print("order-3 symmetries:", len(ord3))
# H1
types = Counter()
for p in ord3:
    seen, cyc = set(), []
    for S in closed:
        if S in seen: continue
        c = [S]; T = img(S, p)
        while T != S: c.append(T); T = img(T, p)
        seen |= set(c); cyc.append(len(c))
    types[tuple(sorted(cyc))] += 1
print("H1 cycle types of order-3 symmetries on the 7 closed sets:", dict(types))
# H2 / H3 / E1 on every distinct triple
triples = set()
for p in ord3:
    for S in closed:
        A, B, C = S, img(S, p), img(img(S, p), p)
        if A != B: triples.add((A, B, C))
print("distinct ordered triples:", len(triples))
fixed_sets = {S for S in closed if all(img(S, p) == S for p in ord3[:1])}
res = Counter(); land = Counter()
for (A, B, C) in sorted(triples, key=str)[:6]:
    UA, UB, UC = units[A], units[B], units[C]
    esc = 0
    for u in UA:
        for v in UB:
            w = omul2(u, v)
            if any(x % 2 for x in w): esc += 1; land["quarter-integer rung"] += 1; continue
            w2 = tuple(x//2 for x in w)
            if w2 in UA or w2 in UB or w2 in UC: continue
            esc += 1
            where = [T for T in closed if T not in (A, B, C) and w2 in units[T]]
            if where: land["another closed set"] += 1
            elif any(w2 in units[T] for T in systems): land["an open labeling"] += 1
            else: land["half-integer, in no set"] += 1
    res[esc] += 1
print("H2/H3 escapes per triple (count -> #triples checked):", dict(res), "| 1/12 would be 4800")
print("E1 where escapes land:", dict(land))
