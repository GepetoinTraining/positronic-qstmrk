exec(open('t6_octonion.py').read().split("results = []")[0])   # reuse table, systems, unit_set, omul2
from itertools import permutations
from collections import Counter
# 1) symmetries of the table among coordinate relabelings (fixing e0), exact signs
auts = [p for p in permutations(range(1,8))
        if all(MUL[(p[i-1],p[j-1])] == (MUL[(i,j)][0], (p[MUL[(i,j)][1]-1] if MUL[(i,j)][1] else 0))
               for i in range(1,8) for j in range(1,8) if i != j)]
print("sign-exact symmetries of the table:", len(auts))
def img(S, p):
    m = {0:0, **{i: p[i-1] for i in range(1,8)}}
    return frozenset(frozenset(m[x] for x in b) for b in S)
units = {S: unit_set(S) for S in systems}
closed = {S for S in systems if all(tuple(x//2 for x in omul2(u,v)) in units[S] and all(x%2==0 for x in omul2(u,v))
                                     for u in list(units[S])[:60] for v in units[S])}
seen, orbits = set(), []
for S in systems:
    if S in seen: continue
    orb = {img(S, p) for p in auts}; seen |= orb
    orbits.append((len(orb), sum(1 for T in orb if T in closed), frozenset(BASE) in orb))
print("orbits of the 30 (size, closed inside, contains aligned):", orbits)
# 2) catch graph: for each non-closed set, which sets catch its half-integer fallout
catch = Counter()
for S in systems:
    if S in closed: continue
    U = units[S]
    for u in U:
        for v in U:
            w = omul2(u, v)
            if any(x % 2 for x in w): continue
            w2 = tuple(x//2 for x in w)
            if w2 in U: continue
            for T in systems:
                if w2 in units[T]: catch[("closed" if T in closed else "open")] += 1
print("who catches the half-integer fallout (counted per catching set):", dict(catch))
# 3) rotary kernel: is a loop of calls a single call?  L_u L_v vs L_{uv} on a closed set
S = next(iter(closed)); U = list(units[S])
def L(u, x): return tuple(y//2 for y in omul2(u, x))
diff = sum(1 for u in U[:40] for v in U[:40] if any(L(u, L(v, x)) != L(L(u, v), x) for x in U))
print(f"pairs (u,v) where applying v then u differs from applying the single unit uv: {diff} of 1600")
