# Does the 6-bit signed-zero tryte carry the binary tetrahedral group (the Hurwitz units),
# with the double-cover sign -1 realised as the magnitude flip (corner<->centre, edge<->face)?
# Lane states in ring order: [+1, -1, +0, -0].  sigma = shift by one: +1->-1->+0->-0->+1.
from itertools import product
L = ['+1', '-1', '+0', '-0']
def d4():                      # symmetries of the 4-ring, as permutations of lane-state indices
    out = []
    for r in range(4):
        out.append(tuple((x + r) % 4 for x in range(4)))
        out.append(tuple((-x + r) % 4 for x in range(4)))
    return out
D4 = d4()
sigma = D4[2]                   # shift by 1
flip_lane = tuple((x + 2) % 4 for x in range(4))
assert [L[flip_lane[i]] for i in range(4)] == ['+0', '-0', '+1', '-1']   # magnitude flip, sign kept
assert tuple(sigma[sigma[x]] for x in range(4)) == flip_lane             # sigma^2 = flip: i^2 = -1

# group element: (perm of lanes p, per-lane D4 maps (a0,a1,a2)); acts on state s=(s0,s1,s2):
#   new[p[l]] = a[l][s[l]]
def act(g, s):
    p, a = g; new = [None]*3
    for l in range(3): new[p[l]] = a[l][s[l]]
    return tuple(new)
STATES = list(product(range(4), repeat=3))
def as_perm(g): return tuple(STATES.index(act(g, s)) for s in STATES)
def comp(P, Q): return tuple(P[Q[x]] for x in range(len(Q)))       # P after Q
def inv(P):
    r = [0]*len(P)
    for i, x in enumerate(P): r[x] = i
    return tuple(r)
ID = tuple(range(64))
F = as_perm(((0,1,2), (flip_lane,)*3))

diag = [as_perm(((0,1,2), a)) for a in product(D4, repeat=3)]
cyc  = [as_perm(((1,2,0), a)) for a in product(D4, repeat=3)]   # lane x->y->z
I_cands = [g for g in diag if comp(g, g) == F]
found = None
for w in cyc:
    if comp(w, comp(w, w)) != F: continue
    wi = inv(w)
    for i in I_cands:
        j = comp(w, comp(i, wi)); k = comp(w, comp(j, wi))
        if comp(j, j) == F and comp(k, k) == F and comp(i, comp(j, k)) == F and comp(w, comp(k, wi)) == i:
            found = (i, w); break
    if found: break
print("binary tetrahedral action with -1 = magnitude flip found:", bool(found))
if found:
    i, w = found
    G = {ID}; frontier = [ID]
    while frontier:
        nxt = []
        for g in frontier:
            for h in (i, w):
                x = comp(h, g)
                if x not in G: G.add(x); nxt.append(x)
        frontier = nxt
    print("group order:", len(G), "(2T = 24)")
    centre = [g for g in G if all(comp(g, h) == comp(h, g) for h in G)]
    print("centre = {id, flip}:", set(centre) == {ID, F})
    seen, orbits = set(), []
    for s in range(64):
        if s in seen: continue
        orb = {g[s] for g in G}; seen |= orb
        cls = lambda t: sum(1 for x in STATES[t] if x >= 2)   # zero lanes
        orbits.append((len(orb), sorted({cls(t) for t in orb})))
    print("orbits (size, zero-lane classes inside):", orbits)
    print("free action (every orbit size 24)?", all(o[0] == 24 for o in orbits))

# Wider search: let i, j, k permute lanes too (full wreath product D4 wr S3, order 3072).
from itertools import permutations
ALL = [as_perm((p, a)) for p in permutations(range(3)) for a in product(D4, repeat=3)]
sq = [g for g in ALL if comp(g, g) == F]
W  = [g for g in ALL if comp(g, comp(g, g)) == F and comp(g, g) != ID]
print("elements squaring to flip:", len(sq), "| candidates for w (w^3 = flip):", len(W))
hit = None
sqset = set(sq)
for w in W:
    wi = inv(w)
    for i in sq:
        j = comp(w, comp(i, wi)); k = comp(w, comp(j, wi))
        if j in sqset and k in sqset and comp(w, comp(k, wi)) == i and \
           (comp(i, comp(j, k)) == F or comp(k, comp(j, i)) == F) and comp(i, j) != comp(j, i):
            hit = (i, w); break
    if hit: break
print("2T with -1 = flip anywhere in D4 wr S3:", bool(hit))
