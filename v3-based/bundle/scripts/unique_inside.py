import sys
exec(open('t6_octonion.py').read().split("results = []")[0])
from itertools import permutations
auts = [p for p in permutations(range(1,8))
        if all(MUL[(p[i-1],p[j-1])] == (MUL[(i,j)][0], (p[MUL[(i,j)][1]-1] if MUL[(i,j)][1] else 0))
               for i in range(1,8) for j in range(1,8) if i != j)]
def img(S, p):
    m = {0:0, **{i: p[i-1] for i in range(1,8)}}; return frozenset(frozenset(m[x] for x in b) for b in S)
def compose(p,q): return tuple(p[q[i]-1] for i in range(7))
def order(p):
    k,r=1,p
    while r!=tuple(range(1,8)): r=compose(p,r); k+=1
    return k
units = {S: unit_set(S) for S in systems}
def is_closed(S):
    U=units[S]; return all(all(x%2==0 for x in omul2(u,v)) and tuple(x//2 for x in omul2(u,v)) in U for u in U for v in U)
closed=[S for S in systems if is_closed(S)]
ord3=[p for p in auts if order(p)==3]
seen=set()
for p in ord3:
    for S in closed:
        A,B,C=S,img(S,p),img(img(S,p),p)
        if A==B: continue
        key=frozenset((A,B,C))
        if key in seen: continue
        seen.add(key)
        UA,UB,UC=units[A],units[B],units[C]
        union=UA|UB|UC
        caught=set(); caught_n=0
        for u in UA:
            for v in UB:
                w=omul2(u,v)
                if any(x%2 for x in w): continue
                w2=tuple(x//2 for x in w)
                if w2 in union: caught.add(w2); caught_n+=1
        print(f"triple: |A∪B∪C|={len(union)}  |A∩B|={len(UA&UB)} |A∩B∩C|={len(UA&UB&UC)}  "
              f"caught products={caught_n}  UNIQUE caught values={len(caught)}  squared={len(caught)**2}")
print("distinct unordered triples:", len(seen))
