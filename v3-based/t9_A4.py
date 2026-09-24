exec(open('/home/claude/t6/t6_octonion.py').read().split("results = []")[0])
from itertools import permutations
auts=[p for p in permutations(range(1,8)) if all(MUL[(p[i-1],p[j-1])]==(MUL[(i,j)][0],(p[MUL[(i,j)][1]-1] if MUL[(i,j)][1] else 0)) for i in range(1,8) for j in range(1,8) if i!=j)]
def img(S,p):
    m={0:0,**{i:p[i-1] for i in range(1,8)}}; return frozenset(frozenset(m[x] for x in b) for b in S)
def comp(p,q): return tuple(p[q[i]-1] for i in range(7))
def order(p):
    k,r=1,p
    while r!=tuple(range(1,8)): r=comp(p,r); k+=1
    return k
units={S:unit_set(S) for S in systems}
def closed(S):
    U=units[S]; return all(all(x%2==0 for x in omul2(u,v)) and tuple(x//2 for x in omul2(u,v)) in U for u in U for v in U)
C=[S for S in systems if closed(S)]
done_types=set()
for p in [q for q in auts if order(q)==3]:
    hub=[S for S in C if img(S,p)==S][0]
    for S in C:
        if S==hub: continue
        A,B,Cc=S,img(S,p),img(img(S,p),p)
        UA,UB,UC=units[A],units[B],units[Cc]
        typ=len(UA&UB&UC)
        if typ in done_types: continue
        done_types.add(typ)
        by_hub=by_other=0
        for u in UA:
            for v in UB:
                w=omul2(u,v)
                if any(x%2 for x in w): continue
                w2=tuple(x//2 for x in w)
                if w2 in UA or w2 in UB or w2 in UC: continue
                if w2 in units[hub]: by_hub+=1
                elif any(w2 in units[T] for T in C): by_other+=1
        print(f"A4 triple type (shared by all three = {typ}): caught by hub {by_hub}, by a non-hub closed set {by_other}")
    if len(done_types)==2: break
