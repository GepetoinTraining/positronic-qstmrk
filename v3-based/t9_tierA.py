import random
R = random.Random(2026)
# ---------- A1 receipts stop drift
def gate(v, d):
    w = (v[0]+d[0], v[1]+d[1]); return (w, True) if all(0 <= x <= 2 for x in w) else (v, False)
H = [(1,-1), (-1,1)]                     # 7/11 and 11/7 on (v7, v11)
ok_log = drift_nolog = 0
for _ in range(1000):
    start = (R.randint(0,2), R.randint(0,2)); v = start; seq = []
    for _ in range(200):
        d = R.choice(H); v, applied = gate(v, d); seq.append((d, applied))
    a = v                                  # with refusal log
    for d, applied in reversed(seq):
        if applied: a, _ = gate(a, (-d[0], -d[1]))
    ok_log += (a == start)
    b = v                                  # without log: blindly invert every step
    for d, _ in reversed(seq): b, _ = gate(b, (-d[0], -d[1]))
    drift_nolog += (b != start)
print(f"A1 (a) with refusal log: {ok_log}/1000 return exactly | (b) without log: {drift_nolog}/1000 drift")
# ---------- walk helpers
def fwd(s, p): return (s[1], s[2], (-s[2] + s[1] + 2*s[0]) % p)
def bwd(s, p):                            # solve for the dropped register: 2*a0 = a3 + a2 - a1
    a1, a2, a3 = s; i2 = pow(2, -1, p); return (((a3 + a2 - a1) * i2) % p, a1, a2)
def orbits(p):
    seen = {}; halves = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                s = (a, b, c)
                if s == (0,0,0) or s in seen: continue
                o = []; t = s
                while True:
                    seen[t] = (len(halves), len(o)); o.append(t); t = fwd(t, p)
                    if t == s: break
                halves.append(o)
    return halves, seen
# ---------- A2 pointerization
for p in (19, 29, 41):
    halves, addr = orbits(p)
    states = [s for o in halves for s in o]
    total = p**3 - 1
    bij = len(states) == total == len(set(states)) and all(halves[h][i] == s for s, (h, i) in addr.items())
    print(f"A2 p={p}: halves={len(halves)} sizes={sorted({len(o) for o in halves})}  bijective={bij}  "
          f"collisions={len(states)-len(set(states))} gaps={total-len(set(states))}")
    # ---------- A3 exact rewind
    fails = half_change = 0
    for _ in range(2000):
        s = states[R.randrange(total)]; h0 = addr[s][0]; t = s
        for _ in range(1000):
            t = fwd(t, p); half_change += (addr[t][0] != h0)
        for _ in range(1000): t = bwd(t, p)
        fails += (t != s)
    print(f"A3 p={p}: rewind failures {fails}/2000, half changes {half_change}")
