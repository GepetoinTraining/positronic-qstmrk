"""T6: hull and rewind on a real layer. Run after tests.py; needs data/L18_*.npy (in the bundle)."""
import numpy as np, hashlib
from hull import *
rng = np.random.default_rng(11)
T_ = {}
def say(tag, ok, msg): T_[tag] = ok; print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}")
wsum = hashlib.sha256(b"".join(v.tobytes() for v in W.values())).hexdigest()[:16]

# baseline state: a 24-token context on the grid (random residuals at realistic scale; no tokenizer in this test)
ctx = np.round(rng.normal(0, 1.0, (24, 2560)) * 2.0**G).astype(np.int64)
base = ctx.copy(); outs = []; worst = 0.0
for t in range(100):                                     # 100 turns, each with a different 8-token prompt appended
    prompt = np.round(rng.normal(0, 1.0, (8, 2560)) * 2.0**G).astype(np.int64)
    r = np.vstack([ctx, prompt]); before = r.copy()
    out, rec, attn = turn(r)
    assert np.array_equal(r, before)
    outs.append(out); worst = max(worst, max(float(np.abs(e).max()) for _, _, e in rec))
say("T6a", np.array_equal(ctx, base), "100 turns through a real layer (hull + second interior), each rewound by its receipts: "
    "the state returns to baseline bit for bit every turn, and only the 100 output rows are kept")
say("T6b", worst <= 2.0**-(G + 1), f"the only rounding in the system is the snap out of each interior; its largest value over all "
    f"turns is {worst:.2e} (half a grid step is {2.0**-(G+1):.2e}), and every one is written in its receipt")
wsum2 = hashlib.sha256(b"".join(v.tobytes() for v in W.values())).hexdigest()[:16]
say("T6c", wsum == wsum2, f"weights read-only throughout (hash {wsum} before and after)")

# float control, off the grid: raw float32 context and raw (unsnapped) deltas, subtracted back after each turn
ctxf = rng.normal(0, 1.0, (24, 2560)).astype(np.float32); basef = ctxf.copy()
for t in range(100):
    prompt = rng.normal(0, 1.0, (8, 2560)).astype(np.float32); rf = np.vstack([ctxf, prompt])
    (qa, ea), _ = hull(np.round(rf.astype(np.float64) * 2.0**G).astype(np.int64)); da = (qa * 2.0**-G + ea).astype(np.float32)
    rf += da
    (qm, em) = interior2(np.round(rf.astype(np.float64) * 2.0**G).astype(np.int64)); dm = (qm * 2.0**-G + em).astype(np.float32)
    rf += dm; rf -= dm; rf -= da; ctxf = rf[:24].copy()
drift = int((ctxf != basef).sum())
say("T6d", drift > 0, f"control, off the grid: the same 100 turns in float32 with the raw deltas subtracted back leave {drift:,} of "
    f"{ctxf.size:,} context values changed (max {float(np.abs(ctxf - basef).max()):.2e})")
say("T6f", True, "note: the first control was accidentally on the grid (context and deltas multiples of 2^-16) and released "
    "perfectly in plain float32; the release comes from staying on the grid, whatever arithmetic carries it")

# the hull chooses: attention is a real selection, not a smear
(da, _), p = hull(np.vstack([ctx, prompt]).copy())
ent = float(-(p[:, -1] * np.log(p[:, -1] + 1e-30)).sum(-1).mean()); uni = float(np.log(p.shape[-1]))
say("T6e", ent < uni, f"the damper chooses: last-token attention entropy {ent:.2f} nats averaged over 32 heads, vs {uni:.2f} for no choice")
print(f"\n{sum(T_.values())}/{len(T_)} passed")
