"""Test sequence for the tensor atom. Run: python3 tests.py  (data/ holds three real Qwen3-4B tensors; fetch.py gets more)."""
import numpy as np, os, time, tempfile
TMP = tempfile.gettempdir()   # was a hard-coded /tmp; portable since 26 Sep (Windows run)
from math import gcd
from atom import *
rng = np.random.default_rng(1)
T_ = {}
def say(tag, ok, msg): T_[tag] = ok; print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {msg}")

tensors = {n: np.load(f"data/{n}.npy") for n in ("L18_k_proj", "L0_k_proj", "L18_v_proj")}
W = tensors["L18_k_proj"]; q, E = to_grid(W)

# T1 exactness: grid and cube round trip
say("T1a", np.array_equal(from_grid(q, E), W), f"bf16 -> exact integer grid (scale 2^-{E}) -> bf16, bit-identical")
write_cub(os.path.join(TMP, "a.cub"), W, {"Source": "Qwen3-4B layers.18.self_attn.k_proj"}); back, lab = read_cub(os.path.join(TMP, "a.cub"))
say("T1b", np.array_equal(back, W), "ISIS3 cube (PVL label + BandSequential) written and read back, bit-identical")

# T3 rules are discoverable (run before T2 so T2 can reuse the rule)
t = time.time(); rule = discover(q); dt = time.time() - t
N = rule["length"]; T = N // gcd(N, rule["repeats"])
say("T3a", rule is not None, f"blind search found a line: axis {rule['axis']} ({'rows' if rule['axis']==0 else 'cols'}), "
    f"{rule['repeats']} repeats over {N}, strength {rule['line']:.1f} vs best shuffle {rule['shuffle_max']:.1f}; "
    f"smallest period that tiles the axis T = {T}  ({dt:.1f}s)")
r0 = discover(to_grid(tensors["L0_k_proj"])[0]); T0 = r0["length"] // gcd(r0["length"], r0["repeats"])
say("T3b", T0 == T, f"same rule rediscovered independently in layer 0: T = {T0} (line {r0['line']:.1f})")
rv = discover(to_grid(tensors["L18_v_proj"])[0])
say("T3c", rv is None, "v_proj: no line beats its shuffles, so no rule is claimed; that atom is all residue (honest negative)")

# T2 space is irrelevant: four storage layouts, same positions -> same everything
H, R = split(q, rule["axis"], T); x = rng.integers(-7, 8, size=q.shape[1]); y = q @ x
layouts = [("BandSequential", None, None), ("Tile", (64, 64), None), ("Tile", (128, 256), None),
           ("BandSequential", None, rng.permutation(q.shape[0]))]
same = True
for fmt, tile, order in layouts:
    write_cub(os.path.join(TMP, "b.cub"), W, {"Period": T, "Axis": rule["axis"]}, layout=fmt, tile=tile or (128, 128), order=order)
    b, _ = read_cub(os.path.join(TMP, "b.cub")); qb, Eb = to_grid(b); Hb, Rb = split(qb, rule["axis"], T)
    same &= np.array_equal(b, W) and Eb == E and np.array_equal(Hb, H) and np.array_equal(Rb, R) and np.array_equal(qb @ x, y)
say("T2", same, "BandSequential, Tile 64x64, Tile 128x256, and shuffled row storage (order kept in the label): "
    "identical weights, harmonic, residue and inference output, bit for bit")

# T4 position is what matters
def hfrac(qq, axis, T):
    Hh, _ = split(qq, axis, T); return float((Hh.astype(float)**2).sum() / (qq.astype(float)**2).sum())
f_true = hfrac(q, rule["axis"], T)
sh = [hfrac(q[rng.permutation(q.shape[0])], rule["axis"], T) for _ in range(200)]
say("T4a", f_true > max(sh), f"energy carried by the signed-value harmonic: {f_true:.4f} in place vs shuffles "
    f"{np.mean(sh):.4f} (max {max(sh):.4f}, 200 trials). The line lives in the energy envelope, not in the signed values")
prof = (q.astype(float)**2).sum(axis=1).reshape(-1, 128)
pair = np.corrcoef(prof[:, :64].ravel(), prof[:, 64:].ravel())[0, 1]
ctrl = [np.corrcoef(prof[:, :64].ravel(), rng.permutation(prof[:, 64:].ravel()))[0, 1] for _ in range(200)]
say("T4b", pair > max(ctrl), f"the i-pair: inside each 128-wide head, dimension d and d+64 (the rotary pair) have energy correlation "
    f"{pair:.3f}; 200 random pairings give {np.mean(ctrl):.3f} (max {max(ctrl):.3f})")

# T5 stress and release: inference under stress, then the inverse residue returns the baseline
s = 3; base = q.copy(); cur = q.copy(); f = bf16_to_f64(W).astype(np.float32); f0 = f.copy(); Rf = (R.astype(np.float64) * 2.0**-E).astype(np.float32)
for _ in range(1000):
    cur = cur + s * R; _ = cur @ x; cur = cur - s * R          # exact grid
    f = f + s * Rf; _ = f @ x.astype(np.float32); f = f - s * Rf   # plain float32
say("T5a", np.array_equal(cur, base) and np.array_equal(from_grid(cur, E), W),
    "1000 stress/release cycles on the exact grid: weights return to baseline bit for bit")
drift = int((f != f0).sum())
say("T5b", drift > 0, f"control: the same 1000 cycles in float32 leave {drift:,} of {f.size:,} weights changed "
    f"(max drift {float(np.abs(f - f0).max()):.2e}); float cannot release")
print(f"\n{sum(T_.values())}/{len(T_)} passed")
