# Atlas Engine — LOG

Legend as in the spec: **[M]** measured on this rig · **[F]** forced by a file or theorem · **[B]** bet.

## 1. 2026-09-13, night — step 1 on the 4B already on disk: intake, census, seal

Spec §9 step 1 names Qwen3-1.7B from a fresh download. The 1.7B is not on this machine
(Hub manifest: two safetensors, 4.06 GB); the Qwen3-4B safetensors are, under
`D:\folded-weights\monster-group-fourier\data\qwen3-4b`. The intake is model-agnostic, so it was
built and run on the 4B tonight; the 1.7B run is the same command once the files are here.

**Built** (`intake/`): `cells.py` (the bf16 pattern as ⟨m|n⟩ on its sheet, decompose/recompose by
shifts and masks, the census), `intake.py` (headers → u16 memmap → census → closed alphabet →
Morton 128×128 grids of dictionary ids → static unfold from disk → seal), `gate_source.py`
(gate 1: byte needles over the intake's own source; "defloat" is exempted as the verb).

**Inherited, verified tonight** [F]:
- V5 + V5.1 crates build clean (release, 11 s); lib tests 176 + 20 green.
- Centipede `carillon-defloat selftest`: 65,280 finite bf16 exact; mutations refused.
- The recorded 4B oracle is in `v5/docs/HANDOFF.md`: argmax 27, top-5 [27, 32664, 86119, 64, 82], seq-8 all positions.

**Measured, Qwen3-4B, whole model** [M] — receipt `receipts/intake-qwen3-4b-20260913/RECEIPT.txt`,
seal `d7c3ebcc…88ddf7`, 236.7 s, grids 8.04 GB under `cells/qwen3-4b/`:

| gate | result |
|---|---|
| 1 source needles | 3 files, none |
| 2 census | 65,280 / 65,280 finite patterns round-trip exactly; non-finite in model: 0 |
| 3 static unfold | 398 / 398 grids byte-identical to source after disk read-back |

| group | cells | distinct patterns | odd atoms | binades |
|---|---|---|---|---|
| embed | 388,956,160 | 6,049 | 128 | 32 [−33..−2] |
| projections (36 layers) | 3,633,315,840 | 7,486 | 128 | 38 [−38..0] |
| norms (145 vectors) | 196,096 | 1,822 | 128 | 26 [−20..5] |
| embed + projections | 4,022,272,000 | **7,532** | 128 | **38** [−38..0] |
| all | 4,022,468,096 | 8,102 | 128 | 43 [−38..5] |

- The spec's and the paper's 7,532 / 38 binades is the **fabric without the norms**; V5 keeps norms in
  book 37, outside the dictionary. Whole-model is 8,102 / 43. Both are right; the spec should say which.
- Prop. 1's "exactly 38 binades × 128 atoms × sign" is not a product (that would be 9,728): it is
  128 atoms present in every binade, with the tails partial. Only 25 binades carry all 256 patterns;
  the low tail thins from 255 at binade −27 to 1 at −38, the high tail from 203 at −1 to 1 at 5.
- Hypothesis 1 (closed alphabet) now holds **model-wide**, not only on layer 17: every tensor group,
  including the norms, uses all 128 odd atoms.
- **No exact zero** of either sign appears anywhere in the 4.02 B cells.

**Open**: the Qwen3-0.6B " Paris" pass the spec mentions is not on this disk under that name (the paper
places it in a sibling web session). Gates 4–8 wait on step 2.

## 2. 2026-09-13, later — step 1 on the spec's target, Qwen3-1.7B

Downloaded from the Hub (two safetensors, 4.06 GB) into `models/qwen3-1.7b/`; config: hidden 2048,
intermediate 6144, 28 layers, 16 q / 8 kv heads, head_dim 128, vocab 151,936, tied, rope_theta 1e6,
eps 1e-6 — every dim a multiple of 128, head shape identical to the 4B, so V5's cos/sin, exp and
rsqrt tables carry over unchanged. [F]

**Measured, Qwen3-1.7B, whole model** [M] — `receipts/intake-qwen3-1.7b-20260913/RECEIPT.txt`,
seal `775b6fb7…16ada2`, 121 s, grids 4.06 GB under `cells/qwen3-1.7b/`:

| gate | result |
|---|---|
| 2 census | 65,280 / 65,280 exact; non-finite in model: 0 |
| 3 static unfold | 311 / 311 grids byte-identical after disk read-back |

| | cells | distinct patterns | odd atoms | binades |
|---|---|---|---|---|
| all (311 tensors) | 2,031,739,904 | 7,910 | 128 | 44 [−39..7] |

Again all 128 odd atoms, no exact zero of either sign anywhere. Hypothesis 1 holds on the second model.

**Anchor recorded once** (`receipts/anchor-qwen3-1.7b.json`, transformers 5.9.0, torch 2.5.1, bf16, CPU,
25.9 s, seal `4c82b1a8…808f3`): seq-1 token [1]; seq-8 tokens [1..8] → per-position argmax
[25, 3555, 4, 5, 6, 25010, 89401, 3764], top-5 at the last position [3764, 1812, 25010, 353, 1057];
"The capital of France is". The reference is never run again.

**Engine** (`atlas/`, Rust, depends on the v5 crate for RMSNorm/RoPE/softmax/SiLU only): the three
surfaces stay separate — `Atoms` (per id: sheet, m, n), `Place` (Morton rank, computed), `Grid` (ids
tile-major, never un-tiled); a value is composed only where an op needs it. matmul is the bucket:
exact odd-core products, rung adds, one i128 per output cell at 2^-64, one collapse at the gather.
Lane declared [2^-64, 2^52], refusal above, sticky below (V5's proven window). 10 tests green:
census, gate 1 needles, Morton order, single products and sums bit-identical to V5's `bf16_mul` /
`bf16_add` inside the lane, refusal above the lane.

## 3. 2026-09-13, 03:30 — step 2: the forward runs from the cells alone; the float anchor is retired

**Gate 4 as first written** (top-5 ids equal the transformers oracle), Qwen3-1.7B, three anchors [M]:

| anchor | argmax | per-position argmax | top-5 |
|---|---|---|---|
| seq-1 [1] | 25 = oracle | 1/1 | equal |
| seq-8 [1..8] | 3764 = oracle | 8/8 | 5th differs: mine 1027 (0x4187) over 1057 (0x4186); oracle the reverse |
| "The capital of France is" | 12095 = oracle | 5/5 | 4th/5th differ: mine 7407 (0x419f) over 32671 (0x419e); oracle the reverse |

Every argmax and all 14 per-position argmaxes equal the float reference. The two top-5 misses are
neighbours ONE bf16 ULP apart whose order the float pipeline inverts — the reference's own rounding
chain (fp32 RMSNorm then cast, cos/sin cast to bf16, three roundings per RoPE pair), not ours.

**Decision** ⟨ink, pgarcia 2026-09-13⟩: "remove the anchor, we're more precise than the model's
floats." The transformers recording is retired to `receipts/retired/`; it is never the gate again.
The reference is the engine's own first sealed receipt, `receipts/reference-qwen3-1.7b.txt`.

| gate | result |
|---|---|
| 5 replay | 3 logit rows (seq-1, seq-8, France) reproduce byte-identical, run after run |
| 8 cells alone | safetensors moved out of reach; the same three hashes |
| 1 needles | crate gate green (`gate::source_float_free`) |

Timing, 1.7B, this box, no tuning: seq-1 1.1 s, seq-8 7.9 s, 5-token prompt 4.8 s; surfaces load 2.5 s.

Open, in order: the deferred-divide softmax (B4, bit-identical to this receipt or it does not land);
the clock rewrite of the accumulate (step 3, same receipt); the frame surface as a written file.

**Qwen3-4B, same engine, same session** [M]: seq-1 argmax 27, top-5 [27, 32664, 86119, 64, 82]; seq-8
per-position [27, 220, 4, 5, 6, 25010, 873, 3764] — equal, id for id, to V5's recorded anchor
(`v5/docs/HANDOFF.md`), which V5 had verified against transformers in May. The surfaces engine and the
dense engine agree on every id; the weights were never a matrix. seq-1 2.4 s, seq-8 18.3 s, load 9.3 s.
Reference receipt: `receipts/reference-qwen3-4b.txt`.

## 4. The footprint of the tiniest running kernel [M] (`atlas_kernel_bench`, 1.7B, layer-0 q_proj, real activation, one thread)

| object | bytes |
|---|---|
| Bucket, the hot state of one output cell | 32 |
| activation element decomposed (sheet, m, n) | 9 |
| alphabet entry (sheet, m, n, pattern) | 6 × 7,910 ids = 47,460, L1/L2-resident |
| weight cell (id) | 2 |
| tile (128×128 ids) | 32,768 |
| tile line, the inner loop's operand | 256 of ids + 1,152 of activation |
| Place, per tile | 4 |

| grain | terms | time | ns / term |
|---|---|---|---|
| one tile line, L1-hot | 128 | 446 ns | **3.49** (the ALU cost: 4 lookups, 1 multiply, 1 shift, 1 i128 add) |
| one output cell + collapse | 2,048 | 12.1 µs | 5.89 (ids streamed from L2/L3) |
| one tile row, 128 cells | 262,144 | 1.67 ms | 6.35 |

Multiply-up: seq-1 = 1,720,451,072 matmul terms → 10.1 s single-thread at 5.89 ns; 0.63 s ideal on
16 threads; measured full pass 1.05 s (everything included). seq-8 = 13.8 G terms → 81 s / 5.1 s ideal /
7.6 s measured. Ids streamed per seq-1 pass: 3,440 MB. The gap between 3.49 (L1-hot) and 6.35 (streamed)
is the memory side; the gap between 0.63 ideal and 1.05 measured is 16 hyperthreads on 8 cores plus the
non-matmul ops. No SIMD anywhere yet: the term is scalar i128.

## 5. The 256-lane kernel, one lane per (odd atom, sheet) ⟨pgarcia 2026-09-13 ~04:00⟩ [M]

`atlas_fiber_bench`, layer-0 q_proj (2048×2048), three real activations. Each output row folded by
(sheet, atom) into 256 lanes of (column, rung); a term is the activation core shifted by the weight
rung and added; each lane sum is multiplied by its atom once at the gather; sign folds as a subtraction.

| | result |
|---|---|
| fold | 4,194,304 entries, 12.6 MB at 3 B each, 255.8 of 256 lanes non-empty per row, 0.09 s |
| bit-identity vs the bucket | **0 mismatches** over 2,048 cells × 3 activations |
| scalar, one thread | bucket 5.68 ns/term; fiber **8.89** ns/term |

The multiply leaves the tick and the bits do not move. Scalar, it is slower: the lane walks the
activation by column index (a gather, 9 B per hit, cache-hostile) instead of streaming 128 contiguous
ids, and carries 3 B per entry instead of 2. A scalar x86 multiply costs one cycle, so removing it buys
nothing on a scalar core; the form pays only where the lane IS the vector — 16 shift-adds per tick on
AVX2, the whole tile per tick on the GPU — and where the gather is free (a tile resident per lane).
The kernel is correct and ready; its speed is a property of the lane width, not of this loop.

## 6. The radix chain, crank on the highest odd ⟨pgarcia 2026-09-13 ~04:20, after mm-solver's crank⟩ [M]

Σ_j (2j+1)·a_j = S_0 + 2·Σ_{j≥1} S_j with S_j the suffix sums from 255 down: the gather over the
128 odds is 255 adds and one shift, no multiply, every odd's multiple produced on the way. The same
chain over rungs (×2 = one shift, one add per rung; negative rungs by halving from the deepest up,
dropped bits to sticky) removes the per-term shift, and a term becomes one add into its own id's bucket.

`atlas_radix_bench`, layer-0 q_proj (rungs [−35..−2], span 34 → 8,704 buckets per output cell against
2,048 terms), three real activations, one thread:

| kernel | bit-identity vs bucket | ns / term |
|---|---|---|
| production bucket (multiply + shift + add, ids streamed) | — | 5.71 |
| A: fiber lanes + odd chain (shift + add per term; gather multiply-free) | **0 mismatches** | 6.57 |
| B: full radix (ONE ADD per term; rung chain + odd chain; activation prefixed once, 5.8 µs/token) | **0 mismatches** | 11.45 |

Both chains reproduce the bucket's bits on all 6,144 cells. Scalar, A is within 15% of the streaming
bucket and beats the multiply gather of §5 (8.89 → 6.57); B pays for computing 4.25× more buckets than
terms, plus 139 KB of scratch per cell. The term in B is the smallest it can be — one add — so B's
cost is entirely the bucket count and the gather; that is the shape step 2 has to change.

**C: the base as the denominator, ×4 by planes** ⟨pgarcia 2026-09-13 ~04:45⟩ [M]. The tensor's lowest
rung K is its one denominator 2^K; activations are prefixed at scale 64 + K (K ≤ 0), so the rung chain
is doubling only, from the top rung down, and lands at scale 64 exactly — no halving, no per-rung
sticky; the prefix floor n_x ≥ −64 − K is the production floor (n_x + n_w ≥ −64 with n_w ≥ K) by
another door. Four activation planes (four tokens) ride one walk of the ids: four buckets per id, four
chains, four collapses.

| kernel | bit-identity vs bucket | ns per term per token |
|---|---|---|
| production bucket, one token per walk | — | 5.68 |
| B: full radix, one token | 0 | 11.34 |
| **C: full radix, four tokens per walk** | **0 mismatches, 2,048 cells × 4 tokens** | **6.14** |

The "more than we should" is per weight row, not per token: shared across the four planes it costs
half of B, and the ids — the only stream that touches memory — are read once for four tokens. The
term is one add; the chain is one shift and one add; nothing else is left in the kernel.

## 7. The 508-token pass (127 × 4), the measurement that retired the sizing ⟨pgarcia 2026-09-13 ~04:50⟩ [M]

Ids 1..508 through the 1.7B as one pass, production bucket kernel, 16 threads. 554.1 s: 28 layers at a
flat 16.4 s each (463 s), then 91 s for the tied head. Cores sampled mid-run: 15.05 of 16 busy.

- **1.09 s per token against 1.05 s for seq-1: batching bought nothing**, because the kernel walks the
  ids once per token. The stream was fed 508 times. That is the queuer, not the bucket.
- The head is the first place the re-walk leaves L3: 311 MB of embed ids × 508 = 158 GB from DRAM;
  compute alone predicts 56 s, measured 91 — the last 35 s are the feed.
- Layers flat at 16.4 s: attention's O(n²) serial walk is not visible yet at 508.
- The first eight positions reproduce the seq-8 reference exactly; the causal prefix is invariant.
- No lane refusal in 508 positions: activations stay inside [2^-64, 2^52] on a real long input.
- 177 distinct predictions; 43 positions predicted the next consecutive id (the model found the
  counting in the noise); the tail collapses to newline/space (198/220).

The machine (32 GB, 8 cores, 96 MB L3) was never the limit. The schedule is: ids once per turn with
four planes rotating on them, attention as a reduction at the gather. The 1.09 vs 1.05 is the receipt.

## 8. The dictionary as the 2-adic hyper object [M] (after the five papers, 2026-09-13 ~05:20)

Read against `cells.pdf`: the sheets ⟨m|n⟩∓ are the ∓1 shifts (Mersenne/Fermat), not the sign; every bf16
value is a 0-sheet name ⟨m|n⟩⁰ with a sign beside it. The core m is a vertex of the 7-cube of mantissa
gates; the rung n is the band; Thm 8.1 (Babbage's carriage, B_k = 2^k + past) is the rung chain of §6
read as a theorem; the center 1 (the empty product, the bar's fixed point) is the void the kernel sits in.

**Measured on both dictionaries: the object is exactly a binade box.** Not one cell lies outside the box
[bmin..bmax] cut through the (odd, rung) rectangle; the interior of the box is completely full — every
odd at every rung — and only the two tails thin.

| model, sign | binade box | cells in box | present | full binades (all 128 odds) | tails |
|---|---|---|---|---|---|
| 1.7B + | [−35..7] | 5,504 | 4,384 | −25..4 (30 of 43) | low −35..−26, high 5..7 |
| 1.7B − | [−39..2] | 5,376 | 3,526 | −24..−2 (23 of 42) | low −39..−25, high −1..2 |
| 4B + | [−38..5] | 5,632 | 4,337 | −26..2 (29 of 44) | low −38..−27, high 3..5 |
| 4B − | [−36..0] | 4,736 | 3,765 | −27..−2 (26 of 37) | low −36..−28, high −1..0 |

The holes are contiguous on the cube (each hole has ~6 of 7 cube-neighbours that are holes): the tails are
shrinking corners of the 7-cube, the diagonal where the binade cut meets the rung. Inside the full band
the dictionary is the identity on patterns — (sign, rung, odd) IS the bf16 pattern, no table — and only
the tails need a lookup. Asymmetry: + reaches 3–5 binades higher than − on both models (norms, all
positive) and − reaches lower.

So the hyper object, from the right side of the torus: sign × a solid interval of full 7-cubes along the
rung axis × two sampled tails. 1.4 KB of occupancy bits describes the whole alphabet.

## 9. The radix kernel under the whole forward — gate 5 holds ⟨2026-09-13 ~05:45⟩ [M]

`atlas/src/fold.rs`: the fold by (sign, odd, rung) per output row is the third surface, built at load
(198 tensors, 7,815 MB, 14.7 s, 16 threads); every projection and the tied head run `matmul_fold`:
activations prefixed once per token at scale 64 + K (K the tensor's lowest rung, its one denominator),
one add per term with four planes per walk of the columns, the rung chain doubling-only from the top
rung down (Horner with gaps), the odd chain with the crank on 255, one collapse per plane. Switch:
`ATLAS_KERNEL=radix`; the bucket kernel remains the default and the twin.

| | bucket kernel | radix kernel | reference hash |
|---|---|---|---|
| seq-1 | 1.1 s | 1.1 s | 536cd9f4… equal |
| seq-8 | 7.9 s | **4.4 s** | 29353094… equal |
| France (5 tokens) | 4.8 s | **3.1 s** | c3e34060… equal |

**gate 5: PASS — 3 logit rows byte-identical to the reference receipt.** The kernel Pedro taught tonight
(fiber → crank on the highest odd → base as denominator → ×4 by planes) now carries the full model, and
the bits did not move. Speed-up is the ids walked once per four tokens; seq-1 has one plane and is
unchanged, as it must be. No multiply and no per-term shift remain in the weight path.

## 10. The 508 found the floor; the kernels are made exact ⟨2026-09-13 ~06:30⟩ [M]

The 508-token turn under the radix kernel: 373 s (bucket 554 s), argmax equal, **logits hash not equal**
(9642638c… vs ed4d9482…). Diagnosis: the bucket truncated per TERM at 2^-64; the radix truncated per
ACTIVATION at the prefix scale 2^-(64+K). Same integer when nothing is tiny; different by a sticky bit at
a rounding tie when something is. Counted: even the three anchors had 862 of 7,254,016 activations under
the prefix floor — gate 5 passed there by luck. Two floors are two definitions.

**Fix: one definition, exact.** Both kernels now compute the exact integer Σ m_x·m_w·2^(n_x+n_w+F) with
F = ACT_FLOOR − rmin per tensor (F = 2·ACT_FLOOR for activation×activation), in a 256-bit lane
(`matmul::Wide`), with NO truncation before the single collapse. The one declared floor is on the
activation: patterns under 2^-90 are dropped with sticky, identically in every kernel. An exact sum is
order-independent, so the bucket and the radix chain are bit-identical by construction, not by test.
The hot loop of the radix kernel stays i128 (bucket sums of prefixed activations, ≤ 2^123); only the
chains and the twin's per-term add are 256-bit.

| | exact bucket | exact radix | reference |
|---|---|---|---|
| seq-1 | 1.4 s | 1.7 s | equal |
| seq-8 | 10.6 s | 7.0 s | equal |
| France | 6.6 s | 4.7 s | equal |
| activations under the floor | 0 of 7,254,016 | 0 of 7,254,016 | — |

gate 5 PASS on both. The reference hashes did not move: the old floors never mattered on the anchors.
Cost of exactness: ~1.5× on the chains. The 508 under both exact kernels is running; the two hashes
must be equal, and that equality is the receipt this section waits for.

**§10 closed** ⟨~07:40⟩ [M]. The 508 under both exact kernels:

| kernel | 508 hash | s |
|---|---|---|
| bucket with the per-term floor (start of the night) | ed4d9482… | 554 |
| exact bucket, 256-bit lanes, 8 threads | **ed4d9482…** | 1,479 |
| exact radix, 256-bit chains, 16 threads (shared box) | **ed4d9482…** | 801 |

Identical by construction, identical by receipt. The activation floor at 2^-90 was touched by 2 of
263,217,152 activations and moved nothing. The reference receipts (3 rows) hold on both.
GOAL.md step 2 is met: one turn = one forward, judged by the engine's own receipts.

## 11. ball.rs — the function, step 1 ⟨~07:30⟩

`atlas/src/ball.rs`: BALL(a, k) = Rot(q)^k · R(a). A cell is the point (place, ±odd, rung). Rot(q) is
the integer sandwich q·v·q̄ with q from SPIN_Q, never divided, the scale |q|^{2k} riding as the frame;
R is tensor_mesh's half-angle torus map, homogeneous of degree 0 so the frame scale cancels exactly,
roots declared at `digits` by integer sqrt; big integers (num-bigint), no float. O is the inverse
renderer; `tight()` the gate. Tests (4) green: Mᵀ M = |q|⁴ I for all six seals; the periodicity census —
**the norm-2 tick closes in 4, the five odd-prime ticks have infinite order** (q^k real for no k ≤ 512):
the door refuses a radial turn; sandwich = matrix; tight on a synthetic cloud at k = 0, 1, 5.

Two corrections to the goal as first written: run-003's 91/64° yaw is the carrier's (ffmpeg v360, float)
exchange rate, not a tick of the function, so step 4 cannot hash against the GIF by construction — the
function's frames get their own receipt and the GIF keeps its seam; and the card is 8 GB, not 16.

**§11 gate, every cell of one tensor** ⟨~07:55⟩ [M]. `atlas_ball`, layer-0 k_proj, all 2,097,152 cells as
points (place, ±odd, rung), W = H = 512, digits = 12, seal q = (1,1,1,0), |q|² = 3:

| | frame scale | tight L0 | tight L3 | occupied cells | table sha256 | time |
|---|---|---|---|---|---|---|
| k = 0 | 1 | true | true | 3,665 / 262,144 | 2749521b… | 9.2 s |
| k = 4 | 81 | true | true | 1,062 / 262,144 | 7cf4b5ab… | 10.8 s |

All six seals: Mᵀ M = |q|⁴ I exactly; period 4 for norm 2, none ≤ 512 for 3, 5, 7, 11, 13. R and O are
inverses on every cell before and after the turn; the frame scale rides as |q|^{2k} and cancels in R.
Receipt: `receipts/ball-qwen3-1.7b-model_layers_0_self_attn_k_proj_weight-q3-k4.txt`. GOAL.md step 1 met.

## 12. The fold as a cube with rung bands ⟨2026-09-13 ~08:40⟩ [M]

`atlas/src/cub.rs`, `atlas_cub write|info`. ISIS3-style PVL header (V5 master.cub lineage), then per tensor
a tile index (u64 per line-tile of 128 rows), then the body: per tile, per BAND = rung from the top down,
per line: `u16 n; n × (u8 lane, u8 count); Σcount × u16 column`. The lane byte is (sign, odd); the band
index is the rung; nothing per bucket is stored. Mapped with V5's `mmap`, opened in 0–1 ms.

The kernel over it is the carriage read as a stream: for a tile, walk the bands top-down; every touched
(line, lane) accumulator doubles by the gap in bands since it was last touched and adds the band's column
sums (the ↑ tick, everywhere at once); after the last band the odd chain per (line, plane) is the gather.
256-bit lanes, the one activation floor; the same exact integer as `fold.rs` and `matmul.rs`.

| | |
|---|---|
| cube, Qwen3-1.7B | **5,058 MB**, 197 tensors, written in 24 s from folds built in 7.5 s |
| tied head | `lm_head.weight` in the safetensors is byte-identical to the embedding (sha 90fa0553…); sealed as one surface, not written twice (−947 MB) |
| load | mmap, 0 ms (was 15 s of fold-building, 7.8 GB resident) |
| gate 5, `ATLAS_KERNEL=cub` | **PASS** — seq-1 1.9 s, seq-8 12.5 s, France 6.2 s, three rows byte-identical |

The cube is under the 8 GB card with room for the planes: GOAL.md step 3 has its file. Bands per tensor
32–44; rungs [−46..−3] on the embedding, [−38..−1] on the projections.

**§12, the 508 under the cube** ⟨~09:00⟩ [M]: hash **ed4d9482…**, 733.5 s, 2 of 263,217,152 activations at the
floor. Four kernels on the 508 now — bucket with the old floor 554 s, exact bucket 1,479 s, exact radix 801 s,
cube 734 s — one hash. The cube is the fastest of the exact three at 508 positions.

## 13. The two clocks: acquisition and checker ⟨handshake-paper.pdf; pgarcia 2026-09-13 ~09:20⟩ [M]

"The clock you use for the first inference and another without anything; we cache the first and the
second is what runtime uses — a digital checker: if we need the same logit, pick it up."

`clock.rs`: the acquisition clock stamps every position with its teeth on the coprime rings 7, 11, 13
(V5.1's phase kernel, driven at 773, backward exact) and the CRT coordinate in 0..1001 — the projected
fourth axis, a coordinate not a gear; no single cadence can lock the addressing (tests: CRT round-trips
all 1001; the joint address is a permutation of the cycle; the clock reverses to the origin).
`cache.rs`: the ACQUISITION — per layer per position (x_in, k roped, v), the final h, the logits row —
sealed under the prefix of token ids; the checker `contained_prefix` admits a re-slice only when the
whole prefix matches; `pick(layer, address)` is the lookup. `turn.rs`: a turn computes only the positions
the checker did not admit, attending over the picked-up k, v, with the same ops in the same order as the
full pass, then extends the acquisition. The acquisition is annotated, never re-timed.

Judged (`atlas_turn`, cube kernel):

| turn | picked up | computed | hash vs reference |
|---|---|---|---|
| acquire [1] | 0 | 1 | seq1 EQUAL |
| re-slice [1..8] over the [1] acquisition | 1 | 7 | seq8 EQUAL |
| France (prefix differs → fresh acquisition) | 0 | 5 | france EQUAL |
| acquire [1..4] | 0 | 4 | — |
| re-slice [1..8] over the [1..4] acquisition | 4 | 4 | seq8 EQUAL |

**PASS — every re-slice equals the full re-parse, bit for bit.** The re-slice is free exactly when it is
information-contained in the acquisition (the sealed prefix), which is the paper's enabling condition
read on transcripts instead of k-space. Addresses of positions 0..7: 0, 773, 545, 317, 89, 862, 634, 406.

## 14. Prediction 6, first reading: the ringing is as white as the bell ⟨~10:10⟩ [M]

probs·V made an exact reduction at the gather (`matmul_dense`, one collapse per (position, d)) in place
of V5's per-step f3 chain: the three reference hashes did NOT change (the chain never lost a bit at
≤ 8 terms); v1 files kept under `receipts/v1-f3-attention/`, identical bytes.

`atlas_capture` writes the residual stream entering layer L, exact, sealed; `intake/ringing.py` is the
INSTRUMENT (float as ruler, as the paper's Prop. 4; the exact NTT power identity P_k = X_k·X_{N−k} over
Z/998244353 is computed beside it, in integers, as the vector's rotation-free receipt).

"The capital of France is", 5 positions × 2048, flatness = geometric/arithmetic mean of power (1 = white):

| layer | ordering | ringing | shuffled null | ratio |
|---|---|---|---|---|
| 14 | index | 0.6031 | 0.6033 | 1.000 |
| 14 | Fiedler unsigned (q_proj column correlation) | 0.6045 | 0.6119 | 0.988 |
| 14 | Fiedler signed | 0.5969 | 0.6045 | 0.988 |
| 17 | index | 0.6117 | 0.6116 | 1.000 |
| the bell: layer-14 q_proj rows, index | — | 0.5611 | 0.5607 | 1.001 |

Power by frequency decile at position 0: 0.101 0.098 0.101 0.101 0.099 0.099 0.101 0.101 0.098 0.102.
**Along the hidden axis the residual stream is white in all three orderings, to the third digit, like
the weights.** On a 5-token prompt Prediction 6 does not hold on that axis. The other axis — along the
positions, per dimension — needs the 508 capture (running) and is where a wave over the sequence would
live; that reading follows.

**§14 second reading — along the positions** ⟨~13:45⟩ [M]. `intake/ringing_positions.py` (instrument;
all 2048 dims, sequence of 508 positions per dim, flatness vs the same sequence shuffled). Two prompts,
each captured exactly at layers 0 (embeddings only), 1 and 14, sealed (`receipts/capture-…-508tok*.meta`):
the ramp (ids 1..508, the §10 query) and a white prompt (508 ids from seed 20260913,
`receipts/tokens-random508-seed20260913.txt`, `--tag random`).

| prompt | layer | flatness | null | ratio | power in lowest decile | lag-1 autocorr |
|---|---|---|---|---|---|---|
| ramp 1..508 | 0 | 0.532 | 0.561 | 0.947 | 0.196 | 0.13 |
| ramp 1..508 | 1 | 0.453 | 0.562 | 0.805 | 0.354 | 0.36 |
| ramp 1..508 | 14 | 0.378 | 0.564 | **0.670** | 0.452 | 0.54 |
| white (random ids) | 0 | 0.563 | 0.562 | 1.002 | 0.098 | — |
| white (random ids) | 1 | 0.561 | 0.563 | 0.996 | 0.108 | — |
| white (random ids) | 14 | 0.549 | 0.564 | **0.973** | 0.150 | 0.12 |

Hidden axis on the white prompt, 508 positions, index order: 0.5787 vs 0.5813 (ratio 0.996) — white, as
before. Position 0 is a sink at layer 14 (|x| = 15,407 vs median 150, both prompts); dropping it does not
move any ratio, so the colour is not the sink. Removing the per-position common mode moves the ramp's
ratio 0.670 → 0.713 only.

Where the colour lives (ramp, layer 14, sink dropped): the single largest bin is k = 1, the half-wave
over the whole sequence (12 % of the power in one bin of 254), then k = 3 and k = 2; the top position-mode
holds 13 % of the variance with flatness 0.03 along positions. On the white prompt the largest bin holds
1.4 % and the top mode is itself white (0.52).

**Reading.** Along the positions the residual stream at layer 14 is NOT white, in both prompts —
so Prediction 6 holds on the sequence axis and fails on the hidden axis. But the size of the wave is the
prompt's: a prompt that enters at 0.947 leaves layer 14 at 0.670 (the model amplifies a slow structure it
was given, layer on layer: 0.947 → 0.805 → 0.670), while a white prompt leaves at 0.973 with 15 % of its
power in the lowest decile — a faint slow component the model adds on its own, real to three digits
(508 × 2048 sequences averaged), and nothing like a bell. The wave is in the ringing only when the strike
had one. Instrument: float as ruler, exact bf16 patterns as input, seals in the .meta files.

**508 under the exact probs·V gather** (cube kernel, shared box with the captures): hash
63708bfa… (was ed4d9482… under V5's per-step f3 chain), argmax 220, top-5 [220, 16, 68, 17, 82]
unchanged, 1,210 s. The three reference rows did not move (≤ 8 terms never lost a bit); at 508 terms
the chain did, and the exact gather is the one the receipt now names. Row `tokens 63708bfa…` appended
to `receipts/reference-qwen3-1.7b.txt` as the 508's reference.

Gate 1 after today's instruments ⟨~14:00⟩: `intake/ringing.py` and `intake/ringing_positions.py` are
rulers, never in the pass; exempted by name in `gate_source.py` beside the retired oracle (6 intake files
pass). Three `as_secs_f64` timing prints in `atlas_forward` replaced by whole seconds from millis (the
same lesson as §4). The crate's own gate (`gate::source_float_free`, 11 files, cargo test 18/18) passes;
the Python gate run over the crate hits only the word in prose and in that gate's own identifier.

## 15. The maelstrom: the number generation as an object ⟨pgarcia 2026-09-13 ~15:30⟩ [M]

Pedro, explaining it to his wife: from the oscillator {1, i, −i, −1} come the quotient pairs 1/2, 1/4
and 2, 4; from them 4/3, 3/4 and 3/2, 2/3, whose product is "one that is a generational rule"; those
pairs give 3 and 5; every number must be found in the maelstrom before the integer side uses it; the
seams between twin primes are 4 = 2² and 12 = 2²·3; the step sequence is a Dedekind lattice and that
lattice is the fractal of the map. Lucas is the second half; the mapping lives in D:\shiat
(here-we-go-again ledger, centipede docs) and D:\folded-weights\monster-group-fourier (read as data).

What those folders state, verified here independently (`atlas/src/maelstrom.rs`, 5 gates, 23/23 tests):
φⁿ as the integer pair (a, b) = a + bφ has coordinates (F(n−1), F(n)) and trace 2a + b = L(n) — one
object, two rays. The coordinate ray mints 2, 3, 5, 13; the trace ray mints 2, 3, 4, 7, 11; 5 divides
no Lucas number (residues mod 5 cycle 2, 1, 3, 4, period 4 — the oscillator, at the point where 5
refuses the trace ray). The ladder 1..13 closes only with both rays: trace ray alone gives
1,2,3,4,6,7,8,9,11,12. Twin seams inside the ladder: (3,5)→4 = 2², (5,7)→6 = 2·3, (11,13)→12 = 2²·3,
each pair with one prime per ray (F4/L2 | F5; F5 | L4; L5 | F7); every twin seam after (3,5) is 6k,
so (3,5) is the one twin pair whose seam is the pure square — the first squaring. Beyond 13 twins are
not ray-born. Correction to the spoken form: 12 is 6 doubled, not 6 squared.

The maelstrom itself is the mediant tree (Stern–Brocot): each row minted by one integer add per
numerator and per denominator; neighbours always b·c − a·d = 1 (the 1 that is a rule); every positive
rational once; the largest number minted at depth k is F(k+2) (rows 0..16 checked: 1, 2, 3, 5, 8, …,
2584). The pair (F(n−1), F(n)) is the tree's zigzag node F(n−1)/F(n) — so the coordinate ray IS the
tree's Fibonacci path, and the loop_axle gears 377/987, 610/1597 with determinant 1 are its neighbours.

The bridge: Minkowski's ? sends the mediant tree onto the dyadic tree (continued-fraction run lengths →
runs of bits), exact on rationals. Its inverse gives every cell ⟨m|n⟩ (value m·2ⁿ, a dyadic) a unique
rational partner. `atlas_maelstrom` (0.2 s) over all 65,278 finite non-zero patterns: round trip
?(partner) = value for all; 32,639 magnitudes → 32,639 distinct partners; partner denominators ≤ 13
bits. ? preserves depth: the k-bit dyadic ⟨1|−k⟩ partners with 1/(k+1), k deep (⟨1|−133⟩ → 1/134).
The deep wall is the integers: an integer cell is its own partner (? is the identity on integers) and
sits N − 1 deep on the rightmost path toward the cusp, up to 255·2¹²⁰ − 1 for the largest cell —
depth is an integer, never a string; the first run of the bin hung writing that path. The zigzag F(n)/F(n+1) maps to
1/2, 3/4, 5/8, 11/16, 21/32, 43/64, 85/128, … (Jacobsthal over 2ᵏ) → 2/3 = 0.1010…, path LRLR….
Samples: ⟨3|−2⟩ → 2/3, ⟨5|−3⟩ → 3/5, ⟨255|−8⟩ → 8/9, ⟨129|−8⟩ → 8/15, ⟨1|−126⟩ → 1/127.
Receipt `receipts/maelstrom.txt`; partner table `receipts/maelstrom-partners.tsv`, sha256
7a8131b0b81a9e5a…. The odd side and the rung side are one tree read two ways; the dictionary now has
a Farey address per cell beside its (sign, odd, rung).

Not on disk anywhere before today (checked all three folders): Stern–Brocot, Farey, Minkowski ?,
Dedekind cuts, the maelstrom, the written oscillator. Two repairs owed upstream: centipede's spine
names rung 7 Fibonacci and rung 13 Lucas while the ray births run the other way (7 = L4, 13 = F7);
monster-group-fourier's proof has κ(5) = 4 but its frozen imaging LUT hardcodes 1. Crate gate now
12 files (`maelstrom.rs` added); `ball.rs` header fenced so it is no longer a doctest.

**The frame** ⟨~16:00⟩ [M]. Pedro: "that's a hyperobject… you cannot calculate it arithmetically, it's
geometry, it's literally a math function." So the tree address is not a path but a frame: the matrix
whose columns are the node's two Farey parents, node = their mediant, b·c − a·d = 1. A step right
replaces the left parent by the node, a step left the right one; a run of n equal steps is one
multiply-add, so `tree_frame` costs O(terms of the continued fraction) whatever the depth. The largest
integer cell ⟨255|120⟩ sits 255·2¹²⁰ − 1 deep on the right wall and its frame is
[[N − 1, 1], [1, 0]] in one evaluation; ⟨1|−133⟩ → 1/134 has frame [[0, 1], [1, 133]]. Gate: every
cell's frame has determinant 1 and mediant equal to the partner (6/6 tests). The tree is the dual of
the Farey tessellation of the hyperbolic plane; a node is a modular-group element, depth is
hyperbolic distance, the right wall runs to the cusp at ∞ — the void with the kernel in it — the
rungs are the horocycle steps and the odds wind around. Receipt re-sealed, partner sha unchanged
7a8131b0…; frames appended.

## 16. "Separate id and dictionary": the first measurement ⟨pgarcia 2026-09-13 ~16:40⟩ [M]

Pedro: "1/value = id, value is value; once you separate them you double the mmap and need two LUTs,
but you get triangulation and selection, no longer calculation at runtime." The smallest form of
"selection instead of calculation" is the odd product of a term taken from a table instead of an imul
(`atlas_select_bench`, L1-hot, one thread, 4,096 rows × 2,048 terms, integer timing, three runs):

| term | ns / term |
|---|---|
| (a) calculation: imul + shift + add | 1.66 – 1.84 |
| (b) selection: odd LUT 128×128 + shift + add | 2.12 – 2.28 |
| (c) selection: signed LUT 256×256 (sign beside) + shift + add, no branch | 1.72 – 1.78 |

All three accumulate the identical integer. The multiply is not the cost on this CPU: a 32 KB table
lookup is slower than imul, and folding the sign into the table only buys back the branch. The 3.49 ns
of the real tile line (§4) is the four dictionary lookups and the id stream, not the arithmetic
(1.7 ns here without them). Also measured (§15 tail): in the trained column order the planes are white —
1.00 cells per run for the full key, sign alone runs at exactly 2 — so the ids' 2 B/cell placement is
what any second chart must replace, not the product. Where "1/value = id" lands is therefore the
placement or the accumulation, not the term; the reading is open.

## 17. The fibers to the integers: id is a position, the dictionary is the value map ⟨pgarcia ~17:00⟩ [M]

"That's exactly your 2-adic hyper object's fibers to the integers." The odd residues mod 2ᴺ are the
group {±1} × Z/2^(N−2): every odd m is ±5ᵏ for one sign and one k. So an odd is a POSITION (sign, k) on
a torus and a cell ⟨m|n⟩ is a point (sign, k, n) of the bundle: the fiber over the integer is the rung
ladder, k the place along the torus. Products of values are sums of positions, m_x·m_w = ±5^(k_x+k_w);
the inverse is −k — "1/value = id". The two maps are separate objects: the log (odd → position, the
id side) and the antilog (position → odd, the dictionary side). `atlas/src/fiber.rs`, 4 gates, 28/28:

- the 128 atoms are exactly the 8-bit torus, k mod 64 with sign (1 → +0, 255 → −0, 5 → +1, 3 → −35);
- on the 16-bit torus (k mod 16,384) every product of two atoms is exact as a position sum,
  all 128 × 128 pairs (255·255 → 65,025);
- 1/m = 5^(−k): m·(1/m) ≡ 1 mod 2¹⁶ for every atom (1/3 → 43,691);
- every one of the 65,278 finite non-zero patterns is a distinct point (sign, k, n) of the bundle.

Receipt: the 128-atom position table in `receipts/maelstrom.txt`, sha256 b1ae5b6f3efa9935…. Crate gate
13 files. Nothing in the module computes a product: positions add, values are selected.

What it changes and what it does not (§16): per term on this core a selection is not cheaper than an
imul, so the fiber is not a faster multiply; it is the id as a position that composes. The row's
accumulation Σ ±5^(k_x+k_w)·2^(n_x+n_w) is a sum over points of the bundle; grouping by point is the
fold's (key, count) with the key now additive. Next: the kernel that walks positions instead of values —
prefix the activation as positions, add per term, select once per distinct point at the collapse — and
its receipt against the reference rows.

## 18. The kernel that walks positions ⟨~17:40⟩ [M]

`atlas/src/fiberkernel.rs`, `ATLAS_KERNEL=fiber`. The id side: every atom is a point (sign, neg, k, n) —
the value's sign beside the cell, the residue's sign and position k on the 16-bit torus (m = ±5ᵏ mod
2¹⁶), the rung. The activation row is read as points once per token. A term is: positions ADD
(k_x + k_w mod 2¹⁴), the odd product SELECTED from the antilog — with the residue sign choosing between
5ᵏ and 2¹⁶ − 5ᵏ, branchless — rungs add, one shift, one add into the same 256-bit lane. No multiply in
the walk. Same terms in the same order as the bucket, so bit-identical by construction; the unit gate
(256×256 grid, 512-atom dictionary, 3 activations) and gate 5 both hold: 29/29 tests, the three reference
rows byte-identical. The dictionary as positions: 7,910 points, antilog 16,384 entries, 144 KB.

| kernel | seq-1 | seq-8 | France |
|---|---|---|---|
| bucket (imul) | 1 s | 13 s | 8 s |
| fiber (positions add, value selected) | 2 s | 18 s | 11 s |

The receipt is the point: the first lesson of the separation cost was the residue sign — I folded it
into the value sign and the unit gate caught it (products with a negative residue are 2¹⁶ − 5ᵏ, not
5ᵏ). The time says what §16 said: on this core a selection is dearer than an imul, and the position
kernel carries two more bytes per id (neg, k) than the atom. What it is for is not the term: the id is
now a position that composes — the fold's key is additive — and the walk never touches a value.
That is the object Pedro asked for, judged; where it pays is the next measurement (the card, and the
additive key in the fold's collapse).

## 19. The observer, measured before it was built ⟨pgarcia ~18:10⟩ [M]

Pedro: "stage one from the other, a pipeline: the kernel cannot know what's being calculated; a
comparator defines what's what from the outside; something observes the bit shifting and then calls
the ids." The exact form of that comparator: after each stage, with T the exact partial sum and B a
bound on what has not been read, the bf16 rounding is determined when the top bit cannot move and T is
farther than B from every tie point (odd multiples of half an ulp); from then on the ids are never
called. Two stagings, measured in Python on layer 14 with the France residual stream as the activation
(exact integers, 320 and 240 (row, position) cells):

| staging | bound for the unread part | q_proj spared | down_proj spared |
|---|---|---|---|
| bands from the top (the carriage), ids of lower bands unread | count · max‖x‖ · 255 · 2ʳ | 0.0 % | — |
| columns by activation magnitude, weights unread | ‖x_j‖ · 255 · 2^rmax | 0.1 % | 0.1 % |
| columns by activation magnitude, weights read (spares adds only) | ‖x_j‖ · ‖w_j‖ | 1.3 % | 1.3 % |

Bands: determination came only 15–22 bands down a 20-band span, i.e. at the bottom. Columns: a bf16
dot product of 2,048 comparable terms is not decided by any small subset; its 8-bit rounding needs
essentially every term, and the bounds are loose by the activation's range (the sink). So on the exact
receipt an observer cannot spare id calls: at most 1.3 % of the adds when the weights are already
read. Not built; the measurement is the receipt. Where staging still pays is overlap, not omission:
a blind stage that only adds positions, a stage that selects, a stage that shifts and collapses — on a
device where those run as streams (the card) — and that is a latency question, measured there.

## 20. The footprint ⟨pgarcia ~18:30⟩ [M]

Peak memory of one gate-5 run (seq-1, seq-8, France) per kernel, sampled every 100 ms plus the OS peak,
32,694 MB box:

| kernel | peak working set | peak private | what it is |
|---|---|---|---|
| bucket | 3,888 MB | 3,889 MB | the ids (3,875 MB of .grid) + dictionary + activations |
| fiber | 3,888 MB | 3,890 MB | same ids + 144 KB of positions |
| cub, before | 8,839 MB | 4,028 MB | the ids loaded AND the 4,824 MB file mapped — the ids twice |
| **cub, now** | **5,556 MB** | **1,190 MB** | the map (file-backed, shared) + embedding grid 622 MB + norms |
| radix | 13,775 MB | 14,183 MB | ids + in-memory folds (~9.9 GB) |

Fix in `Model::load`: under `ATLAS_KERNEL=cub` only the embedding grid (rows of patterns for
`embed_row`) and the norms are read; every projection lives in the mapped file. Gate 5 PASS, 2/16/9 s.
On disk: cells 3,875 MB (= the safetensors, 3,875 MB — the ids are the weights, byte for byte),
fold.cub 4,824 MB, DICT 15.4 KB. The dictionary is 15 KB; the footprint is placement.

Per position, the acquisition (two clocks) keeps 28 × (2048 + 1024 + 1024) × 2 B = 229 KB of slices
plus 304 KB of logits = 533 KB; 508 positions = 271 MB; a 32k transcript would be 17 GB — the logits
cache should keep only what the checker needs (a row hash and the argmax) rather than the row.
Working memory of the cube kernel per thread: 128 lines × 256 lanes × 8 planes × 32 B = 8.4 MB.
The card (8 GB): the map at 4,824 MB fits with 3 GB to spare for the embedding rows and the lanes.

## 21. Sequencing and masking the ids: what recurs ⟨pgarcia ~18:50⟩ [M]

Pedro: "monotonic growth as long as we sequence the values and mask them — a, b, c appear in sequence,
that gets masked, and again and again." Whether the id stream has sequences to mask is a measurement:
the entropy of one id (the cap for any coding), the entropy of an id given the previous one (do
sequences carry anything a lone id does not), and the count of recurring triples against a shuffled
stream (what recurs by chance). Layer 14 and the embedding, ids in the walk order:

| tensor | ids | H(id) bits | cap B/cell | H(id | prev) | shuffled | recurring trigrams | shuffled |
|---|---|---|---|---|---|---|---|
| q_proj L14 | 4.2 M | 10.63 | 1.33 | 9.85 | 9.86 | 0.22 % | 0.21 % |
| down_proj L14 | 12.6 M | 10.55 | 1.32 | 10.23 | 10.23 | 0.74 % | 0.73 % |
| embed_tokens | 311 M | 10.52 | 1.31 | 10.42 | 10.49 | 18.36 % | 16.41 % |

In the projections a sequence carries nothing beyond its ids: the conditional entropy equals the
shuffled null to the second digit, triples recur exactly as often as chance, 4-grams recur 8 times in
4 million. The embedding's excess is whole duplicate rows: 1,992 of 151,936 rows (1.31 %) are copies,
the largest class 503 identical rows (unused token slots: 124, 125, 177, 178, …, 151160); with the
duplicates removed the recurrence is 17.28 % vs 16.88 % null. So the only sequence mask this model
offers is "these 1,992 embedding rows are those 1,992 rows", 4 MB of a 622 MB grid.

The ceiling is the single-id entropy: 10.5 bits, 1.32 B/cell against 2.00 — 34 % less stream, at a
decode per id, and only the stream is the cost (§4: the 508's head re-walk was 158 GB from DRAM). The
fold's (key, count) rows sit near that cap already (0.77–1.52 B/cell, §16) but need the ids beside
them to place the activation. Sequences that DO recur are the transcript's tokens, and there the mask
is exact only for an identical prefix — which is what the checker of the two clocks already does (§13);
a recurring n-gram at another position has different slices because attention sees the whole prefix.

## 22. The sheet of paper ⟨pgarcia ~19:10⟩ [M]

Pedro: "don't think of a table of weights as some random float thing, think of it as the piece of
paper: each value makes the plane go up or down, sequential, signed, one centimeter per unit; the sheet
is crimply already and the function explains how it got that way." Built as said: each row walked in
sequence, the height after each value is the running sum (the prefix, the ↑ tick), and the profile is
the crumpled sheet. Control: the same row's values shuffled, walked the same way. Layer 14, 64 rows:

| | q_proj | down_proj |
|---|---|---|
| profile flatness (1 = white) | 0.0046 | 0.0016 |
| same values shuffled, then walked | 0.0043 | 0.0016 |
| energy in the lowest 1 % / 10 % of modes | 0.925 / 0.992 | 0.973 / 0.997 |
| end height over the walk's own scale (free walk ≈ 0.80, a bridge to 0 ≈ 0) | 0.68 | 0.85 |
| increments: excess kurtosis; cells beyond 4σ | 0.44; 0.050 % | 0.80; 0.073 % |

The sheet is smooth: 99 % of its energy sits in the lowest tenth of its modes, hills and valleys, a
crumpled paper. And the shuffled paper crumples identically. So the crimp is the walk's — integrating
white increments gives 1/f² whatever their order — and the function that "explains how it got that
way" is the summation itself, which the kernel already is (the carriage). The values under it stay
white; the walk does not return to zero (rows are not bridges); creases, cells beyond 4σ, are one or
two per row, ten times Gaussian but not a structure. To 99 % of the energy the sheet is a few hundred
modes; to the last bit it is the 2,048 increments. The receipt is bits, so the sheet is a picture of
the carriage, not a shorter model of the weights.

**Regeneration test** ⟨~19:20⟩: weights regenerated from the sheet with only its lowest modes kept, then
differentiated back to increments (q_proj L14, 64 rows, a random activation as the probe):

| modes kept | weights' energy regenerated | dot-product error (median) |
|---|---|---|
| 1 % | 21 % | 107 % |
| 10 % | 33 % | 99 % |
| 50 % | 75 % | 61 % |
| 90 % | 97 % | 37 % |

The sheet's 99 % is the profile's energy, which integration piles into the low modes; the increments'
energy is flat across modes, so keeping a tenth of the sheet keeps a third of the weights and none of
the dot product. The paper cannot be regenerated from its shape; the dictionary (15 KB) can, the
placement cannot.

## 23. The three random accesses, and the tokenizer ⟨pgarcia ~19:40⟩ [M]

The tape is fixed (§22 tail); three reads in the forward are not on the tape. Pedro read them back:
attention pairs positions → "that's the comparator": the one data-dependent selection in the pass,
the function that decides what it keeps. The embedding row is the token decomposed into values →
"grab the word and match it PRIOR to starting; build a customish tokenizer": the vocabulary IS the
first column of the embedding page, row i is the word vocab[i], and the matching belongs to a stage
before the pass, not to the pass. The activation prefix he did not see yet — written out below.

**`intake/tokenizer.py`** — text → rows of the embedding page, integers only: NFC, the model's own
pre-tokenization regex, the 256-entry byte-level alphabet (a fixed bijection), BPE merges by integer
rank (lowest rank first, leftmost on ties), the 26 added tokens matched first, longest first; decode
is the inverse. The reference implementation is used only as the gate (ids are integers; equality is
a receipt): 14/14 strings equal — ASCII, tabs and CRLF, contractions, Portuguese, CJK, emoji and
symbols, the chat template, code, 300-char runs, digit runs, combining marks, empty — round-trip
exact, and "The capital of France is" → [785, 6722, 315, 9625, 374], the engine's reference prompt.
Receipt `receipts/tokenizer-qwen3-1.7b.txt`, tokenizer.json sha aeb13307…. `--chat "text"` gives the
one-turn Qwen3 template as ids. Engine bins still take ids; the stage feeds them.

**The activation prefix, spelled out.** In the fold and the cube the weights are grouped by
(sign, odd, rung) per row and the tape says, for each group, WHICH COLUMNS hold it. The activation x
of one token is turned into integers once — every x_j written over one denominator, the tensor's
lowest rung (the base), values below 2⁻⁹⁰ dropped with sticky — that is the "prefix": x pre-scaled
so the walk is adds only. Then for a group (sign, m, n) with columns c₁, c₂, c₃ the kernel adds
x[c₁] + x[c₂] + x[c₃] into that group's lane; the odd chain and the rung chain multiply the lanes
out at the end by adds and shifts. The gather x[c] is the placement read from the other side: the
tape carries the columns, the activation is read at them. It is random access only into a 2,048-entry
vector that lives in L1, so it costs nothing to the stream; it is, however, the reason the
placement cannot be dropped — the columns are the ids.

## 24. The harness: turns composed against the engine held open ⟨pgarcia ~20:10⟩ [M]

"Make a harness and load it to compose turns, not a 34905304-token mess to test." Two stages:
`atlas_serve` (Rust) holds the model and the acquisition and speaks ids over stdin/stdout — `GEN <max>
<ids…>` re-slices the transcript through the checker, then generates greedily, `TOK <id>` as each
token lands, `END picked_up= computed= generated= positions= ms=`; `RESET`; `QUIT`. `intake/chat.py`
(Python) tokenizes with the model's own vocabulary (§23), composes the Qwen3 template over the whole
transcript, streams the decode, and writes `receipts/chat-<model>-<t>.txt`. No KV cache: the transcript
is re-sent whole every turn and the checker says what is contained.

Two things the first run exposed. The checker was all-or-nothing on the common length (§13's gate
never needed more); it now picks up the longest common prefix, exact because every slice is causal,
sealed by that prefix — the two-clocks gate re-run: PASS, every re-slice bit-identical. And the harness
re-rendered the assistant's reply without the think block the template had fed, so the transcript
diverged from the acquisition at "assistant\n" (picked up 20 of 25); the reply is now re-rendered
exactly as generated. Three turns, bucket kernel, one load (6.7 s), greedy, 24 tokens max:

| turn | reply | picked up | computed | generated | s |
|---|---|---|---|---|---|
| "What is the capital of France? Answer in one word." | Paris | 0 | 24 | 2 | 42.0 |
| "And of Italy?" | Rome | 25 | 18 | 3 | 34.4 |
| "Which of the two is further north?" | Paris is further north than Rome. | 45 | 22 | 8 | 53.6 |

Every earlier position is picked up; a turn costs its new tokens. Per computed position ≈ 1.8 s under
the bucket kernel (the tape per position); a generated token is one computed position. The EOS is not
forwarded (the acquisition ends at the last content token; the re-rendered transcript adds
`<|im_end|>` as a new position). Receipt `receipts/chat-qwen3-1.7b-1789328666.txt` and the rerun.

## 25. The oscillator with guesses: exact, fewer passes, no faster on this core ⟨pgarcia ~20:50⟩ [M]

"We just need an oscillator" — the tape cycling, four phases per cycle, the spare phases filled from
the transcript's own sequences, the comparator keeping what agrees. Built on `atlas_serve` (`LOOK g`,
`chat.py --lookahead g`): each pass carries the new token plus up to g tokens that followed the most
recent earlier occurrence of the ending n-gram (n = 4..1) in the transcript; after the pass the
comparator keeps a guess while the argmax before it agrees and refuses the rest, whose slices leave
the acquisition. Output = greedy decoding bit for bit, by construction and by receipt: the two
transcripts (lookahead 0 and 3, cube kernel, two turns, 80 generated tokens) are IDENTICAL.

| turn | lookahead | passes | guessed | accepted | tokens / pass | s / token |
|---|---|---|---|---|---|---|
| "Repeat exactly three times: the quick brown fox…" (33 tokens) | 0 | 33 | — | — | 1.00 | 3.55 |
| same | 3 | 12 | 27 | 21 | 2.75 | 3.68 |
| "List the days of the week, then in reverse" (47 tokens) | 0 | 47 | — | — | 1.00 | 3.42 |
| same | 3 | 33 | 55 | 15 | 1.42 | 4.38 |

The comparator works: on the repeated sentence 21 of 27 guesses were kept and the tape ran 12 times
for 33 tokens. The clock did not move: a pass with four planes costs about four times a pass with one,
because on this core the cost per pass is the adds per (term, plane), not the tape (§4, §16, §18, §19
all said so). So the oscillator with guesses is exact and pass-efficient and exactly as slow, here.
It runs fast where the planes are the parallel dimension and the tape the serial one — the card —
and there the same receipt applies: identical transcript, tokens per pass. Per position in the turn
path the cube takes ~3.5 s (the bucket 1.8 s): attention over the acquisition and the dense probs·V
are on that path too.

## 26. v4.3 / v4.4 read back, and the residue lane measured ⟨pgarcia ~21:30⟩ [M]

Pedro: "take a look at the best exact one we made, without attention, v4.3 inside folded-weights."
Survey (D:\folded-weights\monster-group-fourier): v4.3-D was exact WITH attention — the whole Qwen3-4B
graph as big-rational dot products, one round-to-even per op boundary, weights read from SSD per
call, one thread: seq-8 in 3,835 s (~8 min/token), argmax 8/8 vs the float reference. It was slow
because rsqrt/exp/sin/cos/SiLU emit general rationals, so glyph closure broke at the first input norm
and every matmul fell back to big rationals. The exact one WITHOUT attention is v4.4's substrate
(2026-05-20): glyph lattice + RNS residues, 5.8 GB staged, lane kernel, 3 ns/cell on 8 cores with
AVX-512 (7800X3D), one 2.6M-cell projection in 8.5 ms — and it stopped at attention: "exp doesn't
close on the glyph lattice." Atlas closes that by construction: every activation is a bf16 cell at
every boundary and the transcendentals are V5's LUTs, which is why the exact full forward with
attention is 1.8 s/token here (16 threads) against 480 s there (1 thread).

The transferable lesson was the lane: residue rings fit SIMD, one CRT per output. Measured here
(`atlas_rns_bench`, the cube's band walk on one synthetic row, 2,048 terms in 1,462 entries over 25
bands, one thread, integer timing, two runs):

| lane | ns / term | exact |
|---|---|---|
| (a) Wide: i128 sum per entry, one 256-bit shl+add per entry | 4.4 – 5.6 | reference |
| (b) RNS, 4 rings < 2³¹, Shoup doubling per entry, lazy u64 sums | 15.3 – 15.5 | 256/256 lanes by CRT |
| (b) RNS, 9 rings | 27.0 – 27.2 | 256/256 lanes by CRT |

Exact both ways; 3.5× and 6× slower. The reason is the structure the fold gives the walk: an entry
holds 1.4 terms on average, so the per-entry cost (K ring multiplies) is charged almost per term,
while the Wide lane pays one i128 add per term and one shift per entry. v4.4's 4.5× per lane was
measured against a 33 ns/cell baseline with dense lanes and AVX-512; our scalar baseline is already
4–5 ns and the 5700X3D has AVX2 only. The residue lane is not the CPU lever; the rings stay for the
NTT gather, where the sums are long and the lanes dense. Rig: Zen 3, not Zen 4.

## 27. Step 3: the placement on the card ⟨pgarcia "okay, now the card, step 3" ~22:30⟩ [M]

`atlas/src/card.rs`, `ATLAS_KERNEL=card`. wgpu compute over Vulkan on the RTX 3060 Ti — no CUDA
toolkit on the box, none needed. WGSL, u32 only: the 256-bit lane is nine 32-bit limbs with explicit
carries, positive terms and negative terms in two arrays, so nothing is rounded on the card; the limbs
come back and the collapse is `matmul::Bucket::collapse`, the same code every kernel uses. One thread
per output cell walks the row's tile lines in Morton order, the same lines and the same term order as
`matmul_grid`, so the integer is identical, not merely equal. The dictionary is 7,910 packed u32; the
activation row is packed once per position; every grid is uploaded once (198 grids, 4,063 MB resident
on the 8 GB card, 1.7–5.4 s). Dispatch in chunks of 65,535 workgroups (the first 29-position pass
tripped the per-dispatch cap on the head's 4.4 M cells), readback bounded per chunk.

Gates: unit (card vs bucket on a synthetic 256×256 grid, bit-identical); gate 5 PASS, three rows
byte-identical; the 508 PASS, hash 63708bfa… = the CPU's exact-attention reference (§14).

| | CPU bucket, 16 threads | CPU cube | card |
|---|---|---|---|
| seq-1 / seq-8 / France | 1 / 13 / 8 s | 2 / 16 / 9 s | 1 / 1 / 1 s |
| the 508 | — | 734 s (1,210 with exact attention, shared box) | **72 s** |
| chat, per generated token | 1.8 s | 3.5 s | **1.2–1.3 s** |
| chat with lookahead 3, the repeated sentence | — | 3.68 s (no gain) | **0.57 s** (2.75 tokens/pass) |

The oscillator with guesses pays here, as §25 said it would: on the card a pass with four planes costs
1.55 s against 1.2 s for one, so the 12 passes that carried 33 tokens ran in 18.7 s against 39.8 s.
On the second turn (15 of 55 guesses kept) 1.40 vs 1.34 s/token — the refused planes are nearly free
but not free. What the 1.2 s per token is: not the card's arithmetic (the 508 shows 0.14 s per
position at scale) but ~200 dispatches per pass each with buffer creation, a readback, and the collapse
on the CPU, plus attention, norms, RoPE, softmax and SiLU still on the CPU between them. The next
lever is the staged pipeline on the card itself — activations resident, the collapse and the
per-element ops as further stages, one readback per layer or per pass — which is the pipeline Pedro
drew (§19 tail): stages as streams, none waiting on the other. GOAL.md step 3 is met: the fold
resident on the card, exact, same receipts.

## 28. The staged pipeline on the card, stages one and two ⟨pgarcia ~23:10⟩ [M]

Clocked first (`card::TRACE`, printed by `atlas_serve` per turn): a pass called the card 197 times, and
each call cost ~4 ms of overhead — three buffer creations, a submit, a full sync — against arithmetic
that the 508 had shown to be 0.14 s per position at scale. Two stages, exactness untouched (the collapse
stays on the CPU, the same `Bucket`):

1. persistent buffers written in place (`queue.write_buffer`): the activation, the params, one
   (out, read) pair per dispatch slot, grown on demand;
2. the independent projections of a layer under one submit and one wait — q, k, v together; gate and
   up together — `Card::matmul_many`, `wmat_many` in `turn.rs` and `forward.rs`, one activation upload
   and one readback for the batch.

Per chat pass (13-token prompt, 14 generated), before → after:

| stage | before | after |
|---|---|---|
| pack | 6 ms | 6 ms |
| upload + submit | 336 ms | 30 ms |
| gpu wait | 520 ms | 258 ms |
| collapse (CPU) | 24 ms | 35 ms |
| cpu rest (attention, norms, RoPE, softmax, SiLU, adds) | 159 ms | 76 ms |
| **per token** | **1.05 s** | **0.41 s** |

Second turn (64 positions): 1.24 → 0.47 s/token. Gate 5 PASS, unit gate PASS. What remains is the
sync: 5 waits per layer (qkv, o, gate/up, down, and the head) ≈ 140 per pass at ~1.8 ms each is the
258 ms, and the 76 ms of CPU between them is the elementwise V5 arithmetic on 28 layers. Both go away
together in the third stage — norms, RoPE, softmax, SiLU, the residual adds and attention on the card,
exact ports of V5's primitives, so a layer is one submit and a pass one readback — which is the port
that needs the most care: each primitive gated element-for-element against V5 before it joins.

**§28 timings on the staged card** ⟨~23:30⟩: the 508 unchanged at 72 s, hash 63708bfa… (its time is
the CPU's: exact attention over 508 positions and the elementwise ops, not the card). Chat, two turns:

| turn | lookahead | passes | s / token | cpu rest per pass | gpu wait per pass |
|---|---|---|---|---|---|
| repeated sentence, 33 tokens | 0 | 33 | 0.40 | 123 ms | 226 ms |
| same | 3 | 12 | **0.32** | 295 ms | 445 ms |
| days of the week, 47 tokens, 135 positions | 0 | 47 | 0.46 | 226 ms | 192 ms |
| same | 3 | 33 | 0.58 | 434 ms | 302 ms |

The card's share is now flat; what grows with the context and with the planes is `cpu_rest` — the
exact attention on the CPU (per head, per position, a dense Wide dot over every key) and the per-head
copies in `attention_turn`, then the norms and SiLU. On the second turn with four planes it is half the
pass. Stage three is therefore not optional: attention and the elementwise ops on the card, one submit
per layer. Transcripts with and without lookahead identical (the comparator unchanged).

## 29. Stage three: the whole pass on the card ⟨pgarcia "stage 3, go" 2026-09-14 ~00:30⟩ [M]

V5's primitives read and ported to WGSL verbatim in u32 — `bf16_mul`, `bf16_add`, `bf16_lt`, the
F³ point (`bf16_mul_ext`, `bf16_mul_q30_ext`, `f3_add`, `f3_to_bf16`, `single_bits_to_f3`,
`f3_div_to_bf16`), the four 64-bit spots as two-word emulations (the 8×31 product, the F³ sum's carry,
the 64/32 divide as 33 shift-subtract steps, the RoPE phase and its Q.46 interpolation term) — and on
them the engine's ops: cell decomposition, the nine-limb lane, the bucket's collapse with refusal
flags, the projection over the grid ids, the dense bucket for attention (scores and probs·V with GQA
addressing), RMSNorm, RoPE, the exp-LUT softmax with the scale and the causal mask, SiLU, the exact
add. `atlas/src/card_ops.rs`: one binding set for every kernel; gates element for element —

| gate | cases | result |
|---|---|---|
| bf16 mul / add / lt vs V5 | 200,000 each | 0 mismatches |
| F³ mul_ext / add / div vs V5 | 200,000 each | 0 |
| q30 fmsub / fmadd, interpolation term vs V5 / i64 | 200,000 each | 0 |
| exp-LUT decode + divide vs V5 | 100,000 | 0 |
| RMSNorm (40 rows × 2048), RoPE (16 rows × 4 heads, positions to 31k), softmax (12 × 3 × 40, masked, scaled), SiLU·u (100k), add vs matmul::add (100k) | | 0 |
| dense bucket vs matmul_dense; projection with the collapse on the card vs matmul_grid | | identical |

`atlas/src/card_layer.rs`, `ATLAS_KERNEL=card3`: activations resident, the acquisition's K and V
resident per layer (appended by copy at their positions, truncated by count for refused guesses),
a pass over the new positions as one command stream — per layer norm, q/k/v, the per-head norms,
RoPE, the K/V append, scores, softmax, probs·V, o_proj, add, norm, gate/up, SiLU, down, add; then the
final norm and the head — one submit, one wait, one readback of the logits, the refusal flags read
with them. Params in a 256-byte-slot arena with dynamic offsets; 4,063 MB resident, 2.8 s to load.

Gate 5 PASS (three rows byte-identical) and the two-clocks gate PASS (every re-slice bit-identical)
on the first run: the whole forward, attention included, now runs on the card with the CPU touching
only the tokens, the embedding rows, the checker and the argmax.

**§29 timings** ⟨~00:50⟩, whole pass on the card (`card3`), same box, one job:

| | CPU bucket | card, stage 2 | **card, stage 3** |
|---|---|---|---|
| the 508 (one pass, 508 positions) | 734 s (cube) | 72 s | **31 s**, hash 63708bfa… PASS |
| chat, per generated token, turn 1 / turn 2 | 1.8 s | 0.40 / 0.46 s | **0.39 / 0.37 s** |
| with lookahead 3, turn 1 (21 of 27 guesses kept) | — | 0.32 s | **0.21 s** |
| with lookahead 3, turn 2 (15 of 55 kept) | — | 0.58 s | **0.31 s** |

Transcripts identical with and without guesses. The context no longer costs: turn 2 at 135 positions is
as fast as turn 1, because attention is on the card. A pass with four planes now costs 0.58 s against
0.39 for one — the planes are nearly free, so the oscillator pays on both turns. Per position at scale
the card does 61 ms (the 508); a lone position still takes 0.37 s, which is the shape of the projection
kernel at small n: one thread per output cell walking 2,048 terms with two dependent loads per term,
32 workgroups for a 2048-row projection on a 38-SM card — latency, not arithmetic. The next lever is
inside one kernel: split a cell's walk across a workgroup (64 threads × 32 terms, the activation in
shared memory, limbs reduced once), which multiplies the parallelism of the small-n pass by 64 and
changes nothing about the integer — the reduction of limb sums is associative and exact.

## 30. The walk split across the workgroup, and what the clock actually said ⟨pgarcia 2026-09-14 ~02:00⟩ [M]

"Do it, split the walk across the workgroup." Done — one workgroup per output cell, 64 threads on every
64th column of the tile lines (coalesced ids), private lanes, a shared-memory tree reduction with
carries, sticky OR-reduced, thread 0 collapsing — and it changed nothing: the pass stayed at 1,192 ms
for 8 positions. So the cost was not occupancy. Two instruments answered where it was:

- the first head dispatch, 77 M workgroups in one launch, tripped the GPU watchdog and lost the device
  (zeros, a wgpu panic): projections are now dispatched in bounded chunks (≤ 2²⁰ cells);
- a timestamp-query instrument hung the device and was dropped; a skip-one-kernel instrument
  (`ATLAS_SKIP`) showed the whole pass in the projections (1,192 → 61 ms without them), with every
  "skip a producer" run cheap for the same reason — zeros walk for free.

The projection's per-term path was ~700 lane-cycles. The nine-limb lane was `array<u32, 9>` indexed by
the data-dependent limb, which the compiler keeps in local memory: several dependent memory round trips
per term. Three changes, each gated (6/6 op gates, gate 5, two clocks):

| projection lane | 8 positions | 5 positions |
|---|---|---|
| private array, one thread per cell (§29) | 1,192 ms | 772 ms |
| split across the workgroup, same lane | 1,192 ms | 772 ms |
| nine named registers, branchless select chain | 671 ms | 442 ms |
| one signed two's-complement lane (no sign branch), activations pre-packed once per buffer | **456 ms** | **313 ms** |

On the harness (`card3`, two turns, the same script), per generated token:

| | CPU bucket | card stage 2 | card stage 3 | **stage 3 + register lane** |
|---|---|---|---|---|
| turn 1 / turn 2 | 1.8 s | 0.40 / 0.46 | 0.39 / 0.37 | **0.19 / 0.17** |
| lookahead 3, turn 1 / turn 2 | — | 0.32 / 0.58 | 0.21 / 0.31 | **0.14 / 0.18** |
| the 508 | 734 s | 72 s | 31 s | **26 s** |

Hash 63708bfa… on the 508, three rows byte-identical, every re-slice bit-identical, transcripts
identical with and without guesses. Ten times the CPU per token, and a pass of a lone position now
costs ~170 ms, of which the projections are most: ~300 lane-cycles per term against a rough ideal
of ~80, the remainder being the serial carry ripple (nine dependent steps per term) and load latency.
The next lever, if wanted, is carry-save limbs (per-limb carry counters resolved once per cell), which
cuts the per-term chain to two limbs; exact by the same argument.

## 31. "Are we 100 % sure we're not reading the weights anywhere?" ⟨pgarcia ~02:40⟩ [M]

Two proofs, one static and one physical. Static: gate 8 was a sentence in a doc comment; it is now a
test — `gate::source_never_names_the_weights` scans the 17 runtime files and refuses the words for the
weights' files and their folder in code (37/37 tests). The only reader of the safetensors is
`intake/intake.py`. Physical: with `models/` moved out of the tree, `ATLAS_KERNEL=card3` gate 5 PASS,
three rows byte-identical; the engine never noticed. What the runtime reads: the cells (the ids —
grids or cube), the 15 KB dictionary, the norms, the config, and V5's three math tables (cos/sin,
exp, rsqrt: model-independent). The tokenizer's spec (`tokenizer.json`, in the Hub repo beside the
weights) is read by the text stage only.

One correction to the premise: the cube is not projected from the LUT. The dictionary holds the 7,910
values; WHICH value sits at each of the 2 billion places came from the weights, once, at intake, and
that placement is what the engine walks (§22: it does not regenerate). So the runtime reads no
weights, and the intake must run once on the receiver's own copy — the recipe-and-digest shipping of
the previous answer. Housekeeping: the timestamp instrument, which had hung the device, was removed;
it was also the one place an IEEE type name had crept into a runtime file (gate 1 caught it).

## 32. Pair i ↔ odd 5ⁱ: does anything line up? ⟨pgarcia 2026-09-14 ~04:00⟩ [M]

The head has 64 wavelengths (pair i of the 128 columns, phase step from `phase_step_qwen3.bin`:
6.283 positions per turn at i = 0 to 5,064,820 at i = 63, ratio 1.24094 = 10⁶^(2/128)); the odd torus
has 64 positions (m = ±5ᵏ mod 256). The attention side factors as 43,008 waves = 64 × 4 × 6 × 28 and
the fold's lanes as 256 = 4 × 64. Test: map pair i to odd 5ⁱ and ask whether the q_proj/k_proj rows
that feed pair i use that odd more than others. For layers 0, 14, 27: the 64 × 64 table of (pair i,
torus position k of the weights in pair i's rows), normalised by its marginals; the diagonal is the
hypothesis; the null is 200 permutations of the pair labels.

| layer | proj | diagonal ratio | off-diagonal | z vs permuted | max deviation anywhere | same for a shuffled table |
|---|---|---|---|---|---|---|
| 0 | q | 0.997 | 1.000 | −0.8 | 0.110 | 0.296 |
| 0 | k | 1.008 | 1.000 | +1.4 | 0.166 | 0.336 |
| 14 | q | 1.005 | 1.000 | +1.4 | 0.121 | 0.286 |
| 14 | k | 1.004 | 1.000 | +0.9 | 0.170 | 0.355 |
| 27 | q | 0.996 | 1.000 | −1.1 | 0.160 | 0.315 |
| 27 | k | 0.997 | 1.000 | −0.6 | 0.145 | 0.325 |

Nothing lines up. The rows feeding wavelength i use odd 5ⁱ exactly as often as any other odd, to
half a percent, inside the permutation noise; and no cell of the 64 × 64 table deviates as much as
a shuffled table's largest cell does. The two 64s are the same number, not the same object: one is
the byte's worth of angle on the torus, the other the head's worth of wavelengths, and the weights
know nothing of the pairing. The 4 × 64 shape is shared by the lanes and the spectrum; the contents
are independent.

## 33. The lift: m·5^|n|, and what lines up with the wavelength ⟨pgarcia ~04:30⟩ [M]

"Lift the mantissa by multiplying by 10 until you get a full int, no dot, and re-run the test on the
factors of the values." A value is m·2ⁿ; for n < 0, multiplying by 10 exactly |n| times gives the
integer L = m·5^|n| (fewer never suffice: m is odd). Its factors: the odd's primes, and 5 to the power
of the rung (plus any 5 in m). Test: the head's wavelength (pair i of the row's head) against each
factor feature of L, layers 0, 14, 27, q and k, chi² against 30 permutations of the pair labels:

| feature of L | z, all six tensors | largest cell deviation |
|---|---|---|
| **exponent of 5 (= rung, + v₅(m))** | **+105 to +1,014** | **63.0** (a class present in one pair only) |
| largest prime of m | −1.5 to +2.7 | 0.16 – 0.24 |
| number of prime factors of m | −1.0 to +1.5 | 0.11 – 0.19 |
| m mod 3, 7, 11, 13 | −2.3 to +1.0 | 0.02 – 0.07 |
| sign | −0.6 to +1.1 | 0.01 – 0.02 |

One factor lines up and it is the power of 5: the rung. The magnitude of the weights feeding a wavelength
depends on the wavelength; the odd — its primes, its residues, its sign — does not, at the level of §32.
The profile (mean binade of the weights per pair): spread 0.6 – 1.2 octaves across the 64 pairs in every
tensor; at layer 0 it trends with the pair index (correlation +0.59 for q, +0.31 for k); at layer 14 the
first pairs alternate (−7.19, −6.27, −7.06, −6.28 …), but the alternation is local, not a law (even
pairs below odd in 10 – 22 of 32 across layers); the eight shortest wavelengths sit a quarter-octave
lower than the rest in most layers and higher at layer 21. Each layer and projection has its own
profile, and every extreme rung class lives in one pair.

So the byte's two halves answer differently: the angle (odd) knows nothing of the head's spectrum; the
height (rung) knows it well. For the object: the rung plane is predictable from the pair, so it codes
smaller conditioned on the column, and per-pair lanes can be sized from per-pair rung ranges.
