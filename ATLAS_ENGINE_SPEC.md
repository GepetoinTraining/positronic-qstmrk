# Atlas Engine — Build Spec (clean)

**Scope.** Model download → inference. No intermediates, no float anywhere at runtime, no reference tax. Everything integer. Every stage either rewinds exactly or refuses.
**Date.** 2026-09-13. Node Zero. Sources: cad-systemup core, V5/V5.1 Qwen3-4B engine, centipede pin board, cells paper, ironing note, and this session's measurements.

Legend: **[M]** measured on this rig · **[F]** forced by an existing file or theorem · **[B]** bet — stated as a bet, gated before it is relied on.

---

## 0. Model

Pick by ops, not by size. The engine has proven bijective, float-free primitives for exactly five ops: matmul, RMSNorm, RoPE, softmax, SiLU (V5 `rope.rs`, `softmax.rs`, `rmsnorm.rs`, `bf16.rs`, `matmul*.rs`). A model is "first-buildable" iff it uses only those.

| Candidate | Ops beyond the five | Tied lm_head | Verdict |
|---|---|---|---|
| Qwen3-1.7B / 4B (dense) | none | yes | **first** — anchors exist (4B: argmax 27, top-5 [27, 32664, 86119, 64, 82]) |
| Qwen3-0.6B | none | yes | ran tonight (' Paris'); keep as smoke test |
| Qwen3.5-0.8B / 2B / 4B | Gated DeltaNet (delta rule = matmul+add; exp gate; L2 norm on Q/K; causal conv1d), partial mRoPE (64/256 dims), MTP head, vision tower (skip) | yes | **second** — needs 3 new bijections: exp gate, L2 norm, partial mRoPE |
| Gemma 4 E2B/E4B | per-layer embeddings, GELU, softcapping, audio | — | not now |
| Qwen3.8-Flash-Next | MoE 512 experts | — | no |

Requirements for any target: bf16 safetensors; every tensor dim a multiple of 128 (tile = GCD of dims); tied embeddings (the input sphere is the output sphere).

## 1. Intake — defloat to the glyph as written

1. Read safetensors header; every bf16 pattern as u16. No float type is ever instantiated. **[F: tryte.py, defloat.py]**
2. Each u16 → cell **⟨m | n⟩** on the two-sheeted lattice: core m = odd significand (u8, 128 atoms), rung n = signed 2-exponent, sign as sheet. Carry **both** names, ⟨m|n⟩⁻ and ⟨m|n⟩⁺, as written. **[F: cells paper, Thm 2.2]**
3. Census gate: every finite bf16 (65,280 patterns) round-trips through the cell exactly — the same gate as centipede `selftest`. Fails → abort. **[F]**
4. Dedup values into the value dictionary; sequence intact. Alphabet is closed at runtime (check/lookup only, no writes). **[M: Qwen3-4B → 7,532 = 38 binades × 128 atoms × sign]**
5. Grids: per tensor, cell ids packed in **Morton tiles, 128×128**, tiles ordered by Morton code of (tile_row, tile_col). Static unfold must be byte-identical to the source cells or abort. **[F: v4.4 writer + tiling self-check]**
6. Seal: emit the list of defloated objects, reasons, and kept residuals. `hull + residual == written` verified before signing. **[F: defloat.py]**

## 2. The three surfaces — cold, fibered, checked after

Three dictionaries that know nothing of each other. Runtime composes them per cell **after** the accumulator has landed; they are never on the tick. Arrayed to L3; never "loaded" — mmap'd, page cache is the working set. **[F: mmap.rs, tiled.rs]**

| Surface | Content | Owner | Size |
|---|---|---|---|
| **Atom** | 128 odd significands + sign | universal to bf16; never rebuilt | 1 byte |
| **Frame** | binade per **row** (row = output feature) + per-cell offset | model-specific | row frame ≈ nothing; offset ≈ 2.6 bits/cell **[M]** |
| **Place** | tile schedule, Morton order, row/col | pure geometry | `.meta` |

value = sign · atom · 2^(row_frame + offset). `dimension.row()` provenance is the pair (address, frame). **[F]**

Measured facts this rests on (Qwen3-4B, layer 17): rows carry the scale (row-mean binade std 0.41 vs column 0.10–0.14); cells under a row are white (spectrally flat in every ordering tried, adjacent-pair steps = shuffled); column correlations real (30× null) but rank, not sequence; exact-domain floor ≈ 10.6–11 bits/cell (three-dict split and v4.4 pair fold agree). **[M]** — the bytes are not the objective; cold is a side, not a size.

## 3. Hot path — the clock

The only hot state: the clock and the buckets in flight.

- **Clock** = the 4-cycle {1, i, −i, −1}. Multiplication by i is a swap and a negate: no multiply. The field is 998244353 (≡ 1 mod 4; i = 911660635 exact). **[F: lightfiber.rs, verified]** The tick is **↑** (rung + 1 = ×2 = one shift). **[F: cells Conv. 2.3]** Phase state: one word per prime, add + conditional subtract, exact reverse, no history (`phase_kernel.rs`, 137 primes, 274 bytes). The only coupling between kernels is the 137 gear. **[F: pins/mesh.rs]**
- **Quadrant tiling.** Positive cells on 1, negative on −1. A **double** — a pair re-hit in the grid — crosses to the i / −i side, and that side feeds attention. **[M: q_proj L17: 60.6% of pair slots are re-hits, 81% of the tensor is covered by pairs seen ≥ 2×; model-wide L0 table: 11.8M distinct pairs over 2.01B slots, first occurrences < 1%]** Fibering through top or bottom sheet dedups by itself: ↑ conserves the core, same core = same fiber. **[F: cells Prop 2.4]**
- **Buckets.** One per output cell in flight, **128/128 = numerator/denominator**, glyph^glyph: products are exponent-vector adds, quotients are subtracts, dedup/cancel = strip common factors (`tryte.align` / `tryte.strip_frame`; `glyph_morton` "multiply = exponent add"). Never read mid-pass. **[F; the other rig's glyph accumulator is the reference — port, don't reinvent]**
- **Everywhere at once.** Every cell computes its term on the tick; the only sequential thing is the reduction, done once at the gather as a tree (depth log N), never a walk. CPU caps at SIMD width (AVX2 = 16 lanes); GPU = the whole 128×128 tile per tick. **[F: field.py, ntt.py (FT not FFT), matmul_tiled.rs is the one remaining sweep]**
- **Closure.** At the gather the left side Q\*/Q\* closes: F1/F2 = 1, the bar move, one int. `hinges.closure` P1/P2 = 1. **[F]**

## 4. Ops

| Op | Mechanism | Status |
|---|---|---|
| matmul | bucket accumulate on the tick, single collapse | V5 bit-identical; rewrite as field (§3) |
| RMSNorm | the **frame** applied to activations: γ is the row frame of the activation side (book-37 norms) | V5 `rmsnorm.rs`, rsqrt LUT bijective |
| RoPE | the **joint**: rational rotation about the position axis (`joints.py`); angles carried as Q.30 LUT with proven bijection | V5 `rope.rs` |
| softmax | numerator and denominator kept as glyphs; **no divide until the gather**; exp via bijective LUT (glyph in, glyph out) | V5 `softmax.rs` + bucket |
| SiLU | bijective LUT | V5 |
| attention | **left as-is** — normal softmax attention, fed from the i-side | — |
| *(3.5 only)* delta rule | matmul + add on the bucket; state S (d_k × d_v per head) lives in the torus | new; no new op class |
| *(3.5 only)* exp gate | bijective LUT or refused | new bijection required |
| *(3.5 only)* L2 norm | ratio kept as glyph, root declared at `digits` like `tensor_mesh._mag` | new bijection required |
| *(3.5 only)* partial mRoPE | joint on 64 of 256 dims, three sections | extend `rope.rs` |
| *(3.5 only)* causal conv1d | integer FIR, exact | trivial |

**No transcendentals.** Anything not a shift, a flip, the bar, an integer add, or a proven-bijective LUT is refused, not approximated. No series expansions.

## 5. The torus — context outside the window

- Input sphere = output sphere: lm_head tied to embed. The 36 layers are the path; the surface is one piece. **[F]**
- Context is mounted **outside** the chat window; known matmuls are cached inside the torus turn by turn, in clusterings. Zoom levels are the Dedekind slots (2, 3, 6, 20, 168, 7,581, 7.8M), seeded by frequency, read-only at runtime, rebuilt offline. **[F: embed_lattice.rs; tensor_mesh.at(level)]**
- Query is removal: center on the hole. The next token's forward **starts from the coarse cell** the current pass is converging on; the fine accumulate collapses the region to a point. Calcs run ahead of token output. **[B — gate: coarse-cell prediction must contain the final argmax ≥ 99% on the anchor prompts]**
- *(3.5)* the DeltaNet state matrices are torus-native context: fixed size, O(1) update. **[F: architecture]**

## 6. Output — checked, not produced

The tokenizer/vocab table is **outside** the system. The pass closes to one int on the left side; the answer is that value **looked up** against the embed lattice (a dimension is a lookup, never a measurement). Embed rows collapse injectively to lattice points (149,833 → 149,833, 0 collisions) **[M]**; the logit side needs its own zero-collision count before the lookup is trusted **[B — gate]**. The surviving chord (carillon `locked_mask`) is the answer support; argmax only if the caller insists.

## 7. Gates (each can fail; failure aborts and is logged, never papered)

1. `assert_*_source_float_free` on every file that touched the pass (byte-needle grep, `f32`/`f64` never appear in code).
2. Census round-trip: all 65,280 finite bf16 through the cell, exact.
3. Static unfold of every grid byte-identical to source cells.
4. Anchor: argmax + top-5 IDs equal the **recorded** `transformers` oracle for the model (recorded once, never re-run as a reference tax). Qwen3-4B: 27, [27, 32664, 86119, 64, 82]; seq-8 all positions.
5. Replay: same input from a restored snapshot → identical receipt (`Forge137` replay_exact).
6. Rewind: every operator's inverse ∘ forward = identity (NTT, phase bank, tight R/O, hull+R).
7. Adversary: folded bytes without the cert are indistinguishable from noise (χ², byte entropy, compression ratio).
8. Delete the safetensors from disk; reproduce the anchor from the cells alone.

## 8. Bets, as bets

- **B1** the coarse-cell early start contains the argmax (§5).
- **B2** the i-side (pair re-hits) supplies enough attention slots — "more than we need" (measured on one layer; remeasure on the target model).
- **B3** the numer/denom bucket stays inside 128/128 for a full forward (V5's i64-window at D64 = 15 was already exact at the real activation regime; the pair form needs its own audit).
- **B4** softmax with deferred divide is bit-identical to `ExpLutSoftmax` on the anchor rows.
- **B5** *(3.5)* exp gate and L2 norm admit bijective LUTs at bf16 without a float in the builder path.

## 9. Order of work

1. **Intake + census + seal** on Qwen3-1.7B (fresh, from download). Gates 1–3.
2. **Forward, five ops, bucket accumulator** (numer/denom, deferred divide). Gates 4, 5, 8. Anchor recorded once from `transformers`, then never run again.
3. **Clock rewrite** of the accumulate: quadrant lanes, tick = ↑, reduction as a tree at the gather. Bit-identical to step 2 or it doesn't land.
4. **Torus**: Dedekind-slot embed lattice; coarse-cell early start behind a flag; gate B1.
5. **Qwen3.5-0.8B**: prove the three new bijections one at a time, each against the recorded oracle for one layer; mount DeltaNet state in the torus.
6. Seat the whole thing as a kernel on the centipede pin board, 137 gear, Ball4 closing the orbit.

Nothing in steps 1–4 imports v4.3/v4.4/monster-group-fourier code. V5/V5.1 primitives are the only inherited runtime code; CAD core is the only inherited method. The other rig's glyph accumulator is ported when it arrives; until then step 2 builds it from `tryte` + `glyph.rs`.
