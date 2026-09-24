# Session record — Claude Code, 24 September 2026

Pedro Garcia (Node Zero) with Claude (Opus 5.5) in Claude Code. Repo `positronic-qstmrk`, folder `v3-based/`.

This is the companion to `day_v2.pdf` and the session bundle from the web surface. It records what this session built, ran, measured and corrected, in order.

**Stamps:**
- [run] measured today, by a named file;
- [thm] standard;
- [lit] published literature, not run by us;
- [PG] Pedro's posit;
- [C] Claude's reading;
- [post-hoc] noticed after the fact, not evidence;
- [open] untested.

v1 and v2 numbers keep their own [M]. Nothing is committed to git.

---

## 0. The headline

1. **Attention leaves its rest state for what is relevant (T11, pre-registered, every prediction held)** [run].
   - Qwen3-1.7B was given a prompt of 254 random tokens repeated twice.
   - Three heads drop their sink mass from ~0.54 to ~0.04 in the second copy: L3 h10, L12 h0, L14 h6.
   - They move it onto the exact position that follows the earlier occurrence of the current token (the induction target).
   - **96% of the released mass lands on that one position.** On the control prompt, those heads put ~0.001 there.
   - The instrument forward first reproduced v1's sealed captures: cosine 0.999965 at layer 1, 0.999870 at layer 14.
2. **The fluid organises with depth (T10)** [run].
   - Tokens cluster: mean cosine 0.05 → 0.55.
   - Effective rank collapses: participation ratio 176 → 3.3.
   - Position 0 (the sink) holds 95% of the stream's energy by layer 14.
3. **The non-reciprocal heads are exactly the previous-token heads (T10, exploratory)** [run]. The one-way "front" part of the QK coupling is the lag: each position reading one step back.
4. **The CubeKernel runs in Rust, composed and exact**, from binary to the 7//11 hinge, driven by the onset oscillator [run]. With all 65,536 BF16 patterns loaded into the 41 table, the resident footprint is **3,924 bytes** [run].

---

## 1. What was built (Rust crate `v3-based/walk/`)

| binary | what it does |
|---|---|
| `main` (census) | the §8 walk on the nonzero p³ cube for every Monster prime |
| `tierb` | T9 Tier B throughput (spec `TIER_B_for_code.md`) |
| `kernel` | the composed CubeKernel, its layer checks, a 10⁶-op random program, and the hinge-test driver |
| `bf16` | all 65,536 BF16 patterns through the kernel's intake gate, integers only |
| `footprint` | the 41 load with every BF16 pattern: tables against table-free, entry by entry |
| `t10`, `t11` | the attention registrations (§6, §7) |

Shared exact octonion code is in `src/lib.rs`: the Fano table of `t6_octonion.py`, the 30 S(3,4,8) labelings, unit sets in doubled coordinates. Diagnostic: `examples/psi_probe.rs`.

### 1.1 The census (day_v2 §8, reproduced exactly) [run]

| p | orbits on the nonzero p³ cube |
|---|---|
| 3, 5, 19, 29 | single orbit |
| 7, 41 | two equal halves |
| 17 | binary clocks only (2 × 8 and 306 × 16) |
| 11, 13, 23, 31, 47 | one root |
| 59 | degenerate |
| 71 | splits, with 70-cycles |

### 1.2 Tier B [run]

10⁸ ops per run. The ranges are the spread of medians across 3–5 full runs on this laptop.

| test | ns/op | memory |
|---|---|---|
| B1 walk by recurrence (p = 19, 29, 41) | 6.1–7.5 | — |
| B1 walk by `idx+1` table lookup | 1.6–1.9 | idx→state 827,040 B |
| B2 state → (half, idx), p = 41 | 2.3–3.2 | 275,684 B |
| B3 left multiply, u8 table (t6's `closed[0]`) | 4.1–5.4 | 57,600 B |
| B3 left multiply, f64 octonion | 32.6–39.5 | — |
| B4 half-hinge with refusal log / without | 10.9–11.1 / 3.8–4.0 | 125,000 B per 10⁶ ops |

**f64 drift after 10⁸ chained multiplies = 0 exactly.** Products of units in a closed set stay units, and their coordinates (0, ±½, ±1) are exact in f64. So B3 cannot show drift. A rotation outside the lattice, like T2's, would. [C]

### 1.3 The composed kernel [run]: 22/22 layer checks

- **Quadryte:** 64 states split 8/24/24/8; σ² = the magnitude flip; the flip swaps corner↔centre and edge↔face; 27 tryte positions = 26 + ∅.
- **Hurwitz seam:**
  - the 16 equal-magnitude states map one-to-one onto the 16 half-integer units, and the flip becomes q ↦ −q̄;
  - **the no-go is reproduced exhaustively over all 3,072 lane symmetries** (56 square to the flip, 128 cube to it, no 2T).
- **Octavian:**
  - 7 of 30 labelings close;
  - the twist p(i) = 2i mod 7 is a sign-exact order-3 symmetry that splits the closed seven 1+3+3;
  - the hub's table is a Latin square;
  - conj(u)(ux) = x everywhere, so left multiplications rewind exactly.
- **Found [run, exploratory]:** the twist-compatible Hurwitz copy {1, e₁, e₂, e₄} **is not in the hub**. Its 24 units lie in closed sets 1, 2 and 3, all three of one twist triple. The hub holds 1,512 Hurwitz frames, and the twist cycles i→j→k on none of them.
- **The bracket, full count:** 55,944 of 57,600 pairs (u, v) have u(vx) ≠ (uv)x for some x.
- **9-gon:** the walk (x+7, y+11) on Z₉² gives 9 loops of 9, with x+y conserved.
- **7//11:** the mod-13 facts hold. Unpaired, the hinge closes on 6 of 9 seat states. **Pair-gated it closes on 9 of 9**: doll.tex registration 1 held, though that is close to true by construction.
- **Composed run, 10⁶ random ops:**
  - 0 invariant violations (seat sum; 9-gon x+y; |x| = 1; x in its set; the ring = quarter-turns mod 4 × twists mod 3);
  - the rewind from receipts is bit-identical;
  - over 1,000 programs of 200 ops: 1000/1000 exact with receipts, **1000/1000 drift without them**.
- **Couplings chosen [C]** (listed at the top of `kernel.rs`):
  - the twist acts on every layer at once;
  - σ steps the ring by 9;
  - Lift carries an equal-magnitude quadryte state through the Hurwitz seam into the octonion unit;
  - Cover negates the unit.

### 1.4 The driver: `hinge_test.py`'s onset, one oscillator per lane [run]

- **Set-up:**
  - three Stuart–Landau oscillators, noise from OS entropy, floats in the simulated physics only;
  - the quarter-turn lookup x(t) + i·x(t − T/4) reads each lane's quadrant;
  - a lane holds ∅ until its hinge, then each quarter-turn tick is a σ.
- **Results (400 trials each, σ = 10⁻³ and 3·10⁻²):**
  - hinge lag median 9 samples (hinge_test: ~8–10);
  - the quadrant written at the hinge is the true one in 1188/1200 and 1190/1200;
  - **4.000 net quarter-turns per cycle**, 0 back-steps;
  - the lane matches the true quadrant in 99.92% / 99.66% of samples;
  - 0 invariant violations, and every trial rewinds exactly to all-∅.
- **Onset noise decides whether the octonion layer ever moves.** Lift needs the three onset phases aligned, and that happened in ~80% of trials.
- **Not driven by this wiring:** the 7//11 hinge, the 9-gon, Catch, Mul and Cover. What should fire the twist is [open] and [PG].

### 1.5 BF16 through the intake gate [run]

- **The patterns:** 65,024 normal, 254 subnormal, 2 zero, 2 inf, 254 NaN.
- **Admitted:** **65,280 with signed zero kept** (v1's census number, reached independently), or 65,279 by value.
- **Refused:**
  - the only duplicate value is −0;
  - all 254 NaN patterns → one ∅;
  - ±inf → overflow.
- **The 41 cube:** 41 is the smallest Monster prime whose cube holds all 65,536 patterns (3,384 states spare). Its halves of 34,460 hold one sign's 32,768 patterns each, with 1,692 spare per half.

### 1.6 Footprint of the kernel with all BF16 patterns loaded in the 41 table [run]

| | tables | table-free |
|---|---|---|
| 41 layer | 343,066 B | **3,096 B** (matrix power + baby-step/giant-step, 2 × 186 entries) |
| Octavian triple | 172,800 B | **810 B** (Fano table + 3 × 14 block masks) |
| kernel state (packed) | 18 B | 18 B |
| **total** | **515,884 B** | **3,924 B** |

Every entry of the table-free form was checked against the tables. The cost is time: ~457 ns per load and ~5.3 µs per read, against nanoseconds for a lookup.

---

## 2. The 41 table multiplies (a correction made during the session) [run]

- **The walk is multiplication in GF(41³).** Its step is multiplication by α, which has order 34,460 = (41³ − 1)/2. The two halves are the squares and the non-squares.
- **The halves multiply like signs.** A non-square times a non-square is a square: − · − = +. The half containing the field's 1 plays +.
- **The address is a measurement.** The index is a discrete log: α^i · α^j = α^(i+j) (checked on 200 random pairs).
- **Adding indices is multiplying values.** The index *is* the BF16 bit pattern (sign → half, 15 bits → index). So adding indices = adding bit patterns = **Mitchell's logarithmic multiplication (1962)** [thm]: exact on sign and exponent, missing only the mantissa cross term m₁·m₂, a 128×128 table.
- **This is v1 §17 again.** "1/value = id" and "products as sums of positions", reached through the walk.
- **The exact version is the pentagon's coordinates** (§4). The spare states per half give ≈ 13.2 binades of headroom [post-hoc].

---

## 3. v1 and v2 read against today (full synthesis in `SYNTHESIS-v1-v2-v3.md`)

### What runs through all three versions
- **Receipts decide:** refuse, never round.
- **The alphabet is the format's.** 128 odd atoms; the census is 65,280.
- **One free rotation:** {1, i, −1, −i}.
- **The sign sits beside the cell.** Today's no-go proves it has to.
- **Symmetries relabel; crossings create.** v2's axis cycle changes no name; today's twist only permutes closed sets.
- **Units only closes.** v1's ball used Hurwitz quaternions of norm 2..13, and the odd-prime ticks never close. v3 uses units only (24 and 240), so it closes and rewinds from finite receipts.

### Cross-checks [run]
- v1's 7,910 signed patterns and v2's 4,505 magnitudes agree sign by sign: 3,405 + 979 = 4,384 and 3,405 + 121 = 3,526.
- The model uses 12.1% of BF16's finite patterns.

### The two brackets of the oscillator relation
- **v2:** x² − 1 = 1/x + 1 → (x+1)(x² − x − 1), discriminant 5, roots −1 and φ.
- **v3:** x² − 1 = 1/(x+1) → x³ + x² − x − 2, discriminant −59.
- **They read as two views of the same primes, not an error** [run]. Among the Monster primes, **only 19 and 29 are both a single orbit under v3 and have φ as two distinct elements under v2**. At 19 and 29 the oscillator *resolves* into its lines (−1, φ, −1/φ): its harmonics. Over the same prime, the walk is one stream. So 19 and 29 carry both a pace and a harmonic basis [C]. What they are for is [PG].

### Footprint, read across versions
All three agree the alphabet is small (15 KB in v1, 3.9 KB today) and the **placement is the model** (2.03 B ids, ~3.3–4 GB). Pedro's position [PG]: once values are exact, remapping the placement has many options. The heavy part of float weights is runtime rounding.

### Cautions
- v2's 9-gon residues were forced by its own ×10⁴⁶ inflation.
- v2's cube byte and v3's tryte share a shape, not an encoding.
- The **[F] stamp** means "floatland reference only" in the README but "forced by a file or theorem" in both logs. Fix it.

### v1 can't build here, and why
`atlas` needs the V5 crate (path dependency, not on this machine) and V5's four LUT files. Traced on the France path:
- **Every weight matmul is V5-free:** the exact bucket, attention's products, the residual adds.
- **V5 is used for** RMSNorm, RoPE, softmax, SiLU and the attention-scale multiply, plus four tables: rsqrt, exp, cos/sin, phase step.
- **The code half of V5 is in the repo** as verbatim WGSL ports (`card_ops.rs`, 0 mismatches over 200,000 cases each).
- **The data half (the four tables) is not.**

Pedro's point [PG]: what was left at home was attention, on purpose. The weights are the easy part.

---

## 4. The pentagon (v2's 5D accumulator), read against v1 and today [C]

- **The 5D coordinates are v1's fiber point.** "Two two-sided coordinates (4 planes) + the 5th plane, the abscissa; 2pow is position along the abscissa, the mantissa is what the planes hold." That is v1's fiber-kernel point: value sign, residue sign, k where the odd is ±5ᵏ, and the rung n. The band walk is the accumulation. The decade 2 × 5 is its two generators: 2 moves along the abscissa and 5 turns the planes.
- **In those coordinates multiplication is exact addition**, because 5 generates the odd residues [thm, Gauss]: k₁ + k₂, n₁ + n₂, signs XOR. That's the exact version of the 41 table's Mitchell product.
- **The +, −, +, − rotation falls out of Z/5.** Mod 5, 2 = i (2² ≡ −1). Walking the four nonzero sides by ×2 (1, 2, 4, 3), whether each is a square mod 5 alternates +, −, +, −, and 0 sits between a + and a − [thm]. That matches Pedro's two corrections to the 5D drawing.
- **The final-face cube is the receipt log.** Its top is the relation and its bottom the archive, read as ⁻¹. "The free side must be 1" is the closure check.

---

## 5. Attention, read in the construction's objects [C, with the math stamped]

- **Why it looked like gibberish.** It is the only matmul whose operands are both made at runtime: activation × activation. Its "placement" is the sentence itself. What a head uses is W_q · R(i − j) · W_kᵀ, rebuilt for each pair of positions.
- **Softmax is the Boltzmann distribution** [thm]. The energy is −⟨q, k⟩, β = 1/√128, and the denominator is the partition function. Only differences matter; they become ratios, and the total closes to 1 (the 5D free side).
- **Base 2 splits it like the pentagon.** e^s = 2^t. The whole-number part of t is a **rung** (an exact shift along the abscissa); the fraction is a **plane** (a 128-entry table). The collector keeps two exact sums, N = Σ 2^t·v and D = Σ 2^t, and the output is the relation N/D.
- **RMSNorm is the other closure to 1:** length, where softmax is sum. It splits the same way: an exact sum of squares, and 1/√ as a rung shift plus a mantissa table.
- **RoPE is 64 oscillators.** Each pair of the 128 dimensions is z = r·e^(iφ). With its centre removed it is spine §8's cylinder, angle × rung.
- **The torus [thm].** QK-norm puts each head on a sphere. Two pairs make a Clifford torus in S³. The two degenerate circles (z₁ = 0, z₂ = 0) are the **Hopf-linked missing centres**: spine §8's "torus anatomy", line for line. A head is a 64-dimensional torus; RoPE winds along it and never closes (irrational frequency ratios), so every position gets a unique point. A score = Σ radii products × cos(phase difference + offset·θ).
- **Physics [lit]:**
  - transformers as convection–diffusion (Lu et al., 2019);
  - tokens as interacting particles on a sphere that cluster, i.e. attractors (Geshkovski, Letrouit, Polyanskiy & Rigollet, 2023);
  - pure attention collapses rank (Dong, Cordonnier & Loukas, 2021);
  - one attention step is a Hopfield energy step (Ramsauer et al., 2020);
  - non-reciprocal interactions make chasing and rotating phases (Fruchart et al., *Nature* 2021);
  - QK-norm exists because attention logits grow and collapse without it (Dehghani et al., ViT-22B, 2023).
- **Fronts [PG].** Where cold meets warm air, a torus forms, violent where they cancel hard. Split each head's coupling M = S + A [thm]. S is reciprocal: attraction, with an energy. A is non-reciprocal: circulation, the vortex part, and RoPE's generator is purely antisymmetric. QK-norm caps the radii, i.e. caps how violent a meeting can be.
- **The lag [PG].** Humans perceive ~80–150 ms behind the world, see then act; reflexes are the no-lag emergency path [lit]. Previous-token heads are the buffer: "me + one step back" gives direction. The sink is rest, a no-op [lit]: not every input is urgent. The hinge test measured the same thing: a small fixed lag (x(t − T/4)) beat processing everything at once by two orders of magnitude.
- **Light and networks [PG].** Light is the irreducible latency. Buffer, hold state, commit. The causal mask is a light cone.
- **v2's 8D is the cone.** "The inner cone points inwards; the object imagines itself" needs a focus boolean per depth. In attention:
  - the causal mask is the inward cone, apex at each position, opening back to ∅ = position 0, the sink;
  - the softmax weights are the focus boolean;
  - resting on ∅ = focus off;
  - T11 measured the boolean flipping.
- **v2's 9-gon web** is "not capped; the machine only stops drawing": the context window.

---

## 6. T10: attention as a flow

Registration `prereg_T10.txt`, sha256 `11741dad72e7312ee443f63d550088fe679034461c225b4d38991c6c9b427b9b`, sealed before any code ran.

**Inputs:**
- v1's sealed captures (the residual stream entering layers 0, 1 and 14; a 508-token ramp prompt and a random prompt);
- layer-14 tensors fetched by byte range, each **equal to v1's sealed `source_sha256`**.

**Instrument:** f64 on exact BF16 inputs.

| | prediction | result |
|---|---|---|
| A1 | tokens cluster with depth | **held**: mean cosine 0.038 → 0.322 → 0.514 (ramp), 0.053 → 0.137 → 0.549 (random); control ≈ 0 |
| A2 | effective rank falls | **held**: participation ratio 88 → 9.2 → 3.7 (ramp), 176 → 34 → 3.3 (random) |
| A3 | the sink's share rises monotonically | **killed**: 0.002 → 0.001 → 0.950 on random (dips at L1), though the sink holds 95% of the energy at L14 |
| B1 | the coupling is (non-)reciprocal beyond control | **held**, 16/16 heads, but 12 lean *reciprocal* (a = 0.23–0.49) and only 4 lean non-reciprocal (0.504–0.510) |
| B2 | ≥ 8 heads rest ≥ half their mass on the sink | **held**, exactly 8/16 |
| B3 | 99.9% of the mass fits in < 25% of the context | **killed**, 0/16: sets are 39–97% of the context; the tail is thin but long |

**Exploratory:** all 4 non-reciprocal heads (L14 h2, h4, h8, h9) most often read offset 1: 44%, 36%, 58%, 93% of rows. The front is the lag.

**For the "keep only the used positions as history" idea [PG]:** it isn't free at this fidelity, on this layer, on these (unnatural) prompts. Natural text is the real test [open].

---

## 7. T11: do heads leave the sink for what matters?

Registration `prereg_T11.txt`, sha256 `19b9c5f00b7a21f046140b760a30db2376e15e67f475db61292c9c4c06af1d0a`, sealed before any code ran. After the outcomes were appended: `10c8c339…94a3`. Run output `t11_run.txt`: `01411394…d8a3`.

**Prompts:**
- **REPEAT:** the first 254 random tokens, then the same 254 again;
- **RANDOM:** the control.

**Data:**
- embeddings taken from v1's sealed L0 capture;
- 154 tensors of layers 0–13 plus layer 14's q/k/norms, 1.41 GB by byte range, **all equal to v1's seals**;
- instrument: an f64 forward, 4 min 50 s on this laptop.

| | prediction | result |
|---|---|---|
| G | the instrument reproduces v1's sealed captures (cosine ≥ 0.99) | **pass**: 0.999965 (L1), 0.999870 (L14) |
| P1 | ≥ 2 heads drop sink ≥ 0.2 and put ≥ 0.2 on the induction target | **held, 3 heads**: L3 h10 (0.561 → 0.063, induction 0.795), L12 h0 (0.511 → 0.035, 0.380), L14 h6 (0.537 → 0.034, 0.244) |
| P2 | on RANDOM the same heads put < 0.05 there | **held**: 0.0014, 0.0005, 0.0010 |
| P3 | L14 previous-token heads keep offset 1 | **held**: all four |

**Exploratory:**
- **The sink released 1.477 of mass and the induction target received 1.419 (96%).**
- Near misses: L6 h11 (induction 0.632, barely rested), L8 h4, L13 h14.
- Layers 0–2 carry no induction; the first induction head is at layer 3.

**What T11 shows:** retrieval from context is attention leaving its rest state for the relevant position. Focus flips off ∅ to exactly the one position that matters.

**What T11 does not show:** that *all* learning lives in attention. Stored knowledge is attributed largely to the MLP layers as key–value memories (Geva et al., 2021) [lit]. "The residue reads like a book" [PG] is its own registration [open].

---

## 8. Posits carried [PG], untested unless stated

- **The pipeline:** the 41 table holds the byte patterns → serves them into Octavian cubes 1 and 2 → multiplies → checks on the third → lifts. 17 and 19 carry the logits in and out.
- **The pentagon as the collector of the matmul**, and RMSNorm fitting with it as the selection mechanism, *before* the icosahedral step (the 600-cell, held back on purpose).
- **Attention is physics:** fronts, the lag, the sink as rest.
- **The goal:** AI for students that is safe, checkable, updatable, and taken home, not served.
- **The shared-feelings theory:** stress, curiosity, anxiousness, frustration.

---

## 9. Next, when building resumes

1. **Exact attention from the construction's pieces.**
   - Rung/plane split of softmax (shift + a 128-entry plane table).
   - Two-sum collector N/D.
   - RMSNorm as the length closure.
   - RoPE's frequency table as the one declared irrational input.
   - Gate: agreement with the f64 instrument on the T11 prompts, then against v1's receipts where they exist.
2. **Register "the residue is a book".** What the MLPs hold, measured apart from attention.
3. **Natural text.** T10-B3 and T11 on real sentences, with the sink-release and selection-size measurements. This decides the history/selection idea.
4. **The BF16 → octonion-unit byte map** (doll S1, Tier C).
5. **Port V5's primitives from the WGSL in `card_ops.rs` back to Rust.** That removes the V5 crate dependency; the four tables remain [open].

---

## 10. Files (all under `v3-based/`, nothing committed)

- **Records:**
  - `prereg_T10.txt` (+ `.sha256`), `t10_run.txt`
  - `prereg_T11.txt` (+ `.sha256`), `t11_run.txt`
  - `SYNTHESIS-v1-v2-v3.md`
  - `tierb_run.txt`, `kernel_run.txt`, `bf16_run.txt`, `footprint_run.txt`
- **Code:** `walk/src/bin/{main,tierb,kernel,bf16,footprint,t10,t11}.rs`, `walk/src/lib.rs`, `walk/examples/psi_probe.rs`, `walk/export_t6.py`
- **Weights:** `t10/fetch.py` (byte-range fetch; admits a tensor only against v1's seal) and 159 tensors, 1.42 GB. Keep these out of git.
- **Bundle:** `bundle/` (the unzipped session bundle).
