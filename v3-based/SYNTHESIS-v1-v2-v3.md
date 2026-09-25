# Synthesis — v1, v2, and today's v3

Written by Claude (Opus 5.5) for Pedro Garcia, 24 September 2026. Read-only on v1 and v2: nothing was imported or run from them; every cross-version number below was re-derived today and is marked.

Stamps follow `day_v2.pdf`:
- **[run]** measured today by a named script;
- **[thm]** standard;
- **[PG]** Pedro's posit;
- **[C]** Claude's reading;
- **[post-hoc]** arithmetic noticed after the fact, not evidence;
- **[open]** untested.

v1 and v2 numbers are quoted with their own **[M]**.

**What was read:**
- **v1:** `README.md`, `GOAL.md`, `ATLAS_ENGINE_SPEC.md`, `LOG.md` §1–§33, `native_object.pdf`.
- **v2:** `BUILDER_LOG.md` (all), `BRIEF-turns.md`, `docs/DESIGN-HARNESS-BRIEF-CYCLER.md`, `docs/posit/02-posit-and-protocol.pdf`, `docs/alignment/Prime Abacus.pdf`.
- **v3:** `day_v2.pdf`, `prereg_T9.txt`, `TIER_B_for_code.md`, the session bundle, and today's Rust in `walk/`.

The two-registers letters were read as preparation only. Per v2's house rule they are not summarised here.

---

## 1. The three builds, one line each

| | dates | what it is | its headline receipt |
|---|---|---|---|
| **v1 · Atlas** | 13–14 Sep | Qwen3-1.7B run float-free from its own cells. Every stage either rewinds or refuses. | 508-token hash `63708bfa…` identical on CPU and GPU; 0.17 s/token on the 3060 Ti; the dictionary is 15 KB and the weights are 3,875 MB of placement [M] |
| **v2 · construction** | 14–22 Sep | Numbers built from the floor, not imported; the weights read as geometry | 4,505 distinct magnitudes = 433 handles × 63 residuals; `e` fits the 27-cell cube with centre 20; the cycler's cycle 1 is the complete square 126² [M] |
| **v3 · today** | 23–24 Sep | The day's chain (T2, T6–T9) and the kernel tower built in Rust, driven by the onset oscillator | 22/22 layer checks; 10⁶-op rewind bit-identical; BF16 gate 65,280 / 65,279; the 41 load at 3,924 B [run] |

---

## 2. What carries through all three (the spine)

| thread | v1 | v2 | v3 today |
|---|---|---|---|
| **Receipts decide** | sealed hashes; gates abort, never paper over | sealed receipts, the control always beside the real count | pre-registrations with kill criteria; stamps; a post-hoc bin |
| **The alphabet is the format's, not the model's** | H1: all 128 odd atoms in every tensor group; census 65,280/65,280 [M] | 128 odds; every weight is ±odd/2ᵉ [M] | the gate admits **65,280** (signed zero kept) [run]. That is v1's census number, reached independently |
| **One free rotation: {1, i, −1, −i}** | the clock: ×i is a swap and a negate in Z/998244353, no multiply | the oscillator "turning through {1, i, −i, −1}" | the quadryte lane (σ² = flip); the driver's quarter-turn lookup; ring exponent +9 (5 = 2⁹) |
| **The sign stands beside the cell** | "Sign is beside the cell" (GOAL) | the plane is the sign; 3,405 of 4,505 slots are planes (both signs) | the binary layer is its own bit. The no-go [run] proves it *has* to be: 2T cannot act on the lanes with −1 = flip |
| **Hurwitz quaternions** | the ball: SPIN_Q with norms {2, 3, 5, 7, 11, 13}. Norm 2 closes in 4; the odd-prime norms never close, so the frame scale \|q\|²ᵏ has to ride along [M] | the turns table: the 24 units as signed permutations; c = ½(1+i+j+k) is the axis cycle | **units only** (24 and 240), so every group is finite, closes, and rewinds from a finite receipt. This is the design shift that makes exact rewind cheap [C] |
| **Symmetries relabel; crossings create** | — | a pure axis permutation (c, c²) changes no name; only the "−" reading fills holes [M] | the twist maps closed sets to closed sets (1+3+3) and every invariant holds under it. New states come only from Catch (across sets), Lift and the sign [run] |
| **7, 11, 13** | the acquisition clock: coprime rings 7, 11, 13; CRT coordinate in 0..1001 [M] | — | the 7//11 hinge *inside* Z/13 (7·11 ≡ −1, 7/11 = 2⁴). Same three primes, different structure: a product ring there, residues here [C] |
| **The 9-gon** | — | 140°/220° with the beak at the oscillator; torq residues landing on 0, 40, …, 320 | a walk (x+7, y+11) on Z₉², 9 loops of 9, x+y conserved [thm/run] |
| **3³ plus a sign** | — | the cube byte type for e: 27 cells + side, centre 20, and **−000 unused (the ±0 pair)** [M] | the tryte: 27 positions (26 + ∅) with signed zero. BF16's only duplicate value is **−0** [run] |

Two cautions inside that table:
- **The 9-gon in v2 is not support for v3's 9-gon.** v2's own log shows the residues landed on multiples of 40 because every value had been inflated by ×10^46 ("why so inflated?"): 40 divides V, and the vertex is the core mod 9. That landing is forced by the decimal inflation. [C]
- **The cube and the tryte have the same shape but not the same arithmetic.** v2's cube is sign plus magnitude in ternary: e runs −26…+26 around centre 20, 53 codes. v3's tryte is three balanced-ternary lanes with a signed zero: 27 positions, flag per zero lane. Both are "3³ plus a sign, with one ±0 duplicate". Neither maps onto the other without a declared rule. [C]

---

## 3. Cross-checks re-derived today

| check | result |
|---|---|
| v1's 7,910 signed patterns (LOG §8: + 4,384, − 3,526) against v2's 4,505 magnitudes (3,405 planes + 979 only-plus + 121 only-minus) | **3,405 + 979 = 4,384 ✓; 3,405 + 121 = 3,526 ✓.** The two logs agree cell for cell [run] |
| v1's census 65,280, and 65,278 nonzero patterns / 32,639 magnitudes (§15, §17), against today's gate | 65,280 with signed zero kept; 65,279 by value; 32,640 per sign including zero, so 32,639 nonzero ✓ [run] |
| How much of BF16 the model uses | 7,910 of 65,280 finite patterns (12.1%); 4,505 of 32,639 nonzero magnitudes (13.8%) [run, arithmetic] |
| v2's cube centre | e −6…46, span 52 = 2·26, centre 20 ✓ [arithmetic] |

**One inconsistency in the stamps themselves.**
- The repo README and the v2 cycler brief define **[F] = "floatland reference only"**.
- v1's LOG and v2's BUILDER_LOG define **[F] = "forced by a file or theorem"**.

Same letter, opposite meanings. Worth fixing before anyone reads an [F] number cold.

---

## 4. Where the versions disagree, or v3 changed something

### 4a. The oscillator relation has two brackets (the most important item here)

| | written | polynomial | discriminant | roots |
|---|---|---|---|---|
| v2 (BUILDER_LOG l.177, checked at l.187) | X² − 1 = **1/x + 1** | x³ − 2x − 1 = (x + 1)(x² − x − 1) | **5** | −1, φ, −1/φ (the maelstrom's φ, v1 §15) |
| v3 (day_v2 §8, T8, today's census) | x² − 1 = **1/(x + 1)** | x³ + x² − x − 2 = (x − 1)(x + 1)² − 1 | **−59** | one real root and a complex pair; no rational root |

Walk census on the nonzero p³ cube, both readings [run]:

| p | v3 reading (today's walk) | v2 reading, a(n+3) = 2a(n+1) + a(n) |
|---|---|---|
| 7 | 2 × 171 (two halves) | 3 × 2, 21 × 16 |
| 13 | 3 × 4, 8 × 21, 24 × 84 | 6 × 2, 78 × 28 |
| 17 | 2 × 8, 306 × 16 (**binary clocks**) | 8 × 2, 136 × 36 (not binary) |
| 19 | **1 × 6,858 (one orbit)** | 9 × 2, 2 × 9, 379 × 18 |
| 29 | **1 × 24,388 (one orbit)** | 14 × 2, 4 × 7, 1,738 × 14 |

- Every role the doll gives 17, 19, 29 and 41 exists **only under v3's bracket**. Under v2's, the polynomial has the factor (x + 1), so the walk can never be a single orbit [thm].
- ~~Or one of them is a mis-bracketing.~~ Withdrawn: that treated the difference as an error before checking it.

**The two readings sort the primes together** [run]. Among the Monster primes, only 19 and 29 have both of these:
- v3's walk is a single orbit;
- v2's φ exists mod p as two distinct elements (x² − x − 1 splits).

(3 and 5 are single orbits too, but φ is missing at 3 and a double root at 5, v2's discriminant prime.)

| p | v3: one orbit | v2: φ mod p |
|---|---|---|
| 3 | yes | — |
| 5 | yes | double (3) |
| 11, 31, 41, 59, 71 | no | splits |
| **19** | **yes** | **5, 15** |
| **29** | **yes** | **6, 24** |

Reading [C]: at 19 and 29 the oscillator does not close as one orbit because it *resolves*. It splits into three lines of its own: the −1 line (the flip), the φ line and the −1/φ line. Those are its eigen-directions, i.e. its harmonics. Over the same prime, the walk is a single stream through every state. So 19 and 29 carry both a pace and a harmonic basis.

Whether that is the role intended is **[PG]**; it can be registered and tested [open].

### 4b. "The id is a position": the 41 address is measured, and it composes (corrected)

A first draft said the 41 index was only a storage address. That was wrong. Checked [run]:
- The walk's step is multiplication by α in GF(41³). α has order 34,460 = (41³ − 1)/2.
- The two halves are the squares and the non-squares.
- α^i · α^j = α^(i+j) held on 200 random pairs.
- A non-square times a non-square lands in the squares' half. **The halves multiply like signs** (− · − = +), with the half containing the field's 1 playing +.

So the address is a measurement, a discrete logarithm: products of loaded states are sums of indices.

Since the index *is* the BF16 bit pattern (sign → half, 15 bits → index), adding indices means adding bit patterns. That is **Mitchell's logarithmic multiplication (1962)** [thm]:
- exact on the sign and the exponent;
- it drops only the mantissa cross term m₁·m₂, a 7-bit × 7-bit product (a 128 × 128 table).

This is v1 §17's idea ("1/value = id", products as sums of positions), reached through the walk. v1 uses ±5ᵏ on the odd atoms; the walk uses the index on the whole pattern. [C]

Arithmetic still to declare:
- Index sums wrap mod 34,460, not 32,768.
- The exponent bias is a fixed offset.
- The 1,692 spare states per half are ≈ 13.2 binades of headroom past the top of the range (1,692 / 128) [post-hoc].

So for the pipeline [PG] (41 holds the patterns → Octavian cubes 1 and 2 → multiply → check on the third → lift): the 41 layer already multiplies in log form. The open piece is how the Octavian layer's product relates to it: the **BF16 → octonion unit map is not defined anywhere yet** (§5.2).

### 4c. The footprint: all three agree the alphabet is small and the placement is the model

- **v1 §20:** "The dictionary is 15 KB; the footprint is placement." The ids are the weights byte for byte (3,875 MB).
- **v1 §21:** the id stream's zeroth-order entropy is about 10.5 bits per cell [M]. ~~"caps any coding"~~ Corrected: it bounds only a code that writes each id alone from its frequencies. It is equal for the model and its shuffle (§21's own control), so it cannot measure what the arrangement carries. See §4d below.
- **v1 §22:** the low modes of the "sheet" did not regenerate the weights [M]. That is one model failing, not a bound.
- **v2:** the lightest packing is 3.303 GB (13 bits per cell); "don't pack it back as a table" is still unresolved.
- **Today's 3,924 B** holds the alphabet (all 65,536 patterns as walk positions), the Octavian triple and the kernel state. It does **not** hold the 2,031,739,904 placements.
- Today's result makes the alphabet side very small. It doesn't touch the placement, and nothing in v1, v2 or v3 has shrunk the placement below about 3.3 GB. [C]

### 4d. Why no count of the weights can bound what they carry (Pedro's theorem, 24 Sep)

A quantity that doesn't change when the parts are rearranged can't determine a property that does change when they're rearranged [thm]. Take the object and a rearrangement that destroys the property; the quantity is equal on both. Mass (brains of about the same mass compose very different minds), byte count and zeroth-order entropy are all unchanged by rearrangement. At most they give a ceiling (as the Bekenstein bound does for mass), never the content. What the model is lives in the arrangement: the residue, which is not the weights.

### 4e. BF16 at the door

- v2 says v1 "reproduces floatland's own number" (15 breaks).
- v3 takes BF16 in at the 41 intake. That is consistent with v2's house rule: "float layout: claimed, found and forced by pgarcia. Not a flag."
- Inside the kernel nothing is float; the only floats today are in the driver's simulated physics, as in `hinge_test.py`.

---

## 5. The open seams, in the order they block

1. **The two brackets (4a).** They read as two views of the same primes, not an error. At 19 and 29 there is both a stream and a harmonic basis [run]; what those primes are *for* is [PG]. It can be registered.
2. **BF16 → octonion unit (Tier C, doll S1).** 65,536 patterns against 240 units per cube.
   - What the logs already constrain: the **rung** carries the model's structure. v1 §33: the power of 5 = the rung lines up with the head's wavelength, z up to +1,014. The **odd** carries none of it (§32: nothing lines up, to half a percent).
   - Count match [post-hoc]: 128 odd atoms = 2⁷, and the 240 integral octonions contain 128 even-parity half-sign patterns. So a 7-bit → (7 signs + parity) map exists on paper.
   - But it lives in the E8 coordinates, not the Kirmse coordinates the kernel uses, and it's untested [open].
3. **17 and 19 carry the logits in and out [PG].**
   - v1 already produces exact logits and the argmax, bit-identical.
   - v1's Prediction 5 (the logit collapse is injective) is still a bet.
   - The runtime roles need definitions before anything can be measured.
4. **The placement.** v2's "7D schema" for the handles is not given; no model tried so far regenerates it (v1 §21–§22), and no bound says one can't.
5. **doll.tex registrations.**
   - 1 (pair-gated hinge): held today, but it's close to true by construction.
   - 2 (S1 is lossless) and 3 (the semispace rewinds): open; both need the runtime.
   - 4 (hub catches): killed as T9 A4.

---

## 6. Status at a glance

| | built and passing | measured, not a claim | open |
|---|---|---|---|
| v1 | exact forward, CPU and GPU; two clocks; harness; tokenizer | rows are white; the wave lives along positions (§14); the rung knows the wavelength (§33) | B1–B5 of the spec; Prediction 5 |
| v2 | atlas pipeline; the cube byte; the cycler; turns; the side-run walk | up×downᵀ turns across depth; frontier bands move with the hull | the 7D schema; 4D past cycle 9; 9D cycle 2 (infeasible as written) |
| v3 | kernel tower; driver; BF16 gate; 41 load; Tier B | the seam triple (not the hub); bracket count 55,944/57,600; onset lag 9 samples | the bracket; the byte map; the runtime |
