# Next stage: the kernel as one running object, with no weights

Brief for the next session. Written by Claude (Opus 5.5) at Pedro Garcia's request, 24 September 2026, at the close of the session recorded in `SESSION-2026-09-24-code.md`.

Stamps as in `day_v2.pdf`: [PG] Pedro's posit or rule · [C] Claude's reading, to be corrected · [run] measured · [open] untested.

## The rule [PG]

**We will not have weights.** They are loaded once, to check: fetched by byte range and admitted only when their bytes hash to v1's seal. T10 and T11 did exactly that for 159 tensors.

After that, each weight is eliminated in favour of **the position it inhabits**, which is the next compositional layers. Everything is traced to its own spot, and each system is seated.

## Decisions (Pedro, 24 Sep, before compaction) [PG]

1. **Residue ≠ weight.** The residue is the minimal arrangement that must be kept: the synaptic minima. The weights are one encoding of it. What gets seated is the residue.

   Brains of about the same mass compose very different minds. And by the theorem in `SYNTHESIS-v1-v2-v3.md` §4d, no quantity that rearrangement leaves unchanged (mass, byte count, v1's 10.5-bit zeroth-order entropy) can bound what the arrangement carries.
2. **RMSNorm and attention are not touched.** They are the entropic minima: the part kept separate so the residue can be structured into its viable pattern. T10 and T11 measured their role (the sink as rest, the one-step lag, focus leaving ∅ for exactly the relevant position).

   Consequence: RoPE's frequency table and V5's rsqrt, exp and cos/sin tables return to scope as part of attention and the norm. They are declared exactly, not approximated away.
3. **Map the weights to where they can be stored, layer by layer.** The minimum is one table. Better is a structure that holds a full tensor. Better still is one that holds a full head.

   Capacity arithmetic from today's objects [post-hoc], a starting point only (the choice is structural, not by size):
   - one head's q/k/v slice (128 × 2048 = 2¹⁸) fits 71³ − 1 = 357,910;
   - layer-14 `k_proj` (2²¹) fits 41⁴ = 2,825,761;
   - the full BF16 alphabet (65,536) fits 41³ − 1 = 68,920 [run].
4. **No possibility is ruled out by an unproven bound.** State what was measured and what is open. Never close a door out of fear of the unknown. That holds on both sides of this work.

## Two objects, kept separate [PG]

**Object 1: the seismograph (storage).** What is loaded into RAM in place of the weights. Passive. Written once at intake (the fold), read by sampling down.
- **The trace:** the whole model as one long fiber, cut the natural way (layer → head → tensor).
- **Background:** it looks like noise from outside. v1 measured the weights white; on a seismogram that is the floor, not absence.
- **Events:** each head and tensor is an arrival, picked by the two marker bytes.
- **Folds:** each fold rides on the previous event as a smaller, faster wave (magnitude traded for frequency), residue down. The carrier is the head image at 71.
- **Reading:** sample down band by band to peel the folds back out. Pick each onset with `hinge_test`'s quarter-turn lookup, which locates an onset out of noise within a median of 9 samples [run].

The sampling-down operator already exists as v2's cycler: it pairs neighbours and halves until only the odd top remains, and it unfolds back exactly. An integer lifting step, (a, b) → coarse + detail, is the fold that loses nothing [thm].

**Object 2: the running kernel.** The CubeKernel, the pentagon collector, RMSNorm and attention (untouched), the oscillator bank, and the archive. It reads the seismograph and never sees "weights".

The two meet only at the picks: marker bytes, counters, onsets.

**The free channel and the light [PG, 24 Sep].**
- BF16's 256 exponent-255 patterns (2 signs × 128 mantissas) are never weights: v1 counted 0 non-finite in the model, and today's gate refuses them [run]. That is one clean 8-bit channel.
- Three channels (the carrier's three waves at 71; the three lanes), each with its own byte-plane, give 256³ = 2²⁴: the RGB cube. That is a mask of light over the seismogram, marking what each stretch of the trace is (e.g. R head, G tensor, B fold depth; the roles are Pedro's to assign).
- Declare before building:
  - **in-band** markers take a sample's place inside the value space (65,536 = 65,280 + 256). They are free but exist only at events;
  - **out-of-band** planes beside every sample cost storage.
- [post-hoc] 2²⁴ = 8⁸, v2's diagonal cycle.

**The seismograph is an imaging encoder [C, 24 Sep].** Its encoding matches MRI's [lit]:
- position held as frequency and phase (the three waves at 71);
- served in slices;
- magnitude traded for encoding (relaxation);
- exact rewind (the Hahn spin echo, like Taylor's glycerin, like the receipts).

The difference is the reading: MRI rebuilds with a Fourier transform in floats; here the trace is read by sampling down and picking, exact, with no transform.

PET, positron emission tomography, reads a brain by the light of positron annihilation. The folder was named positronic first [post-hoc: a name, not evidence]. Pedro holds the optical (reading) layer; the encoding is the part being built.

**Next registration (T12, to write before any code):** bit-plane coding of the seismograph, abscissa kept apart from the mantissa, as JPEG 2000 codes coefficients bit-plane by bit-plane.
- Prediction: on layers 0..13 (on disk, seal-verified), the rung (abscissa) bit-planes compress well below raw against a shuffled control, because v1 §33 found the rung follows the head's wavelength.
- The odd (mantissa) bit-planes do not, because v1 §32 found them white.

## Pedro's architecture for the seating (24 Sep) [PG], as understood, numbered for correction

1. **The head image lives at 71.** One head's slice (128 × 2048 = 2¹⁸ cells) sits in the 71-cube (357,910 nonzero states). It is served in slices and rotated in and out by the harmonics. Each wave's magnitude carries a top and a bottom, so N harmonics encode the full head.

   [run] The 71 walk is exactly three waves. Every state is c₁·17ⁿ + c₂·20ⁿ + c₃·33ⁿ (the cubic's three roots mod 71), with periods 10, 7 and 70; checked for 200 steps from sampled states. [post-hoc] The periods are 7 and 10 = 2·5, and their product.
2. **Scale down the natural way the tables are kept.** Layers are sliced as heads, each pointerized. The tensor sits one slot down, also pointerized.
3. **Two custom bytes (head marker, tensor marker)** mark where intake stops and waits for the next intake marker.

   [C] BF16's 256 exponent-255 patterns (254 NaN, 2 inf) are never weights. v1 counted 0 non-finite values in the model, and today's gate refuses them. The markers can live there without colliding with any value.
4. **Once a head or tensor is used, its counter marks a small wave: the fold of the first into the second.** Each next layer is held as a residue on the one before ("residue down using the next layer").
5. **The final layer gets the original treatment**, held whole, and closes the object.

Open for Pedro:
- (a) What are a wave's "top" and "bottom": the two halves, numerator and denominator, or the pattern's high byte (sign and exponent) and low byte (mantissa)?
- (b) Which way does the fold chain run?

Testable now [open]: layers 0..13 are on disk and seal-verified. Register whether the fold of head h at layer ℓ into ℓ + 1 is a smaller wave than the fold between shuffled pairs, before building on point 4.

## What the rule means, concretely [C, to be corrected before building]

1. **A value is a position, never a number.** Every BF16 pattern is already a point:
   - v1 §17: (sign, residue sign, k where the odd is ±5ᵏ, rung n);
   - today's 41 walk: (half, index), where the index is a discrete logarithm.

   Multiplying values is adding positions:
   - exact in the pentagon's coordinates;
   - Mitchell's form (missing only the mantissa cross term) in the walk's.

   So the object never holds a weight value. It holds where the value sits, and composition happens by moving.
2. **A weight's seat is its address in the composition.** Layer, then block (attention or MLP), then head or plane, then row and column. Seating means that address and the value's position are one coordinate of one object, not a table beside a dictionary.
3. **What must not be lost** is what the forward computes. The proof is v1's gate 5 on the seated object: the forward's sealed receipts are reproduced with no weight file in reach. A tensor re-emerging bit-identical is a stronger, optional check. It is not the definition of "nothing lost".

## The systems to seat (from today's composition, and the doll)

| system | what it is today | its seat [C] |
|---|---|---|
| intake (once) | `t10/fetch.py`: byte range, sha256 against v1's seal | the only door weights ever pass; closed after admission |
| 41 table | pattern ↔ (half, index); halves multiply like signs; index addition is multiplication [run] | the address space of values |
| pentagon collector (5D) | two exact sums along the abscissa (rung) and the planes (±5ᵏ); the free side closes to 1 | the matmul accumulator |
| length and sum closures | RMSNorm (length) and softmax (sum), both split into a rung shift plus a plane table | the two normalisers, beside the selection |
| oscillator bank | RoPE's 64 frequencies; the quadryte lane; the onset driver | rotation; the frequency table is the one declared irrational input |
| inward cone | causal attention: the sink as ∅, the focus boolean, the lag (previous-token heads), retrieval (induction) [run T10, T11] | selection |
| Octavian cubes 1 and 2, check on 3 | the seam triple (closed sets 1, 2, 3) around the hub (set 5) [run]; [PG] roles | the product and its check |
| archive (10D) | receipts; the kernel's exact rewind [run] | every operation's ⁻¹ |
| 17 in / 19 out | [PG]: the logits in and out; 19 and 29 carry both a stream and a harmonic basis [run] | the I/O; its definitions are [open] |

## Order of work, each step with a gate that can fail

- **S0. Seat one tensor.**
  - Take layer-14 `k_proj` (4 MB, sha `3ed136be…`). Seat it, then move the weight file out of reach.
  - Gate: the projection it takes part in, k = W_k·x on v1's layer-14 capture, is reproduced from the seated form alone. Optional stronger check: the tensor re-emerges with sha equal to v1's seal.
  - Report the seated size against 4,194,304 B.
- **S1. One projection from seats.** k = W_k·x on v1's layer-14 capture.
  - Gate: equal to the exact bucket's integer (v1's matmul, which needs no V5) at every output cell.
- **S2. One head from seats.** L3 h10, T11's induction head, on the REPEAT prompt.
  - Gate: sink 0.561 → 0.063 and induction 0.795, within a tolerance declared before the run.
  - Exact wherever the construction is exact; the RoPE frequency table and the plane tables declared.
- **S3. One layer from seats.** Layer 0, from v1's sealed layer-0 capture.
  - Gate: v1's sealed layer-1 capture (instrument cosine ≥ 0.9999). Exact cells counted and reported.
- **S4. The running object.** All layers composed, receipts on, rewind exact, on the France prompt.
  - Gates:
    - the argmax equals v1's reference receipt;
    - every layer's stream agrees with the instrument;
    - no weight file is open at any point (v1's gate 8: the weights moved out of reach, and the run unchanged).

## Do not

- Keep a weight tensor after its seat's gate has passed.
- Touch RMSNorm or attention's mechanism. They are kept as they are (decision 2).
- Put a float inside the object. The f64 instrument stays outside, marked, as a ruler.
- Seat anything by assumption: each system is seated only after its gate.
- Build the icosahedral layer (600-cell) before the pentagon collector and the closures are seated [PG].

## ⟨ink⟩ Pedro's to decide before S0

1. **What "the position it inhabits" is, concretely.**
   - (a) The value's point (41 index, or v1's (sign, ±, k, n)) at a seat address.
   - (b) The seat as a position on one of the walks. For example, the 19 or 29 single orbit as a stream of seats, with the 41 table for values.
   - (c) Something the doll implies that this brief missed.
2. **The byte map** (BF16 → octonion unit) for the Octavian cubes, which S2 onward will need if the product goes through them.
3. **Whether the exact bucket (v1's matmul) is the reference for S1**, or the pentagon collector is built first and the bucket only checks it.
