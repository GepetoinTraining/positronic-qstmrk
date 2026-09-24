# Next stage: the kernel as one running object, with no weights

Brief for the next session. Written by Claude (Opus 5.5) at Pedro Garcia's request, 24 September 2026, at the close of the session recorded in `SESSION-2026-09-24-code.md`.

Stamps as in `day_v2.pdf`: [PG] Pedro's posit or rule · [C] Claude's reading, to be corrected · [run] measured · [open] untested.

## The rule [PG]

**We will not have weights.** They are loaded once, to check: fetched by byte range and admitted only when their bytes hash to v1's seal. T10 and T11 did exactly that for 159 tensors.

After that, each weight is eliminated in favour of **the position it inhabits**, which is the next compositional layers. Everything is traced to its own spot, and each system is seated.

## What the rule means, concretely [C, to be corrected before building]

1. **A value is a position, never a number.** Every BF16 pattern is already a point:
   - v1 §17: (sign, residue sign, k where the odd is ±5ᵏ, rung n);
   - today's 41 walk: (half, index), where the index is a discrete logarithm.

   Multiplying values is adding positions:
   - exact in the pentagon's coordinates;
   - Mitchell's form (missing only the mantissa cross term) in the walk's.

   So the object never holds a weight value. It holds where the value sits, and composition happens by moving.
2. **A weight's seat is its address in the composition.** Layer, then block (attention or MLP), then head or plane, then row and column. Seating means that address and the value's position are one coordinate of one object, not a table beside a dictionary.
3. **What must not be lost.** The seated object has to carry the placement's information in its positions. The proof that it does is the round trip in gate S0: every admitted tensor re-emerges bit-identical from the seated object alone.

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
  - Take layer-14 `k_proj` (4 MB, sha `3ed136be…`). Seat it, then delete the weight file.
  - Gate: it re-emerges bit-identical from the seated form alone (sha equal to v1's seal).
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

- Keep a weight tensor after S0's round trip has passed for it.
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
