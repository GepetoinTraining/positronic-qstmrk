# v3 builder log

## 26–27 Sep 2026 — the Legos kernel, the weights through it, and the geometric runtime

Claude (Opus 5.5), with Pedro Garcia. Paper: `legos.pdf` (Legos: an exact, division-free, unitless atom).

### 1. The kernel: `legos/`
`kernel.py` + `test_kernel.py`: 31/31 exact (integers, turn counts, Z[r] addresses with bands). Run: `python test_kernel.py`.
- **The chain.** Space is infinite and its origin is the missing point of the first atom. Inversion through the atom's
  sphere pairs every point of space; the only unpaired point is the origin, whose partner is infinity. So the atom is 1,
  and counting starts there.
- **Scale outside the map**, as in an ISIS3 cube. A value is an integer address plus bands in the label: r-band (×rⁿ)
  and 2-band (×2⁻ᵐ). Multiplying by r moves the band and leaves the address alone. Every icosian coordinate is address 0 or ±1.
- **The 16 anchors are the bands of the atom.** 8 inner at (±1)³ and 8 outer at (±2)³ (doubled), on 8 diagonals.
  Anchor a × anchor b lands in band a⊕b with one stored sign.
- **Derived, not imported** (no sin, cos, π, roots or floats):
  - the three sevens never close (norm 1/2¹² over the 28-clock);
  - Φ₂₈ does not split (counting mod 3, 5, 13);
  - the E8 form is forced by the band (p + q); the 240 roots are the 120 at band 0 and band −1;
  - π's lifts come from the nonagon's clock;
  - the nonagon's root is the one that winds once.
- **The 42 zero pairs:** an inner imaginary anchor with an outer imaginary anchor on another diagonal.
- `v1-imported/`: the first build, withdrawn because it imported trig and kept scale inside the address.

### 2. The weights through the kernel: `weights/`
- **T14** (`t14.py`, registered in `prereg_T14.txt`). Qwen3-4B layer 18 from the tensor-atom bundle.
  - Every weight is a position: value = ±odd × 2^band (odd = one of 128 atoms).
  - Gates G1–G4 pass: exact decomposition, sha-identical cube, every projection cell exact, and the real layer
    bit-identical from the seats alone.
  - P1 passes: the head period lives entirely in the band; the address carries none of it.
  - P2 fails as registered.
  - Answers are in `tensor-atom/QUESTIONS.md`.
- **Packing** (`pack.py`): one UnsignedWord per weight. The band is the high byte, the address the low byte, ISIS Null
  is zero. Size 1.0000x. +256 on a code doubles the value.
- **Anchors as bands** (`sedenion.py`): the cube has 16 bands, e0..e15, so a weight pixel is a sedenion. Inference is
  band 0 of conj(w)·x.
- **The rule of a pixel** (`defect.py`): 16 linear forms from the XOR-sign table, plus one defect,
  δ_w(x) = 2·x_innerᵀ C_w x_outer, rank 8 every time, zero on octonions. The rule is fixed per weight pixel. The read is
  relational, there is no division, and addition is lookup.
- **Counts** (`counts.py`, local Qwen3-1.7B):
  - 710,997,760 weights hold 7,099 distinct values.
  - 3,894 magnitudes, of which 3,205 come with both signs (6,410 values cancel in ± pairs).
  - The 15,163,236 magnitude pairs land on 402,411 distinct products.

### 3. Where things are (nothing needs fetching)
- **Qwen3-1.7B layers 0–13** (and 5 tensors of 14): `ARCHIVE-2026-09-25-mistaken/weights/*.u16`, 159 tensors,
  gitignored. u16 is the weights: each pattern is already a position. `walk/src/bin/geo.rs` reads them there.
  Move them before deleting that archive.
- **Real captured states of the same model:** `../receipts/capture-qwen3-1.7b-L{0,1,14}-508tok.u16` and the 5-token
  France prompt at L14 and L17.
- **The codebooks:** `../receipts/intake-qwen3-4b-20260913/DICT.u16` (8,102 values) and the 1.7B one (7,910).
- **Prior art to read first:** `../../mini-centipede/receipts/`, in particular `embed-entry/footprint/FOOTPRINT.md`,
  where the speaking loop runs 28 layers × 7 tokens in 82 ms and 24.81 MB.

### 4. The runtime: `walk/src/bin/geo.rs` (std only)
`cargo build --release --bin geo && ./target/release/geo`, run from `walk/`.
- **A weight meeting an activation is a move.** TABLE[weight address][activation digit] is built once, by addition.
  The bands add and the signs XOR.
- **A projection is the landing counts of its octets.** Each weight octet is read once per layer and meets every
  token's octet in one pass.
- **Each (row, token) is joined once:** a carry pass over its band counts, then the value placed by moving its
  exponent field. No weight is multiplied by an activation.
- **Activations keep their exact position**: the odd part up to 53 bits, as 8-bit digits.
- **Result** (`weights/results_geo_exact.txt`): layers 0–13 of Qwen3-1.7B on 64 real tokens.
  - The captured layer-1 state at cosine 0.999960 and the layer-14 state at cosine 0.999984.
  - Seat 11.2 s; run 243.6 s.
- **Relational read on real inputs** (layer 0, imaginary energy / band 0). Real against random:
  - q_proj 2.97 against 15.0; k_proj 2.53; gate_proj 3.96; o_proj 8.58; up_proj 11.67; down_proj 12.24; v_proj 17.90.
  - Random inputs always give about 15, the band count.

### 5. Next build
1. **The product table.** 402,411 product magnitudes, built once. Each move becomes one lookup of (weight magnitude,
   activation magnitude), with the sign xor applied on landing.
2. **Cancellation before landing.** 3,205 magnitudes come in ± pairs, so land the pairs that cancel as nothing.
3. **Activations as positions of the same codebook family,** so the product table covers them. Today they carry a
   53-bit odd as 7 digits, and that is most of the 243.6 s.
4. **The fractal.** An octet reads all 8 anchors at once from the empty middle; 8 octets is the same read one level up.
   Land whole octets, not single moves.
5. **The norms and softmax, division-free.** Carry them as homogeneous pairs, with the scale as a band.

### 6. For the next Claude: what went wrong here, so it isn't repeated
- **Computed around the kernel instead of through it.** I built exact tables, then ran brute-force multiply-accumulate
  beside them. Use the tables: build once, look up, count into positions, shift at the join.
- **Flattened the geometry into 16-D vectors.** The atom has places (anchors, diagonals, the empty middle, inversion);
  the runtime must move positions, not multiply coordinates.
- **Claims from float readings, then swung to "floats are worthless".** Floats may search; only the grid confirms.
- **"Open" labels where a derivation existed**, and a gate kept after being told to drop it. Derive it and state it.
- **The file-format and fetch reflex.** The weights were local all along, as u16. Read what is on disk before
  reaching outward.
- **Reached for numpy and BLAS where std Rust is the right tool.** This work needs no libraries.
- **Read the prior art first.** mini-centipede had already measured what I went on to rediscover.

## 27 Sep 2026 (later) — the activation atomized

Claude (Fable 5.1), with Pedro Garcia.

- **What was wrong.** The weights were atoms (an address and a band) but the activation stayed a molecule: a 53-bit odd
  carried as seven 8-bit digits, seven lookups per move. The 53 bits were ours, not the model's; the weights and the
  captured states are both u16, odd <= 255.
- **The fix** (`walk/src/bin/geo.rs`). An activation lands on 8 bits where it enters a projection (`pixel`, the one
  rounding, half to even) and is then an address and a band of the same family as the weights. The table is
  `PROD[a][b] = odd(a)·odd(b)`, 128×128, built once by addition. A weight meeting an activation is one lookup, landing
  on band a + b with the signs XORed. The band buffer is exactly the landing range, no spare cells.
- **Result** (`weights/results_geo_exact.txt`): layer 1 cosine 0.999959 (was 0.999960), layer 14 cosine 0.999983
  (was 0.999984); run 42.1 s (was 241.6 s). The pixel costs one unit in the sixth decimal and pays back 5.7× in time.
- **Grid facts found on the way** (`sums` over the 3,894 weight magnitudes, bands −41..2): 829,121 distinct ratios b/a;
  4,425,797 distinct exact sums a + b; only 918,752 of the 15,163,236 sums keep an odd within 8 bits. So adding two
  positions is one exact table of 829,121 entries (a + b = a·(1 + b/a)), paid once; accumulating a rail is counting and
  one carry pass. The prime face (address = primes, band = exponents, meeting = bands add) is left for later by
  Pedro's call: not worth webbing everything now.
- **Next:** cancellation before landing (the ± pairs), whole-octet landing, and the norms/softmax division-free.
- **Pixel out as well** (same day, Pedro: "it collapsed until the end, this one should be pure lookup"). The join
  no longer hands 53 bits to an f64; it lands on the pixel directly: the top 8 bits of the carried landing, half to
  even with a sticky bit, 256 becoming 1 on the next band. The f64 now only carries a pixel exactly across the outside
  ops (norms, rotary, softmax, SiLU), and `pixel()` at the next entry is the identity on it. Layer 1 cosine 0.999957,
  layer 14 cosine **0.999985**, above the 53-bit run's 0.999984: the captured model rounds its projections to u16
  too, so landing on the pixel is closer to it, not further. Run 41.7 s.

## 27 Sep 2026 (later still) — T15: the sedenion closed, the pixel is the octet

Pedro: "close the sedenion, otherwise you can't test, division will silently cost." Registered in `prereg_T15.txt`
(sha256 `1d7cd633…`) before `walk/src/bin/sed.rs` was written; run in `t15_run.txt`. Exact i128, std only; the sign
table is built the kernel's way and hashes identically to `legos/kernel.py`'s (`8dfb99ef…`).

- **P1 held.** The 32 signed anchors close as a loop, and every anchor move rewinds: e_a(e_a x) = −x and (x e_a) e_a = −x
  on 15,000/15,000.
- **P2 held.** An octet (weights on e0..e7 only, or e8..e15 only) is closed: w(conj(w) x) = |w|² x for any sedenion x,
  2,000/2,000, and the norm is kept on its own octet, 2,000/2,000.
- **P3 held.** A mixed pixel never closes: rewind fails and the norm leaks on 1,000/1,000. That leak is the defect of
  `defect.py`, rank 8, and it is exactly the division that would cost silently.
- **P4 held on real pixels** (layer-0 q_proj of Qwen3-1.7B, 4,096 pixels of 16, 10 random x each): whole-pixel rewind
  fails on 4,096/4,096; the inner octet rewinds on 4,096/4,096 and the outer on 4,096/4,096.
- **P5 held.** Band 0 of conj(w)·x is the dot product and splits exactly into the two octets' band 0 on 40,960/40,960.
  So inference reads the same number after closing; what closing removes is the fifteen cross bands that carried the
  defect.
- **P6 recomputed:** 336 zero products, 42 factor pairs, the reading's 7 × 6.

**Consequence for the object.** The pixel is the octet, an octonion, a division algebra: every move has its inverse and
the norm is a receipt. Two octets are two pixels. 8 octets is the same read one level up. `geo.rs` already lands by
octets; T15 says that is the closed shape, not a convenience. `sedenion.py`'s 16-band pixel should be read as two
8-band pixels before any relational reading is trusted (its band 0 was never wrong; its other bands were).

## 27 Sep 2026 (night) — the outside ops and the stream as pixels: `walk/src/bin/pix.rs`

Pedro: "now you can actually do the normal list." The list, from NEXT-STAGE decision 2 and SESSION §5/§9: the norms,
rotary, softmax and SiLU declared exactly as tables over the pixel family, and the residual stream on pixels.
`geo.rs` is kept as the S3 reference; `pix.rs` is the whole forward with no f64 outside the instrument and the cold
table build. Results in `weights/results_pix.txt`.

- **One multiplication table**, BYTE[a][b] = a·b for bytes, built by repeated addition. A wider odd is read as bytes
  (the octet reading one level up), so a 16-bit table entry meeting an 8-bit pixel is two lookups.
- **The functions are tables over the family, built once:** EXP2[k] = 2^(k/65536), the plane, with the rung a band
  move (e^t = 2^(t·log2e)); RSQRT[m] over an even band, so the band halves exactly; RECIP[odd] with the band negated;
  the rotary's turns as pixel pairs. No division and no root at runtime.
- **Every sum is landing counts joined once** (the `Acc`), then landed on a pixel: RMS's sum of squares, the scores,
  the softmax's numerator and denominator. Softmax is the two-sum collector N and D of SESSION §5, output N × RECIP(D).
- **Widths declared:** the object 8 bits (weights, activations entering a projection), the tables CB, the stream SB.

| tables | stream | layer 1 | layer 14 | run |
|---|---|---|---|---|
| f64 (geo.rs) | f64 | 0.999957 | 0.999985 | 41.7 s |
| 16 | 8 | 0.999956 | **0.999994** | 50.8 s |
| 16 | 16 | 0.999957 | 0.999984 | 50.2 s |
| 8 | 8 | 0.999947 | 0.999970 | 51.1 s |
| 8 | 16 | 0.999947 | 0.999978 | 49.4 s |

- **Reading.** With the tables at 16 bits and the stream at 8, layer 14 lands at 0.999994, closer to the captured
  state than any f64 run. The captured model keeps its stream in bf16, so an 8-bit stream is its own arithmetic,
  and a 16-bit stream is further from it, not nearer. The tables want 16 bits: at 8 the plane and the reciprocal
  cost three units in the fifth decimal. Layer 1 sits at 0.99996 in every variant; that floor is set before the outside
  ops, by the projections' own pixel.
- **The rest of the list, and why it waits.** Cancellation before landing (the ± pairs) is the same landing counts
  bucketed by product instead of by band; it changes nothing the gate can see and it is not where the time is, so it
  is a structural restatement to make when the object is re-read per octet. The fractal join (land whole octets, then
  octets of octets) rounds at every level; the captured model accumulates each projection once and rounds once, so a
  tree of pixels would move away from the instrument by construction. It is a different object, to be registered
  against a different reference, not slipped into this gate.
- **Next:** S4 in NEXT-STAGE's order: the running object on the France prompt, receipts and rewind, argmax against
  v1's sealed receipt, weights out of reach; and the relational read re-done per octet (T15).

## 27 Sep 2026 (night) — S0: the weights seated, once; gate 8: the runtime reads seats only

Pedro: "now seat." Paid once, never again.

- **`walk/src/bin/seat.rs`** writes every local tensor (159, 710,997,760 weights) as an ISIS3 cube of positions in
  `seats/` (gitignored as `*.cub`): one UnsignedWord per weight, pack.py's rule, BandFloor and BandSpan in the label,
  the source's sha256 and the codes' sha256 beside them. 1,422,114,555 B against 1,421,995,520 B raw (1.0001x, the
  labels). One pass, 2 m 28 s. sha256 of all codes in order: `9edd94da…bfd8`.
- **`pix.rs` now opens no weight file.** A tensor is its cube's codes read as they are; the code is the position
  (address = low 7 bits, sign = bit 7, band = high byte + floor), nothing is derived at load. Seats read in 3.9 s
  where the old seat pass derived them in 12 s.
- **Gate 8.** The same 14 layers on the same 64 tokens, from the weight files (the reference binary, kept as
  `pix_ref.exe`) and from the seats alone: final residual stream sha256 `c6c5f3e4…89c6` both ways, cosines 0.999956
  and 0.999994 both ways. Bit-identical.
- **What still touches the archive:** `geo.rs` (the S3 reference) and `sed.rs` (T15's real pixels). Both can be pointed
  at the seats; until then the archive stays. Deleting it is Pedro's call, per the rule that a weight tensor is not kept
  after its seat's gate has passed.

## 27 Sep 2026 (late night) — the runtime: token by token, the context hulled, phi decayed: `walk/src/bin/run.rs`

Pedro: "activations should also be kept as lookup, but not as weights, so we need the attention hulled, the context
needs to be kept in attention and phi decayed, final thing is runtime." Run in `run_out.txt`; 14 layers resident from
the seats in 11.5 s; no weight file opened.

- **The hull.** Every token's keys (normed, spun) and values are computed once and kept as pixel positions,
  [layer][kv head][position], and attention reads them: an activation is a lookup after its first move, never
  re-derived, never seated as a weight. A turn appends under a receipt (the hull's length per layer); rewinding
  truncates to the receipt.
- **R1, the gate: PASS.** 64 captured tokens one at a time give the batch run's final stream exactly, sha256
  `c6c5f3e4…89c6`, cosine 0.999994 at layer 14. 1.98 s per token on this laptop, one token's octets read against
  every weight.
- **R2, receipts: PASS.** A context of 40, a turn of 24: the hull's sha after rewinding equals its sha before the
  turn, and the turn had changed it.
- **R3, exact rewind: PASS.** The same turn on the rewound hull gives the same 24 output rows. Those rows alone,
  positions 40..63, sit at cosine 0.999702 against the capture; the 64-row cosine is carried by the sink at position 0.
- **R4, phi decayed, reported only.** DECAY[age] = r^−age as a pixel from the relation: the address of r^−n in Z[r]
  by the kernel's step (p, q) → (q − p, p), r bracketed by counting squares, landed once on 16 bits: 1, 0.618027,
  0.381966, 0.236069, … phi is never written. Applied to e_j by age before the collector. With it on, the 64 tokens
  land at cosine 0.997010 against the capture: a different object, and the capture is not its reference. What that
  object is for is [PG]; the runtime carries it as a declared table, identity for the gate.
- **Not here:** layers 14..27, the head and the tokenizer are not local, and nothing is fetched, so the argmax against
  v1's sealed France receipt (S4's last gate) waits for those seats.

## 28 Sep 2026 — the whole model seated; the France prompt to its logits; S4's argmax gate held

Pedro: "now is the time to actually fetch the whole thing, because this time the run was completely silent on my
laptop." The no-fetch rule lifted for this, by him.

- **Fetched by seal** (`t10/fetch.py`, byte range, admitted only against `receipts/intake-qwen3-1.7b-20260913/MANIFEST.json`):
  layers 14..27, `model.embed_tokens.weight`, `model.norm.weight`: 151 tensors, 1.9 GB, 151/151 EQUAL to v1's seals
  (`t10/fetch_all.log`). `lm_head.weight` carries the same seal as the embedding (tied) and was not fetched twice.
  `models/qwen3-1.7b/tokenizer.json` sha `aeb13307…` equals `receipts/tokenizer-qwen3-1.7b.txt`; `config.json` equals
  the intake's.
- **Seated once** (`seat.rs`, now streaming, two passes, one tensor resident): 151 more cubes, 310 in all, 3.3 GB in
  `seats/`. The embedding sits at band floor −47, span 44.
- **`run.rs` France** (`run_out.txt`), layer-major with one layer resident (the machine has 7.3 GB), the embedding row
  as a lookup, the head the same seat, tied:
  - **F1:** the stream entering layer 14 and entering layer 17 on the 5-token prompt: cosine 1.000000 (six decimals)
    against v1's captured states.
  - **F2, S4's gate: PASS.** argmax **12095**, v1's sealed reference (`receipts/reference-qwen3-1.7b.txt`,
    `forward-qwen3-1.7b-*.txt`: 12095). Top 5 here: 12095, 264, 1112, 32671, 7407 at 22.75, 20.125, 19.875, 19.75,
    19.75; v1's: 12095, 264, 1112, 7407, 279. The first three agree; the fourth and fifth are a tie at the 8-bit
    logit pixel's resolution. 34.4 s for 5 tokens × 28 layers + the head, including reading every layer from disk.
  - R1..R4 on layers 0..13 unchanged: the 64-token stream sha `c6c5f3e4…89c6` again, 1.74 s per token this time.
- **What the runtime is now:** every weight a seat, every activation a pixel, the context a hull of lookups under
  receipts, no division, no root, no float inside; the first token of the model's own answer reproduced from seats
  alone. Left: the tokenizer end to end (`intake/tokenizer.py` on `models/qwen3-1.7b`), generation past one token,
  and the archive's deletion (`geo.rs`, `sed.rs` still read it).
