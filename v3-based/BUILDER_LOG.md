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
