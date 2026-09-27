# Questions back to Claude

One block per question. Keep every block; answers go under the question, never replace it.

## Q-template
- **Asked by / date:**
- **Test it concerns:** (T1a ... T5b, or "new")
- **What was run:**
- **What was expected:**
- **What happened:**
- **Answer (Claude, with the test that backs it):**

All four blocks below come from T14 (`v3-based/weights/t14.py`, results in `v3-based/weights/results_T14.txt`), registered
before any code in `v3-based/prereg_T14.txt` (sha256 073178c4...). Every reading in T14 is exact: integer profiles,
integer period excess, and verdicts that are counts of shuffled controls beaten. No float reaches a verdict. T3 and T4
above were float readings (FFT, ratios, correlation), so their verdicts stand only where T14 confirms them.

The seat, under all four answers: every weight is a position, value = address x 2^band, where the address is the signed
odd part of the BF16 significand and the band is its power of two. All 103,552,256 weights of the bundle decompose and
rebuild exactly (G1). They are stored as ISIS3 cubes with two bands, Address and TwoBand, and read back sha-identical
(G2). All 7 layer-18 projections computed from the cubes equal an independent exact reference at every output cell (G3).
One real layer run from the cubes alone, with np.load refusing the weight files, matches the run from raw weights bit
for bit (G4). The seated run took 19s against 36s for the reference.

## Q1. The multiplicative split
- **Asked by / date:** handover, 26 Sep 2026
- **Test it concerns:** T4a
- **What was run:** T14 P1. Row profiles of L18 q_proj, L18 k_proj and L0 k_proj were read three ways: band profile
  (sum of 4^band), address profile (sum of odd^2) and full profile. The period-128 excess of each was compared against
  100 position shuffles.
- **What was expected:** a line in the band profile and none in the address profile, in all three tensors.
- **What happened:** the band profile beat 100/100 shuffles in all three. The address profile beat 49, 37 and 31, which is
  chance. PASS as registered.
- **Answer (Claude, with the test that backs it):** the split needs no division. BF16 already stores the weight as the
  product of an envelope (2^band) and a unit-scale residue (the odd address). The period lives entirely in the band; the
  address carries none of it. So the object's oscillation is in the bands and the residue is in the addresses.
  Backed by T14 P1.

## Q2. Readings for the tensors with no line
- **Asked by / date:** handover, 26 Sep 2026
- **Test it concerns:** the 4/28 strict test; T3c
- **What was run:** T14 P3, exploratory with no prediction. All 8 tensors, both axes, every power-of-two period that
  tiles, band, address and full profiles.
- **What was expected:** nothing registered.
- **What happened:** v, o, gate, up and down show no line on either axis in any of the three profiles, at any power-of-two
  period. The address profile shows no line anywhere, in any tensor. The lines that do appear are in the rows of q_proj
  and k_proj. One outlier: L0 k_proj columns (the residual-stream axis) show a band line at T = 256. P3 made about 480
  comparisons at 100/100, so a handful of chance hits is expected; this one is not claimed.
- **Answer (Claude, with the test that backs it):** in exact form the negative holds and is sharper than before. Wherever
  a period exists it is in the band, never in the address. For v, o and the MLP, no power-of-two period exists along
  either axis in bands, addresses or full energy. The candidates this does not rule out are non-power-of-two periods, the
  product with a partner tensor, and readings that are not row or column profiles. None of those was tried. Backed by
  T14 P3.

## Q3. The quarter-turn read
- **Asked by / date:** handover, 26 Sep 2026
- **Test it concerns:** T4b
- **What was run:** T14 P2. Inside each 128-row head, on the band profile, D_b = sum of (B_d - B_(d xor 2^b))^2 for
  b = 0..6 (smaller means more shared), plus 200 random pairings.
- **What was expected:** XOR 64 (the rotary pair) is the most shared of the 7, and beats all random pairings, in L18
  q_proj and L18 k_proj.
- **What happened:**
  - L18 q_proj: ranked [64, 4, 2, 1, 8, 16, 32]. XOR 64 is first and beats 200/200 random pairings.
  - L18 k_proj: ranked [2, 4, 64, 8, 1, 16, 32]. XOR 64 is third, though it still beats 200/200 random pairings.
  - L0 k_proj (reported only): ranked [4, 1, 16, 2, 8, 32, 64]. XOR 64 is last and beats 0/200.
  - FAIL as registered, because of L18 k_proj.
- **Answer (Claude, with the test that backs it):** 32 does not carry more. XOR 32, the quarter of the head, is the
  least or second-least shared pairing in every tensor. XOR 64 is the most shared only in L18 q_proj. In k_proj, the
  pairing depends on the layer. T4b's float claim (the i-pair in k_proj L18) survives only as "better than random", not
  as "the strongest pairing". Backed by T14 P2.

## Q4. Scaling the check
- **Asked by / date:** handover, 26 Sep 2026
- **What happened:** not attempted in T14. No network is needed: Qwen3-1.7B layers 0-13 are local as u16
  (`v3-based/ARCHIVE-2026-09-25-mistaken/weights/`, 159 tensors, 710,997,760 weights), with real captured states in
  `receipts/`. The next reading runs there. See `v3-based/BUILDER_LOG.md`.
