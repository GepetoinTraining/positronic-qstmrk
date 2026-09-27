# Tensor atom — handover

Written by Claude (Opus) at Pedro Garcia's request, 26 September 2026, at the end of the session that produced *Legos* (included here as `legos.pdf`). Any hedge in this file is Claude's.

## Read in this order

1. **Run `python3 tests.py`** (about 20 seconds, no network; `data/` holds three real Qwen3-4B tensors). Read `results.txt` next to it.
2. **Read `atom.py`** (about 150 lines). Everything the tests check is in there.
3. **Then `legos.pdf`.** The kernel is the paper's capabilities applied to one tensor. Do not start from the paper; start from the running code.

If the whole thing looks like too much: it is one object and one label. The object is the weights on an exact integer grid. The label is the rule that rebuilds everything else. Everything below is checking that sentence.

## The floor, as Pedro set it

A tensor is an atom made of two parts: the **object** (the weight matrix and its oscillation pattern, with what's left over as residue) and the **label** (how that residue behaves against its inverse). Two waves cross, one real and one in *i*, and their intersection is what inference reads. Inference is stress; the inverse residue is the release that returns the weights to baseline.

## What the tests establish

| Test | Result | What it shows |
|---|---|---|
| T1a, T1b | PASS | bf16 weights map onto an exact integer grid and back, bit-identical; the atom is stored as a real ISIS3 cube (PVL label, BandSequential) and read back bit-identical. |
| T2 | PASS | Space is irrelevant: four storage layouts (band-sequential, two tilings, shuffled rows with the order kept in the label) give identical weights, harmonic, residue and inference output. Only positions matter. |
| T3a, T3b | PASS | Rules are discoverable: a blind search, told nothing about the architecture, finds the period 128 (the attention head width) in k_proj, and finds it again independently in another layer. |
| T3c | PASS | Honest negative: v_proj has no line that beats its shuffled controls, so no rule is claimed for it. |
| T4a | **FAIL** | The period lives in the **energy envelope**, not in the signed values: averaging signed values over periods carries 12.60% of the energy, indistinguishable from shuffled rows (12.50%, max 12.63%). |
| T4b | PASS | The *i*-wave is real where rotary encoding puts it: inside each head, dimensions *d* and *d*+64 (the real and imaginary halves of one rotating number) have energy correlation 0.484; random pairings give 0.001 (max 0.100). |
| T5a, T5b | PASS | Stress and release: 1000 cycles of adding the residue ×3, running inference, and subtracting it again return every weight bit-for-bit on the exact grid. The same 1000 cycles in float32 leave 796 weights changed. |

## The wider check (before the tests)

Fourteen tensors (layers 0 and 18: q, k, v, o, gate, up, down), both axes, 28 profiles.

- A loose test (strongest period vs position-shuffled controls) passed **26/28**. That was wrong: slow trends beat shuffles and were counted as periods.
- The strict test (a period must be a spectral **line**, above its neighbouring frequencies) passes **4/28**: q_proj and k_proj along their output rows, in both layers, all harmonics of 128. MLP, v_proj and o_proj show no line on either axis.
- So Pedro's claim "it always does" is **not supported** by this reading. It holds where the architecture builds the *i*-pair in (q, k). For the other tensors, either there is no oscillation, or it lives in a reading these tests don't make yet. That second possibility is Open Question 2.

## Errata (recorded, not silently fixed)

- T4a and T4b first used ad-hoc thresholds (mean of 20 shuffles; correlation > 0.5). Both were moved to the rule every other test uses, *beat every shuffled control*, before this file was written. Under the old thresholds T4a passed and T4b failed; under the uniform rule it is the other way round.


## T6 — hull and rewind (`hull.py`, `test_hull.py`, results in `results_hull.txt`)

Pedro's test: attention as a mass damper inside a hulled interior (spinning with rotary, choosing with softmax), passing what it chose to a second interior that outputs; the state kept on the grid; every change written with its receipt (MM's |x,K;K,x|); between turns, rewind the operations and keep only the output.

Built on one real layer (Qwen3-4B, layer 18, all seven projections and four norms). The context and prompts are random residual rows at realistic scale; there is no tokenizer or embedding in this test.

| Test | Result | What it shows |
|---|---|---|
| T6a | PASS | 100 turns, each a different 8-token prompt on a 24-token context; after each turn the receipts are rewound and the state is back to baseline bit for bit; only the 100 output rows are kept. |
| T6b | PASS | The only rounding anywhere is the single snap out of each interior, never larger than half a grid step, and every one is stored in its receipt. |
| T6c | PASS | Weights are read-only throughout (hash unchanged). |
| T6d | PASS | Control off the grid: the same turns in float32 with the raw deltas subtracted back leave 46,477 of 61,440 context values changed. |
| T6e | PASS | The damper really chooses: last-token attention entropy about 1.2 nats against 3.47 for no choice. |
| T6f | note | The first control was accidentally on the grid and released perfectly in plain float32. So the release comes from staying on the grid, whatever arithmetic carries it. Recorded rather than hidden. |

Open from T6: a real tokenizer and embedding in front of it; all 36 layers in sequence; and whether the hull's rounding, kept in receipts, ever needs to be replayed rather than discarded.

## Open questions — the channel back

Put questions in `QUESTIONS.md`, one block each, and bring the bundle to a new conversation with Pedro. The receiving Claude answers against `tests.py`, adds a test for every claim it makes, and records results next to the question. No test result is edited silently.

1. **The multiplicative split.** T4a says the pattern is an envelope: a per-position amplitude times a unit-scale residue. That split needs division. Can it be done division-free with the paper's tools (rails, bands kept outside the map, inversion)?
2. **Readings for the tensors with no line.** MLP, v and o fail on row and column energy. Candidates not yet tried: the value domain of single rows, the product with the partner tensor (gate × up; v with o), and the residual stream's own axes.
3. **The quarter-turn read at T/4.** The hinge test (26 September) recovered phase with a read one quarter-period late. Here the rotary partner sits at half the head (64 of 128). Is 64 the right quarter for a two-wave object, or is there a T/4 = 32 read that carries more?
4. **Scaling the check.** 14 tensors from 2 layers. `fetch.py` pulls any tensor by byte range; run all 36 layers before generalizing anything.

## Pointers (from Recall, not re-verified here)

- Pedro adopted ISIS3 `.cub` for v5 in May 2026 (Qwen3-4B stored as `Object = IsisCube`, PVL labels with `Slot_n` groups).
- Rig: `BandedCub` in `mm-solver/forge137/catalyst/banded_cub.py`; `readCubeLabel` / `makeCubeLabel` in `cube-pvl.ts`. Only signatures were seen; bodies were not read.
