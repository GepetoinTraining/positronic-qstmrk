# positronic-qstmrk — Node Zero / the Atlas Engine

Private research repository. **No license yet**: all rights reserved by the authors until one is chosen.

Pedro García (pgarcia) with Claude (Fable 5.1, Opus 5).

Three builds live here, all on Qwen3-1.7B:

1. **v1: the Atlas Engine** (`atlas/`, `intake/`, `receipts/`, `LOG.md`). This is a float-free, all-integer inference engine. It runs the model from its own cells, and every stage either rewinds exactly or refuses with a receipt. The forward pass is bit-identical across the CPU and GPU kernels (RTX 3060 Ti via wgpu).
2. **v2: the construction** (`v2-retry/`). This is a stepped rebuild from the floor: numbers are built, not imported. It holds the manifold-matrix walks, a viewer by dimension, and the geometric reading of the weights: pointers, the cube byte type, residuals and handles, and the cycler.
3. **v3: the kernel** (`v3-based/`). This is the CubeKernel composed in Rust: binary → tryte → quadryte → Hurwitz seam → Octavian Catch → 9-gon → 7//11 hinge, exact, with receipts that rewind. It is driven by the onset oscillator and holds every BF16 pattern in the 41 table in 3,924 bytes. It also carries the attention registrations T10 and T11, run against v1's sealed captures and seal-verified tensors. Next is `v3-based/NEXT-STAGE.md`: the kernel as one running object, with no weights.

Every measured claim carries **[M]** (measured), **[F]** (floatland reference only) or **[B]** (a choice of the builder's). Sealed receipts decide, not the session.

---

## Layout

| path | what |
|---|---|
| `ATLAS_ENGINE_SPEC.md` | the v1 build order ("Node Zero") |
| `GOAL.md` | the v1 goal: the ball as a function, `BALL(a,k) = Rot(q)^k · R(a)` |
| `LOG.md` | the v1 build record, §1–§33 — read this before touching layout or kernels |
| `native_object.pdf` | the hypothesis paper |
| `intake/` | Python: model intake into cells (Morton grids, seal), the integer-only tokenizer, the chat harness, ringing instruments |
| `atlas/` | Rust crate: cells, surfaces, matmul (exact bucket), fold (exact radix), fold cube, fiber kernel, clocks, card kernels (wgpu/WGSL), gates |
| `receipts/` | v1 sealed receipts: reference hashes, forward and chat runs, captures |
| `v2-retry/BUILDER_LOG.md` | the v2 build record, including every miss |
| `v2-retry/docs/` | the posit and protocol (`posit/`), the two-registers papers (`alignment/`) |
| `v2-retry/oscillator/` | Rust: the v2 first job, the oscillator cycles |
| `v2-retry/mm/` | TypeScript: the manifold matrix. Walks into SQLite, the React viewer, and the tables pipeline for the weights |
| `v2-retry/mm/receipts/` | v2 sealed receipts: cycles, namings, hyper limits, table scans |
| `v2-retry/tables/` | v2 model-wide tables: LUTs, cores, planes, odds, roots, measurements (per-table dumps are not in git) |
| `v3-based/SESSION-2026-09-24-code.md` | the v3 session record: what was built, run, measured and corrected |
| `v3-based/NEXT-STAGE.md` | the next stage: the kernel as one running object, with no weights |
| `v3-based/SYNTHESIS-v1-v2-v3.md` | v1 and v2 read against v3, cross-checked |
| `v3-based/prereg_T*.txt` | v3 registrations (T9 from the web surface; T10, T11 here), sealed before running, outcomes appended |
| `v3-based/walk/` | Rust: the census, Tier B, the composed kernel, the BF16 gate, the footprint, T10, T11 |
| `v3-based/t10/fetch.py` | byte-range fetch of single tensors, admitted only against v1's sealed `source_sha256` |
| `v3-based/bundle/`, `day_v2.pdf` | the web surface's session bundle and day record |

## Not in git, and how to get it back

v3's fetched tensors (`v3-based/t10/*.u16`, 1.42 GB) are not in git. From `v3-based/t10`, `python fetch.py <tensor names>` re-fetches them by byte range and admits each only against v1's seal. `t11_names.txt` lists T11's 154.


| what | size | how |
|---|---|---|
| `models/qwen3-1.7b/` | 3.8 GB | download `Qwen/Qwen3-1.7B` from the Hugging Face Hub into this folder (safetensors shards, `config.json`, tokenizer files) |
| `cells/` | ~16 GB | the v1 intake over the model (`intake/intake.py`; see `LOG.md` §1–§2); the fold cube is rebuilt by `atlas_cub` (§12) |
| `v2-retry/mm/data/mm.sqlite` | ~85 MB | `npm run walk` in `v2-retry/mm` (cycles 1–13, graceful failure, receipts are in git) |
| `v2-retry/tables/**/model/`, `v2-retry/tables/qwen3-1.7b/` | ~140 MB for three sample tables | the `v2-retry/mm/src/tables-*.ts` scripts |
| `target/`, `node_modules/` | — | `cargo build --release`, `npm install` |

**External dependency.** `atlas/Cargo.toml` depends on V5 by path: `../../folded-weights/v5/rust`. A second machine needs that crate checked out at the same relative place, or the path changed.

## Setup on another machine

Requirements: Rust 1.93+, Node 22.15+ (TypeScript runs with `--experimental-strip-types`), Python 3.11 with `numpy`, `regex`, `safetensors`, `torch` (intake only; `transformers` is only in the retired anchor script), and a Vulkan GPU for the card kernels.

```bash
git clone https://github.com/GepetoinTraining/positronic-qstmrk.git
```

Then put the weights in `models/qwen3-1.7b/`, and:

```bash
cd v2-retry/mm && npm install
```

```bash
cd atlas && cargo build --release
```

## Running

**v2 construction** (`v2-retry/mm`):

```bash
npm run walk
```

```bash
npm run dev
```

The viewer is at http://localhost:5173, with tabs 2D–9D and the 9-gon. The tables pipeline is one script per step, run from `v2-retry/mm`, e.g.:

```bash
node --experimental-strip-types src/tables-measure.ts
```

In order: `tables-scan`, `tables-pointers`, `tables-cores`, `tables-planes`, `tables-minify` (the cube byte type), `tables-odds`, `tables-roots`, `tables-243`, `tables-weigh`, `tables-bigram` (the cycler), `tables-bigram-model`, `tables-measure`, `tables-cycle-updown`, `tables-osc-rot`. Each script's header comment says what it writes and what it checks.

**v1 engine** (`atlas/`): binaries `atlas_forward`, `atlas_turn`, `atlas_serve`, `atlas_capture`, `atlas_cub`, `atlas_ball`, `atlas_maelstrom`. Kernels are selected with `ATLAS_KERNEL` = `bucket` (default), `radix`, `cub`, `fiber`, `card`, `card3`. Chat through the harness:

```bash
python intake/chat.py --kernel bucket
```

The references are the engine's own receipts (`receipts/reference-qwen3-1.7b.txt`), never a float framework.

## House rules (v2)

- No arithmetic in the construction; every step earned; flag rather than fill.
- The bit has two readings (1/2 relation, 2 turn); 0 only marks a cycle; n/n is n.
- Declare, then determine; never enumerate the infinite.
- Floatland agreement is not validation.
- Structure, not tables: nothing is packed back flat.
- Show attempts, not questions; label [M]/[F]/[B]; seal receipts.

## Where v3 stands (2026-09-24)

- **The kernel runs, composed and exact.** 22/22 layer checks pass. 10⁶ random operations run with 0 invariant violations and rewind bit-identically from receipts. Without receipts, 1000/1000 programs drift.
- **The no-go is reproduced exhaustively.** No 2T acts on the lanes with −1 as the flip, over all 3,072 lane symmetries. The twist-compatible Hurwitz copy lies in one twist triple, not in the hub.
- **BF16 through the intake gate: 65,280** (signed zero kept). The only duplicate value is −0. The 41 walk's halves multiply like signs, and its index is a discrete log: adding indices is Mitchell's multiplication.
- **T11 (pre-registered, every prediction held).** Three heads leave the sink for the induction target on a repeated prompt, and 96% of the released mass lands on that one position. The instrument first reproduced v1's sealed captures (cosine 0.99987 at layer 14).
- **Next:** `v3-based/NEXT-STAGE.md`. The rule is that there will be no weights: load once, check, seat every value at the position it inhabits.

## Where v2 stands (2026-09-15)

- Every bf16 weight is `±odd / 2^e`. The 4,505 distinct values are exactly 4,505 (odd, e) pairs.
- e is the **cube byte type**: 27 cells on/off plus the side, centre 20 (derived, not chosen).
- The odds minify to **63 residuals** (2⁶ with the empty centre) and **11 minifiers**, all ≤ 27 = 3³. 243 = 3⁵ is the one residual no root takes.
- Handle (minifier, e) × residual: **433 × 63 → 4,505**, every pair unique.
- **The cycler** pairs neighbours along the 2-power axis. With the handle off, cycle 1 is the complete square 126² model-wide.
- Against shuffled controls, the arrangement carries structure at `embed_tokens` columns and at `up × downᵀ`. The `up`/`down` plane relation turns across depth: the same plane at layer 0, neutral at 13, the opposite plane at 27.

Full detail and every number: `v2-retry/BUILDER_LOG.md`.
