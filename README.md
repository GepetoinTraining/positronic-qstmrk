# positronic-qstmrk — Node Zero / the Atlas Engine

Private research repository. **No license yet**: all rights reserved by the authors until one is chosen.

Pedro García (pgarcia) with Claude (Fable 5.1, Opus 5).

Two builds live here, both on Qwen3-1.7B:

1. **v1: the Atlas Engine** (`atlas/`, `intake/`, `receipts/`, `LOG.md`). This is a float-free, all-integer inference engine. It runs the model from its own cells, and every stage either rewinds exactly or refuses with a receipt. The forward pass is bit-identical across the CPU and GPU kernels (RTX 3060 Ti via wgpu).
2. **v2: the construction** (`v2-retry/`). This is a stepped rebuild from the floor: numbers are built, not imported. It holds the manifold-matrix walks, a viewer by dimension, and the geometric reading of the weights: pointers, the cube byte type, residuals and handles, and the cycler.

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

## Not in git, and how to get it back

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

## Where v2 stands (2026-09-15)

- Every bf16 weight is `±odd / 2^e`. The 4,505 distinct values are exactly 4,505 (odd, e) pairs.
- e is the **cube byte type**: 27 cells on/off plus the side, centre 20 (derived, not chosen).
- The odds minify to **63 residuals** (2⁶ with the empty centre) and **11 minifiers**, all ≤ 27 = 3³. 243 = 3⁵ is the one residual no root takes.
- Handle (minifier, e) × residual: **433 × 63 → 4,505**, every pair unique.
- **The cycler** pairs neighbours along the 2-power axis. With the handle off, cycle 1 is the complete square 126² model-wide.
- Against shuffled controls, the arrangement carries structure at `embed_tokens` columns and at `up × downᵀ`. The `up`/`down` plane relation turns across depth: the same plane at layer 0, neutral at 13, the opposite plane at 27.

Full detail and every number: `v2-retry/BUILDER_LOG.md`.
