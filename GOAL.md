# /goal — the ball

**Target:** `D:\positronic-qstmrk` (Qwen3-1.7B and 4B, integer-only; `LOG.md` is the build record; `atlas/` the crate).
**Goal in one line:** the spinning ball is a function, not a video. Write the function, run the model as one turn of it, prove the receipts match. The GIF at `run-003` becomes a projection of the function; it stops being the thing.

Corrections carried from the 2026-09-13 session (the goal as first written had these wrong):

| as written | as it is |
|---|---|
| "the 16 GB card" | the card is an RTX 3060 Ti, **8 GB**. The 1.7B fold is 7.8 GB in the load layout; run-length keys per (sign, odd) group are needed before it is resident. |
| "argmax + top-5 equal the recorded oracle" | the transformers oracle was **retired** (pgarcia, 03:30). The reference is the engine's own sealed receipt: `receipts/reference-qwen3-1.7b.txt` (seq-1, seq-8, France) and the 508-token hash `ed4d9482…`, reproduced by the exact radix kernel at 800 s. |
| "`forward_int`, `atlas_fiber_bench.rs` is the per-tick accumulate" | the production kernel is `atlas/src/fold.rs` (`matmul_fold`, `row_kernel`), run with `ATLAS_KERNEL=radix`; the benches are retired under `atlas/retired/` with their numbers in LOG §5–§6. |
| "i64 windows (D64 = 15, proven bit-identical in V5)" for the card | the window was proven on 8 tokens; **at 508 tokens two kernels with different floors diverged** (LOG §10). Both kernels are now exact in 256-bit lanes with one declared activation floor (2^-90). The card path must compute the same exact integer (4×u64 limbs; integer adds are order-free) or declare the same floor — a window is a second definition. |
| "`v5_1_fabric.rs` in the chat outputs" | not on this disk; not used by anything here. |

---

## The function

```
BALL(a, k) = Rot(q)^k · R(a)
```

- `R(a)` — the renderer: atom → cell on the torus cut open. `tensor_mesh.torus_params` (half-angle rationals `t = y/(ρ+x)`, `u = z/(σ+ρ)`, roots declared at `digits`, never floats), `cell_of(t,u,W,H)`. One tensor `T` at max resolution; every zoom is `at(level)`, an integer pooling by 2^level.
- `O(cell)` — the observer, the inverse renderer. `R(a) ∈ C ⇔ a ∈ O(C)` for every atom and cell (`MeshTensor.tight`). This is the check, run after, cold.
- `Rot(q)` — the tick. Exact rational rotation `M(q)/|q|²` (`reconstruct.rotation_matrix`, `joints.joint_motion`), `q` one of the six Hurwitz quaternions in `cubball.rs::SPIN_Q` with `|q|² ∈ {2,3,5,7,11,13}`. Exactly orthogonal, no float, no LUT.
- `k` — the tick count. One turn of the ball = the atlas walked once = one forward pass. `run-003` is `k = 253` frames at yaw `91/64°` = 359.7°; that was the video's shadow of the function.

What's cold: `R`, `O`, the atoms, the frames, the places. What's hot: `k`, and the buckets in flight. The dictionaries are never on the tick (measured 2026-09-13: alphabet in the kernel = 0 B; LOG §7 table).

## Where everything is

| what | path |
|---|---|
| the ball's seals, six quaternions, `spin_glyph` (91/64° = 7·13/2⁶) | `D:\shiat\carillon\centipede\src\cubball.rs` |
| tile order, fold to byte, integer Fourier (998244353, len 512) | `D:\shiat\carillon\centipede\src\lightfiber.rs` |
| run-003: `spin-glyph.cmd`, `CUB3.bin` 94.5 MB, `BALL4.bin` 2.06 MB, the GIF, the mkv | `D:\shiat\carillon\centipede\out\cubball\run-003\` |
| the board: 137 gear, Ball4 centre memory, four-kernels mesh | `D:\shiat\carillon\centipede\src\pins\` |
| `R`/`O` tight pair, torus params, integer pooling | `D:\cad-systemup\core\tensor_mesh.py` |
| exact rotation, spanning forest, part motions | `D:\cad-systemup\core\joints.py`, `core\reconstruct.py` |
| atom/frame, align, strip_frame, refusal | `D:\cad-systemup\core\tryte.py` |
| hull + residual, the seal | `D:\cad-systemup\core\defloat.py` |
| primorial NTT, Good–Thomas | `D:\cad-systemup\core\ntt.py` |
| bijective LUTs (RoPE, exp, rsqrt), V5 primitives, V5 anchors | `D:\folded-weights\v5\rust\src\`, `docs\HANDOFF.md` |
| **this build:** intake (cells, Morton grids, seal), crate (cells, surfaces, matmul = exact bucket, fold = exact radix, forward, gate), receipts, references | `D:\positronic-qstmrk\intake\`, `atlas\`, `receipts\` |
| the papers | `native_object.pdf` (here); `cells.pdf`, `the-unit-from-the-left.pdf`, `ironing-lifted-note.pdf`, `CAPACITY-FLOOR.pdf`, `smikak-1.pdf`, `sisters.pdf` (Downloads) |
| the inverse — where things are laid down and kept | read `LOG.md` §2, §7, §8, §10 before touching layout |

## What to build, in order

1. **`ball.rs` — the function.** `R` and `O` over the model's cells (place = Morton tile schedule; atom = odd u8; frame = rung), `Rot(q)` from `SPIN_Q`, `tick(k)`. Gate: `tight()` over every cell of one tensor; `Rᵀ R = I` exactly as rationals; `Rot^|q|²`-periodicity checked.
2. **One turn = one forward.** `fold.rs` is the per-tick accumulate; the gather is the observer. Gate: the reference receipts hold; the 508 hash `ed4d9482` holds under every kernel (radix 801 s, exact bucket 1,479 s: both ed4d9482 — DONE 2026-09-13).
3. **Resident.** The fold cube (`cells/<model>/fold.cub`, 5,058 MB for the 1.7B, rung bands, LOG §12) on the 8 GB card, exact (limbed integers). The band walk is the kernel: one doubling per band per (line, lane), the odd chain at the gather. Gate: same receipts on the card.
4. **Project.** Render the ball's own turn from `ball.rs` — k ticks of a seal q, the frame scale |q|^{2k} carried, the level-0 cell table hashed per tick — and seal those frames as the function's receipt. The GIF at `run-003` is NOT the target: its 91/64° yaw is the carrier's (ffmpeg v360, float) exchange rate, not a tick of the function, and the odd-prime ticks never close (LOG §11), so the video cannot be reproduced by the function and the function must not be judged by the video. The GIF keeps its seam (17/64°) as its own receipt; the function's frames get theirs.

## Rules carried

- No float anywhere in a hashed byte. Every op is a shift, a flip, the bar, an integer add, or a proven-bijective LUT; anything else is refused with a receipt.
- Never import v4.3/v4.4/monster-group-fourier code; read their data.
- Correct first. A slower bit-identical kernel goes in the log next to June's numbers; a faster wrong one does not exist.
- One floor, declared once, identical in every kernel; above it, exact. Two floors are two definitions (LOG §10; `sisters.pdf`).
- Sign is beside the cell; the ∓ sheets are the Mersenne/Fermat shifts and are unused so far.
- The receipt decides, not the session.


Step 3 DONE 2026-09-13 ~22:30 (LOG §27): the placement resident on the RTX 3060 Ti via wgpu, exact limbed lane, gate 5 and the 508 bit-identical, 508 in 72 s.
