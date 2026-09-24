# Tier B — throughput, for Claude Code on the rig (Rust, --release)
Registered context: T9 (sha256 1d5c329a…7f5b). Tier A correctness already holds; this measures speed only.

## Build (no floats in the hot loop)
1. `walk_step(p)`: (a0,a1,a2) -> (a1, a2, (-a2 + a1 + 2*a0) mod p) on u32 lanes; and `walk_back(p)` with a precomputed 2^-1 mod p.
2. Cold tables, built before timing: for p in {19, 29, 41}: `idx -> state` and `state -> (half, idx)` (flat arrays, p^3 entries).
3. Octavian: 240x240 multiplication table of one closed set as u8 indices (57,600 bytes), built from t6_octonion.py's output.

## Measure (criterion or a plain Instant loop, 10 runs, report median and spread)
- B1 walk: 10^8 steps via recurrence vs via `idx+1` table lookup. Report ns/step.
- B2 pointerize: 10^8 random `state -> (half, idx)` lookups (p=41). ns/lookup.
- B3 rotary kernel: 10^8 left multiplications via the u8 table vs f64 octonion multiply; report ns/op and, for f64, max |norm - 1| after 10^8 chained ops (drift).
- B4 receipts: 10^8 gated half-hinges with refusal log vs without; ns/op and memory per 10^6 ops for the log.

## Report back
A table of ns/op and bytes, plus the f64 drift number. No claim beyond those numbers.
