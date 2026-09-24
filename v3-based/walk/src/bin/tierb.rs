// Tier B (T9): throughput only. Spec: ../../TIER_B_for_code.md.
// No floats in the hot loops except the f64 octonion comparison in B3.

use std::hint::black_box;
use std::time::Instant;
use walk::{closed_table, fano, omul_i, steiner_systems, unit_set, FANO};

const N: u64 = 100_000_000;
const RUNS: usize = 10;
const WALK_PRIMES: [u32; 3] = [19, 29, 41];

// ---------- rng (splitmix64; no crates)
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---------- timing: 10 runs, median and spread in ns/op
fn bench(name: &str, ops: u64, mut f: impl FnMut() -> u64) -> f64 {
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_nanos() as f64 / ops as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = (t[RUNS / 2 - 1] + t[RUNS / 2]) / 2.0;
    println!("{name:<44} median {med:>7.3} ns/op   spread {:.3}..{:.3}", t[0], t[RUNS - 1]);
    med
}

// ---------- 1. walk_step / walk_back on u32 lanes
#[inline(always)]
fn walk_step(p: u32, (a0, a1, a2): (u32, u32, u32)) -> (u32, u32, u32) {
    (a1, a2, (a1 + 2 * a0 + p - a2) % p)
}

#[inline(always)]
fn walk_back(p: u32, inv2: u32, (a1, a2, a3): (u32, u32, u32)) -> (u32, u32, u32) {
    // 2*a0 = a3 + a2 - a1
    (((a3 + a2 + p - a1) % p) * inv2 % p, a1, a2)
}

fn pack(p: u32, (a, b, c): (u32, u32, u32)) -> usize {
    ((a * p + b) * p + c) as usize
}

// ---------- 2. cold tables: idx -> state, state -> (half, idx)
struct WalkTables {
    p: u32,
    idx_to_state: Vec<(u32, u32, u32)>, // halves laid out back to back
    base: Vec<u32>,                     // start of each half in idx_to_state
    state_to_hi: Vec<u32>,              // flat p^3; (half << 31) | idx_in_half; zero state = u32::MAX
}

fn build_tables(p: u32) -> WalkTables {
    let n = (p * p * p) as usize;
    let mut state_to_hi = vec![u32::MAX; n];
    let mut idx_to_state = Vec::with_capacity(n - 1);
    let mut base = Vec::new();
    for start in 1..n as u32 {
        if state_to_hi[start as usize] != u32::MAX {
            continue;
        }
        let half = base.len() as u32;
        base.push(idx_to_state.len() as u32);
        let s0 = (start / (p * p), start / p % p, start % p);
        let mut s = s0;
        let mut i = 0u32;
        loop {
            state_to_hi[pack(p, s)] = (half << 31) | i;
            idx_to_state.push(s);
            i += 1;
            s = walk_step(p, s);
            if s == s0 {
                break;
            }
        }
    }
    assert!(base.len() <= 2, "p={p}: more than two orbits");
    WalkTables { p, idx_to_state, base, state_to_hi }
}

#[inline(always)]
fn omul_f(u: &[f64; 8], v: &[f64; 8]) -> [f64; 8] {
    let mut w = [0f64; 8];
    for i in 0..8 {
        for j in 0..8 {
            let t = u[i] * v[j];
            if FANO.1[i][j] > 0 { w[FANO.0[i][j]] += t } else { w[FANO.0[i][j]] -= t }
        }
    }
    w
}

fn main() {
    let mut rng = Rng(2026);

    // ---------------- build (cold, untimed)
    let tables: Vec<WalkTables> = WALK_PRIMES.iter().map(|&p| build_tables(p)).collect();
    for t in &tables {
        let p = t.p;
        let inv2 = (p + 1) / 2;
        let n = (p * p * p) as usize;
        let mut ok = true;
        for i in 1..n as u32 {
            let s = (i / (p * p), i / p % p, i % p);
            ok &= walk_back(p, inv2, walk_step(p, s)) == s;
        }
        let sizes: Vec<u32> = (0..t.base.len())
            .map(|h| t.base.get(h + 1).copied().unwrap_or(t.idx_to_state.len() as u32) - t.base[h])
            .collect();
        println!("build p={p}: halves {sizes:?}, walk_back inverts walk_step on all states: {ok}");
    }

    let (fidx, fsgn) = fano();
    let systems = steiner_systems();
    let closed: Vec<Vec<u8>> = systems
        .iter()
        .filter_map(|s| closed_table(&fidx, &fsgn, &unit_set(s)))
        .collect();
    println!("build octonions: {} labelings of S(3,4,8), {} closed", systems.len(), closed.len());
    assert_eq!((systems.len(), closed.len()), (30, 7), "does not reproduce T6 counts; need t6_octonion.py");
    // B3's closed set is t6_octonion.py's closed[0], exported by export_t6.py.
    let raw_u = std::fs::read("t6_units.bin").expect("run export_t6.py first");
    let otab = std::fs::read("t6_table.bin").expect("run export_t6.py first");
    let units_i: Vec<[i32; 8]> = raw_u.chunks(8).map(|c| std::array::from_fn(|k| c[k] as i8 as i32)).collect();
    assert_eq!((units_i.len(), otab.len()), (240, 57_600));
    let agrees = (0..240).all(|a| {
        (0..240).all(|b| omul_i(&fidx, &fsgn, &units_i[a], &units_i[b]).map(|x| x / 2) == units_i[otab[a * 240 + b] as usize])
    });
    let mut t6_set = units_i.clone();
    t6_set.sort();
    let same_as_mine = systems.iter().filter(|s| closed_table(&fidx, &fsgn, &unit_set(s)).is_some()).any(|s| {
        let mut m = unit_set(s);
        m.sort();
        m == t6_set
    });
    println!("build octonions: t6 closed[0] loaded; Rust integer multiply agrees with its table: {agrees}; \
              it is one of Rust's 7 closed sets: {same_as_mine}");
    assert!(agrees && same_as_mine);
    let latin = (0..240).all(|a| {
        let mut r = [false; 240];
        let mut c = [false; 240];
        for b in 0..240 {
            r[otab[a * 240 + b] as usize] = true;
            c[otab[b * 240 + a] as usize] = true;
        }
        r.iter().all(|&x| x) && c.iter().all(|&x| x)
    });
    println!("build octonions: table {} bytes, Latin square: {latin}", otab.len());
    let units_f: Vec<[f64; 8]> = units_i.iter().map(|u| u.map(|x| x as f64 / 2.0)).collect();

    // random operand buffers, shared by paired variants (2^16 entries, cycled)
    const BUF: usize = 1 << 16;
    let u_seq: Vec<u8> = (0..BUF).map(|_| rng.below(240) as u8).collect();
    let bits: Vec<u64> = (0..BUF).map(|_| rng.next()).collect();
    println!();

    // ---------------- B1 walk: recurrence vs idx+1 lookup
    for t in &tables {
        let p = black_box(t.p);
        bench(&format!("B1 p={} recurrence", t.p), N, || {
            let mut s = (1u32, 0, 0);
            for _ in 0..N {
                s = walk_step(p, s);
            }
            s.2 as u64
        });
        let len = t.idx_to_state.len() as u32 - t.base.last().copied().unwrap_or(0); // last half
        let (lo, hi) = (t.base.last().copied().unwrap(), t.base.last().copied().unwrap() + len);
        bench(&format!("B1 p={} idx+1 table lookup", t.p), N, || {
            let mut i = lo;
            let mut acc = 0u32;
            for _ in 0..N {
                i += 1;
                if i == hi {
                    i = lo;
                }
                acc ^= t.idx_to_state[i as usize].2;
            }
            acc as u64
        });
    }

    // ---------------- B2 pointerize p=41
    let t41 = &tables[2];
    let rand_states: Vec<(u32, u32, u32)> = (0..BUF)
        .map(|_| t41.idx_to_state[rng.below(t41.idx_to_state.len() as u64) as usize])
        .collect();
    bench("B2 p=41 state -> (half, idx)", N, || {
        let p = black_box(41u32);
        let mut acc = 0u32;
        for k in 0..N as usize {
            acc ^= t41.state_to_hi[pack(p, rand_states[k & (BUF - 1)])];
        }
        acc as u64
    });

    // ---------------- B3 rotary kernel: u8 table vs f64
    bench("B3 left mult, u8 table", N, || {
        let mut x = 0u8;
        for k in 0..N as usize {
            x = otab[u_seq[k & (BUF - 1)] as usize * 240 + x as usize];
        }
        x as u64
    });
    let one = [1.0, 0., 0., 0., 0., 0., 0., 0.];
    bench("B3 left mult, f64 octonion", N, || {
        let mut x = one;
        for k in 0..N as usize {
            x = omul_f(&units_f[u_seq[k & (BUF - 1)] as usize], &x);
        }
        x[0].to_bits()
    });
    let mut x = one;
    let mut drift = 0f64;
    for k in 0..N as usize {
        x = omul_f(&units_f[u_seq[k & (BUF - 1)] as usize], &x);
        let n = x.iter().map(|c| c * c).sum::<f64>().sqrt();
        drift = drift.max((n - 1.0).abs());
    }
    println!("B3 f64 drift: max |norm - 1| over 1e8 chained ops = {drift:e} (final {:e})",
        (x.iter().map(|c| c * c).sum::<f64>().sqrt() - 1.0).abs());

    // ---------------- B4 receipts: gated half-hinges with / without refusal log
    // seats (v7, v11) in {0,1,2}^2; bit 1 = 7/11 (+1,-1), bit 0 = 11/7 (-1,+1).
    // Log = one bit per op (applied or refused); the op sequence itself is the input.
    let mut log = vec![0u64; (N as usize).div_ceil(64)];
    bench("B4 gated half-hinge, with refusal log", N, || {
        let (mut v7, mut v11) = (1i32, 1i32);
        for k in 0..N as usize {
            let d = if bits[(k >> 6) & (BUF - 1)] >> (k & 63) & 1 == 1 { 1 } else { -1 };
            let (w7, w11) = (v7 + d, v11 - d);
            let ok = (0..=2).contains(&w7) && (0..=2).contains(&w11);
            if ok {
                v7 = w7;
                v11 = w11;
            }
            log[k >> 6] |= (ok as u64) << (k & 63);
        }
        (v7 * 3 + v11) as u64
    });
    bench("B4 gated half-hinge, without log", N, || {
        let (mut v7, mut v11) = (1i32, 1i32);
        for k in 0..N as usize {
            let d = if bits[(k >> 6) & (BUF - 1)] >> (k & 63) & 1 == 1 { 1 } else { -1 };
            let (w7, w11) = (v7 + d, v11 - d);
            if (0..=2).contains(&w7) && (0..=2).contains(&w11) {
                v7 = w7;
                v11 = w11;
            }
        }
        (v7 * 3 + v11) as u64
    });
    let applied: u64 = log.iter().map(|w| w.count_ones() as u64).sum();
    println!("B4 log: {} bytes per 1e6 ops (1 bit/op); refused {:.4} of ops",
        1_000_000 / 8, 1.0 - applied as f64 / N as f64);

    println!("\ntable bytes: otab {}, state_to_hi p=41 {}, idx_to_state p=41 {}",
        otab.len(),
        t41.state_to_hi.len() * 4,
        t41.idx_to_state.len() * 12);
}
