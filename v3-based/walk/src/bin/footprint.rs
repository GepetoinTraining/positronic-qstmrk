// Footprint of the kernel with all 65,536 BF16 patterns loaded into the 41 table.
// Loading rule [PG]: sign bit -> half of the 41 walk, low 15 bits -> index along that half.
// Two ways to hold it, each checked entry-by-entry against the other:
//   tables      : pattern -> state and state -> pattern stored flat, Octavian products stored 240x240
//   table-free  : pattern -> state by matrix power, state -> pattern by baby-step/giant-step,
//                 Octavian products computed from the Fano table and the 14 blocks of the set
// Integers only.

use std::collections::HashSet;
use std::time::Instant;
use walk::{omul_i, steiner_systems, unit_set, FANO};

const P: u32 = 41;
type St = [u32; 3];
type M3 = [[u32; 3]; 3];

// ---------- process working set (Windows); 0 elsewhere
#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct Pmc { cb: u32, faults: u32, peak_ws: usize, ws: usize, a: usize, b: usize, c: usize, d: usize, pf: usize, peak_pf: usize }
#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(h: isize, pmc: *mut Pmc, cb: u32) -> i32;
}
fn working_set() -> (usize, usize) {
    #[cfg(windows)]
    unsafe {
        let mut m = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut m, m.cb);
        (m.ws, m.peak_ws)
    }
    #[cfg(not(windows))]
    (0, 0)
}

// ---------- the walk as a matrix: v = (a0, a1, a2)^T, a(n+3) = -a(n+2) + a(n+1) + 2a(n)
const STEP: M3 = [[0, 1, 0], [0, 0, 1], [2, 1, P - 1]];
const BACK: M3 = [[P - 21, 21, 21], [1, 0, 0], [0, 1, 0]]; // 21 = 2^-1 mod 41; a0 = (a3 + a2 - a1)/2
fn mm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum::<u32>() % P))
}
fn mv(a: &M3, v: &St) -> St {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum::<u32>() % P)
}
fn mpow(a: &M3, mut e: u32) -> M3 {
    let mut r: M3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let mut b = *a;
    while e > 0 {
        if e & 1 == 1 { r = mm(&r, &b); }
        b = mm(&b, &b);
        e >>= 1;
    }
    r
}
fn flat(s: &St) -> usize {
    ((s[0] * P + s[1]) * P + s[2]) as usize
}

/// Table-free 41 layer: two start states, and a baby-step table per half for the inverse.
struct Free {
    start: [St; 2],
    len: u32,
    m: u32,
    giant: M3,                      // BACK^m
    baby: [Vec<(u32, u16)>; 2],     // (flat state, j) for j in 0..m, sorted by state
}
impl Free {
    fn new(start: [St; 2], len: u32) -> Self {
        let m = (1..).find(|m: &u32| m * m >= len).unwrap(); // integer ceil(sqrt(len))
        let baby = start.map(|s0| {
            let mut v = Vec::with_capacity(m as usize);
            let mut s = s0;
            for j in 0..m {
                v.push((flat(&s) as u32, j as u16));
                s = mv(&STEP, &s);
            }
            v.sort_unstable();
            v
        });
        Free { start, len, m, giant: mpow(&BACK, m), baby }
    }
    fn load(&self, pattern: u16) -> St {
        mv(&mpow(&STEP, (pattern & 0x7FFF) as u32), &self.start[(pattern >> 15) as usize])
    }
    /// state -> Some(pattern), or None for a spare state (index >= 32,768) or the zero state.
    fn read(&self, s: &St) -> Option<u16> {
        for h in 0..2 {
            let mut t = *s;
            for i in 0..=self.len / self.m {
                if let Ok(k) = self.baby[h].binary_search_by_key(&(flat(&t) as u32), |e| e.0) {
                    let idx = i * self.m + self.baby[h][k].1 as u32;
                    return if idx < 32_768 { Some(((h as u16) << 15) | idx as u16) } else { None };
                }
                t = mv(&self.giant, &t);
            }
        }
        None
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.baby.iter().map(|b| b.len() * std::mem::size_of::<(u32, u16)>()).sum::<usize>()
    }
}

// ---------- Octavian, table-free: a unit index decodes from the set's 14 blocks, products use the Fano table
fn unit_of(blocks: &[u8], i: usize) -> [i32; 8] {
    let mut x = [0; 8];
    if i < 16 {
        x[i / 2] = if i % 2 == 0 { 2 } else { -2 };
    } else {
        let (block, signs) = (blocks[(i - 16) / 16], (i - 16) % 16);
        for (k, c) in (0..8).filter(|c| block >> c & 1 == 1).enumerate() {
            x[c] = if signs >> k & 1 == 1 { -1 } else { 1 };
        }
    }
    x
}
fn index_of(blocks: &[u8], x: &[i32; 8]) -> Option<usize> {
    let support = (0..8).filter(|&c| x[c] != 0).fold(0u8, |m, c| m | 1 << c);
    if support.count_ones() == 1 {
        let c = support.trailing_zeros() as usize;
        return Some(2 * c + (x[c] < 0) as usize);
    }
    let b = blocks.iter().position(|&m| m == support)?;
    let signs = (0..8).filter(|c| support >> c & 1 == 1).enumerate().fold(0, |s, (k, c)| s | ((x[c] < 0) as usize) << k);
    Some(16 + 16 * b + signs)
}
fn omul_free(blocks: &[u8], a: usize, b: usize) -> Option<usize> {
    let w = omul_i(&FANO.0, &FANO.1, &unit_of(blocks, a), &unit_of(blocks, b));
    if w.iter().any(|c| c % 2 != 0) { return None; }
    index_of(blocks, &w.map(|c| c / 2))
}

fn kb(b: usize) -> String {
    format!("{:>9} B ({:.1} KiB)", b, b as f64 / 1024.0)
}

fn main() {
    let (ws0, _) = working_set();
    println!("Footprint: kernel with all 65,536 BF16 patterns loaded in the 41 table\n");

    // ---------------- table-free first (so the working set is measured before any big table exists)
    // starts of the two halves, in the same order as the census: first nonzero flat index, then first unvisited
    let s0: St = [0, 0, 1];
    let mut s = s0;
    let mut seen = vec![false; (P * P * P) as usize];
    let mut len = 0u32;
    loop { seen[flat(&s)] = true; len += 1; s = mv(&STEP, &s); if s == s0 { break; } }
    let f1 = (1..seen.len()).find(|&i| !seen[i]).unwrap() as u32;
    let s1: St = [f1 / (P * P), f1 / P % P, f1 % P];
    drop(seen);
    let free = Free::new([s0, s1], len);

    let systems = steiner_systems();
    let closed: Vec<Vec<u8>> = systems.into_iter().filter(|s| {
        let u = unit_set(s);
        let set: HashSet<[i32; 8]> = u.iter().copied().collect();
        u.iter().all(|a| u.iter().all(|b| {
            let w = omul_i(&FANO.0, &FANO.1, a, b);
            w.iter().all(|c| c % 2 == 0) && set.contains(&w.map(|c| c / 2))
        }))
    }).collect();
    let triple = &closed[1..4]; // the seam triple found by the kernel run: closed sets 1, 2, 3
    let fano_bytes = std::mem::size_of_val(&FANO); // as compiled (usize indices, i32 signs)
    let (ws_free, _) = working_set();

    // ---------------- tables
    let t = Instant::now();
    let mut fwd: Vec<[u8; 3]> = vec![[0; 3]; 65_536];
    let mut inv: Vec<u16> = vec![0; (P * P * P) as usize];
    let mut spare = vec![0u64; (P * P * P) as usize / 64 + 1]; // bit set = state holds no pattern
    for w in spare.iter_mut() { *w = u64::MAX; }
    for h in 0..2u16 {
        let mut s = [s0, s1][h as usize];
        for idx in 0..32_768u16 {
            let pat = (h << 15) | idx;
            fwd[pat as usize] = s.map(|c| c as u8);
            inv[flat(&s)] = pat;
            spare[flat(&s) / 64] &= !(1 << (flat(&s) % 64));
            s = mv(&STEP, &s);
        }
    }
    let oct_tables: Vec<Vec<u8>> = triple.iter().map(|blocks| {
        (0..240 * 240).map(|ab| omul_free(blocks, ab / 240, ab % 240).expect("closed") as u8).collect()
    }).collect();
    let build_ms = t.elapsed().as_millis();
    let (ws_tab, peak) = working_set();

    // ---------------- checks: table-free == tables, entry by entry
    let t = Instant::now();
    let fwd_ok = (0..=u16::MAX).all(|p| free.load(p).map(|c| c as u8) == fwd[p as usize]);
    let load_ns = t.elapsed().as_nanos() as f64 / 65_536.0;
    let t = Instant::now();
    let mut inv_ok = true;
    let mut spare_n = 0;
    for a in 0..P { for b in 0..P { for c in 0..P {
        let st = [a, b, c];
        let is_spare = spare[flat(&st) / 64] >> (flat(&st) % 64) & 1 == 1;
        match free.read(&st) {
            Some(p) => inv_ok &= !is_spare && inv[flat(&st)] == p,
            None => { inv_ok &= is_spare; spare_n += 1; }
        }
    }}}
    let read_ns = t.elapsed().as_nanos() as f64 / (P * P * P) as f64;
    let oct_ok = triple.iter().zip(&oct_tables).all(|(blocks, tab)| (0..240 * 240).all(|ab| omul_free(blocks, ab / 240, ab % 240) == Some(tab[ab] as usize)));
    let latin = oct_tables.iter().all(|t| (0..240).all(|a| (0..240).map(|b| t[a * 240 + b]).collect::<HashSet<_>>().len() == 240));

    println!("checks:");
    println!("  [{}] load: matrix power gives the same 41-state as the table for all 65,536 patterns ({load_ns:.0} ns/pattern)", if fwd_ok { "ok" } else { "FAIL" });
    println!("  [{}] read: baby-step/giant-step returns the same pattern as the table for all {} states; {} states hold no pattern (3,384 spare + the zero state) ({read_ns:.0} ns/state)",
        if inv_ok { "ok" } else { "FAIL" }, P * P * P, spare_n);
    println!("  [{}] Octavian: products computed from Fano + 14 blocks equal the stored 240x240 tables, all 3 sets of the triple; Latin: {latin}", if oct_ok { "ok" } else { "FAIL" });
    assert!(fwd_ok && inv_ok && oct_ok && latin && spare_n == 3_385);

    let state_bytes = 3 + 8 + 1 + 2 + 2 + 1 + 1; // packed: quad, unit (i8 x 8, coords in -2..2), set, seats, 9-gon, ring, live
    println!("\nresident data, tables:");
    println!("  41 load  pattern -> state   {}", kb(fwd.len() * 3));
    println!("  41 read  state -> pattern   {}", kb(inv.len() * 2));
    println!("  41 spare bitmap             {}", kb(spare.len() * 8));
    println!("  Octavian triple 3 x 240^2   {}", kb(oct_tables.iter().map(|t| t.len()).sum()));
    println!("  kernel state (packed)       {}", kb(state_bytes));
    let tab_total = fwd.len() * 3 + inv.len() * 2 + spare.len() * 8 + 3 * 57_600 + state_bytes;
    println!("  total                       {}   (built in {build_ms} ms)", kb(tab_total));
    println!("\nresident data, table-free:");
    println!("  41 walk: 2 starts, 2 matrices, 2 baby-step tables of {}   {}", free.m, kb(free.bytes()));
    println!("  Octavian: Fano table + 3 x 14 block masks  {}", kb(fano_bytes + 3 * 14));
    println!("  kernel state (packed)       {}", kb(state_bytes));
    let free_total = free.bytes() + fano_bytes + 3 * 14 + state_bytes;
    println!("  total                       {}", kb(free_total));
    println!("\nprocess working set (includes the executable and runtime):");
    println!("  at start {}, after table-free setup {}, after tables {} (peak {})", kb(ws0), kb(ws_free), kb(ws_tab), kb(peak));
    println!("\nfor scale: the 65,536 patterns themselves as raw BF16 = {}", kb(65_536 * 2));
}
