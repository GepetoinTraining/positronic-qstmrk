//! `atlas_rns_bench` — the lane in residue rings against the 256-bit lane (after v4.4's lane
//! kernel; LOG §26). The cube's walk on one row: entries (lane = odd, band = rung) from the top
//! rung down, the activation prefixed to integers, per term one add into its lane, per entry a
//! lazy doubling by the band gap. Two lanes:
//!   (a) Wide: i128 sum of the entry's columns, then `Wide::shl(gap)` + add — the kernel today;
//!   (b) RNS: K prime rings < 2^31; per term K u64 adds of the column's residues (lazy, reduced
//!       once per entry); the doubling is one Shoup multiply per ring per entry; the exact lane
//!       is rebuilt once per row by CRT and must equal (a) bit for bit.
//! Integer timing. One thread, L1/L2-hot. `atlas_rns_bench [terms] [rows]`.

use std::time::Instant;

use atlas::matmul::Wide;
use num_bigint::BigInt;
use num_traits::{One, Zero};

const PRIMES: [u64; 9] = [998244353, 985661441, 754974721, 167772161, 469762049, 1004535809, 1811939329, 2013265921, 2113929217];

#[inline(always)]
fn shoup(a: u64, w: u64, wp: u64, p: u64) -> u64 {
    // a*w mod p with w' = floor(w*2^64/p) precomputed; a, w < p < 2^31
    let q = ((a as u128 * wp as u128) >> 64) as u64;
    let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
    if r >= p { r - p } else { r }
}

fn wide_to_big(w: Wide) -> BigInt {
    let neg = w.is_neg();
    let (hi, lo): (u128, u128) = w.mag();
    let v = (BigInt::from(hi) << 128u32) + BigInt::from(lo);
    if neg { -v } else { v }
}

fn crt(res: &[u64], primes: &[u64]) -> BigInt {
    // Garner: x = r0 + p0*(t1 + p1*(t2 + ...)), then centre into (-M/2, M/2]
    let mut m = BigInt::one();
    let mut x = BigInt::zero();
    for (i, (&r, &p)) in res.iter().zip(primes.iter()).enumerate() {
        let pb = BigInt::from(p);
        if i == 0 {
            x = BigInt::from(r);
            m = pb;
            continue;
        }
        // t = (r - x) * m^-1 mod p
        let xm = ((&x % &pb) + &pb) % &pb;
        let xm_u: u64 = xm.try_into().unwrap();
        let mm: BigInt = ((&m % &pb) + &pb) % &pb;
        let mm_u: u64 = mm.try_into().unwrap();
        let inv = modinv(mm_u, p);
        let diff = (r + p - xm_u) % p;
        let t = (diff as u128 * inv as u128 % p as u128) as u64;
        x += &m * BigInt::from(t);
        m *= pb;
    }
    if &x > &(&m >> 1) { x - m } else { x }
}

fn modinv(a: u64, p: u64) -> u64 {
    // Fermat: a^(p-2)
    let (mut r, mut b, mut e) = (1u128, a as u128 % p as u128, p - 2);
    while e > 0 {
        if e & 1 == 1 { r = r * b % p as u128; }
        b = b * b % p as u128;
        e >>= 1;
    }
    r as u64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let terms: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let rows: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    const FRAC: i32 = 64;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    // the activation, prefixed: pv[col] = ±m_x · 2^(n_x + FRAC), n_x in [-16, 0]
    let pv: Vec<i128> = (0..terms).map(|_| {
        let m = ((next() % 128) * 2 + 1) as i128;
        let n = FRAC - (next() % 17) as i32;
        let v = m << n;
        if next() & 1 == 1 { -v } else { v }
    }).collect();
    // the weights: lane = odd index 0..128 with sign folded as lane 128..256, band = rung index (top = 0)
    let bands = 25usize;
    let lane_of: Vec<usize> = (0..terms).map(|_| (next() % 256) as usize).collect();
    let band_of: Vec<usize> = (0..terms).map(|_| { let g = (next() % 1000) as f_free::U; f_free::band(g, bands) }).collect();
    // entries: (band, lane) -> cols, walked bands top-down
    let mut cols_by: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); 256]; bands];
    for j in 0..terms { cols_by[band_of[j]][lane_of[j]].push(j as u32); }
    let mut entries: Vec<(usize, usize, Vec<u32>)> = Vec::new();
    for b in 0..bands {
        for l in 0..256 {
            if !cols_by[b][l].is_empty() {
                entries.push((b, l, cols_by[b][l].clone()));
            }
        }
    }
    let n_entries = entries.len();

    // (a) Wide
    let t = Instant::now();
    let mut lanes_a = [Wide::ZERO; 256];
    let mut chk_a = 0i128;
    for _ in 0..rows {
        let mut acc = [Wide::ZERO; 256];
        let mut last = [0u8; 256];
        let mut touched = [false; 256];
        for (b, l, cols) in entries.iter() {
            let gap = if touched[*l] { (*b as u8 - last[*l]) as u32 } else { 0 };
            let mut sum = 0i128;
            for &c in cols { sum += pv[c as usize]; }
            let mut x = acc[*l];
            if gap > 0 { x = x.shl(gap); }
            x.add(Wide::from_i128(sum));
            acc[*l] = x;
            last[*l] = *b as u8;
            touched[*l] = true;
        }
        for l in 0..256 {
            if touched[l] { let fin = (bands - 1) as u32 - last[l] as u32; acc[l] = acc[l].shl(fin); }
        }
        lanes_a = acc;
        chk_a = chk_a.wrapping_add(acc[7].mag().1 as i128);
    }
    let ns_a = t.elapsed().as_nanos() * 100 / (rows * terms) as u128;

    // (b) RNS with K rings
    let mut report = String::new();
    for &k in &[4usize, 9] {
        let primes = &PRIMES[..k];
        // residues of the prefixed activation, per column: K u64 contiguous
        let mut pvr: Vec<u64> = Vec::with_capacity(terms * k);
        for j in 0..terms {
            let v = pv[j];
            for &p in primes {
                let m = (v.unsigned_abs() % p as u128) as u64;
                pvr.push(if v < 0 && m != 0 { p - m } else { m });
            }
        }
        // 2^g mod p and its Shoup precomputation, g = 0..=bands
        let pow2: Vec<Vec<(u64, u64)>> = primes.iter().map(|&p| (0..=bands as u32).map(|g| { let w = ((1u128 << g) % p as u128) as u64; let wp = (((w as u128) << 64) / p as u128) as u64; (w, wp) }).collect()).collect();
        let t = Instant::now();
        let mut lanes_b: Vec<u64> = vec![0; 256 * k];
        let mut chk_b = 0u64;
        for _ in 0..rows {
            let mut acc: Vec<u64> = vec![0; 256 * k];
            let mut last = [0u8; 256];
            let mut touched = [false; 256];
            for (b, l, cols) in entries.iter() {
                let gap = if touched[*l] { (*b as u8 - last[*l]) as usize } else { 0 };
                let base = l * k;
                // lazy sum of residues: u64 adds, no reduction inside the entry
                let mut sum = [0u64; 9];
                for &c in cols {
                    let r = &pvr[c as usize * k..c as usize * k + k];
                    for i in 0..k { sum[i] += r[i]; }
                }
                for i in 0..k {
                    let p = primes[i];
                    let mut x = acc[base + i];
                    if gap > 0 { let (w, wp) = pow2[i][gap]; x = shoup(x, w, wp, p); }
                    acc[base + i] = (x + sum[i] % p) % p;
                }
                last[*l] = *b as u8;
                touched[*l] = true;
            }
            for l in 0..256 {
                if touched[l] { let fin = bands - 1 - last[l] as usize; for i in 0..k { let (w, wp) = pow2[i][fin]; acc[l * k + i] = shoup(acc[l * k + i], w, wp, primes[i]); } }
            }
            chk_b = chk_b.wrapping_add(acc[7 * k]);
            lanes_b = acc;
        }
        let ns_b = t.elapsed().as_nanos() * 100 / (rows * terms) as u128;
        // exactness: every lane rebuilt by CRT equals the Wide lane
        let t2 = Instant::now();
        let mut equal = 0;
        for l in 0..256 {
            let big = crt(&lanes_b[l * k..l * k + k], primes);
            if big == wide_to_big(lanes_a[l]) { equal += 1; }
        }
        let crt_us = t2.elapsed().as_micros();
        report.push_str(&format!("(b) RNS K={}  {}.{:02} ns/term  lanes equal to Wide by CRT: {}/256  (CRT of 256 lanes: {} us per row)  chk {}\n", k, ns_b / 100, ns_b % 100, equal, crt_us, chk_b));
    }
    println!("format=atlas.rns_bench.v1 terms={} rows={} entries/row={} bands={} one thread", terms, rows, n_entries, bands);
    println!("(a) Wide      {}.{:02} ns/term  (i128 sum per entry, Wide shl+add per entry)  chk {}", ns_a / 100, ns_a % 100, chk_a);
    print!("{}", report);
}

mod f_free {
    pub type U = u64;
    /// a bell over the bands (the binade census is peaked): g in 0..1000 -> band index
    pub fn band(g: U, bands: usize) -> usize {
        let c = [1u64, 2, 4, 8, 16, 32, 56, 80, 100, 110, 112, 110, 100, 80, 56, 32, 16, 8, 4, 2, 1, 1, 1, 1, 1];
        let total: u64 = c[..bands].iter().sum();
        let mut x = g * total / 1000;
        for (i, &w) in c[..bands].iter().enumerate() { if x < w { return i; } x -= w; }
        bands - 1
    }
}
