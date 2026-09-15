//! `atlas_select_bench` — selection instead of calculation (pgarcia: "1/value = id, value is
//! value; two LUTs; triangulation and selection, no longer calculation at runtime").
//!
//! The smallest form: the odd part of a term, m_x * m_w (two 8-bit odds), as (a) an imul and
//! (b) a selection from the 128 x 128 table of odd products; the rung is a shift either way.
//! Same bits by construction; the number is ns per term, L1-hot, one thread, 2,048-term rows.
//!
//!   atlas_select_bench

use std::time::Instant;

const TERMS: usize = 2048;
const ROWS: usize = 4096;

#[inline(always)]
fn odd_idx(m: u16) -> usize {
    (m as usize) >> 1 // odd m in 1..=255 -> 0..=127
}

fn main() {
    // the second LUT: odd x odd -> product (16,384 entries, 32 KB, exact, bijective on (m_x, m_w) up to order)
    let mut lut = vec![0u32; 128 * 128];
    for i in 0..128 {
        for j in 0..128 {
            lut[i * 128 + j] = ((2 * i + 1) * (2 * j + 1)) as u32;
        }
    }
    // a row of weights and an activation, decomposed: odd m (1..=255 odd), rung n
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mw: Vec<u16> = (0..TERMS).map(|_| ((next() % 128) * 2 + 1) as u16).collect();
    let nw: Vec<i32> = (0..TERMS).map(|_| -((next() % 24) as i32) - 8).collect();
    let mx: Vec<u16> = (0..TERMS).map(|_| ((next() % 128) * 2 + 1) as u16).collect();
    let nx: Vec<i32> = (0..TERMS).map(|_| -((next() % 16) as i32)).collect();
    let sx: Vec<bool> = (0..TERMS).map(|_| next() & 1 == 1).collect();
    let sw: Vec<bool> = (0..TERMS).map(|_| next() & 1 == 1).collect();
    const BASE: i32 = 64; // one denominator 2^-64 for the row

    // (a) calculation: imul + shift + add
    let t = Instant::now();
    let mut acc_a: i128 = 0;
    for _ in 0..ROWS {
        let mut acc: i128 = 0;
        for j in 0..TERMS {
            let p = (mx[j] as u32 * mw[j] as u32) as i128;
            let sh = (BASE + nx[j] + nw[j]) as u32;
            let v = p << sh;
            acc += if sx[j] ^ sw[j] { -v } else { v };
        }
        acc_a = acc_a.wrapping_add(acc);
    }
    let ns_a = t.elapsed().as_nanos() * 100 / (ROWS * TERMS) as u128; // hundredths of a ns, integer

    // (b) selection: LUT + shift + add
    let t = Instant::now();
    let mut acc_b: i128 = 0;
    for _ in 0..ROWS {
        let mut acc: i128 = 0;
        for j in 0..TERMS {
            let p = lut[odd_idx(mx[j]) * 128 + odd_idx(mw[j])] as i128;
            let sh = (BASE + nx[j] + nw[j]) as u32;
            let v = p << sh;
            acc += if sx[j] ^ sw[j] { -v } else { v };
        }
        acc_b = acc_b.wrapping_add(acc);
    }
    let ns_b = t.elapsed().as_nanos() * 100 / (ROWS * TERMS) as u128; // hundredths of a ns, integer

    // (c) selection with the rung folded in: (m_x, m_w) -> product already shifted by the
    //     weight's rung is not a table (rung is per cell); but the SIGN can be selected: a
    //     256 x 256 signed table keyed by the signed odd (sign beside): one lookup, one shift, one add
    let mut slut = vec![0i32; 256 * 256];
    for i in 0..256 {
        for j in 0..256 {
            let a = (2 * (i & 127) + 1) as i32 * if i >= 128 { -1 } else { 1 };
            let b = (2 * (j & 127) + 1) as i32 * if j >= 128 { -1 } else { 1 };
            slut[i * 256 + j] = a * b;
        }
    }
    let kx: Vec<usize> = (0..TERMS).map(|j| odd_idx(mx[j]) + if sx[j] { 128 } else { 0 }).collect();
    let kw: Vec<usize> = (0..TERMS).map(|j| odd_idx(mw[j]) + if sw[j] { 128 } else { 0 }).collect();
    let t = Instant::now();
    let mut acc_c: i128 = 0;
    for _ in 0..ROWS {
        let mut acc: i128 = 0;
        for j in 0..TERMS {
            let p = slut[kx[j] * 256 + kw[j]] as i128;
            let sh = (BASE + nx[j] + nw[j]) as u32;
            acc += p << sh;
        }
        acc_c = acc_c.wrapping_add(acc);
    }
    let ns_c = t.elapsed().as_nanos() * 100 / (ROWS * TERMS) as u128; // hundredths of a ns, integer

    println!("format=atlas.select_bench.v1  rows={} terms={} L1-hot one thread", ROWS, TERMS);
    println!("(a) calculation  imul + shift + add             {}.{:02} ns/term   acc {}", ns_a / 100, ns_a % 100, acc_a);
    println!("(b) selection    odd LUT 128x128 + shift + add  {}.{:02} ns/term   acc {}  identical {}", ns_b / 100, ns_b % 100, acc_b, acc_a == acc_b);
    println!("(c) selection    signed LUT 256x256 + shift + add {}.{:02} ns/term   acc {}  identical {}", ns_c / 100, ns_c % 100, acc_c, acc_a == acc_c);
}
