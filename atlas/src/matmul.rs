//! The bucket accumulate (spec §3): `y[s,o] = Σ_i x[s,i] · W[o,i]` — EXACT.
//!
//! Every term is the exact product of two cells: signs xor, odd cores multiply (≤ 255·255,
//! 16 bits), rungs add. The bucket is one 256-bit integer per output cell at a fixed scale
//! 2^-F; a term enters by a shift and an add, and there is NO truncation anywhere before the
//! single collapse to bf16 at the gather (round-to-nearest-even, sticky for the bits below
//! the significand). An exact sum is order-independent, so any kernel that computes the same
//! integer — the bucket here, the radix chain in `fold.rs` — is bit-identical by construction.
//!
//! The one declared floor is on the ACTIVATION: a pattern with rung below −ACT_FLOOR (2^-90,
//! far under anything a normalized stream produces; bf16's own normal floor is 2^-126) is
//! dropped with sticky. It is the same floor in every kernel, applied once per activation, so
//! the set of terms that enter is identical everywhere. Above the lane's top a term is
//! REFUSED (panic with the scale), never wrapped.
//!
//! Weights are read through the surfaces: id → (sign, m, n) tables; the grid is walked
//! tile-major (`Grid::line`), 128 contiguous ids per step.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cells;
use crate::surfaces::{Atoms, Grid};

/// Activations below 2^-ACT_FLOOR are dropped (sticky). The same floor in every kernel.
pub const ACT_FLOOR: i32 = 90;
/// Fractional bits for activation × activation products (both operands ≥ 2^-ACT_FLOOR).
pub const FRAC_DENSE: i32 = 2 * ACT_FLOOR;
/// Fractional bits for a weight tensor with lowest rung `rmin`: every term is exact.
#[inline(always)]
pub const fn frac_for(rmin: i32) -> i32 {
    ACT_FLOOR - rmin
}
const PARALLEL_MIN_MACS: usize = 1 << 18;
const STEAL_FACTOR: usize = 8;

// ── 256-bit signed integer: value = hi·2^128 + lo ────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wide {
    pub hi: i128,
    pub lo: u128,
}

impl Wide {
    pub const ZERO: Wide = Wide { hi: 0, lo: 0 };

    #[inline(always)]
    pub fn from_i128(v: i128) -> Wide {
        Wide { hi: if v < 0 { -1 } else { 0 }, lo: v as u128 }
    }
    #[inline(always)]
    pub fn add(&mut self, o: Wide) {
        let (lo, c) = self.lo.overflowing_add(o.lo);
        self.lo = lo;
        self.hi = self.hi.wrapping_add(o.hi).wrapping_add(c as i128);
    }
    #[inline(always)]
    pub fn sub(&mut self, o: Wide) {
        let (lo, b) = self.lo.overflowing_sub(o.lo);
        self.lo = lo;
        self.hi = self.hi.wrapping_sub(o.hi).wrapping_sub(b as i128);
    }
    /// Left shift by k < 256. The value must stay within 255 bits (asserted by the caller's lane).
    #[inline(always)]
    pub fn shl(self, k: u32) -> Wide {
        if k == 0 {
            self
        } else if k < 128 {
            Wide { hi: (self.hi << k) | ((self.lo >> (128 - k)) as i128), lo: self.lo << k }
        } else {
            Wide { hi: (self.lo as i128) << (k - 128), lo: 0 }
        }
    }
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.hi == 0 && self.lo == 0
    }
    #[inline(always)]
    pub fn is_neg(&self) -> bool {
        self.hi < 0
    }
    pub fn neg(self) -> Wide {
        let mut z = Wide::ZERO;
        z.sub(self);
        z
    }
    /// Magnitude as (hi, lo) unsigned.
    pub fn mag(self) -> (u128, u128) {
        let m = if self.is_neg() { self.neg() } else { self };
        (m.hi as u128, m.lo)
    }
}

#[inline(always)]
fn top_bit(hi: u128, lo: u128) -> i32 {
    if hi != 0 {
        128 + (127 - hi.leading_zeros() as i32)
    } else {
        127 - lo.leading_zeros() as i32
    }
}

/// Bits [lo_bit, lo_bit+n) of the 256-bit magnitude as u32 (n ≤ 32).
#[inline(always)]
fn bits(hi: u128, lo: u128, lo_bit: i32, n: u32) -> u32 {
    let v: u128 = if lo_bit >= 128 {
        hi >> (lo_bit - 128)
    } else if lo_bit + n as i32 <= 128 {
        lo >> lo_bit
    } else {
        (lo >> lo_bit) | (hi << (128 - lo_bit))
    };
    (v & ((1u128 << n) - 1)) as u32
}

/// True if any bit below `bit` is set.
#[inline(always)]
fn any_below(hi: u128, lo: u128, bit: i32) -> bool {
    if bit <= 0 {
        false
    } else if bit >= 128 {
        lo != 0 || (hi & ((1u128 << (bit - 128)) - 1)) != 0
    } else {
        (lo & ((1u128 << bit) - 1)) != 0
    }
}

// ── the bucket ───────────────────────────────────────────────────────────────

/// One activation row, decomposed once; activations under the floor dropped (sticky).
pub struct XRow {
    pub sheet: Vec<u8>,
    pub m: Vec<u32>,
    pub n: Vec<i32>,
    pub sticky: bool,
}

impl XRow {
    pub fn new(x: &[u16]) -> XRow {
        let mut r = XRow { sheet: Vec::with_capacity(x.len()), m: Vec::with_capacity(x.len()), n: Vec::with_capacity(x.len()), sticky: false };
        for &p in x {
            assert!(cells::is_finite(p), "activation carries a non-finite pattern {:#06x}", p);
            let (s, m, n) = cells::decompose(p);
            if m != 0 && (n as i32) < -ACT_FLOOR {
                r.sticky = true;
                r.sheet.push(s);
                r.m.push(0);
                r.n.push(0);
                continue;
            }
            r.sheet.push(s);
            r.m.push(m as u32);
            r.n.push(n as i32);
        }
        r
    }
}

/// The exact bucket for one output cell at scale 2^-frac.
#[derive(Clone, Copy)]
pub struct Bucket {
    pub acc: Wide,
    pub sticky: bool,
    pub frac: i32,
}

impl Bucket {
    #[inline(always)]
    pub const fn new(frac: i32) -> Bucket {
        Bucket { acc: Wide::ZERO, sticky: false, frac }
    }
    /// Add the exact product (sx, mx, nx)·(sw, mw, nw). Both operands already above the floor.
    #[inline(always)]
    pub fn add(&mut self, sx: u8, mx: u32, nx: i32, sw: u8, mw: u32, nw: i32) {
        if mx == 0 || mw == 0 {
            return;
        }
        let sh = nx + nw + self.frac;
        assert!(sh >= 0, "bucket: term below the exact scale (nx {} nw {} frac {})", nx, nw, self.frac);
        assert!(sh <= 236, "bucket refused: term scale 2^{} exceeds the lane", sh - self.frac);
        let term = Wide::from_i128((mx * mw) as i128).shl(sh as u32);
        if (sx ^ sw) != 0 {
            self.acc.sub(term);
        } else {
            self.acc.add(term);
        }
    }
    /// Add a term whose odd product was SELECTED (fiber kernel): `prod` = m_x·m_w, `nsum` = n_x + n_w.
    #[inline(always)]
    pub fn add_product(&mut self, sign: u8, prod: u32, nsum: i32) {
        let sh = nsum + self.frac;
        assert!(sh >= 0, "bucket: term below the exact scale (nsum {} frac {})", nsum, self.frac);
        assert!(sh <= 236, "bucket refused: term scale 2^{} exceeds the lane", sh - self.frac);
        let term = Wide::from_i128(prod as i128).shl(sh as u32);
        if sign != 0 {
            self.acc.sub(term);
        } else {
            self.acc.add(term);
        }
    }
    /// Add an already-scaled wide value (the radix chain's lane total), same scale.
    #[inline(always)]
    pub fn add_wide(&mut self, w: Wide) {
        self.acc.add(w);
    }

    /// The single collapse: exact integer → bf16 pattern, round-to-nearest-even.
    pub fn collapse(&self) -> u16 {
        if self.acc.is_zero() {
            return 0;
        }
        let sign: u16 = if self.acc.is_neg() { 0x8000 } else { 0 };
        let (hi, lo) = self.acc.mag();
        let hb = top_bit(hi, lo);
        let (mut sig, round, low_sticky) = if hb >= 8 {
            (bits(hi, lo, hb - 7, 8), bits(hi, lo, hb - 8, 1), any_below(hi, lo, hb - 8))
        } else {
            ((lo as u32) << (7 - hb), 0u32, false)
        };
        let sticky = low_sticky || self.sticky;
        let mut exp = hb - self.frac + 127;
        if round == 1 && (sticky || (sig & 1) == 1) {
            sig += 1;
            if sig == 0x100 {
                sig >>= 1;
                exp += 1;
            }
        }
        assert!(exp < 0xFF, "collapse refused: value overflows bf16");
        if exp < 1 {
            return sign; // below bf16's normal floor: the same flush V5 makes
        }
        sign | ((exp as u16) << 7) | ((sig & 0x7F) as u16)
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u16);
unsafe impl Send for SendPtr {}

fn thread_cap() -> usize {
    std::env::var("ATLAS_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

pub(crate) fn run_cells<F>(cell: F, in_f: usize, y: &mut [u16])
where
    F: Fn(usize) -> u16 + Copy + Send + Sync,
{
    let total = y.len();
    let n_threads = if total.saturating_mul(in_f) < PARALLEL_MIN_MACS { 1 } else { thread_cap().min(total) };
    if n_threads == 1 {
        for (idx, slot) in y.iter_mut().enumerate() {
            *slot = cell(idx);
        }
        return;
    }
    let chunk = total.div_ceil(n_threads * STEAL_FACTOR).max(1);
    let n_chunks = total.div_ceil(chunk);
    let cursor = AtomicUsize::new(0);
    let cursor_ref = &cursor;
    let yptr = SendPtr(y.as_mut_ptr());
    std::thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(move || {
                let yptr = yptr;
                loop {
                    let c = cursor_ref.fetch_add(1, Ordering::Relaxed);
                    if c >= n_chunks {
                        break;
                    }
                    let start = c * chunk;
                    let end = (start + chunk).min(total);
                    for idx in start..end {
                        // SAFETY: disjoint in-bounds ranges; scope joins before y is read.
                        unsafe { *yptr.0.add(idx) = cell(idx) };
                    }
                }
            });
        }
    });
}

/// `y = x · Wᵀ` with W a grid `[out_f, in_f]` read through the surfaces. `x` is `[seq, in_f]` patterns.
pub fn matmul_grid(x: &[u16], seq: usize, in_f: usize, w: &Grid, atoms: &Atoms) -> Vec<u16> {
    assert_eq!(x.len(), seq * in_f, "matmul_grid: x is not [seq, in_f]");
    assert_eq!(w.cols, in_f, "matmul_grid: grid cols {} != in_f {}", w.cols, in_f);
    let out_f = w.rows;
    let frac = frac_for(w.rmin);
    let rows: Vec<XRow> = (0..seq).map(|s| XRow::new(&x[s * in_f..(s + 1) * in_f])).collect();
    let rows_ref = &rows;
    let tcs = w.tile_cols();
    let cell = move |idx: usize| -> u16 {
        let s = idx / out_f;
        let o = idx % out_f;
        let xr = &rows_ref[s];
        let mut b = Bucket::new(frac);
        b.sticky = xr.sticky;
        for tc in 0..tcs {
            let line = w.line(o, tc);
            let base = tc * w.tw;
            for (j, &id) in line.iter().enumerate() {
                let id = id as usize;
                let i = base + j;
                b.add(xr.sheet[i], xr.m[i], xr.n[i], atoms.sheet[id], atoms.m[id] as u32, atoms.n[id] as i32);
            }
        }
        b.collapse()
    };
    let mut y = vec![0u16; seq * out_f];
    run_cells(cell, in_f, &mut y);
    y
}

/// `y = x · Wᵀ` with both operands dense pattern matrices (activation × activation, e.g. Q·Kᵀ).
pub fn matmul_dense(x: &[u16], w: &[u16], seq: usize, in_f: usize, out_f: usize) -> Vec<u16> {
    assert_eq!(x.len(), seq * in_f);
    assert_eq!(w.len(), out_f * in_f);
    let xs: Vec<XRow> = (0..seq).map(|s| XRow::new(&x[s * in_f..(s + 1) * in_f])).collect();
    let ws: Vec<XRow> = (0..out_f).map(|o| XRow::new(&w[o * in_f..(o + 1) * in_f])).collect();
    let (xs, ws) = (&xs, &ws);
    let cell = move |idx: usize| -> u16 {
        let s = idx / out_f;
        let o = idx % out_f;
        let (a, c) = (&xs[s], &ws[o]);
        let mut b = Bucket::new(FRAC_DENSE);
        b.sticky = a.sticky || c.sticky;
        for i in 0..in_f {
            b.add(a.sheet[i], a.m[i], a.n[i], c.sheet[i], c.m[i], c.n[i]);
        }
        b.collapse()
    };
    let mut y = vec![0u16; seq * out_f];
    run_cells(cell, in_f, &mut y);
    y
}

/// Elementwise bf16 add of two pattern vectors through the bucket (exact sum, one collapse).
pub fn add(a: &[u16], b: &[u16]) -> Vec<u16> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&p, &q)| {
            let (sp, mp, np) = cells::decompose(p);
            let (sq, mq, nq) = cells::decompose(q);
            let mut k = Bucket::new(ACT_FLOOR);
            for (s, m, n) in [(sp, mp, np), (sq, mq, nq)] {
                if m != 0 && (n as i32) < -ACT_FLOOR {
                    k.sticky = true;
                } else {
                    k.add(s, m as u32, n as i32, 0, 1, 0);
                }
            }
            k.collapse()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::bf16::{bf16_add, bf16_mul};

    fn pat(sheet: u8, m: u8, n: i16) -> u16 {
        cells::recompose(sheet, m, n)
    }

    #[test]
    fn wide_shift_and_add_round_trip() {
        let a = Wide::from_i128(0x1234_5678_9abc_def0).shl(200);
        let mut b = a;
        b.add(a);
        assert_eq!(b, Wide::from_i128(0x1234_5678_9abc_def0).shl(201));
        let mut c = b;
        c.sub(a);
        assert_eq!(c, a);
        let n = Wide::from_i128(-5).shl(3);
        assert!(n.is_neg());
        assert_eq!(n.mag(), (0, 40));
    }

    #[test]
    fn single_product_matches_v5_bf16_mul() {
        // every finite normal pattern above the activation floor, sampled
        let samples: Vec<u16> = (0..65536u32)
            .step_by(97)
            .map(|p| p as u16)
            .filter(|&p| cells::is_finite(p) && ((p >> 7) & 0xFF) > 60 && ((p >> 7) & 0xFF) < 150)
            .collect();
        for &a in &samples {
            for &b in samples.iter().step_by(7) {
                let (sa, ma, na) = cells::decompose(a);
                let (sb, mb, nb) = cells::decompose(b);
                let mut k = Bucket::new(FRAC_DENSE);
                k.add(sa, ma as u32, na as i32, sb, mb as u32, nb as i32);
                let ours = k.collapse();
                let theirs = bf16_mul(a, b);
                // V5 flushes subnormal results to 0 and overflows to inf; we refuse overflow and flush too
                if (theirs & 0x7F80) != 0x7F80 {
                    assert_eq!(ours, theirs, "{:#06x} * {:#06x}", a, b);
                }
            }
        }
    }

    #[test]
    fn add_matches_v5_bf16_add() {
        let samples: Vec<u16> = (0..65536u32)
            .step_by(211)
            .map(|p| p as u16)
            .filter(|&p| cells::is_finite(p) && ((p >> 7) & 0xFF) > 60 && ((p >> 7) & 0xFF) < 250)
            .collect();
        for &a in &samples {
            for &b in samples.iter().step_by(5) {
                let ours = add(&[a], &[b])[0];
                let theirs = bf16_add(a, b);
                if (theirs & 0x7F80) != 0x7F80 {
                    assert_eq!(ours, theirs, "{:#06x} + {:#06x}", a, b);
                }
            }
        }
    }

    #[test]
    fn dot_is_exact_before_the_collapse() {
        let one = pat(0, 1, 0);
        let neg = pat(1, 1, 0);
        let x = [one, one, neg];
        let w = [one, one, one];
        assert_eq!(matmul_dense(&x, &w, 1, 3, 1)[0], one);
        let x = [pat(0, 3, 0)];
        let w = [pat(0, 5, -5)];
        assert_eq!(matmul_dense(&x, &w, 1, 1, 1)[0], pat(0, 15, -5));
        // a tiny term must not be lost: 1.0 + 2^-20 - 1.0 = 2^-20 exactly (bf16 could not hold the middle)
        let x = [one, pat(0, 1, -20), neg];
        let w3 = [one, one, one];
        assert_eq!(matmul_dense(&x, &w3, 1, 3, 1)[0], pat(0, 1, -20));
    }

    #[test]
    fn lane_refuses_above_the_top() {
        let r = std::panic::catch_unwind(|| {
            let mut k = Bucket::new(FRAC_DENSE);
            k.add(0, 1, 60, 0, 1, 60);
            k.collapse()
        });
        assert!(r.is_err(), "a term above the lane must be refused, not wrapped");
    }
}
