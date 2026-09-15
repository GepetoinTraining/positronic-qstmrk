//! The fold — the third surface, and the radix kernel over it (spec §3, LOG §5–§6, pgarcia 2026-09-13).
//!
//! Every output row of a weight tensor is folded by (sign, odd core, rung): for each such bucket,
//! the columns that carry it. The fold is static — it is the frame surface written per row — and
//! it is built once at load from the Morton grid and the alphabet.
//!
//! The kernel over it is the one this night arrived at:
//!   - activations are prefixed ONCE per token at scale 2^-ACT_FLOOR (the one declared floor,
//!     identical in every kernel); the tensor's lowest rung K is its one denominator 2^K and
//!     the chain lands every bucket at scale 2^-(ACT_FLOOR − K), the bucket kernel's scale —
//!     the same exact integer, so the same bits, by construction;
//!   - a term is ONE ADD: the prefixed activation into its bucket, up to PLANES transcript
//!     positions per walk of the columns (the turn re-parses the whole transcript; no KV cache);
//!   - per (sign, odd) the rung chain from the highest rung down — Babbage's carriage, ×2 is one
//!     shift, one add per rung — lands every bucket at scale 2^-ACC_FRAC exactly, doubling only;
//!   - per sign the odd chain with the crank on 255: Σ(2j+1)·a_j = S_0 + 2·ΣS_j by suffix sums —
//!     255 adds and one shift, no multiply;
//!   - the two signs subtract, one collapse per plane at the gather.
//! No multiply and no per-term shift anywhere in the pass. Bit-identical to `matmul::matmul_grid`
//! (LOG §6, variant C: 0 mismatches over 8,192 cells) and judged by the reference receipt.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Diagnostic: activations that hit the prefix floor (truncated with sticky) across the run.
pub static PREFIX_TRUNCATIONS: AtomicU64 = AtomicU64::new(0);
pub static PREFIX_TOTAL: AtomicU64 = AtomicU64::new(0);

use crate::cells;
use crate::matmul::{frac_for, Bucket, Wide, ACT_FLOOR};
use crate::surfaces::{Atoms, Grid};

/// Buckets of one output row, sorted by (sign, odd, rung DESCENDING) so the rung chain walks
/// them in Horner order; `cols` holds the columns of every bucket back to back.
pub struct RowFold {
    /// (bucket key, count). key = sign·128·rspan + odd_idx·rspan + (rmax − rung)
    pub buckets: Vec<(u16, u16)>,
    pub cols: Vec<u16>,
}

pub struct TensorFold {
    pub rows: usize,
    pub cols: usize,
    /// lowest rung in the tensor: the one denominator 2^rmin
    pub rmin: i32,
    pub rmax: i32,
    pub rspan: usize,
    pub rows_f: Vec<RowFold>,
}

impl TensorFold {
    pub fn from_grid(g: &Grid, atoms: &Atoms) -> TensorFold {
        // rung span of the tensor
        let (rmin, rmax) = (g.rmin, g.rmax);
        let rspan = (rmax - rmin + 1) as usize;
        assert!(256 * rspan <= 65536, "rung span {} too wide for u16 bucket keys", rspan);
        let n = g.rows;
        let mut rows_f: Vec<Option<RowFold>> = (0..n).map(|_| None).collect();
        // fold rows in parallel
        let cursor = AtomicUsize::new(0);
        let cursor_ref = &cursor;
        let out_ptr = SendPtr(rows_f.as_mut_ptr());
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1).min(n.max(1));
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(move || {
                    let out_ptr = out_ptr;
                    let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(g.cols);
                    loop {
                        let o = cursor_ref.fetch_add(1, Ordering::Relaxed);
                        if o >= n {
                            break;
                        }
                        pairs.clear();
                        for tc in 0..g.tile_cols() {
                            for (j, &id) in g.line(o, tc).iter().enumerate() {
                                let id = id as usize;
                                let m = atoms.m[id];
                                if m == 0 {
                                    continue;
                                }
                                let key = (atoms.sheet[id] as usize) * 128 * rspan
                                    + ((m as usize - 1) / 2) * rspan
                                    + (rmax - atoms.n[id] as i32) as usize;
                                pairs.push((key as u16, (tc * g.tw + j) as u16));
                            }
                        }
                        pairs.sort_unstable();
                        let mut buckets: Vec<(u16, u16)> = Vec::new();
                        let mut cols: Vec<u16> = Vec::with_capacity(pairs.len());
                        for &(k, c) in &pairs {
                            match buckets.last_mut() {
                                Some((lk, cnt)) if *lk == k => *cnt += 1,
                                _ => buckets.push((k, 1)),
                            }
                            cols.push(c);
                        }
                        // SAFETY: each o written by exactly one thread; scope joins before read.
                        unsafe { *out_ptr.0.add(o) = Some(RowFold { buckets, cols }) };
                    }
                });
            }
        });
        TensorFold { rows: g.rows, cols: g.cols, rmin, rmax, rspan, rows_f: rows_f.into_iter().map(|r| r.unwrap()).collect() }
    }

    pub fn bytes(&self) -> usize {
        self.rows_f.iter().map(|r| r.buckets.len() * 4 + r.cols.len() * 2).sum()
    }
}

struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        SendPtr(self.0)
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}

/// One activation plane prefixed at the declared floor: value·2^ACT_FLOOR exactly, signed;
/// an activation below 2^-ACT_FLOOR is dropped with sticky (the same floor as `XRow`).
pub struct Plane {
    pub v: Vec<i128>,
    pub sticky: bool,
}

pub fn prefix(x: &[u16]) -> Plane {
    let mut v = Vec::with_capacity(x.len());
    let mut sticky = false;
    for &p in x {
        assert!(cells::is_finite(p), "activation carries a non-finite pattern {:#06x}", p);
        let (s, m, n) = cells::decompose(p);
        let mx = m as i128;
        let sh = n as i32 + ACT_FLOOR;
        let t = if mx == 0 {
            0
        } else if sh >= 0 {
            assert!(sh <= 112, "prefix refused: activation scale 2^{} exceeds the lane", sh - ACT_FLOOR);
            mx << sh
        } else {
            sticky = true;
            PREFIX_TRUNCATIONS.fetch_add(1, Ordering::Relaxed);
            0
        };
        v.push(if s != 0 { -t } else { t });
    }
    PREFIX_TOTAL.fetch_add(x.len() as u64, Ordering::Relaxed);
    Plane { v, sticky }
}

#[inline(always)]
fn odd_chain(a: &[Wide; 128]) -> Wide {
    let mut s = Wide::ZERO;
    let mut t = Wide::ZERO;
    for j in (1..128).rev() {
        s.add(a[j]);
        t.add(s);
    }
    s.add(a[0]);
    s.add(t.shl(1));
    s
}

/// Planes per walk of a row's columns: the transcript positions that ride one read of the ids.
pub const PLANES: usize = 64;

/// The radix kernel for one output row against up to PLANES planes. Returns one pattern per plane.
#[inline(never)]
pub fn row_kernel(f: &TensorFold, row: &RowFold, planes: &[Plane], out: &mut [u16], lane: &mut Vec<[[Wide; 128]; 2]>) {
    let np = planes.len();
    debug_assert!(np >= 1 && np <= PLANES);
    let rspan = f.rspan;
    lane.clear();
    lane.resize(np, [[Wide::ZERO; 128]; 2]);
    let mut c0 = 0usize;
    let mut i = 0usize;
    while i < row.buckets.len() {
        let (key, cnt) = row.buckets[i];
        let key = key as usize;
        let sign = key / (128 * rspan);
        let odd = (key / rspan) % 128;
        let group_base = sign * 128 * rspan + odd * rspan;
        // walk this (sign, odd) group: buckets in descending rung; Horner with gaps
        let mut acc = [Wide::ZERO; PLANES];
        let mut prev_k: Option<usize> = None;
        let mut j = i;
        let mut c = c0;
        while j < row.buckets.len() {
            let (k2, n2) = row.buckets[j];
            let k2 = k2 as usize;
            if k2 / rspan != group_base / rspan {
                break;
            }
            let dk = k2 - group_base; // = rmax − rung, ascending as rung descends
            if let Some(pk) = prev_k {
                let gap = (dk - pk) as u32;
                for p in 0..np {
                    acc[p] = acc[p].shl(gap);
                }
            }
            let cols = &row.cols[c..c + n2 as usize];
            for p in 0..np {
                let pv = &planes[p].v;
                let mut s = 0i128;
                for &col in cols {
                    s += pv[col as usize];
                }
                acc[p].add(Wide::from_i128(s));
            }
            prev_k = Some(dk);
            c += n2 as usize;
            j += 1;
        }
        // land at scale ACC_FRAC: remaining doublings down to rmin
        let last_dk = prev_k.unwrap();
        let fin = (rspan - 1 - last_dk) as u32;
        for p in 0..np {
            lane[p][sign][odd] = acc[p].shl(fin);
        }
        let _ = cnt;
        c0 = c;
        i = j;
    }
    let frac = frac_for(f.rmin);
    for p in 0..np {
        let mut total = odd_chain(&lane[p][0]);
        total.sub(odd_chain(&lane[p][1]));
        out[p] = Bucket { acc: total, sticky: planes[p].sticky, frac }.collapse();
    }
}

/// `y = x · Wᵀ` over the fold: `x` is `[seq, in_f]` patterns, processed four planes per walk.
pub fn matmul_fold(x: &[u16], seq: usize, in_f: usize, f: &TensorFold) -> Vec<u16> {
    assert_eq!(x.len(), seq * in_f);
    assert_eq!(f.cols, in_f);
    let out_f = f.rows;
    let mut y = vec![0u16; seq * out_f];
    let planes_all: Vec<Plane> = (0..seq).map(|s| prefix(&x[s * in_f..(s + 1) * in_f])).collect();
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
    let n_threads = if out_f * in_f * seq < (1 << 18) { 1 } else { threads.min(out_f) };
    let yptr = SendPtr(y.as_mut_ptr());
    let cursor = AtomicUsize::new(0);
    let (cursor_ref, planes_ref) = (&cursor, &planes_all);
    let chunk = 16usize;
    let n_chunks = out_f.div_ceil(chunk);
    std::thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(move || {
                let yptr = yptr;
                let mut outbuf = [0u16; PLANES];
                let mut lane: Vec<[[Wide; 128]; 2]> = Vec::with_capacity(PLANES);
                loop {
                    let cidx = cursor_ref.fetch_add(1, Ordering::Relaxed);
                    if cidx >= n_chunks {
                        break;
                    }
                    for o in cidx * chunk..((cidx + 1) * chunk).min(out_f) {
                        let row = &f.rows_f[o];
                        let mut s0 = 0;
                        while s0 < seq {
                            let np = (seq - s0).min(PLANES);
                            row_kernel(f, row, &planes_ref[s0..s0 + np], &mut outbuf, &mut lane);
                            for p in 0..np {
                                // SAFETY: (s0+p, o) cells are disjoint across threads; scope joins before y is read.
                                unsafe { *yptr.0.add((s0 + p) * out_f + o) = outbuf[p] };
                            }
                            s0 += np;
                        }
                    }
                }
            });
        }
    });
    y
}
