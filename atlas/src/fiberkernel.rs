//! `fiberkernel.rs` — the kernel that walks positions (LOG §17; pgarcia: "separate id and
//! dictionary; 1/value = id; triangulation and selection, no calculation at runtime").
//!
//! The id side: every atom of the dictionary is a point (sign, k, n) of the bundle — k its
//! position on the 16-bit torus (m = ±5^k mod 2^16), n its rung. The activation row is decomposed
//! to points once per token. A term is then: signs xor, positions ADD, the odd product SELECTED
//! from the antilog (exact, the product of two 8-bit odds is a 16-bit residue), rungs add, one
//! shift, one add into the same 256-bit lane as the bucket. No multiply anywhere in the walk.
//! Same terms in the same order as `matmul::matmul_grid`, so the logits are bit-identical by
//! construction; gate 5 judges it like every kernel. `ATLAS_KERNEL=fiber`.

use crate::fiber::Fiber;
use crate::matmul::{frac_for, run_cells, Bucket, XRow};
use crate::surfaces::{Atoms, Grid};

/// The dictionary read as positions: one point per atom id.
pub struct FiberAtoms {
    /// the value's sign (beside the cell)
    pub sign: Vec<u8>,
    /// the residue's sign: m = (-1)^neg · 5^k mod 2^16
    pub neg: Vec<u8>,
    pub k: Vec<u32>,
    pub n: Vec<i32>,
    /// position -> odd residue mod 2^16 (the value side, selected once per term)
    pub antilog: Vec<u32>,
}

impl FiberAtoms {
    pub fn from_atoms(atoms: &Atoms) -> FiberAtoms {
        let f = Fiber::new(16);
        let n_ids = atoms.m.len();
        let mut out = FiberAtoms { sign: Vec::with_capacity(n_ids), neg: Vec::with_capacity(n_ids), k: Vec::with_capacity(n_ids), n: Vec::with_capacity(n_ids), antilog: f.antilog.clone() };
        for id in 0..n_ids {
            let m = atoms.m[id] as u32;
            assert!(m != 0, "fiber kernel refused: atom {} is an exact zero and the torus has no zero", id);
            let (s, k) = f.position(m);
            out.sign.push(atoms.sheet[id]);
            out.neg.push(s as u8);
            out.k.push(k);
            out.n.push(atoms.n[id] as i32);
        }
        out
    }
    pub fn bytes(&self) -> usize {
        self.sign.len() * (1 + 1 + 4 + 4) + self.antilog.len() * 4
    }
}

/// One activation row as points (zero cells flagged; the floor and sticky as in `XRow`).
pub struct XPos {
    pub sign: Vec<u8>,
    pub neg: Vec<u8>,
    pub k: Vec<u32>,
    pub n: Vec<i32>,
    pub zero: Vec<bool>,
    pub sticky: bool,
}

impl XPos {
    pub fn new(x: &[u16], f: &Fiber) -> XPos {
        let r = XRow::new(x);
        let len = r.m.len();
        let mut p = XPos { sign: Vec::with_capacity(len), neg: Vec::with_capacity(len), k: Vec::with_capacity(len), n: r.n.clone(), zero: Vec::with_capacity(len), sticky: r.sticky };
        for i in 0..len {
            if r.m[i] == 0 {
                p.sign.push(r.sheet[i]);
                p.neg.push(0);
                p.k.push(0);
                p.zero.push(true);
            } else {
                let (s, k) = f.position(r.m[i]);
                p.sign.push(r.sheet[i]);
                p.neg.push(s as u8);
                p.k.push(k);
                p.zero.push(false);
            }
        }
        p
    }
}

/// `y = x · Wᵀ` over the grid, walking positions. Bit-identical to `matmul_grid`.
pub fn matmul_positions(x: &[u16], seq: usize, in_f: usize, w: &Grid, fa: &FiberAtoms) -> Vec<u16> {
    assert_eq!(x.len(), seq * in_f, "matmul_positions: x is not [seq, in_f]");
    assert_eq!(w.cols, in_f, "matmul_positions: grid cols {} != in_f {}", w.cols, in_f);
    let out_f = w.rows;
    let frac = frac_for(w.rmin);
    let f = Fiber::new(16);
    let rows: Vec<XPos> = (0..seq).map(|s| XPos::new(&x[s * in_f..(s + 1) * in_f], &f)).collect();
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
                if xr.zero[i] {
                    continue;
                }
                let kk = (xr.k[i] + fa.k[id]) & 0x3FFF;
                let r = fa.antilog[kk as usize] as i32;
                // the residue's sign selects between 5^k and 2^16 - 5^k, branchless: r + (65536 - 2r) & mask
                let mask = -((xr.neg[i] ^ fa.neg[id]) as i32);
                let prod = (r + ((65536 - 2 * r) & mask)) as u32;
                b.add_product(xr.sign[i] ^ fa.sign[id], prod, xr.n[i] + fa.n[id]);
            }
        }
        b.collapse()
    };
    let mut y = vec![0u16; seq * out_f];
    run_cells(cell, in_f, &mut y);
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells;
    use crate::matmul::matmul_grid;
    use crate::surfaces::Place;
    use std::sync::Arc;

    #[test]
    fn positions_reproduce_the_bucket_bit_for_bit() {
        // a dictionary of 512 finite non-zero patterns in a sane binade box, a 256 x 256 grid, 3 activations
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut dict = Vec::new();
        while dict.len() < 512 {
            let sign = ((next() & 1) as u16) << 15;
            let exp = (110 + (next() % 30)) as u16; // 2^-17 .. 2^12
            let frac = (next() % 128) as u16;
            let p = sign | (exp << 7) | frac;
            if cells::is_finite(p) && !dict.contains(&p) {
                dict.push(p);
            }
        }
        let atoms = Atoms::from_dict(&dict);
        let (rows, cols) = (256usize, 256usize);
        let ids: Vec<u16> = (0..rows * cols).map(|_| (next() % 512) as u16).collect();
        let (mut rmin, mut rmax) = (i32::MAX, i32::MIN);
        for &id in &ids {
            rmin = rmin.min(atoms.n[id as usize] as i32);
            rmax = rmax.max(atoms.n[id as usize] as i32);
        }
        let grid = Grid { rows, cols, th: 128, tw: 128, ids, place: Arc::new(Place::new(2, 2)), rmin, rmax };
        let x: Vec<u16> = (0..3 * cols).map(|_| dict[(next() % 512) as usize]).collect();
        let fa = FiberAtoms::from_atoms(&atoms);
        let a = matmul_grid(&x, 3, cols, &grid, &atoms);
        let b = matmul_positions(&x, 3, cols, &grid, &fa);
        assert_eq!(a, b, "positions vs bucket");
        assert!(a.iter().any(|&v| v != 0));
    }
}
