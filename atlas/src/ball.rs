//! `ball.rs` — the function (GOAL.md step 1):
//!
//! ```text
//! BALL(a, k) = Rot(q)^k · R(a)
//! ```
//!
//! - a cell of the model is a point: (place, ±odd, rung) — the three surfaces as the three
//!   axes; place = the cell's offset in the Morton-tiled grid, odd = the core m with the sign
//!   beside it, rung = n.
//! - Rot(q): the tick. q one of the six Hurwitz quaternions (cubball::SPIN_Q), applied as the
//!   integer sandwich q·v·q̄ — NEVER divided; the scale |q|²ᵏ rides as the frame (cubball,
//!   "rotation_scale_is_the_prefix_product"). Exactly orthogonal: Mᵀ M = |q|⁴·I as integers.
//! - R(a): the renderer, tensor_mesh.torus_params — t = y/(ρ+x), u = z/(σ+ρ), ρ and σ declared
//!   at `digits` by integer square root; homogeneous of degree 0, so the frame scale cancels
//!   exactly; cell_of(t, u, W, H). Big integers throughout (num-bigint); no float.
//! - O(cell): the observer, the inverse renderer. tight: R(a) ∈ C ⇔ a ∈ O(C).
//!
//! Gates: Mᵀ M = |q|⁴ I; the periodicity census (q^k real for some k ≤ bound?); tight() over
//! every cell of one tensor at k = 0 and at k ticks.

use std::collections::HashMap;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use crate::cells;
use crate::surfaces::{Atoms, Grid};

/// The six Hurwitz seals of the ball, norm = the axis prime (cubball.rs::SPIN_Q).
pub const SPIN_Q: [[i64; 4]; 6] = [[1, 1, 0, 0], [1, 1, 1, 0], [2, 1, 0, 0], [2, 1, 1, 1], [3, 1, 1, 0], [3, 2, 0, 0]];
pub const PRIMES: [i64; 6] = [2, 3, 5, 7, 11, 13];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quat {
    pub w: BigInt,
    pub x: BigInt,
    pub y: BigInt,
    pub z: BigInt,
}

impl Quat {
    pub fn from_i64(q: [i64; 4]) -> Quat {
        Quat { w: q[0].into(), x: q[1].into(), y: q[2].into(), z: q[3].into() }
    }
    pub fn one() -> Quat {
        Quat { w: BigInt::one(), x: BigInt::zero(), y: BigInt::zero(), z: BigInt::zero() }
    }
    pub fn mul(&self, b: &Quat) -> Quat {
        let a = self;
        Quat {
            w: &a.w * &b.w - &a.x * &b.x - &a.y * &b.y - &a.z * &b.z,
            x: &a.w * &b.x + &a.x * &b.w + &a.y * &b.z - &a.z * &b.y,
            y: &a.w * &b.y - &a.x * &b.z + &a.y * &b.w + &a.z * &b.x,
            z: &a.w * &b.z + &a.x * &b.y - &a.y * &b.x + &a.z * &b.w,
        }
    }
    pub fn conj(&self) -> Quat {
        Quat { w: self.w.clone(), x: -&self.x, y: -&self.y, z: -&self.z }
    }
    pub fn norm(&self) -> BigInt {
        &self.w * &self.w + &self.x * &self.x + &self.y * &self.y + &self.z * &self.z
    }
    pub fn pow(&self, k: u32) -> Quat {
        let mut r = Quat::one();
        for _ in 0..k {
            r = r.mul(self);
        }
        r
    }
    pub fn is_real(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }
    /// The integer rotation matrix M(q) = |q|²·R(q).
    pub fn matrix(&self) -> [[BigInt; 3]; 3] {
        let (w, x, y, z) = (&self.w, &self.x, &self.y, &self.z);
        let two = BigInt::from(2);
        [
            [w * w + x * x - y * y - z * z, &two * (x * y - w * z), &two * (x * z + w * y)],
            [&two * (x * y + w * z), w * w - x * x + y * y - z * z, &two * (y * z - w * x)],
            [&two * (x * z - w * y), &two * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    }
    /// The sandwich q·v·q̄ on a pure vector: exact, scaled by |q|².
    pub fn rotate(&self, v: &[BigInt; 3]) -> [BigInt; 3] {
        let pv = Quat { w: BigInt::zero(), x: v[0].clone(), y: v[1].clone(), z: v[2].clone() };
        let r = self.mul(&pv).mul(&self.conj());
        debug_assert!(r.w.is_zero());
        [r.x, r.y, r.z]
    }
}

/// Gate: Mᵀ M = |q|⁴ · I exactly.
pub fn orthogonal(q: &Quat) -> bool {
    let m = q.matrix();
    let n2 = q.norm().pow(2);
    for i in 0..3 {
        for j in 0..3 {
            let s: BigInt = (0..3).map(|k| &m[k][i] * &m[k][j]).sum();
            let want = if i == j { n2.clone() } else { BigInt::zero() };
            if s != want {
                return false;
            }
        }
    }
    true
}

/// Periodicity census: the smallest k in 1..=bound with q^k real, if any.
pub fn period(q: &Quat, bound: u32) -> Option<u32> {
    let mut r = Quat::one();
    for k in 1..=bound {
        r = r.mul(q);
        if r.is_real() {
            return Some(k);
        }
    }
    None
}

// ── the renderer ─────────────────────────────────────────────────────────────

pub type Pt = [BigInt; 3];

/// ⌊√(a² + b²) · 10^digits⌋ as an integer over 10^digits — the declared magnitude.
fn mag_declared(a: &BigInt, b: &BigInt, scale: &BigInt) -> BigInt {
    let n2 = (a * a + b * b) * scale * scale;
    n2.sqrt()
}

/// Half-angle torus parameters as exact rationals (num, den), den > 0, folded to [−1, 1].
/// Coordinates relative to the center are homogeneous: any common scale cancels.
pub fn torus_params(p: &Pt, c: &Pt, digits: u32) -> ((BigInt, BigInt), (BigInt, BigInt)) {
    let scale = BigInt::from(10).pow(digits);
    let x = &p[0] - &c[0];
    let y = &p[1] - &c[1];
    let z = &p[2] - &c[2];
    // ρ = mag(x, y)/scale ; t = y / (ρ + x) = y·scale / (mag + x·scale)
    let rho_s = mag_declared(&x, &y, &scale); // ρ·scale
    let t_den = &rho_s + &x * &scale;
    let t = if t_den.is_zero() { (BigInt::zero(), BigInt::one()) } else { (&y * &scale, t_den) };
    // σ = mag(ρ, z)/scale ; u = z / (σ + ρ) — ρ and z at the same scale: σ_s = mag(rho_s, z·scale)
    let z_s = &z * &scale;
    let sigma_s = mag_declared(&rho_s, &z_s, &BigInt::one());
    let u_den = &sigma_s + &rho_s;
    let u = if u_den.is_zero() { (BigInt::zero(), BigInt::one()) } else { (z_s, u_den) };
    (fold(t), fold(u))
}

/// Fold |t| > 1 to 1/t; normalize the sign to the denominator.
fn fold((n, d): (BigInt, BigInt)) -> (BigInt, BigInt) {
    let (mut n, mut d) = if d.is_negative() { (-n, -d) } else { (n, d) };
    if n.abs() > d {
        let (a, b) = (d, n);
        n = a;
        d = b;
        if d.is_negative() {
            n = -n;
            d = -d;
        }
    }
    (n, d)
}

/// cell_of: i = ⌊(t + 1)·W/2⌋ with t = n/d exact.
pub fn cell_of(t: &(BigInt, BigInt), u: &(BigInt, BigInt), w: usize, h: usize) -> (usize, usize) {
    let idx = |(n, d): &(BigInt, BigInt), size: usize| -> usize {
        // floor(((n + d) · size) / (2 d))
        let num = (n + d) * BigInt::from(size);
        let den = d * BigInt::from(2);
        let q = num.div_floor(&den);
        let q: i64 = q.try_into().unwrap_or(i64::MAX);
        (q.max(0) as usize).min(size - 1)
    };
    (idx(t, w), idx(u, h))
}

/// The points of one tensor: (place, ±m, n) per cell, in file order.
pub fn points_of(g: &Grid, atoms: &Atoms) -> Vec<Pt> {
    let mut out = Vec::with_capacity(g.ids.len());
    for (place, &id) in g.ids.iter().enumerate() {
        let id = id as usize;
        let m = atoms.m[id] as i64;
        let signed_m = if atoms.sheet[id] != 0 { -m } else { m };
        out.push([BigInt::from(place as i64), BigInt::from(signed_m), BigInt::from(atoms.n[id] as i64)]);
    }
    out
}

/// The frame is measured, not chosen: centroid (as num/den) and the axis of largest second moment.
/// Returns the center scaled by `n` (so it is an integer) and the permutation; callers scale points by n.
pub fn measure_frame(points: &[Pt]) -> (Pt, BigInt, [usize; 3]) {
    let n = BigInt::from(points.len() as i64);
    let mut sum = [BigInt::zero(), BigInt::zero(), BigInt::zero()];
    for p in points {
        for k in 0..3 {
            sum[k] += &p[k];
        }
    }
    // second moments about the centroid, scaled by n²: Σ (n·p − sum)²
    let mut m2 = [BigInt::zero(), BigInt::zero(), BigInt::zero()];
    for p in points {
        for k in 0..3 {
            let d = &p[k] * &n - &sum[k];
            m2[k] += &d * &d;
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| m2[b].cmp(&m2[a]).then(a.cmp(&b)));
    let perm = [order[2], order[1], order[0]];
    (sum, n, perm)
}

pub struct Mesh {
    pub w: usize,
    pub h: usize,
    pub digits: u32,
    pub cells: HashMap<(usize, usize), Vec<u32>>,
    pub where_: Vec<(usize, usize)>,
}

impl Mesh {
    /// R over every point: render into cells at max resolution. Points are given already in the
    /// measured frame (reframed and scaled so the center is an integer).
    pub fn chunk(points: &[Pt], center: &Pt, w: usize, h: usize, digits: u32) -> Mesh {
        let mut cells: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        let mut where_ = Vec::with_capacity(points.len());
        for (aid, p) in points.iter().enumerate() {
            let (t, u) = torus_params(p, center, digits);
            let c = cell_of(&t, &u, w, h);
            cells.entry(c).or_default().push(aid as u32);
            where_.push(c);
        }
        Mesh { w, h, digits, cells, where_ }
    }
    pub fn render(&self, aid: usize, level: u32) -> (usize, usize) {
        let (i, j) = self.where_[aid];
        (i >> level, j >> level)
    }
    pub fn observe(&self, cell: (usize, usize), level: u32) -> Vec<u32> {
        let k = 1usize << level;
        let mut out = Vec::new();
        for di in 0..k {
            for dj in 0..k {
                if let Some(v) = self.cells.get(&(cell.0 * k + di, cell.1 * k + dj)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
    }
    /// tight: every atom is observed where it renders, and every observed atom renders there.
    pub fn tight(&self, level: u32) -> bool {
        // atoms were pushed in increasing id order, so each cell's list is sorted
        for aid in 0..self.where_.len() {
            let c0 = self.where_[aid];
            let Some(list) = self.cells.get(&c0) else { return false };
            if list.binary_search(&(aid as u32)).is_err() {
                return false;
            }
            let _ = level; // the pooled cell contains its children by construction of `observe`
        }
        let k = 1usize << level;
        let pooled: std::collections::HashSet<(usize, usize)> = self.cells.keys().map(|&(i, j)| (i / k, j / k)).collect();
        for c in pooled {
            for aid in self.observe(c, level) {
                if self.render(aid as usize, level) != c {
                    return false;
                }
            }
        }
        true
    }
    /// The pooled tensor at a level: cell → count.
    pub fn at(&self, level: u32) -> HashMap<(usize, usize), usize> {
        let mut out = HashMap::new();
        for (&(i, j), v) in &self.cells {
            *out.entry((i >> level, j >> level)).or_insert(0) += v.len();
        }
        out
    }
    pub fn silhouette(&self, level: u32) -> String {
        let chars: Vec<char> = " .:-=+*#%@".chars().collect();
        let t = self.at(level);
        let (w, h) = (self.w >> level, self.h >> level);
        let mx = t.values().copied().max().unwrap_or(1).max(1);
        let mut rows = Vec::new();
        for j in (0..h).rev() {
            let row: String = (0..w)
                .map(|i| {
                    let c = t.get(&(i, j)).copied().unwrap_or(0);
                    chars[((c * (chars.len() - 1) + mx - 1) / mx).min(chars.len() - 1)]
                })
                .collect();
            rows.push(row);
        }
        rows.join("\n")
    }
    /// Receipt of the frame: the sorted (cell, count) table as bytes.
    pub fn table_bytes(&self, level: u32) -> Vec<u8> {
        let t = self.at(level);
        let mut v: Vec<((usize, usize), usize)> = t.into_iter().collect();
        v.sort();
        let mut out = Vec::with_capacity(v.len() * 12);
        for ((i, j), c) in v {
            out.extend_from_slice(&(i as u32).to_le_bytes());
            out.extend_from_slice(&(j as u32).to_le_bytes());
            out.extend_from_slice(&(c as u32).to_le_bytes());
        }
        out
    }
}

/// BALL(a, k): rotate every point k ticks of q (exact, scaled by |q|^{2k}) about the measured
/// center, then render. Returns the mesh and the scale carried as the frame.
pub fn ball(points: &[Pt], q: &Quat, k: u32, w: usize, h: usize, digits: u32) -> (Mesh, BigInt) {
    let (sum, n, perm) = measure_frame(points);
    let qk = q.pow(k);
    let scale = qk.norm(); // |q|^{2k}
    // reframe: p' = n·p − sum (center at the origin, integer), permuted so the hole's axis is z
    let mut rotated: Vec<Pt> = Vec::with_capacity(points.len());
    for p in points {
        let v = [&p[perm[0]] * &n - &sum[perm[0]], &p[perm[1]] * &n - &sum[perm[1]], &p[perm[2]] * &n - &sum[perm[2]]];
        rotated.push(if k == 0 { v } else { qk.rotate(&v) });
    }
    let zero: Pt = [BigInt::zero(), BigInt::zero(), BigInt::zero()];
    (Mesh::chunk(&rotated, &zero, w, h, digits), scale)
}

/// The cell-based `is_finite` is not needed here; keep the module's dependency explicit.
#[allow(dead_code)]
fn _cells_link() -> usize {
    cells::FINITE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seals_are_orthogonal_and_norms_are_the_primes() {
        for (i, q) in SPIN_Q.iter().enumerate() {
            let q = Quat::from_i64(*q);
            assert_eq!(q.norm(), BigInt::from(PRIMES[i]));
            assert!(orthogonal(&q));
        }
    }

    #[test]
    fn only_the_norm_two_tick_closes() {
        let periods: Vec<Option<u32>> = SPIN_Q.iter().map(|q| period(&Quat::from_i64(*q), 512)).collect();
        assert_eq!(periods[0], Some(4)); // (1+i)^4 = -4: a quarter turn, four ticks close
        for p in &periods[1..] {
            assert_eq!(*p, None); // odd-prime ticks never close: the door refuses a radial turn
        }
    }

    #[test]
    fn sandwich_matches_matrix() {
        let q = Quat::from_i64([2, 1, 1, 1]);
        let v = [BigInt::from(3), BigInt::from(-4), BigInt::from(12)];
        let r = q.rotate(&v);
        let m = q.matrix();
        for i in 0..3 {
            let s: BigInt = (0..3).map(|k| &m[i][k] * &v[k]).sum();
            assert_eq!(r[i], s);
        }
        // |q v q̄|² = |q|⁴ |v|²
        let n_r: BigInt = r.iter().map(|c| c * c).sum();
        let n_v: BigInt = v.iter().map(|c| c * c).sum();
        assert_eq!(n_r, q.norm().pow(2) * n_v);
    }

    #[test]
    fn tight_on_a_small_cloud() {
        let pts: Vec<Pt> = (0..500i64).map(|i| [BigInt::from(i), BigInt::from((i * 7) % 255 - 127), BigInt::from((i * 3) % 40 - 20)]).collect();
        let q = Quat::from_i64([1, 1, 1, 0]);
        for k in [0u32, 1, 5] {
            let (mesh, _) = ball(&pts, &q, k, 64, 64, 12);
            assert!(mesh.tight(0), "tight fails at k={}", k);
            assert!(mesh.tight(2));
        }
    }
}
