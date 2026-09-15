//! `maelstrom.rs` — the number generation, as an object (pgarcia, 2026-09-13 afternoon).
//!
//! Four things, all integers, all adds and shifts (Euclid's division is the one exact divide):
//!
//! 1. The mediant tree (Stern-Brocot). Each row is minted from the previous by one integer add
//!    per numerator and one per denominator. Every positive rational appears exactly once, in
//!    lowest terms. Two neighbours a/b, c/d always satisfy b*c - a*d = 1 — the "one that is a
//!    generational rule". The largest number minted at depth k is F(k+2): the Fibonacci frontier.
//! 2. The two rays. phi as an integer pair (a, b) = a + b*phi with phi^2 = phi + 1. The pair
//!    coordinates of phi^n are (F(n-1), F(n)); the trace 2a + b is L(n). One object, two rays.
//!    Which integers are born on which ray, and the ladder 1..13 that closes only with both.
//! 3. The seam. Twin primes p, p+2 and the number between; inside the ladder each twin pair has
//!    one prime per ray; (3,5) is the only twin pair whose seam is not a multiple of 6 — it is 2^2.
//! 4. The bridge. Minkowski's question-mark function sends the mediant tree onto the dyadic tree:
//!    a rational's continued-fraction run lengths become runs of bits. Its inverse gives every
//!    cell <m|n> (value m * 2^n, a dyadic rational) a unique rational partner in the mediant tree,
//!    and the round trip is the gate. The odd side and the rung side are one tree read two ways.

use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

// ---- 1 + 2: sequences and the pair ring -------------------------------------------------------

pub fn fib(n: usize) -> u128 {
    let (mut a, mut b) = (0u128, 1u128);
    for _ in 0..n {
        let t = a + b;
        a = b;
        b = t;
    }
    a
}

pub fn lucas(n: usize) -> u128 {
    let (mut a, mut b) = (2u128, 1u128);
    for _ in 0..n {
        let t = a + b;
        a = b;
        b = t;
    }
    a
}

/// a + b*phi, phi^2 = phi + 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub a: i128,
    pub b: i128,
}

impl Pair {
    pub const ONE: Pair = Pair { a: 1, b: 0 };
    pub const PHI: Pair = Pair { a: 0, b: 1 };
    pub fn mul(self, o: Pair) -> Pair {
        Pair { a: self.a * o.a + self.b * o.b, b: self.a * o.b + self.b * o.a + self.b * o.b }
    }
    pub fn add(self, o: Pair) -> Pair {
        Pair { a: self.a + o.a, b: self.b + o.b }
    }
    /// Galois conjugate phi -> 1 - phi.
    pub fn conj(self) -> Pair {
        Pair { a: self.a + self.b, b: -self.b }
    }
    /// Tr(a + b phi) = 2a + b  (= self + conj, phi part zero).
    pub fn trace(self) -> i128 {
        2 * self.a + self.b
    }
    pub fn norm(self) -> i128 {
        self.a * self.a + self.a * self.b - self.b * self.b
    }
}

pub fn phi_pow(n: usize) -> Pair {
    let mut x = Pair::ONE;
    for _ in 0..n {
        x = x.mul(Pair::PHI);
    }
    x
}

pub fn is_prime(n: u128) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2u128;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

pub fn factors(mut n: u128) -> Vec<u128> {
    let mut out = Vec::new();
    let mut d = 2u128;
    while d * d <= n {
        while n % d == 0 {
            out.push(d);
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Which ray mints v: (Fibonacci index, Lucas index), values >= 2 only.
pub fn ray_of(v: u128) -> (Option<usize>, Option<usize>) {
    let mut f = None;
    let mut l = None;
    for i in 0..60 {
        if fib(i) == v && v >= 2 {
            f = Some(i);
        }
        if lucas(i) == v && v >= 2 {
            l = Some(i);
        }
    }
    (f, l)
}

/// Ray values <= cap: (coordinate ray = Fibonacci, trace ray = Lucas), values >= 2.
pub fn ray_values(cap: u128) -> (Vec<u128>, Vec<u128>) {
    let mut f = Vec::new();
    let mut l = Vec::new();
    for i in 0..60 {
        let (x, y) = (fib(i), lucas(i));
        if x >= 2 && x <= cap && !f.contains(&x) {
            f.push(x);
        }
        if y >= 2 && y <= cap && !l.contains(&y) {
            l.push(y);
        }
    }
    (f, l)
}

/// Licensed primes: primes on the used ray(s) plus the prime factors of on-ray composites.
pub fn licensed_primes(cap: u128, two_ray: bool) -> BTreeSet<u128> {
    let (f, l) = ray_values(cap);
    let mut vals = l;
    if two_ray {
        vals.extend(f);
    }
    let mut out = BTreeSet::new();
    for v in vals {
        for p in factors(v) {
            out.insert(p);
        }
    }
    out
}

/// The ladder: integers 1..=cap whose prime factors are all licensed.
pub fn ladder(cap: u128, two_ray: bool) -> Vec<u128> {
    let lic = licensed_primes(cap, two_ray);
    (1..=cap).filter(|&n| factors(n).iter().all(|p| lic.contains(p))).collect()
}

/// Twin primes (p, p+2) with p+2 <= cap: (p, p+2, seam, seam factors).
pub fn twin_seams(cap: u128) -> Vec<(u128, u128, u128, Vec<u128>)> {
    (3..cap - 1).filter(|&p| is_prime(p) && is_prime(p + 2)).map(|p| (p, p + 2, p + 1, factors(p + 1))).collect()
}

// ---- 1: the mediant tree -----------------------------------------------------------------------

/// Rows of the Stern-Brocot tree to `depth`: row k holds the fractions minted at depth k
/// (row 0 = 1/1). Also returns the final ordered sequence including the walls 0/1 and 1/0.
pub fn stern_brocot(depth: usize) -> (Vec<Vec<(u128, u128)>>, Vec<(u128, u128)>) {
    let mut seq: Vec<(u128, u128)> = vec![(0, 1), (1, 0)];
    let mut rows = Vec::new();
    for _ in 0..=depth {
        let mut next = Vec::with_capacity(seq.len() * 2);
        let mut row = Vec::new();
        for w in seq.windows(2) {
            let m = (w[0].0 + w[1].0, w[0].1 + w[1].1);
            next.push(w[0]);
            next.push(m);
            row.push(m);
        }
        next.push(*seq.last().unwrap());
        seq = next;
        rows.push(row);
    }
    (rows, seq)
}

/// Every pair of neighbours in the sequence has determinant 1.
pub fn neighbours_unimodular(seq: &[(u128, u128)]) -> bool {
    seq.windows(2).all(|w| w[0].1 * w[1].0 - w[0].0 * w[1].1 == 1)
}

pub fn row_max(row: &[(u128, u128)]) -> u128 {
    row.iter().map(|&(p, q)| p.max(q)).max().unwrap_or(0)
}

// ---- 4: the bridge -----------------------------------------------------------------------------

/// Continued fraction [a0; a1, ..., an] of p/q > 0 by Euclid, canonical (last term > 1 unless n = 0).
pub fn cf(p: &BigInt, q: &BigInt) -> Vec<BigInt> {
    let (mut p, mut q) = (p.clone(), q.clone());
    let mut out = Vec::new();
    while !q.is_zero() {
        let (d, r) = p.div_rem(&q);
        out.push(d);
        p = q;
        q = r;
    }
    out
}

pub fn from_cf(a: &[BigInt]) -> (BigInt, BigInt) {
    let (mut p0, mut q0) = (BigInt::one(), BigInt::zero());
    let (mut p1, mut q1) = (a[0].clone(), BigInt::one());
    for ai in &a[1..] {
        let p2 = ai * &p1 + &p0;
        let q2 = ai * &q1 + &q0;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
    }
    (p1, q1)
}

/// ?(p/q) as a dyadic num / 2^k, num odd or k = 0:
/// ?([a0; a1..an]) = a0 + sum_{i>=1} (-1)^(i+1) 2^(1 - (a1 + .. + ai)).
pub fn qmark(p: &BigInt, q: &BigInt) -> (BigInt, u32) {
    let a = cf(p, q);
    if a.len() == 1 {
        return (a[0].clone(), 0);
    }
    let sums: Vec<u32> = a[1..]
        .iter()
        .scan(0u32, |s, x| {
            *s += u32::try_from(x).unwrap();
            Some(*s)
        })
        .collect();
    let k = sums.last().unwrap() - 1;
    let mut num = &a[0] << k;
    for (i, s) in sums.iter().enumerate() {
        let term = BigInt::one() << (k - (s - 1));
        if i % 2 == 0 {
            num += term
        } else {
            num -= term
        }
    }
    reduce_dyadic(num, k)
}

fn reduce_dyadic(mut num: BigInt, mut k: u32) -> (BigInt, u32) {
    while k > 0 && (&num % 2u32).is_zero() {
        num >>= 1;
        k -= 1;
    }
    (num, k)
}

/// The inverse: the rational whose ? image is num / 2^k. Greedy on the alternating sum.
pub fn qmark_inv(num: &BigInt, k: u32) -> (BigInt, BigInt) {
    let (num, k) = reduce_dyadic(num.clone(), k);
    let a0 = &num >> k;
    let mut r: BigInt = &num - (&a0 << k);
    let mut a = vec![a0];
    let mut s_prev = 0u32;
    while !r.is_zero() {
        let bits = r.bits() as u32;
        let t = k - bits;
        let pow2 = (&r & (&r - 1u32)).is_zero();
        let s_minus_1 = if pow2 { t + 1 } else { t };
        let s_i = s_minus_1 + 1;
        a.push(BigInt::from(s_i - s_prev));
        s_prev = s_i;
        r = (BigInt::one() << (k - s_minus_1)) - r;
    }
    from_cf(&a)
}

/// The rational partner of the cell <m|n> (value m * 2^n): ?^-1 applied to the dyadic value.
pub fn cell_partner(m: u16, n: i32) -> (BigInt, BigInt) {
    if n >= 0 {
        return (BigInt::from(m) << (n as u32), BigInt::one());
    }
    qmark_inv(&BigInt::from(m), (-n) as u32)
}

/// The dyadic value of a cell as (num, k): m / 2^k with k = -n (k = 0 when n >= 0).
pub fn cell_dyadic(m: u16, n: i32) -> (BigInt, u32) {
    if n >= 0 {
        (BigInt::from(m) << (n as u32), 0)
    } else {
        reduce_dyadic(BigInt::from(m), (-n) as u32)
    }
}

/// Depth of a rational in the Stern-Brocot tree: sum of the continued-fraction terms minus 1.
/// A dyadic 1/2^k sits at depth 2^k - 1 on the leftmost path, so this is a BigInt, never a string.
pub fn tree_depth(p: &BigInt, q: &BigInt) -> BigInt {
    cf(p, q).iter().sum::<BigInt>() - 1
}

/// The frame of a rational in the tree: the matrix [[a, b], [c, d]] whose columns are its two
/// Farey parents a/c (left) and b/d (right), so the node is their mediant (a+b)/(c+d) and
/// b*c - a*d = 1. A step right replaces the left parent by the node (left += right); a step
/// left replaces the right parent (right += left). A run of n equal steps is one multiply-add,
/// so the frame costs O(terms) of the continued fraction, never a walk: the run of 2^k - 1 lefts
/// below 1/2^k is right = (1, 2^k - 1), one shift. This is the hyperobject: the depth 2^133 - 1
/// of the deepest cell is one evaluation (pgarcia: "it's geometry, a function").
pub fn tree_frame(p: &BigInt, q: &BigInt) -> [BigInt; 4] {
    let a = cf(p, q);
    // the root frame: parents 0/1 (left) and 1/0 (right), node 1/1
    let (mut a_, mut b_, mut c_, mut d_) = (BigInt::zero(), BigInt::one(), BigInt::one(), BigInt::zero());
    for (i, ai) in a.iter().enumerate() {
        let n = if i + 1 == a.len() { ai - 1 } else { ai.clone() };
        if i % 2 == 0 {
            // n steps right: left column += n * right column
            a_ += &n * &b_;
            c_ += &n * &d_;
        } else {
            // n steps left: right column += n * left column
            b_ += &n * &a_;
            d_ += &n * &c_;
        }
    }
    [a_, b_, c_, d_]
}

/// Stern-Brocot path of a rational from its continued fraction: R^a0 L^a1 R^a2 ... last run one
/// shorter. Only for shallow rationals (the caller checks `tree_depth` first).
pub fn tree_path(p: &BigInt, q: &BigInt) -> String {
    let a = cf(p, q);
    let mut s = String::new();
    for (i, ai) in a.iter().enumerate() {
        let mut n = u64::try_from(ai).unwrap();
        if i + 1 == a.len() {
            n -= 1;
        }
        let c = if i % 2 == 0 { 'R' } else { 'L' };
        for _ in 0..n {
            s.push(c);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_rays() {
        for n in 1..40 {
            let x = phi_pow(n);
            assert_eq!((x.a as u128, x.b as u128), (fib(n - 1), fib(n)), "coordinates are Fibonacci");
            assert_eq!(x.trace() as u128, lucas(n), "trace is Lucas");
            assert_eq!(x.add(x.conj()), Pair { a: x.trace(), b: 0 }, "conjugate part cancels");
            assert_eq!(x.norm(), if n % 2 == 0 { 1 } else { -1 }, "phi^n psi^n = (-1)^n");
        }
        let m5: Vec<u128> = (0..8).map(|n| lucas(n) % 5).collect();
        assert_eq!(m5, vec![2, 1, 3, 4, 2, 1, 3, 4], "Lucas mod 5: period 4");
    }

    #[test]
    fn ladder_closes_only_with_both_rays() {
        assert_eq!(ladder(13, false), vec![1, 2, 3, 4, 6, 7, 8, 9, 11, 12]);
        assert_eq!(ladder(13, true), (1..=13).collect::<Vec<_>>());
        let lic = licensed_primes(30, true);
        assert!(lic.contains(&5) && lic.contains(&13) && lic.contains(&7) && lic.contains(&11));
    }

    #[test]
    fn twins_straddle_inside_the_ladder() {
        let t = twin_seams(13);
        assert_eq!(t.iter().map(|x| (x.0, x.1, x.2)).collect::<Vec<_>>(), vec![(3, 5, 4), (5, 7, 6), (11, 13, 12)]);
        assert_eq!(t[0].3, vec![2, 2]);
        assert_eq!(t[2].3, vec![2, 2, 3]);
        for (p, q, _, _) in &t {
            let (fp, lp) = ray_of(*p);
            let (fq, lq) = ray_of(*q);
            assert!((fp.is_some() && lq.is_some()) || (lp.is_some() && fq.is_some()), "one prime per ray");
        }
        for (_, _, seam, _) in twin_seams(200).iter().skip(1) {
            assert_eq!(seam % 6, 0, "every seam after (3,5) is 6k");
        }
    }

    #[test]
    fn mediant_tree_frontier() {
        let (rows, seq) = stern_brocot(14);
        assert!(neighbours_unimodular(&seq));
        for (k, row) in rows.iter().enumerate() {
            assert_eq!(row_max(row), fib(k + 2), "row {} max", k);
            assert_eq!(row.len(), 1 << k);
        }
        let mut seen = BTreeSet::new();
        for (p, q) in &seq[1..seq.len() - 1] {
            assert_eq!(p.gcd(q), 1);
            assert!(seen.insert((*p, *q)), "each rational once");
        }
    }

    #[test]
    fn question_mark_bridge() {
        let b = |x: i64| BigInt::from(x);
        assert_eq!(qmark(&b(1), &b(2)), (b(1), 1));
        assert_eq!(qmark(&b(1), &b(3)), (b(1), 2));
        assert_eq!(qmark(&b(2), &b(3)), (b(3), 2));
        assert_eq!(qmark(&b(3), &b(4)), (b(7), 3));
        assert_eq!(qmark(&b(7), &b(2)), (b(7), 1));
        assert_eq!(qmark_inv(&b(3), 2), (b(2), b(3)));
        assert_eq!(qmark_inv(&b(7), 1), (b(7), b(2)));
        // the zigzag F(n)/F(n+1) maps to the alternating bits 0.1010.. truncated, i.e. toward 2/3
        for n in 2..20 {
            let (num, k) = qmark(&BigInt::from(fib(n)), &BigInt::from(fib(n + 1)));
            let diff: BigInt = &num * 3u32 - (BigInt::one() << (k + 1));
            assert!(diff.magnitude() <= BigInt::from(2).magnitude(), "3*?(F_n/F_n+1) = 2^(k+1) +- 1..2 at n={}", n);
        }
        // every cell has a partner and the round trip is exact
        for m in (1u16..=255).step_by(2) {
            for n in -140i32..=4 {
                let (p, q) = cell_partner(m, n);
                let (num, k) = cell_dyadic(m, n);
                assert_eq!(qmark(&p, &q), (num, k), "round trip <{}|{}>", m, n);
            }
        }
        assert_eq!(tree_path(&b(2), &b(3)), "LR");
        assert_eq!(tree_path(&b(3), &b(5)), "LRL");
    }

    #[test]
    fn frame_is_geometry_not_a_walk() {
        let b = |x: i64| BigInt::from(x);
        assert_eq!(tree_frame(&b(1), &b(1)), [b(0), b(1), b(1), b(0)]);
        assert_eq!(tree_frame(&b(1), &b(2)), [b(0), b(1), b(1), b(1)]);
        assert_eq!(tree_frame(&b(2), &b(3)), [b(1), b(1), b(2), b(1)]);
        // every cell: parents are Farey neighbours (det 1) and the node is their mediant
        for m in (1u16..=255).step_by(2) {
            for n in -140i32..=4 {
                let (p, q) = cell_partner(m, n);
                let [a, bb, c, d] = tree_frame(&p, &q);
                assert_eq!(&bb * &c - &a * &d, BigInt::one(), "det <{}|{}>", m, n);
                assert_eq!((&a + &bb, &c + &d), (p, q), "mediant <{}|{}>", m, n);
            }
        }
        // ? preserves depth: the k-bit dyadic 1/2^k partners with 1/(k+1), k deep, right parent 1/k
        let (p, q) = cell_partner(1, -133);
        assert_eq!((p.clone(), q.clone()), (b(1), b(134)));
        assert_eq!(tree_depth(&p, &q), b(133));
        assert_eq!(tree_frame(&p, &q), [b(0), b(1), b(1), b(133)]);
        // the deep wall is the integers: the largest cell 255 * 2^120 sits N - 1 deep on the
        // rightmost path, and its frame [[N-1, 1], [1, 0]] is one multiply-add, not a walk
        let (p, q) = cell_partner(255, 120);
        let big: BigInt = BigInt::from(255) << 120;
        assert_eq!((p.clone(), q.clone()), (big.clone(), b(1)));
        assert_eq!(tree_depth(&p, &q), &big - 1);
        assert_eq!(tree_frame(&p, &q), [&big - 1, b(1), b(1), b(0)]);
    }
}
