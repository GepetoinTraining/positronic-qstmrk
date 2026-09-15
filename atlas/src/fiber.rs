//! `fiber.rs` — the 2-adic hyper object's fibers to the integers (pgarcia, 2026-09-13 ~17:00).
//!
//! The odd residues mod 2^N form the group {±1} × Z/2^(N-2): every odd m is ±5^k for exactly one
//! sign and one k mod 2^(N-2). So an odd is a POSITION (sign, k) on a torus, and a cell <m|n> is
//! (sign, k, n): the fiber over the integer is the rung ladder, the position along the torus is k.
//! Products of values are sums of positions: m_x * m_w = ±5^(k_x + k_w), 1/m = 5^(-k) — "1/value
//! = id". The two maps are kept apart: the log (odd -> position) and the antilog (position ->
//! odd), the id side and the dictionary side. A product of two 8-bit odds is exact as a 16-bit
//! residue, so positions are taken mod 2^14 on the 16-bit torus; the 8-bit torus (k mod 64) is
//! the 128 atoms themselves. Nothing here computes a product: positions add, the value is selected.

/// The torus of odd residues mod 2^bits: antilog[k] = 5^k mod 2^bits (k mod 2^(bits-2)), and
/// log[m >> 1] = (negative, k) with m = (-1)^negative * 5^k mod 2^bits.
pub struct Fiber {
    pub bits: u32,
    pub antilog: Vec<u32>,
    pub log: Vec<(bool, u32)>,
}

impl Fiber {
    pub fn new(bits: u32) -> Fiber {
        assert!((3..=30).contains(&bits));
        let modulus: u64 = 1u64 << bits;
        let period = 1usize << (bits - 2);
        let mut antilog = Vec::with_capacity(period);
        let mut log = vec![(false, u32::MAX); 1usize << (bits - 1)];
        let mut v: u64 = 1;
        for k in 0..period {
            antilog.push(v as u32);
            log[(v as usize) >> 1] = (false, k as u32);
            let neg = (modulus - v) as usize;
            log[neg >> 1] = (true, k as u32);
            v = (v * 5) % modulus;
        }
        assert!(log.iter().all(|&(_, k)| k != u32::MAX), "every odd residue has a position");
        Fiber { bits, antilog, log }
    }

    #[inline(always)]
    pub fn period(&self) -> u32 {
        1u32 << (self.bits - 2)
    }

    /// Position of an odd m (as a residue mod 2^bits): (sign, k).
    #[inline(always)]
    pub fn position(&self, m: u32) -> (bool, u32) {
        debug_assert!(m & 1 == 1 && (m as u64) < (1u64 << self.bits));
        self.log[(m as usize) >> 1]
    }

    /// The odd residue at a position.
    #[inline(always)]
    pub fn value(&self, neg: bool, k: u32) -> u32 {
        let v = self.antilog[(k & (self.period() - 1)) as usize];
        if neg {
            ((1u64 << self.bits) - v as u64) as u32
        } else {
            v
        }
    }

    /// Position of the product: signs xor, positions add.
    #[inline(always)]
    pub fn mul(&self, a: (bool, u32), b: (bool, u32)) -> (bool, u32) {
        (a.0 ^ b.0, (a.1 + b.1) & (self.period() - 1))
    }

    /// Position of the inverse: 1/value = -k.
    #[inline(always)]
    pub fn inv(&self, a: (bool, u32)) -> (bool, u32) {
        (a.0, (self.period() - a.1) & (self.period() - 1))
    }
}

/// A cell <m|n> with sign as a point of the fiber bundle: (sign, k on the 16-bit torus, rung n).
#[inline(always)]
pub fn cell_position(f: &Fiber, negative: bool, m: u8, n: i16) -> (bool, u32, i16) {
    let (s, k) = f.position(m as u32);
    (negative ^ s, k, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_128_atoms_are_the_8_bit_torus() {
        let f = Fiber::new(8);
        assert_eq!(f.period(), 64);
        let mut seen = std::collections::BTreeSet::new();
        for m in (1u32..256).step_by(2) {
            let (s, k) = f.position(m);
            assert!(k < 64);
            assert!(seen.insert((s, k)), "one position per odd");
            assert_eq!(f.value(s, k), m, "antilog(log m) = m");
        }
        assert_eq!(seen.len(), 128);
        assert_eq!(f.position(1), (false, 0));
        assert_eq!(f.position(255), (true, 0), "255 = -1 mod 256");
        assert_eq!(f.position(5), (false, 1));
        assert_eq!(f.position(3), (true, 35), "3 = -5^35 mod 256 (5^35 = 253)");
    }

    #[test]
    fn products_are_position_sums_exactly() {
        let f = Fiber::new(16);
        for mx in (1u32..256).step_by(2) {
            for mw in (1u32..256).step_by(2) {
                let px = f.position(mx);
                let pw = f.position(mw);
                let (s, k) = f.mul(px, pw);
                assert_eq!(f.value(s, k), mx * mw, "{} * {}", mx, mw);
            }
        }
    }

    #[test]
    fn one_over_value_is_minus_k() {
        let f = Fiber::new(16);
        for m in (1u32..256).step_by(2) {
            let p = f.position(m);
            let (s, k) = f.inv(p);
            let inv = f.value(s, k) as u64;
            assert_eq!((inv * m as u64) % (1 << 16), 1, "m * (1/m) = 1 mod 2^16 for m = {}", m);
        }
    }

    #[test]
    fn every_finite_cell_is_a_point_of_the_bundle() {
        let f = Fiber::new(16);
        let mut seen = std::collections::BTreeSet::new();
        let mut count = 0;
        for p in 0u16..=0xFFFF {
            let e = (p >> 7) & 0xFF;
            if e == 0xFF || (p & 0x7FFF) == 0 {
                continue;
            }
            let (sheet, m, n) = crate::cells::decompose(p);
            let pt = cell_position(&f, sheet == 1, m, n);
            assert!(seen.insert(pt), "distinct point per pattern {:#06x}", p);
            count += 1;
        }
        assert_eq!(count, 65278);
    }
}
