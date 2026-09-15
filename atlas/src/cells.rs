//! The bf16 pattern as a cell ⟨m|n⟩ on its sheet — the Rust twin of `intake/cells.py`.
//!
//! pattern = sheet(1) | exp(8) | mant(7). Normals carry the implicit 1: sig = 128|mant;
//! subnormals sig = mant. tz = trailing zeros of sig; m = sig >> tz (odd, or 0 for zero);
//! n = (exp − 134) + tz for normals, (1 − 134) + tz for subnormals. value = ±m·2^n.
//! Both names of zero (±0) are kept as (sheet, 0, 0). Non-finite patterns have no cell.

/// Number of finite bf16 patterns: 65,536 − 256 (exp == 255).
pub const FINITE: usize = 65_280;

#[inline(always)]
pub fn is_finite(p: u16) -> bool {
    ((p >> 7) & 0xFF) != 0xFF
}

/// pattern → (sheet, odd core m, rung n). Caller guarantees `is_finite`.
#[inline(always)]
pub fn decompose(p: u16) -> (u8, u8, i16) {
    let sheet = (p >> 15) as u8;
    let exp = ((p >> 7) & 0xFF) as i32;
    let mant = (p & 0x7F) as i32;
    let (sig, base) = if exp != 0 { (mant | 0x80, exp - 127 - 7) } else { (mant, 1 - 127 - 7) };
    if sig == 0 {
        return (sheet, 0, 0);
    }
    let tz = sig.trailing_zeros() as i32;
    (sheet, (sig >> tz) as u8, (base + tz) as i16)
}

/// (sheet, m, n) → pattern. Exact inverse of `decompose` on every finite pattern.
pub fn recompose(sheet: u8, m: u8, n: i16) -> u16 {
    if m == 0 {
        return (sheet as u16) << 15;
    }
    let hibit = 7 - m.leading_zeros() as i32; // 0..7
    let sig = (m as i32) << (7 - hibit); // 128..255
    let rung_sig = n as i32 - (7 - hibit);
    let exp = rung_sig + 7 + 127;
    if exp >= 1 {
        ((sheet as u16) << 15) | ((exp as u16) << 7) | ((sig & 0x7F) as u16)
    } else {
        let shift = (1 - exp).min(8);
        ((sheet as u16) << 15) | ((sig >> shift) as u16)
    }
}

/// The binade of a cell: floor(log2 |value|) = n + hibit(m). Meaningless for m = 0.
#[inline(always)]
pub fn binade(m: u8, n: i16) -> i16 {
    n + (7 - m.leading_zeros() as i16)
}

/// Gate 2: every finite pattern round-trips. Returns (checked, exact).
pub fn census() -> (usize, usize) {
    let mut checked = 0;
    let mut exact = 0;
    for p in 0..=u16::MAX {
        if !is_finite(p) {
            continue;
        }
        checked += 1;
        let (s, m, n) = decompose(p);
        if recompose(s, m, n) == p {
            exact += 1;
        }
    }
    (checked, exact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_is_exact() {
        assert_eq!(census(), (FINITE, FINITE));
    }

    #[test]
    fn spot_cells() {
        assert_eq!(decompose(0x3F80), (0, 1, 0)); // 1.0
        assert_eq!(decompose(0xC000), (1, 1, 1)); // -2.0
        assert_eq!(decompose(0x3E20), (0, 5, -5)); // 0.15625 = 5/32
        assert_eq!(decompose(0x0001), (0, 1, -133)); // smallest subnormal
        assert_eq!(decompose(0x8000), (1, 0, 0)); // -0
        assert_eq!(decompose(0x4040), (0, 3, 0)); // 3.0
        assert_eq!(decompose(0x7F7F), (0, 255, 120)); // max finite
        assert_eq!(binade(5, -5), -3);
    }

    #[test]
    fn constants_used_by_the_forward() {
        // eps = 1e-6 → nearest bf16 is (1 + 6/128)·2^-20 = pattern 0x3586
        assert_eq!(decompose(0x3586), (0, 67, -26)); // 134/128 · 2^-20 = 67 · 2^-26
        // attn scale = 1/sqrt(128) → nearest bf16 is (1 + 53/128)·2^-4 = pattern 0x3DB5
        assert_eq!(decompose(0x3DB5), (0, 181, -11)); // 181/128 · 2^-4 = 181 · 2^-11
    }
}
