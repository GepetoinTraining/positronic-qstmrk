//! `clock.rs` — the two clocks (handshake-paper.pdf, 2026-07-20; pgarcia 2026-09-13 ~09:10).
//!
//! The acquisition clock stamps every position of the first inference with its residues on
//! coprime rings — the phase bank of V5.1's `phase_kernel` (one state word per prime, driven at
//! 773, forward is add + conditional subtract, backward exact, no history). No single cadence
//! can resonance-lock the addressing: a lock on any one ring is toured out by the others.
//! The three rings the paper names, 7·11·13 = 1001, give the projected fourth axis by CRT: a
//! coordinate, not a gear, free at no cost. The runtime clock carries nothing and asks the
//! checker (`cache.rs`).
//!
//! Integer only.

pub const DRIVE: u16 = 773;
pub const RINGS: [u16; 3] = [7, 11, 13];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Phase {
    pub prime: u16,
    pub tooth: u16,
    pub stride: u16,
}

impl Phase {
    pub fn new(prime: u16) -> Self {
        Self { prime, tooth: 0, stride: DRIVE % prime }
    }
    #[inline(always)]
    pub fn forward(&mut self) {
        let x = self.tooth + self.stride;
        self.tooth = if x >= self.prime { x - self.prime } else { x };
    }
    #[inline(always)]
    pub fn backward(&mut self) {
        if self.tooth < self.stride {
            self.tooth += self.prime;
        }
        self.tooth -= self.stride;
    }
}

/// The address of a position: its teeth on the three rings, and the CRT coordinate in 0..1001.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub teeth: [u16; 3],
    pub crt: u16,
}

/// The acquisition clock: ticked once per position from the origin.
#[derive(Clone, Debug)]
pub struct Clock {
    pub rings: [Phase; 3],
    pub pos: u64,
}

impl Clock {
    pub fn origin() -> Clock {
        Clock { rings: [Phase::new(RINGS[0]), Phase::new(RINGS[1]), Phase::new(RINGS[2])], pos: 0 }
    }
    pub fn tick(&mut self) {
        for r in &mut self.rings {
            r.forward();
        }
        self.pos += 1;
    }
    pub fn untick(&mut self) {
        for r in &mut self.rings {
            r.backward();
        }
        self.pos -= 1;
    }
    pub fn address(&self) -> Address {
        let teeth = [self.rings[0].tooth, self.rings[1].tooth, self.rings[2].tooth];
        Address { teeth, crt: crt_1001(teeth) }
    }
    /// The address of position `pos` from the origin, without keeping a clock: tooth = pos·DRIVE mod p.
    pub fn address_of(pos: u64) -> Address {
        let teeth = [
            ((pos % 7) * (DRIVE as u64 % 7) % 7) as u16,
            ((pos % 11) * (DRIVE as u64 % 11) % 11) as u16,
            ((pos % 13) * (DRIVE as u64 % 13) % 13) as u16,
        ];
        Address { teeth, crt: crt_1001(teeth) }
    }
}

/// Chinese remainder over 7·11·13: the unique x in 0..1001 with x ≡ teeth[i] (mod ring i).
pub fn crt_1001(teeth: [u16; 3]) -> u16 {
    // precomputed: M_i = 1001/p_i, y_i = M_i^{-1} mod p_i
    // 143 mod 7 = 3, inv 5 (3·5=15≡1); 91 mod 11 = 3, inv 4 (12≡1); 77 mod 13 = 12, inv 12 (144≡1)
    let x = (teeth[0] as u32 * 143 * 5 + teeth[1] as u32 * 91 * 4 + teeth[2] as u32 * 77 * 12) % 1001;
    x as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crt_round_trips_every_position_of_the_cycle() {
        for x in 0..1001u32 {
            let teeth = [(x % 7) as u16, (x % 11) as u16, (x % 13) as u16];
            assert_eq!(crt_1001(teeth) as u32, x);
        }
    }

    #[test]
    fn clock_matches_direct_address_and_reverses() {
        let mut c = Clock::origin();
        for pos in 0..3000u64 {
            assert_eq!(c.address(), Clock::address_of(pos));
            c.tick();
        }
        for _ in 0..3000 {
            c.untick();
        }
        assert_eq!(c.address(), Clock::address_of(0));
        assert_eq!(c.pos, 0);
    }

    #[test]
    fn no_single_ring_locks_the_cycle() {
        // the joint address of positions 0..1001 is a permutation of the cycle: no two positions
        // within one tour share an address, whatever cadence a transcript happens to have
        let mut seen = vec![false; 1001];
        for pos in 0..1001u64 {
            let a = Clock::address_of(pos);
            assert!(!seen[a.crt as usize], "collision at {}", pos);
            seen[a.crt as usize] = true;
        }
    }
}
