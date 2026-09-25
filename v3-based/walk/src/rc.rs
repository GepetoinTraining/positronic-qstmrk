// The integer binary range coder verified by T12 (0 different over 704,643,072 weights), shared from here on.
// LZMA-style: 11-bit probabilities, adaptation shift 5.
pub struct Enc {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}
impl Enc {
    pub fn new() -> Self {
        Enc { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1, out: Vec::new() }
    }
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }
    pub fn bit(&mut self, p: &mut u16, b: u32) {
        let bound = (self.range >> 11) * (*p as u32);
        if b == 0 {
            self.range = bound;
            *p += (2048 - *p) >> 5;
        } else {
            self.low += bound as u64;
            self.range -= bound;
            *p -= *p >> 5;
        }
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.shift_low();
        }
    }
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

pub struct Dec<'a> {
    code: u32,
    range: u32,
    src: &'a [u8],
    pos: usize,
}
impl<'a> Dec<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        let mut d = Dec { code: 0, range: 0xFFFF_FFFF, src, pos: 0 };
        for _ in 0..5 {
            d.code = (d.code << 8) | d.next() as u32;
        }
        d
    }
    fn next(&mut self) -> u8 {
        let b = *self.src.get(self.pos).unwrap_or(&0);
        self.pos += 1;
        b
    }
    pub fn bit(&mut self, p: &mut u16) -> u32 {
        let bound = (self.range >> 11) * (*p as u32);
        let b = if self.code < bound {
            self.range = bound;
            *p += (2048 - *p) >> 5;
            0
        } else {
            self.code -= bound;
            self.range -= bound;
            *p -= *p >> 5;
            1
        };
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next() as u32;
        }
        b
    }
}

pub fn bucket(p: u16) -> usize {
    let e = ((p >> 7) & 0xFF) as usize;
    e.clamp(110, 125) - 110
}

pub fn shuffle(p: &mut [u16]) {
    let mut s: u64 = 20260914;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..p.len()).rev() {
        let j = ((next() as u128 * (i as u128 + 1)) >> 64) as usize;
        p.swap(i, j);
    }
}
