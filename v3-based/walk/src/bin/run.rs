// S4, as far as the local seats go: the running object, token by token, with the context hulled.
// Qwen3-1.7B layers 0..13 from their seats (../seats; no weight file is opened), fed the captured tokens one at a time.
// The hull keeps every token's keys and values as pixel positions: an activation is computed once and looked up after,
// never re-derived and never seated as a weight. A turn appends to the hull under a receipt (the hull's length per
// layer); rewinding a turn truncates the hull back to its receipt, bit for bit, and only the turn's output is kept.
// The context is phi-decayed by age: e_j is moved down the r-band by (i - j), through DECAY[age], a table built once
// from the relation r^2 = r + 1 with r bracketed by counting squares (phi is never written). DECAY = 1 for the gate.
// Gate: the stream after 64 tokens, one at a time, equals the batch run of pix.rs (final stream sha256).
// Arithmetic as pix.rs, on pixels,
// against the captured layer-1 and layer-14 states. geo.rs is the S3 reference (projections exact, outside ops in f64);
// this file moves the outside ops (norms, rotary, softmax, SiLU) and the residual stream onto pixels too.
//
// A pixel is sign, odd, band (value = +-odd * 2^band), with the odd below 2^bits. Two widths are declared:
//   OB: the object's width, 8 bits. Weights, activations entering a projection, the residual stream.
//   CB: the width of the declared tables and constants used by the outside ops.
// The only multiplication table is BYTE[a][b] = a * b for bytes, built once by repeated addition; a wider odd is
// read as bytes (the octet reading one level up). The outside functions are tables over the pixel family, built once:
//   EXP2[k] = 2^(k / 65536)  (the plane; the rung is a band move)      e^t = 2^(t * LOG2E) = EXP2[frac] << floor
//   RSQRT[m] = 1 / sqrt(m)    m even-banded, so the band halves exactly     RECIP[odd] = 1 / odd, band negated
// Every sum is exact landing counts joined once by a carry pass, then landed on a pixel: the one rounding, at the pixel.
// No division and no root is computed at runtime; f64 appears only in the table build (cold) and in the instrument.
use std::fs;
use std::thread;
use std::time::Instant;

const HID: usize = 2048;
const H: usize = 16;
const KV: usize = 8;
const D: usize = 128;
const INTER: usize = 6144;
const TOK: usize = 64;
const OB: u32 = 8;                         // the object's pixel width
const CB: u32 = 16;
const SB: u32 = 8;
const SDIR: &str = "../seats";               // the seats: the only place the runtime reads weights from
const CAP: &str = "../../receipts";

fn read_u16(path: &str) -> Vec<u16> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn sha256_hex(data: &[u8]) -> String {
    // minimal SHA-256, std only
    const K: [u32; 64] = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut m = data.to_vec(); let bl = (data.len() as u64) * 8; m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); } m.extend_from_slice(&bl.to_be_bytes());
    for ch in m.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([ch[4*i], ch[4*i+1], ch[4*i+2], ch[4*i+3]]); }
        for i in 16..64 { let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3); let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10); w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1); }
        let (mut a, mut b, mut c, mut d, mut e_, mut f, mut g, mut hh) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e_.rotate_right(6) ^ e_.rotate_right(11) ^ e_.rotate_right(25); let chh = (e_ & f) ^ (!e_ & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(chh).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22); let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e_; e_ = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e_, f, g, hh].iter().enumerate() { h[i] = h[i].wrapping_add(*v); }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

// ---------- the pixel ----------
#[derive(Clone, Copy, Default)]
struct X { neg: bool, band: i32, odd: u32, live: bool }
const DEAD: X = X { neg: false, band: 0, odd: 0, live: false };
const ONE: X = X { neg: false, band: 0, odd: 1, live: true };

/// An integer magnitude times 2^band, landed on `bits`: half to even, everything below is sticky.
fn land(mut mag: u128, mut band: i32, neg: bool, bits: u32) -> X {
    if mag == 0 { return DEAD; }
    let top = 128 - mag.leading_zeros();
    if top > bits {
        let sh = top - bits; let low = mag & ((1u128 << sh) - 1); let half = 1u128 << (sh - 1);
        mag >>= sh; band += sh as i32;
        if low > half || (low == half && mag & 1 == 1) { mag += 1; }
    }
    let t = mag.trailing_zeros(); mag >>= t; band += t as i32;
    X { neg, band, odd: mag as u32, live: true }
}
fn from_f64(v: f64, bits: u32) -> X {
    if v == 0.0 || !v.is_finite() { return DEAD; }
    let b = v.to_bits(); let ex = ((b >> 52) & 0x7FF) as i32; let frac = b & ((1u64 << 52) - 1);
    let (m, e) = if ex == 0 { (frac, -1074) } else { (frac | (1u64 << 52), ex - 1075) };
    land(m as u128, e, b >> 63 == 1, bits)
}
fn to_f64(x: X) -> f64 {                     // the instrument only
    if !x.live { return 0.0; }
    let v = (x.odd as f64) * 2f64.powi(x.band); if x.neg { -v } else { v }
}
fn stored_pixel(b: u16) -> X {               // the u16 as a position (bf16 layout)
    let s = b >> 15 == 1; let ef = ((b >> 7) & 0xFF) as i32; let f = (b & 0x7F) as u32;
    assert!(ef != 255, "not a position");
    let m = if ef > 0 { 128 + f } else { f };
    if m == 0 { return DEAD; }
    land(m as u128, if ef > 0 { ef - 134 } else { -133 }, s, OB)
}

// ---------- the one table: bytes times bytes, by repeated addition ----------
fn byte_table() -> Vec<u16> {
    let mut t = vec![0u16; 256 * 256];
    for a in 0..256usize { let mut acc = 0u32; for b in 0..256usize { t[a * 256 + b] = acc as u16; acc += a as u32; } }
    t
}
/// The product of two odds by byte lookups: a wider odd is bytes, the octet reading one level up.
fn omul(t: &[u16], a: u32, b: u32) -> u128 {
    let mut acc: u128 = 0; let mut ka = 0;
    let mut x = a;
    while x != 0 {
        let da = (x & 0xFF) as usize; let mut y = b; let mut kb = 0;
        while y != 0 { let db = (y & 0xFF) as usize; acc += (t[da * 256 + db] as u128) << (8 * (ka + kb)); y >>= 8; kb += 1; }
        x >>= 8; ka += 1;
    }
    acc
}
fn pmul(t: &[u16], a: X, b: X, bits: u32) -> X {
    if !a.live || !b.live { return DEAD; }
    land(omul(t, a.odd, b.odd), a.band + b.band, a.neg ^ b.neg, bits)
}
fn padd(a: X, b: X, bits: u32) -> X {
    if !a.live { return if b.live { land(b.odd as u128, b.band, b.neg, bits) } else { DEAD }; }
    if !b.live { return land(a.odd as u128, a.band, a.neg, bits); }
    let (hi, lo) = if a.band >= b.band { (a, b) } else { (b, a) };
    let gap = (hi.band - lo.band) as u32;
    let (mag, band, neg) = if gap > 100 {          // the small one only breaks a tie: one sticky bit, its way
        let m = if hi.neg == lo.neg { ((hi.odd as u128) << 2) + 1 } else { ((hi.odd as u128) << 2) - 1 };
        (m, hi.band - 2, hi.neg)
    } else {
        let sh = ((hi.odd as i128) << gap) * if hi.neg { -1 } else { 1 };
        let sl = (lo.odd as i128) * if lo.neg { -1 } else { 1 };
        let s = sh + sl; (s.unsigned_abs(), lo.band, s < 0)
    };
    land(mag, band, neg, bits)
}
fn pneg(a: X) -> X { X { neg: !a.neg, ..a } }
fn pmax(a: X, b: X) -> X {                    // exact compare: sign first, then magnitude on a common band
    let sign = |x: X| -> i8 { if !x.live { 0 } else if x.neg { -1 } else { 1 } };
    let (sa, sb) = (sign(a), sign(b));
    if sa != sb { return if sa > sb { a } else { b }; }
    if sa == 0 { return a; }
    let ge = |x: X, y: X| -> bool {            // |x| >= |y|
        let (mx, my) = (x.odd as u128, y.odd as u128);
        if x.band >= y.band { (mx << (x.band - y.band).min(100)) >= my } else { mx >= (my << (y.band - x.band).min(100)) }
    };
    let a_big = ge(a, b);
    if sa > 0 { if a_big { a } else { b } } else if a_big { b } else { a }
}

// ---------- the accumulator: exact landing counts, joined once ----------
struct Acc { s: Vec<i64>, floor: i32, lo: usize, hi: usize }
impl Acc {
    fn new() -> Acc { Acc { s: vec![0; 512], floor: -256, lo: usize::MAX, hi: 0 } }
    #[inline] fn add(&mut self, mag: u128, band: i32, neg: bool) {
        if mag == 0 { return; }
        let i = (band - self.floor) as usize; assert!(i + 62 < 512, "band out of range");
        // a magnitude wider than 62 bits is split into two cells
        let (lo, hi) = ((mag & ((1u128 << 62) - 1)) as i64, (mag >> 62) as i64);
        if neg { self.s[i] -= lo; if hi != 0 { self.s[i + 62] -= hi; } } else { self.s[i] += lo; if hi != 0 { self.s[i + 62] += hi; } }
        self.lo = self.lo.min(i); self.hi = self.hi.max(i + if hi != 0 { 62 } else { 0 });
    }
    fn fin(&mut self, bits: u32) -> X {
        if self.lo == usize::MAX { return DEAD; }
        let mut bv: Vec<u8> = Vec::with_capacity(self.hi - self.lo + 80); let mut carry: i64 = 0;
        for i in self.lo..=self.hi { let v = self.s[i] + carry; let bit = v & 1; bv.push(bit as u8); carry = (v - bit) >> 1; self.s[i] = 0; }
        while carry != 0 && carry != -1 { let bit = carry & 1; bv.push(bit as u8); carry = (carry - bit) >> 1; }
        let negative = carry == -1;
        if negative { let mut c = 1u8; for b in bv.iter_mut() { let v = (1 - *b) + c; *b = v & 1; c = v >> 1; } if c == 1 { bv.push(1); } }
        let base = self.floor + self.lo as i32; self.lo = usize::MAX; self.hi = 0;
        let Some(top) = bv.iter().rposition(|&b| b == 1) else { return DEAD; };
        let start = top.saturating_sub(119);
        let mut m: u128 = 0; for k in (start..=top).rev() { m = (m << 1) | bv[k] as u128; }
        if bv[..start].iter().any(|&b| b == 1) { m = (m << 1) | 1; return land(m, base + start as i32 - 1, negative, bits); }
        land(m, base + start as i32, negative, bits)
    }
}
fn dot(t: &[u16], acc: &mut Acc, xs: &[X], ys: &[X], bits: u32) -> X {
    for (a, b) in xs.iter().zip(ys) { if a.live && b.live { acc.add(omul(t, a.odd, b.odd), a.band + b.band, a.neg ^ b.neg); } }
    acc.fin(bits)
}

// ---------- the declared tables, built once ----------
struct Tables { byte: Vec<u16>, exp2: Vec<X>, rsqrt: Vec<X>, recip: Vec<X>, log2e: X, scale: X, eps: X, osc: Vec<(X, X)> }
fn tables() -> Tables {
    let exp2 = (0..65536).map(|k| from_f64(2f64.powf(k as f64 / 65536.0), CB)).collect();
    let rsqrt = (0..(1usize << (CB + 1))).map(|m| if m == 0 { DEAD } else { from_f64(1.0 / (m as f64).sqrt(), CB) }).collect();
    let recip = (0..(1usize << CB)).map(|o| if o == 0 { DEAD } else { from_f64(1.0 / o as f64, CB) }).collect();
    let mut osc = Vec::with_capacity(TOK * D / 2);
    for pos in 0..TOK { for i in 0..D / 2 { let a = pos as f64 * 1e6f64.powf(-((2 * i) as f64) / D as f64); osc.push((from_f64(a.cos(), CB), from_f64(a.sin(), CB))); } }
    Tables { byte: byte_table(), exp2, rsqrt, recip, log2e: from_f64(std::f64::consts::LOG2_E, CB),
             scale: from_f64(1.0 / (D as f64).sqrt(), CB), eps: from_f64(1e-6, CB), osc }
}
/// e^t: the rung is a band move, the plane is EXP2 at 16 bits.
fn pexp(tb: &Tables, t: X) -> X {
    if !t.live { return ONE; }
    let u = pmul(&tb.byte, t, tb.log2e, CB);
    let (i, k): (i64, u32) = if u.band >= 0 {
        if u.band > 40 { return if u.neg { DEAD } else { X { neg: false, band: 200, odd: 1, live: true } }; }
        (((u.odd as i64) << u.band) * if u.neg { -1 } else { 1 }, 0)
    } else {
        let fb = (-u.band) as u32; let ip = if fb >= 64 { 0 } else { (u.odd as u64) >> fb } as i64;
        let f = if fb >= 64 { u.odd as u64 } else { (u.odd as u64) & ((1u64 << fb) - 1) };
        let f16 = if fb >= 16 { if fb - 16 >= 64 { 0 } else { f >> (fb - 16) } } else { f << (16 - fb) } as u32;
        if !u.neg { (ip, f16) } else if f == 0 { (-ip, 0) } else if f16 == 0 { (-ip - 1, 65535) } else { (-ip - 1, 65536 - f16) }
    };
    if i < -150 { return DEAD; }
    if i > 150 { return X { neg: false, band: 200, odd: 1, live: true }; }
    let p = tb.exp2[(k & 0xFFFF) as usize]; X { band: p.band + i as i32, ..p }
}
fn prsqrt(tb: &Tables, p: X) -> X {
    if !p.live { return DEAD; }
    let (m, e) = if p.band & 1 != 0 { (p.odd << 1, p.band - 1) } else { (p.odd, p.band) };
    let r = tb.rsqrt[m as usize]; X { band: r.band - e / 2, ..r }
}
fn precip(tb: &Tables, p: X) -> X {
    if !p.live { return DEAD; }
    let r = tb.recip[p.odd as usize]; X { band: r.band - p.band, neg: p.neg, ..r }
}

// ---------- weights: the seat, read as it is ----------
/// A seated tensor: the codes of its ISIS3 cube, one UnsignedWord per weight, and the band floor from the label.
/// code = (band - floor) * 256 + sign * 128 + (odd - 1) / 2; 0 is the zero. Nothing is derived at load: the code is the position.
struct W { code: Vec<u16>, floor: i32, lo: i32, hi: i32 }
fn read_seat(path: &str) -> W {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let end = raw.windows(5).position(|w| w == b"
End
").expect("label") + 5;
    let txt = std::str::from_utf8(&raw[..end]).unwrap();
    let field = |k: &str| -> i64 { let i = txt.find(&format!("{k} = ")).unwrap() + k.len() + 3; txt[i..].split_whitespace().next().unwrap().parse().unwrap() };
    let (start, s, l, floor, span) = (field("StartByte") as usize, field("Samples") as usize, field("Lines") as usize, field("BandFloor") as i32, field("BandSpan") as i32);
    let body = &raw[start - 1..start - 1 + 2 * s * l];
    W { code: body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(), floor, lo: floor + 1, hi: floor + span }
}
fn seat_pixel(code: u16, floor: i32) -> X {
    if code == 0 { return DEAD; }
    X { neg: code & 128 != 0, band: (code >> 8) as i32 + floor, odd: 2 * (code & 127) as u32 + 1, live: true }
}

/// The octet read (as geo.rs), the landing counts joined to a pixel of OB. out[t][r] for x[t] (T x cols).
fn project(w: &W, rows: usize, cols: usize, x: &[Vec<X>], tab: &[u16]) -> Vec<Vec<X>> {
    let t_n = x.len();
    let live = || x.iter().flatten().filter(|p| p.live);
    let (xlo, xhi) = (live().map(|p| p.band).min().unwrap(), live().map(|p| p.band).max().unwrap());
    let g0 = w.lo + xlo; let gn = (w.hi - w.lo + xhi - xlo + 1) as usize;
    let oct = cols / 8;
    let mut xo = vec![DEAD; oct * t_n * 8];    // [octet][token][8]
    for (t, row) in x.iter().enumerate() {
        for (j, p) in row.iter().enumerate() { let mut q = *p; q.band -= xlo; xo[((j / 8) * t_n + t) * 8 + j % 8] = q; }
    }
    let nthreads = 8; let chunk = rows.div_ceil(nthreads);
    let parts: Vec<(usize, Vec<Vec<X>>)> = thread::scope(|sc| {
        let hs: Vec<_> = (0..nthreads).map(|th| {
            let xo = &xo;
            sc.spawn(move || {
                let r_lo = th * chunk; let r_hi = ((th + 1) * chunk).min(rows);
                let mut part = vec![vec![DEAD; r_hi.saturating_sub(r_lo)]; t_n];
                let mut s = vec![0i64; t_n * gn]; let mut bits = Vec::with_capacity(gn + 64);
                for r in r_lo..r_hi {
                    s.iter_mut().for_each(|v| *v = 0);
                    for o in 0..oct {
                        let wb0 = r * cols + o * 8;
                        for a in 0..8 {
                            let c = w.code[wb0 + a]; if c == 0 { continue; }
                            let trow = &tab[((2 * (c & 127) as usize) + 1) * 256..][..256];        // the address
                            let wneg = ((c >> 7) & 1) as u8; let woff = (c >> 8) as usize - 1;      // the sign, the band
                            for t in 0..t_n {
                                let p = &xo[(o * t_n + t) * 8 + a]; if !p.live { continue; }
                                let v = trow[p.odd as usize] as i64;                     // the move
                                let cell = &mut s[t * gn + woff + p.band as usize];       // the landing
                                if wneg ^ (p.neg as u8) != 0 { *cell -= v } else { *cell += v }
                            }
                        }
                    }
                    for t in 0..t_n { part[t][r - r_lo] = join(&s[t * gn..(t + 1) * gn], g0, &mut bits); }
                }
                (r_lo, part)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = vec![vec![DEAD; rows]; t_n];
    for (r_lo, part) in parts { for (t, row) in part.into_iter().enumerate() { out[t][r_lo..r_lo + row.len()].copy_from_slice(&row); } }
    out
}
fn join(s: &[i64], g0: i32, bits: &mut Vec<u8>) -> X {
    bits.clear(); let mut carry: i64 = 0;
    for &c in s { let v = c + carry; let bit = v & 1; bits.push(bit as u8); carry = (v - bit) >> 1; }
    while carry != 0 && carry != -1 { let bit = carry & 1; bits.push(bit as u8); carry = (carry - bit) >> 1; }
    let negative = carry == -1;
    if negative { let mut c = 1u8; for b in bits.iter_mut() { let v = (1 - *b) + c; *b = v & 1; c = v >> 1; } if c == 1 { bits.push(1); } }
    let Some(top) = bits.iter().rposition(|&b| b == 1) else { return DEAD; };
    let start = top.saturating_sub(119);
    let mut m: u128 = 0; for k in (start..=top).rev() { m = (m << 1) | bits[k] as u128; }
    if bits[..start].iter().any(|&b| b == 1) { return land((m << 1) | 1, g0 + start as i32 - 1, negative, OB); }
    land(m, g0 + start as i32, negative, OB)
}

// ---------- the outside ops, on pixels ----------
fn rms(tb: &Tables, acc: &mut Acc, x: &[X], g: &[X]) -> Vec<X> {
    for v in x { if v.live { acc.add(omul(&tb.byte, v.odd, v.odd), 2 * v.band, false); } }
    let mut ms = acc.fin(CB); ms.band -= (x.len().trailing_zeros()) as i32;          // / n, n a power of two
    let k = prsqrt(tb, padd(ms, tb.eps, CB));
    x.iter().zip(g).map(|(&v, &w)| pmul(&tb.byte, pmul(&tb.byte, v, w, CB), k, OB)).collect()
}
fn spin(tb: &Tables, v: &mut [X], pos: usize) {
    for i in 0..D / 2 {
        let (c, s) = tb.osc[pos * (D / 2) + i];
        let (x1, x2) = (v[i], v[i + D / 2]);
        v[i] = padd(pmul(&tb.byte, x1, c, CB), pneg(pmul(&tb.byte, x2, s, CB)), OB);
        v[i + D / 2] = padd(pmul(&tb.byte, x2, c, CB), pmul(&tb.byte, x1, s, CB), OB);
    }
}
fn silu(tb: &Tables, g: X, u: X) -> X {
    if !g.live { return DEAD; }
    let sig = precip(tb, padd(ONE, pexp(tb, pneg(g)), CB));
    pmul(&tb.byte, pmul(&tb.byte, g, sig, CB), u, OB)
}

struct Layer { q: W, k: W, v: W, o: W, gate: W, up: W, down: W, n_in: Vec<X>, n_post: Vec<X>, n_q: Vec<X>, n_k: Vec<X> }
fn load_layer(l: usize) -> Layer {
    let p = |n: &str| format!("{SDIR}/L{l}.{n}.cub");
    let norm = |n: &str| { let w = read_seat(&p(n)); w.code.iter().map(|&c| seat_pixel(c, w.floor)).collect::<Vec<_>>() };
    Layer { q: read_seat(&p("self_attn.q_proj")), k: read_seat(&p("self_attn.k_proj")), v: read_seat(&p("self_attn.v_proj")),
            o: read_seat(&p("self_attn.o_proj")), gate: read_seat(&p("mlp.gate_proj")), up: read_seat(&p("mlp.up_proj")),
            down: read_seat(&p("mlp.down_proj")), n_in: norm("input_layernorm"), n_post: norm("post_attention_layernorm"),
            n_q: norm("self_attn.q_norm"), n_k: norm("self_attn.k_norm") }
}

fn compare(r: &[Vec<X>], l: usize) -> String {
    let cap = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L{l}-508tok.u16"));
    let (mut dotp, mut na, mut nb) = (0f64, 0f64, 0f64);
    for ti in 0..r.len() { for d in 0..HID { let (x, y) = (to_f64(r[ti][d]), to_f64(stored_pixel(cap[ti * HID + d]))); dotp += x * y; na += x * x; nb += y * y; } }
    format!("cosine {:.6} against the captured state (instrument)", dotp / (na * nb).sqrt())
}


// ---------- the hull: the context as pixel positions, kept under receipts ----------
struct Hull { k: Vec<Vec<Vec<Vec<X>>>>, v: Vec<Vec<Vec<Vec<X>>>> }   // [layer][kv head][position][D]
impl Hull {
    fn new(layers: usize) -> Hull { Hull { k: vec![vec![Vec::new(); KV]; layers], v: vec![vec![Vec::new(); KV]; layers] } }
    fn receipt(&self) -> Vec<usize> { self.k.iter().map(|l| l[0].len()).collect() }
    fn rewind(&mut self, receipt: &[usize]) { for (l, &n) in receipt.iter().enumerate() { for h in 0..KV { self.k[l][h].truncate(n); self.v[l][h].truncate(n); } } }
    fn sha(&self) -> String {
        let mut b = Vec::new();
        for l in 0..self.k.len() { for h in 0..KV { for (kk, vv) in self.k[l][h].iter().zip(&self.v[l][h]) { for p in kk.iter().chain(vv) { push_pixel(&mut b, p); } } } }
        sha256_hex(&b)
    }
}
fn push_pixel(b: &mut Vec<u8>, p: &X) {
    if p.live { b.push(p.neg as u8); b.extend_from_slice(&p.band.to_le_bytes()); b.extend_from_slice(&p.odd.to_le_bytes()); } else { b.extend_from_slice(&[0xFF; 9]); }
}

/// r^-age as a pixel, from the relation: the address (p, q) of r^-n in Z[r] by the kernel's _down, r bracketed by
/// counting squares (sqrt 5 = 2r - 1), landed once on CB bits. phi is never written. age 0 is 1.
fn decay_table(n: usize, on: bool) -> Vec<X> {
    if !on { return vec![ONE; n]; }
    let k = 40u32; let five = 5u128 << (2 * k);
    let mut s = 0u128; let mut bit = 1u128 << 63;            // integer square root of 5 * 4^k by counting squares
    while bit > 0 { if (s + bit) * (s + bit) <= five { s += bit; } bit >>= 1; }
    let r_num = ((1u128 << k) + s) as i128;                    // 2r * 2^k = (1 + sqrt 5) * 2^k
    let (mut p, mut q) = (1i128, 0i128); let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let val = p * (1i128 << (k + 1)) + q * r_num;           // (p + q r) * 2^(k+1)
        out.push(land(val.unsigned_abs(), -(k as i32 + 1), val < 0, CB));
        let (np, nq) = (q - p, p); p = np; q = nq;              // address times r^-1 = r - 1
    }
    out
}

/// One token through one layer. The hull grows by one position per kv head; attention reads the hull.
fn step(x_in: &mut Vec<X>, ly: &Layer, l: usize, hull: &mut Hull, tb: &Tables, decay: &[X], pos: usize, acc: &mut Acc) {
    let x = rms(tb, acc, x_in, &ly.n_in);
    let one = vec![x];
    let q = project(&ly.q, H * D, HID, &one, &tb.byte).remove(0);
    let k = project(&ly.k, KV * D, HID, &one, &tb.byte).remove(0);
    let v = project(&ly.v, KV * D, HID, &one, &tb.byte).remove(0);
    for h in 0..KV {
        let mut z = rms(tb, acc, &k[h * D..(h + 1) * D], &ly.n_k); spin(tb, &mut z, pos);
        hull.k[l][h].push(z); hull.v[l][h].push(v[h * D..(h + 1) * D].to_vec());     // the activation, kept as lookup
    }
    let mut att = vec![DEAD; H * D];
    for h in 0..H {
        let kv = h / (H / KV); let mut qz = rms(tb, acc, &q[h * D..(h + 1) * D], &ly.n_q); spin(tb, &mut qz, pos);
        let keys = &hull.k[l][kv]; let vals = &hull.v[l][kv]; let n = keys.len();
        let sc: Vec<X> = (0..n).map(|j| pmul(&tb.byte, dot(&tb.byte, acc, &qz, &keys[j], CB), tb.scale, CB)).collect();
        let m = sc.iter().fold(sc[0], |a, &b| pmax(a, b));
        let e: Vec<X> = sc.iter().enumerate().map(|(j, &s)| pmul(&tb.byte, pexp(tb, padd(s, pneg(m), CB)), decay[n - 1 - j], CB)).collect();
        for &w in &e { if w.live { acc.add(w.odd as u128, w.band, false); } }
        let rz = precip(tb, acc.fin(CB));
        for d in 0..D {
            for (j, &w) in e.iter().enumerate() { let vv = vals[j][d]; if w.live && vv.live { acc.add(omul(&tb.byte, w.odd, vv.odd), w.band + vv.band, vv.neg); } }
            att[h * D + d] = pmul(&tb.byte, acc.fin(CB), rz, OB);
        }
    }
    let o = project(&ly.o, HID, H * D, &vec![att], &tb.byte).remove(0);
    for d in 0..HID { x_in[d] = padd(x_in[d], o[d], SB); }
    let x2 = rms(tb, acc, x_in, &ly.n_post);
    let two = vec![x2];
    let g = project(&ly.gate, INTER, HID, &two, &tb.byte).remove(0);
    let u = project(&ly.up, INTER, HID, &two, &tb.byte).remove(0);
    let act: Vec<X> = (0..INTER).map(|j| silu(tb, g[j], u[j])).collect();
    let dn = project(&ly.down, HID, INTER, &vec![act], &tb.byte).remove(0);
    for d in 0..HID { x_in[d] = padd(x_in[d], dn[d], SB); }
}

/// A turn: the tokens lo..hi, one at a time, through every layer; returns the output rows (the stream after the last layer).
fn turn(cap0: &[u16], lo: usize, hi: usize, layers: &[Layer], hull: &mut Hull, tb: &Tables, decay: &[X]) -> Vec<Vec<X>> {
    let mut acc = Acc::new(); let mut out = Vec::new();
    for t in lo..hi {
        let mut x: Vec<X> = cap0[t * HID..(t + 1) * HID].iter().map(|&b| stored_pixel(b)).collect();
        for (l, ly) in layers.iter().enumerate() { step(&mut x, ly, l, hull, tb, decay, t, &mut acc); }
        out.push(x);
    }
    out
}
fn rows_sha(rows: &[Vec<X>]) -> String { let mut b = Vec::new(); for r in rows { for p in r { push_pixel(&mut b, p); } } sha256_hex(&b) }
fn compare_rows(rows: &[Vec<X>], l: usize, offset: usize) -> String {
    let cap = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L{l}-508tok.u16"));
    let (mut dotp, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (i, r) in rows.iter().enumerate() { for d in 0..HID { let (x, y) = (to_f64(r[d]), to_f64(stored_pixel(cap[(offset + i) * HID + d]))); dotp += x * y; na += x * x; nb += y * y; } }
    format!("cosine {:.6} against the captured state (instrument)", dotp / (na * nb).sqrt())
}

const LAYERS: usize = 28;
const VOCAB: usize = 151936;
const FRANCE: [usize; 5] = [785, 6722, 315, 9625, 374];        // "The capital of France is", v1's reference prompt

/// The France prompt through every layer, token by token, from the embedding seat to the logits.
/// The embedding row is a lookup (the token decomposed into values); the head is the same seat, tied.
/// Layer-major, one layer resident at a time (7 GB of RAM here): each layer's hull depends only on its own earlier
/// tokens, so the result is the token-major one, bit for bit.
fn france(embed: &W, norm: &[X], tb: &Tables, decay: &[X]) {
    let mut hull = Hull::new(LAYERS); let mut acc = Acc::new();
    let cap14 = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L14-5tok.u16"));
    let cap17 = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L17-5tok.u16"));
    let mut at14: Vec<Vec<X>> = Vec::new(); let mut at17: Vec<Vec<X>> = Vec::new();
    let t0 = Instant::now();
    let mut xs: Vec<Vec<X>> = FRANCE.iter().map(|&id| embed.code[id * HID..(id + 1) * HID].iter().map(|&c| seat_pixel(c, embed.floor)).collect()).collect();
    for l in 0..LAYERS {
        if l == 14 { at14 = xs.clone(); } if l == 17 { at17 = xs.clone(); }
        let ly = load_layer(l);
        for t in 0..FRANCE.len() { step(&mut xs[t], &ly, l, &mut hull, tb, decay, t, &mut acc); }
    }
    let last = xs.pop().unwrap();
    let cos = |rows: &[Vec<X>], cap: &[u16]| -> f64 {
        let (mut dp, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (i, r) in rows.iter().enumerate() { for d in 0..HID { let (x, y) = (to_f64(r[d]), to_f64(stored_pixel(cap[i * HID + d]))); dp += x * y; na += x * x; nb += y * y; } }
        dp / (na * nb).sqrt()
    };
    println!("[F1] the stream entering layer 14 against the captured 5-token state: cosine {:.6}; entering layer 17: cosine {:.6}", cos(&at14, &cap14), cos(&at17, &cap17));
    let h = rms(tb, &mut acc, &last, norm);
    let logits = project(embed, VOCAB, HID, &vec![h], &tb.byte).remove(0);
    let mut order: Vec<usize> = (0..VOCAB).collect();
    order.sort_by(|&a, &b| { let m = pmax(logits[a], logits[b]); if m.odd == logits[a].odd && m.band == logits[a].band && m.neg == logits[a].neg && m.live == logits[a].live { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater } });
    let top: Vec<String> = order[..5].iter().map(|&i| format!("{i} ({:.4})", to_f64(logits[i]))).collect();
    println!("[F2] the last token's logits, top 5 ids: {}", top.join(", "));
    println!("[F2] argmax {} in {:.1}s for 5 tokens x {LAYERS} layers + the head", order[0], t0.elapsed().as_secs_f64());
}

fn main() {
    let tb = tables();
    let cap0 = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L0-508tok.u16"));
    let decay_off = decay_table(TOK, false);
    {
        let embed = read_seat(&format!("{SDIR}/embed_tokens.cub"));
        let norm_w = read_seat(&format!("{SDIR}/norm.cub")); let norm: Vec<X> = norm_w.code.iter().map(|&c| seat_pixel(c, norm_w.floor)).collect();
        france(&embed, &norm, &tb, &decay_off);
    }
    let ts = Instant::now(); let layers: Vec<Layer> = (0..14).map(load_layer).collect(); let layers = &layers[..];
    println!("seats read: 14 layers resident in {:.1}s", ts.elapsed().as_secs_f64());
    let decay_off = decay_table(TOK, false);

    // R1, the gate: 64 tokens one at a time equal the batch run (pix.rs: final stream sha256 c6c5f3e4...)
    let mut hull = Hull::new(14); let tr = Instant::now();
    let out = turn(&cap0, 0, TOK, layers, &mut hull, &tb, &decay_off);
    let run_s = tr.elapsed().as_secs_f64();
    let batch = "c6c5f3e465d778b1348cca31be80172b29d9f8f7a846eac95dd9359187f189c6";
    let sha = rows_sha(&out);
    println!("[R1] {}  token by token, {TOK} tokens: {}", if sha == batch { "PASS" } else { "FAIL" }, compare(&out, 14));
    println!("[R1] final stream sha256 {sha}  equals the batch run's {}: {}", &batch[..8], sha == batch);
    println!("[R1] {run_s:.1}s, {:.2}s per token; the hull holds {} positions x 14 layers x {KV} kv heads", run_s / TOK as f64, hull.k[0][0].len());

    // R2, R3: receipts and rewind. A context of 40, a turn of 24, rewound, and the turn again.
    let mut hull = Hull::new(14);
    let _ctx = turn(&cap0, 0, 40, layers, &mut hull, &tb, &decay_off);
    let receipt = hull.receipt(); let before = hull.sha();
    let out1 = turn(&cap0, 40, TOK, layers, &mut hull, &tb, &decay_off); let grown = hull.sha();
    hull.rewind(&receipt); let after = hull.sha();
    let out2 = turn(&cap0, 40, TOK, layers, &mut hull, &tb, &decay_off);
    println!("[R2] {}  the hull after the turn, rewound to its receipt ({} positions per layer): sha equal to before the turn: {}; the turn had changed it: {}",
             if after == before && grown != before { "PASS" } else { "FAIL" }, receipt[0], after == before, grown != before);
    println!("[R3] {}  the same turn on the rewound hull gives the same 24 output rows: {}; those rows: {}",
             if rows_sha(&out1) == rows_sha(&out2) { "PASS" } else { "FAIL" }, rows_sha(&out1) == rows_sha(&out2), compare_rows(&out1, 14, 40));

    // R4: the phi decay, declared from the relation. Not the captured model; reported only.
    let decay_on = decay_table(TOK, true);
    println!("[R4] note  DECAY[0..6] = {}", (0..6).map(|a| format!("{:.6}", to_f64(decay_on[a]))).collect::<Vec<_>>().join(", "));
    let mut hull = Hull::new(14);
    let outd = turn(&cap0, 0, TOK, layers, &mut hull, &tb, &decay_on);
    println!("[R4] note  phi-decayed context, 64 tokens: {} (a different object; the capture is not its reference)", compare(&outd, 14));
}
