// The whole forward as pixels, std only. Qwen3-1.7B layers 0..13 from their seats (../seats, written once by seat.rs;
// no weight file is opened here), on the captured tokens,
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

fn forward(r: &mut Vec<Vec<X>>, ly: &Layer, tb: &Tables) {
    let t = r.len(); let mut acc = Acc::new();
    let x: Vec<Vec<X>> = r.iter().map(|v| rms(tb, &mut acc, v, &ly.n_in)).collect();
    let q = project(&ly.q, H * D, HID, &x, &tb.byte);
    let k = project(&ly.k, KV * D, HID, &x, &tb.byte);
    let v = project(&ly.v, KV * D, HID, &x, &tb.byte);
    let mut qh = vec![vec![vec![DEAD; D]; t]; H]; let mut kh = vec![vec![vec![DEAD; D]; t]; KV];
    for ti in 0..t {
        for h in 0..H { let mut z = rms(tb, &mut acc, &q[ti][h * D..(h + 1) * D], &ly.n_q); spin(tb, &mut z, ti); qh[h][ti] = z; }
        for h in 0..KV { let mut z = rms(tb, &mut acc, &k[ti][h * D..(h + 1) * D], &ly.n_k); spin(tb, &mut z, ti); kh[h][ti] = z; }
    }
    let heads: Vec<Vec<Vec<X>>> = thread::scope(|sc| {
        let hs: Vec<_> = (0..H).map(|h| {
            let (qh, kh, v) = (&qh, &kh, &v);
            sc.spawn(move || {
                let kv = h / (H / KV); let mut out = vec![vec![DEAD; D]; t]; let mut acc = Acc::new();
                for i in 0..t {
                    let sc: Vec<X> = (0..=i).map(|j| pmul(&tb.byte, dot(&tb.byte, &mut acc, &qh[h][i], &kh[kv][j], CB), tb.scale, CB)).collect();
                    let m = sc.iter().fold(sc[0], |a, &b| pmax(a, b));
                    let e: Vec<X> = sc.iter().map(|&s| pexp(tb, padd(s, pneg(m), CB))).collect();
                    for &w in &e { if w.live { acc.add(w.odd as u128, w.band, false); } }
                    let z = acc.fin(CB); let rz = precip(tb, z);
                    for d in 0..D {
                        for (j, &w) in e.iter().enumerate() { let vv = v[j][kv * D + d]; if w.live && vv.live { acc.add(omul(&tb.byte, w.odd, vv.odd), w.band + vv.band, vv.neg); } }
                        out[i][d] = pmul(&tb.byte, acc.fin(CB), rz, OB);
                    }
                }
                out
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut att = vec![vec![DEAD; H * D]; t];
    for (h, out) in heads.into_iter().enumerate() { for i in 0..t { att[i][h * D..(h + 1) * D].copy_from_slice(&out[i]); } }
    let o = project(&ly.o, HID, H * D, &att, &tb.byte);
    for ti in 0..t { for d in 0..HID { r[ti][d] = padd(r[ti][d], o[ti][d], SB); } }
    let x2: Vec<Vec<X>> = r.iter().map(|v| rms(tb, &mut acc, v, &ly.n_post)).collect();
    let g = project(&ly.gate, INTER, HID, &x2, &tb.byte);
    let u = project(&ly.up, INTER, HID, &x2, &tb.byte);
    let act: Vec<Vec<X>> = (0..t).map(|ti| (0..INTER).map(|j| silu(tb, g[ti][j], u[ti][j])).collect()).collect();
    let dn = project(&ly.down, HID, INTER, &act, &tb.byte);
    for ti in 0..t { for d in 0..HID { r[ti][d] = padd(r[ti][d], dn[ti][d], SB); } }
}

fn compare(r: &[Vec<X>], l: usize) -> String {
    let cap = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L{l}-508tok.u16"));
    let (mut dotp, mut na, mut nb) = (0f64, 0f64, 0f64);
    for ti in 0..r.len() { for d in 0..HID { let (x, y) = (to_f64(r[ti][d]), to_f64(stored_pixel(cap[ti * HID + d]))); dotp += x * y; na += x * x; nb += y * y; } }
    format!("cosine {:.6} against the captured state (instrument)", dotp / (na * nb).sqrt())
}

fn main() {
    let tt = Instant::now(); let tb = tables(); let table_s = tt.elapsed().as_secs_f64();
    let cap0 = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L0-508tok.u16"));
    let mut r: Vec<Vec<X>> = (0..TOK).map(|t| cap0[t * HID..(t + 1) * HID].iter().map(|&b| stored_pixel(b)).collect()).collect();
    let (mut seat_s, mut run_s) = (0f64, 0f64);
    for l in 0..14 {
        let ts = Instant::now(); let ly = load_layer(l); seat_s += ts.elapsed().as_secs_f64();
        let tr = Instant::now(); forward(&mut r, &ly, &tb); run_s += tr.elapsed().as_secs_f64();
        if l == 0 { println!("[L0->L1]  {}", compare(&r, 1)); }
    }
    println!("[L0->L14] {}", compare(&r, 14));
    let mut bytes = Vec::with_capacity(TOK * HID * 8);
    for row in &r { for p in row { if p.live { bytes.push(p.neg as u8); bytes.extend_from_slice(&p.band.to_le_bytes()); bytes.extend_from_slice(&p.odd.to_le_bytes()); } else { bytes.extend_from_slice(&[0xFF; 9]); } } }
    println!("final stream sha256 {}", sha256_hex(&bytes));
    println!("pixels: object {OB} bits, tables {CB} bits, stream {SB} bits; tables {table_s:.1}s; seats read {seat_s:.1}s; 14 layers x {TOK} tokens {run_s:.1}s");
}
