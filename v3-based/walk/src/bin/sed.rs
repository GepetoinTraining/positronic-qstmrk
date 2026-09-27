// T15: closing the sedenion pixel. Exact integers (i128), std only. Registered in ../prereg_T15.txt before this file.
// The product is the kernel's: Cayley-Dickson doubling; SIGN[a][b] with landing a xor b (legos/kernel.py).
use std::fs;

const WDIR: &str = "../ARCHIVE-2026-09-25-mistaken/weights";
const SEED: u64 = 0x5EDE_0015;

// ---------- the product, as the kernel builds it ----------
fn cd(x: &[i128], y: &[i128]) -> Vec<i128> {
    let n = x.len();
    if n == 1 { return vec![x[0] * y[0]]; }
    let h = n / 2; let (a, b) = x.split_at(h); let (c, d) = y.split_at(h);
    let conj = |v: &[i128]| { let mut w = v.to_vec(); for t in &mut w[1..] { *t = -*t; } w };
    let (ac, db, da, bc) = (cd(a, c), cd(&conj(d), b), cd(d, a), cd(b, &conj(c)));
    ac.iter().zip(&db).map(|(p, q)| p - q).chain(da.iter().zip(&bc).map(|(p, q)| p + q)).collect()
}
fn e(i: usize) -> Vec<i128> { let mut v = vec![0; 16]; v[i] = 1; v }
fn sign_table() -> [[i128; 16]; 16] {
    let mut s = [[0i128; 16]; 16];
    for a in 0..16 { for b in 0..16 { let p = cd(&e(a), &e(b)); s[a][b] = p.iter().sum(); } }
    s
}
fn mul(s: &[[i128; 16]; 16], x: &[i128], y: &[i128]) -> Vec<i128> {
    let mut out = vec![0i128; 16];
    for a in 0..16 { if x[a] != 0 { for b in 0..16 { if y[b] != 0 { out[a ^ b] += s[a][b] * x[a] * y[b]; } } } }
    out
}
fn conj(v: &[i128]) -> Vec<i128> { let mut w = v.to_vec(); for t in &mut w[1..] { *t = -*t; } w }
fn nrm(v: &[i128]) -> i128 { v.iter().map(|t| t * t).sum() }
fn scaled(v: &[i128], k: i128) -> Vec<i128> { v.iter().map(|t| t * k).collect() }

// ---------- a seeded xorshift ----------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    fn small(&mut self) -> i128 { (self.next() % 101) as i128 - 50 }
    fn sed(&mut self, lo: usize, hi: usize) -> Vec<i128> { let mut v = vec![0i128; 16]; for i in lo..hi { v[i] = self.small(); } v }
}

// ---------- real pixels: the u16 as a position ----------
fn stored_position(b: u16) -> Option<(i128, i32)> {
    let s = (b >> 15) as i128; let ef = ((b >> 7) & 0xFF) as i32; let f = (b & 0x7F) as i128;
    assert!(ef != 255, "not a position");
    let m = if ef > 0 { 128 + f } else { f };
    if m == 0 { return None; }
    let t = m.trailing_zeros() as i32; let odd = m >> t;
    Some((if s == 1 { -odd } else { odd }, (if ef > 0 { ef - 134 } else { -133 }) + t))
}
fn pixel(bits: &[u16]) -> Vec<i128> {
    let pos: Vec<Option<(i128, i32)>> = bits.iter().map(|&b| stored_position(b)).collect();
    let lo = pos.iter().flatten().map(|p| p.1).min().unwrap_or(0);
    pos.iter().map(|p| match p { Some((o, g)) => o << (g - lo), None => 0 }).collect()
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

fn main() {
    let s = sign_table();
    // the table against the recursive product on every pair, and its hash for the Python cross-check
    let mut table_ok = true;
    for a in 0..16 { for b in 0..16 { table_ok &= mul(&s, &e(a), &e(b)) == cd(&e(a), &e(b)); } }
    let flat: Vec<u8> = s.iter().flatten().map(|v| if *v < 0 { b'-' } else { b'+' }).collect();
    println!("[T]  {}  SIGN table = recursive product on all 256 pairs; sha256 of the 256 signs (+/-, row-major): {}",
             if table_ok { "PASS" } else { "FAIL" }, sha256_hex(&flat));

    let mut rng = Rng(SEED);
    // P1: the 32 signed anchors close; anchor moves rewind
    let mut closed = true;
    for a in 0..16 { for b in 0..16 { let p = mul(&s, &e(a), &e(b)); closed &= p.iter().filter(|v| **v != 0).count() == 1 && p[a ^ b].abs() == 1; } }
    let (mut left, mut right, mut tot) = (0usize, 0usize, 0usize);
    for _ in 0..1000 {
        let x = rng.sed(0, 16);
        for a in 1..16 {
            let ea = e(a); tot += 1;
            if mul(&s, &ea, &mul(&s, &ea, &x)) == scaled(&x, -1) { left += 1; }
            if mul(&s, &mul(&s, &x, &ea), &ea) == scaled(&x, -1) { right += 1; }
        }
    }
    println!("[P1] {}  32 signed anchors closed (one anchor lands, sign +-1): {closed}; e_a(e_a x) = -x on {left}/{tot}; (x e_a) e_a = -x on {right}/{tot}",
             if closed && left == tot && right == tot { "PASS" } else { "FAIL" });

    // P2: the octet closes: rewind and norm
    let mut p2 = true; let (mut rw, mut nm, mut n2) = (0usize, 0usize, 0usize);
    for oct in 0..2 {
        let (lo, hi) = if oct == 0 { (0, 8) } else { (8, 16) };
        for _ in 0..1000 {
            let w = rng.sed(lo, hi); let x = rng.sed(0, 16); let xs = rng.sed(lo, hi);
            let back = mul(&s, &w, &mul(&s, &conj(&w), &x)); n2 += 1;
            if back == scaled(&x, nrm(&w)) { rw += 1; } else { p2 = false; }
            if nrm(&mul(&s, &w, &xs)) == nrm(&w) * nrm(&xs) { nm += 1; } else { p2 = false; }
        }
    }
    println!("[P2] {}  octet w (inner 1000, outer 1000): w(conj(w) x) = |w|^2 x for any x on {rw}/{n2}; |w x|^2 = |w|^2 |x|^2 for x on the same octet on {nm}/{n2}",
             if p2 { "PASS" } else { "FAIL" });

    // P3: the mixed pixel does not close
    let (mut fail_rw, mut fail_nm) = (0usize, 0usize);
    for _ in 0..1000 {
        let w = rng.sed(0, 16); let x = rng.sed(0, 16);
        if mul(&s, &w, &mul(&s, &conj(&w), &x)) != scaled(&x, nrm(&w)) { fail_rw += 1; }
        if nrm(&mul(&s, &w, &x)) != nrm(&w) * nrm(&x) { fail_nm += 1; }
    }
    println!("[P3] {}  mixed w, 1000 (w, x): rewind fails on {fail_rw}/1000; norm not kept on {fail_nm}/1000 (registered: >= 99%)",
             if fail_rw >= 990 { "HELD" } else { "KILLED" });

    // P4 and P5: real pixels, layer-0 q_proj of Qwen3-1.7B, first 4096 pixels of 16 weights
    let raw = fs::read(format!("{WDIR}/model.layers.0.self_attn.q_proj.weight.u16")).expect("q_proj u16");
    let bits: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let (mut whole_fail, mut in_ok, mut out_ok, mut band0_ok, mut n) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut px_whole_fail, mut px_in_ok, mut px_out_ok) = (0usize, 0usize, 0usize);
    for p in 0..4096 {
        let w = pixel(&bits[p * 16..(p + 1) * 16]);
        let mut w_in = w.clone(); for t in &mut w_in[8..] { *t = 0; }
        let mut w_out = w.clone(); for t in &mut w_out[..8] { *t = 0; }
        let (mut wf, mut iok, mut ook) = (0, 0, 0);
        for _ in 0..10 {
            let x = rng.sed(0, 16); n += 1;
            if mul(&s, &w, &mul(&s, &conj(&w), &x)) != scaled(&x, nrm(&w)) { whole_fail += 1; wf += 1; }
            if mul(&s, &w_in, &mul(&s, &conj(&w_in), &x)) == scaled(&x, nrm(&w_in)) { in_ok += 1; iok += 1; }
            if mul(&s, &w_out, &mul(&s, &conj(&w_out), &x)) == scaled(&x, nrm(&w_out)) { out_ok += 1; ook += 1; }
            let b0 = mul(&s, &conj(&w), &x)[0]; let b0s = mul(&s, &conj(&w_in), &x)[0] + mul(&s, &conj(&w_out), &x)[0];
            let dot: i128 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
            if b0 == b0s && b0 == dot { band0_ok += 1; }
        }
        if wf > 0 { px_whole_fail += 1; } if iok == 10 { px_in_ok += 1; } if ook == 10 { px_out_ok += 1; }
    }
    println!("[P4] {}  4096 real pixels x 10 x: whole-pixel rewind fails on {px_whole_fail}/4096 pixels ({whole_fail}/{n} pairs); inner octet rewinds on {px_in_ok}/4096 ({in_ok}/{n}); outer octet on {px_out_ok}/4096 ({out_ok}/{n})",
             if px_in_ok == 4096 && px_out_ok == 4096 && px_whole_fail >= 4055 { "PASS" } else { "FAIL" });
    println!("[P5] {}  band 0 of conj(w) x = band 0 of conj(w_in) x + band 0 of conj(w_out) x = w . x on {band0_ok}/{n}",
             if band0_ok == n { "PASS" } else { "FAIL" });

    // P6: the 42 zero pairs and the 336 zero products, from the table
    let mut zero_products = 0usize; let mut pairs = std::collections::BTreeSet::new();
    for a in 0..16 { for b in (a + 1)..16 { for c in 0..16 { for d in (c + 1)..16 {
        for sa in [1i128, -1] { for sc in [1i128, -1] {
            let mut x = e(a); x[b] = sa; let mut y = e(c); y[d] = sc;
            if mul(&s, &x, &y).iter().all(|v| *v == 0) { zero_products += 1; pairs.insert((a, b)); }
        } }
    } } } }
    let reading: usize = (1..8).map(|o| (9..16).filter(|s| *s != o + 8).count()).sum();
    println!("[P6] note  zero products (e_a +- e_b)(e_c +- e_d) = 0: {zero_products}; distinct left factor pairs: {}; the reading's count 7 x 6 = {reading}",
             pairs.len());
}
