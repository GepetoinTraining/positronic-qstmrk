// The geometric read, std only, exact values. Qwen3-1.7B layers 0..13 from the archived u16 weights, on the captured
// tokens (u16 positions), against the captured layer-1 and layer-14 states.
//
// A value is a position: sign, odd, band (value = +-odd * 2^band). A weight is its u16 read as a position: sign bit,
// band field, address field (odd <= 255, one of 128 atoms). An activation is a pixel of the same family: it lands on
// 8 bits where it enters a projection (the one rounding, half to even), and from there it is an address and a band.
// A weight meeting an activation is one move: PROD[weight address][activation address] (built once, by addition),
// landing on band = weight band + activation band, sign = sign xor sign. A projection is the landing counts of its
// octets: each weight octet read once, meeting every token's octet in one pass. Each (row, token) is joined once: a
// carry pass over its band counts, landing on a pixel (top 8 bits, half to even). No weight is multiplied by an activation.
// The pixel is the octet: 8 weights on 8 anchors, an octonion, closed with exact rewind (T15). Two octets are two
// pixels; the 16-anchor sedenion pixel mixes them and its norm leaks (the defect). Band 0, the dot product, is the same.
// Norms, rotary, softmax and SiLU are kept as they are (outside the object).
use std::fs;
use std::thread;
use std::time::Instant;

const HID: usize = 2048;
const H: usize = 16;
const KV: usize = 8;
const D: usize = 128;
const INTER: usize = 6144;
const TOK: usize = 64;
const WDIR: &str = "../ARCHIVE-2026-09-25-mistaken/weights";
const CAP: &str = "../../receipts";

fn read_u16(path: &str) -> Vec<u16> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// 2^e placed on v by moving its exponent field (no multiply).
fn place(v: f64, e: i32) -> f64 {
    if v == 0.0 { return 0.0; }
    let b = v.to_bits(); let ex = ((b >> 52) & 0x7FF) as i32 + e;
    assert!(ex > 0 && ex < 0x7FF, "out of range");
    f64::from_bits((b & !(0x7FFu64 << 52)) | ((ex as u64) << 52))
}

// ---------- weights: the u16 as a position ----------
struct W { idx: Vec<u8>, neg: Vec<u8>, band: Vec<i16>, lo: i32, hi: i32 }
const NONE: i16 = i16::MIN;
fn stored_position(b: u16) -> (u8, u8, i16) {
    let s = (b >> 15) as u8; let ef = ((b >> 7) & 0xFF) as i32; let f = (b & 0x7F) as u32;
    assert!(ef != 255, "not a position");
    let m = if ef > 0 { 128 + f } else { f };
    if m == 0 { return (0, 0, NONE); }
    let t = m.trailing_zeros(); let odd = m >> t;
    (((odd - 1) / 2) as u8, s, ((if ef > 0 { ef - 134 } else { -133 }) + t as i32) as i16)
}
fn stored_value(b: u16) -> f64 {
    let (i, s, g) = stored_position(b); if g == NONE { return 0.0; }
    let v = place((2 * i as u32 + 1) as f64, g as i32); if s == 1 { -v } else { v }
}
fn seat(bits: &[u16]) -> W {
    let mut w = W { idx: Vec::with_capacity(bits.len()), neg: Vec::with_capacity(bits.len()), band: Vec::with_capacity(bits.len()), lo: i32::MAX, hi: i32::MIN };
    for &b in bits {
        let (i, s, g) = stored_position(b); w.idx.push(i); w.neg.push(s); w.band.push(g);
        if g != NONE { w.lo = w.lo.min(g as i32); w.hi = w.hi.max(g as i32); }
    }
    w
}

// ---------- activations: pixels of the weight family (sign, address of an odd <= 255, band) ----------
const NOX: X = X { neg: 0, band: 0, idx: 0, live: false };
#[derive(Clone, Copy)]
struct X { neg: u8, band: i32, idx: u8, live: bool }
fn pixel(v: f64) -> X {
    if v == 0.0 { return NOX; }
    let b = v.to_bits(); let ex = ((b >> 52) & 0x7FF) as i32; let frac = b & ((1u64 << 52) - 1);
    let (mut m, mut e) = if ex == 0 { (frac, -1074) } else { (frac | (1u64 << 52), ex - 1075) };
    let top = 64 - m.leading_zeros() as i32;
    if top > 8 {                                  // land on 8 bits: the one rounding, half to even
        let sh = (top - 8) as u32; let low = m & ((1u64 << sh) - 1); let half = 1u64 << (sh - 1);
        m >>= sh; e += sh as i32;
        if low > half || (low == half && m & 1 == 1) { m += 1; }
    }
    let t = m.trailing_zeros(); let odd = m >> t;   // 256 after a round-up is 1 on the next band
    X { neg: (b >> 63) as u8, band: e + t as i32, idx: ((odd - 1) / 2) as u8, live: true }
}

// ---------- the table, built once: PROD[a][b] = odd(a) * odd(b), by repeated addition ----------
fn table() -> Vec<u16> {
    let mut t = vec![0u16; 128 * 128];
    for a in 0..128usize { let step = (2 * a + 1) as u32; let mut acc = step; for b in 0..128usize { t[a * 128 + b] = acc as u16; acc += 2 * step; } }
    t
}

/// The oscillator bank, built once: the rotary's turn for every (position, frequency pair), the declared input.
/// A turn acts on the pair (d, d+64): the real wave and the i-wave.
fn oscillator(t_n: usize) -> Vec<(f64, f64)> {
    let mut o = Vec::with_capacity(t_n * D / 2);
    for pos in 0..t_n { for i in 0..D / 2 { let a = pos as f64 * 1e6f64.powf(-((2 * i) as f64) / D as f64); o.push((a.cos(), a.sin())); } }
    o
}

// ---------- the join: carry pass over band counts, value placed by shifting ----------
fn join(s: &[i64], g0: i32, bits: &mut Vec<u8>) -> f64 {
    bits.clear(); let mut carry: i64 = 0;
    for &c in s { let v = c + carry; let bit = v & 1; bits.push(bit as u8); carry = (v - bit) >> 1; }
    while carry != 0 && carry != -1 { let bit = carry & 1; bits.push(bit as u8); carry = (carry - bit) >> 1; }
    let negative = carry == -1;
    if negative {                                 // magnitude of a two's-complement digit string: invert, add one
        let mut c = 1u8; for b in bits.iter_mut() { let v = (1 - *b) + c; *b = v & 1; c = v >> 1; }
        if c == 1 { bits.push(1); }
    }
    let Some(top) = bits.iter().rposition(|&b| b == 1) else { return 0.0; };
    let lo = top.saturating_sub(7);               // the pixel: the top 8 bits of the landing, nothing else leaves
    let mut m: u64 = 0; for k in (lo..=top).rev() { m = (m << 1) | bits[k] as u64; }
    if lo > 0 {                                   // half to even, with sticky; 256 is 1 on the next band
        let half = bits[lo - 1] == 1; let sticky = bits[..lo - 1].iter().any(|&b| b == 1);
        if half && (sticky || m & 1 == 1) { m += 1; }
    }
    let v = place(m as f64, g0 + lo as i32);      // the pixel carried exactly as an f64 for the outside ops
    if negative { -v } else { v }
}

/// The octet read. out[t][r] for x[t] (T x cols).
fn project(w: &W, rows: usize, cols: usize, x: &[Vec<f64>], tab: &[u16]) -> Vec<Vec<f64>> {
    let t_n = x.len();
    let xs: Vec<Vec<X>> = x.iter().map(|v| v.iter().map(|&f| pixel(f)).collect()).collect();
    let live = || xs.iter().flatten().filter(|p| p.live);
    let (xlo, xhi) = (live().map(|p| p.band).min().unwrap(), live().map(|p| p.band).max().unwrap());
    let g0 = w.lo + xlo; let gn = (w.hi - w.lo + xhi - xlo + 1) as usize;
    let oct = cols / 8;
    let mut xo = vec![NOX; oct * t_n * 8];        // [octet][token][8]
    for (t, row) in xs.iter().enumerate() {
        for (j, p) in row.iter().enumerate() { let mut q = *p; q.band -= xlo; xo[((j / 8) * t_n + t) * 8 + j % 8] = q; }
    }
    let nthreads = 8; let chunk = rows.div_ceil(nthreads);
    let parts: Vec<(usize, Vec<Vec<f64>>)> = thread::scope(|sc| {
        let hs: Vec<_> = (0..nthreads).map(|th| {
            let xo = &xo;
            sc.spawn(move || {
                let r_lo = th * chunk; let r_hi = ((th + 1) * chunk).min(rows);
                let mut part = vec![vec![0f64; r_hi.saturating_sub(r_lo)]; t_n];
                let mut s = vec![0i64; t_n * gn]; let mut bits = Vec::with_capacity(gn + 64);
                for r in r_lo..r_hi {
                    s.iter_mut().for_each(|v| *v = 0);
                    for o in 0..oct {
                        let wb0 = r * cols + o * 8;
                        for a in 0..8 {
                            let wb = w.band[wb0 + a]; if wb == NONE { continue; }
                            let trow = &tab[(w.idx[wb0 + a] as usize) * 128..][..128];
                            let wneg = w.neg[wb0 + a]; let woff = (wb as i32 - w.lo) as usize;
                            for t in 0..t_n {
                                let p = &xo[(o * t_n + t) * 8 + a]; if !p.live { continue; }
                                let v = trow[p.idx as usize] as i64;                    // the move
                                let cell = &mut s[t * gn + woff + p.band as usize];      // the landing
                                if wneg ^ p.neg != 0 { *cell -= v } else { *cell += v }
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
    let mut out = vec![vec![0f64; rows]; t_n];
    for (r_lo, part) in parts { for (t, row) in part.into_iter().enumerate() { out[t][r_lo..r_lo + row.len()].copy_from_slice(&row); } }
    out
}

// ---------- the layer (norms, rotary, softmax, SiLU kept as they are) ----------
fn rms(x: &[f64], g: &[f64]) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let k = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(g).map(|(v, w)| v * k * w).collect()
}
fn spin(v: &mut [f64], pos: usize, osc: &[(f64, f64)]) {
    for i in 0..D / 2 {
        let (c, s) = osc[pos * (D / 2) + i];                 // the turn, read from the bank
        let (x1, x2) = (v[i], v[i + D / 2]);
        v[i] = x1 * c - x2 * s; v[i + D / 2] = x2 * c + x1 * s;
    }
}

struct Layer { q: W, k: W, v: W, o: W, gate: W, up: W, down: W, n_in: Vec<f64>, n_post: Vec<f64>, n_q: Vec<f64>, n_k: Vec<f64> }
fn load_layer(l: usize) -> Layer {
    let p = |n: &str| format!("{WDIR}/model.layers.{l}.{n}.weight.u16");
    let norm = |n: &str| read_u16(&p(n)).into_iter().map(stored_value).collect::<Vec<_>>();
    Layer { q: seat(&read_u16(&p("self_attn.q_proj"))), k: seat(&read_u16(&p("self_attn.k_proj"))), v: seat(&read_u16(&p("self_attn.v_proj"))),
            o: seat(&read_u16(&p("self_attn.o_proj"))), gate: seat(&read_u16(&p("mlp.gate_proj"))), up: seat(&read_u16(&p("mlp.up_proj"))),
            down: seat(&read_u16(&p("mlp.down_proj"))), n_in: norm("input_layernorm"), n_post: norm("post_attention_layernorm"),
            n_q: norm("self_attn.q_norm"), n_k: norm("self_attn.k_norm") }
}

fn forward(r: &mut Vec<Vec<f64>>, ly: &Layer, tab: &[u16], osc: &[(f64, f64)]) {
    let t = r.len();
    let x: Vec<Vec<f64>> = r.iter().map(|v| rms(v, &ly.n_in)).collect();
    let q = project(&ly.q, H * D, HID, &x, tab);
    let k = project(&ly.k, KV * D, HID, &x, tab);
    let v = project(&ly.v, KV * D, HID, &x, tab);
    let mut qh = vec![vec![vec![0f64; D]; t]; H]; let mut kh = vec![vec![vec![0f64; D]; t]; KV];
    for ti in 0..t {
        for h in 0..H { let mut z = rms(&q[ti][h * D..(h + 1) * D], &ly.n_q); spin(&mut z, ti, osc); qh[h][ti] = z; }
        for h in 0..KV { let mut z = rms(&k[ti][h * D..(h + 1) * D], &ly.n_k); spin(&mut z, ti, osc); kh[h][ti] = z; }
    }
    let heads: Vec<Vec<Vec<f64>>> = thread::scope(|sc| {
        let hs: Vec<_> = (0..H).map(|h| {
            let (qh, kh, v) = (&qh, &kh, &v);
            sc.spawn(move || {
                let kv = h / (H / KV); let mut out = vec![vec![0f64; D]; t];
                for i in 0..t {
                    let sc: Vec<f64> = (0..=i).map(|j| qh[h][i].iter().zip(&kh[kv][j]).map(|(a, b)| a * b).sum::<f64>() / (D as f64).sqrt()).collect();
                    let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect(); let z: f64 = e.iter().sum();
                    for (j, w) in e.iter().enumerate() { for d in 0..D { out[i][d] += w / z * v[j][kv * D + d]; } }
                }
                out
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut att = vec![vec![0f64; H * D]; t];
    for (h, out) in heads.into_iter().enumerate() { for i in 0..t { att[i][h * D..(h + 1) * D].copy_from_slice(&out[i]); } }
    let o = project(&ly.o, HID, H * D, &att, tab);
    for ti in 0..t { for d in 0..HID { r[ti][d] += o[ti][d]; } }
    let x2: Vec<Vec<f64>> = r.iter().map(|v| rms(v, &ly.n_post)).collect();
    let g = project(&ly.gate, INTER, HID, &x2, tab);
    let u = project(&ly.up, INTER, HID, &x2, tab);
    let act: Vec<Vec<f64>> = (0..t).map(|ti| (0..INTER).map(|j| g[ti][j] / (1.0 + (-g[ti][j]).exp()) * u[ti][j]).collect()).collect();
    let dn = project(&ly.down, HID, INTER, &act, tab);
    for ti in 0..t { for d in 0..HID { r[ti][d] += dn[ti][d]; } }
}

fn compare(r: &[Vec<f64>], l: usize) -> String {
    let cap = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L{l}-508tok.u16"));
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for ti in 0..r.len() { for d in 0..HID { let (x, y) = (r[ti][d], stored_value(cap[ti * HID + d])); dot += x * y; na += x * x; nb += y * y; } }
    format!("cosine {:.6} against the captured state (instrument)", dot / (na * nb).sqrt())
}

fn main() {
    let tab = table();
    let osc = oscillator(TOK);
    let cap0 = read_u16(&format!("{CAP}/capture-qwen3-1.7b-L0-508tok.u16"));
    let mut r: Vec<Vec<f64>> = (0..TOK).map(|t| cap0[t * HID..(t + 1) * HID].iter().map(|&b| stored_value(b)).collect()).collect();
    let (mut seat_s, mut run_s) = (0f64, 0f64);
    for l in 0..14 {
        let ts = Instant::now(); let ly = load_layer(l); seat_s += ts.elapsed().as_secs_f64();
        let tr = Instant::now(); forward(&mut r, &ly, &tab, &osc); run_s += tr.elapsed().as_secs_f64();
        if l == 0 { println!("[L0->L1]  {}", compare(&r, 1)); }
    }
    println!("[L0->L14] {}", compare(&r, 14));
    println!("seat {seat_s:.1}s; 14 layers x {TOK} tokens {run_s:.1}s");
}
