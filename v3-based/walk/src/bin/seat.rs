// S0: seat the weights, once. Every local u16 tensor of Qwen3-1.7B (all 28 layers, the embedding, the final norm) becomes an
// ISIS3 cube of positions: one UnsignedWord per weight, the rule and the band floor in the label, the tensor in its own
// order (pack.py's rule):  code = (band - floor) * 256 + sign * 128 + (odd - 1) / 2;  code 0 is Null, the zero.
// After this the runtime (pix.rs) reads seats only. The weight files are never opened again.
// Run from walk/:  cargo run --release --bin seat
use std::fs;

const WDIRS: [&str; 2] = ["../ARCHIVE-2026-09-25-mistaken/weights", "../t10"];   // layers 0..13, then the fetched rest
const SDIR: &str = "../seats";

fn sha256_hex(data: &[u8]) -> String {
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

/// The u16 as a position: (sign, odd address 1..255, band), or None for zero.
fn position(b: u16) -> Option<(u8, u32, i32)> {
    let s = (b >> 15) as u8; let ef = ((b >> 7) & 0xFF) as i32; let f = (b & 0x7F) as u32;
    assert!(ef != 255, "not a position");
    let m = if ef > 0 { 128 + f } else { f };
    if m == 0 { return None; }
    let t = m.trailing_zeros() as i32; Some((s, m >> t, (if ef > 0 { ef - 134 } else { -133 }) + t))
}

fn shape(name: &str) -> (usize, usize) {
    match name {
        "self_attn.q_proj" | "self_attn.o_proj" => (2048, 2048),
        "self_attn.k_proj" | "self_attn.v_proj" => (1024, 2048),
        "mlp.gate_proj" | "mlp.up_proj" => (6144, 2048),
        "mlp.down_proj" => (2048, 6144),
        "input_layernorm" | "post_attention_layernorm" => (1, 2048),
        "self_attn.q_norm" | "self_attn.k_norm" => (1, 128),
        "embed_tokens" => (151936, 2048),
        "norm" => (1, 2048),
        _ => panic!("{name}"),
    }
}

fn main() {
    fs::create_dir_all(SDIR).unwrap();
    let names = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm", "self_attn.q_proj",
                 "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
    let (mut n_t, mut raw_b, mut seat_b, mut n_w) = (0usize, 0usize, 0usize, 0usize);
    let mut all = Vec::new();
    let mut jobs: Vec<(String, String, &str)> = Vec::new();          // (source file, seat name, shape name)
    for l in 0..28 { for name in names { jobs.push((format!("model.layers.{l}.{name}.weight.u16"), format!("L{l}.{name}"), name)); } }
    jobs.push(("model.embed_tokens.weight.u16".into(), "embed_tokens".into(), "embed_tokens"));
    jobs.push(("model.norm.weight.u16".into(), "norm".into(), "norm"));
    let mut skipped = 0usize;
    for (file, seat_name, name) in &jobs {
        let name = *name; let l = seat_name.clone();
        {
            let dst = format!("{SDIR}/{seat_name}.cub");
            if fs::metadata(&dst).is_ok() { skipped += 1; continue; }              // seated already: paid once
            let Some(src) = WDIRS.iter().map(|d| format!("{d}/{file}")).find(|p| fs::metadata(p).is_ok()) else { println!("  missing: {file}"); continue; };
            let raw = fs::read(&src).unwrap(); let raw_len = raw.len(); let raw_sha = sha256_hex(&raw);
            let (lines, samples) = shape(name); assert_eq!(raw_len / 2, lines * samples, "{src}");
            let bit = |i: usize| u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);                       // pass one: the band range
            for i in 0..lines * samples { if let Some((_, _, g)) = position(bit(i)) { lo = lo.min(g); hi = hi.max(g); } }
            let floor = lo - 1; let span = hi - floor;
            assert!(span <= 254, "band span too wide for one byte");
            let mut body: Vec<u8> = Vec::with_capacity(raw_len);                 // pass two: the codes
            for i in 0..lines * samples {
                let c: u16 = match position(bit(i)) { Some((s, odd, g)) => (((g - floor) as u16) << 8) | ((s as u16) << 7) | ((odd - 1) / 2) as u16, None => 0 };
                body.extend_from_slice(&c.to_le_bytes());
            }
            drop(raw); let body_sha = sha256_hex(&body);
            let label = |start: usize| format!(
                "Object = IsisCube\n  Object = Core\n    StartByte = {start}\n    Format = BandSequential\n    Group = Dimensions\n      Samples = {samples}\n      Lines = {lines}\n      Bands = 1\n    End_Group\n    Group = Pixels\n      Type = UnsignedWord\n      ByteOrder = Lsb\n      Base = 0.0\n      Multiplier = 1.0\n    End_Group\n  End_Object\n  Group = Atom\n    BandFloor = {floor}\n    BandSpan = {span}\n    Rule = \"code = (band - BandFloor) * 256 + sign * 128 + (odd - 1) / 2; value = +-odd * 2^band; Null (0) is zero\"\n    Source = \"Qwen3-1.7B {file} (bf16 as u16)\"\n    SourceSha256 = {}\n    CodeSha256 = {}\n  End_Group\nEnd_Object\nEnd\n",
                raw_sha, body_sha);
            let h = label(0); let start = label(h.len() + 1).len() + 1;
            let mut out = label(start).into_bytes(); out.resize(start - 1, b' '); out.extend_from_slice(&body);
            let dst = format!("{SDIR}/{seat_name}.cub"); fs::write(&dst, &out).unwrap();
            n_t += 1; raw_b += raw_len; seat_b += out.len(); n_w += lines * samples;
            all.extend_from_slice(body_sha.as_bytes());                         // the hash of the code hashes, in order
            println!("  {l:34} {lines:6} x {samples:5}  floor {floor:4}  span {span:3}  seat {} B  source {}", out.len(), &raw_sha[..16]);
        }
    }
    println!("[S0] {skipped} tensors were seated already and were not touched");
    println!("[S0] {n_t} tensors, {n_w} weights seated in {SDIR}/ as ISIS3 cubes: {seat_b} B against raw {raw_b} B ({:.4}x, the rest is the labels)", seat_b as f64 / raw_b as f64);
    println!("[S0] sha256 of the tensors' code hashes in order (this pass only): {}", sha256_hex(&all));
}
