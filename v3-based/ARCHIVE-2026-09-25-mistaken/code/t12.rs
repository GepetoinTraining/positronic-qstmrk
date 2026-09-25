// T12 (prereg_T12.txt, sealed 69b68544…): the seismograph's bit-planes, real against shuffled.
// Integer LZMA-style binary range coder; three streams per tensor (EXP, SIGN, MANT); causal contexts
// (own prefix x left/up neighbours). Every real stream is decoded back and compared pattern by pattern.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const KINDS: [(&str, usize, usize); 7] = [
    ("self_attn.q_proj", 2048, 2048),
    ("self_attn.k_proj", 1024, 2048),
    ("self_attn.v_proj", 1024, 2048),
    ("self_attn.o_proj", 2048, 2048),
    ("mlp.gate_proj", 6144, 2048),
    ("mlp.up_proj", 6144, 2048),
    ("mlp.down_proj", 2048, 6144),
];
const LAYERS: usize = 14;

struct Enc {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}
impl Enc {
    fn new() -> Self {
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
    fn bit(&mut self, p: &mut u16, b: u32) {
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
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct Dec<'a> {
    code: u32,
    range: u32,
    src: &'a [u8],
    pos: usize,
}
impl<'a> Dec<'a> {
    fn new(src: &'a [u8]) -> Self {
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
    fn bit(&mut self, p: &mut u16) -> u32 {
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

fn bucket(p: u16) -> usize {
    let e = ((p >> 7) & 0xFF) as usize;
    e.clamp(110, 125) - 110
}

struct Models {
    exp: Vec<u16>,  // 16 x 16 x 256
    sign: Vec<u16>, // 8
    mant: Vec<u16>, // 16 x 128
}
impl Models {
    fn new() -> Self {
        Models { exp: vec![1024; 16 * 16 * 256], sign: vec![1024; 8], mant: vec![1024; 16 * 128] }
    }
}

// Causal neighbours of cell i in a rows x cols raster: (left, up, left-left); absent = 0.
fn nbrs(p: &[u16], i: usize, cols: usize) -> (u16, u16, u16) {
    let c = i % cols;
    let l = if c >= 1 { p[i - 1] } else { 0 };
    let u = if i >= cols { p[i - cols] } else { 0 };
    let ll = if c >= 2 { p[i - 2] } else { 0 };
    (l, u, ll)
}

fn ctx(l: u16, u: u16, ll: u16) -> (usize, usize, usize) {
    let e = (bucket(l) * 16 + bucket(u)) * 256;
    let s = ((l >> 15) as usize) << 2 | ((u >> 15) as usize) << 1 | (ll >> 15) as usize;
    let m = ((((l >> 5) & 3) as usize) * 4 + ((u >> 5) & 3) as usize) * 128;
    (e, s, m)
}

fn encode(p: &[u16], cols: usize) -> [Vec<u8>; 3] {
    let mut m = Models::new();
    let (mut ee, mut es, mut em) = (Enc::new(), Enc::new(), Enc::new());
    for i in 0..p.len() {
        let (l, u, ll) = nbrs(p, i, cols);
        let (ce, cs, cm) = ctx(l, u, ll);
        let v = p[i];
        let e = ((v >> 7) & 0xFF) as u32;
        let mut node = 1usize;
        for k in (0..8).rev() {
            let b = (e >> k) & 1;
            ee.bit(&mut m.exp[ce + node], b);
            node = node * 2 + b as usize;
        }
        es.bit(&mut m.sign[cs], (v >> 15) as u32);
        let mt = (v & 0x7F) as u32;
        let mut node = 1usize;
        for k in (0..7).rev() {
            let b = (mt >> k) & 1;
            em.bit(&mut m.mant[cm + node], b);
            node = node * 2 + b as usize;
        }
    }
    [ee.finish(), es.finish(), em.finish()]
}

fn decode(s: &[Vec<u8>; 3], n: usize, cols: usize) -> Vec<u16> {
    let mut m = Models::new();
    let (mut de, mut ds, mut dm) = (Dec::new(&s[0]), Dec::new(&s[1]), Dec::new(&s[2]));
    let mut p = vec![0u16; n];
    for i in 0..n {
        let (l, u, ll) = nbrs(&p, i, cols);
        let (ce, cs, cm) = ctx(l, u, ll);
        let mut node = 1usize;
        for _ in 0..8 {
            node = node * 2 + de.bit(&mut m.exp[ce + node]) as usize;
        }
        let e = (node - 256) as u16;
        let sg = ds.bit(&mut m.sign[cs]) as u16;
        let mut node = 1usize;
        for _ in 0..7 {
            node = node * 2 + dm.bit(&mut m.mant[cm + node]) as usize;
        }
        let mt = (node - 128) as u16;
        p[i] = sg << 15 | e << 7 | mt;
    }
    p
}

fn shuffle(p: &mut [u16]) {
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

#[derive(Clone, Default)]
struct Row {
    layer: usize,
    kind: usize,
    n: u64,
    real: [u64; 3],
    shuf: [u64; 3],
    diff: u64,
}

// ratio a/b as a fixed-point string with 4 decimals, integer arithmetic only
fn r4(a: u64, b: u64) -> String {
    let x = (a as u128 * 10000 + b as u128 / 2) / b as u128;
    format!("{}.{:04}", x / 10000, x % 10000)
}
// bytes -> bits per weight, 3 decimals
fn bpw(bytes: u64, n: u64) -> String {
    let x = (bytes as u128 * 8 * 1000 + n as u128 / 2) / n as u128;
    format!("{}.{:03}", x / 1000, x % 1000)
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("selftest") {
        let mut p: Vec<u16> = (0..64u32 * 64).map(|i| (i.wrapping_mul(2654435761) >> 16) as u16).collect();
        p[0] = 0x3C00;
        let s = encode(&p, 64);
        let back = decode(&s, p.len(), 64);
        let first = p.iter().zip(&back).position(|(a, b)| a != b);
        println!("selftest: bytes {:?}, first mismatch {first:?}", s.iter().map(|x| x.len()).collect::<Vec<_>>());
        if let Some(i) = first {
            println!("  at {i}: want {:04x} got {:04x}", p[i], back[i]);
        }
        return;
    }
    let jobs: Vec<(usize, usize)> = (0..LAYERS).flat_map(|l| (0..KINDS.len()).map(move |k| (l, k))).collect();
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(vec![Row::default(); jobs.len()]);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= jobs.len() {
                    break;
                }
                let (layer, kind) = jobs[j];
                let (name, r, c) = KINDS[kind];
                let path = format!("../t10/model.layers.{layer}.{name}.weight.u16");
                let b = std::fs::read(&path).expect(&path);
                assert_eq!(b.len(), r * c * 2, "{path}");
                let p: Vec<u16> = b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
                let sr = encode(&p, c);
                let back = decode(&sr, p.len(), c);
                let diff = p.iter().zip(&back).filter(|(a, b)| a != b).count() as u64;
                let mut q = p.clone();
                shuffle(&mut q);
                let ss = encode(&q, c);
                let row = Row {
                    layer,
                    kind,
                    n: p.len() as u64,
                    real: [sr[0].len() as u64, sr[1].len() as u64, sr[2].len() as u64],
                    shuf: [ss[0].len() as u64, ss[1].len() as u64, ss[2].len() as u64],
                    diff,
                };
                eprintln!("L{layer:2} {name:18} exp {} sign {} mant {} diff {diff}", r4(row.real[0], row.shuf[0]), r4(row.real[1], row.shuf[1]), r4(row.real[2], row.shuf[2]));
                rows.lock().unwrap()[j] = row;
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    let secs = t0.elapsed().as_secs();

    println!("T12 run (walk/src/bin/t12.rs), prereg sha 69b68544…, {} tensors, {threads} threads, {secs} s", rows.len());
    println!("Columns: real/shuffled byte ratio per plane group (below 1 = the arrangement pays), then real bits/weight.");
    println!("{:>3} {:18} {:>9} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>7} | diff", "L", "tensor", "weights", "EXP", "SIGN", "MANT", "EXPb", "SGNb", "MNTb", "total");
    for r in &rows {
        let tot = r.real.iter().sum::<u64>();
        println!(
            "{:>3} {:18} {:>9} {} {} {} | {:>6} {:>6} {:>6} {:>7} | {}",
            r.layer, KINDS[r.kind].0, r.n,
            r4(r.real[0], r.shuf[0]), r4(r.real[1], r.shuf[1]), r4(r.real[2], r.shuf[2]),
            bpw(r.real[0], r.n), bpw(r.real[1], r.n), bpw(r.real[2], r.n), bpw(tot, r.n), r.diff
        );
    }
    let n: u64 = rows.iter().map(|r| r.n).sum();
    let sum = |f: &dyn Fn(&Row) -> u64| rows.iter().map(f).sum::<u64>();
    let (er, es) = (sum(&|r| r.real[0]), sum(&|r| r.shuf[0]));
    let (sr, ss) = (sum(&|r| r.real[1]), sum(&|r| r.shuf[1]));
    let (mr, ms) = (sum(&|r| r.real[2]), sum(&|r| r.shuf[2]));
    let diff = sum(&|r| r.diff);
    let exp_wins = rows.iter().filter(|r| r.real[0] < r.shuf[0]).count();

    println!();
    println!("Receipt: {n} weights decoded back from the real streams, {diff} different{}", if diff == 0 { "" } else { " — T12 VOID" });
    println!("Totals: EXP real {er} B, shuffled {es} B, ratio {}", r4(er, es));
    println!("        SIGN real {sr} B, shuffled {ss} B, ratio {}", r4(sr, ss));
    println!("        MANT real {mr} B, shuffled {ms} B, ratio {}", r4(mr, ms));
    let p1a = er * 100 <= es * 99;
    let p1b = exp_wins >= 88;
    println!(
        "P1 {}: EXP real/shuffled {} (needs <= 0.9900: {}); real < shuffled in {exp_wins}/98 tensors (needs >= 88: {})",
        if p1a && p1b { "HELD" } else { "KILLED" }, r4(er, es), p1a, p1b
    );
    let p2 = mr * 1000 >= ms * 995;
    println!("P2 {}: MANT real/shuffled {} (needs >= 0.9950)", if p2 { "HELD" } else { "KILLED" }, r4(mr, ms));

    println!();
    println!("Exploratory:");
    println!("  bits/weight, real:     EXP {} of 8, SIGN {} of 1, MANT {} of 7, total {} of 16", bpw(er, n), bpw(sr, n), bpw(mr, n), bpw(er + sr + mr, n));
    println!("  bits/weight, shuffled: EXP {} of 8, SIGN {} of 1, MANT {} of 7, total {} of 16", bpw(es, n), bpw(ss, n), bpw(ms, n), bpw(es + ss + ms, n));
    println!("  (v2's best table: 13.00 bits/weight)");
    println!("  By kind (layers 0..13): EXP ratio, SIGN ratio, MANT ratio, real total bits/weight");
    for (k, (name, _, _)) in KINDS.iter().enumerate() {
        let rs: Vec<&Row> = rows.iter().filter(|r| r.kind == k).collect();
        let s = |f: &dyn Fn(&Row) -> u64| rs.iter().map(|r| f(r)).sum::<u64>();
        let nk = s(&|r| r.n);
        println!(
            "    {name:18} {} {} {} {}",
            r4(s(&|r| r.real[0]), s(&|r| r.shuf[0])), r4(s(&|r| r.real[1]), s(&|r| r.shuf[1])),
            r4(s(&|r| r.real[2]), s(&|r| r.shuf[2])), bpw(s(&|r| r.real.iter().sum()), nk)
        );
    }
    println!("  By layer: EXP ratio, SIGN ratio, MANT ratio, real total bits/weight");
    for l in 0..LAYERS {
        let rs: Vec<&Row> = rows.iter().filter(|r| r.layer == l).collect();
        let s = |f: &dyn Fn(&Row) -> u64| rs.iter().map(|r| f(r)).sum::<u64>();
        let nl = s(&|r| r.n);
        println!(
            "    L{l:<2} {} {} {} {}",
            r4(s(&|r| r.real[0]), s(&|r| r.shuf[0])), r4(s(&|r| r.real[1]), s(&|r| r.shuf[1])),
            r4(s(&|r| r.real[2]), s(&|r| r.shuf[2])), bpw(s(&|r| r.real.iter().sum()), nl)
        );
    }
}
