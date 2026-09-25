// T13 (prereg_T13.txt, sealed f6e331d5…): the fold between layers. Target slice (l+1, h) coded with contexts taken
// only from a reference cell at the same (row, col): SAME (l, h), OTHER (l, h+1 mod H), SHUF (l, h) shuffled, NONE.
// T12's verified range coder (walk::rc). Every SAME stream is decoded back and compared cell by cell.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use walk::rc::{bucket, shuffle, Dec, Enc};

// (name, rows, cols, heads, heads-along-columns)
const KINDS: [(&str, usize, usize, usize, bool); 7] = [
    ("self_attn.q_proj", 2048, 2048, 16, false),
    ("self_attn.k_proj", 1024, 2048, 8, false),
    ("self_attn.v_proj", 1024, 2048, 8, false),
    ("self_attn.o_proj", 2048, 2048, 16, true),
    ("mlp.gate_proj", 6144, 2048, 0, false),
    ("mlp.up_proj", 6144, 2048, 0, false),
    ("mlp.down_proj", 2048, 6144, 0, false),
];
const PAIRS: usize = 13; // (l, l+1), l = 0..12
const REFS: [&str; 4] = ["SAME", "OTHER", "SHUF", "NONE"];

fn load(layer: usize, kind: usize) -> Vec<u16> {
    let (name, r, c, _, _) = KINDS[kind];
    let path = format!("../t10/model.layers.{layer}.{name}.weight.u16");
    let b = std::fs::read(&path).expect(&path);
    assert_eq!(b.len(), r * c * 2, "{path}");
    b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect()
}

// Head h's slice, row-major within the slice.
fn slice(w: &[u16], kind: usize, h: usize) -> Vec<u16> {
    let (_, r, c, _, by_col) = KINDS[kind];
    if by_col {
        (0..r).flat_map(|i| w[i * c + 128 * h..i * c + 128 * h + 128].iter().copied()).collect()
    } else {
        w[128 * h * c..128 * (h + 1) * c].to_vec()
    }
}

struct Models {
    exp: Vec<u16>,
    sign: Vec<u16>,
    mant: Vec<u16>,
}
impl Models {
    fn new() -> Self {
        Models { exp: vec![1024; 16 * 256], sign: vec![1024; 2], mant: vec![1024; 4 * 128] }
    }
}

fn ctx(r: u16) -> (usize, usize, usize) {
    (bucket(r) * 256, (r >> 15) as usize, (((r >> 5) & 3) as usize) * 128)
}

fn encode(t: &[u16], rf: &[u16]) -> [Vec<u8>; 3] {
    let mut m = Models::new();
    let (mut ee, mut es, mut em) = (Enc::new(), Enc::new(), Enc::new());
    for i in 0..t.len() {
        let (ce, cs, cm) = ctx(rf[i]);
        let v = t[i];
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

fn decode(s: &[Vec<u8>; 3], rf: &[u16]) -> Vec<u16> {
    let mut m = Models::new();
    let (mut de, mut ds, mut dm) = (Dec::new(&s[0]), Dec::new(&s[1]), Dec::new(&s[2]));
    rf.iter()
        .map(|&r| {
            let (ce, cs, cm) = ctx(r);
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
            sg << 15 | e << 7 | (node - 128) as u16
        })
        .collect()
}

#[derive(Clone, Default)]
struct Row {
    pair: usize,
    kind: usize,
    head: usize, // usize::MAX for a whole MLP tensor
    n: u64,
    bytes: [[u64; 3]; 4], // [ref][EXP, SIGN, MANT]
    diff: u64,
}

// One target against its four references.
fn measure(pair: usize, kind: usize, head: usize, t: &[u16], same: Vec<u16>, other: Vec<u16>) -> Row {
    let mut shuf = same.clone();
    shuffle(&mut shuf);
    let none = vec![0u16; t.len()];
    let mut row = Row { pair, kind, head, n: t.len() as u64, ..Default::default() };
    for (k, rf) in [&same, &other, &shuf, &none].into_iter().enumerate() {
        let s = encode(t, rf);
        row.bytes[k] = [s[0].len() as u64, s[1].len() as u64, s[2].len() as u64];
        if k == 0 {
            let back = decode(&s, rf);
            row.diff = t.iter().zip(&back).filter(|(a, b)| a != b).count() as u64;
        }
    }
    row
}

fn r4(a: u64, b: u64) -> String {
    let x = (a as u128 * 10000 + b as u128 / 2) / b as u128;
    format!("{}.{:04}", x / 10000, x % 10000)
}

fn tot(b: &[u64; 3]) -> u64 {
    b.iter().sum()
}

fn main() {
    let jobs: Vec<(usize, usize)> = (0..PAIRS).flat_map(|l| (0..KINDS.len()).map(move |k| (l, k))).collect();
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::<Row>::new());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= jobs.len() {
                    break;
                }
                let (l, kind) = jobs[j];
                let (name, r, c, heads, _) = KINDS[kind];
                let (lo, hi) = (load(l, kind), load(l + 1, kind));
                let mut out = Vec::new();
                if heads > 0 {
                    for h in 0..heads {
                        let t = slice(&hi, kind, h);
                        out.push(measure(l, kind, h, &t, slice(&lo, kind, h), slice(&lo, kind, (h + 1) % heads)));
                    }
                } else {
                    // Whole MLP tensor. OTHER (exploratory, not fixed by the registration): the reference with its
                    // rows rotated by 128, i.e. a different block of rows at the same layer.
                    let mut other = lo.clone();
                    other.rotate_left(128 * c);
                    out.push(measure(l, kind, usize::MAX, &hi, lo, other));
                }
                let s = |k: usize| out.iter().map(|x| tot(&x.bytes[k])).sum::<u64>();
                let d: u64 = out.iter().map(|x| x.diff).sum();
                eprintln!("L{l:2}->{:2} {name:18} {r}x{c} SAME/SHUF {} SAME/OTHER {} SAME/NONE {} diff {d}", l + 1, r4(s(0), s(2)), r4(s(0), s(1)), r4(s(0), s(3)));
                rows.lock().unwrap().extend(out);
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|x| (x.pair, x.kind, x.head));
    let secs = t0.elapsed().as_secs();

    let heads: Vec<&Row> = rows.iter().filter(|x| x.head != usize::MAX).collect();
    let mlp: Vec<&Row> = rows.iter().filter(|x| x.head == usize::MAX).collect();
    let sum = |rs: &[&Row], k: usize, g: Option<usize>| -> u64 {
        rs.iter().map(|x| match g { Some(g) => x.bytes[k][g], None => tot(&x.bytes[k]) }).sum()
    };
    let diff: u64 = rows.iter().map(|x| x.diff).sum();
    let cells: u64 = rows.iter().map(|x| x.n).sum();

    println!("T13 run (walk/src/bin/t13.rs), prereg sha f6e331d5…, {} head slices + {} MLP tensors, {threads} threads, {secs} s", heads.len(), mlp.len());
    println!("Receipt: {cells} target cells decoded back from the SAME streams, {diff} different{}", if diff == 0 { "" } else { " — T13 VOID" });
    println!();
    println!("Head slices (q, k, v, o; l -> l+1, l = 0..12), total bytes:");
    for (k, name) in REFS.iter().enumerate() {
        let b = [sum(&heads, k, Some(0)), sum(&heads, k, Some(1)), sum(&heads, k, Some(2))];
        println!("  {name:5} {:>11} B   (EXP {:>10}, SIGN {:>9}, MANT {:>10})", tot(&b), b[0], b[1], b[2]);
    }
    let (same, other, shuf, none) = (sum(&heads, 0, None), sum(&heads, 1, None), sum(&heads, 2, None), sum(&heads, 3, None));
    let p1 = same * 100 <= shuf * 99;
    let p2 = same * 100 <= other * 99;
    println!("P1 {}: SAME/SHUF {} (needs <= 0.9900)", if p1 { "HELD" } else { "KILLED" }, r4(same, shuf));
    println!("P2 {}: SAME/OTHER {} (needs <= 0.9900)", if p2 { "HELD" } else { "KILLED" }, r4(same, other));
    println!();
    println!("Exploratory:");
    println!("  heads, SAME/NONE {}, OTHER/SHUF {}, SHUF/NONE {}", r4(same, none), r4(other, shuf), r4(shuf, none));
    println!("  heads, by plane group SAME/SHUF: EXP {}, SIGN {}, MANT {}",
        r4(sum(&heads, 0, Some(0)), sum(&heads, 2, Some(0))), r4(sum(&heads, 0, Some(1)), sum(&heads, 2, Some(1))), r4(sum(&heads, 0, Some(2)), sum(&heads, 2, Some(2))));
    println!("  by kind: SAME/SHUF, SAME/OTHER, OTHER/SHUF, SAME/NONE");
    for k in 0..KINDS.len() {
        let rs: Vec<&Row> = rows.iter().filter(|x| x.kind == k).collect();
        let s = |i| sum(&rs, i, None);
        println!("    {:18} {} {} {} {}", KINDS[k].0, r4(s(0), s(2)), r4(s(0), s(1)), r4(s(1), s(2)), r4(s(0), s(3)));
    }
    println!("  by layer pair (heads only): SAME/SHUF, SAME/OTHER");
    for l in 0..PAIRS {
        let rs: Vec<&Row> = heads.iter().copied().filter(|x| x.pair == l).collect();
        println!("    L{l:<2}->L{:<2} {} {}", l + 1, r4(sum(&rs, 0, None), sum(&rs, 2, None)), r4(sum(&rs, 0, None), sum(&rs, 1, None)));
    }
    let wins = heads.iter().filter(|x| tot(&x.bytes[0]) < tot(&x.bytes[1])).count();
    println!("  head slices where SAME < OTHER: {wins}/{}", heads.len());
    let mut best: Vec<&&Row> = heads.iter().collect();
    best.sort_by_key(|x| (tot(&x.bytes[0]) as u128 * 1_000_000 / tot(&x.bytes[1]) as u128) as u64);
    println!("  strongest SAME/OTHER slices:");
    for x in best.iter().take(8) {
        println!("    L{}->L{} {} h{:<2} SAME/OTHER {} SAME/SHUF {}", x.pair, x.pair + 1, KINDS[x.kind].0, x.head, r4(tot(&x.bytes[0]), tot(&x.bytes[1])), r4(tot(&x.bytes[0]), tot(&x.bytes[2])));
    }
}
