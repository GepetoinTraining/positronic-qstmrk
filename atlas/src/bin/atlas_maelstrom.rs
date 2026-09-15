//! `atlas_maelstrom` — the map, sealed: the ladder with its rays, the twin seams, the mediant
//! tree's frontier, the question-mark images of the Fibonacci zigzag, and the rational partner
//! of every finite bf16 cell (the dictionary as one tree read two ways). Integers only.
//!
//!   atlas_maelstrom <root>
//!
//! Writes receipts/maelstrom.txt (the tables, the census, the sha256 of the partner table) and
//! receipts/maelstrom-partners.tsv (pattern, sign, m, n, partner p, q, tree depth).

use std::fmt::Write as _;
use std::path::PathBuf;

use atlas::cells::decompose;
use atlas::fiber::Fiber;
use atlas::maelstrom::*;
use num_bigint::BigInt;
use sha2::{Digest, Sha256};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&a[1]);
    let mut r = String::from("format=atlas.maelstrom.v1\n");

    // 2: the two rays
    writeln!(r, "\n== the two rays: phi^n = (F(n-1), F(n)) = a + b*phi, trace 2a+b = L(n) ==").unwrap();
    writeln!(r, "{:>3} {:>8} {:>8} {:>8} {:>6} {:>5}", "n", "a", "b", "trace", "norm", "L%5").unwrap();
    for n in 0..=13 {
        let x = phi_pow(n);
        writeln!(r, "{:>3} {:>8} {:>8} {:>8} {:>6} {:>5}", n, x.a, x.b, x.trace(), x.norm(), lucas(n) % 5).unwrap();
    }

    // the ladder
    writeln!(r, "\n== the ladder 1..13 ==").unwrap();
    writeln!(r, "trace ray only (Lucas): {:?}", ladder(13, false)).unwrap();
    writeln!(r, "two rays:               {:?}", ladder(13, true)).unwrap();
    writeln!(r, "{:>3} {:>10} {:>10}  factors", "n", "coord ray", "trace ray").unwrap();
    for n in 1..=13u128 {
        let (f, l) = ray_of(n);
        writeln!(r, "{:>3} {:>10} {:>10}  {:?}", n, f.map_or("-".to_string(), |i| format!("F{}", i)), l.map_or("-".to_string(), |i| format!("L{}", i)), factors(n)).unwrap();
    }

    // 3: the seam
    writeln!(r, "\n== twin seams (p, p+2, seam, factors, ray of p, ray of p+2) ==").unwrap();
    for (p, q, seam, fac) in twin_seams(120) {
        let ray = |v: u128| {
            let (f, l) = ray_of(v);
            match (f, l) {
                (Some(i), Some(j)) => format!("F{},L{}", i, j),
                (Some(i), None) => format!("F{}", i),
                (None, Some(j)) => format!("L{}", j),
                _ => "-".to_string(),
            }
        };
        writeln!(r, "({:>3},{:>3}) {:>4}  {:<16} {:<6} {:<6} {}", p, q, seam, format!("{:?}", fac), ray(p), ray(q), if seam % 6 == 0 { "6k" } else { "2^2" }).unwrap();
    }

    // 1: the mediant tree
    let (rows, seq) = stern_brocot(16);
    writeln!(r, "\n== mediant tree to depth 16: neighbours unimodular = {} ==", neighbours_unimodular(&seq)).unwrap();
    writeln!(r, "{:>5} {:>8} {:>8} {:>8}", "row", "minted", "max", "F(k+2)").unwrap();
    for (k, row) in rows.iter().enumerate() {
        writeln!(r, "{:>5} {:>8} {:>8} {:>8}", k, row.len(), row_max(row), fib(k + 2)).unwrap();
    }

    // 4: the bridge on the zigzag
    writeln!(r, "\n== ?(F(n)/F(n+1)) as num/2^k (toward 2/3 = 0.1010...) ==").unwrap();
    for n in 1..=12 {
        let (num, k) = qmark(&BigInt::from(fib(n)), &BigInt::from(fib(n + 1)));
        writeln!(r, "n={:>2}  {}/{}  ->  {}/2^{}  path {}", n, fib(n), fib(n + 1), num, k, tree_path(&BigInt::from(fib(n)), &BigInt::from(fib(n + 1)))).unwrap();
    }

    // every finite bf16 cell -> its rational partner
    let mut tsv = String::from("pattern\tsheet\tm\tn\tp\tq\tdepth\n");
    let mut h = Sha256::new();
    let (mut count, mut max_depth, mut max_q_bits) = (0usize, BigInt::from(0), 0u64);
    let mut distinct = std::collections::BTreeSet::new();
    for p in 0u16..=0xFFFF {
        let e = (p >> 7) & 0xFF;
        if e == 0xFF || (p & 0x7FFF) == 0 {
            continue; // inf/nan and the two zeros
        }
        let (sheet, m, n) = decompose(p);
        let (pp, qq) = cell_partner(m as u16, n as i32);
        let (num, k) = cell_dyadic(m as u16, n as i32);
        assert_eq!(qmark(&pp, &qq), (num, k), "round trip {:#06x}", p);
        let depth = tree_depth(&pp, &qq);
        if depth > max_depth {
            max_depth = depth.clone();
        }
        max_q_bits = max_q_bits.max(qq.bits());
        if sheet == 0 {
            distinct.insert((pp.clone(), qq.clone()));
        }
        let line = format!("{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\n", p, sheet, m, n, pp, qq, depth);
        h.update(line.as_bytes());
        tsv.push_str(&line);
        count += 1;
    }
    let sha = format!("{:x}", h.finalize());
    writeln!(r, "\n== the bridge over the dictionary ==").unwrap();
    writeln!(r, "finite patterns: {}  magnitudes: {}  distinct partners (sheet 0): {}", count, count / 2, distinct.len()).unwrap();
    writeln!(r, "round trip ?(partner) = value: all {}  max tree depth: {} (= 255*2^120 - 1: the largest integer cell, on the right wall toward the cusp)  max partner denominator bits: {}", count, max_depth, max_q_bits).unwrap();
    writeln!(r, "sample partners:").unwrap();
    for &(m, n) in &[(1u16, -1i32), (3, -2), (5, -3), (3, -1), (255, -8), (129, -8), (1, -126), (255, 0), (3, 4)] {
        let (pp, qq) = cell_partner(m, n);
        let depth = tree_depth(&pp, &qq);
        let path = if depth <= BigInt::from(64) { tree_path(&pp, &qq) } else { format!("(depth {})", depth) };
        writeln!(r, "  <{}|{}> = {}/2^{}  ->  {}/{}  path {}", m, n, m, -n, pp, qq, path).unwrap();
    }
    writeln!(r, "\n== the frame: parents (left a/c, right b/d), node = mediant, b*c - a*d = 1; O(terms), never a walk ==").unwrap();
    for &(m, n) in &[(1u16, -1i32), (3, -2), (255, -8), (1, -133), (255, -133), (129, -133), (255, 0), (255, 120)] {
        let (pp, qq) = cell_partner(m, n);
        let [a, b, c, d] = tree_frame(&pp, &qq);
        writeln!(r, "  <{}|{}> -> {}/{}  depth {}  left {}/{}  right {}/{}  det {}", m, n, pp, qq, tree_depth(&pp, &qq), a, c, b, d, &b * &c - &a * &d).unwrap();
    }
    writeln!(r, "partner table sha256 {}", sha).unwrap();

    // the fibers: the 128 atoms as positions ±5^k on the 8-bit torus, and on the 16-bit torus where products are exact
    let f8 = Fiber::new(8);
    let f16 = Fiber::new(16);
    writeln!(r, "\n== the fibers: odd m = ±5^k; the 8-bit torus (k mod 64) and the 16-bit torus (k mod 16384, products exact) ==").unwrap();
    writeln!(r, "{:>4} {:>6} {:>6} | {:>6} {:>6}", "m", "sign8", "k8", "sign16", "k16").unwrap();
    let mut hf = Sha256::new();
    for m in (1u32..256).step_by(2) {
        let (s8, k8) = f8.position(m);
        let (s16, k16) = f16.position(m);
        let line = format!("{:>4} {:>6} {:>6} | {:>6} {:>6}\n", m, if s8 { "-" } else { "+" }, k8, if s16 { "-" } else { "+" }, k16);
        hf.update(line.as_bytes());
        if m <= 31 || m >= 241 {
            r.push_str(&line);
        } else if m == 33 {
            writeln!(r, "   … (all 128 rows hashed)").unwrap();
        }
    }
    let (s, k) = f16.mul(f16.position(255), f16.position(255));
    writeln!(r, "255 * 255 = position sum -> {} (exact 65025)", f16.value(s, k)).unwrap();
    let (s, k) = f16.inv(f16.position(3));
    writeln!(r, "1/3 = position -k -> {} (3 * {} = 1 mod 2^16)", f16.value(s, k), f16.value(s, k)).unwrap();
    writeln!(r, "fiber table sha256 {:x}", hf.finalize()).unwrap();

    print!("{}", r);
    std::fs::write(root.join("receipts").join("maelstrom.txt"), &r).unwrap();
    std::fs::write(root.join("receipts").join("maelstrom-partners.tsv"), &tsv).unwrap();
    println!("receipt: receipts/maelstrom.txt, receipts/maelstrom-partners.tsv");
}
