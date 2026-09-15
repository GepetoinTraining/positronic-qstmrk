//! The Pedro Fibonacci set over a TPB run (pgarcia, 2026-09-14).
//!
//! Each cycle closes a square: the reference r² is whole as r²/r². The relation just before it closes
//! is its limit in two dimensions — the Lim(inf) Q* produces — and it is where one dimension crosses
//! into the next (cycle 1: 3/4 = 3¹/2²; cycle 2: 8/9 = 2³/3²). Constructively it is always collapsed
//! to factors:
//!   denominator  the reference's own factors, held twice (the square).
//!   numerator    the two neighbours of the whole r — the integers one hop below and one hop above
//!                it — held together. Each neighbour is read from the ledger in its own factors.
//! When the numerator is an entry that arrived as an unnamed atom, the limit collapses it: the entry is
//! deduped into its factors, and every later square built on it collapses with it.
//!
//! Machine side, flagged: names are decimal readings; the neighbours of r are read as r − 1 and r + 1;
//! a rod's depth is kept as a bead count; `shadow_composite` is an outside check, reported, never used.

use std::collections::BTreeMap;
use std::path::Path;

/// Rods: atom name → beads.
pub type Rods = BTreeMap<u64, u32>;

pub struct Run {
    pub cycles: usize,
    pub seam: Vec<u64>,
    pub squares: BTreeMap<u64, u64>, // square name → its atom
    pub rectangle: Option<(u64, u64)>,
}

fn words(p: &Path) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn read(run: &Path) -> Run {
    let receipt = std::fs::read_to_string(run.join("receipt.txt")).unwrap();
    let cycles = receipt.split("cycles=").nth(1).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse().ok()).unwrap();
    let aw = words(&run.join("atoms.tpb"));
    let cw = words(&run.join("charts.tpb"));
    let sw = words(&run.join("seam.tpb"));
    let name: BTreeMap<u32, u64> = aw.chunks_exact(2).enumerate().map(|(i, r)| ((i * 8) as u32, r[1] as u64 + 1)).collect();
    let mut squares = BTreeMap::new();
    let mut rectangle = None;
    let mut i = 0usize;
    while i < cw.len() {
        let n = cw[i] as usize;
        let f: Vec<(u64, u32)> = (0..n).map(|k| (name[&cw[i + 1 + 2 * k]], cw[i + 2 + 2 * k])).collect();
        if n == 1 && f[0].1 == 2 {
            squares.insert(f[0].0 * f[0].0, f[0].0);
        }
        if n == 2 {
            rectangle = Some((f[0].0, f[1].0));
        }
        i += 1 + 2 * n;
    }
    Run { cycles, seam: sw.iter().map(|o| name[o]).collect(), squares, rectangle }
}

pub fn hold(a: &Rods, b: &Rods) -> Rods {
    let mut out = a.clone();
    for (&k, &v) in b {
        *out.entry(k).or_insert(0) += v;
    }
    out
}

pub struct Ledger<'r> {
    pub run: &'r Run,
    pub collapsed: BTreeMap<u64, Rods>,
}

impl<'r> Ledger<'r> {
    /// A number in its factors, as far as the ledger has collapsed it.
    pub fn factors(&self, n: u64) -> Option<Rods> {
        if n == 1 {
            return Some(Rods::new());
        }
        if let Some(r) = self.collapsed.get(&n) {
            return Some(r.clone());
        }
        if let Some((a, b)) = self.run.rectangle {
            if a * b == n {
                return Some(hold(&self.factors(a)?, &self.factors(b)?));
            }
        }
        if let Some(&a) = self.run.squares.get(&n) {
            let f = self.factors(a)?;
            return Some(hold(&f, &f));
        }
        if self.run.seam.contains(&n) {
            return Some(Rods::from([(n, 1)]));
        }
        None
    }
}

pub struct Lim {
    pub cycle: usize,
    pub whole: u64,
    pub below: u64,
    pub above: u64,
    pub num: Rods,
    pub den: Rods,
    /// the entry this limit collapsed, if it had arrived as an unnamed atom
    pub deduped: Option<u64>,
}

pub fn pedro_fibonacci(run: &Run) -> (Vec<Lim>, Ledger<'_>) {
    let mut ledger = Ledger { run, collapsed: BTreeMap::new() };
    let mut out = Vec::new();
    for k in 0..run.cycles.min(run.seam.len()) {
        let r = run.seam[k];
        let fr = ledger.factors(r).expect("reference not in the ledger");
        let den = hold(&fr, &fr);
        let (below, above) = (r - 1, r + 1);
        let (Some(fb), Some(fa)) = (ledger.factors(below), ledger.factors(above)) else {
            break; // a neighbour the run has not reached: the set stops here
        };
        let num = hold(&fb, &fa);
        let n = below * above;
        let mut deduped = None;
        if let Some(current) = ledger.factors(n) {
            if current == Rods::from([(n, 1)]) && num != current {
                ledger.collapsed.insert(n, num.clone());
                deduped = Some(n);
            }
        }
        out.push(Lim { cycle: k + 1, whole: r, below, above, num, den, deduped });
    }
    (out, ledger)
}

pub fn sup(mut v: u32) -> String {
    const D: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let mut s = Vec::new();
    loop {
        s.push(D[(v % 10) as usize]);
        v /= 10;
        if v == 0 {
            break;
        }
    }
    s.iter().rev().collect()
}

pub fn rods_text(r: &Rods) -> String {
    if r.is_empty() {
        return "1".into();
    }
    r.iter().map(|(a, b)| format!("{}{}", a, sup(*b))).collect::<Vec<_>>().join(" × ")
}

pub fn relation_text(num: &Rods, den: &Rods) -> String {
    let wrap = |r: &Rods| if r.len() > 1 { format!("({})", rods_text(r)) } else { rods_text(r) };
    format!("{}/{}", wrap(num), wrap(den))
}

/// Outside check only: whether the line would split n. Reported, never used by the ledger.
pub fn shadow_composite(n: u64) -> bool {
    n > 3 && (2..n).take_while(|d| d * d <= n).any(|d| n % d == 0)
}
