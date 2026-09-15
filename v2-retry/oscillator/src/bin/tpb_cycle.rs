//! `tpb_cycle` — Opus numbers interacting in two dimensions, written into the TPB (pgarcia, 2026-09-14).
//!
//! Declared by pgarcia for this run: magnitude up to 2 (squaring only), ONE rectangle, no update.
//!
//! The cycle (my version):
//!   cycle k     the reference is the square chart of the k-th entry on the seam (cycle 1: the bit).
//!   covered     every chart already written is paired, seat for seat, against the reference's walk;
//!               where it ends, that block is covered, and the embedding is written as a glue record.
//!   Opus        every block of the walk no chart covers is a new unnamed entry. Its ring is that block.
//!               It is appended to the seam, and its line chart (¹) and square chart (²) are written.
//!   rectangle   once, after cycle 1: the bit's line crossed with O1's line.
//!
//! The TPB (topology backup): append-only files of written addresses (byte offsets, u32 LE).
//!   atoms.tpb   per entry [origin chart, end]: the entry IS the block of its origin chart ending there.
//!               The bit's record points at the floor chart: given, not found.
//!   charts.tpb  per chart [factors, (atom, level)…]: a local shape. Floor: no factors. Square: one
//!               factor at level 2. The rectangle: two factors at level 1.
//!   glue.tpb    per embedding [inner chart, outer chart, end]: the inner chart's seats are the block of
//!               the outer chart's walk ending at `end`. Charts glued along these form the store.
//!   seam.tpb    per arrival [atom]: the line of identities 1/2, 1/O1, 1/O2, … toward ∅, the seam
//!               between the relational side and the integer side.
//! Nothing is ever rewritten; a run refuses to write into a TPB that already exists.
//!
//! Machine side (wiring, flagged): the walk of a reference chart is a run of seat addresses; pairing a
//! chart against it advances an address per seat; `reached` is indexed by address; the decimal names
//! in seam-names.txt and in the printout are the machine reading of seats, for the reader only.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

struct AtomRec {
    offset: u32,
    ring_end: u32, // wiring: the last seat address of its ring
}

struct ChartRec {
    offset: u32,
    factors: Vec<(usize, u32)>, // (atom index, level beads)
}

struct Store {
    dir: PathBuf,
    atoms_b: Vec<u8>,
    charts_b: Vec<u8>,
    glue_b: Vec<u8>,
    seam_b: Vec<u8>,
    atoms: Vec<AtomRec>,
    charts: Vec<ChartRec>,
    seam: Vec<usize>,
    l1: Vec<usize>, // chart index of each atom's line
    l2: Vec<usize>, // chart index of each atom's square
}

fn put(buf: &mut Vec<u8>, words: &[u32]) -> u32 {
    let at = buf.len() as u32;
    for w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    at
}

impl Store {
    fn atom(&mut self, origin_chart: u32, end: u32) -> usize {
        let offset = put(&mut self.atoms_b, &[origin_chart, end]);
        self.atoms.push(AtomRec { offset, ring_end: end });
        let a = self.atoms.len() - 1;
        put(&mut self.seam_b, &[offset]);
        self.seam.push(a);
        a
    }
    fn chart(&mut self, factors: Vec<(usize, u32)>) -> usize {
        let mut words = vec![factors.len() as u32];
        for &(a, l) in &factors {
            words.push(self.atoms[a].offset);
            words.push(l);
        }
        let offset = put(&mut self.charts_b, &words);
        self.charts.push(ChartRec { offset, factors });
        self.charts.len() - 1
    }
    fn glue(&mut self, inner: usize, outer: usize, end: u32) {
        let (i, o) = (self.charts[inner].offset, self.charts[outer].offset);
        put(&mut self.glue_b, &[i, o, end]);
    }
    fn arrive(&mut self, origin: u32, end: u32) -> usize {
        let a = self.atom(origin, end);
        let c1 = self.chart(vec![(a, 1)]);
        let c2 = self.chart(vec![(a, 2)]);
        self.l1.push(c1);
        self.l2.push(c2);
        a
    }
    /// The chart's seat dimensions: one ring per level bead of every factor.
    fn dims(&self, c: usize) -> Vec<u32> {
        let mut d = Vec::new();
        for &(a, l) in &self.charts[c].factors {
            for _ in 0..l {
                d.push(self.atoms[a].ring_end);
            }
        }
        d
    }
}

/// Pair a chart's own seats, one at a time, against a walk of `walk` seat addresses.
/// Some(address of the last paired seat) when the chart's seats run out first or together.
fn end_in(dims: &[u32], walk: u32) -> Option<u32> {
    let mut digits = vec![0u32; dims.len()];
    let mut p: u32 = 0;
    loop {
        if p >= walk {
            return None;
        }
        let mut i = 0;
        loop {
            if i == dims.len() {
                return Some(p);
            }
            if digits[i] < dims[i] {
                digits[i] += 1;
                break;
            }
            digits[i] = 0;
            i += 1;
        }
        p += 1;
    }
}

/// Every seat of a chart, walked: the length of its run of addresses.
fn walk_of(dims: &[u32]) -> u32 {
    end_in(dims, u32::MAX).unwrap() + 1
}

fn mark(s: &Store, a: usize) -> String {
    if s.atoms[a].offset == 0 {
        "2".to_string() // the bit, given
    } else {
        format!("O{}", a) // shadow: arrival mark
    }
}

fn main() {
    let cycles: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tpb").join(format!("run-c{}", cycles));
    if root.exists() {
        eprintln!("refused: {} is already written; the TPB is append-only", root.display());
        std::process::exit(2);
    }
    let mut s = Store { dir: root.clone(), atoms_b: Vec::new(), charts_b: Vec::new(), glue_b: Vec::new(), seam_b: Vec::new(), atoms: Vec::new(), charts: Vec::new(), seam: Vec::new(), l1: Vec::new(), l2: Vec::new() };

    // the floor chart, then the bit: a ring of two seats, its own origin
    let floor = s.chart(Vec::new());
    let bit = s.atom(0, 1);
    let c1 = s.chart(vec![(bit, 1)]);
    let c2 = s.chart(vec![(bit, 2)]);
    s.l1.push(c1);
    s.l2.push(c2);
    let _ = floor;

    let mut out = String::new();
    writeln!(out, "format=v2.tpb_cycle.v1 cycles={} | declared: magnitude <= 2 (squares only), one rectangle (2¹ × O1¹), no update", cycles).unwrap();

    let mut rectangle_written = false;
    for k in 0..cycles {
        let Some(&atom) = s.seam.get(k) else {
            writeln!(out, "cycle {:>3} refused: the seam has no entry to square", k + 1).unwrap();
            break;
        };
        let reference = s.l2[atom];
        let walk = walk_of(&s.dims(reference));
        let mut reached: Vec<bool> = (0..walk).map(|_| false).collect();
        let n_charts = s.charts.len();
        let mut covered = 0usize;
        for c in 0..n_charts {
            if c == reference {
                continue;
            }
            if let Some(end) = end_in(&s.dims(c), walk) {
                if !reached[end as usize] {
                    reached[end as usize] = true;
                    covered += 1;
                }
                s.glue(c, reference, end); // every embedding is written, second addresses included
            }
        }
        reached[(walk - 1) as usize] = true; // the reference covers its own walk
        let ref_offset = s.charts[reference].offset;
        let mut new = 0usize;
        for end in 0..walk {
            if !reached[end as usize] {
                let a = s.arrive(ref_offset, end);
                let l1 = s.l1[a];
                s.glue(l1, reference, end);
                new += 1;
            }
        }
        writeln!(out, "cycle {:>3}  reference {}²  walk {:>5}  covered {:>4}  new Opus {:>4}  seam {:>5}", k + 1, mark(&s, atom), walk, covered + 1, new, s.seam.len()).unwrap();
        if !rectangle_written && s.seam.len() > 1 {
            s.chart(vec![(s.seam[0], 1), (s.seam[1], 1)]);
            rectangle_written = true;
            writeln!(out, "           the one rectangle written: 2¹ × O1¹").unwrap();
        }
    }

    // the seam, as the line of identities toward ∅
    let n = s.seam.len();
    let ids: Vec<String> = s.seam.iter().take(8).map(|&a| format!("1/{}", mark(&s, a))).collect();
    writeln!(out, "\nseam: {}, … , 1/{} → ∅   ({} entries)", ids.join(", "), mark(&s, s.seam[n - 1]), n).unwrap();
    let names: Vec<String> = s.seam.iter().map(|&a| (s.atoms[a].ring_end + 1).to_string()).collect();
    writeln!(out, "shadow names, first 48: {}", names.iter().take(48).cloned().collect::<Vec<_>>().join(" ")).unwrap();

    // write the TPB
    std::fs::create_dir_all(&s.dir).unwrap();
    let files = [("atoms.tpb", &s.atoms_b), ("charts.tpb", &s.charts_b), ("glue.tpb", &s.glue_b), ("seam.tpb", &s.seam_b)];
    let mut h = Sha256::new();
    for (name, bytes) in files.iter() {
        let mut f = std::fs::OpenOptions::new().create_new(true).write(true).open(s.dir.join(name)).unwrap();
        f.write_all(bytes).unwrap();
        h.update(bytes);
        writeln!(out, "tpb {:<11} {:>9} B  {:>7} words", name, bytes.len(), bytes.len() / 4).unwrap();
    }
    let names_txt = s.seam.iter().zip(names.iter()).map(|(&a, nm)| format!("1/{}\t{}", mark(&s, a), nm)).collect::<Vec<_>>().join("\n");
    std::fs::write(s.dir.join("seam-names.txt"), names_txt + "\n").unwrap();

    // readback: every entry after the bit is placed, by a glue record, as the block of its origin chart
    let words = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let aw = words(&std::fs::read(s.dir.join("atoms.tpb")).unwrap());
    let gw = words(&std::fs::read(s.dir.join("glue.tpb")).unwrap());
    let sw = words(&std::fs::read(s.dir.join("seam.tpb")).unwrap());
    let mut placed = 0usize;
    for (a, rec) in s.atoms.iter().enumerate().skip(1) {
        let (origin, end) = (aw[(rec.offset / 4) as usize], aw[(rec.offset / 4) as usize + 1]);
        let l1 = s.charts[s.l1[a]].offset;
        if gw.chunks_exact(3).any(|g| g[0] == l1 && g[1] == origin && g[2] == end) {
            placed += 1;
        }
    }
    let seam_ok = sw.len() == s.atoms.len() && sw.iter().zip(s.atoms.iter()).all(|(w, r)| *w == r.offset);
    writeln!(out, "readback: {} of {} found entries placed in their origin chart; seam order {}", placed, s.atoms.len() - 1, if seam_ok { "intact" } else { "BROKEN" }).unwrap();

    h.update(out.as_bytes());
    let seal = format!("{:x}", h.finalize());
    writeln!(out, "seal sha256 {}", seal).unwrap();
    std::fs::write(s.dir.join("receipt.txt"), &out).unwrap();
    print!("{}", out);
}
