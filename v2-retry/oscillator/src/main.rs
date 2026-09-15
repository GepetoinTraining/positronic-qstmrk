//! The oscillator, third attempt (pgarcia, 2026-09-14): cycle 1, with the ? cell entered in the
//! OpusNumber instead of named off the line.
//!
//! Target, pgarcia's line with the shadow name 3 replaced by the ledger's first entry:
//!     2{Q*(4/O1, O1/4, O1/2, 2/O1, 1/O1), Pq(4/2, [2/4, 1/2], 1/4), Pn(2)}
//!
//! OpusNumber (coined by pgarcia): a ledger that includes, in order of arrival, things not yet named.
//! An entry needs no name: its identity is its own slot 1/entry, which makes it different from
//! everything else. Entries are kept with the region they were found as, so where they stand is a
//! receipt, not a comparison.
//!
//! Cycle 1:
//!   shape    rod 2 takes one bead per seat of its own ring: the square. Its seats are the paths of
//!            nested ring choices, walked inside the square only (the atom's own geometry, not a line).
//!   held     the floor, rod 2 •, rod 2 ••; each covers the first block of the square's seats that
//!            pairs with its own seats.
//!   ? cell   the block of the square no held cell covers exactly. It is included in the OpusNumber.
//!   turns    within a pair, the cell whose region holds the other's reads as the turn. All regions
//!            here are blocks of one walk, so one always holds the other.
//!   Pq, Q*   as before: pairs deepest first, turn reading then relation reading, nothing over the
//!            floor, one site in one bracket (a/b and c/d are one site when a held with d is, rod for
//!            rod, b held with c).
//!
//! Shadow (machine side): the decimal names of the bit's own cells (1, 2, 4), the ledger's arrival
//! mark (O1), and the rail's rod count in Pn. The structure never reads any of them.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

const TARGET: &str = "2{Q*(4/O1, O1/4, O1/2, 2/O1, 1/O1), Pq(4/2, [2/4, 1/2], 1/4), Pn(2)}";

/// Update 1, as written by pgarcia: the atoms with their levels, then the crossings at each level.
const UPDATE_TARGET: [&str; 3] = ["2(2¹, 2²), 3(3¹, 3²)", "6(2¹ × 3¹)", "36(2² × 3²)"];

#[derive(Clone, Copy)]
struct Bead;

#[derive(Clone)]
struct Rod {
    atom: usize, // rail address (wiring)
    stack: Vec<Bead>,
}

#[derive(Clone, Default)]
struct Cell {
    rods: Vec<Rod>,
}

/// A seat of a shape: one ring choice per bead (ring seat addresses are wiring).
type Path = Vec<usize>;
type Region = Vec<Path>;

/// Something on the relational side: its rods, the region it covers in the shape, and how it is written.
#[derive(Clone)]
struct Item {
    cell: Cell,
    region: Region,
    label: String,
}

/// An OpusNumber entry: a thing found and not named. Its identity is its slot.
struct Entry {
    region: Region,
    rod: usize, // its rod on the rail
}

struct OpusNumber {
    entries: Vec<Entry>,
}

impl OpusNumber {
    /// Include a new unnamed thing; the returned slot is its identity (1/slot).
    fn include(&mut self, region: Region, rod: usize) -> usize {
        self.entries.push(Entry { region, rod });
        self.entries.len() - 1
    }
    fn mark(&self, slot: usize) -> String {
        format!("O{}", slot + 1) // shadow: arrival mark for the reader
    }
}

fn same_stack(a: &[Bead], b: &[Bead]) -> bool {
    let (mut x, mut y) = (a.iter(), b.iter());
    loop {
        match (x.next(), y.next()) {
            (None, None) => return true,
            (Some(_), Some(_)) => continue,
            _ => return false,
        }
    }
}

fn stack_on<'c>(c: &'c Cell, atom: usize) -> &'c [Bead] {
    c.rods.iter().find(|r| r.atom == atom).map(|r| r.stack.as_slice()).unwrap_or(&[])
}

fn hold(a: &Cell, b: &Cell) -> Cell {
    let mut out = a.clone();
    for rod in &b.rods {
        match out.rods.iter_mut().find(|r| r.atom == rod.atom) {
            Some(r) => r.stack.extend(rod.stack.iter().copied()),
            None => out.rods.push(rod.clone()),
        }
    }
    out
}

fn same_cell(a: &Cell, b: &Cell) -> bool {
    a.rods.iter().all(|r| same_stack(&r.stack, stack_on(b, r.atom)))
        && b.rods.iter().all(|r| same_stack(&r.stack, stack_on(a, r.atom)))
}

fn same_site(p: &(Item, Item), q: &(Item, Item)) -> bool {
    same_cell(&hold(&p.0.cell, &q.1.cell), &hold(&p.1.cell, &q.0.cell))
}

/// The seats of a shape: nested ring choices, one level per bead.
fn seats(stack: &[Bead], ring: &[Bead]) -> Region {
    let mut out: Region = vec![Path::new()];
    for _bead in stack {
        let mut next = Region::new();
        for p in &out {
            for (seat, _) in ring.iter().enumerate() {
                let mut q = p.clone();
                q.push(seat);
                next.push(q);
            }
        }
        out = next;
    }
    out
}

/// The block of the shape's seats that pairs, seat for seat, with a cell's own seats.
fn block(shape: &Region, own: &Region) -> Region {
    shape.iter().zip(own.iter()).map(|(s, _)| s.clone()).collect()
}

fn same_path(a: &Path, b: &Path) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Every seat of `b` stands in `a`.
fn holds(a: &Region, b: &Region) -> bool {
    b.iter().all(|p| a.iter().any(|q| same_path(p, q)))
}

fn same_region(a: &Region, b: &Region) -> bool {
    holds(a, b) && holds(b, a)
}

fn is_floor(i: &Item) -> bool {
    i.cell.rods.is_empty()
}

/// Turn reading first, then relation reading; nothing over the floor.
fn readings(a: &Item, b: &Item) -> Vec<(Item, Item)> {
    let (top, bottom) = if holds(&a.region, &b.region) { (a, b) } else { (b, a) };
    let mut out = Vec::new();
    for (x, y) in [(top, bottom), (bottom, top)] {
        if !is_floor(y) {
            out.push((x.clone(), y.clone()));
        }
    }
    out
}

fn group(rels: Vec<(Item, Item)>) -> Vec<Vec<(Item, Item)>> {
    let mut groups: Vec<Vec<(Item, Item)>> = Vec::new();
    for r in rels {
        match groups.iter_mut().find(|g| same_site(&g[0], &r)) {
            Some(g) => g.push(r),
            None => groups.push(vec![r]),
        }
    }
    groups
}

fn write_groups(groups: &[Vec<(Item, Item)>]) -> String {
    let rel = |r: &(Item, Item)| format!("{}/{}", r.0.label, r.1.label);
    groups
        .iter()
        .map(|g| if g.len() == 1 { rel(&g[0]) } else { format!("[{}]", g.iter().map(rel).collect::<Vec<_>>().join(", ")) })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Shadow: the decimal name of a stack on the bit's rod.
fn bit_name(stack: &[Bead]) -> String {
    let mut n = 1u64;
    for _ in stack {
        n *= 2;
    }
    n.to_string()
}

fn main() {
    let cycles: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut out = String::new();
    writeln!(out, "format=v2.oscillator.v3 cycles={}", cycles).unwrap();

    // the rail: rod 0 is the bit's atom, a ring of two seats
    let bit_ring = vec![Bead, Bead];
    let mut rail_rods: Vec<()> = vec![()];
    let mut opus = OpusNumber { entries: Vec::new() };

    // the shape: one bead per seat of the ring — the square
    let shape_stack: Vec<Bead> = bit_ring.clone();
    let shape = seats(&shape_stack, &bit_ring);

    // held: floor, then one bead more at a time up to the shape; deepest first
    let mut held: Vec<Item> = Vec::new();
    let mut stack: Vec<Bead> = Vec::new();
    let floor_region = block(&shape, &seats(&stack, &bit_ring));
    held.push(Item { cell: Cell::default(), region: floor_region, label: bit_name(&stack) });
    for _seat in &shape_stack {
        stack.push(Bead);
        let region = block(&shape, &seats(&stack, &bit_ring));
        held.push(Item { cell: Cell { rods: vec![Rod { atom: 0, stack: stack.clone() }] }, region, label: bit_name(&stack) });
    }
    held.reverse();

    // Pq
    let mut pq = Vec::new();
    for (i, a) in held.iter().enumerate() {
        for b in &held[i + 1..] {
            pq.extend(readings(a, b));
        }
    }
    let pq = group(pq);

    // the ? cell: every block of the square's walk that no held cell covers exactly
    let mut unreached: Vec<Region> = Vec::new();
    let mut walk = Region::new();
    for s in &shape {
        walk.push(s.clone());
        if !held.iter().any(|h| same_region(&h.region, &walk)) {
            unreached.push(walk.clone());
        }
    }
    assert!(unreached.len() == 1, "cycle 1 refused: the square leaves more than one uncovered block");
    let region = unreached.pop().unwrap();
    rail_rods.push(());
    let slot = opus.include(region.clone(), rail_rods.len() - 1);
    let q = Item { cell: Cell { rods: vec![Rod { atom: opus.entries[slot].rod, stack: vec![Bead] }] }, region, label: opus.mark(slot) };

    // Q*
    let mut qs = Vec::new();
    for h in &held {
        qs.extend(readings(h, &q));
    }
    let qs = group(qs);

    let line = format!("{}{{Q*({}), Pq({}), Pn({})}}", bit_name(&[Bead]), write_groups(&qs), write_groups(&pq), rail_rods.len());
    writeln!(out, "{}", line).unwrap();
    for (s, e) in opus.entries.iter().enumerate() {
        let inside: Vec<&str> = held.iter().filter(|h| holds(&h.region, &e.region)).map(|h| h.label.as_str()).collect();
        let holding: Vec<&str> = held.iter().filter(|h| holds(&e.region, &h.region) && !same_region(&h.region, &e.region)).map(|h| h.label.as_str()).collect();
        writeln!(out, "OpusNumber {} | identity 1/{} | inside {} | holds {}", opus.mark(s), opus.mark(s), inside.join(" "), holding.join(" ")).unwrap();
    }
    writeln!(out, "target {}", TARGET).unwrap();
    writeln!(out, "gate cycle 1: {}", if line == TARGET { "EQUAL" } else { "DIFFERENT" }).unwrap();

    // ── update 1: measure what cycle 1 left and collapse it where it is whole ─────────────────────
    //
    // O1's ring: walk the blocks inside O1's own region; each block must be covered by a held cell or
    // by O1 itself. When every seat is covered, O1/O1 is not the floor — it is O1 whole, determined.
    writeln!(out, "\nupdate 1").unwrap();
    let e_region = opus.entries[slot].region.clone();
    let mut walk = Region::new();
    let mut o1_ring: Vec<Bead> = Vec::new();
    let mut seats_named: Vec<String> = Vec::new();
    for s in &e_region {
        walk.push(s.clone());
        let by = held
            .iter()
            .find(|h| same_region(&h.region, &walk))
            .map(|h| h.label.clone())
            .or_else(|| if same_region(&e_region, &walk) { Some(opus.mark(slot)) } else { None });
        match by {
            Some(label) => {
                o1_ring.push(Bead);
                seats_named.push(format!("{}/{}", label, opus.mark(slot)));
            }
            None => panic!("update 1 refused: a seat of {} is not covered", opus.mark(slot)),
        }
    }
    let o1_name = o1_ring.len().to_string(); // shadow: the whole's name, read off its own ring
    writeln!(out, "collapse {}/{}: seats {} all held -> whole, OpusNumber {} is named {}", opus.mark(slot), opus.mark(slot), seats_named.join(", "), opus.mark(slot), o1_name).unwrap();

    // the rail's atoms, each with its own ring
    let atoms: Vec<(String, Vec<Bead>)> = vec![(bit_name(&[Bead]), bit_ring.clone()), (o1_name, o1_ring)];

    // levels of cycle 1: one per seat of the bit's ring — each level its own shape, not a stack on a line
    let mut levels: Vec<Vec<Bead>> = Vec::new();
    let mut level: Vec<Bead> = Vec::new();
    for _seat in &bit_ring {
        level.push(Bead);
        levels.push(level.clone());
    }
    let sup = |l: &[Bead]| -> &str {
        match l.len() {
            1 => "¹",
            2 => "²",
            3 => "³",
            _ => "ⁿ",
        }
    }; // shadow: the level's mark

    let atom_line = atoms
        .iter()
        .map(|(name, _)| format!("{}({})", name, levels.iter().map(|l| format!("{}{}", name, sup(l))).collect::<Vec<_>>().join(", ")))
        .collect::<Vec<_>>()
        .join(", ");

    // Pqs: at each level, every atom's shape crossed with every other's — seats of the crossing are
    // the seats of each shape taken together; its name is the machine reading of those seats
    let mut pq_lines: Vec<String> = Vec::new();
    for l in &levels {
        let mut crossed: Vec<Vec<Path>> = vec![Vec::new()];
        for (_, ring) in &atoms {
            let shape = seats(l, ring);
            let mut next = Vec::new();
            for c in &crossed {
                for s in &shape {
                    let mut t = c.clone();
                    t.push(s.clone());
                    next.push(t);
                }
            }
            crossed = next;
        }
        let factors = atoms.iter().map(|(name, _)| format!("{}{}", name, sup(l))).collect::<Vec<_>>().join(" × ");
        pq_lines.push(format!("{}({})", crossed.len(), factors));
    }

    writeln!(out, "{}", atom_line).unwrap();
    for p in &pq_lines {
        writeln!(out, "{}", p).unwrap();
    }
    let got: Vec<&str> = std::iter::once(atom_line.as_str()).chain(pq_lines.iter().map(|s| s.as_str())).collect();
    let equal = got.len() == UPDATE_TARGET.len() && got.iter().zip(UPDATE_TARGET.iter()).all(|(a, b)| a == b);
    writeln!(out, "gate update 1: {}", if equal { "EQUAL" } else { "DIFFERENT" }).unwrap();
    if cycles > 1 {
        writeln!(out, "cycle 2 refused: not declared").unwrap();
    }

    let mut h = Sha256::new();
    h.update(out.as_bytes());
    let seal = format!("{:x}", h.finalize());
    print!("{}", out);
    println!("seal sha256 {}", seal);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("receipts");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("oscillator-v4-update1.txt"), format!("{}seal sha256 {}\n", out, seal)).unwrap();
}
