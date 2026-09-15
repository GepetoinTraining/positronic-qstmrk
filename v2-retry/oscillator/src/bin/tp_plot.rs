//! `tp_plot` — step 3 seen as an image: the tuples of what cycle 2 found, deduped into fibers, placed by
//! neighbours (pgarcia, 2026-09-14).
//!
//!   tp_plot [run-dir]        default tpb/run-c2  →  <run-dir>/cycle2.svg and <run-dir>/tp.txt
//!
//! Reads the TPB only:
//!   found     the floor, every chart glued into cycle 2's reference (O1²), the reference itself, and
//!             every entry whose origin is that reference.
//!   seam      every entry (the bit and the Opus arrivals), as identities 1/entry toward ∅.
//! Step 3:
//!   tuples    every (a, b) over the found; tuples that name one site form a fiber, and the fiber is
//!             one node. The node is drawn once; its tuples are the beads under it.
//!   neighbours two nodes p/q and r/s are neighbours when |ps − qr| is the unit (§1.1). Distance is
//!             hops between neighbours.
//!   place     row = hops from the floor 1/1; column = hops from the seam; the integer side (turn
//!             reading) to the right of the seam, the relational side to the left.
//!   dedupe    the Pedro Fibonacci set (pf.rs): the relation before each square closes, collapsed to
//!             factors; an entry it names is a crossing, not an atom.
//! Machine side, flagged: the decimal names read from ring ends, the fiber test (cross products), the
//! neighbour test (|ps − qr|), and every pixel coordinate. These are lookups the construction has not
//! yet replaced (§8.8 gap 2: the form of the admissibility lookup is not fixed).

#[path = "../pf.rs"]
mod pf;

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn words(p: &Path) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

struct Node {
    p: u64,
    q: u64,
    tuples: Vec<(u64, u64)>,
}

fn label(p: u64, q: u64) -> String {
    if q == 1 { p.to_string() } else { format!("{}/{}", p, q) }
}

fn bfs(adj: &[Vec<usize>], sources: &[usize]) -> Vec<Option<u32>> {
    let mut d: Vec<Option<u32>> = adj.iter().map(|_| None).collect();
    let mut q = VecDeque::new();
    for &s in sources {
        d[s] = Some(0);
        q.push_back(s);
    }
    while let Some(u) = q.pop_front() {
        let du = d[u].unwrap();
        for &v in &adj[u] {
            if d[v].is_none() {
                d[v] = Some(du + 1);
                q.push_back(v);
            }
        }
    }
    d
}

fn main() {
    let run: PathBuf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tpb").join("run-c2"));
    let aw = words(&run.join("atoms.tpb"));
    let cw = words(&run.join("charts.tpb"));
    let gw = words(&run.join("glue.tpb"));
    let sw = words(&run.join("seam.tpb"));

    // atoms: offset -> (origin chart, name)
    let mut atom_name: BTreeMap<u32, u64> = BTreeMap::new();
    let mut atom_origin: BTreeMap<u32, u32> = BTreeMap::new();
    for (i, r) in aw.chunks_exact(2).enumerate() {
        let off = (i * 8) as u32;
        atom_origin.insert(off, r[0]);
        atom_name.insert(off, r[1] as u64 + 1); // shadow: the ring's seats read as a name
    }
    // charts: offset -> factors
    let mut charts: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
    let mut i = 0usize;
    while i < cw.len() {
        let n = cw[i] as usize;
        let f: Vec<(u32, u32)> = (0..n).map(|k| (cw[i + 1 + 2 * k], cw[i + 2 + 2 * k])).collect();
        charts.insert((i * 4) as u32, f);
        i += 1 + 2 * n;
    }
    let chart_name = |f: &[(u32, u32)]| -> u64 { f.iter().map(|&(a, l)| atom_name[&a].pow(l)).product() };

    // cycle 2's reference: the square chart of the seam's second entry
    let o1 = sw[1];
    let reference = *charts.iter().find(|(_, f)| f.len() == 1 && f[0] == (o1, 2)).map(|(o, _)| o).unwrap();
    let mut found: Vec<u64> = vec![1, chart_name(&charts[&reference])];
    for g in gw.chunks_exact(3).filter(|g| g[1] == reference) {
        found.push(chart_name(&charts[&g[0]]));
    }
    for (&off, &origin) in &atom_origin {
        if origin == reference {
            found.push(atom_name[&off]);
        }
    }
    found.sort();
    found.dedup();
    let seam_names: Vec<u64> = sw.iter().map(|o| atom_name[o]).collect();

    // tuples -> fibers -> nodes
    let mut by_site: BTreeMap<(u64, u64), Vec<(u64, u64)>> = BTreeMap::new();
    for &a in &found {
        for &b in &found {
            // a/a is a whole, not the floor (update 1: 3/3 is 3); every other tuple joins its site
            let key = if a == b { (a, 1) } else { let g = gcd(a, b); (a / g, b / g) };
            by_site.entry(key).or_default().push((a, b));
        }
    }
    let nodes: Vec<Node> = by_site.into_iter().map(|((p, q), tuples)| Node { p, q, tuples }).collect();
    let index: BTreeMap<(u64, u64), usize> = nodes.iter().enumerate().map(|(i, n)| ((n.p, n.q), i)).collect();

    // neighbours
    let mut adj: Vec<Vec<usize>> = nodes.iter().map(|_| Vec::new()).collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            let (a, b) = (&nodes[i], &nodes[j]);
            if (a.p * b.q).abs_diff(a.q * b.p) == 1 {
                adj[i].push(j);
                adj[j].push(i);
                edges.push((i, j));
            }
        }
    }
    let floor = index[&(1, 1)];
    let seam: Vec<usize> = seam_names.iter().filter_map(|&n| index.get(&(1, n)).copied()).collect();
    let d0 = bfs(&adj, &[floor]);
    let ds = bfs(&adj, &seam);

    // dedupe: the Pedro Fibonacci set — the relation before each square closes, collapsed to factors
    let pf_run = pf::read(&run);
    let (lims, _ledger) = pf::pedro_fibonacci(&pf_run);
    let pf_line = lims.iter().map(|l| pf::relation_text(&l.num, &l.den)).collect::<Vec<_>>().join(", ");
    let mut dedupe: Vec<String> = Vec::new();
    for l in &lims {
        if let Some(n) = l.deduped {
            if seam_names.contains(&n) {
                dedupe.push(format!("{} ≡ {}  (cycle {}: {})", n, pf::rods_text(&l.num), l.cycle, pf::relation_text(&l.num, &l.den)));
            }
        }
    }

    // integer ↔ id distances for every seam entry
    let mut id_hops: Vec<String> = Vec::new();
    for &o in &seam_names {
        if let (Some(&i), Some(&j)) = (index.get(&(o, 1)), index.get(&(1, o))) {
            let d = bfs(&adj, &[i]);
            id_hops.push(format!("{} ↔ 1/{}: {} hops", o, o, d[j].map(|x| x.to_string()).unwrap_or("unreached".into())));
        }
    }

    // ── place ───────────────────────────────────────────────────────────────────────────────────
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Side {
        Relation,
        Seam,
        Integer,
    }
    let side = |i: usize| -> Side {
        let n = &nodes[i];
        if seam.contains(&i) || i == floor {
            Side::Seam
        } else if n.p > n.q {
            Side::Integer
        } else {
            Side::Relation
        }
    };
    let mut groups: BTreeMap<(Side, u32, u32), Vec<usize>> = BTreeMap::new();
    let (mut max_d0, mut max_ds) = (0u32, 0u32);
    for i in 0..nodes.len() {
        let r = d0[i].unwrap_or(99);
        let c = if side(i) == Side::Seam { 0 } else { ds[i].unwrap_or(9) };
        max_d0 = max_d0.max(r);
        max_ds = max_ds.max(c);
        groups.entry((side(i), r, c)).or_default().push(i);
    }
    let _ = max_ds;
    let (row_h, step): (i64, i64) = (92, 44);
    let (top, left) = (150i64, 40i64);
    // each column is as wide as its widest row; columns are laid outward from the seam
    let mut widths: BTreeMap<(Side, u32), i64> = BTreeMap::new();
    for ((sd, _, c), g) in &groups {
        let w = g.len() as i64 * step + 28;
        let e = widths.entry((*sd, *c)).or_insert(0);
        *e = (*e).max(w);
    }
    let seam_w = widths.get(&(Side::Seam, 0)).copied().unwrap_or(80).max(80);
    let rel_total: i64 = widths.iter().filter(|((s, _), _)| *s == Side::Relation).map(|(_, w)| *w).sum();
    let int_total: i64 = widths.iter().filter(|((s, _), _)| *s == Side::Integer).map(|(_, w)| *w).sum();
    let centre = left + rel_total + seam_w / 2;
    let width = centre + seam_w / 2 + int_total + left;
    let height = top + (max_d0 as i64 + 2) * row_h;
    let base = |sd: Side, c: u32| -> i64 {
        let inner: i64 = widths.iter().filter(|((s, k), _)| *s == sd && *k < c).map(|(_, w)| *w).sum();
        let own = widths[&(sd, c)];
        match sd {
            Side::Seam => centre,
            Side::Integer => centre + seam_w / 2 + inner + own / 2,
            Side::Relation => centre - seam_w / 2 - inner - own / 2,
        }
    };
    let mut pos: Vec<(i64, i64)> = nodes.iter().map(|_| (0, 0)).collect();
    for ((sd, r, c), g) in &groups {
        let len = g.len() as i64;
        let base_x = base(*sd, *c);
        for (k, &i) in g.iter().enumerate() {
            let x = base_x + (k as i64) * step - (len - 1) * step / 2;
            let y = top + (*r as i64) * row_h + if len > 1 { ((k as i64) % 2) * 22 - 11 } else { 0 };
            pos[i] = (x, y);
        }
    }

    // ── draw ────────────────────────────────────────────────────────────────────────────────────
    let mut svg = String::new();
    writeln!(svg, r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Consolas, 'Courier New', monospace">"##, w = width, h = height).unwrap();
    writeln!(svg, r##"<rect width="{}" height="{}" fill="#f4f1ec"/>"##, width, height).unwrap();
    writeln!(svg, r##"<text x="{}" y="40" font-size="20" fill="#1b1a18">Cycle 2 — tuples deduped into fibers, placed by neighbours</text>"##, left).unwrap();
    writeln!(svg, r##"<text x="{}" y="64" font-size="12" fill="#5a554d">found: {} · seam: {} · {} tuples → {} nodes · {} neighbour pairs</text>"##, left,
        found.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" "), seam_names.iter().map(|n| format!("1/{}", n)).collect::<Vec<_>>().join(" "),
        nodes.iter().map(|n| n.tuples.len()).sum::<usize>(), nodes.len(), edges.len()).unwrap();
    writeln!(svg, r##"<text x="{}" y="84" font-size="12" fill="#5a554d">row = hops from 1 · column = hops from the seam · beads under a node = its tuples · hover a node for the fiber</text>"##, left).unwrap();
    writeln!(svg, r##"<text x="{}" y="104" font-size="13" fill="#a23b2a">Pedro Fibonacci set: {}</text>"##, left, pf_line).unwrap();
    writeln!(svg, r##"<text x="{}" y="128" font-size="13" fill="#3d6b8c">◀ relational side</text>"##, left).unwrap();
    writeln!(svg, r##"<text x="{}" y="128" font-size="13" fill="#b5651d" text-anchor="end">integer side ▶</text>"##, width - left).unwrap();

    for &(i, j) in &edges {
        let (a, b) = (pos[i], pos[j]);
        writeln!(svg, r##"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#8c867c" stroke-opacity="0.28" stroke-width="1"/>"##, a.0, a.1, b.0, b.1).unwrap();
    }
    // the seam, from 1/2 toward ∅
    let seam_y: Vec<i64> = seam.iter().map(|&i| pos[i].1).collect();
    let (sy0, sy1) = (seam_y.iter().min().copied().unwrap_or(top), seam_y.iter().max().copied().unwrap_or(top));
    writeln!(svg, r##"<line x1="{c}" y1="{}" x2="{c}" y2="{}" stroke="#1b1a18" stroke-width="2" stroke-dasharray="6 5"/>"##, sy0, sy1 + row_h, c = centre).unwrap();
    writeln!(svg, r##"<text x="{}" y="{}" font-size="18" fill="#1b1a18" text-anchor="middle">∅</text>"##, centre, sy1 + row_h + 22).unwrap();

    let deduped: Vec<u64> = seam_names.iter().copied().filter(|o| dedupe.iter().any(|d| d.starts_with(&format!("{} ≡", o)))).collect();
    for (i, n) in nodes.iter().enumerate() {
        let (x, y) = pos[i];
        let (fill, ink) = if i == floor {
            ("#1b1a18", "#f4f1ec")
        } else if seam.contains(&i) {
            ("#2b2a27", "#f4f1ec")
        } else if n.q == 1 && deduped.contains(&n.p) {
            ("#a23b2a", "#f4f1ec")
        } else if n.q == 1 && seam_names.contains(&n.p) {
            ("#b5651d", "#f4f1ec")
        } else if n.p > n.q {
            ("#e9c9a4", "#1b1a18")
        } else {
            ("#bcd3e2", "#1b1a18")
        };
        let fiber = n.tuples.iter().map(|&(a, b)| format!("{}/{}", a, b)).collect::<Vec<_>>().join(" ≡ ");
        writeln!(svg, r##"<g><title>{} · fiber: {} · hops from 1: {} · from seam: {}</title>"##, label(n.p, n.q), fiber,
            d0[i].map(|v| v.to_string()).unwrap_or("-".into()), ds[i].map(|v| v.to_string()).unwrap_or("-".into())).unwrap();
        writeln!(svg, r##"<circle cx="{}" cy="{}" r="17" fill="{}" stroke="#1b1a18" stroke-width="0.8"/>"##, x, y, fill).unwrap();
        writeln!(svg, r##"<text x="{}" y="{}" font-size="11" fill="{}" text-anchor="middle">{}</text>"##, x, y + 4, ink, label(n.p, n.q)).unwrap();
        let k = n.tuples.len() as i64;
        for b in 0..k {
            writeln!(svg, r##"<circle cx="{}" cy="{}" r="2" fill="#1b1a18"/>"##, x + b * 5 - (k - 1) * 5 / 2, y + 24).unwrap();
        }
        writeln!(svg, "</g>").unwrap();
    }
    // legend
    let ly = height - 40;
    let legend = [("#1b1a18", "floor 1"), ("#2b2a27", "seam: 1/entry"), ("#b5651d", "entry, whole"), ("#a23b2a", "entry that dedupes (a crossing)"), ("#e9c9a4", "integer side"), ("#bcd3e2", "relational side")];
    let mut lx = left;
    for (c, t) in legend {
        writeln!(svg, r##"<circle cx="{}" cy="{}" r="7" fill="{}" stroke="#1b1a18" stroke-width="0.6"/><text x="{}" y="{}" font-size="12" fill="#1b1a18">{}</text>"##, lx, ly, c, lx + 12, ly + 4, t).unwrap();
        lx += 30 + 8 * t.chars().count() as i64;
    }
    writeln!(svg, "</svg>").unwrap();
    std::fs::write(run.join("cycle2.svg"), &svg).unwrap();

    let mut rep = String::new();
    writeln!(rep, "format=v2.tp_plot.v1 run={}", run.display()).unwrap();
    writeln!(rep, "found: {:?}", found).unwrap();
    writeln!(rep, "seam: {}", seam_names.iter().map(|n| format!("1/{}", n)).collect::<Vec<_>>().join(", ")).unwrap();
    writeln!(rep, "tuples {} -> nodes {} ; neighbour pairs {}", nodes.iter().map(|n| n.tuples.len()).sum::<usize>(), nodes.len(), edges.len()).unwrap();
    writeln!(rep, "rows (hops from 1): {} ; seam columns: {}", max_d0, max_ds).unwrap();
    writeln!(rep, "Pedro Fibonacci set: {}", pf_line).unwrap();
    for h in &id_hops {
        writeln!(rep, "{}", h).unwrap();
    }
    for d in &dedupe {
        writeln!(rep, "dedupe: {}", d).unwrap();
    }
    std::fs::write(run.join("tp.txt"), &rep).unwrap();
    print!("{}", rep);
    println!("svg: {}", run.join("cycle2.svg").display());
}
