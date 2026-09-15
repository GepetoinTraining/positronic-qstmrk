//! `atlas_fiber_bench` — the 256-lane kernel (pgarcia, 2026-09-13 ~04:00): one lane per
//! (odd atom, sheet). With the atom fixed per lane the weight's core leaves the inner loop:
//! a term is the activation's core shifted by the weight's rung, then added. The lane sums
//! are multiplied by their atom once each at the gather, sign folded as a subtraction.
//!
//! Judged first by bit-identity with the production bucket on every output cell of
//! layer-0 q_proj, then by ns per term against the scalar kernel.
//!
//!   atlas_fiber_bench <root> <model-id>

use std::path::PathBuf;
use std::time::Instant;

use atlas::cells;
use atlas::forward::{rmsnorm_rows, Engine};
use atlas::matmul::{Bucket, XRow, ACC_FRAC};
use atlas::surfaces::{Atoms, Grid, Model};

fn find_intake_dir(root: &std::path::Path, id: &str) -> PathBuf {
    let prefix = format!("intake-{}-", id);
    let mut best: Option<PathBuf> = None;
    for e in std::fs::read_dir(root.join("receipts")).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with(&prefix) && best.as_ref().map_or(true, |b| name > b.file_name().unwrap().to_string_lossy().to_string()) {
            best = Some(p);
        }
    }
    best.unwrap()
}

/// One output row folded by (sheet, atom): 256 lanes of (column, rung).
struct RowFibers {
    /// lane = sheet*128 + (m-1)/2 ; entries (col u16, rung i8)
    lanes: Vec<Vec<(u16, i8)>>,
}

fn fold_row(g: &Grid, atoms: &Atoms, o: usize) -> RowFibers {
    let mut lanes: Vec<Vec<(u16, i8)>> = (0..256).map(|_| Vec::new()).collect();
    for tc in 0..g.tile_cols() {
        for (j, &id) in g.line(o, tc).iter().enumerate() {
            let id = id as usize;
            let m = atoms.m[id];
            if m == 0 {
                continue;
            }
            let lane = (atoms.sheet[id] as usize) * 128 + ((m as usize) - 1) / 2;
            let col = (tc * g.tw + j) as u16;
            let rung = atoms.n[id];
            assert!(rung >= -128 && rung <= 127);
            lanes[lane].push((col, rung as i8));
        }
    }
    RowFibers { lanes }
}

/// The 256-lane kernel for one output cell: shift-add per term, multiply per lane.
#[inline(never)]
fn fiber_dot(xr: &XRow, f: &RowFibers) -> u16 {
    let mut total: i128 = 0;
    let mut sticky = false;
    for (lane, entries) in f.lanes.iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let mut acc: i128 = 0;
        for &(col, rung) in entries {
            let i = col as usize;
            let mx = xr.m[i];
            if mx == 0 {
                continue;
            }
            let sh = xr.n[i] + rung as i32 + ACC_FRAC;
            let t = if sh >= 0 {
                (mx as i128) << sh
            } else {
                let r = -sh;
                if r >= 128 {
                    sticky = true;
                    continue;
                }
                if ((mx as i128) & ((1i128 << r) - 1)) != 0 {
                    sticky = true;
                }
                (mx as i128) >> r
            };
            if xr.sheet[i] != 0 {
                acc -= t;
            } else {
                acc += t;
            }
        }
        let m = (2 * (lane % 128) + 1) as i128;
        if lane >= 128 {
            total -= m * acc;
        } else {
            total += m * acc;
        }
    }
    Bucket { acc: total, sticky }.collapse()
}

#[inline(never)]
fn cell_dot(xr: &XRow, g: &Grid, atoms: &Atoms, o: usize) -> u16 {
    let mut b = Bucket::new();
    for tc in 0..g.tile_cols() {
        let line = g.line(o, tc);
        let base = tc * g.tw;
        for (j, &id) in line.iter().enumerate() {
            let id = id as usize;
            let i = base + j;
            b.add(xr.sheet[i], xr.m[i], xr.n[i], atoms.sheet[id], atoms.m[id] as u32, atoms.n[id] as i32);
        }
    }
    b.collapse()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&args[1]);
    let id = args[2].clone();
    let intake = find_intake_dir(&root, &id);
    let model = Model::load(&id, &intake, &root.join("cells").join(&id)).unwrap();
    let cfg = model.cfg.clone();
    let eng = Engine::new(model, &PathBuf::from("D:/folded-weights/v5/tests/oracles/lut")).unwrap();
    let m = &eng.model;
    let g = m.grid("model.layers.0.self_attn.q_proj.weight");

    // a real activation row, and a second one from a different token for coverage
    let mut rows = Vec::new();
    for tok in [1usize, 785, 9625] {
        let x0 = m.embed_row(tok);
        let h = rmsnorm_rows(&eng, &x0, 1, cfg.hidden, m.norm("model.layers.0.input_layernorm.weight"));
        rows.push(XRow::new(&h));
    }

    // fold every output row of the tensor by (sheet, atom)
    let t = Instant::now();
    let fibers: Vec<RowFibers> = (0..g.rows).map(|o| fold_row(g, &m.atoms, o)).collect();
    let fold_s = t.elapsed().as_secs_f64();
    let nonempty: usize = fibers.iter().map(|f| f.lanes.iter().filter(|l| !l.is_empty()).count()).sum();
    let entries: usize = fibers.iter().map(|f| f.lanes.iter().map(|l| l.len()).sum::<usize>()).sum();
    println!("folded {} rows by (sheet, atom) in {:.2}s: {} entries ({} B at 3 B each), {:.1} non-empty lanes per row of 256",
        g.rows, fold_s, entries, entries * 3, nonempty as f64 / g.rows as f64);

    // bit-identity on every output cell, three activations
    let mut mism = 0usize;
    for (k, xr) in rows.iter().enumerate() {
        for o in 0..g.rows {
            let a = cell_dot(xr, g, &m.atoms, o);
            let b = fiber_dot(xr, &fibers[o]);
            if a != b {
                mism += 1;
                if mism <= 5 {
                    println!("  mismatch act {} row {}: bucket {:#06x} fiber {:#06x}", k, o, a, b);
                }
            }
        }
    }
    println!("bit-identity: {} mismatches over {} output cells x {} activations", mism, g.rows, rows.len());

    // timing, single thread, one full tensor row of outputs each
    let xr = &rows[0];
    let t = Instant::now();
    let mut s = 0u32;
    for o in 0..g.rows {
        s ^= cell_dot(xr, g, &m.atoms, o) as u32;
    }
    let ns_bucket = t.elapsed().as_nanos() as f64 / (g.rows * g.cols) as f64;
    let t = Instant::now();
    let mut s2 = 0u32;
    for o in 0..g.rows {
        s2 ^= fiber_dot(xr, &fibers[o]) as u32;
    }
    let ns_fiber = t.elapsed().as_nanos() as f64 / (g.rows * g.cols) as f64;
    println!("single thread, {} cells x {} terms: bucket {:.3} ns/term, 256-lane fiber {:.3} ns/term (sinks {} {})",
        g.rows, g.cols, ns_bucket, ns_fiber, s & 1, s2 & 1);
    let _ = cells::FINITE;
}
