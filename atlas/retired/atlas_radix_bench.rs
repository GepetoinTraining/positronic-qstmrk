//! `atlas_radix_bench` — the radix chain with the crank on the highest odd (pgarcia,
//! 2026-09-13 ~04:20, after mm-solver's crank: one integer advances, every gear reads).
//!
//! Σ_j (2j+1)·a_j = S_0 + 2·Σ_{j≥1} S_j, S_j the suffix sums from the highest odd down.
//! So the gather over the 128 odds is 255 adds and one shift, no multiply, and every odd's
//! multiple is produced on the way — every odd rotates together.
//!
//! Variant A (odd chain): per term shift-add into its (sheet, atom) lane as the fiber kernel;
//!   the gather is the suffix chain from 255 down.
//! Variant B (full radix): per term ONE ADD into its own dictionary id's bucket (atom, rung),
//!   the activation pre-shifted once per token to the bucket scale; then per atom the rung
//!   chain from the highest rung down (×2 = one shift, one add per rung; negative rungs by
//!   halving from the deepest up, dropped bits to sticky); then the odd chain. No multiply
//!   and no per-term shift anywhere. It computes 128 × (rung span) buckets per output cell —
//!   more than the 2,048 terms — and every one of them is an add.
//!
//! Judged by bit-identity with the production bucket on every output cell of layer-0
//! q_proj for three real activations, then ns per term.

use std::path::PathBuf;
use std::time::Instant;

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

// ── production bucket, for identity ──────────────────────────────────────────
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

/// The odd chain: Σ_j (2j+1)·a_j with the crank on the highest odd. 255 adds, one shift.
#[inline(always)]
fn odd_chain(a: &[i128; 128]) -> i128 {
    let mut s: i128 = 0;
    let mut t: i128 = 0;
    for j in (1..128).rev() {
        s += a[j];
        t += s;
    }
    s += a[0];
    s + (t << 1)
}

// ── variant A: fiber lanes + odd chain ───────────────────────────────────────
struct RowFibers {
    lanes: Vec<Vec<(u16, i8)>>, // 256 lanes: sheet*128 + (m-1)/2
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
            lanes[lane].push(((tc * g.tw + j) as u16, atoms.n[id] as i8));
        }
    }
    RowFibers { lanes }
}

#[inline(never)]
fn fiber_chain_dot(xr: &XRow, f: &RowFibers) -> u16 {
    let mut sticky = false;
    let mut lane_acc = [[0i128; 128]; 2];
    for (lane, entries) in f.lanes.iter().enumerate() {
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
        lane_acc[lane / 128][lane % 128] = acc;
    }
    let total = odd_chain(&lane_acc[0]) - odd_chain(&lane_acc[1]);
    Bucket { acc: total, sticky }.collapse()
}

// ── variant B: full radix — one add per term, rung chain, odd chain ───────────
/// Per row, static: for each id-bucket (sheet, atom, rung) the columns that carry it,
/// flattened: `cols` with `starts` per bucket index b = sheet*128*R + atom_idx*R + (rung - rmin).
struct RowRadix {
    rmin: i32,
    rspan: usize,
    starts: Vec<u32>, // len = 2*128*rspan + 1
    cols: Vec<u16>,
}

fn fold_row_radix(g: &Grid, atoms: &Atoms, o: usize, rmin: i32, rspan: usize) -> RowRadix {
    let nb = 2 * 128 * rspan;
    let mut lists: Vec<Vec<u16>> = (0..nb).map(|_| Vec::new()).collect();
    for tc in 0..g.tile_cols() {
        for (j, &id) in g.line(o, tc).iter().enumerate() {
            let id = id as usize;
            let m = atoms.m[id];
            if m == 0 {
                continue;
            }
            let b = (atoms.sheet[id] as usize) * 128 * rspan + (((m as usize) - 1) / 2) * rspan + (atoms.n[id] as i32 - rmin) as usize;
            lists[b].push((tc * g.tw + j) as u16);
        }
    }
    let mut starts = Vec::with_capacity(nb + 1);
    let mut cols = Vec::new();
    for l in &lists {
        starts.push(cols.len() as u32);
        cols.extend_from_slice(l);
    }
    starts.push(cols.len() as u32);
    RowRadix { rmin, rspan, starts, cols }
}

/// The activation pre-shifted once per token to the bucket scale 2^-ACC_FRAC, signed.
struct XFixed {
    v: Vec<i128>,
    sticky: bool,
}

fn prefix_activation(xr: &XRow) -> XFixed {
    let mut v = Vec::with_capacity(xr.m.len());
    let mut sticky = false;
    for i in 0..xr.m.len() {
        let mx = xr.m[i] as i128;
        let sh = xr.n[i] + ACC_FRAC;
        let t = if mx == 0 {
            0
        } else if sh >= 0 {
            mx << sh
        } else {
            let r = -sh;
            if r < 128 && (mx & ((1i128 << r) - 1)) != 0 {
                sticky = true;
            }
            if r >= 128 { 0 } else { mx >> r }
        };
        v.push(if xr.sheet[i] != 0 { -t } else { t });
    }
    XFixed { v, sticky }
}

#[inline(never)]
fn radix_dot(xf: &XFixed, r: &RowRadix, scratch: &mut Vec<i128>) -> u16 {
    let rspan = r.rspan;
    let nb = 2 * 128 * rspan;
    scratch.clear();
    scratch.resize(nb, 0);
    // one add per term
    for b in 0..nb {
        let (s, e) = (r.starts[b] as usize, r.starts[b + 1] as usize);
        if s == e {
            continue;
        }
        let mut acc = 0i128;
        for &c in &r.cols[s..e] {
            acc += xf.v[c as usize];
        }
        scratch[b] = acc;
    }
    // rung chain per (sheet, atom), then odd chain per sheet
    let mut sticky = xf.sticky;
    let mut lane_acc = [[0i128; 128]; 2];
    let zero_idx = (-r.rmin) as usize; // bucket index of rung 0 within an atom's span
    for sheet in 0..2 {
        for a in 0..128 {
            let base = sheet * 128 * rspan + a * rspan;
            let bk = &scratch[base..base + rspan];
            // non-negative rungs (if any in the span): from the top down, R = 2R + B_n
            let mut hi: i128 = 0;
            if zero_idx < rspan {
                for k in (zero_idx..rspan).rev() {
                    hi = (hi << 1) + bk[k];
                }
            }
            // negative rungs: from the deepest up, Q = (Q >> 1) + B_n; at the end Q holds
            // Σ 2^(n − n_last) B_n and is halved (−n_last) more times. Dropped bits → sticky.
            let neg_count = zero_idx.min(rspan);
            let mut lo: i128 = 0;
            if neg_count > 0 {
                for k in 0..neg_count {
                    if lo & 1 != 0 {
                        sticky = true;
                    }
                    lo = (lo >> 1) + bk[k];
                }
                let n_last = r.rmin + neg_count as i32 - 1; // ≤ -1
                let fin = (-n_last) as u32;
                if (lo & ((1i128 << fin) - 1)) != 0 {
                    sticky = true;
                }
                lo >>= fin;
            }
            lane_acc[sheet][a] = hi + lo;
        }
    }
    let total = odd_chain(&lane_acc[0]) - odd_chain(&lane_acc[1]);
    Bucket { acc: total, sticky }.collapse()
}


// ── variant C: base as denominator, x4 by planes ──────────────────────────────
/// The tensor's lowest rung K is the one denominator 2^K. Activations are prefixed at scale
/// ACC_FRAC + rmin (rmin ≤ 0) so the rung chain is doubling only, from the highest rung down,
/// and lands at scale ACC_FRAC exactly. Four activation planes (four tokens) ride one walk of
/// the ids: four buckets per id, four lane arrays, four collapses.
fn prefix_plane(xr: &XRow, rmin: i32) -> XFixed {
    let mut v = Vec::with_capacity(xr.m.len());
    let mut sticky = false;
    for i in 0..xr.m.len() {
        let mx = xr.m[i] as i128;
        let sh = xr.n[i] + ACC_FRAC + rmin;
        let t = if mx == 0 {
            0
        } else if sh >= 0 {
            mx << sh
        } else {
            let r = -sh;
            if r < 128 && (mx & ((1i128 << r) - 1)) != 0 {
                sticky = true;
            }
            if r >= 128 { 0 } else { mx >> r }
        };
        v.push(if xr.sheet[i] != 0 { -t } else { t });
    }
    XFixed { v, sticky }
}

#[inline(never)]
fn radix4_dot(planes: &[XFixed; 4], r: &RowRadix, scratch: &mut Vec<[i128; 4]>) -> [u16; 4] {
    let rspan = r.rspan;
    let nb = 2 * 128 * rspan;
    scratch.clear();
    scratch.resize(nb, [0; 4]);
    // one add per term per plane; the ids walked once
    for b in 0..nb {
        let (s, e) = (r.starts[b] as usize, r.starts[b + 1] as usize);
        if s == e {
            continue;
        }
        let mut acc = [0i128; 4];
        for &c in &r.cols[s..e] {
            let c = c as usize;
            acc[0] += planes[0].v[c];
            acc[1] += planes[1].v[c];
            acc[2] += planes[2].v[c];
            acc[3] += planes[3].v[c];
        }
        scratch[b] = acc;
    }
    // rung chain: doubling only, from the top rung down; then the odd chain; per sheet, per plane
    let mut lane_acc = [[[0i128; 128]; 2]; 4];
    for sheet in 0..2 {
        for a in 0..128 {
            let base = sheet * 128 * rspan + a * rspan;
            let mut hi = [0i128; 4];
            for k in (0..rspan).rev() {
                let bk = &scratch[base + k];
                for p in 0..4 {
                    hi[p] = (hi[p] << 1) + bk[p];
                }
            }
            for p in 0..4 {
                lane_acc[p][sheet][a] = hi[p];
            }
        }
    }
    let mut out = [0u16; 4];
    for p in 0..4 {
        let total = odd_chain(&lane_acc[p][0]) - odd_chain(&lane_acc[p][1]);
        out[p] = Bucket { acc: total, sticky: planes[p].sticky }.collapse();
    }
    out
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

    let mut rows = Vec::new();
    for tok in [1usize, 785, 9625] {
        let x0 = m.embed_row(tok);
        let h = rmsnorm_rows(&eng, &x0, 1, cfg.hidden, m.norm("model.layers.0.input_layernorm.weight"));
        rows.push(XRow::new(&h));
    }

    // rung span of this tensor
    let (mut rmin, mut rmax) = (i32::MAX, i32::MIN);
    for tc in 0..g.tile_cols() {
        for o in 0..g.rows {
            for &id in g.line(o, tc) {
                let id = id as usize;
                if m.atoms.m[id] != 0 {
                    rmin = rmin.min(m.atoms.n[id] as i32);
                    rmax = rmax.max(m.atoms.n[id] as i32);
                }
            }
        }
    }
    let rspan = (rmax - rmin + 1) as usize;
    println!("layer-0 q_proj rungs [{}..{}], span {}; buckets per output cell = 2 x 128 x {} = {}", rmin, rmax, rspan, rspan, 256 * rspan);

    let t = Instant::now();
    let fibers: Vec<RowFibers> = (0..g.rows).map(|o| fold_row(g, &m.atoms, o)).collect();
    let radix: Vec<RowRadix> = (0..g.rows).map(|o| fold_row_radix(g, &m.atoms, o, rmin, rspan)).collect();
    println!("folds built in {:.2}s (radix: {} B of starts per row + 2 B per term)", t.elapsed().as_secs_f64(), (256 * rspan + 1) * 4);

    // identity
    let mut mism_a = 0usize;
    let mut mism_b = 0usize;
    let mut scratch = Vec::new();
    for xr in rows.iter() {
        let xf = prefix_activation(xr);
        for o in 0..g.rows {
            let want = cell_dot(xr, g, &m.atoms, o);
            let a = fiber_chain_dot(xr, &fibers[o]);
            let b = radix_dot(&xf, &radix[o], &mut scratch);
            if a != want {
                mism_a += 1;
                if mism_a <= 3 {
                    println!("  A mismatch row {}: bucket {:#06x} chain {:#06x}", o, want, a);
                }
            }
            if b != want {
                mism_b += 1;
                if mism_b <= 3 {
                    println!("  B mismatch row {}: bucket {:#06x} radix {:#06x}", o, want, b);
                }
            }
        }
    }
    println!("bit-identity over {} cells x {} activations: A (fiber + odd chain) {} mismatches; B (full radix) {} mismatches", g.rows, rows.len(), mism_a, mism_b);

    // variant C: four tokens per turn, base as denominator
    let mut rows4 = Vec::new();
    for tok in [1usize, 785, 9625, 374] {
        let x0 = m.embed_row(tok);
        let h = rmsnorm_rows(&eng, &x0, 1, cfg.hidden, m.norm("model.layers.0.input_layernorm.weight"));
        rows4.push(XRow::new(&h));
    }
    let planes: [XFixed; 4] = [prefix_plane(&rows4[0], rmin), prefix_plane(&rows4[1], rmin), prefix_plane(&rows4[2], rmin), prefix_plane(&rows4[3], rmin)];
    let mut scratch4: Vec<[i128; 4]> = Vec::new();
    let mut mism_c = 0usize;
    for o in 0..g.rows {
        let got = radix4_dot(&planes, &radix[o], &mut scratch4);
        for p in 0..4 {
            let want = cell_dot(&rows4[p], g, &m.atoms, o);
            if got[p] != want {
                mism_c += 1;
                if mism_c <= 3 {
                    println!("  C mismatch row {} plane {}: bucket {:#06x} radix4 {:#06x}", o, p, want, got[p]);
                }
            }
        }
    }
    println!("bit-identity C (base as denominator, x4 planes) over {} cells x 4 tokens: {} mismatches", g.rows, mism_c);
    let t = Instant::now();
    let mut s4 = 0u32;
    for o in 0..g.rows {
        let r4 = radix4_dot(&planes, &radix[o], &mut scratch4);
        s4 ^= (r4[0] ^ r4[1] ^ r4[2] ^ r4[3]) as u32;
    }
    let ns4 = t.elapsed().as_nanos() as f64 / (4.0 * g.rows as f64 * g.cols as f64);
    println!("C: {:.3} ns per term per token, four tokens per walk of the ids (sink {})", ns4, s4 & 1);

    // timing
    let xr = &rows[0];
    let n = (g.rows * g.cols) as f64;
    let t = Instant::now();
    let mut s = 0u32;
    for o in 0..g.rows {
        s ^= cell_dot(xr, g, &m.atoms, o) as u32;
    }
    let ns0 = t.elapsed().as_nanos() as f64 / n;
    let t = Instant::now();
    let mut s1 = 0u32;
    for o in 0..g.rows {
        s1 ^= fiber_chain_dot(xr, &fibers[o]) as u32;
    }
    let ns1 = t.elapsed().as_nanos() as f64 / n;
    let t = Instant::now();
    let xf = prefix_activation(xr);
    let pre = t.elapsed().as_nanos() as f64;
    let t = Instant::now();
    let mut s2 = 0u32;
    for o in 0..g.rows {
        s2 ^= radix_dot(&xf, &radix[o], &mut scratch) as u32;
    }
    let ns2 = t.elapsed().as_nanos() as f64 / n;
    println!("single thread, ns/term: bucket {:.3} | A fiber+odd-chain {:.3} | B full radix {:.3} (activation prefix {:.1} us once per token) (sinks {}{}{})",
        ns0, ns1, ns2, pre / 1e3, s & 1, s1 & 1, s2 & 1);
}
