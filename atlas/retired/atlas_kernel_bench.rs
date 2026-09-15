//! `atlas_kernel_bench` — the footprint of the tiniest running kernel, measured, so it can
//! be multiplied. Single thread, real weights (layer 0 q_proj), a real activation row (the
//! embedding of token 1 through the layer-0 input norm), the production `Bucket`.
//!
//!   atlas_kernel_bench <root> <model-id>
//!
//! Reports: static bytes of every hot object; ns per term (MAC) at three grains — one tile
//! line (128 terms), one output cell (in_f terms), one tile row (128 cells) — and the
//! multiply-up: terms per pass for this model, predicted single-thread time, threads on this
//! box, predicted parallel time, and the measured seq-1 pass for comparison.

use std::path::PathBuf;
use std::time::Instant;

use atlas::forward::{forward_all, rmsnorm_rows, Engine};
use atlas::matmul::{Bucket, XRow};
use atlas::surfaces::{Atoms, Grid, Model, Place};

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

fn terms_per_pass(c: &atlas::surfaces::Config, seq: usize) -> u64 {
    let h = c.hidden as u64;
    let per_layer = h * c.q_dim() as u64 + 2 * h * c.kv_dim() as u64 + c.q_dim() as u64 * h + 3 * h * c.intermediate as u64;
    (per_layer * c.n_layers as u64 + h * c.vocab as u64) * seq as u64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&args[1]);
    let id = args[2].clone();
    let intake = find_intake_dir(&root, &id);
    let model = Model::load(&id, &intake, &root.join("cells").join(&id)).unwrap();
    let cfg = model.cfg.clone();
    let lut = PathBuf::from("D:/folded-weights/v5/tests/oracles/lut");
    let eng = Engine::new(model, &lut).unwrap();
    let m = &eng.model;

    // ── static footprint ─────────────────────────────────────────────────────
    println!("== static bytes ==");
    println!("Bucket (hot state of one output cell)      {} B", std::mem::size_of::<Bucket>());
    println!("XRow per activation element (sheet,m,n)    {} B", 1 + 4 + 4);
    println!("Atoms per dictionary id (sheet,m,n,pattern) {} B  x {} ids = {} B (L1/L2 resident)", 1 + 1 + 2 + 2, m.atoms.len(), 6 * m.atoms.len());
    println!("Grid per weight cell (id)                  2 B");
    println!("Place per tile (rank)                      4 B");
    let g = m.grid("model.layers.0.self_attn.q_proj.weight");
    println!("one tile                                   {} B of ids ({}x{})", g.th * g.tw * 2, g.th, g.tw);
    println!("one tile line (the inner loop's operand)   {} B of ids + {} B of activation", g.tw * 2, g.tw * 9);
    println!("Place for this tensor                      {} B ({} tiles)", g.place.rank.len() * 4, g.place.rank.len());

    // ── a real activation row ────────────────────────────────────────────────
    let x0 = m.embed_row(1);
    let h = rmsnorm_rows(&eng, &x0, 1, cfg.hidden, m.norm("model.layers.0.input_layernorm.weight"));
    let xr = XRow::new(&h);

    // ── one tile line: 128 terms ─────────────────────────────────────────────
    let reps = 200_000u64;
    let line = g.line(0, 0);
    let t = Instant::now();
    let mut sink = 0i128;
    for r in 0..reps {
        let mut b = Bucket::new();
        let base = (r as usize % g.tile_cols()) * g.tw;
        let line = g.line(0, r as usize % g.tile_cols());
        for (j, &id) in line.iter().enumerate() {
            let id = id as usize;
            let i = base + j;
            b.add(xr.sheet[i], xr.m[i], xr.n[i], m.atoms.sheet[id], m.atoms.m[id] as u32, m.atoms.n[id] as i32);
        }
        sink ^= b.acc;
    }
    let ns_line = t.elapsed().as_nanos() as f64 / reps as f64;
    let _ = line;
    println!("\n== single thread, layer 0 q_proj ({}x{}), real activation ==", g.rows, g.cols);
    println!("one tile line   128 terms   {:8.1} ns  = {:.3} ns/term   (sink {})", ns_line, ns_line / 128.0, sink & 1);

    // ── one output cell: in_f terms + collapse ───────────────────────────────
    let reps = 20_000usize;
    let t = Instant::now();
    let mut s2 = 0u32;
    for r in 0..reps {
        s2 ^= cell_dot(&xr, g, &m.atoms, r % g.rows) as u32;
    }
    let ns_cell = t.elapsed().as_nanos() as f64 / reps as f64;
    println!("one output cell {} terms  {:8.1} ns  = {:.3} ns/term + collapse (sink {})", g.cols, ns_cell, ns_cell / g.cols as f64, s2 & 1);

    // ── one tile row: 128 output cells ───────────────────────────────────────
    let t = Instant::now();
    let mut s3 = 0u32;
    for o in 0..128 {
        s3 ^= cell_dot(&xr, g, &m.atoms, o) as u32;
    }
    let ns_row = t.elapsed().as_nanos() as f64;
    println!("one tile row    128 cells   {:8.1} us = {:.3} ns/term (sink {})", ns_row / 1e3, ns_row / (128.0 * g.cols as f64), s3 & 1);

    // ── multiply-up ──────────────────────────────────────────────────────────
    let ns_term = ns_cell / g.cols as f64;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("\n== multiply-up for {} ==", id);
    for seq in [1usize, 8] {
        let terms = terms_per_pass(&cfg, seq);
        let single = terms as f64 * ns_term / 1e9;
        println!("seq-{}: {:>14} matmul terms/pass  -> {:7.2} s single-thread, {:6.2} s ideal on {} threads", seq, terms, single, single / threads as f64, threads);
    }
    let t = Instant::now();
    let _ = forward_all(&eng, &[1], &mut |_| {});
    let seq1 = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = forward_all(&eng, &[1, 2, 3, 4, 5, 6, 7, 8], &mut |_| {});
    let seq8 = t.elapsed().as_secs_f64();
    println!("measured full pass: seq-1 {:.2} s, seq-8 {:.2} s (attention, norms, RoPE, softmax, SiLU included; threads = {})", seq1, seq8, threads);
    println!("bytes of ids streamed per seq-1 pass: {} MB", terms_per_pass(&cfg, 1) * 2 / 1_000_000);
}
