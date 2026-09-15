//! `atlas_capture` — the residual stream at a layer, exact, for Prediction 6 (native_object.pdf):
//! "the residual stream at layer L for a single prompt is not spectrally white in at least one
//! of the orderings under which the weights were." The engine writes the exact bf16 patterns;
//! the instrument (float, marked) reads them elsewhere.
//!
//!   atlas_capture <root> <model-id> <layer> [--tag <name>] --tokens t1 t2 ...
//!
//! Writes receipts/capture-<model>-L<layer>-<n>tok.u16 : [seq, hidden] u16 LE patterns of the
//! residual stream ENTERING layer <layer> (after <layer> decoder layers), plus a .meta line.

use std::path::PathBuf;

use atlas::forward::{decoder_layer, Engine};
use atlas::surfaces::Model;
use sha2::{Digest, Sha256};

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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&a[1]);
    let id = &a[2];
    let layer: usize = a[3].parse().unwrap();
    let tag = a.iter().position(|s| s == "--tag").map(|i| format!("-{}", a[i + 1])).unwrap_or_default();
    let ti = a.iter().position(|s| s == "--tokens").expect("--tokens");
    let tokens: Vec<usize> = a[ti + 1..].iter().map(|s| s.parse().unwrap()).collect();
    let intake = find_intake_dir(&root, id);
    let model = Model::load(id, &intake, &root.join("cells").join(id)).unwrap();
    let eng = Engine::new(model, &PathBuf::from("D:/folded-weights/v5/tests/oracles/lut")).unwrap();
    let c = &eng.model.cfg;
    let seq = tokens.len();
    let mut x = Vec::with_capacity(seq * c.hidden);
    for &t in &tokens {
        x.extend(eng.model.embed_row(t));
    }
    for l in 0..layer {
        x = decoder_layer(&eng, x, seq, l);
        eprintln!("layer {}/{}", l + 1, layer);
    }
    let mut bytes = Vec::with_capacity(x.len() * 2);
    for &p in &x {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    let mut h = Sha256::new();
    h.update(&bytes);
    let sha = format!("{:x}", h.finalize());
    let stem = format!("capture-{}-L{}-{}tok{}", id, layer, seq, tag);
    let out = root.join("receipts").join(format!("{}.u16", stem));
    std::fs::write(&out, &bytes).unwrap();
    let meta = format!("model={} layer={} seq={} hidden={} tokens={:?} sha256={}\n", id, layer, seq, c.hidden, tokens, sha);
    std::fs::write(root.join("receipts").join(format!("{}.meta", stem)), &meta).unwrap();
    print!("{}", meta);
    println!("wrote {}", out.display());
}
