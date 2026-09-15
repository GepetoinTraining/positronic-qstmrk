//! `atlas_cub` — write the fold cube (rung bands) for a model, or print its header.
//!
//!   atlas_cub write <root> <model-id>     → <root>/cells/<model-id>/fold.cub
//!   atlas_cub info  <root> <model-id>

use std::path::PathBuf;
use std::time::Instant;

use atlas::cub::{write_cub, CubFold};
use atlas::fold::TensorFold;
use atlas::surfaces::Model;

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
    if a.len() < 4 {
        eprintln!("usage: atlas_cub write|info <root> <model-id>");
        std::process::exit(2);
    }
    let root = PathBuf::from(&a[2]);
    let id = &a[3];
    let out = root.join("cells").join(id).join("fold.cub");
    match a[1].as_str() {
        "write" => {
            let intake = find_intake_dir(&root, id);
            let seal = std::fs::read_to_string(intake.join("RECEIPT.txt")).unwrap_or_default().lines().find(|l| l.starts_with("seal=")).map(|l| l[5..].to_string()).unwrap_or_default();
            let t = Instant::now();
            let model = Model::load(id, &intake, &root.join("cells").join(id)).unwrap();
            let mut names: Vec<String> = model.grids.keys().cloned().collect();
            names.sort();
            // a tied head that is byte-identical to the embedding is the same surface: seal it, do not store it twice
            if model.cfg.tied && model.grids.contains_key("lm_head.weight") {
                let same = model.grid("lm_head.weight").ids == model.grid("model.embed_tokens.weight").ids;
                if same {
                    names.retain(|n| n != "lm_head.weight");
                    println!("lm_head.weight is byte-identical to model.embed_tokens.weight (tied): sealed as one surface, not written twice");
                }
            }
            let mut folds: Vec<(String, TensorFold)> = Vec::new();
            for n in &names {
                folds.push((n.clone(), TensorFold::from_grid(model.grid(n), &model.atoms)));
            }
            println!("folds built: {} tensors in {} ms", folds.len(), t.elapsed().as_millis());
            let refs: Vec<(String, &TensorFold)> = folds.iter().map(|(n, f)| (n.clone(), f)).collect();
            let t = Instant::now();
            let specs = write_cub(&out, id, &seal, &refs).unwrap();
            let bytes = std::fs::metadata(&out).unwrap().len();
            println!("wrote {} ({} tensors, {} MB) in {} ms", out.display(), specs.len(), bytes / 1_000_000, t.elapsed().as_millis());
        }
        "info" => {
            let c = CubFold::open(&out).unwrap();
            println!("{} : {} MB, {} tensors", out.display(), c.bytes() / 1_000_000, c.specs.len());
            for s in &c.specs {
                println!("  {:44} lines {:>6} samples {:>5} rungs [{:>3}..{:>2}] bands {:>2} tiles {:>5} body {:>9} B", s.name, s.rows, s.cols, s.rmin, s.rmax, s.bands(), s.tiles, s.body_bytes);
            }
        }
        other => panic!("unknown command {}", other),
    }
}
