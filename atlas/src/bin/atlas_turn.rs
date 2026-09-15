//! `atlas_turn` — the two clocks, judged: acquire, then re-slice through the checker, and the
//! logits of every position must equal the full re-parse's reference hashes.
//!
//!   atlas_turn <root> <model-id>
//!
//! Sequence: turn [1] (fresh acquisition) → hash vs reference seq1; turn [1..8] (7 picked-up
//! positions? no: 1 picked up, 7 computed) → hash of all 8 rows vs reference seq8; turn France
//! (prefix differs → fresh) → hash vs reference france; then turn [1..8] again from a fresh
//! acquisition of [1..4] → same seq8 hash (a re-slice from a 4-position acquisition).

use std::path::PathBuf;
use std::time::Instant;

use atlas::cache::Acquisition;
use atlas::forward::Engine;
use atlas::surfaces::Model;
use atlas::turn::forward_turn;
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

fn sha_rows(logits: &[u16]) -> String {
    let mut h = Sha256::new();
    for &p in logits {
        h.update(p.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&a[1]);
    let id = &a[2];
    let intake = find_intake_dir(&root, id);
    let model = Model::load(id, &intake, &root.join("cells").join(id)).unwrap();
    let eng = Engine::new(model, &PathBuf::from("D:/folded-weights/v5/tests/oracles/lut")).unwrap();
    let reference = std::fs::read_to_string(root.join("receipts").join(format!("reference-{}.txt", id))).unwrap_or_default();
    let want = |label: &str| -> String { reference.lines().find(|l| l.starts_with(label)).map(|l| l.split_whitespace().nth(1).unwrap_or("").to_string()).unwrap_or_default() };

    let mut acq = Acquisition::empty(eng.model.cfg.n_layers);
    let mut lc: Vec<Vec<u16>> = Vec::new();
    let mut report = String::from("format=atlas.turn.v1\n");
    let mut pass = true;
    let steps: [(&str, Vec<usize>, &str); 5] = [
        ("acquire [1]", vec![1], "seq1"),
        ("re-slice [1..8] over the [1] acquisition", (1..=8).collect(), "seq8"),
        ("France: prefix differs, fresh acquisition", vec![785, 6722, 315, 9625, 374], "france"),
        ("acquire [1..4]", (1..=4).collect(), ""),
        ("re-slice [1..8] over the [1..4] acquisition", (1..=8).collect(), "seq8"),
    ];
    for (label, tokens, refkey) in steps.iter() {
        let t = Instant::now();
        let (logits, kept) = forward_turn(&eng, &mut acq, tokens, &mut lc);
        let h = sha_rows(&logits);
        let secs = t.elapsed().as_millis();
        let verdict = if refkey.is_empty() {
            "(no reference row)".to_string()
        } else {
            let w = want(refkey);
            let ok = w == h;
            pass &= ok;
            format!("vs reference {}: {}", refkey, if ok { "EQUAL" } else { "DIFFERENT" })
        };
        let line = format!("[{}] positions={} picked_up={} computed={} addresses={:?} hash={} {} {} ms\n", label, tokens.len(), kept, tokens.len() - kept,
            acq.addresses.iter().map(|ad| ad.crt).collect::<Vec<_>>(), &h[..16], verdict, secs);
        print!("{}", line);
        report.push_str(&line);
    }
    let v = format!("gate (two clocks): {}\n", if pass { "PASS — every re-slice equals the full re-parse, bit for bit" } else { "FAIL" });
    print!("{}", v);
    report.push_str(&v);
    let out = root.join("receipts").join(format!("turn-{}.txt", id));
    std::fs::write(&out, report).unwrap();
    println!("receipt: {}", out.display());
}
