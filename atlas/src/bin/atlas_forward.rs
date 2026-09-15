//! `atlas_forward` — run the engine on the anchor inputs. The reference is the engine's own
//! first sealed receipt (receipts/reference-<model>.txt): a replay must reproduce every logit
//! hash (gate 5). The float reference was retired 2026-09-13 (pgarcia: "we're more precise
//! than the model's floats"); if receipts/anchor-<model>.json is present it is still judged,
//! as an observation, never as the gate.
//!
//!   atlas_forward <root> <model-id> [--lut <dir>] [--tokens 1 2 3 ...]
//!
//! Reads: <root>/receipts/intake-<model-id>-*/ (DICT.u16, MANIFEST.json, config.json),
//!        <root>/cells/<model-id>/*.grid, the V5 LUTs. Nothing else — no safetensors (gate 8).
//! Writes: <root>/receipts/forward-<model-id>-<stamp>.txt with the results and the
//!         sha256 of every logit row (gate 5: a replay must reproduce this hash).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use atlas::forward::{argmax, forward_all, topk, Engine};
use atlas::surfaces::Model;
use sha2::{Digest, Sha256};

fn find_intake_dir(root: &Path, id: &str) -> PathBuf {
    let prefix = format!("intake-{}-", id);
    let mut best: Option<PathBuf> = None;
    for e in fs::read_dir(root.join("receipts")).expect("receipts dir") {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with(&prefix) && best.as_ref().map_or(true, |b| name > b.file_name().unwrap().to_string_lossy().to_string()) {
            best = Some(p);
        }
    }
    best.unwrap_or_else(|| panic!("no intake receipt for {}", id))
}

fn json_ints_after(text: &str, key: &str) -> Option<Vec<usize>> {
    let k = format!("\"{}\": [", key);
    let p = text.find(&k)? + k.len();
    let e = p + text[p..].find(']')?;
    Some(text[p..e].split(',').filter_map(|s| s.trim().parse().ok()).collect())
}

fn json_int_after(text: &str, key: &str) -> Option<usize> {
    let k = format!("\"{}\": ", key);
    let p = text.find(&k)? + k.len();
    let rest = &text[p..];
    let e = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..e].parse().ok()
}

struct Anchor {
    name: String,
    tokens: Vec<usize>,
    argmax: usize,
    top5: Vec<usize>,
    per_pos: Vec<usize>,
}

fn load_anchors(path: &Path) -> Vec<Anchor> {
    let text = fs::read_to_string(path).expect("anchor json");
    let mut out = Vec::new();
    for name in ["seq1", "seq8", "france"] {
        let k = format!("\"{}\": {{", name);
        let Some(p) = text.find(&k) else { continue };
        let body = &text[p..];
        let end = body.find("\n  }").unwrap_or(body.len());
        let body = &body[..end];
        out.push(Anchor {
            name: name.to_string(),
            tokens: json_ints_after(body, "tokens").unwrap(),
            argmax: json_int_after(body, "argmax").unwrap(),
            top5: json_ints_after(body, "top5").unwrap(),
            per_pos: json_ints_after(body, "per_position_argmax").unwrap(),
        });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: atlas_forward <root> <model-id> [--lut <dir>] [--tokens ...]");
        std::process::exit(2);
    }
    let root = PathBuf::from(&args[1]);
    let id = args[2].clone();
    let mut lut = PathBuf::from("D:/folded-weights/v5/tests/oracles/lut");
    let mut explicit: Option<Vec<usize>> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--lut" => {
                lut = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--tokens" => {
                explicit = Some(args[i + 1..].iter().map(|s| s.parse().unwrap()).collect());
                break;
            }
            other => panic!("unknown arg {}", other),
        }
    }

    let intake = find_intake_dir(&root, &id);
    let cells_dir = root.join("cells").join(&id);
    let t0 = Instant::now();
    let model = Model::load(&id, &intake, &cells_dir).expect("load surfaces");
    println!(
        "surfaces loaded: {} grids, {} norms, alphabet {} ids; cfg hidden={} inter={} layers={} heads={}/{} vocab={} ({}s)",
        model.grids.len(), model.norms.len(), model.atoms.len(), model.cfg.hidden, model.cfg.intermediate,
        model.cfg.n_layers, model.cfg.n_q_heads, model.cfg.n_kv_heads, model.cfg.vocab, t0.elapsed().as_millis() as u64 / 1000
    );
    let eng = Engine::new(model, &lut).expect("LUTs");

    let mut report = String::new();
    report.push_str(&format!("format=atlas.forward.v1\nmodel={}\nintake={}\n", id, intake.display()));
    let vocab = eng.model.cfg.vocab;

    let run = |tokens: &[usize], label: &str, report: &mut String| -> (usize, Vec<usize>, Vec<usize>, Vec<u16>) {
        let t = Instant::now();
        let mut last = 0usize;
        let logits = forward_all(&eng, tokens, &mut |l| {
            last = l;
            if (l + 1) % 7 == 0 {
                eprintln!("  [{}] layer {}/{} ({}s)", label, l + 1, eng.model.cfg.n_layers, t.elapsed().as_millis() as u64 / 1000);
            }
        });
        let seq = tokens.len();
        let per_pos: Vec<usize> = (0..seq).map(|s| argmax(&logits[s * vocab..(s + 1) * vocab])).collect();
        let lastrow = &logits[(seq - 1) * vocab..seq * vocab];
        let am = argmax(lastrow);
        let t5 = topk(lastrow, 5);
        let mut h = Sha256::new();
        for &p in &logits {
            h.update(p.to_le_bytes());
        }
        let hash = format!("{:x}", h.finalize());
        let line = format!(
            "[{}] tokens={:?} argmax={} top5={:?} per_position_argmax={:?} logits_sha256={} seconds={}\n",
            label, tokens, am, t5, per_pos, hash, t.elapsed().as_millis() as u64 / 1000
        );
        print!("{}", line);
        report.push_str(&line);
        let _ = last;
        (am, t5, per_pos, lastrow.to_vec())
    };
    let vals = |row: &[u16], ids: &[usize]| -> Vec<String> { ids.iter().map(|&i| format!("{}:{:#06x}", i, row[i])).collect() };

    if let Some(tokens) = explicit {
        let _ = run(&tokens, "tokens", &mut report);
    } else {
        let anchor_path = root.join("receipts").join(format!("anchor-{}.json", id));
        let anchors = if anchor_path.exists() { load_anchors(&anchor_path) } else { Vec::new() };
        if anchors.is_empty() {
            let _ = run(&[1], "seq1", &mut report);
            let _ = run(&(1..=8).collect::<Vec<_>>(), "seq8", &mut report);
            let _ = run(&[785, 6722, 315, 9625, 374], "france", &mut report); // "The capital of France is"
        } else {
            let mut pass = true;
            for a in &anchors {
                let (am, t5, pp, row) = run(&a.tokens, &a.name, &mut report);
                let ok_am = am == a.argmax;
                let ok_t5 = t5 == a.top5;
                let ok_pp = pp == a.per_pos;
                if !ok_t5 {
                    let d = format!("[diag:{}] mine {:?} recorded {:?}\n", a.name, vals(&row, &t5), vals(&row, &a.top5));
                    print!("{}", d);
                    report.push_str(&d);
                }
                let verdict = format!(
                    "[gate 4:{}] argmax {} (recorded {}) top5 {} per_position {}\n",
                    a.name,
                    if ok_am { "ok" } else { "FAIL" },
                    a.argmax,
                    if ok_t5 { "ok".to_string() } else { format!("FAIL recorded {:?}", a.top5) },
                    if ok_pp { "ok".to_string() } else { format!("FAIL recorded {:?}", a.per_pos) }
                );
                print!("{}", verdict);
                report.push_str(&verdict);
                pass &= ok_am && ok_t5 && ok_pp;
            }
            let v = format!("gate 4: {}\n", if pass { "PASS — argmax, top-5 and every position equal the recorded oracle" } else { "FAIL" });
            print!("{}", v);
            report.push_str(&v);
        }
    }
    // gate 5 — replay against the engine's own reference receipt
    let reference = root.join("receipts").join(format!("reference-{}.txt", id));
    let hashes: Vec<(String, String)> = report
        .lines()
        .filter(|l| l.starts_with('[') && l.contains("logits_sha256="))
        .map(|l| {
            let label = l[1..l.find(']').unwrap()].to_string();
            let h = l.split("logits_sha256=").nth(1).unwrap().split_whitespace().next().unwrap().to_string();
            (label, h)
        })
        .collect();
    if reference.exists() {
        let want = fs::read_to_string(&reference).unwrap();
        let mut ok = true;
        for (label, h) in &hashes {
            let line = format!("{} {}", label, h);
            if !want.lines().any(|l| l == line) {
                ok = false;
                let v = format!("[gate 5:{}] FAIL — logits hash {} not in {}
", label, h, reference.display());
                print!("{}", v);
                report.push_str(&v);
            }
        }
        let v = format!("gate 5: {} — {} logit rows {} the reference receipt
", if ok { "PASS" } else { "FAIL" }, hashes.len(), if ok { "byte-identical to" } else { "differ from" });
        print!("{}", v);
        report.push_str(&v);
    } else {
        let body: String = hashes.iter().map(|(l, h)| format!("{} {}
", l, h)).collect();
        fs::write(&reference, &body).expect("write reference");
        let v = format!("gate 5: reference receipt written to {} ({} logit rows); every later run must reproduce it
", reference.display(), hashes.len());
        print!("{}", v);
        report.push_str(&v);
    }
    let stamp = {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        format!("{}", secs)
    };
    let out = root.join("receipts").join(format!("forward-{}-{}.txt", id, stamp));
    fs::write(&out, &report).expect("write receipt");
    println!("receipt: {}", out.display());
    let tr = atlas::fold::PREFIX_TRUNCATIONS.load(std::sync::atomic::Ordering::Relaxed);
    let tot = atlas::fold::PREFIX_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    if tot > 0 {
        println!("prefix floor: {} of {} activations truncated (sticky)", tr, tot);
    }
}
