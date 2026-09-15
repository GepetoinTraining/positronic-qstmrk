//! `atlas_ball` — BALL(a, k) over one real tensor: gates and a projection receipt.
//!
//!   atlas_ball <root> <model-id> <tensor-name> <prime in {2,3,5,7,11,13}> <k> [W H digits]
//!
//! Prints: the seal gates (orthogonality, periodicity census), tight() at k = 0 and k, the
//! frame scale |q|^{2k}, the silhouette at level 5, and the sha256 of the level-0 cell table
//! at both k — the frame's receipt. Writes receipts/ball-<model>-<tensor>-q<p>-k<k>.txt.

use std::path::PathBuf;
use std::time::Instant;

use atlas::ball::{ball, orthogonal, period, points_of, Quat, PRIMES, SPIN_Q};
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

fn sha(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: atlas_ball <root> <model-id> <tensor> <prime> <k> [W H digits rows]");
        std::process::exit(2);
    }
    let root = PathBuf::from(&a[1]);
    let id = &a[2];
    let tensor = &a[3];
    let prime: i64 = a[4].parse().unwrap();
    let k: u32 = a[5].parse().unwrap();
    let w: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(512);
    let h: usize = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(512);
    let digits: u32 = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(12);
    let rows_limit: usize = a.get(9).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
    let qi = PRIMES.iter().position(|&p| p == prime).expect("prime must be one of 2 3 5 7 11 13");
    let q = Quat::from_i64(SPIN_Q[qi]);

    let mut rep = String::new();
    let mut line = |s: String, rep: &mut String| {
        println!("{}", s);
        rep.push_str(&s);
        rep.push('\n');
    };
    line(format!("format=atlas.ball.v1\nmodel={} tensor={} q=({},{},{},{}) |q|^2={} k={} W={} H={} digits={}", id, tensor, SPIN_Q[qi][0], SPIN_Q[qi][1], SPIN_Q[qi][2], SPIN_Q[qi][3], prime, k, w, h, digits), &mut rep);

    // seal gates
    for (i, s) in SPIN_Q.iter().enumerate() {
        let qq = Quat::from_i64(*s);
        line(format!("[seal {}] |q|^2={} orthogonal(M^T M = |q|^4 I)={} period(<=512)={}", PRIMES[i], qq.norm(), orthogonal(&qq), period(&qq, 512).map(|p| p.to_string()).unwrap_or("none".into())), &mut rep);
    }

    let intake = find_intake_dir(&root, id);
    let model = Model::load(id, &intake, &root.join("cells").join(id)).unwrap();
    let g = model.grid(tensor);
    let mut pts = points_of(g, &model.atoms);
    if rows_limit < g.rows {
        pts.truncate(rows_limit * g.cols); // file order = tile-major; the first rows_limit*cols places
    }
    line(format!("points: {} cells of {} ({}x{}){}", pts.len(), tensor, g.rows, g.cols, if rows_limit < g.rows { format!(" [first {} rows of places]", rows_limit) } else { String::new() }), &mut rep);

    for kk in [0u32, k] {
        let t = Instant::now();
        let (mesh, scale) = ball(&pts, &q, kk, w, h, digits);
        let tight0 = mesh.tight(0);
        let tight3 = mesh.tight(3);
        let bytes = mesh.table_bytes(0);
        let occupied = mesh.cells.len();
        line(format!("[k={}] frame scale |q|^2k={} ({} digits) | tight(level 0)={} tight(level 3)={} | occupied cells {} of {} | table sha256={} | {} ms",
            kk, if scale.to_string().len() > 40 { format!("{}...", &scale.to_string()[..40]) } else { scale.to_string() }, scale.to_string().len(),
            tight0, tight3, occupied, w * h, sha(&bytes), t.elapsed().as_millis()), &mut rep);
        line(format!("[k={}] silhouette, level 5 ({}x{}):\n{}", kk, w >> 5, h >> 5, mesh.silhouette(5)), &mut rep);
        if !(tight0 && tight3) {
            line("GATE FAIL: R and O are not inverses".into(), &mut rep);
        }
    }
    let out = root.join("receipts").join(format!("ball-{}-{}-q{}-k{}.txt", id, tensor.replace('.', "_"), prime, k));
    std::fs::write(&out, rep).unwrap();
    println!("receipt: {}", out.display());
}
