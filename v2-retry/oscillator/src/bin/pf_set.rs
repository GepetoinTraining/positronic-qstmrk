//! `pf_set` — the Pedro Fibonacci set of a TPB run, collapsed to factors (pgarcia, 2026-09-14).
//!
//!   pf_set [run-dir]     default tpb/run-c32  →  <run-dir>/pf.txt (written once)

#[path = "../pf.rs"]
mod pf;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn main() {
    let run_dir: PathBuf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tpb").join("run-c32"));
    let run = pf::read(&run_dir);
    let (lims, ledger) = pf::pedro_fibonacci(&run);

    let mut out = String::new();
    writeln!(out, "format=v2.pf_set.v1 run={} cycles={}", run_dir.display(), run.cycles).unwrap();
    writeln!(out, "Pedro Fibonacci set: the relation before each square closes, collapsed to factors").unwrap();
    for l in &lims {
        let fb = ledger_text(&ledger, l.below);
        let fa = ledger_text(&ledger, l.above);
        let note = match l.deduped {
            Some(n) => format!("   collapses entry {} → {}", n, pf::rods_text(&l.num)),
            None => String::new(),
        };
        writeln!(out, "cycle {:>2}  whole {:>4}²  neighbours {} × {}  →  {}{}", l.cycle, l.whole, fb, fa, pf::relation_text(&l.num, &l.den), note).unwrap();
    }
    let deduped: Vec<u64> = lims.iter().filter_map(|l| l.deduped).collect();
    writeln!(out, "\nentries collapsed by the set: {:?}", deduped).unwrap();

    // outside check, reported only: arrivals still standing as atoms that the line would split
    let standing: Vec<u64> = run.seam.iter().copied().filter(|&n| ledger.factors(n) == Some(pf::Rods::from([(n, 1)])) && pf::shadow_composite(n)).collect();
    writeln!(out, "shadow check (not used): {} of {} seam entries still stand as atoms though the line splits them; first: {:?}", standing.len(), run.seam.len(), standing.iter().take(16).collect::<Vec<_>>()).unwrap();

    let path = run_dir.join("pf.txt");
    let mut f = std::fs::OpenOptions::new().create_new(true).write(true).open(&path).unwrap_or_else(|_| {
        eprintln!("refused: {} is already written", path.display());
        std::process::exit(2)
    });
    use std::io::Write as _;
    f.write_all(out.as_bytes()).unwrap();
    print!("{}", out);
}

fn ledger_text(ledger: &pf::Ledger, n: u64) -> String {
    ledger.factors(n).map(|r| if r.len() > 1 { format!("({})", pf::rods_text(&r)) } else { pf::rods_text(&r) }).unwrap_or_else(|| "?".into())
}
