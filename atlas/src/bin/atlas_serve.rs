//! `atlas_serve` — the engine held open: model loaded once, the acquisition kept between turns,
//! ids in and ids out over stdin/stdout. The tokenizer stage (intake/tokenizer.py, or the chat
//! harness intake/chat.py) sits in front of it; this side never sees text.
//!
//!   atlas_serve <root> <model-id>
//!
//! Protocol, one command per line:
//!   GEN <max_new> <id> <id> ...   run the transcript through the two clocks (old positions picked
//!                                 up, new ones computed), then generate greedily up to max_new
//!                                 tokens, printing `TOK <id>` as each is produced; stops at an EOS
//!                                 id; ends with `END picked_up=<n> computed=<m> generated=<g> ms=<t>`
//!   LOOK <g>                      lookahead: fill up to g extra planes per pass with the tokens
//!                                 that followed the last occurrence of the current n-gram in the
//!                                 transcript (its own sequences, cached); the comparator keeps a
//!                                 guess only if the argmax before it agrees, so the output is
//!                                 greedy decoding bit for bit; 0 = off (default)
//!   RESET                         drop the acquisition
//!   QUIT
//! Every generated token is appended to the acquisition, so the next GEN with the same transcript
//! plus the reply is a re-slice, not a re-parse.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use atlas::cache::Acquisition;
use atlas::forward::{argmax, Engine};
use atlas::surfaces::Model;
use atlas::turn::forward_turn;

const EOS: [usize; 2] = [151645, 151643]; // <|im_end|>, <|endoftext|> (generation_config.json)

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

/// The transcript's own guess: the tokens that followed the most recent earlier occurrence of the
/// longest n-gram (n = 4..1) ending the transcript. Up to `g` of them.
fn lookup(tokens: &[usize], g: usize) -> Vec<usize> {
    let len = tokens.len();
    for n in (1..=4.min(len.saturating_sub(1))).rev() {
        let pat = &tokens[len - n..];
        // most recent earlier occurrence, not overlapping the final one
        let mut i = len - n;
        while i > 0 {
            i -= 1;
            if &tokens[i..i + n] == pat {
                let start = i + n;
                let end = (start + g).min(len);
                if end > start {
                    return tokens[start..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&a[1]);
    let id = &a[2];
    let intake = find_intake_dir(&root, id);
    let t = Instant::now();
    let model = Model::load(id, &intake, &root.join("cells").join(id)).unwrap();
    let eng = Engine::new(model, &PathBuf::from("D:/folded-weights/v5/tests/oracles/lut")).unwrap();
    let vocab = eng.model.cfg.vocab;
    let mut acq = Acquisition::empty(eng.model.cfg.n_layers);
    let mut lc: Vec<Vec<u16>> = Vec::new();
    let mut look: usize = 0;
    let out = std::io::stdout();
    let mut out = out.lock();
    writeln!(out, "READY model={} layers={} vocab={} kernel={} load_ms={}", id, eng.model.cfg.n_layers, vocab, std::env::var("ATLAS_KERNEL").unwrap_or_else(|_| "bucket".into()), t.elapsed().as_millis()).unwrap();
    out.flush().unwrap();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("GEN") => {
                let max_new: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(64);
                let mut tokens: Vec<usize> = it.map(|s| s.parse().unwrap()).collect();
                let t = Instant::now();
                let (logits, kept) = forward_turn(&eng, &mut acq, &tokens, &mut lc);
                let computed = tokens.len() - kept;
                let mut next = argmax(&logits[(tokens.len() - 1) * vocab..tokens.len() * vocab]);
                let mut generated = 0usize;
                let (mut passes, mut guessed, mut accepted) = (1usize, 0usize, 0usize);
                'gen: while generated < max_new {
                    writeln!(out, "TOK {}", next).unwrap();
                    out.flush().unwrap();
                    generated += 1;
                    tokens.push(next);
                    if EOS.contains(&next) || generated >= max_new {
                        break;
                    }
                    // the oscillator's spare planes: the transcript's own guess of what follows
                    let guesses = if look > 0 { lookup(&tokens, look) } else { Vec::new() };
                    let mut cand = tokens.clone();
                    cand.extend_from_slice(&guesses);
                    let (logits, _) = forward_turn(&eng, &mut acq, &cand, &mut lc);
                    passes += 1;
                    guessed += guesses.len();
                    let pos = tokens.len() - 1;
                    next = argmax(&logits[pos * vocab..(pos + 1) * vocab]);
                    // the comparator: keep a guess while the argmax before it agrees
                    for (gi, &gt) in guesses.iter().enumerate() {
                        if gt != next {
                            break;
                        }
                        writeln!(out, "TOK {}", next).unwrap();
                        out.flush().unwrap();
                        generated += 1;
                        accepted += 1;
                        tokens.push(next);
                        if EOS.contains(&next) || generated >= max_new {
                            break 'gen;
                        }
                        let p = pos + 1 + gi;
                        next = argmax(&logits[p * vocab..(p + 1) * vocab]);
                    }
                    // refused guesses leave the acquisition
                    if acq.len() > tokens.len() {
                        acq.truncate(tokens.len());
                        lc.truncate(tokens.len());
                    }
                }
                if acq.len() > tokens.len() {
                    acq.truncate(tokens.len());
                    lc.truncate(tokens.len());
                }
                let total_ms = t.elapsed().as_millis();
                let tr = atlas::card::trace_take();
                if tr[5] > 0 {
                    let ms = |i: usize| tr[i] / 1_000_000;
                    writeln!(out, "TRACE card calls={} pack={}ms upload+submit={}ms gpu_wait={}ms collapse={}ms cpu_rest={}ms", tr[5], ms(0), ms(1), ms(2), ms(4), total_ms as u64 - (ms(0) + ms(1) + ms(2) + ms(4))).unwrap();
                }
                writeln!(out, "END picked_up={} computed={} generated={} positions={} passes={} guessed={} accepted={} ms={}", kept, computed, generated, acq.len(), passes, guessed, accepted, total_ms).unwrap();
                out.flush().unwrap();
            }
            Some("LOOK") => {
                look = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                writeln!(out, "OK lookahead={}", look).unwrap();
                out.flush().unwrap();
            }
            Some("RESET") => {
                acq = Acquisition::empty(eng.model.cfg.n_layers);
                lc.clear();
                writeln!(out, "OK").unwrap();
                out.flush().unwrap();
            }
            Some("QUIT") | None => break,
            Some(other) => {
                writeln!(out, "ERR unknown command {}", other).unwrap();
                out.flush().unwrap();
            }
        }
    }
}
