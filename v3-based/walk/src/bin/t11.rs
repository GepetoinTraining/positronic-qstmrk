// T11 — do heads leave the sink for what matters? Registration: ../../prereg_T11.txt (sha256 19b9c5f0…af1d0a).
// Instrument: f64 Qwen3 forward on exact bf16 inputs, layers streamed from ../t10/*.u16 (each admitted against v1's
// seal by fetch.py). The gate (G) must reproduce v1's sealed captures before anything else is read.
// Run from v3-based/walk:  cargo run --release --bin t11

use std::fs;

const SEQ: usize = 508;
const HID: usize = 2048;
const INTER: usize = 6144;
const HD: usize = 128;
const NQ: usize = 16;
const NKV: usize = 8;
const EPS: f64 = 1e-6;
const THETA: f64 = 1_000_000.0;
const HALF: usize = 254; // REPEAT = first 254 tokens twice; induction target j = i - 253
const F: std::ops::Range<usize> = 64..254;
const S: std::ops::Range<usize> = 318..508;

fn bf16(p: u16) -> f64 {
    f32::from_bits((p as u32) << 16) as f64
}
fn load(path: &str) -> Vec<f64> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks(2).map(|c| bf16(u16::from_le_bytes([c[0], c[1]]))).collect()
}
fn weight(layer: usize, name: &str) -> Vec<f64> {
    load(&format!("../t10/model.layers.{layer}.{name}.weight.u16"))
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn rmsnorm(v: &[f64], g: &[f64]) -> Vec<f64> {
    let ms = v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64;
    let r = 1.0 / (ms + EPS).sqrt();
    v.iter().zip(g).map(|(a, w)| a * r * w).collect()
}
fn rmsnorm_rows(x: &[f64], g: &[f64]) -> Vec<f64> {
    x.chunks(HID).flat_map(|r| rmsnorm(r, g)).collect()
}
/// y[s][o] = sum_i W[o][i] x[s][i]; W is out x inn, rows of x split over threads
fn project(x: &[f64], w: &[f64], inn: usize, out: usize) -> Vec<f64> {
    let mut y = vec![0.0; SEQ * out];
    let chunk = SEQ.div_ceil(8);
    std::thread::scope(|sc| {
        for (t, ys) in y.chunks_mut(chunk * out).enumerate() {
            sc.spawn(move || {
                for (r, yr) in ys.chunks_mut(out).enumerate() {
                    let s = t * chunk + r;
                    let xs = &x[s * inn..(s + 1) * inn];
                    for (o, yo) in yr.iter_mut().enumerate() {
                        *yo = dot(&w[o * inn..(o + 1) * inn], xs);
                    }
                }
            });
        }
    });
    y
}
fn add_into(x: &mut [f64], y: &[f64]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}
fn cos_table() -> Vec<(f64, f64)> {
    (0..SEQ)
        .flat_map(|p| (0..HD / 2).map(move |d| (p as f64 * THETA.powf(-(2.0 * d as f64) / HD as f64)).sin_cos()))
        .collect()
}
fn rope(v: &mut [f64], pos: usize, t: &[(f64, f64)]) {
    for d in 0..HD / 2 {
        let (s, c) = t[pos * (HD / 2) + d];
        let (a, b) = (v[d], v[d + HD / 2]);
        v[d] = a * c - b * s;
        v[d + HD / 2] = b * c + a * s;
    }
}

#[derive(Clone, Default)]
struct Stat {
    sink_f: f64,
    sink_s: f64,
    ind_s: f64,
    top_off_s: usize,
}

/// One layer's attention: per-head stats, and (if v is given) the attention output [SEQ x NQ*HD].
fn attention(xn: &[f64], l: usize, v: Option<&[f64]>, tab: &[(f64, f64)]) -> (Vec<Stat>, Vec<f64>) {
    let (wq, wk) = (weight(l, "self_attn.q_proj"), weight(l, "self_attn.k_proj"));
    let (gq, gk) = (weight(l, "self_attn.q_norm"), weight(l, "self_attn.k_norm"));
    let q = project(xn, &wq, HID, NQ * HD);
    let k = project(xn, &wk, HID, NKV * HD);
    let mut out = vec![0.0; if v.is_some() { SEQ * NQ * HD } else { 0 }];
    let heads = |m: &[f64], width: usize, h: usize, g: &[f64]| -> Vec<Vec<f64>> {
        (0..SEQ)
            .map(|s| {
                let mut x = rmsnorm(&m[s * width + h * HD..s * width + (h + 1) * HD], g);
                rope(&mut x, s, tab);
                x
            })
            .collect()
    };
    let scale = 1.0 / (HD as f64).sqrt();
    let mut stats = vec![Stat::default(); NQ];
    for h in 0..NQ {
        let g = h / (NQ / NKV);
        let (qs, ks) = (heads(&q, NQ * HD, h, &gq), heads(&k, NKV * HD, g, &gk));
        let mut offsets = vec![0usize; SEQ];
        for i in 0..SEQ {
            let sc: Vec<f64> = (0..=i).map(|j| dot(&qs[i], &ks[j]) * scale).collect();
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            let p: Vec<f64> = e.iter().map(|a| a / z).collect();
            if F.contains(&i) {
                stats[h].sink_f += p[0] / F.len() as f64;
            }
            if S.contains(&i) {
                stats[h].sink_s += p[0] / S.len() as f64;
                stats[h].ind_s += p[i - (HALF - 1)] / S.len() as f64;
                let jmax = (1..=i).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()).unwrap();
                offsets[i - jmax] += 1;
            }
            if let Some(v) = v {
                let o = &mut out[i * NQ * HD + h * HD..i * NQ * HD + (h + 1) * HD];
                for (j, pj) in p.iter().enumerate() {
                    let vj = &v[j * NKV * HD + g * HD..j * NKV * HD + (g + 1) * HD];
                    o.iter_mut().zip(vj).for_each(|(a, b)| *a += pj * b);
                }
            }
        }
        stats[h].top_off_s = offsets.iter().enumerate().max_by_key(|(_, c)| **c).unwrap().0;
    }
    (stats, out)
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Run layers 0..=13 (and layer 14's attention stats) on one prompt. Returns per-layer stats and the residual
/// stream entering layers 1 and 14.
fn run(x0: Vec<f64>, tab: &[(f64, f64)], label: &str) -> (Vec<Vec<Stat>>, Vec<f64>, Vec<f64>) {
    let mut x = x0;
    let (mut all, mut x1, mut x14) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..=14 {
        if l == 1 {
            x1 = x.clone();
        }
        let xn = rmsnorm_rows(&x, &weight(l, "input_layernorm"));
        if l == 14 {
            x14 = x.clone();
            all.push(attention(&xn, 14, None, tab).0);
            break;
        }
        let v = project(&xn, &weight(l, "self_attn.v_proj"), HID, NKV * HD);
        let (st, att) = attention(&xn, l, Some(&v), tab);
        all.push(st);
        add_into(&mut x, &project(&att, &weight(l, "self_attn.o_proj"), NQ * HD, HID));
        let h2 = rmsnorm_rows(&x, &weight(l, "post_attention_layernorm"));
        let gate = project(&h2, &weight(l, "mlp.gate_proj"), HID, INTER);
        let up = project(&h2, &weight(l, "mlp.up_proj"), HID, INTER);
        let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        add_into(&mut x, &project(&act, &weight(l, "mlp.down_proj"), INTER, HID));
        eprintln!("  [{label}] layer {l} done");
    }
    (all, x1, x14)
}

fn mean_cos(a: &[f64], b: &[f64]) -> f64 {
    a.chunks(HID).zip(b.chunks(HID)).map(|(u, v)| dot(u, v) / (dot(u, u).sqrt() * dot(v, v).sqrt())).sum::<f64>() / SEQ as f64
}

fn main() {
    let cap = |layer: usize| load(&format!("../../receipts/capture-qwen3-1.7b-L{layer}-508tok-random.u16"));
    let tab = cos_table();
    let emb = cap(0);
    let repeat: Vec<f64> = (0..SEQ).flat_map(|i| emb[(i % HALF) * HID..(i % HALF + 1) * HID].to_vec()).collect();

    println!("T11 — registration sha256 19b9c5f00b7a21f046140b760a30db2376e15e67f475db61292c9c4c06af1d0a\n");
    let (st_rand, x1, x14) = run(emb.clone(), &tab, "random");
    let (g1, g14) = (mean_cos(&x1, &cap(1)), mean_cos(&x14, &cap(14)));
    let gate = g1 >= 0.99 && g14 >= 0.99;
    println!("G  instrument vs v1's sealed captures (RANDOM): mean cosine L1 {g1:.6}, L14 {g14:.6} — {}", if gate { "PASS" } else { "FAIL: T11 void" });
    if !gate {
        return;
    }
    let (st_rep, _, _) = run(repeat, &tab, "repeat");

    println!("\nper layer (REPEAT): heads meeting P1 (sink drop >= 0.2 and induction >= 0.2); strongest induction head");
    let mut p1_heads = Vec::new();
    for l in 0..=14 {
        let best = (0..NQ).max_by(|&a, &b| st_rep[l][a].ind_s.partial_cmp(&st_rep[l][b].ind_s).unwrap()).unwrap();
        let b = &st_rep[l][best];
        let meets: Vec<usize> = (0..NQ).filter(|&h| st_rep[l][h].sink_s <= st_rep[l][h].sink_f - 0.2 && st_rep[l][h].ind_s >= 0.2).collect();
        println!(
            "  L{l:>2}: {:>2} heads {:?} | best h{best:>2}: induction {:.3}, sink F {:.3} -> S {:.3} (RANDOM same rows: induction {:.4}, sink {:.3})",
            meets.len(), meets, b.ind_s, b.sink_f, b.sink_s, st_rand[l][best].ind_s, st_rand[l][best].sink_s
        );
        p1_heads.extend(meets.into_iter().map(|h| (l, h)));
    }
    println!("\nP1 heads (layer, head, sink F -> S, induction S, RANDOM induction S):");
    for &(l, h) in &p1_heads {
        let (r, c) = (&st_rep[l][h], &st_rand[l][h]);
        println!("  L{l:>2} h{h:>2}: sink {:.3} -> {:.3}, induction {:.3} | RANDOM induction {:.4}", r.sink_f, r.sink_s, r.ind_s, c.ind_s);
    }
    let p1 = p1_heads.len() >= 2;
    let p1_kill = p1_heads.is_empty();
    let p2_bad: Vec<_> = p1_heads.iter().filter(|&&(l, h)| st_rand[l][h].ind_s >= 0.05).collect();
    let prev: Vec<(usize, usize)> = [2usize, 4, 8, 9].iter().map(|&h| (h, st_rep[14][h].top_off_s)).collect();
    let lost = prev.iter().filter(|(_, o)| *o != 1).count();
    println!("\nP1 relevance leaves the sink: {} heads — {}", p1_heads.len(), if p1 { "HELD" } else if p1_kill { "KILLED" } else { "not held (1 head), kill clause not met" });
    println!("P2 control (RANDOM mass on j = i - 253 < 0.05 for every P1 head): {}", if p1_heads.is_empty() { "not applicable".to_string() } else if p2_bad.is_empty() { "HELD".into() } else { format!("KILLED by {p2_bad:?}") });
    println!("P3 the lag stays (L14 heads 2, 4, 8, 9 top offset in S): {prev:?} — {}", if lost < 2 { "HELD" } else { "KILLED" });

    let released: f64 = p1_heads.iter().map(|&(l, h)| st_rep[l][h].sink_f - st_rep[l][h].sink_s).sum();
    let landed: f64 = p1_heads.iter().map(|&(l, h)| st_rep[l][h].ind_s).sum();
    println!("\nexploratory: over the P1 heads, sink mass released {released:.3}, induction mass landed {landed:.3}");
}
