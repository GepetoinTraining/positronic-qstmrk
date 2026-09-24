// T10 — attention as a flow. Registration: ../../prereg_T10.txt (sha256 11741dad…9b27b9b), written before this ran.
// Instrument: f64 on exact bf16 inputs (a ruler). Tier A reads v1's sealed captures; Tier B adds layer-14 tensors
// fetched by byte range and admitted only against v1's sealed source_sha256 (../t10/fetch.py).
// Run from v3-based/walk:  cargo run --release --bin t10

use std::fs;

const SEQ: usize = 508;
const HID: usize = 2048;
const HD: usize = 128;
const NQ: usize = 16;
const NKV: usize = 8;
const EPS: f64 = 1e-6;
const THETA: f64 = 1_000_000.0;

fn bf16(p: u16) -> f64 {
    f32::from_bits((p as u32) << 16) as f64
}
fn load(path: &str) -> Vec<f64> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks(2).map(|c| bf16(u16::from_le_bytes([c[0], c[1]]))).collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ======================= Tier A
/// mean pairwise cosine over positions 1..SEQ
fn mean_cosine(x: &[f64]) -> f64 {
    let unit: Vec<Vec<f64>> = (1..SEQ)
        .map(|i| {
            let v = &x[i * HID..(i + 1) * HID];
            let n = dot(v, v).sqrt();
            v.iter().map(|a| a / n).collect()
        })
        .collect();
    let (mut s, mut n) = (0.0, 0usize);
    for i in 0..unit.len() {
        for j in i + 1..unit.len() {
            s += dot(&unit[i], &unit[j]);
            n += 1;
        }
    }
    s / n as f64
}
/// participation ratio (tr G)^2 / ||G||_F^2, G the Gram matrix of positions 1..SEQ
fn participation(x: &[f64]) -> f64 {
    let rows: Vec<&[f64]> = (1..SEQ).map(|i| &x[i * HID..(i + 1) * HID]).collect();
    let (mut tr, mut fro) = (0.0, 0.0);
    for i in 0..rows.len() {
        for j in i..rows.len() {
            let g = dot(rows[i], rows[j]);
            if i == j {
                tr += g;
                fro += g * g;
            } else {
                fro += 2.0 * g * g;
            }
        }
    }
    tr * tr / fro
}
fn sink_share(x: &[f64]) -> f64 {
    let total: f64 = x.iter().map(|a| a * a).sum();
    let s0: f64 = x[..HID].iter().map(|a| a * a).sum();
    s0 / total
}

// ======================= Tier B
fn rmsnorm(v: &[f64], g: &[f64]) -> Vec<f64> {
    let ms = v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64;
    let r = 1.0 / (ms + EPS).sqrt();
    v.iter().zip(g).map(|(a, w)| a * r * w).collect()
}
/// y[s][o] = sum_i W[o][i] x[s][i], rows split over threads
fn project(x: &[f64], w: &[f64], out: usize) -> Vec<f64> {
    let mut y = vec![0.0; SEQ * out];
    let threads = 8;
    let chunk = SEQ.div_ceil(threads);
    std::thread::scope(|sc| {
        for (t, ys) in y.chunks_mut(chunk * out).enumerate() {
            sc.spawn(move || {
                for (r, yr) in ys.chunks_mut(out).enumerate() {
                    let s = t * chunk + r;
                    let xs = &x[s * HID..(s + 1) * HID];
                    for (o, yo) in yr.iter_mut().enumerate() {
                        *yo = dot(&w[o * HID..(o + 1) * HID], xs);
                    }
                }
            });
        }
    });
    y
}
/// Qwen3 rotate_half RoPE on one head vector at position pos
fn rope(v: &mut [f64], pos: usize) {
    for d in 0..HD / 2 {
        let inv = THETA.powf(-(2.0 * d as f64) / HD as f64);
        let (s, c) = (pos as f64 * inv).sin_cos();
        let (a, b) = (v[d], v[d + HD / 2]);
        v[d] = a * c - b * s;
        v[d + HD / 2] = b * c + a * s;
    }
}

struct HeadStats {
    sink: f64,
    select: f64,
    below: f64,
    offset: usize,
    offset_share: f64,
}

/// per-head attention statistics on one capture (layer 14 input)
fn attention_stats(x: &[f64], g_in: &[f64], wq: &[f64], wk: &[f64], gq: &[f64], gk: &[f64]) -> Vec<HeadStats> {
    let xn: Vec<f64> = (0..SEQ).flat_map(|s| rmsnorm(&x[s * HID..(s + 1) * HID], g_in)).collect();
    let q = project(&xn, wq, NQ * HD);
    let k = project(&xn, wk, NKV * HD);
    let head = |m: &[f64], s: usize, h: usize, width: usize, g: &[f64]| -> Vec<f64> {
        let mut v = rmsnorm(&m[s * width + h * HD..s * width + (h + 1) * HD], g);
        rope(&mut v, s);
        v
    };
    let scale = 1.0 / (HD as f64).sqrt();
    (0..NQ)
        .map(|h| {
            let kv = h / (NQ / NKV);
            let qs: Vec<Vec<f64>> = (0..SEQ).map(|s| head(&q, s, h, NQ * HD, gq)).collect();
            let ks: Vec<Vec<f64>> = (0..SEQ).map(|s| head(&k, s, kv, NKV * HD, gk)).collect();
            let (mut sink, mut select, mut below) = (0.0, 0.0, 0.0);
            let mut offsets = vec![0usize; SEQ];
            let rows = SEQ - 64;
            for i in 64..SEQ {
                let sc: Vec<f64> = (0..=i).map(|j| dot(&qs[i], &ks[j]) * scale).collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let p: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                let z: f64 = p.iter().sum();
                sink += p[0] / z;
                let mut sorted: Vec<f64> = p.iter().map(|a| a / z).collect();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let target = 1.0 - (2f64).powi(-10);
                let (mut acc, mut n) = (0.0, 0usize);
                for w in &sorted {
                    acc += w;
                    n += 1;
                    if acc >= target {
                        break;
                    }
                }
                select += n as f64 / (i + 1) as f64;
                // below 2^-24 of the row's largest weight: positions an exact collector could skip with a bound
                below += p.iter().filter(|&&w| w < (2f64).powi(-24)).count() as f64 / (i + 1) as f64;
                let jmax = (1..=i).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()).unwrap();
                offsets[i - jmax] += 1;
            }
            let (offset, cnt) = offsets.iter().enumerate().max_by_key(|(_, c)| **c).map(|(o, c)| (o, *c)).unwrap();
            HeadStats { sink: sink / rows as f64, select: select / rows as f64, below: below / rows as f64, offset, offset_share: cnt as f64 / rows as f64 }
        })
        .collect()
}

/// a = ||A||^2/||B||^2 for B = U^T V (U, V: 128 x 2048), via 128 x 128 products only
fn antisym_share(u: &[f64], v: &[f64]) -> f64 {
    let n = HD;
    fn row(m: &[f64], a: usize) -> &[f64] {
        &m[a * HID..(a + 1) * HID]
    }
    let mut b2 = 0.0; // ||B||^2 = sum_ab (U U^T)_ab (V V^T)_ab
    let mut tr = 0.0; // tr(B^2) = sum_ab C_ab C_ba, C = V U^T
    let mut c = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            c[a * n + b] = dot(row(v, a), row(u, b));
        }
    }
    for a in 0..n {
        for b in 0..n {
            b2 += dot(row(u, a), row(u, b)) * dot(row(v, a), row(v, b));
            tr += c[a * n + b] * c[b * n + a];
        }
    }
    (b2 - tr) / 2.0 / b2
}

fn main() {
    let cap = |layer: usize, tag: &str| load(&format!("../../receipts/capture-qwen3-1.7b-L{layer}-508tok{tag}.u16"));
    let prompts = [("ramp", ""), ("random", "-random")];

    println!("T10 — registration sha256 11741dad72e7312ee443f63d550088fe679034461c225b4d38991c6c9b427b9b\n");
    println!("Tier A — the fluid (captures; positions 1..507 unless stated)");
    println!("{:<7} {:>5} {:>12} {:>24} {:>14} {:>12}", "prompt", "layer", "mean cos", "control band (10 shuf)", "particip.", "sink share");
    let mut a_rows = Vec::new();
    for (name, tag) in prompts {
        for layer in [0usize, 1, 14] {
            let x = cap(layer, tag);
            let cos = mean_cosine(&x);
            let mut rng = Rng(20260924 + layer as u64);
            let ctrl: Vec<f64> = (0..10)
                .map(|_| {
                    let mut y = x.clone();
                    for s in 0..SEQ {
                        rng.shuffle(&mut y[s * HID..(s + 1) * HID]);
                    }
                    mean_cosine(&y)
                })
                .collect();
            let (lo, hi) = ctrl.iter().fold((f64::MAX, f64::MIN), |(l, h), &c| (l.min(c), h.max(c)));
            let pr = participation(&x);
            let sk = sink_share(&x);
            println!("{name:<7} {layer:>5} {cos:>12.5} {:>24} {pr:>14.3} {sk:>12.5}", format!("[{lo:.5}, {hi:.5}]"));
            a_rows.push((name, layer, cos, hi, pr, sk));
        }
    }
    let series = |n: &str, f: &dyn Fn(&(&str, usize, f64, f64, f64, f64)) -> f64| -> Vec<f64> {
        a_rows.iter().filter(|r| r.0 == n).map(|r| f(r)).collect()
    };
    let rising = |v: &[f64]| v.windows(2).all(|w| w[1] > w[0]);
    let falling = |v: &[f64]| v.windows(2).all(|w| w[1] < w[0]);
    let l14 = |n: &str| a_rows.iter().find(|r| r.0 == n && r.1 == 14).unwrap();
    let a1 = rising(&series("ramp", &|r| r.2)) && rising(&series("random", &|r| r.2)) && l14("random").2 > l14("random").3 && l14("ramp").2 > l14("ramp").3;
    let a1_kill = !rising(&series("random", &|r| r.2)) || l14("random").2 <= l14("random").3;
    let a2 = falling(&series("ramp", &|r| r.4)) && falling(&series("random", &|r| r.4));
    let a2_kill = !falling(&series("random", &|r| r.4));
    let a3 = rising(&series("ramp", &|r| r.5)) && rising(&series("random", &|r| r.5));
    println!("\nA1 clustering: {}", if a1 { "HELD" } else if a1_kill { "KILLED" } else { "not held on both prompts, kill clause not met" });
    println!("A2 effective rank: {}", if a2 { "HELD" } else if a2_kill { "KILLED" } else { "not held on both prompts, kill clause not met" });
    println!("A3 sink: {}", if a3 { "HELD" } else { "KILLED" });

    // ---------------- Tier B
    let w = |n: &str| load(&format!("../t10/model.layers.14.{n}.weight.u16"));
    let (g_in, wq, wk, gq, gk) = (w("input_layernorm"), w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.q_norm"), w("self_attn.k_norm"));
    println!("\nTier B — layer 14 (tensors admitted against v1's seals)");
    let uv = |wm: &[f64], h: usize, g: &[f64], perm: Option<&[usize]>| -> Vec<f64> {
        let mut m = vec![0.0; HD * HID];
        for a in 0..HD {
            for i in 0..HID {
                let src = perm.map_or(i, |p| p[i]);
                m[a * HID + i] = g[a] * wm[(h * HD + a) * HID + src];
            }
        }
        m
    };
    let mut rng = Rng(20260924);
    let perms: Vec<Vec<usize>> = (0..10)
        .map(|_| {
            let mut p: Vec<usize> = (0..HID).collect();
            rng.shuffle(&mut p);
            p
        })
        .collect();
    let stats_rand = attention_stats(&cap(14, "-random"), &g_in, &wq, &wk, &gq, &gk);
    let stats_ramp = attention_stats(&cap(14, ""), &g_in, &wq, &wk, &gq, &gk);
    println!("{:>4} {:>7} {:>20} {:>6} | random prompt: {:>7} {:>8} {:>10} {:>14} | ramp: {:>7} {:>8}", "head", "a", "control band", "out?", "sink", "select", "<2^-24", "top offset", "sink", "select");
    let (mut b1_out, mut b2_n, mut b3_n) = (0, 0, 0);
    for h in 0..NQ {
        let u = uv(&wq, h, &gq, None);
        let v = uv(&wk, h / (NQ / NKV), &gk, None);
        let a = antisym_share(&u, &v);
        let ctrl: Vec<f64> = perms.iter().map(|p| antisym_share(&u, &uv(&wk, h / (NQ / NKV), &gk, Some(p)))).collect();
        let (lo, hi) = ctrl.iter().fold((f64::MAX, f64::MIN), |(l, x), &c| (l.min(c), x.max(c)));
        let out = a < lo || a > hi;
        b1_out += out as usize;
        let (r, m) = (&stats_rand[h], &stats_ramp[h]);
        b2_n += (r.sink >= 0.5) as usize;
        b3_n += (r.select < 0.25) as usize;
        println!(
            "{h:>4} {a:>7.4} {:>20} {:>6} | {:>21.3} {:>8.3} {:>10.3} {:>8} ({:>4.0}%) | {:>13.3} {:>8.3}",
            format!("[{lo:.4}, {hi:.4}]"),
            if out { if a > hi { "A-side" } else { "S-side" } } else { "in" },
            r.sink, r.select, r.below, r.offset, 100.0 * r.offset_share, m.sink, m.select
        );
    }
    let verdict = |n: usize, need: usize, kill: usize| if n >= need { "HELD" } else if n < kill { "KILLED" } else { "not held, kill clause not met" };
    println!("\nB1 fronts: {b1_out}/16 heads outside the control band — {}", verdict(b1_out, 8, 4));
    println!("B2 sink: {b2_n}/16 heads send >= 0.5 of their mass to position 0 — {}", verdict(b2_n, 8, 4));
    println!("B3 selection: {b3_n}/16 heads keep 1 - 2^-10 of the mass in < 25% of the context — {}", verdict(b3_n, 8, 4));
}
