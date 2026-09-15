//! `turn.rs` — a chat turn over the acquisition (handshake-paper.pdf; cache.rs).
//!
//! `forward_turn(eng, acq, tokens)`: the checker finds how many leading positions of `tokens`
//! are information-contained in the acquisition (the sealed prefix). Those are picked up,
//! never re-timed. The remaining positions are computed exactly as `forward::forward_all`
//! computes them — the same ops in the same order per output cell, attending over the
//! picked-up k and v of the old positions — so the logits of every position are bit-identical
//! to a full re-parse (gate: the seq-8 reference reproduced from the [1] acquisition).
//! The acquisition is then extended with the new positions and their addresses.

use crate::cache::{Acquisition, Slice};
use crate::forward::{rmsnorm_rows, Engine};
use crate::matmul::{add, matmul_dense};
use crate::surfaces::Model;
use crate::v5::bf16::{bf16_mul, f3_add, f3_div_to_bf16, single_bits_to_f3};
use crate::v5::probe::{RopeVariant, SoftmaxVariant};

const ONE_SINGLE_BITS: u32 = 0x3F80_0000;

fn silu(x: u16, exp_lut: &[u32]) -> u16 {
    let mag = (x & 0x7FFF) as usize;
    let e = single_bits_to_f3(exp_lut[mag]);
    let one = single_bits_to_f3(ONE_SINGLE_BITS);
    let denom = f3_add(one, e);
    let sigma = if (x & 0x8000) == 0 { f3_div_to_bf16(one, denom) } else { f3_div_to_bf16(e, denom) };
    bf16_mul(x, sigma)
}

fn wmat(m: &Model, x: &[u16], seq: usize, in_f: usize, name: &str) -> Vec<u16> {
    if let Some(c) = &m.card {
        c.matmul(x, seq, in_f, name)
    } else if let Some(c) = &m.cub {
        crate::cub::matmul_cub(x, seq, in_f, c, name)
    } else if m.radix() {
        crate::fold::matmul_fold(x, seq, in_f, m.fold(name))
    } else if let Some(f) = &m.fiber {
        crate::fiberkernel::matmul_positions(x, seq, in_f, m.grid(name), f)
    } else {
        crate::matmul::matmul_grid(x, seq, in_f, m.grid(name), &m.atoms)
    }
}

/// Several projections over one activation: on the card one upload, one submit, one wait.
fn wmat_many(m: &Model, x: &[u16], seq: usize, in_f: usize, names: &[&str]) -> Vec<Vec<u16>> {
    if let Some(c) = &m.card {
        c.matmul_many(x, seq, in_f, names)
    } else {
        names.iter().map(|n| wmat(m, x, seq, in_f, n)).collect()
    }
}

fn lname(layer: usize, sub: &str) -> String {
    format!("model.layers.{}.{}.weight", layer, sub)
}

/// Attention for the new positions p0..p0+n over the old k, v (picked up) plus the new.
/// Records the new positions' (x_in, k, v) into the acquisition for this layer.
fn attention_turn(eng: &Engine, acq: &mut Acquisition, x_in: &[u16], h: &[u16], p0: usize, n: usize, layer: usize) -> Vec<u16> {
    let c = &eng.model.cfg;
    let (hd, qd, kvd) = (c.head_dim, c.q_dim(), c.kv_dim());
    let group = c.n_q_heads / c.n_kv_heads;
    let m = &eng.model;
    let (qn, kn, vn) = (lname(layer, "self_attn.q_proj"), lname(layer, "self_attn.k_proj"), lname(layer, "self_attn.v_proj"));
        let mut qkv = wmat_many(m, h, n, c.hidden, &[&qn, &kn, &vn]);
        let v = qkv.pop().unwrap();
        let mut k = qkv.pop().unwrap();
        let mut q = qkv.pop().unwrap();
    let q_g = m.norm(&lname(layer, "self_attn.q_norm"));
    let k_g = m.norm(&lname(layer, "self_attn.k_norm"));
    for s in 0..n {
        let pos = (p0 + s) as u64;
        for qh in 0..c.n_q_heads {
            let off = s * qd + qh * hd;
            let normed = eng.rmsnorm.apply(&q[off..off + hd], q_g, eng.rms_eps);
            let roped = eng.rope.apply(&normed, pos);
            q[off..off + hd].copy_from_slice(&roped);
        }
        for kh in 0..c.n_kv_heads {
            let off = s * kvd + kh * hd;
            let normed = eng.rmsnorm.apply(&k[off..off + hd], k_g, eng.rms_eps);
            let roped = eng.rope.apply(&normed, pos);
            k[off..off + hd].copy_from_slice(&roped);
        }
    }
    // record the new positions (the acquisition is annotated as it is made)
    for s in 0..n {
        acq.record(layer, p0 + s, Slice { x_in: x_in[s * c.hidden..(s + 1) * c.hidden].to_vec(), k: k[s * kvd..(s + 1) * kvd].to_vec(), v: v[s * kvd..(s + 1) * kvd].to_vec() });
    }
    let total = p0 + n;
    let mut attn = vec![0u16; n * qd];
    for qh in 0..c.n_q_heads {
        let kvh = qh / group;
        let mut qm = vec![0u16; n * hd];
        let mut km = vec![0u16; total * hd];
        let mut vm = vec![0u16; total * hd];
        for s in 0..n {
            qm[s * hd..(s + 1) * hd].copy_from_slice(&q[s * qd + qh * hd..s * qd + (qh + 1) * hd]);
        }
        for j in 0..total {
            let sl = &acq.slices[layer][j];
            km[j * hd..(j + 1) * hd].copy_from_slice(&sl.k[kvh * hd..(kvh + 1) * hd]);
            vm[j * hd..(j + 1) * hd].copy_from_slice(&sl.v[kvh * hd..(kvh + 1) * hd]);
        }
        let scores = matmul_dense(&qm, &km, n, hd, total);
        let mut probs_all = vec![0u16; n * total];
        for s in 0..n {
            let i = p0 + s;
            let row: Vec<u16> = (0..total).map(|j| bf16_mul(scores[s * total + j], eng.attn_scale)).collect();
            let probs = eng.softmax.apply(&row, i);
            probs_all[s * total..(s + 1) * total].copy_from_slice(&probs);
        }
        let mut vt = vec![0u16; hd * total];
        for j in 0..total {
            for d in 0..hd {
                vt[d * total + j] = vm[j * hd + d];
            }
        }
        let out = matmul_dense(&probs_all, &vt, n, total, hd);
        for s in 0..n {
            attn[s * qd + qh * hd..s * qd + (qh + 1) * hd].copy_from_slice(&out[s * hd..(s + 1) * hd]);
        }
    }
    wmat(m, &attn, n, qd, &lname(layer, "self_attn.o_proj"))
}

fn mlp(eng: &Engine, h: &[u16], n: usize, layer: usize) -> Vec<u16> {
    let c = &eng.model.cfg;
    let m = &eng.model;
    let (gn, un) = (lname(layer, "mlp.gate_proj"), lname(layer, "mlp.up_proj"));
        let mut gu = wmat_many(m, h, n, c.hidden, &[&gn, &un]);
        let u = gu.pop().unwrap();
        let g = gu.pop().unwrap();
    let act: Vec<u16> = g.iter().zip(u.iter()).map(|(&gi, &ui)| bf16_mul(silu(gi, &eng.softmax.exp_lut), ui)).collect();
    wmat(m, &act, n, c.intermediate, &lname(layer, "mlp.down_proj"))
}

/// One turn. Returns the logits of every position `[len(tokens), vocab]`, old rows picked up.
/// Also returns how many positions were picked up (0 = fresh acquisition).
pub fn forward_turn(eng: &Engine, acq: &mut Acquisition, tokens: &[usize], logits_cache: &mut Vec<Vec<u16>>) -> (Vec<u16>, usize) {
    let c = &eng.model.cfg;
    let vocab = c.vocab;
    let kept = acq.contained_prefix(tokens);
    if kept < acq.len() {
        acq.truncate(kept);
        logits_cache.truncate(kept);
    }
    let p0 = kept;
    let n = tokens.len() - p0;
    if let Some(ce) = &eng.card3 {
        let mut ce = ce.lock().unwrap();
        ce.truncate(kept);
        if n > 0 {
            for &t in &tokens[p0..] {
                acq.push_position(t);
            }
            let mut x = Vec::with_capacity(n * c.hidden);
            for &t in &tokens[p0..] {
                x.extend(eng.model.embed_row(t));
            }
            let logits = ce.forward_positions(&x, p0, n);
            for s in 0..n {
                logits_cache.push(logits[s * vocab..(s + 1) * vocab].to_vec());
            }
        }
        let mut out = Vec::with_capacity(tokens.len() * vocab);
        for row in logits_cache.iter().take(tokens.len()) {
            out.extend_from_slice(row);
        }
        return (out, kept);
    }
    if n > 0 {
        for &t in &tokens[p0..] {
            acq.push_position(t);
        }
        let mut x = Vec::with_capacity(n * c.hidden);
        for &t in &tokens[p0..] {
            x.extend(eng.model.embed_row(t));
        }
        for l in 0..c.n_layers {
            let h = rmsnorm_rows(eng, &x, n, c.hidden, eng.model.norm(&lname(l, "input_layernorm")));
            let a = attention_turn(eng, acq, &x, &h, p0, n, l);
            let x1 = add(&x, &a);
            let h2 = rmsnorm_rows(eng, &x1, n, c.hidden, eng.model.norm(&lname(l, "post_attention_layernorm")));
            let o = mlp(eng, &h2, n, l);
            x = add(&x1, &o);
        }
        let h = rmsnorm_rows(eng, &x, n, c.hidden, eng.model.norm("model.norm.weight"));
        for s in 0..n {
            acq.final_h.push(h[s * c.hidden..(s + 1) * c.hidden].to_vec());
        }
        let logits = wmat(&eng.model, &h, n, c.hidden, "model.embed_tokens.weight");
        for s in 0..n {
            logits_cache.push(logits[s * vocab..(s + 1) * vocab].to_vec());
        }
    }
    let mut out = Vec::with_capacity(tokens.len() * vocab);
    for row in logits_cache.iter().take(tokens.len()) {
        out.extend_from_slice(row);
    }
    (out, kept)
}
