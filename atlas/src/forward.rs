//! The forward (spec §4): the five ops composed as the trained model composes them,
//! dims from the config, weights through the surfaces, never through a dense matrix.
//!
//! matmul  — this crate's bucket (`matmul::matmul_grid`, `matmul_dense`).
//! RMSNorm — V5 `RmsNorm` (rsqrt bijective LUT), γ from the norm surface.
//! RoPE    — V5 `GeometricRopeIntLut` (Q.30 cos/sin LUT, proven bijection).
//! softmax — V5 `ExpLutSoftmax` (exp bijective LUT; the divide is V5's f3 divide for now —
//!           the deferred-divide bucket is the next move, gate B4).
//! probs·V — the exact bucket (`matmul_dense`), one collapse per (position, d): a reduction
//!           at the gather, not a walk (since 2026-09-13 ~09:45; references re-sealed as v2).
//! SiLU    — exp LUT + one divide, as V5 (copied: V5's is private).
//! attention — left as-is.
//!
//! Residual adds go through the bucket (`matmul::add`), one collapse each, as V5's `add_into`.

use crate::matmul::{add, matmul_dense, matmul_grid};
use crate::surfaces::Model;
use crate::v5::bf16::{bf16_mul, f3_add, f3_div_to_bf16, single_bits_to_f3};
use crate::v5::probe::{RopeVariant, SoftmaxVariant};
use crate::v5::rmsnorm::RmsNorm;
use crate::v5::rope::GeometricRopeIntLut;
use crate::v5::softmax::ExpLutSoftmax;

/// bf16 pattern of 1e-6 (rms_norm_eps): (1 + 6/128)·2^-20. Verified in cells::tests.
pub const RMS_EPS: u16 = 0x3586;
/// bf16 pattern of 1/sqrt(128) (attention scale, head_dim 128): (1 + 53/128)·2^-4.
pub const ATTN_SCALE_HD128: u16 = 0x3DB5;
const ONE_SINGLE_BITS: u32 = 0x3F80_0000;

pub struct Engine {
    pub model: Model,
    pub rope: GeometricRopeIntLut,
    pub softmax: ExpLutSoftmax,
    pub rmsnorm: RmsNorm,
    pub rms_eps: u16,
    pub attn_scale: u16,
    /// Stage three: the whole pass on the card (`ATLAS_KERNEL=card3`).
    pub card3: Option<std::sync::Mutex<crate::card_layer::CardEngine>>,
}

impl Engine {
    pub fn new(model: Model, lut_dir: &std::path::Path) -> std::io::Result<Engine> {
        assert_eq!(model.cfg.head_dim, 128, "the RoPE phase table and the attention scale are for head_dim 128");
        let cs = std::fs::read(lut_dir.join("cossin_q30_n65536.bin"))?;
        let ps = std::fs::read(lut_dir.join("phase_step_qwen3.bin"))?;
        let rope = GeometricRopeIntLut {
            cossin_lut: GeometricRopeIntLut::decode_cossin_q30(&cs),
            phase_step: GeometricRopeIntLut::decode_phase_step(&ps),
            head_dim: model.cfg.head_dim,
        };
        let e = std::fs::read(lut_dir.join("exp_n32768.bin"))?;
        let softmax = ExpLutSoftmax { exp_lut: ExpLutSoftmax::decode_exp_lut(&e) };
        let r = std::fs::read(lut_dir.join("rsqrt_n32768.bin"))?;
        let rmsnorm = RmsNorm { rsqrt_lut: RmsNorm::decode_rsqrt_lut(&r) };
        let mut card3 = None;
        if std::env::var("ATLAS_KERNEL").map(|v| v == "card3").unwrap_or(false) {
            let t = std::time::Instant::now();
            let ce = crate::card_layer::CardEngine::new(&model, lut_dir, RMS_EPS, ATTN_SCALE_HD128).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            eprintln!("card3: {} — {} MB resident, whole pass on the card, {} ms", ce.adapter_name, ce.resident_bytes / 1_000_000, t.elapsed().as_millis());
            card3 = Some(std::sync::Mutex::new(ce));
        }
        Ok(Engine { model, rope, softmax, rmsnorm, rms_eps: RMS_EPS, attn_scale: ATTN_SCALE_HD128, card3 })
    }
}

fn silu(x: u16, exp_lut: &[u32]) -> u16 {
    let mag = (x & 0x7FFF) as usize;
    let e = single_bits_to_f3(exp_lut[mag]);
    let one = single_bits_to_f3(ONE_SINGLE_BITS);
    let denom = f3_add(one, e);
    let sigma = if (x & 0x8000) == 0 { f3_div_to_bf16(one, denom) } else { f3_div_to_bf16(e, denom) };
    bf16_mul(x, sigma)
}

/// Every weight matmul goes through here: the bucket over the grid, or the radix kernel over the fold.
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
        matmul_grid(x, seq, in_f, m.grid(name), &m.atoms)
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

pub fn rmsnorm_rows(eng: &Engine, x: &[u16], rows: usize, dim: usize, gamma: &[u16]) -> Vec<u16> {
    assert_eq!(x.len(), rows * dim);
    let mut out = Vec::with_capacity(x.len());
    for r in 0..rows {
        out.extend(eng.rmsnorm.apply(&x[r * dim..(r + 1) * dim], gamma, eng.rms_eps));
    }
    out
}

pub fn attention(eng: &Engine, h: &[u16], seq: usize, layer: usize) -> Vec<u16> {
    let c = &eng.model.cfg;
    let (hd, qd, kvd) = (c.head_dim, c.q_dim(), c.kv_dim());
    let group = c.n_q_heads / c.n_kv_heads;
    let m = &eng.model;
    let (qn, kn, vn) = (lname(layer, "self_attn.q_proj"), lname(layer, "self_attn.k_proj"), lname(layer, "self_attn.v_proj"));
        let mut qkv = wmat_many(m, h, seq, c.hidden, &[&qn, &kn, &vn]);
        let v = qkv.pop().unwrap();
        let mut k = qkv.pop().unwrap();
        let mut q = qkv.pop().unwrap();
    let q_g = m.norm(&lname(layer, "self_attn.q_norm"));
    let k_g = m.norm(&lname(layer, "self_attn.k_norm"));
    for s in 0..seq {
        for qh in 0..c.n_q_heads {
            let off = s * qd + qh * hd;
            let normed = eng.rmsnorm.apply(&q[off..off + hd], q_g, eng.rms_eps);
            let roped = eng.rope.apply(&normed, s as u64);
            q[off..off + hd].copy_from_slice(&roped);
        }
        for kh in 0..c.n_kv_heads {
            let off = s * kvd + kh * hd;
            let normed = eng.rmsnorm.apply(&k[off..off + hd], k_g, eng.rms_eps);
            let roped = eng.rope.apply(&normed, s as u64);
            k[off..off + hd].copy_from_slice(&roped);
        }
    }
    let mut attn = vec![0u16; seq * qd];
    for qh in 0..c.n_q_heads {
        let kvh = qh / group;
        let mut qm = vec![0u16; seq * hd];
        let mut km = vec![0u16; seq * hd];
        let mut vm = vec![0u16; seq * hd];
        for s in 0..seq {
            let qo = s * qd + qh * hd;
            let ko = s * kvd + kvh * hd;
            qm[s * hd..(s + 1) * hd].copy_from_slice(&q[qo..qo + hd]);
            km[s * hd..(s + 1) * hd].copy_from_slice(&k[ko..ko + hd]);
            vm[s * hd..(s + 1) * hd].copy_from_slice(&v[ko..ko + hd]);
        }
        let scores = matmul_dense(&qm, &km, seq, hd, seq);
        // probs · V as an exact reduction at the gather: one bucket per (position, d), one collapse
        let mut probs_all = vec![0u16; seq * seq];
        for i in 0..seq {
            let row: Vec<u16> = (0..seq).map(|j| bf16_mul(scores[i * seq + j], eng.attn_scale)).collect();
            let probs = eng.softmax.apply(&row, i);
            probs_all[i * seq..(i + 1) * seq].copy_from_slice(&probs);
        }
        let mut vt = vec![0u16; hd * seq];
        for j in 0..seq {
            for d in 0..hd {
                vt[d * seq + j] = vm[j * hd + d];
            }
        }
        let out = matmul_dense(&probs_all, &vt, seq, seq, hd);
        for i in 0..seq {
            attn[i * qd + qh * hd..i * qd + (qh + 1) * hd].copy_from_slice(&out[i * hd..(i + 1) * hd]);
        }
    }
    wmat(m, &attn, seq, qd, &lname(layer, "self_attn.o_proj"))
}

pub fn mlp(eng: &Engine, h: &[u16], seq: usize, layer: usize) -> Vec<u16> {
    let c = &eng.model.cfg;
    let m = &eng.model;
    let (gn, un) = (lname(layer, "mlp.gate_proj"), lname(layer, "mlp.up_proj"));
        let mut gu = wmat_many(m, h, seq, c.hidden, &[&gn, &un]);
        let u = gu.pop().unwrap();
        let g = gu.pop().unwrap();
    let act: Vec<u16> = g.iter().zip(u.iter()).map(|(&gi, &ui)| bf16_mul(silu(gi, &eng.softmax.exp_lut), ui)).collect();
    wmat(m, &act, seq, c.intermediate, &lname(layer, "mlp.down_proj"))
}

pub fn decoder_layer(eng: &Engine, x: Vec<u16>, seq: usize, layer: usize) -> Vec<u16> {
    let c = &eng.model.cfg;
    let h = rmsnorm_rows(eng, &x, seq, c.hidden, eng.model.norm(&lname(layer, "input_layernorm")));
    let x = add(&x, &attention(eng, &h, seq, layer));
    let h2 = rmsnorm_rows(eng, &x, seq, c.hidden, eng.model.norm(&lname(layer, "post_attention_layernorm")));
    let out = mlp(eng, &h2, seq, layer);
    add(&x, &out)
}

/// Per-position logits `[seq, vocab]`; `progress(layer)` after each layer.
pub fn forward_all(eng: &Engine, tokens: &[usize], progress: &mut dyn FnMut(usize)) -> Vec<u16> {
    let c = &eng.model.cfg;
    let seq = tokens.len();
    assert!(seq > 0);
    let mut x = Vec::with_capacity(seq * c.hidden);
    for &t in tokens {
        x.extend(eng.model.embed_row(t));
    }
    if let Some(ce) = &eng.card3 {
        let mut ce = ce.lock().unwrap();
        ce.reset();
        let out = ce.forward_positions(&x, 0, seq);
        progress(c.n_layers - 1);
        return out;
    }
    for l in 0..c.n_layers {
        x = decoder_layer(eng, x, seq, l);
        progress(l);
    }
    let h = rmsnorm_rows(eng, &x, seq, c.hidden, eng.model.norm("model.norm.weight"));
    assert!(c.tied, "untied lm_head is not a surface this engine has");
    wmat(&eng.model, &h, seq, c.hidden, "model.embed_tokens.weight")
}

/// bf16 ordering on patterns: true if a < b as signed magnitudes.
fn lt(a: u16, b: u16) -> bool {
    let key = |p: u16| -> i32 {
        let mag = (p & 0x7FFF) as i32;
        if p & 0x8000 != 0 { -mag } else { mag }
    };
    key(a) < key(b)
}

pub fn argmax(logits: &[u16]) -> usize {
    let mut best = 0;
    for (i, &p) in logits.iter().enumerate() {
        if lt(logits[best], p) {
            best = i;
        }
    }
    best
}

pub fn topk(logits: &[u16], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        if lt(logits[a], logits[b]) {
            std::cmp::Ordering::Greater
        } else if lt(logits[b], logits[a]) {
            std::cmp::Ordering::Less
        } else {
            a.cmp(&b)
        }
    });
    idx.truncate(k);
    idx
}
