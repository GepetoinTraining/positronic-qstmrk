//! `card_layer.rs` — stage three: the whole pass on the card (LOG §29). `ATLAS_KERNEL=card3`.
//!
//! Every activation lives on the card; the acquisition's K and V per layer live on the card; a
//! pass over the new positions is one command stream — per layer: norm, q/k/v projections, the
//! per-head norms, RoPE, the K/V append, scores, softmax, probs·V, o_proj, add, norm, gate/up,
//! SiLU, down, add — then the final norm and the head; one submit, one wait, one readback of the
//! logits. Every kernel is the gated port in `card_ops.rs`, in the CPU path's order, so the
//! logits are the same bytes gate 5 and the 508 already sealed.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::card_ops::Ops;
use crate::matmul::{frac_for, FRAC_DENSE};
use crate::surfaces::{Grid, Model};

/// cells per projection dispatch: bounded so no single dispatch outlives the GPU watchdog
const CHUNK: usize = 1 << 20;
const SLOT: u64 = 256; // uniform slot stride (dynamic offset alignment)

struct GridBufs {
    ids: wgpu::Buffer,
    rank: wgpu::Buffer,
    rows: usize,
    cols: usize,
    tcs: u32,
    place_tile_cols: u32,
    th: u32,
    tw: u32,
    frac: i32,
}

struct Work {
    buf: wgpu::Buffer,
    cap: usize, // in u32
}

pub struct CardEngine {
    pub ops: Ops,
    hidden: usize,
    inter: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    vocab: usize,
    rms_eps: u32,
    attn_scale: u32,
    grids: HashMap<String, GridBufs>,
    atoms: wgpu::Buffer,
    norms: HashMap<String, wgpu::Buffer>,
    rsqrt: wgpu::Buffer,
    exp: wgpu::Buffer,
    cossin: wgpu::Buffer,
    steps: wgpu::Buffer,
    dummy: wgpu::Buffer,
    err: wgpu::Buffer,
    err_read: wgpu::Buffer,
    /// per layer: (K, V) resident, rows of kvd patterns
    kv: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    kv_cap: usize,
    pub kv_len: usize,
    params: wgpu::Buffer,
    params_cap: usize,
    work: HashMap<&'static str, Work>,
    logits_read: Option<(wgpu::Buffer, usize)>,
    pub resident_bytes: usize,
    pub adapter_name: String,
}

fn pack_atom(sheet: u8, m: u8, n: i16) -> u32 {
    ((sheet as u32) << 31) | ((m as u32) << 16) | ((n as i32 + 32768) as u32 & 0xFFFF)
}

/// One dispatch in the stream.
struct Op<'a> {
    kernel: &'static str,
    bufs: [&'a wgpu::Buffer; 6],
    params: [u32; 12],
    groups: (u32, u32),
}

impl CardEngine {
    pub fn new(model: &Model, lut_dir: &std::path::Path, rms_eps: u16, attn_scale: u16) -> Result<CardEngine, String> {
        let ops = Ops::new()?;
        let d = &ops.device;
        let mk_init = |label: &str, v: &[u32]| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(v), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST });
        let a = &model.atoms;
        let packed: Vec<u32> = (0..a.m.len()).map(|i| pack_atom(a.sheet[i], a.m[i], a.n[i])).collect();
        let atoms = mk_init("atoms", &packed);
        let mut resident = packed.len() * 4;
        let mut grids = HashMap::new();
        let mut names: Vec<&String> = model.grids.keys().collect();
        names.sort();
        for name in names {
            let g: &Grid = &model.grids[name];
            let ids = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(name), contents: bytemuck::cast_slice(&g.ids), usage: wgpu::BufferUsages::STORAGE });
            let rank = mk_init("rank", &g.place.rank);
            resident += g.ids.len() * 2 + g.place.rank.len() * 4;
            grids.insert(name.clone(), GridBufs { ids, rank, rows: g.rows, cols: g.cols, tcs: (g.cols / g.tw) as u32, place_tile_cols: g.place.tile_cols as u32, th: g.th as u32, tw: g.tw as u32, frac: frac_for(g.rmin) });
        }
        let mut norms = HashMap::new();
        for (name, v) in &model.norms {
            let u: Vec<u32> = v.iter().map(|&p| p as u32).collect();
            norms.insert(name.clone(), mk_init(name, &u));
        }
        let read_lut = |f: &str| -> Result<Vec<u32>, String> {
            let b = std::fs::read(lut_dir.join(f)).map_err(|e| format!("{}: {}", f, e))?;
            Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let rsqrt = mk_init("rsqrt", &read_lut("rsqrt_n32768.bin")?);
        let exp = mk_init("exp", &read_lut("exp_n32768.bin")?);
        let cossin = mk_init("cossin", &read_lut("cossin_q30_n65536.bin")?);
        let steps64 = crate::v5::rope::GeometricRopeIntLut::decode_phase_step(&std::fs::read(lut_dir.join("phase_step_qwen3.bin")).map_err(|e| e.to_string())?);
        let steps_lo: Vec<u32> = steps64.iter().map(|&s| { assert!(s < (1u64 << 32), "phase step exceeds 32 bits"); s as u32 }).collect();
        let steps = mk_init("steps", &steps_lo);
        let dummy = mk_init("dummy", &[0u32]);
        let err = mk_init("err", &[0u32]);
        let err_read = d.create_buffer(&wgpu::BufferDescriptor { label: Some("err read"), size: 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let c = &model.cfg;
        let params_cap = 4096usize;
        let params = d.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: SLOT * params_cap as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut eng = CardEngine {
            hidden: c.hidden, inter: c.intermediate, layers: c.n_layers, heads: c.n_q_heads, kv_heads: c.n_kv_heads, hd: c.head_dim, vocab: c.vocab,
            rms_eps: rms_eps as u32, attn_scale: attn_scale as u32,
            grids, atoms, norms, rsqrt, exp, cossin, steps, dummy, err, err_read,
            kv: Vec::new(), kv_cap: 0, kv_len: 0, params, params_cap, work: HashMap::new(), logits_read: None,
            resident_bytes: resident, adapter_name: ops.adapter_name.clone(), ops,
        };
        eng.ensure_kv(256);
        Ok(eng)
    }

    fn ensure_kv(&mut self, positions: usize) {
        if positions <= self.kv_cap {
            return;
        }
        let cap = positions.max(self.kv_cap * 2).max(256);
        let kvd = self.kv_heads * self.hd;
        let bytes = (cap * kvd * 4) as u64;
        let mut new = Vec::with_capacity(self.layers);
        let mut enc = self.ops.device.create_command_encoder(&Default::default());
        for l in 0..self.layers {
            let k = self.ops.device.create_buffer(&wgpu::BufferDescriptor { label: Some("K"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let v = self.ops.device.create_buffer(&wgpu::BufferDescriptor { label: Some("V"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            if l < self.kv.len() && self.kv_len > 0 {
                let old = (self.kv_len * kvd * 4) as u64;
                enc.copy_buffer_to_buffer(&self.kv[l].0, 0, &k, 0, old);
                enc.copy_buffer_to_buffer(&self.kv[l].1, 0, &v, 0, old);
            }
            new.push((k, v));
        }
        self.ops.queue.submit(Some(enc.finish()));
        self.kv = new;
        self.kv_cap = cap;
    }

    fn work(&mut self, name: &'static str, len: usize) {
        let need = len.max(1);
        let grow = match self.work.get(name) { Some(w) => w.cap < need, None => true };
        if grow {
            let buf = self.ops.device.create_buffer(&wgpu::BufferDescriptor { label: Some(name), size: (need * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.work.insert(name, Work { buf, cap: need });
        }
    }

    fn w(&self, name: &str) -> &wgpu::Buffer {
        &self.work[name].buf
    }

    fn push_pack<'a>(&'a self, st: &mut Vec<Op<'a>>, xb: &'static str, pb: &'static str, len: usize) {
        st.push(Op { kernel: "k_pack", bufs: [self.w(xb), self.w(pb), &self.dummy, &self.dummy, &self.dummy, &self.dummy], params: [len as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], groups: (((len as u32) + 63) / 64, 1) });
    }

    /// `xb` must be a PACKED buffer (push_pack) — k_proj_wg reads cells, not patterns.
    fn push_proj<'a>(&'a self, st: &mut Vec<Op<'a>>, gname: &str, xb: &'static str, yb: &'static str, seq: usize) {
        let g = &self.grids[gname];
        let cells = seq * g.rows;
        let mut cell0 = 0usize;
        while cell0 < cells {
            let nc = (cells - cell0).min(CHUNK);
            let gx = (nc as u32).min(65_535);
            let gy = ((nc as u32) + 65_534) / 65_535;
            st.push(Op { kernel: "k_proj_wg", bufs: [&g.ids, &g.rank, &self.atoms, self.w(xb), self.w(yb), &self.dummy],
                params: [g.rows as u32, g.cols as u32, seq as u32, 0, g.frac as u32, 0, g.tcs, g.place_tile_cols, g.th, g.tw, cell0 as u32, nc as u32], groups: (gx, gy) });
            cell0 += nc;
        }
    }

    fn push_norm<'a>(&'a self, st: &mut Vec<Op<'a>>, xb: &'static str, gname: &str, yb: &'static str, rows: usize, dim: usize) {
        st.push(Op { kernel: "k_rmsnorm", bufs: [self.w(xb), &self.norms[gname], &self.rsqrt, self.w(yb), &self.dummy, &self.dummy],
            params: [rows as u32, dim as u32, self.rms_eps, 0, 0, 0, 0, 0, 0, 0, 0, 0], groups: (((rows as u32) + 63) / 64, 1) });
    }

    fn push_rope<'a>(&'a self, st: &mut Vec<Op<'a>>, xb: &'static str, yb: &'static str, nheads: usize, n: usize) {
        let hd = self.hd;
        let cells = n * nheads * (hd / 2);
        st.push(Op { kernel: "k_rope", bufs: [self.w(xb), &self.steps, &self.cossin, self.w(yb), self.w("pos"), &self.dummy],
            params: [cells as u32, (nheads * hd) as u32, hd as u32, nheads as u32, 0, 0, 0, 0, 0, 0, 0, 0], groups: (((cells as u32) + 63) / 64, 1) });
    }

    fn push_add<'a>(&'a self, st: &mut Vec<Op<'a>>, ab: &'static str, bb: &'static str, yb: &'static str, len: usize) {
        st.push(Op { kernel: "k_add", bufs: [self.w(ab), self.w(bb), self.w(yb), &self.dummy, &self.dummy, &self.dummy], params: [len as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], groups: (((len as u32) + 63) / 64, 1) });
    }

    /// Drop positions beyond `len` from the resident acquisition (the refused guesses).
    pub fn truncate(&mut self, len: usize) {
        if len < self.kv_len {
            self.kv_len = len;
        }
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    /// The pass: `x` = embedding rows of the `n` new positions p0..p0+n (the acquisition holds 0..p0 on the card).
    /// Returns the logits `[n, vocab]` as patterns and appends the positions' K/V.
    pub fn forward_positions(&mut self, x: &[u16], p0: usize, n: usize) -> Vec<u16> {
        assert_eq!(x.len(), n * self.hidden);
        assert_eq!(p0, self.kv_len, "card acquisition holds {} positions, transcript prefix is {}", self.kv_len, p0);
        let total = p0 + n;
        self.ensure_kv(total);
        let (hidden, inter, heads, kvh, hd) = (self.hidden, self.inter, self.heads, self.kv_heads, self.hd);
        let (qd, kvd) = (heads * hd, kvh * hd);
        for (name, len) in [("x", n * hidden), ("h", n * hidden), ("q", n * qd), ("k", n * kvd), ("v", n * kvd), ("q2", n * qd), ("k2", n * kvd), ("q3", n * qd), ("k3", n * kvd),
                            ("scores", n * heads * total), ("probs", n * heads * total), ("attn", n * qd), ("o", n * hidden), ("x1", n * hidden), ("h2", n * hidden),
                            ("g", n * inter), ("u", n * inter), ("act", n * inter), ("dn", n * hidden), ("pos", n), ("logits", n * self.vocab),
                            ("hp", n * hidden), ("attnp", n * qd), ("h2p", n * hidden), ("actp", n * inter)] {
            self.work(name, len);
        }
        let lbytes = (n * self.vocab * 4) as u64;
        let need_new = match &self.logits_read { Some((_, cap)) => (*cap as u64) < lbytes, None => true };
        if need_new {
            let b = self.ops.device.create_buffer(&wgpu::BufferDescriptor { label: Some("logits read"), size: lbytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.logits_read = Some((b, lbytes as usize));
        }
        let xin: Vec<u32> = x.iter().map(|&p| p as u32).collect();
        self.ops.queue.write_buffer(self.w("x"), 0, bytemuck::cast_slice(&xin));
        let positions: Vec<u32> = (0..n).map(|s| (p0 + s) as u32).collect();
        self.ops.queue.write_buffer(self.w("pos"), 0, bytemuck::cast_slice(&positions));
        self.ops.queue.write_buffer(&self.err, 0, bytemuck::cast_slice(&[0u32]));

        let trace = std::env::var("ATLAS_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        // the stream
        let mut stream: Vec<Op> = Vec::new();
        let groups = |cells: usize| (((cells as u32) + 63) / 64, 1u32);
        let lname = |l: usize, s: &str| format!("model.layers.{}.{}.weight", l, s);
        let d = &self.dummy;
        let group = (heads / kvh) as u32;
        for l in 0..self.layers {
            self.push_norm(&mut stream, "x", &lname(l, "input_layernorm"), "h", n, hidden);
            self.push_pack(&mut stream, "h", "hp", n * hidden);
            self.push_proj(&mut stream, &lname(l, "self_attn.q_proj"), "hp", "q", n);
            self.push_proj(&mut stream, &lname(l, "self_attn.k_proj"), "hp", "k", n);
            self.push_proj(&mut stream, &lname(l, "self_attn.v_proj"), "hp", "v", n);
            self.push_norm(&mut stream, "q", &lname(l, "self_attn.q_norm"), "q2", n * heads, hd);
            self.push_norm(&mut stream, "k", &lname(l, "self_attn.k_norm"), "k2", n * kvh, hd);
            self.push_rope(&mut stream, "q2", "q3", heads, n);
            self.push_rope(&mut stream, "k2", "k3", kvh, n);
            // K/V append is a copy, encoded at the same point of the stream (see below)
            stream.push(Op { kernel: "copy_kv", bufs: [self.w("k3"), &self.kv[l].0, self.w("v"), &self.kv[l].1, d, d], params: [l as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], groups: (0, 0) });
            let cells = n * heads * total;
            stream.push(Op { kernel: "k_scores", bufs: [self.w("q3"), &self.kv[l].0, self.w("scores"), d, d, d],
                params: [cells as u32, hd as u32, heads as u32, total as u32, FRAC_DENSE as u32, 0, group, qd as u32, kvd as u32, 0, 0, 0], groups: groups(cells) });
            stream.push(Op { kernel: "k_softmax", bufs: [self.w("scores"), &self.exp, self.w("probs"), self.w("pos"), d, d],
                params: [(n * heads) as u32, total as u32, heads as u32, self.attn_scale, 0, 0, 0, 0, 0, 0, 0, 0], groups: groups(n * heads) });
            let cells = n * heads * hd;
            stream.push(Op { kernel: "k_pv", bufs: [self.w("probs"), &self.kv[l].1, self.w("attn"), d, d, d],
                params: [cells as u32, hd as u32, heads as u32, total as u32, FRAC_DENSE as u32, 0, group, qd as u32, kvd as u32, 0, 0, 0], groups: groups(cells) });
            self.push_pack(&mut stream, "attn", "attnp", n * qd);
            self.push_proj(&mut stream, &lname(l, "self_attn.o_proj"), "attnp", "o", n);
            self.push_add(&mut stream, "x", "o", "x1", n * hidden);
            self.push_norm(&mut stream, "x1", &lname(l, "post_attention_layernorm"), "h2", n, hidden);
            self.push_pack(&mut stream, "h2", "h2p", n * hidden);
            self.push_proj(&mut stream, &lname(l, "mlp.gate_proj"), "h2p", "g", n);
            self.push_proj(&mut stream, &lname(l, "mlp.up_proj"), "h2p", "u", n);
            stream.push(Op { kernel: "k_silu", bufs: [self.w("g"), self.w("u"), &self.exp, self.w("act"), d, d], params: [(n * inter) as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], groups: groups(n * inter) });
            self.push_pack(&mut stream, "act", "actp", n * inter);
            self.push_proj(&mut stream, &lname(l, "mlp.down_proj"), "actp", "dn", n);
            self.push_add(&mut stream, "x1", "dn", "x", n * hidden);
        }
        self.push_norm(&mut stream, "x", "model.norm.weight", "h", n, hidden);
        self.push_pack(&mut stream, "h", "hp", n * hidden);
        self.push_proj(&mut stream, "model.embed_tokens.weight", "hp", "logits", n);

        // params arena
        let nops = stream.len();
        assert!(nops <= self.params_cap, "params arena: {} ops > {}", nops, self.params_cap);
        let mut arena = vec![0u32; nops * (SLOT as usize / 4)];
        for (i, op) in stream.iter().enumerate() {
            arena[i * 64..i * 64 + 12].copy_from_slice(&op.params);
        }
        self.ops.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&arena));
        let t1 = std::time::Instant::now();
        let skip: Vec<String> = std::env::var("ATLAS_SKIP").map(|v| v.split(',').map(|s| s.to_string()).collect()).unwrap_or_default();
        // encode
        let mut enc = self.ops.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pass") });
        let kv_row_bytes = (kvd * 4) as u64;
        for (i, op) in stream.iter().enumerate() {
            if op.kernel == "copy_kv" {
                enc.copy_buffer_to_buffer(op.bufs[0], 0, op.bufs[1], (p0 as u64) * kv_row_bytes, (n as u64) * kv_row_bytes);
                enc.copy_buffer_to_buffer(op.bufs[2], 0, op.bufs[3], (p0 as u64) * kv_row_bytes, (n as u64) * kv_row_bytes);
                continue;
            }
            if skip.iter().any(|k| k == op.kernel) {
                continue; // timing experiments only: the pass is then wrong by construction
            }
            let bind = self.ops.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.ops.layout, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: std::num::NonZeroU64::new(48) }) },
                wgpu::BindGroupEntry { binding: 1, resource: op.bufs[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: op.bufs[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: op.bufs[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: op.bufs[3].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: op.bufs[4].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: op.bufs[5].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.err.as_entire_binding() },
            ] });
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.ops.pipelines[op.kernel]);
            pass.set_bind_group(0, &bind, &[(i as u32) * SLOT as u32]);
            pass.dispatch_workgroups(op.groups.0, op.groups.1, 1);
        }
        // readback
        let lr = &self.logits_read.as_ref().unwrap().0;
        enc.copy_buffer_to_buffer(self.w("logits"), 0, lr, 0, lbytes);
        enc.copy_buffer_to_buffer(&self.err, 0, &self.err_read, 0, 4);
        let t2 = std::time::Instant::now();
        self.ops.queue.submit(Some(enc.finish()));
        let t3 = std::time::Instant::now();
        let (tx, rx) = std::sync::mpsc::channel();
        lr.slice(..lbytes).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let (tx2, rx2) = std::sync::mpsc::channel();
        self.err_read.slice(..).map_async(wgpu::MapMode::Read, move |r| tx2.send(r).unwrap());
        self.ops.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        rx2.recv().unwrap().unwrap();
        if trace {
            eprintln!("pass n={} p0={} ops={} build={}ms encode={}ms submit={}ms wait={}ms", n, p0, nops, (t1 - t0).as_millis(), (t2 - t1).as_millis(), (t3 - t2).as_millis(), t3.elapsed().as_millis());
        }
        let e: u32 = bytemuck::cast_slice::<u8, u32>(&self.err_read.slice(..).get_mapped_range())[0];
        self.err_read.unmap();
        assert_eq!(e, 0, "card refused: flags {:#x} (1 = lane overflow, 2 = bf16 overflow, 4 = term below the exact scale)", e);
        let out: Vec<u16> = bytemuck::cast_slice::<u8, u32>(&lr.slice(..lbytes).get_mapped_range()).iter().map(|&v| v as u16).collect();
        lr.unmap();
        drop(stream);
        self.kv_len = total;
        out
    }
}
