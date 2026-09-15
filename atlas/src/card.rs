//! `card.rs` — GOAL.md step 3: the placement resident on the card, the exact bucket walked there
//! (LOG §27, §28). wgpu compute (Vulkan/DX12, no toolkit), WGSL, u32 only — the 256-bit lane is
//! nine 32-bit limbs with explicit carries, one array for positive terms and one for negative, so
//! nothing is rounded before the collapse. The collapse itself is NOT on the card: the limbs come
//! back and go through `matmul::Bucket::collapse`, the same code every kernel uses, so gate 5 is
//! the same three rows. `ATLAS_KERNEL=card`.
//!
//! Per output cell (position, row): walk the row's tile lines in Morton order (the same lines
//! `matmul_grid` walks), one thread per cell; per term: dictionary lookup (packed u32), product of
//! the two odds (< 2^16), shift by n_x + n_w + frac placed into limb sh/32, add with carry.
//! Terms in the same order as the bucket, so the integer is identical, not merely equal.
//!
//! Staged (§28): buffers are persistent and written in place; the independent matmuls of a layer
//! (q, k, v; gate, up) are encoded together — one activation upload, one submit, one wait, one
//! readback — through `matmul_many`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use wgpu::util::DeviceExt;

use crate::matmul::{frac_for, Bucket, Wide, XRow};
use crate::surfaces::{Atoms, Grid, Model};

const LIMBS: usize = 9;
const WG: u32 = 64;
/// at most 65,535 workgroups per dispatch
const CHUNK: usize = 65_535 * WG as usize;

/// Stage clocks, nanoseconds, accumulated across calls: pack, upload+submit, gpu wait, (unused), collapse, calls.
pub static TRACE: [AtomicU64; 6] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

pub fn trace_take() -> [u64; 6] {
    let mut out = [0u64; 6];
    for (i, t) in TRACE.iter().enumerate() {
        out[i] = t.swap(0, Ordering::Relaxed);
    }
    out
}

fn tick(i: usize, t: Instant) -> Instant {
    TRACE[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Instant::now()
}

const SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    seq: u32,
    frac: i32,
    tcs: u32,
    tile_cols_of_place: u32,
    th: u32,
    tw: u32,
    cell0: u32,
    ncells: u32,
    _p2: u32,
    _p3: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;     // u16 ids, two per u32, grid layout
@group(0) @binding(2) var<storage, read> rank: array<u32>;    // Morton rank of tile (tr, tc)
@group(0) @binding(3) var<storage, read> atoms: array<u32>;   // per id: bit31 sign, bits 16..24 m, bits 0..16 n+32768
@group(0) @binding(4) var<storage, read> xact: array<u32>;    // per (position, col): same packing, m = 0 for dropped
@group(0) @binding(5) var<storage, read_write> out: array<u32>; // per cell: 9 limbs pos, 9 limbs neg

fn add_term(acc: ptr<function, array<u32, 9>>, prod: u32, sh: u32) {
    let limb = sh >> 5u;
    let bit = sh & 31u;
    let lo = prod << bit;
    var hi = 0u;
    if (bit != 0u) { hi = prod >> (32u - bit); }
    var s = (*acc)[limb] + lo;
    var c = select(0u, 1u, s < lo);
    (*acc)[limb] = s;
    var l = limb + 1u;
    let v = hi + c;
    s = (*acc)[l] + v;
    c = select(0u, 1u, s < v);
    (*acc)[l] = s;
    loop {
        if (c == 0u) { break; }
        l = l + 1u;
        if (l >= 9u) { break; }
        s = (*acc)[l] + 1u;
        c = select(0u, 1u, s == 0u);
        (*acc)[l] = s;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.ncells) { return; }
    let cell = P.cell0 + gid.x;
    let s = cell / P.rows;
    let o = cell % P.rows;
    var pos: array<u32, 9>;
    var neg: array<u32, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) { pos[i] = 0u; neg[i] = 0u; }
    let tr = o / P.th;
    let ln = o % P.th;
    let xbase = s * P.cols;
    for (var tc = 0u; tc < P.tcs; tc = tc + 1u) {
        let t = rank[tr * P.tile_cols_of_place + tc];
        let base = t * P.th * P.tw + ln * P.tw;
        for (var j = 0u; j < P.tw; j = j + 1u) {
            let idx = base + j;
            let word = ids[idx >> 1u];
            var id = word & 0xFFFFu;
            if ((idx & 1u) != 0u) { id = word >> 16u; }
            let a = atoms[id];
            let x = xact[xbase + tc * P.tw + j];
            let mx = (x >> 16u) & 0xFFu;
            if (mx == 0u) { continue; }
            let mw = (a >> 16u) & 0xFFu;
            let nx = i32(x & 0xFFFFu) - 32768;
            let nw = i32(a & 0xFFFFu) - 32768;
            let sh = u32(nx + nw + P.frac);
            let prod = mx * mw;
            if (((x ^ a) & 0x80000000u) != 0u) {
                add_term(&neg, prod, sh);
            } else {
                add_term(&pos, prod, sh);
            }
        }
    }
    let ob = gid.x * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) { out[ob + i] = pos[i]; out[ob + 9u + i] = neg[i]; }
}
"#;

pub struct CardGrid {
    pub ids: wgpu::Buffer,
    pub rank: wgpu::Buffer,
    pub rows: usize,
    pub cols: usize,
    pub th: usize,
    pub tw: usize,
    pub place_tile_cols: usize,
    pub rmin: i32,
    pub bytes: usize,
}

/// Persistent working buffers (written in place, grown on demand).
struct Pool {
    x: wgpu::Buffer,
    x_cap: usize,
    /// one (params, out, read) triple per dispatch slot
    params: Vec<wgpu::Buffer>,
    out: Vec<wgpu::Buffer>,
    read: Vec<wgpu::Buffer>,
}

pub struct Card {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    atoms: wgpu::Buffer,
    pool: Mutex<Pool>,
    pub grids: HashMap<String, CardGrid>,
    pub resident_bytes: usize,
    pub adapter_name: String,
}

fn pack_atom(sheet: u8, m: u8, n: i16) -> u32 {
    ((sheet as u32) << 31) | ((m as u32) << 16) | ((n as i32 + 32768) as u32 & 0xFFFF)
}

/// One dispatch: which grid, which cell range, which slot it lands in.
struct Job<'a> {
    name: &'a str,
    cell0: usize,
    ncells: usize,
    slot: usize,
}

impl Card {
    pub fn new(model: &Model) -> Result<Card, String> {
        let mut names: Vec<&String> = model.grids.keys().collect();
        names.sort();
        Card::build(&model.atoms, names.iter().map(|n| (n.as_str(), &model.grids[*n])))
    }

    pub fn build<'a>(atoms: &Atoms, grids: impl Iterator<Item = (&'a str, &'a Grid)>) -> Result<Card, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).ok_or("no adapter")?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.max(6);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("atlas card"), required_features: wgpu::Features::empty(), required_limits: limits, memory_hints: wgpu::MemoryHints::Performance }, None)).map_err(|e| format!("{:?}", e))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("bucket"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let entry = |i: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding: i, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bucket layout"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, ro), entry(2, ro), entry(3, ro), entry(4, ro), entry(5, wgpu::BufferBindingType::Storage { read_only: false })],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("bucket"), layout: Some(&pl), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let packed: Vec<u32> = (0..atoms.m.len()).map(|i| pack_atom(atoms.sheet[i], atoms.m[i], atoms.n[i])).collect();
        let atoms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("atoms"), contents: bytemuck::cast_slice(&packed), usage: wgpu::BufferUsages::STORAGE });
        let pool = Pool { x: device.create_buffer(&wgpu::BufferDescriptor { label: Some("x"), size: 4 * 8192, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }), x_cap: 8192, params: Vec::new(), out: Vec::new(), read: Vec::new() };
        let mut card = Card { device, queue, pipeline, layout, atoms: atoms_buf, pool: Mutex::new(pool), grids: HashMap::new(), resident_bytes: packed.len() * 4, adapter_name: format!("{} ({:?})", info.name, info.backend) };
        for (name, g) in grids {
            card.upload(name, g);
        }
        Ok(card)
    }

    fn upload(&mut self, name: &str, g: &Grid) {
        let ids = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(name), contents: bytemuck::cast_slice(&g.ids), usage: wgpu::BufferUsages::STORAGE });
        let rank = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("rank"), contents: bytemuck::cast_slice(&g.place.rank), usage: wgpu::BufferUsages::STORAGE });
        self.resident_bytes += g.ids.len() * 2 + g.place.rank.len() * 4;
        self.grids.insert(name.to_string(), CardGrid { ids, rank, rows: g.rows, cols: g.cols, th: g.th, tw: g.tw, place_tile_cols: g.place.tile_cols, rmin: g.rmin, bytes: g.ids.len() * 2 });
    }

    fn ensure_slots(&self, pool: &mut Pool, n: usize) {
        while pool.params.len() < n {
            pool.params.push(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: 48, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            let bytes = (CHUNK * 2 * LIMBS * 4) as u64;
            pool.out.push(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("out"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }));
            pool.read.push(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        }
    }

    /// `y = x · Wᵀ` on the card for one grid. Bit-identical to `matmul_grid`.
    pub fn matmul(&self, x: &[u16], seq: usize, in_f: usize, name: &str) -> Vec<u16> {
        self.matmul_many(x, seq, in_f, &[name]).pop().unwrap()
    }

    /// Several grids over the same activation (q, k, v; gate, up): one upload, one submit, one wait.
    pub fn matmul_many(&self, x: &[u16], seq: usize, in_f: usize, names: &[&str]) -> Vec<Vec<u16>> {
        assert_eq!(x.len(), seq * in_f);
        TRACE[5].fetch_add(names.len() as u64, Ordering::Relaxed);
        let mut t = Instant::now();
        let rows: Vec<XRow> = (0..seq).map(|s| XRow::new(&x[s * in_f..(s + 1) * in_f])).collect();
        let mut xp: Vec<u32> = Vec::with_capacity(seq * in_f);
        for r in &rows {
            for i in 0..in_f {
                xp.push(pack_atom(r.sheet[i], r.m[i] as u8, r.n[i] as i16));
            }
        }
        // the jobs: one per (grid, chunk)
        let mut jobs: Vec<Job> = Vec::new();
        let mut slot = 0usize;
        for &name in names {
            let g = self.grids.get(name).unwrap_or_else(|| panic!("card: no grid {}", name));
            assert_eq!(g.cols, in_f, "card: grid cols {} != in_f {}", g.cols, in_f);
            let frac = frac_for(g.rmin);
            for r in &rows {
                for i in 0..in_f {
                    if r.m[i] != 0 {
                        assert!(r.n[i] + g.rmin + frac >= 0, "card: term below the exact scale");
                    }
                }
            }
            let cells = seq * g.rows;
            let mut cell0 = 0usize;
            while cell0 < cells {
                let ncells = (cells - cell0).min(CHUNK);
                jobs.push(Job { name, cell0, ncells, slot });
                slot += 1;
                cell0 += ncells;
            }
        }
        t = tick(0, t);
        let mut pool = self.pool.lock().unwrap();
        self.ensure_slots(&mut pool, jobs.len());
        if pool.x_cap < xp.len() {
            pool.x = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("x"), size: (xp.len() * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            pool.x_cap = xp.len();
        }
        self.queue.write_buffer(&pool.x, 0, bytemuck::cast_slice(&xp));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut binds = Vec::with_capacity(jobs.len());
        for j in &jobs {
            let g = &self.grids[j.name];
            let frac = frac_for(g.rmin);
            let params: [u32; 12] = [g.rows as u32, g.cols as u32, seq as u32, frac as u32, (g.cols / g.tw) as u32, g.place_tile_cols as u32, g.th as u32, g.tw as u32, j.cell0 as u32, j.ncells as u32, 0, 0];
            self.queue.write_buffer(&pool.params[j.slot], 0, bytemuck::cast_slice(&params));
            binds.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pool.params[j.slot].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: g.ids.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: g.rank.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.atoms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: pool.x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: pool.out[j.slot].as_entire_binding() },
                ],
            }));
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            for (j, b) in jobs.iter().zip(binds.iter()) {
                pass.set_bind_group(0, b, &[]);
                pass.dispatch_workgroups(((j.ncells as u32) + WG - 1) / WG, 1, 1);
            }
        }
        for j in &jobs {
            let bytes = (j.ncells * 2 * LIMBS * 4) as u64;
            enc.copy_buffer_to_buffer(&pool.out[j.slot], 0, &pool.read[j.slot], 0, bytes);
        }
        self.queue.submit(Some(enc.finish()));
        t = tick(1, t);
        // one wait for every readback
        let mut rxs = Vec::with_capacity(jobs.len());
        for j in &jobs {
            let bytes = (j.ncells * 2 * LIMBS * 4) as u64;
            let (tx, rx) = std::sync::mpsc::channel();
            pool.read[j.slot].slice(..bytes).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            rxs.push(rx);
        }
        self.device.poll(wgpu::Maintain::Wait);
        for rx in rxs {
            rx.recv().unwrap().unwrap();
        }
        t = tick(2, t);
        // collapse, per grid
        let mut outs: Vec<Vec<u16>> = names.iter().map(|n| vec![0u16; seq * self.grids[*n].rows]).collect();
        for j in &jobs {
            let g = &self.grids[j.name];
            let frac = frac_for(g.rmin);
            let gi = names.iter().position(|n| *n == j.name).unwrap();
            let bytes = (j.ncells * 2 * LIMBS * 4) as u64;
            let data = pool.read[j.slot].slice(..bytes).get_mapped_range();
            let limbs: &[u32] = bytemuck::cast_slice(&data);
            let y = &mut outs[gi];
            for ci in 0..j.ncells {
                let c = j.cell0 + ci;
                let p = &limbs[ci * 18..ci * 18 + 9];
                let n = &limbs[ci * 18 + 9..ci * 18 + 18];
                let mut d = [0u32; 9];
                let mut borrow = 0u64;
                for i in 0..9 {
                    let v = p[i] as u64;
                    let w = n[i] as u64 + borrow;
                    if v >= w {
                        d[i] = (v - w) as u32;
                        borrow = 0;
                    } else {
                        d[i] = (v + (1u64 << 32) - w) as u32;
                        borrow = 1;
                    }
                }
                let negative = borrow == 1;
                assert!(d[8] == if negative { 0xFFFF_FFFF } else { 0 }, "card refused: cell overflows the 256-bit lane");
                let lo = (d[0] as u128) | ((d[1] as u128) << 32) | ((d[2] as u128) << 64) | ((d[3] as u128) << 96);
                let hi = ((d[4] as u128) | ((d[5] as u128) << 32) | ((d[6] as u128) << 64) | ((d[7] as u128) << 96)) as i128;
                let s = c / g.rows;
                y[c] = Bucket { acc: Wide { hi, lo }, sticky: rows[s].sticky, frac }.collapse();
            }
            drop(data);
            pool.read[j.slot].unmap();
        }
        tick(4, t);
        outs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells;
    use crate::matmul::matmul_grid;
    use crate::surfaces::Place;
    use std::sync::Arc;

    #[test]
    fn card_reproduces_the_bucket_bit_for_bit() {
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut dict = Vec::new();
        while dict.len() < 512 {
            let sign = ((next() & 1) as u16) << 15;
            let exp = (110 + (next() % 30)) as u16;
            let frac = (next() % 128) as u16;
            let p = sign | (exp << 7) | frac;
            if cells::is_finite(p) && !dict.contains(&p) {
                dict.push(p);
            }
        }
        let atoms = Atoms::from_dict(&dict);
        let (rows, cols) = (256usize, 256usize);
        let mut mk = || -> Grid {
            let ids: Vec<u16> = (0..rows * cols).map(|_| (next() % 512) as u16).collect();
            let (mut rmin, mut rmax) = (i32::MAX, i32::MIN);
            for &id in &ids {
                rmin = rmin.min(atoms.n[id as usize] as i32);
                rmax = rmax.max(atoms.n[id as usize] as i32);
            }
            Grid { rows, cols, th: 128, tw: 128, ids, place: Arc::new(Place::new(2, 2)), rmin, rmax }
        };
        let g1 = mk();
        let g2 = mk();
        let x: Vec<u16> = (0..3 * cols).map(|_| dict[(next() % 512) as usize]).collect();
        let card = match Card::build(&atoms, [("t1", &g1), ("t2", &g2)].into_iter()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no card: {} (test skipped)", e);
                return;
            }
        };
        let a1 = matmul_grid(&x, 3, cols, &g1, &atoms);
        let a2 = matmul_grid(&x, 3, cols, &g2, &atoms);
        let b = card.matmul_many(&x, 3, cols, &["t1", "t2"]);
        assert_eq!(a1, b[0], "card vs bucket, grid 1");
        assert_eq!(a2, b[1], "card vs bucket, grid 2");
        assert_eq!(a1, card.matmul(&x, 3, cols, "t1"));
        assert!(a1.iter().any(|&v| v != 0));
    }
}
