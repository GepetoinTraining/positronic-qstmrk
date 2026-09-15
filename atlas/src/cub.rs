//! `cub.rs` — the fold as a cube with rung bands (ISIS3 PVL header, V5 `.cub` lineage).
//!
//! The third surface, written once by the intake side and mapped at load. Layout:
//!
//!   [ PVL header, ASCII, padded to 64 B ]
//!   [ index: per tensor, per line-tile, a u64 LE byte offset of the tile block ]
//!   [ body: per tensor, per line-tile of 128 rows, per BAND (rung, top down), per line:
//!       u16 n_entries; n_entries × (u8 lane, u8 count); Σcount × u16 column ]
//!
//! A band is a rung. Walking a tile's bands from the top rung down is the carriage: every
//! (line, lane) accumulator doubles once per band (lazily, by the gap since it was last
//! touched) and adds the band's column sums — the tick ↑ everywhere at once. After the last
//! band the odd chain per (line, plane) is the gather. Nothing per bucket is stored: the band
//! index is the rung, the lane byte is (sign, odd). Same exact integer as `fold.rs` and
//! `matmul.rs`, hence the same bits.
//!
//! Integer only. Gated by `gate::source_float_free`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::fold::{prefix, Plane, TensorFold};
use crate::matmul::{frac_for, Bucket, Wide};
use crate::v5::mmap::Mmap;

pub const LINE_TILE: usize = 128;
const HEADER_PAD: usize = 64;

#[derive(Clone, Debug)]
pub struct CubTensorSpec {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub rmin: i32,
    pub rmax: i32,
    pub tiles: usize,
    /// 0-based byte offset of this tensor's tile index (u64 × tiles) in the file
    pub index_start: usize,
    /// 0-based byte offset of the first tile block
    pub body_start: usize,
    pub body_bytes: usize,
}

impl CubTensorSpec {
    pub fn bands(&self) -> usize {
        (self.rmax - self.rmin + 1) as usize
    }
}

// ── writer ────────────────────────────────────────────────────────────────────

/// Serialize one row's buckets (sorted by (sign, odd, rung desc) as `TensorFold` keeps them)
/// into per-band entry lists: band → Vec<(lane, count, cols slice range)>.
fn row_by_band(f: &TensorFold, o: usize) -> Vec<Vec<(u8, u16, usize, usize)>> {
    let rspan = f.rspan;
    let row = &f.rows_f[o];
    let mut by_band: Vec<Vec<(u8, u16, usize, usize)>> = (0..rspan).map(|_| Vec::new()).collect();
    let mut c = 0usize;
    for &(key, cnt) in &row.buckets {
        let key = key as usize;
        let lane = key / rspan; // sign·128 + odd_idx
        let band = key % rspan; // rmax − rung
        by_band[band].push((lane as u8, cnt, c, c + cnt as usize));
        c += cnt as usize;
    }
    by_band
}

/// Write every fold as one cube. `folds` in a fixed order (sorted by name).
pub fn write_cub(path: &Path, model_id: &str, intake_seal: &str, folds: &[(String, &TensorFold)]) -> std::io::Result<Vec<CubTensorSpec>> {
    // pass 1: sizes
    let mut specs: Vec<CubTensorSpec> = Vec::new();
    let mut tile_sizes: Vec<Vec<u64>> = Vec::new();
    for (name, f) in folds {
        assert!(f.rows % LINE_TILE == 0, "{}: rows {} not a multiple of the line tile", name, f.rows);
        let tiles = f.rows / LINE_TILE;
        let mut sizes = Vec::with_capacity(tiles);
        for t in 0..tiles {
            let mut bytes = 0u64;
            for o in t * LINE_TILE..(t + 1) * LINE_TILE {
                let bb = row_by_band(f, o);
                for band in &bb {
                    let mut entries = 0u64;
                    let mut cols = 0u64;
                    for &(_, cnt, _, _) in band {
                        entries += (cnt as u64).div_ceil(255);
                        cols += cnt as u64;
                    }
                    bytes += 2 + entries * 2 + cols * 2;
                }
            }
            sizes.push(bytes);
        }
        specs.push(CubTensorSpec { name: name.clone(), rows: f.rows, cols: f.cols, rmin: f.rmin, rmax: f.rmax, tiles, index_start: 0, body_start: 0, body_bytes: sizes.iter().sum::<u64>() as usize });
        tile_sizes.push(sizes);
    }
    // layout: header, then all indices, then all bodies
    let mut header = compose_header(model_id, intake_seal, &specs, 0, 0);
    let mut hlen = header.len().div_ceil(HEADER_PAD) * HEADER_PAD;
    // fixed-point: header length depends on offsets which depend on header length; iterate twice
    for _ in 0..3 {
        let mut off = hlen;
        for (i, s) in specs.iter_mut().enumerate() {
            s.index_start = off;
            off += 8 * tile_sizes[i].len();
        }
        for s in specs.iter_mut() {
            s.body_start = off;
            off += s.body_bytes;
        }
        header = compose_header(model_id, intake_seal, &specs, hlen, off);
        let nh = header.len().div_ceil(HEADER_PAD) * HEADER_PAD;
        if nh == hlen {
            break;
        }
        hlen = nh;
    }
    let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
    let mut hb = header.into_bytes();
    hb.resize(hlen, b' ');
    w.write_all(&hb)?;
    // indices
    for (i, s) in specs.iter().enumerate() {
        let mut off = s.body_start as u64;
        for &sz in &tile_sizes[i] {
            w.write_all(&off.to_le_bytes())?;
            off += sz;
        }
    }
    // bodies
    for (name, f) in folds {
        let _ = name;
        let tiles = f.rows / LINE_TILE;
        for t in 0..tiles {
            let rows_bb: Vec<Vec<Vec<(u8, u16, usize, usize)>>> = (t * LINE_TILE..(t + 1) * LINE_TILE).map(|o| row_by_band(f, o)).collect();
            for band in 0..f.rspan {
                for (li, o) in (t * LINE_TILE..(t + 1) * LINE_TILE).enumerate() {
                    let entries = &rows_bb[li][band];
                    let n: u64 = entries.iter().map(|&(_, cnt, _, _)| (cnt as u64).div_ceil(255)).sum();
                    w.write_all(&(n as u16).to_le_bytes())?;
                    let cols = &f.rows_f[o].cols;
                    for &(lane, cnt, _, _) in entries {
                        let mut left = cnt;
                        while left > 0 {
                            let take = left.min(255);
                            w.write_all(&[lane, take as u8])?;
                            left -= take;
                        }
                    }
                    for &(_, _, a, b) in entries {
                        for &c in &cols[a..b] {
                            w.write_all(&c.to_le_bytes())?;
                        }
                    }
                }
            }
        }
    }
    w.flush()?;
    Ok(specs)
}

fn compose_header(model_id: &str, intake_seal: &str, specs: &[CubTensorSpec], header_len: usize, file_len: usize) -> String {
    let mut s = String::with_capacity(16 * 1024);
    s.push_str("Object = IsisCube\n");
    s.push_str("  /* Atlas Engine fold cube: the third surface, banded by rung. Lineage: V5 master.cub. */\n");
    s.push_str("  Object = Core\n");
    s.push_str(&format!("    StartByte   = {}\n", header_len + 1));
    s.push_str("    Format      = RungBanded\n");
    s.push_str(&format!("    LineTile    = {}\n", LINE_TILE));
    s.push_str("    Group = Pixels\n      Type = UnsignedWord\n      ByteOrder = Lsb\n      Meaning = \"column index; lane byte = sign*128 + (odd-1)/2; band = rmax - rung\"\n    End_Group\n");
    s.push_str("  End_Object\n");
    s.push_str("  Object = Label\n");
    s.push_str(&format!("    Model       = \"{}\"\n", model_id));
    s.push_str(&format!("    IntakeSeal  = {}\n", intake_seal));
    s.push_str("    Floor       = \"activations below 2^-90 dropped with sticky; above it every term exact\"\n");
    s.push_str(&format!("    Tensors     = {}\n", specs.len()));
    s.push_str(&format!("    FileBytes   = {}\n", file_len));
    s.push_str("  End_Object\n");
    for t in specs {
        s.push_str("  Object = Tensor\n");
        s.push_str(&format!("    Name       = \"{}\"\n", t.name));
        s.push_str(&format!("    Lines      = {}\n    Samples    = {}\n", t.rows, t.cols));
        s.push_str(&format!("    RungMin    = {}\n    RungMax    = {}\n    Bands      = {}\n", t.rmin, t.rmax, t.bands()));
        s.push_str(&format!("    Tiles      = {}\n    IndexByte  = {}\n    BodyByte   = {}\n    BodyBytes  = {}\n", t.tiles, t.index_start + 1, t.body_start + 1, t.body_bytes));
        s.push_str("  End_Object\n");
    }
    s.push_str("End_Object\nEnd\n");
    s
}

// ── reader ────────────────────────────────────────────────────────────────────

pub struct CubFold {
    mmap: Mmap,
    pub specs: Vec<CubTensorSpec>,
    by_name: HashMap<String, usize>,
}

fn pvl_val<'a>(text: &'a str, key: &str, from: usize) -> Option<(&'a str, usize)> {
    let p = text[from..].find(key)? + from;
    let eq = text[p..].find('=')? + p + 1;
    let end = text[eq..].find('\n')? + eq;
    Some((text[eq..end].trim().trim_matches('"'), end))
}

impl CubFold {
    pub fn open(path: &Path) -> std::io::Result<CubFold> {
        let mmap = Mmap::open(path)?;
        let bytes = mmap.as_slice();
        let end = bytes.windows(4).position(|w| w == b"End\n").map(|p| p + 4).unwrap_or(HEADER_PAD);
        let text = std::str::from_utf8(&bytes[..end]).expect("header ascii").to_string();
        let mut specs = Vec::new();
        let mut pos = 0;
        while let Some(p) = text[pos..].find("Object = Tensor") {
            let base = pos + p;
            let (name, _) = pvl_val(&text, "Name", base).unwrap();
            let (rows, _) = pvl_val(&text, "Lines", base).unwrap();
            let (cols, _) = pvl_val(&text, "Samples", base).unwrap();
            let (rmin, _) = pvl_val(&text, "RungMin", base).unwrap();
            let (rmax, _) = pvl_val(&text, "RungMax", base).unwrap();
            let (tiles, _) = pvl_val(&text, "Tiles", base).unwrap();
            let (ib, _) = pvl_val(&text, "IndexByte", base).unwrap();
            let (bb, _) = pvl_val(&text, "BodyByte", base).unwrap();
            let (bbytes, e) = pvl_val(&text, "BodyBytes", base).unwrap();
            specs.push(CubTensorSpec {
                name: name.to_string(),
                rows: rows.parse().unwrap(),
                cols: cols.parse().unwrap(),
                rmin: rmin.parse().unwrap(),
                rmax: rmax.parse().unwrap(),
                tiles: tiles.parse().unwrap(),
                index_start: ib.parse::<usize>().unwrap() - 1,
                body_start: bb.parse::<usize>().unwrap() - 1,
                body_bytes: bbytes.parse().unwrap(),
            });
            pos = e;
        }
        let by_name = specs.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
        Ok(CubFold { mmap, specs, by_name })
    }
    pub fn spec(&self, name: &str) -> &CubTensorSpec {
        &self.specs[*self.by_name.get(name).unwrap_or_else(|| panic!("no tensor {} in cube", name))]
    }
    fn tile_block(&self, s: &CubTensorSpec, t: usize) -> &[u8] {
        let b = self.mmap.as_slice();
        let ix = s.index_start + 8 * t;
        let start = u64::from_le_bytes(b[ix..ix + 8].try_into().unwrap()) as usize;
        let end = if t + 1 < s.tiles {
            let ix2 = ix + 8;
            u64::from_le_bytes(b[ix2..ix2 + 8].try_into().unwrap()) as usize
        } else {
            s.body_start + s.body_bytes
        };
        &b[start..end]
    }
    pub fn bytes(&self) -> usize {
        self.mmap.len()
    }
}

// ── the band-walk kernel ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct SendPtr(*mut u16);
unsafe impl Send for SendPtr {}

/// Planes carried per band walk of a tile (accumulator = 128 lines × 256 lanes × planes × 32 B).
pub const CUB_PLANES: usize = 8;

/// `y = x · Wᵀ` over the cube: `x` `[seq, in_f]` patterns. Bit-identical to `matmul_fold`.
pub fn matmul_cub(x: &[u16], seq: usize, in_f: usize, cub: &CubFold, name: &str) -> Vec<u16> {
    let s = cub.spec(name);
    assert_eq!(s.cols, in_f, "{}: cube cols {} != in_f {}", name, s.cols, in_f);
    assert_eq!(x.len(), seq * in_f);
    let out_f = s.rows;
    let bands = s.bands();
    let frac = frac_for(s.rmin);
    let planes_all: Vec<Plane> = (0..seq).map(|p| prefix(&x[p * in_f..(p + 1) * in_f])).collect();
    let mut y = vec![0u16; seq * out_f];
    let yptr = SendPtr(y.as_mut_ptr());
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1).min(s.tiles);
    let cursor = AtomicUsize::new(0);
    let (cursor_ref, planes_ref) = (&cursor, &planes_all);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(move || {
                let yptr = yptr;
                let mut acc: Vec<Wide> = Vec::new();
                let mut last: Vec<u8> = vec![0; LINE_TILE * 256];
                let mut touched: Vec<bool> = vec![false; LINE_TILE * 256];
                loop {
                    let t = cursor_ref.fetch_add(1, Ordering::Relaxed);
                    if t >= s.tiles {
                        break;
                    }
                    let block = cub.tile_block(s, t);
                    let mut p0 = 0usize;
                    while p0 < seq {
                        let np = (seq - p0).min(CUB_PLANES);
                        let planes = &planes_ref[p0..p0 + np];
                        acc.clear();
                        acc.resize(LINE_TILE * 256 * np, Wide::ZERO);
                        for v in touched.iter_mut() {
                            *v = false;
                        }
                        // walk the bands top-down: the carriage
                        let mut pos = 0usize;
                        for band in 0..bands {
                            for line in 0..LINE_TILE {
                                let n = u16::from_le_bytes([block[pos], block[pos + 1]]) as usize;
                                pos += 2;
                                let ent = &block[pos..pos + 2 * n];
                                pos += 2 * n;
                                let total: usize = ent.chunks_exact(2).map(|e| e[1] as usize).sum();
                                let cols = &block[pos..pos + 2 * total];
                                pos += 2 * total;
                                let mut c = 0usize;
                                for e in ent.chunks_exact(2) {
                                    let lane = e[0] as usize;
                                    let cnt = e[1] as usize;
                                    let idx = line * 256 + lane;
                                    // lazy doubling by the gap in bands since last touched
                                    let gap = if touched[idx] { (band as u8 - last[idx]) as u32 } else { 0 };
                                    let base = idx * np;
                                    for p in 0..np {
                                        let pv = &planes[p].v;
                                        let mut sum = 0i128;
                                        for cc in cols[2 * c..2 * (c + cnt)].chunks_exact(2) {
                                            sum += pv[u16::from_le_bytes([cc[0], cc[1]]) as usize];
                                        }
                                        let mut a = acc[base + p];
                                        if gap > 0 {
                                            a = a.shl(gap);
                                        }
                                        a.add(Wide::from_i128(sum));
                                        acc[base + p] = a;
                                    }
                                    last[idx] = band as u8;
                                    touched[idx] = true;
                                    c += cnt;
                                }
                            }
                        }
                        // the gather: land every lane at rmin, then the odd chain per line per plane
                        for line in 0..LINE_TILE {
                            for p in 0..np {
                                let mut pos_lanes = [Wide::ZERO; 128];
                                let mut neg_lanes = [Wide::ZERO; 128];
                                for lane in 0..256 {
                                    let idx = line * 256 + lane;
                                    if !touched[idx] {
                                        continue;
                                    }
                                    let fin = (bands - 1) as u32 - last[idx] as u32;
                                    let a = acc[idx * np + p].shl(fin);
                                    if lane < 128 {
                                        pos_lanes[lane] = a;
                                    } else {
                                        neg_lanes[lane - 128] = a;
                                    }
                                }
                                let mut total = odd_chain(&pos_lanes);
                                total.sub(odd_chain(&neg_lanes));
                                let out = Bucket { acc: total, sticky: planes[p].sticky, frac }.collapse();
                                let o = t * LINE_TILE + line;
                                // SAFETY: disjoint (position, row) cells per tile; scope joins before y is read.
                                unsafe { *yptr.0.add((p0 + p) * out_f + o) = out };
                            }
                        }
                        p0 += np;
                    }
                }
            });
        }
    });
    y
}

#[inline(always)]
fn odd_chain(a: &[Wide; 128]) -> Wide {
    let mut s = Wide::ZERO;
    let mut t = Wide::ZERO;
    for j in (1..128).rev() {
        s.add(a[j]);
        t.add(s);
    }
    s.add(a[0]);
    s.add(t.shl(1));
    s
}
