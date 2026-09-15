//! The three surfaces (spec §2), loaded cold from the intake's receipt and grids.
//!
//! - `Atoms`: per dictionary id, (sheet, m, n) — decomposed once from DICT.u16.
//! - `Place`: the Morton tile schedule for a (tile_rows, tile_cols) shape — computed.
//! - `Grid`:  a tensor's ids exactly as written on disk (tile-major), plus the Place
//!            that addresses them. `line(r, tc)` returns 128 contiguous ids of row r
//!            in tile-column tc; nothing is ever un-tiled.
//!
//! `Model` owns the config, the atoms, every 2-D grid by tensor name, and the 1-D
//! norm vectors (already composed to patterns: they are 145 tiny vectors and are the
//! γ of RMSNorm, an activation-side operand).

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::cells;

pub const TILE: usize = 128;

// ── config ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub tied: bool,
}

fn find_int(text: &str, key: &str) -> Option<usize> {
    let k = format!("\"{}\"", key);
    let p = text.find(&k)? + k.len();
    let rest = &text[p..];
    let c = rest.find(':')? + 1;
    let rest = rest[c..].trim_start();
    let end = rest.find(|ch: char| !ch.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn find_bool(text: &str, key: &str) -> Option<bool> {
    let k = format!("\"{}\"", key);
    let p = text.find(&k)? + k.len();
    let rest = text[p..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

impl Config {
    pub fn from_json(text: &str) -> Config {
        let g = |k: &str| find_int(text, k).unwrap_or_else(|| panic!("config: missing {}", k));
        Config {
            hidden: g("hidden_size"),
            intermediate: g("intermediate_size"),
            n_layers: g("num_hidden_layers"),
            n_q_heads: g("num_attention_heads"),
            n_kv_heads: g("num_key_value_heads"),
            head_dim: g("head_dim"),
            vocab: g("vocab_size"),
            tied: find_bool(text, "tie_word_embeddings").unwrap_or(false),
        }
    }
    pub fn q_dim(&self) -> usize {
        self.n_q_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

// ── atoms ─────────────────────────────────────────────────────────────────────

/// The alphabet, per dictionary id. `pattern` is the id's bf16 pattern, kept only for
/// the activation-side composition (embedding rows, γ); the matmul never reads it.
pub struct Atoms {
    pub sheet: Vec<u8>,
    pub m: Vec<u8>,
    pub n: Vec<i16>,
    pub pattern: Vec<u16>,
}

impl Atoms {
    pub fn from_dict(dict: &[u16]) -> Atoms {
        let mut a = Atoms {
            sheet: Vec::with_capacity(dict.len()),
            m: Vec::with_capacity(dict.len()),
            n: Vec::with_capacity(dict.len()),
            pattern: dict.to_vec(),
        };
        for &p in dict {
            assert!(cells::is_finite(p), "dict carries a non-finite pattern {:#06x}", p);
            let (s, m, n) = cells::decompose(p);
            a.sheet.push(s);
            a.m.push(m);
            a.n.push(n);
        }
        a
    }
    pub fn len(&self) -> usize {
        self.pattern.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pattern.is_empty()
    }
}

// ── place ─────────────────────────────────────────────────────────────────────

fn spread(mut x: u64) -> u64 {
    x &= 0xFFFF_FFFF;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Morton code of (tile_row, tile_col): row bits odd, column bits even — as the intake.
#[inline]
pub fn morton(tr: usize, tc: usize) -> u64 {
    spread(tc as u64) | (spread(tr as u64) << 1)
}

/// The schedule: `rank[tr * tile_cols + tc]` = position of that tile in the file.
pub struct Place {
    pub tile_rows: usize,
    pub tile_cols: usize,
    pub rank: Vec<u32>,
}

impl Place {
    pub fn new(tile_rows: usize, tile_cols: usize) -> Place {
        let n = tile_rows * tile_cols;
        let mut keys: Vec<(u64, u32)> = Vec::with_capacity(n);
        for tr in 0..tile_rows {
            for tc in 0..tile_cols {
                keys.push((morton(tr, tc), (tr * tile_cols + tc) as u32));
            }
        }
        keys.sort_unstable();
        let mut rank = vec![0u32; n];
        for (pos, (_, idx)) in keys.iter().enumerate() {
            rank[*idx as usize] = pos as u32;
        }
        Place { tile_rows, tile_cols, rank }
    }
}

// ── grid ──────────────────────────────────────────────────────────────────────

pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub th: usize,
    pub tw: usize,
    pub ids: Vec<u16>,
    pub place: Arc<Place>,
    /// lowest and highest rung present in this tensor (its one denominator is 2^rmin)
    pub rmin: i32,
    pub rmax: i32,
}

impl Grid {
    /// 128 (tw) contiguous ids: row `r`, tile column `tc`.
    #[inline(always)]
    pub fn line(&self, r: usize, tc: usize) -> &[u16] {
        let tr = r / self.th;
        let ln = r % self.th;
        let t = self.place.rank[tr * self.place.tile_cols + tc] as usize;
        let base = t * self.th * self.tw + ln * self.tw;
        &self.ids[base..base + self.tw]
    }
    pub fn tile_cols(&self) -> usize {
        self.cols / self.tw
    }
}

// ── model ─────────────────────────────────────────────────────────────────────

pub struct Model {
    pub id: String,
    pub cfg: Config,
    pub atoms: Atoms,
    pub grids: HashMap<String, Grid>,
    pub norms: HashMap<String, Vec<u16>>,
    /// The third surface: every grid folded by (sign, odd, rung) per row. Built when
    /// `ATLAS_KERNEL=radix`; the forward then runs `fold::matmul_fold` for every projection.
    pub folds: HashMap<String, crate::fold::TensorFold>,
    /// The fold cube (rung bands), mapped when `ATLAS_KERNEL=cub`.
    pub cub: Option<crate::cub::CubFold>,
    /// The dictionary as positions on the 2-adic torus (`ATLAS_KERNEL=fiber`): the id side.
    pub fiber: Option<crate::fiberkernel::FiberAtoms>,
    /// The placement resident on the card (`ATLAS_KERNEL=card`, GOAL step 3).
    pub card: Option<crate::card::Card>,
}

fn read_u16(path: &Path) -> io::Result<Vec<u16>> {
    let b = fs::read(path)?;
    assert!(b.len() % 2 == 0, "{}: odd byte length", path.display());
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// Manifest entries (name, shape) scanned from MANIFEST.json (sort_keys: "name" precedes "shape").
fn scan_manifest(text: &str) -> Vec<(String, Vec<usize>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(p) = text[pos..].find("\"name\": \"") {
        let start = pos + p + 9;
        let end = start + text[start..].find('"').unwrap();
        let name = text[start..end].to_string();
        let sp = end + text[end..].find("\"shape\": [").unwrap() + 10;
        let se = sp + text[sp..].find(']').unwrap();
        let shape: Vec<usize> = text[sp..se]
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        out.push((name, shape));
        pos = se;
    }
    out
}

impl Model {
    /// `receipts`: the intake receipt dir (DICT.u16, MANIFEST.json, config.json);
    /// `cells_dir`: where the `.grid` files are.
    pub fn load(id: &str, receipts: &Path, cells_dir: &Path) -> io::Result<Model> {
        let cfg = Config::from_json(&fs::read_to_string(receipts.join("config.json"))?);
        let atoms = Atoms::from_dict(&read_u16(&receipts.join("DICT.u16"))?);
        let manifest = fs::read_to_string(receipts.join("MANIFEST.json"))?;
        let mut grids = HashMap::new();
        let mut norms = HashMap::new();
        let mut places: HashMap<(usize, usize), Arc<Place>> = HashMap::new();
        // Under the cube the projections live in the mapped file; only the embedding grid (rows of
        // patterns for `embed_row`) and the norms are read. The footprint is then the map, not the ids.
        let cub_only = std::env::var("ATLAS_KERNEL").map(|v| v == "cub").unwrap_or(false);
        for (name, shape) in scan_manifest(&manifest) {
            if cub_only && shape.len() == 2 && name != "model.embed_tokens.weight" {
                continue;
            }
            let path: PathBuf = cells_dir.join(format!("{}.grid", name));
            let ids = read_u16(&path)?;
            if shape.len() == 1 {
                // a norm vector: compose to patterns now (activation-side operand)
                let v: Vec<u16> = ids.iter().map(|&i| atoms.pattern[i as usize]).collect();
                assert_eq!(v.len(), shape[0]);
                norms.insert(name, v);
                continue;
            }
            let (rows, cols) = (shape[0], shape[1]);
            assert_eq!(ids.len(), rows * cols, "{}: grid size", name);
            let key = (rows / TILE, cols / TILE);
            let place = places
                .entry(key)
                .or_insert_with(|| Arc::new(Place::new(key.0, key.1)))
                .clone();
            let (mut rmin, mut rmax) = (i32::MAX, i32::MIN);
            for &id in &ids {
                let id = id as usize;
                if atoms.m[id] != 0 {
                    rmin = rmin.min(atoms.n[id] as i32);
                    rmax = rmax.max(atoms.n[id] as i32);
                }
            }
            if rmin > rmax {
                rmin = 0;
                rmax = 0;
            }
            grids.insert(name, Grid { rows, cols, th: TILE, tw: TILE, ids, place, rmin, rmax });
        }
        let mut folds = HashMap::new();
        if std::env::var("ATLAS_KERNEL").map(|v| v == "radix").unwrap_or(false) {
            let t = std::time::Instant::now();
            let mut bytes = 0usize;
            for (name, g) in grids.iter() {
                let f = crate::fold::TensorFold::from_grid(g, &atoms);
                bytes += f.bytes();
                folds.insert(name.clone(), f);
            }
            eprintln!("folds built: {} tensors, {} MB, {} ms", folds.len(), bytes / 1_000_000, t.elapsed().as_millis());
        }
        let mut cub = None;
        if std::env::var("ATLAS_KERNEL").map(|v| v == "cub").unwrap_or(false) {
            let t = std::time::Instant::now();
            let c = crate::cub::CubFold::open(&cells_dir.join("fold.cub"))?;
            eprintln!("fold cube mapped: {} tensors, {} MB, {} ms", c.specs.len(), c.bytes() / 1_000_000, t.elapsed().as_millis());
            cub = Some(c);
        }
        let mut fiber = None;
        if std::env::var("ATLAS_KERNEL").map(|v| v == "fiber").unwrap_or(false) {
            let f = crate::fiberkernel::FiberAtoms::from_atoms(&atoms);
            eprintln!("fiber atoms: {} positions, antilog {} entries, {} KB", f.k.len(), f.antilog.len(), f.bytes() / 1000);
            fiber = Some(f);
        }
        let mut model = Model { id: id.to_string(), cfg, atoms, grids, norms, folds, cub, fiber, card: None };
        if std::env::var("ATLAS_KERNEL").map(|v| v == "card").unwrap_or(false) {
            let t = std::time::Instant::now();
            let c = crate::card::Card::new(&model).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            eprintln!("card: {} — {} grids resident, {} MB, {} ms", c.adapter_name, c.grids.len(), c.resident_bytes / 1_000_000, t.elapsed().as_millis());
            model.card = Some(c);
        }
        Ok(model)
    }

    pub fn radix(&self) -> bool {
        !self.folds.is_empty()
    }
    pub fn fold(&self, name: &str) -> &crate::fold::TensorFold {
        self.folds.get(name).unwrap_or_else(|| panic!("no fold {}", name))
    }

    pub fn grid(&self, name: &str) -> &Grid {
        self.grids.get(name).unwrap_or_else(|| panic!("no grid {}", name))
    }
    pub fn norm(&self, name: &str) -> &[u16] {
        self.norms.get(name).unwrap_or_else(|| panic!("no norm {}", name))
    }

    /// One embedding row composed to patterns (the input surface).
    pub fn embed_row(&self, token: usize) -> Vec<u16> {
        let g = self.grid("model.embed_tokens.weight");
        assert!(token < g.rows, "token {} >= vocab {}", token, g.rows);
        let mut out = Vec::with_capacity(g.cols);
        for tc in 0..g.tile_cols() {
            for &id in g.line(token, tc) {
                out.push(self.atoms.pattern[id as usize]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_is_z_order() {
        let p = Place::new(3, 5);
        // tile (tr,tc) row-major index → rank; from the intake's smoke test the file order is
        // [0,1,5,6,2,3,7,8,10,11,12,13,4,9,14]
        let order = [0usize, 1, 5, 6, 2, 3, 7, 8, 10, 11, 12, 13, 4, 9, 14];
        for (pos, idx) in order.iter().enumerate() {
            assert_eq!(p.rank[*idx] as usize, pos);
        }
    }

    #[test]
    fn config_parses() {
        let c = Config::from_json(r#"{"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": true}"#);
        assert_eq!((c.hidden, c.intermediate, c.n_layers, c.n_q_heads, c.n_kv_heads, c.head_dim, c.vocab, c.tied), (2048, 6144, 28, 16, 8, 128, 151936, true));
    }
}
