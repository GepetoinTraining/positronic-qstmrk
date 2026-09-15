//! Atlas Engine — spec §2–§4, step 2 of §9.
//!
//! The model is three cold surfaces that know nothing of each other and are
//! composed per cell only when a value is needed:
//!
//!   - `surfaces::Atoms`  — the closed alphabet: per dictionary id, its sheet, its
//!                          odd core m and its rung n. Universal to bf16; the model
//!                          chose nothing here (native_object.pdf, Prop. 1).
//!   - `surfaces::Place`  — the Morton schedule: which tile holds (row, col). Pure
//!                          geometry, computed, never stored.
//!   - `surfaces::Grid`   — the ids, tile-major as the intake wrote them. Never
//!                          un-tiled into a dense matrix.
//!
//! The five ops: matmul is this crate's `matmul` (exact products of odd cores,
//! exponent adds, one i128 bucket per output cell, one collapse at the gather —
//! spec §3 "Buckets"); RMSNorm, RoPE, softmax and SiLU are V5's proven
//! bijective-lookup primitives, inherited unchanged (spec §4).
//!
//! Discipline: integer/byte only in every module here. `gate::source_float_free`
//! greps this crate's own source for the two IEEE type names.

pub mod cells;
pub mod surfaces;
pub mod matmul;
pub mod fold;
pub mod cub;
pub mod clock;
pub mod cache;
pub mod turn;
pub mod forward;
pub mod gate;
pub mod ball;
pub mod maelstrom;
pub mod fiber;
pub mod fiberkernel;
pub mod card;
pub mod card_ops;
pub mod card_layer;

pub use v5;
