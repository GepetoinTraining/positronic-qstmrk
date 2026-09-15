//! `cache.rs` — the acquisition and the checker (handshake-paper.pdf; pgarcia 2026-09-13):
//! "the clock you use for the first inference and another without anything; we cache the first
//! and the second is what runtime uses — a digital checker: if we need the same logit, pick it up."
//!
//! The first inference over a transcript is the ACQUISITION: every position's per-layer
//! activations (the layer input x, and the roped-and-normed k and v it produced) are kept,
//! stamped with the position's ring address and sealed under the prefix of token ids that
//! produced them. It is annotated, never re-timed.
//!
//! A later turn over a longer transcript is a RE-SLICING. It is free exactly when it is
//! information-contained in the acquisition: the causal prefix is identical (the seal matches),
//! so every old position's activations are what the full pass would recompute — bit for bit,
//! because the pass is exact (LOG §7: the first 8 positions of the 508 equal the seq-8
//! reference). The runtime computes only the new positions, attending over the old k, v picked
//! up from the cache, and extends the acquisition. If the seal does not match the prefix, the
//! checker refuses and the turn is a fresh acquisition.
//!
//! Integer only; the seal is sha256 over the token ids.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::clock::{Address, Clock};

/// One position's kept state for one layer.
#[derive(Clone)]
pub struct Slice {
    pub x_in: Vec<u16>, // the residual stream entering the layer
    pub k: Vec<u16>,    // normed + roped keys, kv_dim
    pub v: Vec<u16>,    // values, kv_dim
}

pub struct Acquisition {
    pub tokens: Vec<usize>,
    pub seal: String,
    pub addresses: Vec<Address>,
    /// layers × positions
    pub slices: Vec<Vec<Slice>>,
    /// the final normed hidden per position (the head's input), for pickup of logits
    pub final_h: Vec<Vec<u16>>,
    pub by_address: HashMap<(usize, Address), usize>,
}

pub fn seal_of(tokens: &[usize]) -> String {
    let mut h = Sha256::new();
    for &t in tokens {
        h.update((t as u32).to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

impl Acquisition {
    pub fn empty(n_layers: usize) -> Acquisition {
        Acquisition { tokens: Vec::new(), seal: seal_of(&[]), addresses: Vec::new(), slices: (0..n_layers).map(|_| Vec::new()).collect(), final_h: Vec::new(), by_address: HashMap::new() }
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    /// The checker: how many leading positions of `tokens` are information-contained here.
    /// The whole prefix must match token for token — a seal over the prefix, not a guess.
    /// The longest prefix of `tokens` that is information-contained in the acquisition: every
    /// slice at a position depends only on the tokens up to it (causal), so the common prefix is
    /// exactly what can be picked up; the seal of that prefix is the receipt.
    pub fn contained_prefix(&self, tokens: &[usize]) -> usize {
        let n = self.tokens.len().min(tokens.len());
        let k = (0..n).take_while(|&i| self.tokens[i] == tokens[i]).count();
        if k > 0 && seal_of(&self.tokens[..k]) != seal_of(&tokens[..k]) {
            return 0;
        }
        k
    }
    /// Pick up a position's slice by its ring address (the digital checker's lookup).
    pub fn pick(&self, layer: usize, addr: Address) -> Option<&Slice> {
        self.by_address.get(&(layer, addr)).map(|&p| &self.slices[layer][p])
    }
    pub fn push_position(&mut self, token: usize) {
        let pos = self.tokens.len() as u64;
        let addr = Clock::address_of(pos);
        self.tokens.push(token);
        self.addresses.push(addr);
        self.seal = seal_of(&self.tokens);
    }
    pub fn record(&mut self, layer: usize, pos: usize, slice: Slice) {
        debug_assert_eq!(self.slices[layer].len(), pos);
        self.slices[layer].push(slice);
        self.by_address.insert((layer, self.addresses[pos]), pos);
    }
    /// Truncate to the first `n` positions (a re-slice that diverges after n).
    pub fn truncate(&mut self, n: usize) {
        self.tokens.truncate(n);
        self.addresses.truncate(n);
        for l in self.slices.iter_mut() {
            l.truncate(n);
        }
        self.final_h.truncate(n);
        self.by_address.retain(|&(_, _), &mut p| p < n);
        self.seal = seal_of(&self.tokens);
    }
}
