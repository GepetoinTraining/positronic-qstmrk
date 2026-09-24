// CubeKernel, composed: binary -> tryte -> quadryte -> [Hurwitz seam] -> Octavian Catch -> 9-gon -> 7//11 hinge.
// Layer definitions: bundle/documents/doll.tex §2 and day_v2.pdf §5-7. Exact integers throughout.
//
// Couplings between layers that the documents leave as readings are chosen here and marked [C]:
//   [C] the twist (half-hinge 7/11) acts on every layer at once: seats (+1,-1), ring exponent +4,
//       quadryte lanes cycled x->y->z, Octavian coordinates by the order-3 table symmetry p(i) = 2i mod 7,
//       closed set moved by p, 9-gon stepped by (+7,+11).
//   [C] a quarter-turn sigma on a lane also steps the ring exponent by 9 (5 = 2^9, order 4).
//   [C] Lift: an equal-magnitude quadryte state is mapped to a half-integer Hurwitz unit (the [thm]
//       bijection), embedded where the twist cycles i -> j -> k, and left-multiplies the Octavian unit.
//       [run] that embedding is not in the hub: it lies in the three sets of one twist triple.
//   [C] the binary layer's op (Cover) is x -> -x on the Octavian unit: the double-cover sign as its own bit.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::time::Instant;
use walk::{omul_i, steiner_systems, unit_set, FANO};

type Unit = [i32; 8]; // octonion, doubled coordinates (x2 = 2x)
type Quad = [u8; 3]; // three lanes, each in ring order [+1, -1, +0, -0]
type Perm64 = [u8; 64];

const LANE: [&str; 4] = ["+1", "-1", "+0", "-0"];
const N_RUN: usize = 1_000_000;

// ---------- rng (splitmix64; no crates)
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn check(name: &str, ok: bool) {
    println!("  [{}] {name}", if ok { "ok" } else { "FAIL" });
    assert!(ok, "{name}");
}

// ======================= binary, tryte, quadryte
// lane index k in Z4; bit 0 = sign (the binary layer), bit 1 = magnitude is zero.
fn sigma(q: Quad, l: usize, fwd: bool) -> Quad {
    let mut r = q;
    r[l] = (q[l] + if fwd { 1 } else { 3 }) % 4;
    r
}
fn mag_flip(q: Quad) -> Quad {
    q.map(|k| (k + 2) % 4)
}
fn zero_lanes(q: Quad) -> usize {
    q.iter().filter(|&&k| k >= 2).count()
}
fn class_name(q: Quad) -> &'static str {
    ["corner", "edge", "face", "centre"][zero_lanes(q)]
}
fn cycle_lanes(q: Quad, fwd: bool) -> Quad {
    // fwd: lane x -> y -> z -> x
    if fwd { [q[2], q[0], q[1]] } else { [q[1], q[2], q[0]] }
}
fn tryte_view(q: Quad, live: u8) -> String {
    let lane = |l: usize| if live >> l & 1 == 1 { LANE[q[l] as usize] } else { "∅" };
    format!("({},{},{})", lane(0), lane(1), lane(2))
}
fn qidx(q: Quad) -> usize {
    (q[0] as usize * 4 + q[1] as usize) * 4 + q[2] as usize
}
fn qof(i: usize) -> Quad {
    [(i / 16) as u8, (i / 4 % 4) as u8, (i % 4) as u8]
}

// ======================= Hurwitz seam (quaternions, doubled coordinates)
fn hmul(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    let r = [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
    r.map(|x| x / 2) // (2a)(2b) = 4ab -> doubled ab
}
fn hurwitz24() -> Vec<[i32; 4]> {
    let mut h = Vec::new();
    for i in 0..4 {
        for s in [2, -2] {
            let mut x = [0; 4];
            x[i] = s;
            h.push(x);
        }
    }
    for m in 0..16 {
        h.push(std::array::from_fn(|i| if m >> i & 1 == 1 { -1 } else { 1 }));
    }
    h
}
/// [thm] equal-magnitude quadryte states <-> half-integer Hurwitz units:
/// real sign = corner (+) / centre (-), i, j, k signs = lane signs.
fn phi(q: Quad) -> Option<[i32; 4]> {
    match zero_lanes(q) {
        0 | 3 => {
            let r = if zero_lanes(q) == 0 { 1 } else { -1 };
            let s = |k: u8| if k & 1 == 1 { -1 } else { 1 };
            Some([r, s(q[0]), s(q[1]), s(q[2])])
        }
        _ => None,
    }
}

// ---------- the no-go (day_v2 §5 Prop. 2), ported from tryte_hurwitz.py
fn comp(p: &Perm64, q: &Perm64) -> Perm64 {
    std::array::from_fn(|x| p[q[x] as usize])
}
fn inv(p: &Perm64) -> Perm64 {
    let mut r = [0u8; 64];
    for (i, &x) in p.iter().enumerate() {
        r[x as usize] = i as u8;
    }
    r
}
fn nogo() -> (usize, usize, usize, bool) {
    let d4: Vec<[u8; 4]> = (0..4u8)
        .flat_map(|r| [std::array::from_fn(|x| (x as u8 + r) % 4), std::array::from_fn(|x| (4 - x as u8 + r) % 4)])
        .collect();
    let perms3 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let as_perm = |lp: [usize; 3], a: [[u8; 4]; 3]| -> Perm64 {
        std::array::from_fn(|s| {
            let q = qof(s);
            let mut n = [0u8; 3];
            for l in 0..3 {
                n[lp[l]] = a[l][q[l] as usize];
            }
            qidx(n) as u8
        })
    };
    let mut all = Vec::with_capacity(3072);
    for lp in perms3 {
        for a in &d4 {
            for b in &d4 {
                for c in &d4 {
                    all.push(as_perm(lp, [*a, *b, *c]));
                }
            }
        }
    }
    let id: Perm64 = std::array::from_fn(|x| x as u8);
    let f: Perm64 = std::array::from_fn(|s| qidx(mag_flip(qof(s))) as u8);
    let sq: Vec<Perm64> = all.iter().filter(|g| comp(g, g) == f).copied().collect();
    let sqset: HashSet<Perm64> = sq.iter().copied().collect();
    let w: Vec<Perm64> = all.iter().filter(|g| comp(g, &comp(g, g)) == f && comp(g, g) != id).copied().collect();
    let mut hit = false;
    'outer: for w in &w {
        let wi = inv(w);
        for i in &sq {
            let j = comp(w, &comp(i, &wi));
            let k = comp(w, &comp(&j, &wi));
            if sqset.contains(&j)
                && sqset.contains(&k)
                && comp(w, &comp(&k, &wi)) == *i
                && (comp(i, &comp(&j, &k)) == f || comp(&k, &comp(&j, i)) == f)
                && comp(i, &j) != comp(&j, i)
            {
                hit = true;
                break 'outer;
            }
        }
    }
    (all.len(), sq.len(), w.len(), hit)
}

// ======================= Octavian
fn lmul(v: &Unit, x: &Unit) -> Option<Unit> {
    let w = omul_i(&FANO.0, &FANO.1, v, x); // = 4 vx
    if w.iter().any(|c| c % 2 != 0) { None } else { Some(w.map(|c| c / 2)) }
}
fn conj(x: &Unit) -> Unit {
    std::array::from_fn(|i| if i == 0 { x[0] } else { -x[i] })
}
fn neg(x: &Unit) -> Unit {
    x.map(|c| -c)
}
fn norm4(x: &Unit) -> i32 {
    x.iter().map(|c| c * c).sum()
}
/// The order-3 table symmetry p(i) = 2i mod 7 on labels 1..7 (7 fixed), 0 fixed.
fn p_of(i: usize) -> usize {
    if i == 0 { 0 } else { let v = 2 * i % 7; if v == 0 { 7 } else { v } }
}
fn twist_unit(x: &Unit, fwd: bool) -> Unit {
    let mut y = [0; 8];
    for i in 0..8 {
        if fwd { y[p_of(i)] = x[i] } else { y[i] = x[p_of(i)] }
    }
    y
}

struct Octavian {
    closed: Vec<Vec<Unit>>,         // the 7 closed sets, in steiner_systems order
    member: HashMap<Unit, u8>,      // unit -> bitmask of closed sets containing it
    open_member: HashSet<Unit>,     // units of the 23 labelings that do not close
    set_twist: [[usize; 7]; 2],     // closed set index under p (fwd) / p^-1
    hub: usize,
    psi: [Unit; 3],                 // images of i, j, k (twist cycles them i -> j -> k)
    seam: Vec<usize>,               // closed sets containing all 24 embedded Hurwitz units
}

impl Octavian {
    fn build() -> Self {
        let systems = steiner_systems();
        let is_closed = |u: &Vec<Unit>| {
            let s: HashSet<Unit> = u.iter().copied().collect();
            u.iter().all(|a| u.iter().all(|b| lmul(a, b).map_or(false, |w| s.contains(&w))))
        };
        let mut closed_sys = Vec::new();
        let mut closed = Vec::new();
        let mut open_member = HashSet::new();
        for s in &systems {
            let u = unit_set(s);
            if is_closed(&u) {
                closed_sys.push(s.clone());
                closed.push(u);
            } else {
                open_member.extend(u);
            }
        }
        let mut member: HashMap<Unit, u8> = HashMap::new();
        for (c, u) in closed.iter().enumerate() {
            for x in u {
                *member.entry(*x).or_default() |= 1 << c;
            }
        }
        let img = |s: &Vec<u8>, fwd: bool| -> Vec<u8> {
            let map = |i: usize| if fwd { p_of(i) } else { (0..8).find(|&j| p_of(j) == i).unwrap() };
            let mut t: Vec<u8> = s.iter().map(|b| (0..8).filter(|i| b >> i & 1 == 1).fold(0u8, |m, i| m | 1 << map(i))).collect();
            t.sort_unstable();
            t
        };
        let mut set_twist = [[0usize; 7]; 2];
        for (d, fwd) in [true, false].into_iter().enumerate() {
            for c in 0..closed.len() {
                let t = img(&closed_sys[c], fwd);
                set_twist[d][c] = closed_sys.iter().position(|s| *s == t).expect("p maps closed sets to closed sets");
            }
        }
        let hubs: Vec<usize> = (0..closed.len()).filter(|&c| set_twist[0][c] == c).collect();
        let hub = hubs[0];
        // psi: find I with J = p(I), K = p(J), IJ = K and all 16 (±1±I±J±K)/2 in one closed set.
        // [run] none exists in the hub (examples/psi_probe.rs); search every closed set.
        let mut psi = None;
        'search: for set in &closed {
            let s: HashSet<Unit> = set.iter().copied().collect();
            for i in set.iter().filter(|x| x[0] == 0) {
                let j = twist_unit(i, true);
                let k = twist_unit(&j, true);
                if lmul(i, &j) != Some(k) || lmul(&j, &k) != Some(*i) || lmul(&k, i) != Some(j) {
                    continue;
                }
                let halves_in = (0..16).all(|m| {
                    let sg = |b: usize| if m >> b & 1 == 1 { -1 } else { 1 };
                    let v: Unit = std::array::from_fn(|c| sg(0) * if c == 0 { 2 } else { 0 } + sg(1) * i[c] + sg(2) * j[c] + sg(3) * k[c]);
                    v.iter().all(|c| c % 2 == 0) && s.contains(&v.map(|c| c / 2))
                });
                if halves_in {
                    psi = Some([*i, j, k]);
                    break 'search;
                }
            }
        }
        let n_hubs = hubs.len();
        let mut oct = Octavian { closed, member, open_member, set_twist, hub, psi: psi.expect("no twist-compatible Hurwitz copy"), seam: vec![] };
        let h24 = hurwitz24();
        oct.seam = (0..oct.closed.len())
            .filter(|&c| h24.iter().all(|q| oct.member.get(&oct.embed(*q)).map_or(false, |m| m >> c & 1 == 1)))
            .collect();
        assert_eq!(n_hubs, 1);
        oct
    }
    /// Embed a doubled Hurwitz quaternion as an octonion unit via psi.
    fn embed(&self, q: [i32; 4]) -> Unit {
        let [i, j, k] = &self.psi;
        std::array::from_fn(|c| (q[0] * if c == 0 { 2 } else { 0 } + q[1] * i[c] + q[2] * j[c] + q[3] * k[c]) / 2)
    }
}

// ======================= 9-gon, 7//11 hinge, ring mod 13
fn pow_mod(b: u32, e: u32, m: u32) -> u32 {
    (0..e).fold(1, |r, _| r * b % m)
}
fn log2_13(x: u32) -> u32 {
    (0..12).find(|&e| pow_mod(2, e, 13) == x).unwrap()
}
fn seat_gate(seat: (i8, i8), d: i8) -> Option<(i8, i8)> {
    let w = (seat.0 + d, seat.1 - d);
    if (0..=2).contains(&w.0) && (0..=2).contains(&w.1) { Some(w) } else { None }
}

// ======================= the composed kernel
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct State {
    quad: Quad,
    x: Unit,
    set: usize,
    seat: (i8, i8),
    nine: (u8, u8),
    ring: u8, // exponent e of 2^e mod 13
    live: u8, // bit l set: lane l has been written; clear: lane holds the sentinel (empty), never read as a number
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Cover,
    Sigma(usize, bool),
    Mul(Unit),
    Lift(bool),
    Catch(Unit),
    Hinge(bool), // true = 7/11, false = 11/7
    Write(usize, u8), // driver: the lane's first reading at its onset hinge (empty -> ring state k)
    Erase(usize),     // blind inverse of Write only
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ev {
    Done,
    Moved { hub: bool },
    Rung,
    OpenLabeling,
    NoSet,
    NoImage,
    Gate,
    Empty,
    Occupied,
}

#[derive(Clone, Copy)]
struct Receipt {
    op: Op,       // resolved: Lift carries the unit it multiplied by, in Catch form
    applied: bool,
    set_before: u8,
    ev: Ev,
}

struct Kernel {
    oct: Octavian,
}

impl Kernel {
    /// Left-multiply by v and classify where the product lands.
    fn mult(&self, st: &mut State, v: &Unit) -> (bool, Ev) {
        let Some(w) = lmul(v, &st.x) else { return (false, Ev::Rung) };
        let mask = self.oct.member.get(&w).copied().unwrap_or(0);
        if mask >> st.set & 1 == 1 {
            st.x = w;
            (true, Ev::Done)
        } else if mask != 0 {
            st.x = w;
            st.set = mask.trailing_zeros() as usize;
            (true, Ev::Moved { hub: st.set == self.oct.hub })
        } else if self.oct.open_member.contains(&w) {
            (false, Ev::OpenLabeling)
        } else {
            (false, Ev::NoSet)
        }
    }

    fn twist(&self, st: &mut State, fwd: bool) {
        st.ring = (st.ring + if fwd { 4 } else { 8 }) % 12;
        st.quad = cycle_lanes(st.quad, fwd);
        st.live = if fwd { (st.live << 1 | st.live >> 2) & 7 } else { (st.live >> 1 | st.live << 2) & 7 };
        st.x = twist_unit(&st.x, fwd);
        st.set = self.oct.set_twist[!fwd as usize][st.set];
        st.nine = if fwd { ((st.nine.0 + 7) % 9, (st.nine.1 + 11) % 9) } else { ((st.nine.0 + 2) % 9, (st.nine.1 + 7) % 9) };
    }

    fn apply(&self, st: &mut State, op: Op) -> Receipt {
        let set_before = st.set as u8;
        let (applied, ev, op) = match op {
            Op::Cover => {
                st.x = neg(&st.x);
                (true, Ev::Done, op)
            }
            Op::Sigma(l, _) if st.live >> l & 1 == 0 => (false, Ev::Empty, op),
            Op::Sigma(l, fwd) => {
                st.quad = sigma(st.quad, l, fwd);
                st.ring = (st.ring + if fwd { 9 } else { 3 }) % 12;
                (true, Ev::Done, op)
            }
            Op::Write(l, _) if st.live >> l & 1 == 1 => (false, Ev::Occupied, op),
            Op::Write(l, k) => {
                st.quad[l] = k;
                st.live |= 1 << l;
                st.ring = (st.ring + 9 * k) % 12; // k quarter-turns from +1
                (true, Ev::Done, op)
            }
            Op::Erase(l) if st.live >> l & 1 == 0 => (false, Ev::Empty, op),
            Op::Erase(l) => {
                st.ring = (st.ring + 3 * st.quad[l]) % 12;
                st.quad[l] = 0;
                st.live &= !(1 << l);
                (true, Ev::Done, op)
            }
            Op::Mul(u) | Op::Catch(u) => {
                let (a, e) = self.mult(st, &u);
                (a, e, op)
            }
            Op::Lift(_) if st.live != 7 => (false, Ev::Empty, op),
            Op::Lift(fwd) => match phi(st.quad) {
                None => (false, Ev::NoImage, op),
                Some(q) => {
                    let h = self.oct.embed(q);
                    let v = if fwd { h } else { conj(&h) };
                    let (a, e) = self.mult(st, &v);
                    (a, e, Op::Catch(v))
                }
            },
            Op::Hinge(fwd) => match seat_gate(st.seat, if fwd { 1 } else { -1 }) {
                None => (false, Ev::Gate, op),
                Some(s) => {
                    st.seat = s;
                    self.twist(st, fwd);
                    (true, Ev::Done, op)
                }
            },
        };
        Receipt { op, applied, set_before, ev }
    }

    /// Exact inverse of an applied receipt (no gate consulted: the receipt says it happened).
    fn undo(&self, st: &mut State, r: &Receipt) {
        if !r.applied {
            return;
        }
        match r.op {
            Op::Cover => st.x = neg(&st.x),
            Op::Sigma(l, fwd) => {
                st.quad = sigma(st.quad, l, !fwd);
                st.ring = (st.ring + if fwd { 3 } else { 9 }) % 12;
            }
            Op::Mul(v) | Op::Catch(v) => {
                st.x = lmul(&conj(&v), &st.x).expect("conj(v)(vx) = x");
                st.set = r.set_before as usize;
            }
            Op::Lift(_) => unreachable!("Lift is resolved to Catch in the receipt"),
            Op::Write(l, k) => {
                st.ring = (st.ring + 3 * k) % 12;
                st.quad[l] = 0;
                st.live &= !(1 << l);
            }
            Op::Erase(_) => unreachable!("Erase appears only in blind rewinds"),
            Op::Hinge(fwd) => {
                st.seat = (st.seat.0 - if fwd { 1 } else { -1 }, st.seat.1 + if fwd { 1 } else { -1 });
                self.twist(st, !fwd);
            }
        }
    }

    /// Blind inverse: the same op inverted and sent back through the gates, with no log.
    fn blind_inverse(op: Op) -> Op {
        match op {
            Op::Cover => Op::Cover,
            Op::Sigma(l, fwd) => Op::Sigma(l, !fwd),
            Op::Mul(v) => Op::Mul(conj(&v)),
            Op::Catch(v) => Op::Catch(conj(&v)),
            Op::Lift(fwd) => Op::Lift(!fwd),
            Op::Hinge(fwd) => Op::Hinge(!fwd),
            Op::Write(l, _) => Op::Erase(l),
            Op::Erase(_) => unreachable!(),
        }
    }

    fn invariants(&self, st: &State, s0: &State) -> bool {
        let lane_sum = |q: Quad| q.iter().map(|&k| k as i32).sum::<i32>();
        st.seat.0 + st.seat.1 == s0.seat.0 + s0.seat.1
            && (st.nine.0 + st.nine.1) % 9 == (s0.nine.0 + s0.nine.1) % 9
            && norm4(&st.x) == 4
            && self.oct.member.get(&st.x).map_or(false, |m| m >> st.set & 1 == 1)
            // S1 readout: the ring's Z4 part counts quarter-turns, its Z3 part counts twists
            && (st.ring as i32 - s0.ring as i32).rem_euclid(4) == (lane_sum(st.quad) - lane_sum(s0.quad)).rem_euclid(4)
            && (st.ring as i32 - s0.ring as i32).rem_euclid(3) == ((st.seat.0 - s0.seat.0) as i32).rem_euclid(3)
    }

    fn random_op(&self, rng: &mut Rng, st: &State) -> Op {
        match rng.below(10) {
            0 => Op::Cover,
            1 | 2 => Op::Sigma(rng.below(3), true),
            3 | 4 => Op::Mul(self.oct.closed[st.set][rng.below(240)]),
            5 | 6 => Op::Lift(true),
            7 => {
                let others: Vec<usize> = (0..7).filter(|&c| c != st.set).collect();
                let c = others[rng.below(others.len())];
                Op::Catch(self.oct.closed[c][rng.below(240)])
            }
            _ => Op::Hinge(rng.next() & 1 == 1),
        }
    }
}

fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::Cover => "Cover",
        Op::Sigma(..) => "Sigma",
        Op::Mul(_) => "Mul",
        Op::Lift(_) => "Lift",
        Op::Catch(_) => "Catch",
        Op::Hinge(_) => "Hinge",
        Op::Write(..) => "Write",
        Op::Erase(_) => "Erase",
    }
}

fn trace_line(op: Op, r: &Receipt, st: &State, k: &Kernel) -> String {
    let xyz = ['x', 'y', 'z'];
    let name = match op {
        Op::Sigma(l, f) => format!("Sigma{} lane {}", if f { "" } else { "^-1" }, xyz[l]),
        Op::Mul(_) => "Mul (in set)".into(),
        Op::Catch(_) => "Catch (other set)".into(),
        Op::Hinge(f) => (if f { "Hinge 7/11" } else { "Hinge 11/7" }).into(),
        Op::Write(l, q) => format!("Write lane {} = {}", xyz[l], LANE[q as usize]),
        _ => op_kind(&op).into(),
    };
    let ev = format!("{:?}", r.ev).replace("Moved { hub: true }", "caught by hub").replace("Moved { hub: false }", "caught by non-hub");
    format!("{name:<18} {:<18} {}", if r.applied { ev } else { format!("refused: {ev}") }, show(st, k))
}

fn show(st: &State, k: &Kernel) -> String {
    let sign = st.x.iter().find(|&&c| c != 0).map_or('?', |&c| if c < 0 { '-' } else { '+' });
    format!(
        "tryte {:<10} {:<6} | cover {} | oct set {}{} | seats {:?} | 9-gon {:?} | ring 2^{}={}",
        tryte_view(st.quad, st.live),
        if st.live == 7 { class_name(st.quad) } else { "-" },
        sign,
        st.set,
        if st.set == k.oct.hub { "(hub)" } else { "     " },
        st.seat,
        st.nine,
        st.ring,
        pow_mod(2, st.ring as u32, 13)
    )
}

// ======================= the driver: hinge_test.py's onset, one oscillator per lane
// Floats live here only (the analog side). The kernel receives integers: a quadrant, then its steps.
const SPC: usize = 100; // samples per cycle
const CYCLES: usize = 70;
const N_S: usize = SPC * CYCLES;
const ON: usize = 10 * SPC; // 10 cycles below threshold, then the drive crosses it
const Q: usize = SPC / 4; // quarter turn = the i, as a sample offset
const SIGMAS: [f64; 2] = [1e-3, 3e-2]; // process noise levels, as in hinge_test.py
const MEAS: f64 = 1e-4; // instrument noise on the shadow
const TRIALS: usize = 400;

#[cfg(windows)]
#[link(name = "advapi32")]
unsafe extern "system" {
    #[link_name = "SystemFunction036"]
    fn rtl_gen_random(buf: *mut u8, len: u32) -> u8;
}
fn os_entropy(buf: &mut [u8]) {
    #[cfg(windows)]
    unsafe {
        assert!(rtl_gen_random(buf.as_mut_ptr(), buf.len() as u32) != 0)
    }
    #[cfg(not(windows))]
    {
        use std::io::Read;
        std::fs::File::open("/dev/urandom").unwrap().read_exact(buf).unwrap();
    }
}
/// n standard normals from kernel entropy, Box-Muller as in hinge_test.py (cos branch).
fn urandom_normal(n: usize) -> Vec<f64> {
    let mut b = vec![0u8; 16 * n];
    os_entropy(&mut b);
    let u = |i: usize| (u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()) as f64 + 0.5) / 2f64.powi(64);
    (0..n).map(|i| (-2.0 * u(i).ln()).sqrt() * (2.0 * PI * u(n + i)).cos()).collect()
}
/// Quadrant of a+ib as a lane state: 1, i, -1, -i -> +1, -1, +0, -0 (sigma = i).
fn quadrant(re: f64, im: f64) -> u8 {
    match (re >= 0.0, im >= 0.0) {
        (true, true) => 0,
        (false, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
    }
}
/// Stuart-Landau onset, dz = [(mu + i w) z - |z|^2 z] dt + sigma dW, per lane. Returns (z, shadow x).
fn onset(sigma: f64) -> Vec<(Vec<(f64, f64)>, Vec<f64>)> {
    let (dt, w) = (1.0 / SPC as f64, 2.0 * PI);
    let noise = urandom_normal(3 * 3 * N_S);
    (0..3)
        .map(|l| {
            let nz = &noise[l * 3 * N_S..(l + 1) * 3 * N_S];
            let (mut zr, mut zi) = (0.0f64, 0.0f64);
            let mut z = Vec::with_capacity(N_S);
            let mut x = Vec::with_capacity(N_S);
            let s = sigma * (dt / 2.0).sqrt();
            for k in 0..N_S {
                let mu = if k < ON { -0.5 } else { 0.5 };
                let r2 = zr * zr + zi * zi;
                let (dr, di) = ((mu * zr - w * zi - r2 * zr) * dt, (mu * zi + w * zr - r2 * zi) * dt);
                zr += dr + s * nz[3 * k];
                zi += di + s * nz[3 * k + 1];
                z.push((zr, zi));
                x.push(zr + MEAS * nz[3 * k + 2]);
            }
            (z, x)
        })
        .collect()
}

#[derive(Default)]
struct Drive {
    hinge_lag: Vec<i64>,   // read hinge - true hinge, samples, per lane
    write_agree: usize,    // written quadrant == true z's quadrant at that sample
    ticks: [usize; 3],     // sigma forward, sigma back, two-quadrant jump (not applied)
    track: (usize, usize), // (lane == true quadrant, samples) after each lane's hinge
    lift: HashMap<Ev, usize>,
    trials_lifting: usize,
    broken: usize,
    rewound: usize,
    ops: usize,
}

fn drive(k: &Kernel, d: &mut Drive, sigma: f64, trace: bool) {
    let sim = onset(sigma);
    let hinge_amp = 10.0 * sigma; // hinge = clear of the noise floor
    let s0 = State { quad: [0; 3], x: [2, 0, 0, 0, 0, 0, 0, 0], set: k.oct.seam[0], seat: (1, 1), nine: (0, 0), ring: 0, live: 0 };
    let mut st = s0;
    let mut log: Vec<Receipt> = Vec::new();
    let mut lifted = false;
    let mut shown = 0;
    let mut step = |st: &mut State, op: Op, ks: usize, log: &mut Vec<Receipt>, d: &mut Drive| {
        let r = k.apply(st, op);
        if trace && (shown < 14 || (matches!(op, Op::Lift(_)) && r.applied && shown < 18)) {
            println!("  cycle {:>6.2} {}", ks as f64 / SPC as f64, trace_line(op, &r, st, k));
            shown += 1;
        }
        d.broken += !k.invariants(st, &s0) as usize;
        log.push(r);
        r
    };
    for ks in Q..N_S {
        for l in 0..3 {
            let (z, x) = (&sim[l].0, &sim[l].1);
            let (re, im) = (x[ks], x[ks - Q]); // the quarter-turn lookup: no transform, a memory read
            let q = quadrant(re, im);
            if st.live >> l & 1 == 0 {
                if re * re + im * im > hinge_amp * hinge_amp {
                    let kt = z.iter().position(|(a, b)| a * a + b * b > hinge_amp * hinge_amp).unwrap_or(N_S);
                    d.hinge_lag.push(ks as i64 - kt as i64);
                    d.write_agree += (quadrant(z[ks].0, z[ks].1) == q) as usize;
                    step(&mut st, Op::Write(l, q), ks, &mut log, d);
                }
                continue;
            }
            match (q + 4 - st.quad[l]) % 4 {
                0 => {}
                2 => d.ticks[2] += 1,
                delta => {
                    d.ticks[if delta == 1 { 0 } else { 1 }] += 1;
                    step(&mut st, Op::Sigma(l, delta == 1), ks, &mut log, d);
                    let r = step(&mut st, Op::Lift(true), ks, &mut log, d);
                    *d.lift.entry(r.ev).or_default() += 1;
                    lifted |= r.applied;
                }
            }
            d.track.0 += (quadrant(z[ks].0, z[ks].1) == st.quad[l]) as usize;
            d.track.1 += 1;
        }
    }
    d.trials_lifting += lifted as usize;
    d.ops += log.len();
    if trace {
        println!("  ... {} ops in this trial; end: {}", log.len(), show(&st, k));
    }
    for r in log.iter().rev() {
        k.undo(&mut st, r);
    }
    d.rewound += (st == s0) as usize;
}

fn main() {
    println!("CubeKernel — layer checks (each reproduces a documented claim or tests a coupling)\n");

    // ---------- binary / tryte / quadryte
    println!("binary, tryte, quadryte:");
    let all_q: Vec<Quad> = (0..64).map(qof).collect();
    let classes: Vec<usize> = (0..4).map(|z| all_q.iter().filter(|q| zero_lanes(**q) == z).count()).collect();
    check(&format!("4^3 = 64 states: corners/edges/faces/centre = {classes:?} (doc: 8/24/24/8)"), classes == [8, 24, 24, 8]);
    check("sigma^2 = magnitude flip on every lane (each lane is {1,i,-1,-i})",
        all_q.iter().all(|&q| (0..3).all(|l| sigma(sigma(q, l, true), l, true)[l] == mag_flip(q)[l])));
    check("magnitude flip is an involution; swaps corner<->centre, edge<->face",
        all_q.iter().all(|&q| mag_flip(mag_flip(q)) == q && zero_lanes(mag_flip(q)) == 3 - zero_lanes(q)));
    let positions: HashSet<[i8; 3]> = all_q.iter().map(|q| q.map(|k| if k >= 2 { 0 } else if k == 0 { 1 } else { -1 })).collect();
    check(&format!("tryte positions ignoring zero sign: {} = 26 faces + the empty centre", positions.len()), positions.len() == 27);
    check("sigma on different lanes commutes (the quadryte is abelian)",
        all_q.iter().all(|&q| sigma(sigma(q, 0, true), 1, true) == sigma(sigma(q, 1, true), 0, true)));

    // ---------- Hurwitz seam
    println!("Hurwitz seam:");
    let h24 = hurwitz24();
    let hset: HashSet<[i32; 4]> = h24.iter().copied().collect();
    check("24 Hurwitz units closed under multiplication",
        h24.iter().all(|a| h24.iter().all(|b| hset.contains(&hmul(*a, *b)))));
    let eq_mag: Vec<Quad> = all_q.iter().copied().filter(|q| phi(*q).is_some()).collect();
    let imgs: HashSet<[i32; 4]> = eq_mag.iter().map(|q| phi(*q).unwrap()).collect();
    check(&format!("{} equal-magnitude states biject with the 16 half-integer units", eq_mag.len()),
        eq_mag.len() == 16 && imgs.len() == 16 && imgs.iter().all(|h| h.iter().all(|c| c.abs() == 1)));
    check("magnitude flip becomes q -> -conj(q) under the bijection",
        eq_mag.iter().all(|&q| { let h = phi(q).unwrap(); phi(mag_flip(q)) == Some([-h[0], h[1], h[2], h[3]]) }));
    let t = Instant::now();
    let (n_all, n_sq, n_w, hit) = nogo();
    check(&format!("no-go: no 2T with -1 = flip among all {n_all} lane symmetries ({n_sq} square to flip, {n_w} cube to it; {:.1}s)",
        t.elapsed().as_secs_f64()), n_all == 3072 && !hit);

    // ---------- Octavian
    println!("Octavian Catch:");
    let t = Instant::now();
    let oct = Octavian::build();
    check(&format!("7 closed of 30 labelings; unique hub of the twist p(i)=2i mod 7 is set {} ({:.1}s)",
        oct.hub, t.elapsed().as_secs_f64()), oct.closed.len() == 7);
    let auto = (0..8).all(|i| (0..8).all(|j| {
        FANO.0[p_of(i)][p_of(j)] == p_of(FANO.0[i][j]) && FANO.1[p_of(i)][p_of(j)] == FANO.1[i][j]
    }));
    check("p is a sign-exact symmetry of the table, order 3", auto && (0..8).all(|i| p_of(p_of(p_of(i))) == i));
    let cyc: Vec<usize> = (0..7).map(|c| {
        let (mut n, mut d) = (1, oct.set_twist[0][c]);
        while d != c { d = oct.set_twist[0][d]; n += 1; }
        n
    }).collect();
    let mut split = cyc.clone();
    split.sort();
    check(&format!("twist splits the closed seven as 1+3+3 (cycle lengths {split:?}; T7 H1)"), split == [1, 3, 3, 3, 3, 3, 3]);
    let hub = &oct.closed[oct.hub];
    let hidx: HashMap<Unit, usize> = hub.iter().enumerate().map(|(i, u)| (*u, i)).collect();
    let tab: Vec<u8> = hub.iter().flat_map(|a| hub.iter().map(|b| hidx[&lmul(a, b).unwrap()] as u8).collect::<Vec<_>>()).collect();
    let latin = (0..240).all(|a| (0..240).map(|b| tab[a * 240 + b]).collect::<HashSet<_>>().len() == 240
        && (0..240).map(|b| tab[b * 240 + a]).collect::<HashSet<_>>().len() == 240);
    check("hub's 240x240 table is a Latin square (unique addressing)", latin);
    let bracket = (0..240).map(|u| (0..240).filter(|&v| {
        let uv = tab[u * 240 + v] as usize;
        (0..240).any(|x| tab[u * 240 + tab[v * 240 + x] as usize] != tab[uv * 240 + x])
    }).count()).sum::<usize>();
    println!("  [run] pairs (u,v) where applying v then u differs from the single unit uv: {bracket} of 57600 (the bracket must be kept)");
    check("rewind by conjugate is exact on the hub: conj(u)(ux) = x for all u, x",
        hub.iter().all(|u| hub.iter().all(|x| lmul(&conj(u), &lmul(u, x).unwrap()) == Some(*x))));
    let psi_hom = h24.iter().all(|a| h24.iter().all(|b| lmul(&oct.embed(*a), &oct.embed(*b)) == Some(oct.embed(hmul(*a, *b)))));
    check(&format!("Hurwitz embeds as a subring, multiplication preserved (I, J, K = {:?}, {:?}, {:?})", oct.psi[0], oct.psi[1], oct.psi[2]), psi_hom);
    let seam_cycle = oct.seam.len() == 3 && oct.seam.iter().all(|&c| oct.set_twist[0][c] != c && oct.seam.contains(&oct.set_twist[0][c]));
    println!("  [run] the twist-compatible Hurwitz copy is not in the hub; its 24 units lie in sets {:?}, all three of one twist triple: {seam_cycle}", oct.seam);
    assert!(seam_cycle && !oct.seam.contains(&oct.hub));
    check("seam is twist-equivariant: embed(phi(cycle lanes)) = p(embed(phi(q))) on all 16",
        eq_mag.iter().all(|&q| oct.embed(phi(cycle_lanes(q, true)).unwrap()) == twist_unit(&oct.embed(phi(q).unwrap()), true)));

    // ---------- 9-gon
    println!("9-gon:");
    let mut seen = HashSet::new();
    let mut loops = Vec::new();
    for s in 0..81u32 {
        if seen.contains(&s) { continue; }
        let (mut x, mut y, mut n) = (s / 9, s % 9, 0);
        let inv = (x + y) % 9;
        loop {
            seen.insert(x * 9 + y);
            x = (x + 7) % 9; y = (y + 11) % 9; n += 1;
            assert_eq!((x + y) % 9, inv);
            if x * 9 + y == s { break; }
        }
        loops.push(n);
    }
    check(&format!("walk (x+7, y+11) on Z9^2: {} loops of {:?}, x+y conserved", loops.len(), loops.iter().collect::<HashSet<_>>()),
        loops.len() == 9 && loops.iter().all(|&n| n == 9));

    // ---------- 7//11 hinge and the ring mod 13
    println!("7//11 hinge, ring mod 13:");
    let (l7, l11, l5) = (log2_13(7), log2_13(11), log2_13(5));
    let twist_e = (l7 + 12 - l11) % 12;
    check(&format!("7 = 2^{l7}, 11 = 2^{l11}, 7*11 = {} = -1, 7/11 = 2^{twist_e} (order 3), 5 = 2^{l5} (order 4)", 7 * 11 % 13),
        l7 == 11 && l11 == 7 && 7 * 11 % 13 == 12 && twist_e == 4 && l5 == 9);
    let seats: Vec<(i8, i8)> = (0..3).flat_map(|a| (0..3).map(move |b| (a, b))).collect();
    let unpaired = |first: i8| seats.iter().filter(|&&s| {
        let a = seat_gate(s, first).unwrap_or(s);
        seat_gate(a, -first).unwrap_or(a) == s
    }).count();
    let paired = |first: i8| seats.iter().filter(|&&s| {
        let out = seat_gate(s, first).and_then(|a| seat_gate(a, -first)).unwrap_or(s); // both or neither
        out == s
    }).count();
    check(&format!("unpaired hinge closes on {} of 9 seat states starting 7/11 ({} starting 11/7); doc: 6 of 9",
        unpaired(1), unpaired(-1)), unpaired(1) == 6 && unpaired(-1) == 6);
    check(&format!("pair-gated hinge closes on {} of 9 (doll.tex registration 1: kill any drift)", paired(1).min(paired(-1))),
        paired(1) == 9 && paired(-1) == 9);
    let k = Kernel { oct };
    let s0 = State { quad: [0, 0, 0], x: [2, 0, 0, 0, 0, 0, 0, 0], set: k.oct.seam[0], seat: (1, 1), nine: (0, 0), ring: 0, live: 7 };
    let mut sq = s0;
    let mut sw = s0;
    k.apply(&mut sq, Op::Sigma(0, true));
    k.apply(&mut sq, Op::Hinge(true));
    k.apply(&mut sw, Op::Hinge(true));
    k.apply(&mut sw, Op::Sigma(0, true));
    check("the twist does not commute with a quarter-turn (the non-commuting tier is reached)", sq != sw);
    let mut s3 = s0;
    for _ in 0..3 { k.twist(&mut s3, true); }
    check("three twists return every layer except the 9-gon, which has moved 3 of its 9 steps",
        s3.quad == s0.quad && s3.x == s0.x && s3.set == s0.set && s3.ring == s0.ring && s3.nine == (3, 6));

    // ======================= the composed run
    println!("\nComposed kernel — first 16 ops of a random program (start: {})", show(&s0, &k));
    let mut rng = Rng(2026);
    let mut st = s0;
    let mut log: Vec<Receipt> = Vec::with_capacity(N_RUN);
    let mut ops: Vec<Op> = Vec::with_capacity(N_RUN);
    let mut broken = 0usize;
    let t = Instant::now();
    for n in 0..N_RUN {
        let op = k.random_op(&mut rng, &st);
        let r = k.apply(&mut st, op);
        if n < 16 {
            println!("  {n:>2} {}", trace_line(op, &r, &st, &k));
        }
        if !k.invariants(&st, &s0) { broken += 1; }
        log.push(r);
        ops.push(op);
    }
    let fwd_ns = t.elapsed().as_nanos() as f64 / N_RUN as f64;
    let end = st;

    let mut tally: HashMap<(&str, Ev), usize> = HashMap::new();
    for (op, r) in ops.iter().zip(&log) {
        let name = op_kind(op);
        *tally.entry((name, r.ev)).or_default() += 1;
    }
    let mut rows: Vec<_> = tally.into_iter().collect();
    rows.sort_by_key(|((n, e), _)| (*n, format!("{e:?}")));
    println!("\n{N_RUN} ops, {fwd_ns:.0} ns/op forward (with invariant checks). Events:");
    for ((n, e), c) in rows {
        println!("  {n:<6} {:<24} {c:>7}", format!("{e:?}"));
    }
    check(&format!("invariants held on every step (seat sum, 9-gon x+y, |x|=1, x in its set, ring = Z4 quarter-turns x Z3 twists): {} violations", broken), broken == 0);

    // rewind with receipts
    let t = Instant::now();
    for r in log.iter().rev() { k.undo(&mut st, r); }
    let back_ns = t.elapsed().as_nanos() as f64 / N_RUN as f64;
    check(&format!("rewind with receipts over {N_RUN} ops returns bit-identical ({back_ns:.0} ns/op)"), st == s0);
    println!("  receipt: {} bytes as stored; the minimum it must carry is 1 applied bit + 3 bits of set (+ the op stream)",
        std::mem::size_of::<Receipt>());

    // blind rewind: same ops inverted, back through the gates, no log (A1 shape: 1000 sequences x 200 ops)
    let (mut ok_log, mut drift_blind) = (0, 0);
    for _ in 0..1000 {
        let start = State { quad: qof(rng.below(64)), x: k.oct.closed[k.oct.seam[0]][rng.below(240)], set: k.oct.seam[0],
            seat: [(0, 2), (1, 1), (2, 0)][rng.below(3)], nine: (rng.below(9) as u8, rng.below(9) as u8), ring: 0, live: 7 };
        let mut s = start;
        let mut seq = Vec::new();
        for _ in 0..200 {
            let op = k.random_op(&mut rng, &s);
            seq.push((op, k.apply(&mut s, op)));
        }
        let mut a = s;
        for (_, r) in seq.iter().rev() { k.undo(&mut a, r); }
        ok_log += (a == start) as usize;
        let mut b = s;
        for (op, _) in seq.iter().rev() { k.apply(&mut b, Kernel::blind_inverse(*op)); }
        drift_blind += (b != start) as usize;
    }
    check(&format!("1000 random programs of 200 ops: with receipts {ok_log}/1000 return exactly"), ok_log == 1000);
    println!("  [run] without receipts (inverse ops sent back through the gates): {drift_blind}/1000 drift");
    println!("\nend state after {N_RUN} ops: {}", show(&end, &k));
    // ======================= driven by the onset (hinge_test.py), one oscillator per lane
    println!("\nDriven kernel — three Stuart-Landau onsets (noise from OS entropy), read by the quarter-turn lookup");
    println!("trial 1 (lanes hold ∅ until their hinge; each quarter-turn read is a sigma, then a Lift is offered):");
    for (n, &sigma) in SIGMAS.iter().enumerate() {
        let mut d = Drive::default();
        let t = Instant::now();
        for trial in 0..TRIALS {
            drive(&k, &mut d, sigma, n == 0 && trial == 0);
        }
        let secs = t.elapsed().as_secs_f64();
        let mut lag: Vec<i64> = d.hinge_lag.iter().map(|x| x.abs()).collect();
        lag.sort();
        let n_l = d.hinge_lag.len();
        println!("
sigma = {sigma:e}: {TRIALS} trials x 3 lanes ({secs:.1}s, {} kernel ops)", d.ops);
        println!("  [run] read hinge vs true hinge: median |lag| {} samples (hinge_test: ~8-10), max {}", lag[n_l / 2], lag[n_l - 1]);
        println!("  [run] quadrant written at the hinge equals the true object's quadrant: {}/{}", d.write_agree, n_l);
        println!("  [run] quarter-turn ticks: {} forward, {} back, {} two-quadrant jumps (not applied); {:.3} net per lane-cycle after each hinge",
            d.ticks[0], d.ticks[1], d.ticks[2], (d.ticks[0] as f64 - d.ticks[1] as f64) / (d.track.1 as f64 / SPC as f64));
        println!("  [run] lane equals the true quadrant on {:.2}% of post-hinge samples", 100.0 * d.track.0 as f64 / d.track.1 as f64);
        let mut lr: Vec<_> = d.lift.iter().collect();
        lr.sort_by_key(|(e, _)| format!("{e:?}"));
        println!("  [run] Lift after each tick: {}; trials where it ever fired: {}/{TRIALS}",
            lr.iter().map(|(e, c)| format!("{e:?} {c}")).collect::<Vec<_>>().join(", "), d.trials_lifting);
        check(&format!("invariants on every driven op: {} violations", d.broken), d.broken == 0);
        check(&format!("every trial rewinds from its receipts to all-∅: {}/{TRIALS}", d.rewound), d.rewound == TRIALS);
    }
}
