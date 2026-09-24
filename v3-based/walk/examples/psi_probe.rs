// Diagnostic: which Hurwitz copies live in the twist's hub, and how does the twist act on them?
use std::collections::HashSet;
use walk::{omul_i, steiner_systems, unit_set, FANO};
type Unit = [i32; 8];
fn lmul(v: &Unit, x: &Unit) -> Option<Unit> {
    let w = omul_i(&FANO.0, &FANO.1, v, x);
    if w.iter().any(|c| c % 2 != 0) { None } else { Some(w.map(|c| c / 2)) }
}
fn p_of(i: usize) -> usize { if i == 0 { 0 } else { let v = 2 * i % 7; if v == 0 { 7 } else { v } } }
fn tw(x: &Unit, fwd: bool) -> Unit { let mut y = [0; 8]; for i in 0..8 { if fwd { y[p_of(i)] = x[i] } else { y[i] = x[p_of(i)] } } y }
fn main() {
    let closed: Vec<Vec<Unit>> = steiner_systems().iter().map(|s| unit_set(s)).filter(|u| {
        let s: HashSet<Unit> = u.iter().copied().collect();
        u.iter().all(|a| u.iter().all(|b| lmul(a, b).map_or(false, |w| s.contains(&w))))
    }).collect();
    for (c, set) in closed.iter().enumerate() {
        let s: HashSet<Unit> = set.iter().copied().collect();
        let fixed = set.iter().all(|x| s.contains(&tw(x, true)));
        let imag: Vec<&Unit> = set.iter().filter(|x| x[0] == 0).collect();
        // Hurwitz copies: I, J pure imaginary, anticommuting, halves in set
        let mut copies = 0; let mut twisted = [0; 4];
        for i in &imag { for j in &imag {
            let Some(k) = lmul(i, j) else { continue };
            if k[0] != 0 || lmul(j, i) != Some(k.map(|c| -c)) { continue; }
            let halves = (0..16).all(|m| { let sg = |b: usize| if m >> b & 1 == 1 { -1 } else { 1 };
                let v: Unit = std::array::from_fn(|q| sg(0) * if q == 0 { 2 } else { 0 } + sg(1) * i[q] + sg(2) * j[q] + sg(3) * k[q]);
                v.iter().all(|q| q % 2 == 0) && s.contains(&v.map(|q| q / 2)) });
            if !halves { continue; }
            copies += 1;
            for (d, fwd) in [(0, true), (1, false)] {
                if tw(i, fwd) == **j && tw(*j, fwd) == k { twisted[d] += 1; }
                if tw(i, fwd) == k && tw(&k, fwd) == **i { twisted[2 + d] += 1; }
            }
        }}
        println!("set {c}: twist-fixed {fixed:<5} | ordered Hurwitz frames (I,J) with halves inside: {copies:>4} | twist I->J->K fwd {} back {} | I->K->J fwd {} back {}",
            twisted[0], twisted[1], twisted[2], twisted[3]);
    }
}
