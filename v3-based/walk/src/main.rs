// The §8 cubic walk of day.pdf: a[n+3] = -a[n+2] + a[n+1] + 2a[n] (mod p),
// on the nonzero p^3 cube. Census of orbit lengths over the Monster primes.
// Exact integers only; the recurrence is the only rule.

const MONSTER_PRIMES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[inline]
fn step(p: u64, (a0, a1, a2): (u64, u64, u64)) -> (u64, u64, u64) {
    // -a2 + a1 + 2a0, kept nonnegative by adding p before subtracting
    let next = (p - a2 + a1 + 2 * a0) % p;
    (a1, a2, next)
}

fn index(p: u64, (a0, a1, a2): (u64, u64, u64)) -> usize {
    ((a0 * p + a1) * p + a2) as usize
}

/// Roots of x^3 + x^2 - x - 2 mod p (the characteristic polynomial).
fn roots(p: u64) -> Vec<u64> {
    (0..p).filter(|&x| (x * x % p * x + x * x + 2 * p - x - 2) % p == 0).collect()
}

fn main() {
    for &p in &MONSTER_PRIMES {
        let n = (p * p * p) as usize;
        let mut seen = vec![false; n];
        seen[0] = true; // the zero state is its own fixed point; excluded
        let mut lengths: Vec<u64> = Vec::new();
        let mut tails = 0u64; // states not on any cycle (only possible when det = 2 ≡ 0, i.e. p = 2)
        for start in 1..n {
            if seen[start] {
                continue;
            }
            let s0 = ((start as u64) / (p * p), (start as u64) / p % p, (start as u64) % p);
            // follow until we revisit a state
            let mut path = vec![start];
            let mut s = s0;
            seen[start] = true;
            loop {
                s = step(p, s);
                let i = index(p, s);
                if seen[i] {
                    if let Some(pos) = path.iter().position(|&q| q == i) {
                        lengths.push((path.len() - pos) as u64);
                        tails += pos as u64;
                    } else {
                        tails += path.len() as u64;
                    }
                    break;
                }
                seen[i] = true;
                path.push(i);
            }
        }
        lengths.sort_unstable();
        let mut hist: Vec<(u64, usize)> = Vec::new();
        for &l in &lengths {
            match hist.last_mut() {
                Some((len, c)) if *len == l => *c += 1,
                _ => hist.push((l, 1)),
            }
        }
        let binary = lengths.iter().all(|l| l.is_power_of_two());
        let h: Vec<String> = hist.iter().map(|(l, c)| format!("{c}x{l}")).collect();
        println!(
            "p={p:>2}  states={:>6}  cycles={:>5}  roots={:?}  binary={}  tails={}  [{}]",
            n - 1,
            lengths.len(),
            roots(p),
            binary,
            tails,
            h.join(" ")
        );
    }
}
