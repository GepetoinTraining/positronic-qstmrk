// All 65,536 BF16 bit patterns through the kernel's intake, integers only (no float anywhere).
// Rails (spine.tex §0): a reached value is m * 2^e with m in [2^30, 2^31); an exact zero is an event;
// overflow is an error, never a wrap; the sentinel ∅ is not a number and is never read as one.
// Gate: first admission wins; any later input with the same kernel identity is refused and logged.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Class {
    Normal,
    Subnormal,
    Zero,
    Inf,
    NaN,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Rung { neg: bool, m: u32, e: i32 }, // value = ±m * 2^e, m in [2^30, 2^31)
    Zero { neg: bool },                  // exact zero, sign kept (the tryte's signed zero)
    Overflow { neg: bool },              // ±inf: refused, overflow is an error
    Empty,                               // NaN: the sentinel ∅
}

/// Exact decode of one BF16 pattern: sign 1, exponent 8 (bias 127), mantissa 7.
fn decode(b: u16) -> (Class, Kernel) {
    let neg = b >> 15 == 1;
    let ex = (b >> 7 & 0xFF) as i32;
    let f = (b & 0x7F) as u32;
    let (class, big, e) = match (ex, f) {
        (255, 0) => return (Class::Inf, Kernel::Overflow { neg }),
        (255, _) => return (Class::NaN, Kernel::Empty),
        (0, 0) => return (Class::Zero, Kernel::Zero { neg }),
        (0, _) => (Class::Subnormal, f, -126 - 7),       // f * 2^-133
        _ => (Class::Normal, 128 + f, ex - 127 - 7),      // (1.f) * 2^(ex-127)
    };
    let shift = 30 - (31 - big.leading_zeros() as i32); // bring the top bit to 2^30
    (class, Kernel::Rung { neg, m: big << shift, e: e - shift })
}

/// Exact comparison of two rung values (same sign assumed positive): compare by exponent of the top bit, then mantissa.
fn rung_lt(a: (u32, i32), b: (u32, i32)) -> bool {
    (a.1, a.0) < (b.1, b.0) // m normalised to [2^30, 2^31), so the exponent orders first
}

fn main() {
    let decoded: Vec<(u16, Class, Kernel)> = (0..=u16::MAX).map(|b| { let (c, k) = decode(b); (b, c, k) }).collect();

    println!("BF16: all 65,536 patterns, decoded exactly (integer bit operations only)\n");
    let mut by_class: HashMap<Class, usize> = HashMap::new();
    for (_, c, _) in &decoded {
        *by_class.entry(*c).or_default() += 1;
    }
    for c in [Class::Normal, Class::Subnormal, Class::Zero, Class::Inf, Class::NaN] {
        println!("  {:<10} {:>6}", format!("{c:?}"), by_class[&c]);
    }
    println!("  {:<10} {:>6}", "total", decoded.len());

    // checks on the decode itself
    let pos: Vec<(u32, i32)> = decoded.iter().filter_map(|(b, _, k)| match k {
        Kernel::Rung { neg: false, m, e } if *b < 0x8000 => Some((*m, *e)),
        _ => None,
    }).collect();
    let monotone = pos.windows(2).all(|w| rung_lt(w[0], w[1]));
    println!("\n  [ok? {}] positive finite patterns in bit order are strictly increasing in exact value (so no two are equal)", monotone);
    let mirror = decoded.iter().all(|(b, _, k)| match (k, decoded[(*b ^ 0x8000) as usize].2) {
        (Kernel::Rung { neg, m, e }, Kernel::Rung { neg: n2, m: m2, e: e2 }) => *neg != n2 && *m == m2 && *e == e2,
        _ => true,
    });
    println!("  [ok? {}] flipping the sign bit gives exactly the negated value (sign is its own bit)", mirror);
    assert!(monotone && mirror);

    // ---------- the intake gate, two identities
    for (name, signed_zero_kept) in [("value identity (+0 = -0)", false), ("kernel identity (signed zero kept, as the tryte lanes do)", true)] {
        let mut admitted: HashSet<Kernel> = HashSet::new();
        let mut refusals: HashMap<&str, usize> = HashMap::new();
        let mut log: Vec<(u16, &str)> = Vec::new(); // receipt: which pattern was refused, and why
        let mut empty_seen = false;
        for (b, _, k) in &decoded {
            let key = match *k {
                Kernel::Zero { .. } if !signed_zero_kept => Kernel::Zero { neg: false },
                other => other,
            };
            let why = match key {
                Kernel::Overflow { .. } => Some("overflow (inf)"),
                Kernel::Empty => {
                    let first = !empty_seen;
                    empty_seen = true;
                    Some(if first { "∅ (NaN, first)" } else { "∅ (NaN, duplicate of ∅)" })
                }
                _ if admitted.contains(&key) => Some("duplicate value"),
                _ => {
                    admitted.insert(key);
                    None
                }
            };
            if let Some(w) = why {
                *refusals.entry(w).or_default() += 1;
                log.push((*b, w));
            }
        }
        let dup_values: Vec<String> = log.iter().filter(|(_, w)| *w == "duplicate value").map(|(b, _)| format!("0x{b:04X}")).collect();
        let (pos_n, neg_n) = admitted.iter().fold((0, 0), |(p, n), k| match k {
            Kernel::Rung { neg: false, .. } | Kernel::Zero { neg: false } => (p + 1, n),
            _ => (p, n + 1),
        });
        println!("\nGate by {name}:");
        let mut r: Vec<_> = refusals.iter().collect();
        r.sort();
        for (w, c) in r {
            println!("  refused  {w:<26} {c:>6}");
        }
        println!("  duplicate values refused: {} [{}]", dup_values.len(), dup_values.join(", "));
        println!("  admitted (left over as output): {}   = {} with sign bit clear + {} with sign bit set", admitted.len(), pos_n, neg_n);
        println!("  refusal receipts: {} entries; admitted + refused = {}", log.len(), admitted.len() + log.len());
        assert_eq!(admitted.len() + log.len(), 65_536);
    }

    // ---------- capacity of the walk cubes (the 'p/q = p-cube through q's addressing' reading)
    println!("\nWalk-cube capacity for the 65,536 inputs (nonzero states of the p x p x p register cube):");
    for (p, orbits) in [(17u32, "binary clocks"), (19, "1 orbit"), (29, "1 orbit"), (31, "one root"), (37, "(not a Monster prime)"), (41, "2 halves"), (47, "one root")] {
        let states = p * p * p - 1;
        let fits = if states >= 65_536 { format!("holds all 65,536; {} states spare", states - 65_536) } else { format!("short by {}", 65_536 - states) };
        println!("  p = {p:>2} ({orbits:<22}) {states:>7} states  {fits}");
    }
    let half = (41u32 * 41 * 41 - 1) / 2;
    println!("  p = 41 half = {half}: one sign's 32,768 patterns fit in one half with {} spare", half - 32_768);
}
