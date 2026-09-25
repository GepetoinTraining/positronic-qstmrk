pub mod rc;
// Shared exact octonion machinery: the Fano table of t6_octonion.py, the 30 labelings
// of S(3,4,8), unit sets in doubled coordinates, closure. Used by tierb and kernel.

// ---------- 3. octonions: Fano table, the 30 labelings of S(3,4,8), one closed set
// e_i e_{i+1} = e_{i+3} (indices 1..7 mod 7); index 0 is the real unit.
pub const fn fano() -> ([[usize; 8]; 8], [[i32; 8]; 8]) {
    let mut idx = [[0usize; 8]; 8];
    let mut sgn = [[0i32; 8]; 8];
    let mut i = 0;
    while i < 8 {
        idx[0][i] = i;
        sgn[0][i] = 1;
        idx[i][0] = i;
        sgn[i][0] = 1;
        if i > 0 {
            idx[i][i] = 0;
            sgn[i][i] = -1;
        }
        i += 1;
    }
    let mut k = 0;
    while k < 7 {
        let (a, b, c) = (k % 7 + 1, (k + 1) % 7 + 1, (k + 3) % 7 + 1);
        let cyc = [(a, b, c), (b, c, a), (c, a, b)];
        let mut m = 0;
        while m < 3 {
            let (x, y, z) = cyc[m];
            idx[x][y] = z;
            sgn[x][y] = 1;
            idx[y][x] = z;
            sgn[y][x] = -1;
            m += 1;
        }
        k += 1;
    }
    (idx, sgn)
}

pub fn omul_i(idx: &[[usize; 8]; 8], sgn: &[[i32; 8]; 8], u: &[i32; 8], v: &[i32; 8]) -> [i32; 8] {
    let mut w = [0i32; 8];
    for i in 0..8 {
        for j in 0..8 {
            w[idx[i][j]] += sgn[i][j] * u[i] * v[j];
        }
    }
    w
}

/// All 30 Steiner systems S(3,4,8) on {0..7}, each as 14 block bitmasks, sorted.
pub fn steiner_systems() -> Vec<Vec<u8>> {
    // AG(3,2): 4-sets whose labels xor to 0
    let s0: Vec<[usize; 4]> = (0..256u32)
        .filter(|m| m.count_ones() == 4)
        .map(|m| {
            let v: Vec<usize> = (0..8).filter(|i| m >> i & 1 == 1).collect();
            [v[0], v[1], v[2], v[3]]
        })
        .filter(|b| b[0] ^ b[1] ^ b[2] ^ b[3] == 0)
        .collect();
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut perm: Vec<usize> = (0..8).collect();
    permute(&mut perm, 0, &mut |p| {
        let mut s: Vec<u8> = s0.iter().map(|b| b.iter().fold(0u8, |m, &x| m | 1 << p[x])).collect();
        s.sort_unstable();
        out.push(s);
    });
    out.sort();
    out.dedup();
    out
}

fn permute(v: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
    if k == v.len() {
        f(v);
        return;
    }
    for i in k..v.len() {
        v.swap(k, i);
        permute(v, k + 1, f);
        v.swap(k, i);
    }
}

/// Units in doubled coordinates: ±2e_i (16) and all ±1 sign patterns on each block (224).
pub fn unit_set(s: &[u8]) -> Vec<[i32; 8]> {
    let mut u = Vec::with_capacity(240);
    for i in 0..8 {
        for sg in [2, -2] {
            let mut x = [0; 8];
            x[i] = sg;
            u.push(x);
        }
    }
    for &b in s {
        let pos: Vec<usize> = (0..8).filter(|i| b >> i & 1 == 1).collect();
        for signs in 0..16 {
            let mut x = [0; 8];
            for (k, &i) in pos.iter().enumerate() {
                x[i] = if signs >> k & 1 == 1 { -1 } else { 1 };
            }
            u.push(x);
        }
    }
    u
}

/// Product table as u8 indices if the set is closed, else None.
pub fn closed_table(idx: &[[usize; 8]; 8], sgn: &[[i32; 8]; 8], units: &[[i32; 8]]) -> Option<Vec<u8>> {
    let mut t = vec![0u8; 240 * 240];
    for (a, u) in units.iter().enumerate() {
        for (b, v) in units.iter().enumerate() {
            let w = omul_i(idx, sgn, u, v); // = 4 uv; doubled uv = w/2
            if w.iter().any(|x| x % 2 != 0) {
                return None;
            }
            let h: [i32; 8] = w.map(|x| x / 2);
            t[a * 240 + b] = units.iter().position(|x| *x == h)? as u8;
        }
    }
    Some(t)
}

// Compile-time table (lets the f64 multiply in tierb unroll into registers).
pub const FANO: ([[usize; 8]; 8], [[i32; 8]; 8]) = fano();
