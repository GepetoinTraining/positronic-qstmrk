//! `card_ops.rs` — V5's primitives and the engine's ops on the card, ported to WGSL exactly
//! (LOG §29, stage three).
//!
//! Everything the engine does between and inside the matmuls — `bf16_mul`, `bf16_add`,
//! `bf16_lt`, the F³ point (`bf16_mul_ext`, `bf16_mul_q30_ext`, `f3_add`, `f3_to_bf16`,
//! `single_bits_to_f3`, `f3_div_to_bf16`), and on them RMSNorm, RoPE (Q.30 chart + phase
//! interpolation), the exp-LUT softmax, SiLU, the exact add, the cell decomposition, the bucket
//! with its nine-limb lane and its collapse (round-to-nearest-even with sticky, refusal flagged),
//! the projection over the grid ids and the dense bucket for attention — in u32 only, with the
//! 64-bit spots as two-word emulations.
//!
//! One binding set for every kernel (P uniform + B1..B8 storage + ERR), so the same kernels serve
//! the probe (the gates below, element for element against V5 and the CPU kernels) and the layer
//! pipeline (`card_layer.rs`).

use wgpu::util::DeviceExt;

pub const ACT_FLOOR: i32 = 90;

/// The primitives (bf16.rs), verbatim in u32.
pub const LIB: &str = r#"
struct F3 { sign: u32, exp: i32, mant: u32, sticky: u32 };

fn f3_zero(sign: u32) -> F3 { return F3(sign, 0, 0u, 0u); }

fn bf16_mul(a: u32, b: u32) -> u32 {
    let sign = (a ^ b) & 0x8000u;
    if ((a & 0x7FFFu) == 0u || (b & 0x7FFFu) == 0u) { return sign; }
    let a_exp = i32((a >> 7u) & 0xFFu);
    let b_exp = i32((b >> 7u) & 0xFFu);
    let a_mant = (a & 0x7Fu) | 0x80u;
    let b_mant = (b & 0x7Fu) | 0x80u;
    let prod = a_mant * b_mant;
    var mant_kept: u32; var exp_adj: i32; var rbp: u32;
    if (prod >= 0x8000u) { mant_kept = prod >> 8u; exp_adj = 1; rbp = 7u; } else { mant_kept = prod >> 7u; exp_adj = 0; rbp = 6u; }
    let round_bit = (prod >> rbp) & 1u;
    let sticky = (prod & ((1u << rbp) - 1u)) != 0u;
    let lsb = mant_kept & 1u;
    let round_up = round_bit == 1u && (sticky || lsb == 1u);
    var mant_final = mant_kept + select(0u, 1u, round_up);
    var final_exp = a_exp + b_exp - 127 + exp_adj;
    if ((mant_final & 0x100u) != 0u) { mant_final = mant_final >> 1u; final_exp = final_exp + 1; }
    if (final_exp >= 0xFF) { return sign | 0x7F80u; }
    if (final_exp <= 0) { return sign; }
    return sign | (u32(final_exp) << 7u) | (mant_final & 0x7Fu);
}

fn bf16_add(a: u32, b: u32) -> u32 {
    if ((a & 0x7FFFu) == 0u) { return b; }
    if ((b & 0x7FFFu) == 0u) { return a; }
    let a_sign = (a >> 15u) & 1u;
    let b_sign = (b >> 15u) & 1u;
    let a_exp = i32((a >> 7u) & 0xFFu);
    let b_exp = i32((b >> 7u) & 0xFFu);
    let a_mant = (a & 0x7Fu) | 0x80u;
    let b_mant = (b & 0x7Fu) | 0x80u;
    var la_sign: u32; var la_exp: i32; var la_mant: u32; var sm_sign: u32; var sm_mant: u32; var exp_diff: u32;
    if (a_exp > b_exp || (a_exp == b_exp && a_mant >= b_mant)) {
        la_sign = a_sign; la_exp = a_exp; la_mant = a_mant; sm_sign = b_sign; sm_mant = b_mant; exp_diff = u32(a_exp - b_exp);
    } else {
        la_sign = b_sign; la_exp = b_exp; la_mant = b_mant; sm_sign = a_sign; sm_mant = a_mant; exp_diff = u32(b_exp - a_exp);
    }
    let large = la_mant << 3u;
    let small_full = sm_mant << 3u;
    var small_shifted: u32; var sticky_align: bool;
    if (exp_diff >= 32u) { small_shifted = 0u; sticky_align = small_full != 0u; }
    else if (exp_diff == 0u) { small_shifted = small_full; sticky_align = false; }
    else { let mask = (1u << exp_diff) - 1u; small_shifted = small_full >> exp_diff; sticky_align = (small_full & mask) != 0u; }
    let small = small_shifted | select(0u, 1u, sticky_align);
    var mant: u32;
    if (la_sign == sm_sign) { mant = large + small; } else { mant = large - small; }
    let result_sign = la_sign;
    if (mant == 0u) { return 0u; }
    var exp = la_exp;
    if (mant >= (1u << 11u)) { let lost = mant & 1u; mant = (mant >> 1u) | lost; exp = exp + 1; }
    loop {
        if ((mant & (1u << 10u)) != 0u) { break; }
        mant = mant << 1u; exp = exp - 1;
        if (exp < 1) { return result_sign << 15u; }
    }
    let round_bit = (mant >> 2u) & 1u;
    let sticky = (mant & 3u) != 0u;
    let lsb = (mant >> 3u) & 1u;
    let round_up = round_bit == 1u && (sticky || lsb == 1u);
    var mant_kept = mant >> 3u;
    if (round_up) { mant_kept = mant_kept + 1u; if ((mant_kept & 0x100u) != 0u) { mant_kept = mant_kept >> 1u; exp = exp + 1; } }
    if (exp >= 0xFF) { return (result_sign << 15u) | 0x7F80u; }
    if (exp < 1) { return result_sign << 15u; }
    return (result_sign << 15u) | (u32(exp) << 7u) | (mant_kept & 0x7Fu);
}

fn bf16_sub(a: u32, b: u32) -> u32 { return bf16_add(a, b ^ 0x8000u); }
fn bf16_key(u: u32) -> i32 { if ((u & 0x8000u) != 0u) { return -i32(u & 0x7FFFu) - 1; } return i32(u); }
fn bf16_lt(a: u32, b: u32) -> bool { return bf16_key(a) < bf16_key(b); }

fn bf16_mul_ext(a: u32, b: u32) -> F3 {
    let sign = ((a ^ b) >> 15u) & 1u;
    if ((a & 0x7FFFu) == 0u || (b & 0x7FFFu) == 0u) { return f3_zero(sign); }
    let a_exp = i32((a >> 7u) & 0xFFu);
    let b_exp = i32((b >> 7u) & 0xFFu);
    let prod = ((a & 0x7Fu) | 0x80u) * ((b & 0x7Fu) | 0x80u);
    if (prod >= 0x8000u) { return F3(sign, a_exp + b_exp - 127 + 1, prod << 16u, 0u); }
    return F3(sign, a_exp + b_exp - 127, prod << 17u, 0u);
}

fn mul_8x31(a: u32, q: u32) -> vec2<u32> {
    let lo_part = a * (q & 0xFFFFu);
    let hi_part = a * (q >> 16u);
    let t = hi_part + (lo_part >> 16u);
    let lo = (t << 16u) | (lo_part & 0xFFFFu);
    let hi = t >> 16u;
    return vec2<u32>(hi, lo);
}

fn bf16_mul_q30_ext(a: u32, q: i32) -> F3 {
    let sign_a = (a >> 15u) & 1u;
    let sign_q = select(0u, 1u, q < 0);
    let sign = sign_a ^ sign_q;
    if ((a & 0x7FFFu) == 0u || q == 0) { return f3_zero(sign); }
    let exp_a = i32((a >> 7u) & 0xFFu);
    let mant_a = (a & 0x7Fu) | 0x80u;
    let q_mag = u32(abs(q));
    let p = mul_8x31(mant_a, q_mag);
    var k: u32;
    if (p.x != 0u) { k = 32u + (31u - countLeadingZeros(p.x)); } else { k = 31u - countLeadingZeros(p.y); }
    var mant_norm: u32; var sticky: bool;
    if (k >= 31u) {
        let shift = k - 31u;
        if (shift == 0u) { mant_norm = p.y; sticky = false; }
        else { mant_norm = (p.y >> shift) | (p.x << (32u - shift)); sticky = (p.y & ((1u << shift) - 1u)) != 0u; }
    } else {
        mant_norm = p.y << (31u - k); sticky = false;
    }
    return F3(sign, exp_a + i32(k) - 37, mant_norm, select(0u, 1u, sticky));
}

fn f3_to_bf16(p: F3) -> u32 {
    if (p.mant == 0u) { return 0u; }
    let round_bit = (p.mant >> 23u) & 1u;
    let intra = (p.mant & ((1u << 23u) - 1u)) != 0u;
    let sticky = intra || p.sticky != 0u;
    let lsb = (p.mant >> 24u) & 1u;
    let round_up = round_bit == 1u && (sticky || lsb == 1u);
    var mant8 = p.mant >> 24u;
    var exp = p.exp;
    if (round_up) { mant8 = mant8 + 1u; if ((mant8 & 0x100u) != 0u) { mant8 = mant8 >> 1u; exp = exp + 1; } }
    if (exp >= 0xFF) { return (p.sign << 15u) | 0x7F80u; }
    if (exp < 1) { return p.sign << 15u; }
    return (p.sign << 15u) | (u32(exp) << 7u) | (mant8 & 0x7Fu);
}

fn f3_add(p: F3, q: F3) -> F3 {
    if (p.mant == 0u) { return F3(q.sign, q.exp, q.mant, select(0u, 1u, p.sticky != 0u || q.sticky != 0u)); }
    if (q.mant == 0u) { return F3(p.sign, p.exp, p.mant, select(0u, 1u, p.sticky != 0u || q.sticky != 0u)); }
    var la: F3; var sm: F3;
    if (p.exp > q.exp || (p.exp == q.exp && p.mant >= q.mant)) { la = p; sm = q; } else { la = q; sm = p; }
    let exp_diff = u32(la.exp - sm.exp);
    var sm_m: u32; var sticky_align: bool;
    if (exp_diff >= 32u) { sm_m = 0u; sticky_align = sm.mant != 0u; }
    else if (exp_diff == 0u) { sm_m = sm.mant; sticky_align = false; }
    else { sm_m = sm.mant >> exp_diff; sticky_align = (sm.mant & ((1u << exp_diff) - 1u)) != 0u; }
    var sticky = sticky_align || la.sticky != 0u || sm.sticky != 0u;
    var mant: u32; var sign: u32; var carry = false;
    if (la.sign == sm.sign) { mant = la.mant + sm_m; carry = mant < la.mant; sign = la.sign; }
    else if (la.mant >= sm_m) { mant = la.mant - sm_m; sign = la.sign; }
    else { mant = sm_m - la.mant; sign = sm.sign; }
    if (mant == 0u && !carry) { return F3(0u, 0, 0u, select(0u, 1u, sticky)); }
    var exp = la.exp;
    if (carry) { if ((mant & 1u) != 0u) { sticky = true; } mant = (mant >> 1u) | 0x80000000u; exp = exp + 1; }
    loop {
        if ((mant & 0x80000000u) != 0u) { break; }
        mant = mant << 1u; exp = exp - 1;
    }
    return F3(sign, exp, mant, select(0u, 1u, sticky));
}

fn single_bits_to_f3(bits: u32) -> F3 {
    let sign = (bits >> 31u) & 1u;
    let raw_exp = (bits >> 23u) & 0xFFu;
    let mant23 = bits & 0x7FFFFFu;
    if (raw_exp == 0u) { return f3_zero(sign); }
    return F3(sign, i32(raw_exp), (0x800000u | mant23) << 8u, 0u);
}

fn f3_div_to_bf16(num: F3, den: F3) -> u32 {
    let sign = num.sign ^ den.sign;
    if (num.mant == 0u || den.mant == 0u) { return sign << 15u; }
    var rem = num.mant;
    var q_hi = 0u;
    if (rem >= den.mant) { rem = rem - den.mant; q_hi = 1u; }
    var q = 0u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let top = rem >> 31u;
        rem = rem << 1u;
        if (top == 1u || rem >= den.mant) { rem = rem - den.mant; q = (q << 1u) | 1u; } else { q = q << 1u; }
    }
    let rem_nonzero = rem != 0u;
    var mant_q: u32; var exp_r: i32; var shifted_out = false;
    if (q_hi == 1u) { mant_q = (q >> 1u) | 0x80000000u; exp_r = num.exp - den.exp + 127; shifted_out = (q & 1u) != 0u; }
    else { mant_q = q; exp_r = num.exp - den.exp + 126; }
    let sticky = rem_nonzero || shifted_out || num.sticky != 0u || den.sticky != 0u;
    return f3_to_bf16(F3(sign, exp_r, mant_q, select(0u, 1u, sticky)));
}

fn f3_neg(q: F3) -> F3 { return F3(q.sign ^ 1u, q.exp, q.mant, q.sticky); }
fn bf16_fmsub_q30(a: u32, c: i32, b: u32, d: i32) -> u32 { return f3_to_bf16(f3_add(bf16_mul_q30_ext(a, c), f3_neg(bf16_mul_q30_ext(b, d)))); }
fn bf16_fmadd_q30(a: u32, c: i32, b: u32, d: i32) -> u32 { return f3_to_bf16(f3_add(bf16_mul_q30_ext(a, c), bf16_mul_q30_ext(b, d))); }

fn f3_from_count(n: u32) -> F3 {
    let top = 31u - countLeadingZeros(n);
    return F3(0u, 127 + i32(top), n << (31u - top), 0u);
}

// signed (delta * frac + 2^15) >> 16 (arithmetic shift), delta i32, frac < 2^16
fn interp_term(delta: i32, frac: u32) -> i32 {
    let neg = delta < 0;
    let d = u32(abs(delta));
    let lo_part = (d & 0xFFFFu) * frac;
    let hi_part = (d >> 16u) * frac;
    if (neg) {
        let lo2 = lo_part + 0x7FFFu;
        let c2 = select(0u, 1u, lo2 < lo_part);
        let r2 = hi_part + (lo2 >> 16u) + (c2 << 16u);
        return -i32(r2);
    }
    let lo_sum = lo_part + 0x8000u;
    let carry = select(0u, 1u, lo_sum < lo_part);
    let r = hi_part + (lo_sum >> 16u) + (carry << 16u);
    return i32(r);
}

// ---- the cell and the bucket (cells.rs, matmul.rs) --------------------------------------------

struct Cell { s: u32, m: u32, n: i32 };

fn decomp(p: u32) -> Cell {
    let sheet = (p >> 15u) & 1u;
    let exp = i32((p >> 7u) & 0xFFu);
    let mant = i32(p & 0x7Fu);
    var sig: i32; var base: i32;
    if (exp != 0) { sig = mant | 0x80; base = exp - 127 - 7; } else { sig = mant; base = 1 - 127 - 7; }
    if (sig == 0) { return Cell(sheet, 0u, 0); }
    let tz = i32(countTrailingZeros(u32(sig)));
    return Cell(sheet, u32(sig >> u32(tz)), base + tz);
}

fn lane_add(acc: ptr<function, array<u32, 9>>, prod: u32, sh: u32) {
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

// pos - neg -> magnitude limbs and sign; returns 1 if the value does not fit the 256-bit lane
fn lane_settle(pos: ptr<function, array<u32, 9>>, neg: ptr<function, array<u32, 9>>, mag: ptr<function, array<u32, 9>>) -> vec2<u32> {
    var borrow = 0u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let v = (*pos)[i];
        let w = (*neg)[i];
        let t = v - w;
        let b1 = select(0u, 1u, v < w);
        let d = t - borrow;
        let b2 = select(0u, 1u, t < borrow);
        (*mag)[i] = d;
        borrow = b1 | b2;
    }
    let negative = borrow;
    if (negative == 1u) {
        var c = 1u;
        for (var i = 0u; i < 9u; i = i + 1u) {
            let inv = ~(*mag)[i];
            let s = inv + c;
            c = select(0u, 1u, s < inv);
            (*mag)[i] = s;
        }
    }
    let overflow = select(0u, 1u, (*mag)[8] != 0u || ((*mag)[7] & 0x80000000u) != 0u);
    return vec2<u32>(negative, overflow);
}

fn mag_top_bit(mag: ptr<function, array<u32, 9>>) -> i32 {
    for (var l = 7; l >= 0; l = l - 1) {
        let v = (*mag)[l];
        if (v != 0u) { return l * 32 + 31 - i32(countLeadingZeros(v)); }
    }
    return -1;
}

fn mag_bits8(mag: ptr<function, array<u32, 9>>, lo_bit: i32) -> u32 {
    let limb = u32(lo_bit) >> 5u;
    let off = u32(lo_bit) & 31u;
    var v = (*mag)[limb] >> off;
    if (off > 24u) { v = v | ((*mag)[limb + 1u] << (32u - off)); }
    return v & 0xFFu;
}

fn mag_bit(mag: ptr<function, array<u32, 9>>, bit: i32) -> u32 {
    return ((*mag)[u32(bit) >> 5u] >> (u32(bit) & 31u)) & 1u;
}

fn mag_any_below(mag: ptr<function, array<u32, 9>>, bit: i32) -> bool {
    if (bit <= 0) { return false; }
    let limb = u32(bit) >> 5u;
    let off = u32(bit) & 31u;
    for (var l = 0u; l < limb; l = l + 1u) { if ((*mag)[l] != 0u) { return true; } }
    if (off == 0u) { return false; }
    return ((*mag)[limb] & ((1u << off) - 1u)) != 0u;
}

// Bucket::collapse; returns the pattern, or 0xFFFF with the error flag set (refused)
fn collapse(pos: ptr<function, array<u32, 9>>, neg: ptr<function, array<u32, 9>>, sticky_in: bool, frac: i32) -> u32 {
    var mag: array<u32, 9>;
    let sn = lane_settle(pos, neg, &mag);
    if (sn.y == 1u) { atomicOr(&ERR[0], 1u); return 0xFFFFu; }
    let hb = mag_top_bit(&mag);
    if (hb < 0) { return 0u; }
    let sign = sn.x << 15u;
    var sig: u32; var round: u32; var low_sticky: bool;
    if (hb >= 8) { sig = mag_bits8(&mag, hb - 7); round = mag_bit(&mag, hb - 8); low_sticky = mag_any_below(&mag, hb - 8); }
    else { sig = mag[0] << u32(7 - hb); round = 0u; low_sticky = false; }
    let sticky = low_sticky || sticky_in;
    var exp = hb - frac + 127;
    if (round == 1u && (sticky || (sig & 1u) == 1u)) {
        sig = sig + 1u;
        if (sig == 0x100u) { sig = sig >> 1u; exp = exp + 1; }
    }
    if (exp >= 0xFF) { atomicOr(&ERR[0], 2u); return 0xFFFFu; }
    if (exp < 1) { return sign; }
    return sign | (u32(exp) << 7u) | (sig & 0x7Fu);
}

// ---- the lane as nine named registers (no dynamic indexing: stays in registers) ----------------

struct Lane { l0: u32, l1: u32, l2: u32, l3: u32, l4: u32, l5: u32, l6: u32, l7: u32, l8: u32 };

fn lane_zero() -> Lane { return Lane(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); }

fn lane_to_arr(l: Lane) -> array<u32, 9> {
    var a: array<u32, 9>;
    a[0] = l.l0; a[1] = l.l1; a[2] = l.l2; a[3] = l.l3; a[4] = l.l4; a[5] = l.l5; a[6] = l.l6; a[7] = l.l7; a[8] = l.l8;
    return a;
}

// acc += prod << sh: lo into limb sh/32, hi into the next, carries rippled through all nine, branchless
fn lane_add9(a: ptr<function, Lane>, prod: u32, sh: u32) {
    let limb = sh >> 5u;
    let bit = sh & 31u;
    let lo = prod << bit;
    var hi = 0u;
    if (bit != 0u) { hi = prod >> (32u - bit); }
    var c = 0u;
    { let sel = select(0u, lo, limb == 0u) + 0u; let s1 = (*a).l0 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l0 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 1u) + select(0u, hi, limb == 0u); let s1 = (*a).l1 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l1 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 2u) + select(0u, hi, limb == 1u); let s1 = (*a).l2 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l2 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 3u) + select(0u, hi, limb == 2u); let s1 = (*a).l3 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l3 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 4u) + select(0u, hi, limb == 3u); let s1 = (*a).l4 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l4 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 5u) + select(0u, hi, limb == 4u); let s1 = (*a).l5 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l5 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 6u) + select(0u, hi, limb == 5u); let s1 = (*a).l6 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l6 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 7u) + select(0u, hi, limb == 6u); let s1 = (*a).l7 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l7 = s2; c = c1 | c2; }
    { let sel = select(0u, lo, limb == 8u) + select(0u, hi, limb == 7u); let s1 = (*a).l8 + sel; let c1 = select(0u, 1u, s1 < sel); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l8 = s2; c = c1 | c2; }
}

fn collapse9(pos: ptr<function, Lane>, neg: ptr<function, Lane>, sticky_in: bool, frac: i32) -> u32 {
    var pa = lane_to_arr(*pos);
    var na = lane_to_arr(*neg);
    return collapse(&pa, &na, sticky_in, frac);
}

// acc += ±(prod << sh) in ONE two's-complement lane: ng = 0 (add) or 0xFFFFFFFF (subtract: invert + 1), branchless
fn lane_add_signed(a: ptr<function, Lane>, prod: u32, sh: u32, ng: u32) {
    let limb = sh >> 5u;
    let bit = sh & 31u;
    let lo = prod << bit;
    var hi = 0u;
    if (bit != 0u) { hi = prod >> (32u - bit); }
    var c = ng & 1u;
    { let v = (select(0u, lo, limb == 0u) + 0u) ^ ng; let s1 = (*a).l0 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l0 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 1u) + select(0u, hi, limb == 0u)) ^ ng; let s1 = (*a).l1 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l1 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 2u) + select(0u, hi, limb == 1u)) ^ ng; let s1 = (*a).l2 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l2 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 3u) + select(0u, hi, limb == 2u)) ^ ng; let s1 = (*a).l3 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l3 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 4u) + select(0u, hi, limb == 3u)) ^ ng; let s1 = (*a).l4 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l4 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 5u) + select(0u, hi, limb == 4u)) ^ ng; let s1 = (*a).l5 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l5 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 6u) + select(0u, hi, limb == 5u)) ^ ng; let s1 = (*a).l6 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l6 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 7u) + select(0u, hi, limb == 6u)) ^ ng; let s1 = (*a).l7 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l7 = s2; c = c1 | c2; }
    { let v = (select(0u, lo, limb == 8u) + select(0u, hi, limb == 7u)) ^ ng; let s1 = (*a).l8 + v; let c1 = select(0u, 1u, s1 < v); let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1); (*a).l8 = s2; c = c1 | c2; }
}

// a signed 288-bit lane -> magnitude limbs; returns (negative, overflow)
fn lane_settle_signed(x: ptr<function, array<u32, 9>>, mag: ptr<function, array<u32, 9>>) -> vec2<u32> {
    let negative = (*x)[8] >> 31u;
    if (negative == 1u) {
        var c = 1u;
        for (var i = 0u; i < 9u; i = i + 1u) {
            let inv = ~(*x)[i];
            let s = inv + c;
            c = select(0u, 1u, s < inv);
            (*mag)[i] = s;
        }
    } else {
        for (var i = 0u; i < 9u; i = i + 1u) { (*mag)[i] = (*x)[i]; }
    }
    let overflow = select(0u, 1u, (*mag)[8] != 0u || ((*mag)[7] & 0x80000000u) != 0u);
    return vec2<u32>(negative, overflow);
}

// Bucket::collapse on a settled magnitude
fn collapse_mag(mag: ptr<function, array<u32, 9>>, negative: u32, overflow: u32, sticky_in: bool, frac: i32) -> u32 {
    if (overflow == 1u) { atomicOr(&ERR[0], 1u); return 0xFFFFu; }
    let hb = mag_top_bit(mag);
    if (hb < 0) { return 0u; }
    let sign = negative << 15u;
    var sig: u32; var round: u32; var low_sticky: bool;
    if (hb >= 8) { sig = mag_bits8(mag, hb - 7); round = mag_bit(mag, hb - 8); low_sticky = mag_any_below(mag, hb - 8); }
    else { sig = (*mag)[0] << u32(7 - hb); round = 0u; low_sticky = false; }
    let sticky = low_sticky || sticky_in;
    var exp = hb - frac + 127;
    if (round == 1u && (sticky || (sig & 1u) == 1u)) {
        sig = sig + 1u;
        if (sig == 0x100u) { sig = sig >> 1u; exp = exp + 1; }
    }
    if (exp >= 0xFF) { atomicOr(&ERR[0], 2u); return 0xFFFFu; }
    if (exp < 1) { return sign; }
    return sign | (u32(exp) << 7u) | (sig & 0x7Fu);
}
"#;

/// The kernels, on one binding set. Roles of B1..B8 per kernel are in the comments.
pub const OPS: &str = r#"
struct Params { n: u32, m: u32, k: u32, l: u32, frac: i32, p0: u32, a: u32, b: u32, c: u32, d: u32, e: u32, f: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> B1: array<u32>;
@group(0) @binding(2) var<storage, read_write> B2: array<u32>;
@group(0) @binding(3) var<storage, read_write> B3: array<u32>;
@group(0) @binding(4) var<storage, read_write> B4: array<u32>;
@group(0) @binding(5) var<storage, read_write> B5: array<u32>;
@group(0) @binding(6) var<storage, read_write> B6: array<u32>;
@group(0) @binding(7) var<storage, read_write> ERR: array<atomic<u32>>;
"#;

pub const KERNELS: &str = r#"
// k_proj: y = x · Wᵀ over the grid. B1 ids (u16 pairs), B2 rank, B3 atoms (packed), B4 x patterns [seq×cols],
// B5 out patterns [seq×rows]. P: n=rows m=cols k=seq frac, a=tcs b=place_tile_cols c=th d=tw e=cell0 f=ncells
@compute @workgroup_size(64)
fn k_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.f) { return; }
    let cell = P.e + gid.x;
    let s = cell / P.n;
    let o = cell % P.n;
    var pos = lane_zero();
    var neg = lane_zero();
    let tr = o / P.c;
    let ln = o % P.c;
    let xbase = s * P.m;
    var sticky = false;
    for (var tc = 0u; tc < P.a; tc = tc + 1u) {
        let t = B2[tr * P.b + tc];
        let base = t * P.c * P.d + ln * P.d;
        for (var j = 0u; j < P.d; j = j + 1u) {
            let idx = base + j;
            let word = B1[idx >> 1u];
            var id = word & 0xFFFFu;
            if ((idx & 1u) != 0u) { id = word >> 16u; }
            let x = decomp(B4[xbase + tc * P.d + j]);
            if (x.m == 0u) { continue; }
            if (x.n < -90) { sticky = true; continue; }
            let a = B3[id];
            let mw = (a >> 16u) & 0xFFu;
            let nw = i32(a & 0xFFFFu) - 32768;
            let shi = x.n + nw + P.frac;
            if (shi < 0) { atomicOr(&ERR[0], 4u); continue; }
            let prod = x.m * mw;
            if ((x.s ^ (a >> 31u)) != 0u) { lane_add9(&neg, prod, u32(shi)); } else { lane_add9(&pos, prod, u32(shi)); }
        }
    }
    B5[cell] = collapse9(&pos, &neg, sticky, P.frac);
}


// k_pack: activation patterns -> packed cells (bit31 sign, bits 16..24 m, bits 0..16 n+32768), once per buffer.
// B1 in, B2 out. P: n
@compute @workgroup_size(64)
fn k_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let c = decomp(B1[gid.x]);
    B2[gid.x] = (c.s << 31u) | (c.m << 16u) | (u32(c.n + 32768) & 0xFFFFu);
}

// k_proj_wg: the projection, one WORKGROUP per output cell: 64 threads each walk every 64th column of the
// row's tile lines (coalesced ids), ONE signed two's-complement lane per thread (branchless), a tree
// reduction of the lanes in shared memory with carries, sticky OR-reduced, thread 0 settles and collapses.
// B1 ids, B2 rank, B3 atoms, B4 PACKED x (k_pack), B5 out. Params as k_proj (P.e = cell0, P.f = ncells);
// dispatch (min(ncells, 65535), ceil(ncells / 65535)); cell = P.e + wg.y * 65535 + wg.x.
var<workgroup> red: array<array<u32, 9>, 64>;
var<workgroup> red_sticky: atomic<u32>;

@compute @workgroup_size(64)
fn k_proj_wg(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let local = wg.y * 65535u + wg.x;
    let cell = P.e + local;
    if (tid == 0u) { atomicStore(&red_sticky, 0u); }
    workgroupBarrier();
    var acc = lane_zero();
    let valid = local < P.f;
    if (valid) {
        let s = cell / P.n;
        let o = cell % P.n;
        let tr = o / P.c;
        let ln = o % P.c;
        let xbase = s * P.m;
        var sticky = false;
        for (var tc = 0u; tc < P.a; tc = tc + 1u) {
            let t = B2[tr * P.b + tc];
            let base = t * P.c * P.d + ln * P.d;
            for (var j = tid; j < P.d; j = j + 64u) {
                let idx = base + j;
                let word = B1[idx >> 1u];
                let id = (word >> ((idx & 1u) * 16u)) & 0xFFFFu;
                let x = B4[xbase + tc * P.d + j];
                let mx = (x >> 16u) & 0xFFu;
                let nx = i32(x & 0xFFFFu) - 32768;
                if (mx == 0u) { continue; }
                if (nx < -90) { sticky = true; continue; }
                let a = B3[id];
                let mw = (a >> 16u) & 0xFFu;
                let nw = i32(a & 0xFFFFu) - 32768;
                let shi = nx + nw + P.frac;
                if (shi < 0) { atomicOr(&ERR[0], 4u); continue; }
                let ng = 0u - ((x ^ a) >> 31u);
                lane_add_signed(&acc, mx * mw, u32(shi), ng);
            }
        }
        if (sticky) { atomicOr(&red_sticky, 1u); }
    }
    let la = lane_to_arr(acc);
    for (var i = 0u; i < 9u; i = i + 1u) { red[tid][i] = la[i]; }
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            var c = 0u;
            for (var i = 0u; i < 9u; i = i + 1u) {
                let a = red[tid][i];
                let b = red[tid + stride][i];
                let s1 = a + b;
                let c1 = select(0u, 1u, s1 < a);
                let s2 = s1 + c;
                let c2 = select(0u, 1u, s2 < s1);
                red[tid][i] = s2;
                c = c1 | c2;
            }
        }
        workgroupBarrier();
    }
    if (tid == 0u && valid) {
        var x9: array<u32, 9>;
        var mag: array<u32, 9>;
        for (var i = 0u; i < 9u; i = i + 1u) { x9[i] = red[0][i]; }
        let so = lane_settle_signed(&x9, &mag);
        B5[cell] = collapse_mag(&mag, so.x, so.y, atomicLoad(&red_sticky) != 0u, P.frac);
    }
}

// k_dense: y[s][o] = Σ_i a[s][i]·c[o][i], both pattern matrices with the floor on both (matmul_dense).
// B1 a [seq×in], B2 c [out×in], B3 out [seq×out]. P: n=cells m=in k=out frac
@compute @workgroup_size(64)
fn k_dense(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let s = gid.x / P.k;
    let o = gid.x % P.k;
    var pos = lane_zero();
    var neg = lane_zero();
    var sticky = false;
    for (var i = 0u; i < P.m; i = i + 1u) {
        var x = decomp(B1[s * P.m + i]);
        var w = decomp(B2[o * P.m + i]);
        if (x.m != 0u && x.n < -90) { sticky = true; x.m = 0u; }
        if (w.m != 0u && w.n < -90) { sticky = true; w.m = 0u; }
        if (x.m == 0u || w.m == 0u) { continue; }
        let shi = x.n + w.n + P.frac;
        if (shi < 0) { atomicOr(&ERR[0], 4u); continue; }
        if ((x.s ^ w.s) != 0u) { lane_add9(&neg, x.m * w.m, u32(shi)); } else { lane_add9(&pos, x.m * w.m, u32(shi)); }
    }
    B3[gid.x] = collapse9(&pos, &neg, sticky, P.frac);
}

// k_scores: scores[s][h][j] = q[s][h·hd..] · K[j][kvh·hd..], kvh = h / group. B1 q [n×qd], B2 K [total×kvd],
// B3 out [n×heads×total]. P: n=cells m=hd k=heads l=total frac, a=group b=qd c=kvd
@compute @workgroup_size(64)
fn k_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let j = gid.x % P.l;
    let sh_ = gid.x / P.l;
    let h = sh_ % P.k;
    let s = sh_ / P.k;
    let kvh = h / P.a;
    var pos = lane_zero();
    var neg = lane_zero();
    var sticky = false;
    for (var i = 0u; i < P.m; i = i + 1u) {
        var x = decomp(B1[s * P.b + h * P.m + i]);
        var w = decomp(B2[j * P.c + kvh * P.m + i]);
        if (x.m != 0u && x.n < -90) { sticky = true; x.m = 0u; }
        if (w.m != 0u && w.n < -90) { sticky = true; w.m = 0u; }
        if (x.m == 0u || w.m == 0u) { continue; }
        let shi = x.n + w.n + P.frac;
        if (shi < 0) { atomicOr(&ERR[0], 4u); continue; }
        if ((x.s ^ w.s) != 0u) { lane_add9(&neg, x.m * w.m, u32(shi)); } else { lane_add9(&pos, x.m * w.m, u32(shi)); }
    }
    B3[gid.x] = collapse9(&pos, &neg, sticky, P.frac);
}

// k_pv: attn[s][h·hd+d] = Σ_j probs[s][h][j] · V[j][kvh·hd+d]. B1 probs [n×heads×total], B2 V [total×kvd],
// B3 out [n×qd]. P: n=cells(n·heads·hd) m=hd k=heads l=total frac, a=group b=qd c=kvd
@compute @workgroup_size(64)
fn k_pv(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let d = gid.x % P.m;
    let sh_ = gid.x / P.m;
    let h = sh_ % P.k;
    let s = sh_ / P.k;
    let kvh = h / P.a;
    var pos = lane_zero();
    var neg = lane_zero();
    var sticky = false;
    for (var j = 0u; j < P.l; j = j + 1u) {
        var x = decomp(B1[(s * P.k + h) * P.l + j]);
        var w = decomp(B2[j * P.c + kvh * P.m + d]);
        if (x.m != 0u && x.n < -90) { sticky = true; x.m = 0u; }
        if (w.m != 0u && w.n < -90) { sticky = true; w.m = 0u; }
        if (x.m == 0u || w.m == 0u) { continue; }
        let shi = x.n + w.n + P.frac;
        if (shi < 0) { atomicOr(&ERR[0], 4u); continue; }
        if ((x.s ^ w.s) != 0u) { lane_add9(&neg, x.m * w.m, u32(shi)); } else { lane_add9(&pos, x.m * w.m, u32(shi)); }
    }
    B3[s * P.b + h * P.m + d] = collapse9(&pos, &neg, sticky, P.frac);
}

// k_rmsnorm: one thread per row. B1 x [rows×dim], B2 gamma [dim], B3 rsqrt LUT, B4 out. P: n=rows m=dim k=eps
@compute @workgroup_size(64)
fn k_rmsnorm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let base = gid.x * P.m;
    var sumsq = f3_zero(0u);
    for (var i = 0u; i < P.m; i = i + 1u) { sumsq = f3_add(sumsq, bf16_mul_ext(B1[base + i], B1[base + i])); }
    let mean = f3_div_to_bf16(sumsq, f3_from_count(P.m));
    let s = bf16_add(mean, P.k);
    let r = f3_to_bf16(single_bits_to_f3(B3[s & 0x7FFFu]));
    for (var i = 0u; i < P.m; i = i + 1u) { B4[base + i] = bf16_mul(bf16_mul(B1[base + i], r), B2[i]); }
}

// k_rope: per (row, head, pair). B1 q [rows×width], B2 phase_step lo words [half], B3 cossin [65536×2] (i32 as u32),
// B4 out, B5 position per row. P: n=rows·heads·half m=width k=hd l=heads
@compute @workgroup_size(64)
fn k_rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let half = P.k / 2u;
    let i = gid.x % half;
    let rh = gid.x / half;
    let h = rh % P.l;
    let row = rh / P.l;
    let pos = B5[row];
    let phase = pos * B2[i];                    // (pos · step) mod 2^32: idx = bits 16..31, frac = bits 0..15
    let phase_idx = phase >> 16u;
    let next_idx = (phase_idx + 1u) & 0xFFFFu;
    let frac = phase & 0xFFFFu;
    let c0 = i32(B3[phase_idx * 2u]);
    let s0 = i32(B3[phase_idx * 2u + 1u]);
    let c1 = i32(B3[next_idx * 2u]);
    let s1 = i32(B3[next_idx * 2u + 1u]);
    let c_q30 = c0 + interp_term(c1 - c0, frac);
    let s_q30 = s0 + interp_term(s1 - s0, frac);
    let off = row * P.m + h * P.k;
    let qi = B1[off + i];
    let qj = B1[off + i + half];
    B4[off + i] = bf16_fmsub_q30(qi, c_q30, qj, s_q30);
    B4[off + i + half] = bf16_fmadd_q30(qi, s_q30, qj, c_q30);
}

// k_softmax: one thread per (row, head): scores scaled by attn_scale, masked at position B4[row], probs out.
// B1 scores [rows×heads×total], B2 exp LUT, B3 out, B4 position per row. P: n=rows·heads m=total k=heads l=attn_scale
@compute @workgroup_size(64)
fn k_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let row = gid.x / P.k;
    let pos = B4[row];
    let base = gid.x * P.m;
    var max_score = bf16_mul(B1[base], P.l);
    for (var j = 1u; j <= pos; j = j + 1u) { let v = bf16_mul(B1[base + j], P.l); if (bf16_lt(max_score, v)) { max_score = v; } }
    var z = f3_zero(0u);
    for (var j = 0u; j <= pos; j = j + 1u) {
        let sh = bf16_sub(bf16_mul(B1[base + j], P.l), max_score);
        z = f3_add(z, single_bits_to_f3(B2[sh & 0x7FFFu]));
    }
    for (var j = 0u; j < P.m; j = j + 1u) {
        if (j <= pos) {
            let sh = bf16_sub(bf16_mul(B1[base + j], P.l), max_score);
            B3[base + j] = f3_div_to_bf16(single_bits_to_f3(B2[sh & 0x7FFFu]), z);
        } else { B3[base + j] = 0u; }
    }
}

// k_silu: act = silu(g) · u. B1 g, B2 u, B3 exp LUT, B4 out. P: n
@compute @workgroup_size(64)
fn k_silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    let x = B1[gid.x];
    let e = single_bits_to_f3(B3[x & 0x7FFFu]);
    let one = single_bits_to_f3(0x3F800000u);
    let denom = f3_add(one, e);
    var sigma: u32;
    if ((x & 0x8000u) == 0u) { sigma = f3_div_to_bf16(one, denom); } else { sigma = f3_div_to_bf16(e, denom); }
    B4[gid.x] = bf16_mul(bf16_mul(x, sigma), B2[gid.x]);
}

// k_add: exact sum of two patterns through the bucket at frac 90 (matmul::add). B1 a, B2 b, B3 out. P: n
@compute @workgroup_size(64)
fn k_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.n) { return; }
    var pos = lane_zero();
    var neg = lane_zero();
    let p = decomp(B1[gid.x]);
    let q = decomp(B2[gid.x]);
    if (p.m != 0u) { let shi = p.n + 90; if (shi < 0) { atomicOr(&ERR[0], 4u); } else if (p.s != 0u) { lane_add9(&neg, p.m, u32(shi)); } else { lane_add9(&pos, p.m, u32(shi)); } }
    if (q.m != 0u) { let shi = q.n + 90; if (shi < 0) { atomicOr(&ERR[0], 4u); } else if (q.s != 0u) { lane_add9(&neg, q.m, u32(shi)); } else { lane_add9(&pos, q.m, u32(shi)); } }
    B3[gid.x] = collapse9(&pos, &neg, false, 90);
}

// probes of the primitives (the gates): B1, B2, B3 in; B4 out; P.n elements
@compute @workgroup_size(64) fn p_mul(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = bf16_mul(B1[g.x], B2[g.x]); }
@compute @workgroup_size(64) fn p_add(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = bf16_add(B1[g.x], B2[g.x]); }
@compute @workgroup_size(64) fn p_lt(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = select(0u, 1u, bf16_lt(B1[g.x], B2[g.x])); }
@compute @workgroup_size(64) fn p_mulext(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = f3_to_bf16(bf16_mul_ext(B1[g.x], B2[g.x])); }
@compute @workgroup_size(64) fn p_f3add(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = f3_to_bf16(f3_add(bf16_mul_ext(B1[g.x], B2[g.x]), bf16_mul_ext(B3[g.x], B2[(g.x + 1u) % P.n]))); }
@compute @workgroup_size(64) fn p_f3div(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = f3_div_to_bf16(bf16_mul_ext(B1[g.x], B2[g.x]), bf16_mul_ext(B3[g.x], B3[(g.x + 1u) % P.n])); }
@compute @workgroup_size(64) fn p_fmsub_q30(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = bf16_fmsub_q30(B1[g.x], i32(B3[g.x]), B2[g.x], i32(B3[(g.x + 1u) % P.n])); }
@compute @workgroup_size(64) fn p_fmadd_q30(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = bf16_fmadd_q30(B1[g.x], i32(B3[g.x]), B2[g.x], i32(B3[(g.x + 1u) % P.n])); }
@compute @workgroup_size(64) fn p_interp(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = u32(interp_term(i32(B1[g.x]), B2[g.x] & 0xFFFFu)); }
@compute @workgroup_size(64) fn p_lutdiv(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= P.n) { return; } B4[g.x] = f3_div_to_bf16(single_bits_to_f3(B1[g.x]), single_bits_to_f3(B2[g.x])); }
"#;

pub fn shader_source() -> String {
    format!("{}\n{}\n{}", OPS, LIB, KERNELS)
}

/// A wgpu device with the ops module and its one layout; the probe for the gates and the base of the layer pipeline.
pub struct Ops {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
    pub layout: wgpu::BindGroupLayout,
    pub pipelines: std::collections::HashMap<String, wgpu::ComputePipeline>,
    pub adapter_name: String,
}

pub const ENTRIES: [&str; 14] = ["k_proj", "k_proj_wg", "k_pack", "k_dense", "k_scores", "k_pv", "k_rmsnorm", "k_rope", "k_softmax", "k_silu", "k_add", "p_mul", "p_add", "p_lt"];
pub const PROBES: [&str; 7] = ["p_mulext", "p_f3add", "p_f3div", "p_fmsub_q30", "p_fmadd_q30", "p_interp", "p_lutdiv"];

impl Ops {
    pub fn new() -> Result<Ops, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).ok_or("no adapter")?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.max(8);
        let want = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let features = adapter.features() & want;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("atlas ops"), required_features: features, required_limits: limits, memory_hints: wgpu::MemoryHints::Performance }, None)).map_err(|e| format!("{:?}", e))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ops"), source: wgpu::ShaderSource::Wgsl(shader_source().into()) });
        let entry = |i: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding: i, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let st = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let un = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ops layout"), entries: &[entry(0, un), entry(1, st), entry(2, st), entry(3, st), entry(4, st), entry(5, st), entry(6, st), entry(7, st)] });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let mut pipelines = std::collections::HashMap::new();
        for name in ENTRIES.iter().chain(PROBES.iter()) {
            pipelines.insert(name.to_string(), device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: Some(&pl), module: &module, entry_point: Some(name), compilation_options: Default::default(), cache: None }));
        }
        Ok(Ops { device, queue, module, layout, pipelines, adapter_name: format!("{} ({:?})", info.name, info.backend) })
    }

    /// Run one kernel once, standalone: inputs b1..b6 (missing = a 4-byte dummy), params (12 u32), workgroups;
    /// returns the contents of the buffer at `out_slot` (1-based) and the error flags.
    pub fn run_once(&self, name: &str, bufs: [&[u32]; 6], params: [u32; 12], groups: u32, out_slot: usize) -> (Vec<u32>, u32) {
        let mk = |v: &[u32]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(if v.is_empty() { &[0u32] } else { v }), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
        let bs: Vec<wgpu::Buffer> = bufs.iter().map(|b| mk(b)).collect();
        let err = mk(&[0u32]);
        let pb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &pb, offset: 0, size: None }) },
            wgpu::BindGroupEntry { binding: 1, resource: bs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bs[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bs[4].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bs[5].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: err.as_entire_binding() },
        ] });
        let out_bytes = (bufs[out_slot - 1].len().max(1) * 4) as u64;
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: out_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let eb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[name]);
            pass.set_bind_group(0, &bind, &[0]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&bs[out_slot - 1], 0, &rb, 0, out_bytes);
        enc.copy_buffer_to_buffer(&err, 0, &eb, 0, 4);
        self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let (tx2, rx2) = std::sync::mpsc::channel();
        eb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx2.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        rx2.recv().unwrap().unwrap();
        let out: Vec<u32> = bytemuck::cast_slice(&rb.slice(..).get_mapped_range()).to_vec();
        let e: u32 = bytemuck::cast_slice::<u8, u32>(&eb.slice(..).get_mapped_range())[0];
        rb.unmap();
        eb.unmap();
        (out, e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::bf16::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn patterns(next: &mut impl FnMut() -> u64, n: usize, lo: u32, hi: u32) -> Vec<u32> {
        (0..n)
            .map(|_| {
                let r = next();
                if r % 41 == 0 {
                    return ((r >> 8) & 1) as u32 * 0x8000;
                }
                let sign = ((r >> 1) & 1) as u32;
                let exp = lo + ((r >> 8) % (hi - lo + 1) as u64) as u32;
                let mant = ((r >> 20) & 0x7F) as u32;
                (sign << 15) | (exp << 7) | mant
            })
            .collect()
    }

    fn ops() -> Option<Ops> {
        match Ops::new() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("no card: {} (test skipped)", e);
                None
            }
        }
    }

    fn params(n: u32) -> [u32; 12] {
        let mut p = [0u32; 12];
        p[0] = n;
        p
    }

    fn groups(n: usize) -> u32 {
        ((n as u32) + 63) / 64
    }

    #[test]
    fn card_bf16_mul_add_lt_match_v5() {
        let Some(p) = ops() else { return };
        let mut next = rng(11);
        let n = 200_000;
        let a = patterns(&mut next, n, 90, 160);
        let b = patterns(&mut next, n, 90, 160);
        let o = vec![0u32; n];
        let (m, _) = p.run_once("p_mul", [&a, &b, &[], &o, &[], &[]], params(n as u32), groups(n), 4);
        let (s, _) = p.run_once("p_add", [&a, &b, &[], &o, &[], &[]], params(n as u32), groups(n), 4);
        let (l, _) = p.run_once("p_lt", [&a, &b, &[], &o, &[], &[]], params(n as u32), groups(n), 4);
        let mut bad = 0;
        for i in 0..n {
            if m[i] != bf16_mul(a[i] as u16, b[i] as u16) as u32 { bad += 1; }
            if s[i] != bf16_add(a[i] as u16, b[i] as u16) as u32 { bad += 1; }
            if l[i] != bf16_lt(a[i] as u16, b[i] as u16) as u32 { bad += 1; }
        }
        assert_eq!(bad, 0, "bf16 mul/add/lt mismatches");
    }

    #[test]
    fn card_f3_ops_match_v5() {
        let Some(p) = ops() else { return };
        let mut next = rng(23);
        let n = 200_000;
        let a = patterns(&mut next, n, 100, 150);
        let b = patterns(&mut next, n, 100, 150);
        let c = patterns(&mut next, n, 100, 150);
        let o = vec![0u32; n];
        let (me, _) = p.run_once("p_mulext", [&a, &b, &c, &o, &[], &[]], params(n as u32), groups(n), 4);
        let (ad, _) = p.run_once("p_f3add", [&a, &b, &c, &o, &[], &[]], params(n as u32), groups(n), 4);
        let (dv, _) = p.run_once("p_f3div", [&a, &b, &c, &o, &[], &[]], params(n as u32), groups(n), 4);
        let mut bad = [0; 3];
        for i in 0..n {
            let (ai, bi, ci, bn, cn) = (a[i] as u16, b[i] as u16, c[i] as u16, b[(i + 1) % n] as u16, c[(i + 1) % n] as u16);
            if me[i] != f3_to_bf16(bf16_mul_ext(ai, bi)) as u32 { bad[0] += 1; }
            if ad[i] != f3_to_bf16(f3_add(bf16_mul_ext(ai, bi), bf16_mul_ext(ci, bn))) as u32 { bad[1] += 1; }
            if dv[i] != f3_div_to_bf16(bf16_mul_ext(ai, bi), bf16_mul_ext(ci, cn)) as u32 { bad[2] += 1; }
        }
        assert_eq!(bad, [0, 0, 0], "f3 mul_ext/add/div mismatches");
    }

    #[test]
    fn card_q30_and_interp_match_v5() {
        let Some(p) = ops() else { return };
        let mut next = rng(37);
        let n = 200_000;
        let a = patterns(&mut next, n, 100, 150);
        let b = patterns(&mut next, n, 100, 150);
        let c: Vec<u32> = (0..n).map(|i| {
            let r = next();
            let v: i32 = match i % 97 { 0 => 1 << 30, 1 => -(1 << 30), 2 => 0, _ => ((r % (1u64 << 31)) as i64 - (1i64 << 30)) as i32 };
            v as u32
        }).collect();
        let o = vec![0u32; n];
        let (fs, _) = p.run_once("p_fmsub_q30", [&a, &b, &c, &o, &[], &[]], params(n as u32), groups(n), 4);
        let (fa, _) = p.run_once("p_fmadd_q30", [&a, &b, &c, &o, &[], &[]], params(n as u32), groups(n), 4);
        let mut bad = [0; 2];
        for i in 0..n {
            let (ci, cn) = (c[i] as i32, c[(i + 1) % n] as i32);
            if fs[i] != bf16_fmsub_q30(a[i] as u16, ci, b[i] as u16, cn) as u32 { bad[0] += 1; }
            if fa[i] != bf16_fmadd_q30(a[i] as u16, ci, b[i] as u16, cn) as u32 { bad[1] += 1; }
        }
        assert_eq!(bad, [0, 0], "q30 fmsub/fmadd mismatches");
        let d: Vec<u32> = (0..n).map(|_| { let r = next(); (((r % (1u64 << 32)) as i64 - (1i64 << 31)) as i32) as u32 }).collect();
        let f: Vec<u32> = (0..n).map(|_| (next() & 0xFFFF) as u32).collect();
        let (it, _) = p.run_once("p_interp", [&d, &f, &[], &o, &[], &[]], params(n as u32), groups(n), 4);
        let mut bad = 0;
        for i in 0..n {
            let want = (((d[i] as i32 as i64) * (f[i] as i64) + (1i64 << 15)) >> 16) as i32;
            if it[i] as i32 != want { bad += 1; }
        }
        assert_eq!(bad, 0, "interpolation term mismatches");
    }

    fn lut(name: &str) -> Option<Vec<u32>> {
        let dir = std::path::PathBuf::from("D:/folded-weights/v5/tests/oracles/lut");
        let bytes = std::fs::read(dir.join(name)).ok()?;
        Some(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    #[test]
    fn card_lut_decode_and_divide_match_v5() {
        let Some(p) = ops() else { return };
        let Some(lut) = lut("exp_n32768.bin") else { return };
        let mut next = rng(41);
        let n = 100_000;
        let a: Vec<u32> = (0..n).map(|_| lut[(next() % 32768) as usize]).collect();
        let b: Vec<u32> = (0..n).map(|_| lut[(next() % 4096) as usize]).collect();
        let o = vec![0u32; n];
        let (out, _) = p.run_once("p_lutdiv", [&a, &b, &[], &o, &[], &[]], params(n as u32), groups(n), 4);
        let mut bad = 0;
        for i in 0..n {
            if out[i] != f3_div_to_bf16(single_bits_to_f3(a[i]), single_bits_to_f3(b[i])) as u32 { bad += 1; }
        }
        assert_eq!(bad, 0, "lut decode + divide mismatches");
    }

    #[test]
    fn card_rmsnorm_rope_softmax_silu_add_match_cpu() {
        let Some(p) = ops() else { return };
        let (Some(rs), Some(ex), Some(cs)) = (lut("rsqrt_n32768.bin"), lut("exp_n32768.bin"), lut("cossin_q30_n65536.bin")) else { return };
        let ps_bytes = std::fs::read("D:/folded-weights/v5/tests/oracles/lut/phase_step_qwen3.bin").unwrap();
        let steps = crate::v5::rope::GeometricRopeIntLut::decode_phase_step(&ps_bytes);
        let mut next = rng(53);
        // rmsnorm: 40 rows of 2048 against V5
        let rms = crate::v5::rmsnorm::RmsNorm { rsqrt_lut: rs.clone() };
        let rows = 40;
        let dim = 2048;
        let x = patterns(&mut next, rows * dim, 110, 140);
        let gamma = patterns(&mut next, dim, 120, 132);
        let o = vec![0u32; rows * dim];
        let mut pr = params(rows as u32);
        pr[1] = dim as u32;
        pr[2] = crate::forward::RMS_EPS as u32;
        let (y, e) = p.run_once("k_rmsnorm", [&x, &gamma, &rs, &o, &[], &[]], pr, groups(rows), 4);
        assert_eq!(e, 0);
        let mut bad = 0;
        for r in 0..rows {
            let xr: Vec<u16> = x[r * dim..(r + 1) * dim].iter().map(|&v| v as u16).collect();
            let gr: Vec<u16> = gamma.iter().map(|&v| v as u16).collect();
            let want = rms.apply(&xr, &gr, crate::forward::RMS_EPS);
            for i in 0..dim { if y[r * dim + i] != want[i] as u32 { bad += 1; } }
        }
        assert_eq!(bad, 0, "rmsnorm mismatches");
        // rope: 16 rows × 4 heads of 128, positions 0..15 and some large
        let rope = crate::v5::rope::GeometricRopeIntLut { cossin_lut: crate::v5::rope::GeometricRopeIntLut::decode_cossin_q30(&std::fs::read("D:/folded-weights/v5/tests/oracles/lut/cossin_q30_n65536.bin").unwrap()), phase_step: steps.clone(), head_dim: 128 };
        let (rows, heads, hd) = (16usize, 4usize, 128usize);
        let q = patterns(&mut next, rows * heads * hd, 100, 140);
        let positions: Vec<u32> = (0..rows).map(|r| if r < 8 { r as u32 } else { 1000 + (next() % 30000) as u32 }).collect();
        let steps_lo: Vec<u32> = steps.iter().map(|&s| { assert!(s < (1u64 << 32)); s as u32 }).collect();
        let o = vec![0u32; rows * heads * hd];
        let mut pr = params((rows * heads * hd / 2) as u32);
        pr[1] = (heads * hd) as u32;
        pr[2] = hd as u32;
        pr[3] = heads as u32;
        let (y, e) = p.run_once("k_rope", [&q, &steps_lo, &cs, &o, &positions, &[]], pr, groups(rows * heads * hd / 2), 4);
        assert_eq!(e, 0);
        let mut bad = 0;
        for r in 0..rows {
            for h in 0..heads {
                let off = r * heads * hd + h * hd;
                let qs: Vec<u16> = q[off..off + hd].iter().map(|&v| v as u16).collect();
                let want = crate::v5::probe::RopeVariant::apply(&rope, &qs, positions[r] as u64);
                for i in 0..hd { if y[off + i] != want[i] as u32 { bad += 1; } }
            }
        }
        assert_eq!(bad, 0, "rope mismatches");
        // softmax: 12 rows × 3 heads over total 40, masked at positions, scaled
        let sm = crate::v5::softmax::ExpLutSoftmax { exp_lut: ex.clone() };
        let (rows, heads, total) = (12usize, 3usize, 40usize);
        let scores = patterns(&mut next, rows * heads * total, 118, 134);
        let positions: Vec<u32> = (0..rows).map(|r| (r * 3 + 1).min(total - 1) as u32).collect();
        let o = vec![0u32; rows * heads * total];
        let mut pr = params((rows * heads) as u32);
        pr[1] = total as u32;
        pr[2] = heads as u32;
        pr[3] = crate::forward::ATTN_SCALE_HD128 as u32;
        let (y, e) = p.run_once("k_softmax", [&scores, &ex, &o, &positions, &[], &[]], pr, groups(rows * heads), 3);
        assert_eq!(e, 0);
        let mut bad = 0;
        for r in 0..rows {
            for h in 0..heads {
                let base = (r * heads + h) * total;
                let row: Vec<u16> = scores[base..base + total].iter().map(|&v| bf16_mul(v as u16, crate::forward::ATTN_SCALE_HD128)).collect();
                let want = crate::v5::probe::SoftmaxVariant::apply(&sm, &row, positions[r] as usize);
                for j in 0..total { if y[base + j] != want[j] as u32 { bad += 1; } }
            }
        }
        assert_eq!(bad, 0, "softmax mismatches");
        // silu · u against the CPU's silu, and the exact add against matmul::add
        let n = 100_000;
        let g = patterns(&mut next, n, 100, 140);
        let u = patterns(&mut next, n, 100, 140);
        let o = vec![0u32; n];
        let (y, e) = p.run_once("k_silu", [&g, &u, &ex, &o, &[], &[]], params(n as u32), groups(n), 4);
        assert_eq!(e, 0);
        let one = single_bits_to_f3(0x3F80_0000);
        let mut bad = 0;
        for i in 0..n {
            let x = g[i] as u16;
            let ee = single_bits_to_f3(ex[(x & 0x7FFF) as usize]);
            let denom = f3_add(one, ee);
            let sigma = if (x & 0x8000) == 0 { f3_div_to_bf16(one, denom) } else { f3_div_to_bf16(ee, denom) };
            let want = bf16_mul(bf16_mul(x, sigma), u[i] as u16);
            if y[i] != want as u32 { bad += 1; }
        }
        assert_eq!(bad, 0, "silu mismatches");
        let a = patterns(&mut next, n, 60, 140);
        let b = patterns(&mut next, n, 60, 140);
        let (y, e) = p.run_once("k_add", [&a, &b, &o, &[], &[], &[]], params(n as u32), groups(n), 3);
        assert_eq!(e, 0);
        let au: Vec<u16> = a.iter().map(|&v| v as u16).collect();
        let bu: Vec<u16> = b.iter().map(|&v| v as u16).collect();
        let want = crate::matmul::add(&au, &bu);
        let bad = (0..n).filter(|&i| y[i] != want[i] as u32).count();
        assert_eq!(bad, 0, "add mismatches");
    }

    #[test]
    fn card_dense_and_proj_match_cpu() {
        let Some(p) = ops() else { return };
        let mut next = rng(67);
        // dense: matmul_dense on 6 × 128 against 9 × 128
        let (seq, in_f, out_f) = (6usize, 128usize, 9usize);
        let x = patterns(&mut next, seq * in_f, 100, 140);
        let w = patterns(&mut next, out_f * in_f, 100, 140);
        let o = vec![0u32; seq * out_f];
        let mut pr = params((seq * out_f) as u32);
        pr[1] = in_f as u32;
        pr[2] = out_f as u32;
        pr[4] = crate::matmul::FRAC_DENSE as u32;
        let (y, e) = p.run_once("k_dense", [&x, &w, &o, &[], &[], &[]], pr, groups(seq * out_f), 3);
        assert_eq!(e, 0);
        let xu: Vec<u16> = x.iter().map(|&v| v as u16).collect();
        let wu: Vec<u16> = w.iter().map(|&v| v as u16).collect();
        let want = crate::matmul::matmul_dense(&xu, &wu, seq, in_f, out_f);
        assert_eq!(y.iter().map(|&v| v as u16).collect::<Vec<_>>(), want, "dense");
        // proj: matmul_grid on a synthetic 256×256 grid, collapse on the card
        use crate::surfaces::{Atoms, Grid, Place};
        let mut dict = Vec::new();
        while dict.len() < 512 {
            let sign = ((next() & 1) as u16) << 15;
            let exp = (110 + (next() % 30)) as u16;
            let frac = (next() % 128) as u16;
            let pat = sign | (exp << 7) | frac;
            if crate::cells::is_finite(pat) && !dict.contains(&pat) { dict.push(pat); }
        }
        let atoms = Atoms::from_dict(&dict);
        let (rows, cols) = (256usize, 256usize);
        let ids: Vec<u16> = (0..rows * cols).map(|_| (next() % 512) as u16).collect();
        let (mut rmin, mut rmax) = (i32::MAX, i32::MIN);
        for &id in &ids { rmin = rmin.min(atoms.n[id as usize] as i32); rmax = rmax.max(atoms.n[id as usize] as i32); }
        let grid = Grid { rows, cols, th: 128, tw: 128, ids: ids.clone(), place: std::sync::Arc::new(Place::new(2, 2)), rmin, rmax };
        let xs: Vec<u16> = (0..3 * cols).map(|_| dict[(next() % 512) as usize]).collect();
        let want = crate::matmul::matmul_grid(&xs, 3, cols, &grid, &atoms);
        let ids_u32: Vec<u32> = ids.chunks_exact(2).map(|c| c[0] as u32 | ((c[1] as u32) << 16)).collect();
        let packed: Vec<u32> = (0..atoms.m.len()).map(|i| ((atoms.sheet[i] as u32) << 31) | ((atoms.m[i] as u32) << 16) | ((atoms.n[i] as i32 + 32768) as u32 & 0xFFFF)).collect();
        let xin: Vec<u32> = xs.iter().map(|&v| v as u32).collect();
        let o = vec![0u32; 3 * rows];
        let frac = crate::matmul::frac_for(rmin);
        let pr: [u32; 12] = [rows as u32, cols as u32, 3, 0, frac as u32, 0, 2, 2, 128, 128, 0, (3 * rows) as u32];
        let (y, e) = p.run_once("k_proj", [&ids_u32, &grid.place.rank, &packed, &xin, &o, &[]], pr, groups(3 * rows), 5);
        assert_eq!(e, 0);
        assert_eq!(y.iter().map(|&v| v as u16).collect::<Vec<_>>(), want, "proj");
        let xpk: Vec<u32> = xs.iter().map(|&v| { let (sh, m, n) = crate::cells::decompose(v); ((sh as u32) << 31) | ((m as u32) << 16) | ((n as i32 + 32768) as u32 & 0xFFFF) }).collect();
        let (y2, e2) = p.run_once("k_proj_wg", [&ids_u32, &grid.place.rank, &packed, &xpk, &o, &[]], pr, (3 * rows) as u32, 5);
        assert_eq!(e2, 0);
        assert_eq!(y2.iter().map(|&v| v as u16).collect::<Vec<_>>(), want, "proj_wg");
    }
}
