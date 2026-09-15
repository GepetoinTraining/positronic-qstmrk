"""cells.py — the bf16 pattern as a cell on the two-sheeted lattice (Atlas Engine §1.2).

A bf16 pattern is sixteen bits: sheet(1) | exp(8) | mant(7). No number type is ever
made from it; it is decomposed by shifts and masks into the cell

    ⟨ m | n ⟩ on sheet s        value = s · m · 2^n

    s   the sheet, + or −  (bit 15)
    m   the core: the ODD significand, u8, one of the 128 odd atoms 1,3,…,255; 0 for zero
    n   the rung: the signed 2-exponent that puts the odd core at its place

Normals carry the implicit 1: sig = 128 | mant (8 bits, 128..255). Subnormals carry
sig = mant (0..127). In both, tz = trailing zeros of sig, m = sig >> tz, and
n = (exp − 127 − 7) + tz for normals, (1 − 127 − 7) + tz for subnormals. Zero is
m = 0, n = 0, on whichever sheet the pattern names (−0 and +0 are two names, both kept).

The inverse rebuilds the pattern exactly, and the census gate below proves it on every
one of the 65,280 finite patterns (65,536 minus the 256 with exp = 255). Non-finite
patterns have no cell: they are refused, never mapped.

Integer only. This file is gated by intake/gate_source.py (byte needles).
"""
from __future__ import annotations

import numpy as np

U16 = np.uint16
I32 = np.int32

FINITE_PATTERNS = 65280  # 65536 − 256 (exp == 255: inf / nan)

# bit_length(x) − 1 for x in 0..255 (bit_length(0) := 0 → −1 is never used: m = 0 is zero)
_HIBIT = np.zeros(256, dtype=I32)
for _x in range(1, 256):
    _b = 0
    _y = _x
    while _y > 1:
        _y >>= 1
        _b += 1
    _HIBIT[_x] = _b

# trailing zeros for x in 0..255 (tz(0) := 0)
_TZ = np.zeros(256, dtype=I32)
for _x in range(1, 256):
    _t = 0
    _y = _x
    while _y & 1 == 0:
        _y >>= 1
        _t += 1
    _TZ[_x] = _t


def is_finite(p: np.ndarray) -> np.ndarray:
    """True where the pattern has exp != 255."""
    p = p.astype(U16, copy=False)
    return ((p >> 7) & 0xFF) != 0xFF


def decompose(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pattern u16 → (sheet u8 in {0,1}, core m u8 odd-or-zero, rung n i32).
    Caller guarantees finiteness (see is_finite); non-finite input is a bug here."""
    p = p.astype(U16, copy=False)
    sheet = (p >> 15).astype(np.uint8)
    exp = ((p >> 7) & 0xFF).astype(I32)
    mant = (p & 0x7F).astype(I32)
    normal = exp != 0
    sig = np.where(normal, mant | 0x80, mant)             # 0..255
    tz = _TZ[sig]
    m = (sig >> tz).astype(np.uint8)                       # odd, or 0
    base = np.where(normal, exp - 127 - 7, 1 - 127 - 7)
    n = np.where(sig != 0, base + tz, 0).astype(I32)
    return sheet, m, n


def recompose(sheet: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    """(sheet, m, n) → pattern u16. Exact inverse of decompose on every finite pattern."""
    m32 = m.astype(I32)
    n32 = n.astype(I32)
    k = _HIBIT[m32]                                        # 0..7, top bit of the core
    sig = m32 << (7 - k)                                   # 128..255 (or 0)
    rung_sig = n32 - (7 - k)                               # rung of the 8-bit sig
    exp = rung_sig + 7 + 127                               # biased exponent if normal
    is_zero = m32 == 0
    sub = (exp <= 0) & ~is_zero                            # subnormal: shift sig down
    shift = np.where(sub, 1 - exp, 0)
    sig_sub = sig >> np.minimum(shift, 8)
    mant = np.where(sub, sig_sub, sig & 0x7F)
    exp_out = np.where(sub | is_zero, 0, exp)
    out = ((sheet.astype(I32) << 15) | (exp_out << 7) | mant).astype(U16)
    return out


def census() -> tuple[int, int]:
    """Round-trip every finite bf16 pattern through the cell. Returns (checked, exact)."""
    allp = np.arange(65536, dtype=np.int64).astype(U16)
    fin = allp[is_finite(allp)]
    s, m, n = decompose(fin)
    back = recompose(s, m, n)
    return int(fin.size), int(np.count_nonzero(back == fin))


def binade_of(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The binade a cell sits in: the rung of its 8-bit sig = n + (7 − hibit(m)) … i.e.
    floor(log2 |value|) = n + hibit(m). Integer only."""
    return n.astype(I32) + _HIBIT[m.astype(I32)]
