"""The geometric read: positions move and land; nothing is multiplied at runtime.

A value is a position: sign, address (one of the 128 odd atoms), band. The weights are their u16 as they sit on disk;
the activations are the model's own BF16 positions.
A weight meeting an activation is a move:   sign xor sign,   band + band,   ADDR[a][b]  (the address table, built once).
A projection is where the moves land: each move drops its address result onto its band (landing counts per band),
and each row is joined once at the end by shifting its bands into place. The gate: equal, bit for bit, to the exact
integer reference on the same activations. Real inputs: Qwen3-1.7B layer 0 on the captured tokens.
Run from v3-based/weights:  python geo.py   (numpy)
"""
import os, time
import numpy as np
import t14, pack, rules, real17

ODD = np.arange(1, 256, 2, dtype=np.int64)
ADDR = ODD[:, None] * ODD[None, :]                    # built once: what two atoms make (the madness)

def positions_w(rule):
    code, floor = rule; c = code.astype(np.int64)
    return (c & 127), (c >> 7) & 1, (c >> 8) + floor, c != 0         # address index, sign, band, present

def positions_x(xf):
    bits = real17.to_bf16_bits(xf)                                   # the activation as its BF16 position
    a, b = t14.decompose(bits)
    return (np.abs(a.astype(np.int64)) - 1) // 2, (a < 0).astype(np.int64), b.astype(np.int64), a != 0, bits

def land(pw, px):
    """One token: every weight meets the activation at its column; moves land on bands; rows joined by shifts."""
    aw, sw, ew, nw = pw; ax, sx, fx, nx = px[:4]
    g0 = int(ew[nw].min()) + int(fx[nx].min()); G = int(ew[nw].max()) + int(fx[nx].max()) - g0 + 1
    S = np.zeros((aw.shape[0], G), np.int64)
    for r0 in range(0, aw.shape[0], 1024):
        sl = slice(r0, r0 + 1024)
        live = nw[sl] & nx[None, :]
        g = ew[sl] + fx[None, :] - g0                                # band + band
        A = ADDR[aw[sl], ax[None, :]]                                # the address table, read
        val = np.where(live, np.where(sw[sl] ^ sx[None, :], -A, A), 0)  # sign xor sign
        rows = np.arange(val.shape[0])[:, None]
        key = (rows * G + np.where(live, g, 0)).ravel()               # where each move lands: its row, its band
        S[sl] = np.bincount(key, weights=val.ravel().astype(np.float64), minlength=val.shape[0] * G).reshape(-1, G).astype(np.int64)
    return t14.join([(S[:, k], k) for k in range(G)]), g0            # each row, its bands shifted into place, once

if __name__ == "__main__":
    TOK = 8
    r = real17.capture(0)[:TOK]
    rl, nm = real17.seat_layer(0)
    xs = {}
    def rec(rl_, n, x, keep=None):
        xs[n] = x.copy(); N, s = rules.apply(rl_[n], t14.gridx(x)); return t14.to_float(N, s - t14.G)
    real17.proj = rec
    real17.layer(r, rl, nm)                                         # the real layer 0, to capture each projection's input

    ok_all = True; t_geo = t_ref = 0.0
    print(f"layer 0 of Qwen3-1.7B, {TOK} real tokens: each projection read by landing, against the exact integer reference")
    for n in real17.SHAPE:
        pw = positions_w(rl[n]); tot_ok = True
        for t in range(TOK):
            px = positions_x(xs[n][t])
            t0 = time.time(); Ng, sg = land(pw, px); t_geo += time.time() - t0
            a, b = t14.decompose(px[4]); fmin = int(b[a != 0].min())
            xi = np.where(a != 0, a.astype(np.int64) << np.where(a != 0, b.astype(np.int64) - fmin, 0), 0)[None, :]
            t0 = time.time(); Nr, sr = rules.apply(rl[n], xi); t_ref += time.time() - t0
            tot_ok &= t14.same((Ng[None, :], sg), (Nr, sr + fmin))
        ok_all &= tot_ok
        print(f"  {n:18s} {str(real17.SHAPE[n]):13s} landing == exact reference on all {TOK} tokens: {tot_ok}", flush=True)
    print(f"\n{'PASS' if ok_all else 'FAIL'}  every projection of the real layer, read by moves and landings (no runtime multiply), "
          f"equals the exact reference bit for bit.  landing {t_geo:.1f}s, reference {t_ref:.1f}s ({TOK} tokens x 7 projections)")
