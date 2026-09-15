"""ringing.py — Prediction 6 (native_object.pdf): is the wave in the ringing?

INSTRUMENT, not runtime. The engine wrote the residual stream exactly (bf16 patterns, sealed);
this script reads them and measures spectral flatness with float as the ruler — the same
instrument the paper used on the weights (Prop. 4), marked as such. Nothing here feeds the pass.

    python intake/ringing.py <model-id> <layer> <ntok> [--weights <tensor-name>]

Per position vector (hidden values): flatness = geometric mean / arithmetic mean of the power
spectrum (1.0 = white), in (a) index order, (b) the unsigned Fiedler order of the layer's
q_proj column correlation, (c) the signed Fiedler order — the three orderings of Prop. 4 —
each against the same vector shuffled (the null). Also the weights' own rows under the same
instrument, so the ringing and the bell are read on one ruler. And, exact and separate: the
NTT power identity P_k = X_k · X_{N-k} over Z/998244353 for position 0 (ntt.py's rotation-free
identity), as a receipt of the vector, computed in integers only.
"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, "intake")
import cells  # noqa: E402

M = 998244353
G = 3


def values(pat):
    """bf16 patterns → values (float, the instrument)."""
    s, m, n = cells.decompose(pat)
    v = m.astype(np.float64) * np.exp2(n.astype(np.float64))
    return np.where(s == 1, -v, v)


def flatness(x):
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    p = p[1:]  # drop DC
    p = p[p > 0]
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def deciles(x):
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    p = p[1:]
    tot = p.sum()
    edges = np.linspace(0, len(p), 11).astype(int)
    return [float(p[edges[i]:edges[i + 1]].sum() / tot) for i in range(10)]


def fiedler_orders(w):
    """w: [rows, cols] values. Column correlation → unsigned/signed Laplacian Fiedler orders."""
    c = np.corrcoef(w.T)
    c = np.nan_to_num(c)
    np.fill_diagonal(c, 0)
    out = {}
    for name, a in (("fiedler_unsigned", np.abs(c)), ("fiedler_signed", c)):
        d = np.diag(np.abs(a).sum(1))
        lap = d - a
        vals, vecs = np.linalg.eigh(lap)
        out[name] = np.argsort(vecs[:, 1])
    return out


def ntt_power_identity(pat):
    """Exact: values as integers m·2^(n − nmin), NTT over Z/M (N = 2048), P_k = X_k·X_{N−k} mod M.
    Returns sha256 of the P_k list and a shift-invariance check (rotate the input by 1: same P)."""
    s, m, n = cells.decompose(pat)
    nmin = int(n[m != 0].min())
    ints = [(-1 if int(s[i]) else 1) * int(m[i]) * (1 << (int(n[i]) - nmin)) if m[i] else 0 for i in range(len(pat))]
    N = len(ints)
    assert N & (N - 1) == 0 and (M - 1) % N == 0
    w = pow(G, (M - 1) // N, M)

    def ntt(a):
        a = [x % M for x in a]
        j = 0
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        ln = 2
        while ln <= N:
            wl = pow(w, N // ln, M)
            for i in range(0, N, ln):
                wn = 1
                for k in range(ln // 2):
                    u, v = a[i + k], a[i + k + ln // 2] * wn % M
                    a[i + k], a[i + k + ln // 2] = (u + v) % M, (u - v) % M
                    wn = wn * wl % M
            ln <<= 1
        return a

    X = ntt(ints)
    P = [X[k] * X[(N - k) % N] % M for k in range(N)]
    X2 = ntt(ints[1:] + ints[:1])
    P2 = [X2[k] * X2[(N - k) % N] % M for k in range(N)]
    h = hashlib.sha256(b"".join(int(p).to_bytes(4, "little") for p in P)).hexdigest()
    return h, P == P2


def main(mid, layer, ntok, wname=None):
    stem = f"receipts/capture-{mid}-L{layer}-{ntok}tok"
    meta = open(stem + ".meta").read().strip()
    hidden = int(meta.split("hidden=")[1].split()[0])
    pat = np.fromfile(stem + ".u16", dtype=np.uint16).reshape(-1, hidden)
    print(meta)
    rng = np.random.default_rng(7)
    orders = {"index": np.arange(hidden)}
    wrows = None
    if wname:
        import json, struct, os
        man = json.load(open(f"receipts/intake-{mid}-20260913/MANIFEST.json"))
        d = np.fromfile(f"receipts/intake-{mid}-20260913/DICT.u16", dtype=np.uint16)
        ent = next(t for t in man["tensors"] if t["name"] == wname)
        ids = np.fromfile(ent["grid"], dtype=np.uint16)
        # unfold Morton grid → dense rows (same as intake.unfold)
        sys.path.insert(0, "intake"); import intake as ik
        R, C = ent["shape"]; order = ik.tile_order(R // 128, C // 128)
        dense = d[ik.unfold(ids, order, R, C, 128, 128)]
        wrows = values(dense.reshape(-1))
        wrows = wrows.reshape(R, C)
        orders.update(fiedler_orders(wrows[:2048] if R > 2048 else wrows))
        print(f"orderings from {wname} column correlation: {list(orders)}")
    print(f"\n== ringing: {pat.shape[0]} positions x {hidden} values, layer {layer} ==")
    print(f"{'ordering':18s} {'flatness (mean over positions)':>32s} {'shuffled null':>14s} {'ratio':>7s}")
    for oname, ordr in orders.items():
        f = []; f0 = []
        for i in range(pat.shape[0]):
            v = values(pat[i])[ordr]
            f.append(flatness(v))
            f0.append(np.mean([flatness(rng.permutation(v)) for _ in range(5)]))
        print(f"{oname:18s} {np.mean(f):32.4f} {np.mean(f0):14.4f} {np.mean(f)/np.mean(f0):7.3f}")
    v0 = values(pat[0])
    print("power by frequency decile, position 0, index order:", " ".join(f"{d:.3f}" for d in deciles(v0)))
    if pat.shape[0] >= 64:
        # along positions, per dimension: is the sequence axis white?
        fs = [flatness(values(pat[:, j])) for j in range(0, hidden, 64)]
        fs0 = [np.mean([flatness(rng.permutation(values(pat[:, j]))) for _ in range(3)]) for j in range(0, hidden, 64)]
        print(f"along positions (every 64th dim): flatness {np.mean(fs):.4f} vs shuffled {np.mean(fs0):.4f} ratio {np.mean(fs)/np.mean(fs0):.3f}")
    if wrows is not None:
        f = [flatness(wrows[r]) for r in range(0, min(wrows.shape[0], 1024), 8)]
        f0 = [flatness(rng.permutation(wrows[r])) for r in range(0, min(wrows.shape[0], 1024), 8)]
        print(f"\n== the bell: {wname} rows, index order: flatness {np.mean(f):.4f} vs shuffled {np.mean(f0):.4f} ratio {np.mean(f)/np.mean(f0):.3f}")
    h, inv = ntt_power_identity(pat[0])
    print(f"\n== exact NTT power identity (Z/998244353, N={hidden}), position 0: sha256 {h[:16]}…, rotation-invariant: {inv}")


if __name__ == "__main__":
    w = sys.argv[sys.argv.index("--weights") + 1] if "--weights" in sys.argv else None
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), w)
