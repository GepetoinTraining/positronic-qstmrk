"""ringing_positions.py — Prediction 6 along the sequence axis (companion to ringing.py; INSTRUMENT).

    python intake/ringing_positions.py <model-id> <ntok> <layer> [<layer> ...] [--tag <name>]

For each captured layer: per hidden dimension j, the sequence x_j[pos] (bf16 → float, the ruler),
flatness of its power spectrum along positions vs the same sequence shuffled (null), ALL dims.
Then the mean power spectrum by frequency decile (where the colour lives: decile 1 = drift / slow
wave, decile 10 = alternation), and the same with position 0 removed (the sink) and with the
per-position mean removed (the common mode across dims). The layer-0 capture is the input
control: embeddings only, no attention — colour already there is the prompt's, not the model's.
"""
import sys

import numpy as np

sys.path.insert(0, "intake")
from ringing import values, flatness  # noqa: E402


def spectrum_deciles(x2d):
    """x2d: [pos, dims] → mean over dims of the normalised power spectrum along pos, by decile."""
    x = x2d - x2d.mean(0)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    p = p[1:]
    p = p / p.sum(0, keepdims=True)
    m = p.mean(1)
    edges = np.linspace(0, len(m), 11).astype(int)
    return [float(m[edges[i]:edges[i + 1]].sum()) for i in range(10)]


def read(mid, layer, ntok, tag=""):
    stem = f"receipts/capture-{mid}-L{layer}-{ntok}tok{tag}"
    meta = open(stem + ".meta").read().strip()
    hidden = int(meta.split("hidden=")[1].split()[0])
    pat = np.fromfile(stem + ".u16", dtype=np.uint16).reshape(-1, hidden)
    return values(pat.reshape(-1)).reshape(pat.shape)


def main(mid, ntok, layers, tag=""):
    rng = np.random.default_rng(7)
    print(f"{'layer':>5s} {'dims':>5s} {'flat all':>9s} {'null':>7s} {'ratio':>6s} | {'no pos0':>8s} {'ratio':>6s} | {'common-mode removed':>20s} {'ratio':>6s} | deciles of mean spectrum (all dims)")
    for layer in layers:
        x = read(mid, layer, ntok, tag)  # [pos, hidden]
        pos, hidden = x.shape
        def block(xx):
            f = np.array([flatness(xx[:, j]) for j in range(hidden)])
            f0 = np.array([flatness(rng.permutation(xx[:, j])) for j in range(hidden)])
            return f.mean(), f0.mean()
        fa, na = block(x)
        fb, nb = block(x[1:])
        xc = x - x.mean(1, keepdims=True)
        fc, nc = block(xc)
        dec = spectrum_deciles(x)
        print(f"{layer:5d} {hidden:5d} {fa:9.4f} {na:7.4f} {fa/na:6.3f} | {fb:8.4f} {fb/nb:6.3f} | {fc:20.4f} {fc/nc:6.3f} | " + " ".join(f"{d:.3f}" for d in dec))
        # magnitude profile along positions: is position 0 a sink?
        norms = np.sqrt((x ** 2).sum(1))
        print(f"      |x| by position: p0 {norms[0]:.1f}  p1 {norms[1]:.1f}  median {np.median(norms):.1f}  max {norms.max():.1f} at {int(norms.argmax())}")


if __name__ == "__main__":
    tag = "-" + sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    layers = [int(a) for a in sys.argv[3:] if a.isdigit()]
    main(sys.argv[1], int(sys.argv[2]), layers, tag)
