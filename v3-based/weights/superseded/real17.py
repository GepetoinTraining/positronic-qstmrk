"""A real forward from local u16: Qwen3-1.7B layers 0-13, from the archived BF16 weights (u16 = the weights, read as
positions: band byte and address byte), on the captured real layer-0 input, against the captured layer-1 and layer-14
inputs. Projections exact (rules.apply: integer band windows, BLAS-carried below 2^53, joined by shifts); attention
and the norms kept as they are. Then the relational read (all 16 bands) on the real inputs of layer 0.
Run from v3-based/weights:  python real17.py   (numpy)
"""
import os, time
import numpy as np
import t14, pack, rules

ROOT = os.path.join(t14.HERE, "..", "..")
WDIR = os.path.join(t14.HERE, "..", "ARCHIVE-2026-09-25-mistaken", "weights")
CAP = os.path.join(ROOT, "receipts")
H, KV, D, INTER, HID, TOK = 16, 8, 128, 6144, 2048, 64
SHAPE = {"self_attn.q_proj": (H * D, HID), "self_attn.k_proj": (KV * D, HID), "self_attn.v_proj": (KV * D, HID),
         "self_attn.o_proj": (HID, H * D), "mlp.gate_proj": (INTER, HID), "mlp.up_proj": (INTER, HID), "mlp.down_proj": (HID, INTER)}
NORM = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"]

def u16(path, shape=None):
    a = np.fromfile(path, dtype="<u2"); return a.reshape(shape) if shape else a
def capture(L):
    return t14.bits_to_f64(u16(os.path.join(CAP, f"capture-qwen3-1.7b-L{L}-508tok.u16"), (508, HID))[:TOK])

def seat_layer(L):
    rl = {}
    for n, sh in SHAPE.items():
        code, floor = pack.pack(u16(os.path.join(WDIR, f"model.layers.{L}.{n}.weight.u16"), sh)); rl[n] = (code, floor)
    nm = {n: t14.bits_to_f64(u16(os.path.join(WDIR, f"model.layers.{L}.{n}.weight.u16"))) for n in NORM}
    return rl, nm

def proj(rl, n, x, keep=None):
    xi = t14.gridx(x)
    if keep is not None: keep[n] = xi
    N, s = rules.apply(rl[n], xi); return t14.to_float(N, s - t14.G)

def layer(r, rl, nm, keep=None):
    x = t14.rms(r, nm["input_layernorm"]); T = x.shape[0]; pos = np.arange(T, dtype=np.float64)
    q = proj(rl, "self_attn.q_proj", x, keep).reshape(T, H, D); k = proj(rl, "self_attn.k_proj", x, keep).reshape(T, KV, D)
    v = proj(rl, "self_attn.v_proj", x, keep).reshape(T, KV, D)
    q = t14.spin(t14.rms(q, nm["self_attn.q_norm"]).transpose(1, 0, 2), pos); k = t14.spin(t14.rms(k, nm["self_attn.k_norm"]).transpose(1, 0, 2), pos)
    k = np.repeat(k, H // KV, 0); v = np.repeat(v.transpose(1, 0, 2), H // KV, 0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(D); s = s + np.triu(np.full((T, T), -np.inf), 1)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    r = r + proj(rl, "self_attn.o_proj", (p @ v).transpose(1, 0, 2).reshape(T, H * D), keep)
    x2 = t14.rms(r, nm["post_attention_layernorm"]); g = proj(rl, "mlp.gate_proj", x2, keep)
    return r + proj(rl, "mlp.down_proj", (g / (1 + np.exp(-g))) * proj(rl, "mlp.up_proj", x2, keep), keep)

def to_bf16_bits(v):
    b = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)          # round to nearest even

def compare(ours, L):
    cap = u16(os.path.join(CAP, f"capture-qwen3-1.7b-L{L}-508tok.u16"), (508, HID))[:TOK]
    mine = to_bf16_bits(ours)
    same = int((mine == cap).sum())
    ulp = np.abs(mine.astype(np.int64) - cap.astype(np.int64))[((mine ^ cap) & 0x8000) == 0]
    ref = t14.bits_to_f64(cap)
    cos = float((ours * ref).sum() / np.sqrt((ours * ours).sum() * (ref * ref).sum()))
    return f"BF16-identical {same:,} of {cap.size:,}; largest gap {int(ulp.max())} BF16 steps (same-sign); cosine {cos:.6f} (instrument)"

if __name__ == "__main__":
    t0 = time.time()
    r = capture(0); keep = {}
    for L in range(14):
        rl, nm = seat_layer(L)
        r = layer(r, rl, nm, keep if L == 0 else None)
        if L == 0:
            print(f"[L0->L1] {compare(r, 1)}", flush=True)
            l0 = rl
        del rl
    print(f"[L0->L14] {compare(r, 14)}  ({time.time() - t0:.0f}s for 14 layers, {TOK} tokens)", flush=True)

    print("\n[bands] imaginary / band 0 on layer 0's real inputs, against the same projection on random inputs", flush=True)
    rng = np.random.default_rng(17)
    for n in SHAPE:
        xi = keep[n]; T = xi.shape[0]
        def ratio(x):
            N, s = rules.apply(l0[n], np.concatenate([rules.moved(x, c) for c in range(16)], axis=0))
            E = [sum(int(v) * int(v) for v in N[c * T:(c + 1) * T].ravel()) for c in range(16)]
            return sum(E[1:]) / E[0]
        rand = np.round(rng.normal(0, 1.0, xi.shape) * float(np.abs(xi).std())).astype(np.int64)
        print(f"  {n:18s} real {ratio(xi):8.3f}   random {ratio(rand):8.3f}", flush=True)
    print(f"\ndone ({time.time() - t0:.0f}s)")
