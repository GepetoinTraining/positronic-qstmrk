"""Hull and rewind: one real transformer layer (Qwen3-4B, layer 18) run as two interiors with a receipt layer.

  state        the residual stream, kept on an exact integer grid (value = integer * 2^-G)
  hull         attention: RMSNorm, q/k/v, per-head norm, rotary SPIN, softmax CHOOSE, o_proj. The mass damper.
               The only approximate arithmetic in the system lives here; what leaves it is snapped to the grid.
  interior 2   the MLP (gate, up, down), reading what the hull chose; also snapped to the grid on the way out.
  receipt      every change to state is written as (delta, rounding) -- MM's |x,K;K,x|: forward and receipt in one.
  turn         apply hull, apply interior 2, keep the output row, then rewind every receipt: state returns bit for bit.
"""
import numpy as np
G = 16                                                   # grid: 2^-16
def w(name): b = np.load(f"data/L18_{name}.npy"); return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
W = {k: w(k) for k in ["self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_o_proj", "mlp_gate_proj",
                       "mlp_up_proj", "mlp_down_proj", "input_layernorm", "post_attention_layernorm",
                       "self_attn_q_norm", "self_attn_k_norm"]}
H, KV, D, EPS, THETA = 32, 8, 128, 1e-6, 1e6

def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g
def spin(x, pos):                                        # rotary: dims d and d+64 are one complex number, turned by position
    inv = THETA ** (-np.arange(0, D, 2) / D); a = pos[:, None] * inv[None]; c, s = np.cos(a), np.sin(a)
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
def snap(delta):                                         # leave the hull: one rounding, kept in the receipt
    q = np.round(delta * 2.0**G).astype(np.int64); return q, delta - q * 2.0**-G

def hull(r):
    x = rms(r * 2.0**-G, W["input_layernorm"]); T = x.shape[0]; pos = np.arange(T, dtype=np.float64)
    q = (x @ W["self_attn_q_proj"].T).reshape(T, H, D); k = (x @ W["self_attn_k_proj"].T).reshape(T, KV, D)
    v = (x @ W["self_attn_v_proj"].T).reshape(T, KV, D)
    q = spin(rms(q, W["self_attn_q_norm"]).transpose(1, 0, 2), pos); k = spin(rms(k, W["self_attn_k_norm"]).transpose(1, 0, 2), pos)
    k = np.repeat(k, H // KV, 0); v = np.repeat(v.transpose(1, 0, 2), H // KV, 0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(D); s = s + np.triu(np.full((T, T), -np.inf), 1)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)          # choose
    o = (p @ v).transpose(1, 0, 2).reshape(T, H * D) @ W["self_attn_o_proj"].T
    return snap(o), p

def interior2(r):
    x = rms(r * 2.0**-G, W["post_attention_layernorm"]); g = x @ W["mlp_gate_proj"].T
    return snap(((g / (1 + np.exp(-g))) * (x @ W["mlp_up_proj"].T)) @ W["mlp_down_proj"].T)

def turn(r):
    """One turn on state r (int64). Returns the kept output row, the receipts, and r (mutated then rewound)."""
    receipts = []
    (da, ea), attn = hull(r); r += da; receipts.append(("hull", da, ea))
    (dm, em) = interior2(r); r += dm; receipts.append(("interior2", dm, em))
    out = r[-1].copy()                                   # keep only the output
    for _, d, _ in reversed(receipts): r -= d            # release: rewind every receipt
    return out, receipts, attn
