import json, struct, subprocess, sys, numpy as np
BASE = "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/"
def rng(url, a, b):
    return subprocess.run(["curl", "-sL", "-r", f"{a}-{b}", url], capture_output=True, check=True).stdout
def header(shard):
    url = BASE + shard; n = struct.unpack("<Q", rng(url, 0, 7))[0]
    return url, 8 + n, json.loads(rng(url, 8, 8 + n - 1))
def get(name, shard, cache={}):
    if shard not in cache: cache[shard] = header(shard)
    url, off, h = cache[shard]; t = h[name]; a, b = t["data_offsets"]
    raw = rng(url, off + a, off + b - 1); assert len(raw) == b - a, (name, len(raw), b - a)
    bits = np.frombuffer(raw, dtype="<u2").reshape(t["shape"])      # bf16, kept as its exact bits
    return bits, t["dtype"]
if __name__ == "__main__":
    want = [(f"model.layers.{L}.{k}.weight", s) for L, s in [(0, "model-00001-of-00003.safetensors"), (18, "model-00002-of-00003.safetensors")]
            for k in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]
    for name, shard in want:
        bits, dt = get(name, shard); np.save(name.replace(".", "_") + ".npy", bits)
        print(name, dt, bits.shape, f"{bits.nbytes/1e6:.1f} MB")
