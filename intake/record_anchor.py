"""record_anchor.py — ONE-SHOT ORACLE, NOT RUNTIME CODE. Gate 4's reference.

Runs the reference implementation (transformers, bf16, CPU, greedy) on the anchor inputs
ONCE and writes what it says to receipts/anchor-<model>.json. After this file exists the
reference is never run again ("recorded once, never re-run as a reference tax" — spec §7.4).
The engine is judged against the recorded ids, never against live transformers.

Anchor inputs (mirroring V5's, so the two models' receipts have the same shape):
  seq-1:  token [1]                 → argmax, top-5 ids
  seq-8:  tokens [1..8]             → per-position argmax
  prompt: "The capital of France is" → token ids, argmax, top-5, decoded piece

    python intake/record_anchor.py models/qwen3-1.7b qwen3-1.7b

This is the only file in intake/ that may touch a number type; it is excluded from gate 1
by name (it never runs in the pass).
"""
import hashlib
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)


def main(model_dir, model_id):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "receipts", "anchor-%s.json" % model_id)
    if os.path.exists(out):
        sys.exit("anchor already recorded at %s — not re-running the reference" % out)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    rec = {"model": model_id, "source": os.path.abspath(model_dir), "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "reference": {"transformers": __import__("transformers").__version__, "torch": torch.__version__, "dtype": "bfloat16", "device": "cpu"},
           "config": {k: getattr(cfg, k, None) for k in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                                                          "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings", "rms_norm_eps", "rope_theta"]},
           "anchors": {}}

    def run(ids):
        with torch.no_grad():
            logits = model(torch.tensor([ids]), use_cache=False).logits[0]
        last = logits[-1]
        top5 = torch.topk(last, 5).indices.tolist()
        per_pos = logits.argmax(dim=-1).tolist()
        return {"tokens": ids, "argmax": int(last.argmax()), "top5": top5, "per_position_argmax": per_pos,
                "top5_pieces": [tok.decode([t]) for t in top5]}

    rec["anchors"]["seq1"] = run([1])
    rec["anchors"]["seq8"] = run(list(range(1, 9)))
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    rec["anchors"]["france"] = run(ids)
    rec["seconds"] = round(time.time() - t0, 1)
    body = json.dumps(rec, indent=1, sort_keys=True)
    rec["seal"] = hashlib.sha256(body.encode()).hexdigest()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", newline="\n").write(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
