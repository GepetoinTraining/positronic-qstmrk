"""tokenizer.py — the stage before the pass: text → the rows of the embedding page (pgarcia:
"the embedding row is the token decomposed into values; grab the word and match it PRIOR to
starting; build a customish tokenizer").

The vocabulary IS the first column of the embedding page: row i of `model.embed_tokens.weight`
is the word `vocab[i]`. This module reads the model's own tokenizer.json (NFC normalizer, the
pre-tokenization regex, byte-level alphabet, BPE merges by rank, added tokens) and encodes with
integers only: merge ranks are integers, the byte map is a 256-entry table. No library does the
work; the reference implementation (`tokenizers`) is used ONLY as the gate — token ids are
integers, so equality is a receipt, not an anchor on the retired oracle.

    python intake/tokenizer.py <model-dir> [--gate]            # gate against the reference on a corpus
    python intake/tokenizer.py <model-dir> --encode "text"     # ids
    python intake/tokenizer.py <model-dir> --chat "user text"  # ids of the chat-template turn
"""
import hashlib
import json
import sys
import unicodedata

import regex


def bytes_to_unicode():
    """GPT-2 byte-level alphabet: 256 bytes → 256 printable code points (a fixed bijective table)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tokenizer:
    def __init__(self, model_dir):
        spec = json.load(open(f"{model_dir}/tokenizer.json", encoding="utf-8"))
        self.spec_sha = hashlib.sha256(open(f"{model_dir}/tokenizer.json", "rb").read()).hexdigest()
        m = spec["model"]
        assert m["type"] == "BPE"
        self.vocab = m["vocab"]  # token string (byte-level alphabet) → id
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        merges = m["merges"]
        self.ranks = {}
        for r, mg in enumerate(merges):
            a, b = (mg if isinstance(mg, list) else mg.split(" "))
            self.ranks[(a, b)] = r
        pre = spec["pre_tokenizer"]["pretokenizers"][0]
        assert pre["type"] == "Split"
        self.pat = regex.compile(pre["pattern"]["Regex"])
        assert spec["normalizer"]["type"] == "NFC"
        self.byte_map = bytes_to_unicode()
        self.byte_unmap = {v: k for k, v in self.byte_map.items()}
        self.added = {a["content"]: a["id"] for a in spec["added_tokens"]}
        for k, v in self.added.items():
            self.id_to_token[v] = k
        # added tokens are matched before any pre-tokenization, longest first
        self.added_pat = regex.compile("|".join(regex.escape(t) for t in sorted(self.added, key=len, reverse=True)))
        self.cache = {}

    def bpe(self, word):
        """Merge the byte-level characters of one pre-token by ascending merge rank."""
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            i = best[1]
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        # the same rank never appears twice, so "lowest rank first, leftmost on ties" is the reference order
        self.cache[word] = parts
        return parts

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = []
        pos = 0
        for am in self.added_pat.finditer(text):
            ids.extend(self._encode_plain(text[pos:am.start()]))
            ids.append(self.added[am.group(0)])
            pos = am.end()
        ids.extend(self._encode_plain(text[pos:]))
        return ids

    def _encode_plain(self, text):
        ids = []
        for m in self.pat.finditer(text):
            word = "".join(self.byte_map[b] for b in m.group(0).encode("utf-8"))
            for piece in self.bpe(word):
                ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            t = self.id_to_token[i]
            if t in self.added:
                out.extend(t.encode("utf-8"))
            else:
                out.extend(self.byte_unmap[c] for c in t)
        return out.decode("utf-8", errors="replace")

    def chat(self, user, system=None):
        """The Qwen3 chat template for one user turn (no thinking block), as ids."""
        s = ""
        if system:
            s += f"<|im_start|>system\n{system}<|im_end|>\n"
        s += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return self.encode(s)


CORPUS = [
    "The capital of France is",
    "Hello, world!",
    "  leading spaces and\ttabs\n\nnewlines\r\n",
    "don't stop; they're here, we'll see, I'd say it's 1234 5.67 numbers-and-dashes",
    "Olá, mundo! Coração, ação, São Paulo — açúcar.",
    "日本語のテキスト、中文文本，한국어 텍스트.",
    "emoji 🙂🚀👍🏽 and symbols ∑∫√ ≈ ≠ ±",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "def f(x):\n    return x**2  # comment\n",
    "MixedCASE WORDS and CamelCaseWords and snake_case_words",
    "a" * 300,
    "1234567890" * 5,
    "é́ combining marks é vs é",
    "",
]


def gate(model_dir):
    from tokenizers import Tokenizer as Ref
    ref = Ref.from_file(f"{model_dir}/tokenizer.json")
    tk = Tokenizer(model_dir)
    lines = [f"format=atlas.tokenizer.v1\ntokenizer.json sha256 {tk.spec_sha}\nvocab {len(tk.vocab)} merges {len(tk.ranks)} added {len(tk.added)}\n"]
    ok = 0
    for s in CORPUS:
        mine = tk.encode(s)
        theirs = ref.encode(s, add_special_tokens=False).ids
        same = mine == theirs
        rt = tk.decode(mine) == unicodedata.normalize("NFC", s)
        ok += same
        lines.append(f"[{'EQUAL' if same else 'DIFF '}] round-trip {'ok' if rt else 'NO'} {len(mine):4d} ids  {s[:48]!r}\n")
        if not same:
            lines.append(f"   mine   {mine[:24]}\n   theirs {theirs[:24]}\n")
    france = tk.encode("The capital of France is")
    lines.append(f"France = {france} (the engine's reference prompt: [785, 6722, 315, 9625, 374])\n")
    verdict = f"gate (tokenizer): {'PASS' if ok == len(CORPUS) and france == [785, 6722, 315, 9625, 374] else 'FAIL'} — {ok}/{len(CORPUS)} strings equal to the reference\n"
    lines.append(verdict)
    rep = "".join(lines)
    open("receipts/tokenizer-qwen3-1.7b.txt", "w", encoding="utf-8").write(rep)
    sys.stdout.buffer.write(rep.encode("utf-8"))


if __name__ == "__main__":
    md = sys.argv[1]
    if "--encode" in sys.argv:
        print(Tokenizer(md).encode(sys.argv[sys.argv.index("--encode") + 1]))
    elif "--chat" in sys.argv:
        print(Tokenizer(md).chat(sys.argv[sys.argv.index("--chat") + 1]))
    else:
        gate(md)
