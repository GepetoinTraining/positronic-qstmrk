"""chat.py — the harness: compose turns in text against the engine held open by atlas_serve.

    python intake/chat.py [--kernel cub|bucket|radix|fiber] [--max 64] [--think] [--system "..."] [--lookahead 3]
    python intake/chat.py --script "first user turn" "second user turn" ...   (non-interactive)

Stages: this side tokenizes (intake/tokenizer.py, the model's own vocabulary, integers only),
composes the Qwen3 chat template over the whole transcript, sends the ids, and decodes the ids the
engine streams back. The engine side (atlas_serve) keeps the acquisition: every turn is a re-slice,
old positions picked up, only the new ones computed, and every generated token joins the acquisition.
The transcript is re-sent whole each turn (no KV cache: the checker decides what is contained).
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, "intake")
from tokenizer import Tokenizer  # noqa: E402

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODEL = "qwen3-1.7b"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def template(turns, system=None, think=False):
    s = f"{IM_START}system\n{system}{IM_END}\n" if system else ""
    prefix = "" if think else "<think>\n\n</think>\n\n"
    for role, text in turns:
        # the assistant's turn is re-rendered exactly as it was generated, think block included,
        # so the transcript extends the acquisition token for token
        s += f"{IM_START}{role}\n{prefix if role == 'assistant' else ''}{text}{IM_END}\n"
    s += f"{IM_START}assistant\n{prefix}"
    return s


def main():
    args = sys.argv[1:]
    kernel = args[args.index("--kernel") + 1] if "--kernel" in args else "cub"
    max_new = int(args[args.index("--max") + 1]) if "--max" in args else 64
    think = "--think" in args
    system = args[args.index("--system") + 1] if "--system" in args else None
    look = int(args[args.index("--lookahead") + 1]) if "--lookahead" in args else 0
    script = args[args.index("--script") + 1:] if "--script" in args else None
    tk = Tokenizer(f"{ROOT}/models/{MODEL}")
    env = dict(os.environ, ATLAS_KERNEL=kernel)
    exe = f"{ROOT}/atlas/target/release/atlas_serve.exe"
    p = subprocess.Popen([exe, ROOT, MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1)
    ready = p.stdout.readline().strip()
    print(ready, flush=True)
    if look:
        p.stdin.write(f"LOOK {look}\n")
        p.stdin.flush()
        print(p.stdout.readline().strip(), flush=True)
    turns = []
    log = [ready + "\n"]
    while True:
        if script is not None:
            if not script:
                break
            user = script.pop(0)
            print(f"you> {user}", flush=True)
        else:
            try:
                user = input("you> ")
            except EOFError:
                break
            if user.strip() in ("", "/quit"):
                break
        turns.append(("user", user))
        ids = tk.encode(template(turns, system, think))
        p.stdin.write(f"GEN {max_new} " + " ".join(map(str, ids)) + "\n")
        p.stdin.flush()
        gen = []
        shown = ""
        t0 = time.time()
        print("atlas> ", end="", flush=True)
        while True:
            line = p.stdout.readline()
            if not line:
                print("\n[engine closed]")
                return
            if line.startswith("TOK "):
                gen.append(int(line.split()[1]))
                text = tk.decode([g for g in gen if g not in (151645, 151643)])
                if text.startswith(shown) and not text.endswith("�"):
                    sys.stdout.write(text[len(shown):])
                    sys.stdout.flush()
                    shown = text
            elif line.startswith("TRACE"):
                print(f"\n   [{line.strip()}]", end="", flush=True)
            elif line.startswith("END"):
                break
        reply = tk.decode([g for g in gen if g not in (151645, 151643)])
        secs = time.time() - t0
        print(f"\n   [{line.strip()}  {len(gen)} tokens in {secs:.1f} s = {secs / max(1, len(gen)):.2f} s/token]", flush=True)
        turns.append(("assistant", reply))
        log.append(f"you> {user}\natlas> {reply}\n   [{line.strip()}]\n")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    with open(f"{ROOT}/receipts/chat-{MODEL}-{int(time.time())}.txt", "w", encoding="utf-8") as f:
        f.write(f"format=atlas.chat.v1 kernel={kernel} max_new={max_new} think={think} lookahead={look}\n" + "".join(log))


if __name__ == "__main__":
    main()
