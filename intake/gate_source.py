"""gate_source.py — Gate 1: no source file that touches the pass may carry a float needle.

Byte-grep, as in V5's assert_*_source_float_free: the needles are the two IEEE type names
of 32 and 64 bits, the generic word for that kind of number, and numpy's cast verb. Any hit
aborts. Run: python intake/gate_source.py intake/*.py
"""
import re
import sys

NEEDLES = [b"f" + b"32", b"f" + b"64", b"fl" + b"oat", b"as" + b"type("]
# (spliced so this file passes its own gate; a needle found whole anywhere else fails)


def main(paths):
    bad = 0
    for path in paths:
        if path.endswith(("gate_source.py", "record_anchor.py", "ringing.py", "ringing_positions.py")):
            continue  # the gate itself, the retired oracle, and the two Prediction-6 instruments (rulers, never in the pass)
        data = open(path, "rb").read()
        # "defloat" is the intake's own verb (cad-systemup core/defloat.py); it is not a type name
        data = re.sub(rb"(?i)defloat", b"", data)
        for nd in NEEDLES:
            if nd in data:
                # allow the sanctioned integer-cast spellings only
                if nd == b"as" + b"type(" and all(
                    seg.startswith((b"U16", b"I32", b"np.uint8", b"np.int64", b"np.uint64", b"np.uint32", b"np.int32", b"np.uint16"))
                    for seg in data.split(nd)[1:]
                ):
                    continue
                print(f"GATE 1 FAIL: {path} carries needle {nd!r}")
                bad += 1
    if bad:
        sys.exit(1)
    print(f"gate 1 ok: {len(paths)} files, no needle")


if __name__ == "__main__":
    main(sys.argv[1:])
