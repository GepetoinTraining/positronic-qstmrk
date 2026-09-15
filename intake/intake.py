"""intake.py — Atlas Engine §1: defloat the model to the glyph as written, census, tile, seal.

    python intake/intake.py <safetensors-dir> <model-id> [--cells <dir>] [--receipts <dir>]

Steps (each aborts on failure; nothing is papered):
  1. Read every safetensors header; every tensor must be BF16; each pattern is a u16. No
     number type is ever instantiated — the body is memory-mapped as u16.
  2. Every pattern → cell ⟨m|n⟩ on its sheet (cells.py). Both names of zero are kept.
  3. Census gate: all 65,280 finite patterns round-trip through the cell exactly.
     Any non-finite pattern in the model refuses the intake (listed, never mapped).
  4. Value dictionary: the distinct patterns present, sorted; ids are u16 indices into it.
     The alphabet is closed after this run.
  5. Grids: per tensor, ids packed in 128×128 tiles, tiles ordered by the Morton code of
     (tile_row, tile_col) — row bits at odd positions, column bits at even. 1-D tensors
     are one row of 1×128 tiles in order. Static unfold of every grid file, read back
     from disk, must be byte-identical to the source cells or the run aborts.
  6. Seal: MANIFEST.json (per tensor: shape, tile, tiles, source sha256, grid sha256),
     DICT.u16, and RECEIPT.txt carrying sha256(MANIFEST + DICT), the counts, and the
     defloat report — for bf16 every finite pattern is exact as written, so the
     residual is 0 on every cell and hull == written holds trivially; that is stated,
     and the non-finite refusals (if any) are listed.

Integer only. Gated by intake/gate_source.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402

TILE = 128
U16 = np.uint16


# ---------------------------------------------------------------- safetensors header
def read_headers(sdir):
    """[(name, file, shape, byte_start, byte_end)] for every tensor, sorted by name."""
    out = []
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(sdir, fn)
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        base = 8 + hlen
        for name, spec in hdr.items():
            if name == "__metadata__":
                continue
            if spec["dtype"] != "BF16":
                sys.exit("REFUSED: %s is %s, not BF16" % (name, spec["dtype"]))
            a, b = spec["data_offsets"]
            out.append((name, path, list(spec["shape"]), base + a, base + b))
    out.sort()
    return out


def mmap_u16(path, start, end):
    n = (end - start) // 2
    return np.memmap(path, dtype=U16, mode="r", offset=start, shape=(n,))


# ---------------------------------------------------------------- morton
def _spread(x):
    x &= 0xFFFFFFFF
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


def morton(tr, tc):
    return _spread(tc) | (_spread(tr) << 1)


def tile_order(rows_t, cols_t):
    """Tile index (row-major tr*cols_t+tc) listed in Morton order."""
    keys = [(morton(tr, tc), tr * cols_t + tc) for tr in range(rows_t) for tc in range(cols_t)]
    keys.sort()
    return np.array([k[1] for k in keys], dtype=np.int64)


def fold(src2d, order, th, tw):
    """(R, C) → tiles of th×tw in the given order, each tile row-major, flat."""
    R, C = src2d.shape
    t = src2d.reshape(R // th, th, C // tw, tw).transpose(0, 2, 1, 3).reshape(-1, th * tw)
    return np.ascontiguousarray(t[order].reshape(-1))


def unfold(grid, order, R, C, th, tw):
    t = np.empty(((R // th) * (C // tw), th * tw), dtype=grid.dtype)
    t[order] = grid.reshape(-1, th * tw)
    return t.reshape(R // th, C // tw, th, tw).transpose(0, 2, 1, 3).reshape(R, C)


def sha(b):
    return hashlib.sha256(memoryview(np.ascontiguousarray(b))).hexdigest()


# ---------------------------------------------------------------- main
def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    sdir, model_id = argv[0], argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cells_dir = os.path.join(root, "cells", model_id)
    rec_dir = os.path.join(root, "receipts", "intake-%s-%s" % (model_id, time.strftime("%Y%m%d")))
    if "--cells" in argv:
        cells_dir = argv[argv.index("--cells") + 1]
    if "--receipts" in argv:
        rec_dir = argv[argv.index("--receipts") + 1]
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    t0 = time.time()

    # gate 2 — census
    checked, exact = cells.census()
    if checked != cells.FINITE_PATTERNS or exact != checked:
        sys.exit("GATE 2 FAIL: census %d/%d" % (exact, checked))
    print("gate 2 ok: %d/%d finite bf16 patterns round-trip exactly" % (exact, checked), flush=True)

    tensors = read_headers(sdir)
    print("%d tensors, all BF16" % len(tensors), flush=True)

    # pass A — value census over every tensor (bincount on the u16 pattern, integer)
    counts = np.zeros(65536, dtype=np.int64)
    nonfinite_mask = ((np.arange(65536) >> 7) & 0xFF) == 0xFF
    nonfinite = []
    total = 0
    for name, path, shape, a, b in tensors:
        v = mmap_u16(path, a, b)
        total += v.size
        bc = np.bincount(v, minlength=65536)
        counts += bc
        bad = int(bc[nonfinite_mask].sum())
        if bad:
            nonfinite.append((name, bad))
    if nonfinite:
        for n_, c_ in nonfinite:
            print("REFUSED non-finite: %s (%d patterns)" % (n_, c_))
        sys.exit("GATE 2 FAIL: non-finite patterns present; nothing mapped")
    present = np.nonzero(counts)[0].astype(U16)          # the closed alphabet, sorted
    s, m, n = cells.decompose(present)
    atoms = np.unique(m[m != 0])
    binades = np.unique(cells.binade_of(n, m)[m != 0])
    print("pass A: %d cells; alphabet %d distinct patterns; %d odd atoms; %d binades [%d..%d]; "
          "sheets %d; +0 %d -0 %d; %.1fs" % (
              total, present.size, atoms.size, binades.size, int(binades.min()), int(binades.max()),
              np.unique(s).size, int(counts[0x0000]), int(counts[0x8000]), time.time() - t0), flush=True)
    dict_path = os.path.join(rec_dir, "DICT.u16")
    present.tofile(dict_path)

    # pass B — grids + static unfold (gate 3) + hashes
    manifest = {"format": "atlas.intake.v1", "model": model_id, "source": os.path.abspath(sdir),
                "tile": TILE, "morton": "row bits odd, col bits even; 1-D = one row of 1x128 tiles",
                "dict": "DICT.u16", "dict_len": int(present.size), "tensors": []}
    unfold_ok = 0
    orders = {}
    for name, path, shape, a, b in tensors:
        v = mmap_u16(path, a, b)
        ids = np.searchsorted(present, v).astype(U16)
        if len(shape) == 2:
            R, C = shape
            th = tw = TILE
        elif len(shape) == 1:
            R, C = 1, shape[0]
            th, tw = 1, TILE
        else:
            sys.exit("REFUSED: %s has rank %d" % (name, len(shape)))
        if R % th or C % tw:
            sys.exit("REFUSED: %s shape %s not a multiple of the %dx%d tile" % (name, shape, th, tw))
        key = (R // th, C // tw)
        if key not in orders:
            orders[key] = tile_order(*key)
        order = orders[key]
        grid = fold(ids.reshape(R, C), order, th, tw)
        gpath = os.path.join(cells_dir, name + ".grid")
        grid.tofile(gpath)
        # gate 3: read back from disk, unfold, compare bytes to the source patterns
        back = np.fromfile(gpath, dtype=U16)
        rebuilt = present[unfold(back, order, R, C, th, tw)].reshape(-1)
        if rebuilt.size != v.size or not np.array_equal(rebuilt, v):
            sys.exit("GATE 3 FAIL: static unfold of %s is not byte-identical to source" % name)
        unfold_ok += 1
        manifest["tensors"].append({
            "name": name, "shape": shape, "tile": [th, tw], "tiles": int(order.size),
            "cells": int(v.size), "source_sha256": sha(v),
            "grid": os.path.relpath(gpath, root).replace("\\", "/"), "grid_sha256": sha(grid),
        })
        if unfold_ok % 50 == 0:
            print("  %d/%d grids written and unfolded byte-identical; %.0fs" % (unfold_ok, len(tensors), time.time() - t0), flush=True)
    print("gate 3 ok: %d/%d static unfolds byte-identical" % (unfold_ok, len(tensors)), flush=True)

    # seal
    mpath = os.path.join(rec_dir, "MANIFEST.json")
    mbytes = json.dumps(manifest, indent=1, sort_keys=True).encode()
    open(mpath, "wb").write(mbytes)
    seal = hashlib.sha256(mbytes + open(dict_path, "rb").read()).hexdigest()
    secs = time.time() - t0
    receipt = (
        "format=atlas.intake.v1\nseal=%s\nmodel=%s\ntensors=%d cells=%d seconds=%.1f\n\n" % (seal, model_id, len(tensors), total, secs)
        + "The intake read every tensor's bf16 patterns as u16 and put each on the two-sheeted lattice as\n"
        "the cell <m|n> (odd core, signed rung, sheet), both names of zero kept. No number type was\n"
        "instantiated. Every finite bf16 pattern is an exact dyadic as written, so on every cell\n"
        "written = hull and residual = 0: NOTHING WAS DEFLOATED and nothing was dropped.\n\n"
        + "[gate 1] source needles: intake/gate_source.py over intake/*.py (run by the caller, recorded in the log)\n"
        + "[gate 2] census: %d/%d finite patterns round-trip exactly; non-finite patterns in model: 0\n" % (exact, checked)
        + "[gate 3] static unfold: %d/%d grids byte-identical to source after disk read-back\n\n" % (unfold_ok, len(tensors))
        + "[alphabet] distinct_patterns=%d odd_atoms=%d binades=%d binade_min=%d binade_max=%d sheets=%d plus_zero=%d minus_zero=%d\n" % (
            present.size, atoms.size, binades.size, int(binades.min()), int(binades.max()), np.unique(s).size,
            int(counts[0x0000]), int(counts[0x8000]))
        + "[defloat] rows=0 defloated=0 refused_nonfinite=0 max_residual=0\n"
        + "[grids] dir=%s tile=%d morton=row-odd/col-even\n" % (os.path.relpath(cells_dir, root).replace("\\", "/"), TILE)
    )
    open(os.path.join(rec_dir, "RECEIPT.txt"), "w", newline="\n").write(receipt)
    print(receipt)


if __name__ == "__main__":
    main(sys.argv[1:])
