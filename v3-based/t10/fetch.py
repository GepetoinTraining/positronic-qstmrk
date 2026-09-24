"""Fetch single tensors from the Hub by byte range (the way v1's intake reads them: header → offsets → u16),
and admit each only if its bytes hash to v1's sealed source_sha256 (receipts/intake-qwen3-1.7b-20260913/MANIFEST.json).
Stdlib only. Usage: python fetch.py <tensor name> ...   → writes <name>.u16 beside this script."""
import hashlib, json, os, struct, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/"
MANIFEST = os.path.join(HERE, "..", "..", "receipts", "intake-qwen3-1.7b-20260913", "MANIFEST.json")

def get(url, start, end, tries=5):  # inclusive byte range; a dropped connection is retried, never trusted
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except Exception as e:
            if t == tries - 1:
                raise
    raise IOError(f"short read {url} {start}-{end}")

sealed = {t["name"]: t["source_sha256"] for t in json.load(open(MANIFEST))["tensors"]}
index = json.load(open(os.path.join(HERE, "index.json")))["weight_map"]
names = sys.argv[1:]
headers = {}
for shard in sorted({index[n] for n in names}):
    n = struct.unpack("<Q", get(URL + shard, 0, 7))[0]
    headers[shard] = (8 + n, json.loads(get(URL + shard, 8, 8 + n - 1)))

def one(name):
    path = os.path.join(HERE, name + ".u16")
    if os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == sealed[name]:
        return f"{name}: already on disk, EQUAL to v1 seal", True
    base, head = headers[index[name]]
    spec = head[name]
    assert spec["dtype"] == "BF16", spec
    a, b = spec["data_offsets"]
    data = get(URL + index[name], base + a, base + b - 1)
    h = hashlib.sha256(data).hexdigest()
    if h != sealed[name]:
        return f"{name}: DIFFERENT from v1 seal {sealed[name][:16]} (got {h[:16]}); refused", False
    with open(path, "wb") as f:
        f.write(data)
    return f"{name}: shape {spec['shape']}, {len(data)} bytes, sha256 {h[:16]}… EQUAL to v1 seal", True

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(one, names))
for line, _ in results:
    print(line)
if not all(ok for _, ok in results):
    sys.exit("refused: some bytes do not match the sealed source")
