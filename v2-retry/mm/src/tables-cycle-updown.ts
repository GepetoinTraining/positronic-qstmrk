import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * Cycle the up×downᵀ bigrams again (pgarcia). Per layer, ±residual basis (126 codes):
 *   cycle 1  cell (up[j,c], down[c,j]) — j the intermediate coordinate, c the hidden coordinate read by up, written by down
 *   cycle k  cell (cell k−1 at c, cell k−1 at c+1) along c, while the hidden count still holds a 2 (2048 = 2¹¹ → 11 cycles)
 * Distinct cells per cycle, real against the control (down shuffled in place, Fisher–Yates, LCG seed 20260914).
 * Machine side: cells numbered by rank of their (left, right) key after a numeric sort — a numbering, not a value.
 * Gate: real cycle-2 distinct must equal measure.csv's up×downᵀ distinct_real.
 *   cycle-updown.csv  layer,cycle,held,distinct_real,distinct_shuffled
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const SEED = 20260914;
const H = 2048;
const I = 6144;

const oddRows = readFileSync(`${out}odds.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number));
const residuals = [...new Set(oddRows.map((r) => r[2]))].sort((a, b) => a - b);
const placeOfResidual = new Map(residuals.map((r, i) => [r, i]));
const residualOfOdd = new Map(oddRows.map((r) => [r[0], r[2]]));
const codeOfWord = new Uint8Array(0x10000);
for (let mag = 1; mag < 0x8000; mag++) {
  const E = (mag >> 7) & 0xff;
  let odd = E > 0 ? 128 + (mag & 0x7f) : mag & 0x7f;
  while (odd % 2 === 0) odd /= 2;
  const place = placeOfResidual.get(residualOfOdd.get(odd) ?? 1)!;
  codeOfWord[mag] = place;
  codeOfWord[mag | 0x8000] = 63 + place;
}
const measured = new Map(
  readFileSync(`${out}measure.csv`, "utf8").trim().split("\n").filter((l) => l.startsWith("up×downᵀ,")).map((l) => l.split(",")).map((r) => [r[1], Number(r[3])]),
);

type T = { shard: string; start: number; end: number; base: number };
const tables = new Map<string, T>();
for (const s of ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]) {
  const fd = openSync(dir + s, "r");
  const len = Buffer.alloc(8);
  readSync(fd, len, 0, 8, 0);
  const n = Number(len.readBigUInt64LE(0));
  const head = Buffer.alloc(n);
  readSync(fd, head, 0, n, 8);
  closeSync(fd);
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { data_offsets: [number, number] }>)) {
    if (name !== "__metadata__") tables.set(name, { shard: s, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
const codesOf = (name: string) => {
  const t = tables.get(name)!;
  const fd = openSync(dir + t.shard, "r");
  const bytes = Buffer.alloc(t.end - t.start);
  readSync(fd, bytes, 0, bytes.length, t.base + t.start);
  closeSync(fd);
  const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
  const codes = new Uint8Array(words.length);
  for (let i = 0; i < words.length; i++) codes[i] = codeOfWord[words[i]];
  return codes;
};
let lcg = SEED >>> 0;
const shuffled = (codes: Uint8Array) => {
  const s = codes.slice();
  for (let i = s.length - 1; i > 0; i--) {
    lcg = (Math.imul(lcg, 1664525) + 1013904223) >>> 0;
    const j = Math.floor((lcg / 4294967296) * (i + 1));
    const x = s[i];
    s[i] = s[j];
    s[j] = x;
  }
  return s;
};

/** number every (left, right) key by its rank among the distinct keys; returns [ids, distinct] */
const rank = (keys: Float64Array): [Float64Array, number] => {
  const sorted = keys.slice().sort();
  let d = 0;
  for (let i = 0; i < sorted.length; i++) if (i === 0 || sorted[i] !== sorted[i - 1]) sorted[d++] = sorted[i];
  const ids = new Float64Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    let lo = 0;
    let hi = d - 1;
    const k = keys[i];
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < k) lo = mid + 1;
      else hi = mid;
    }
    ids[i] = lo;
  }
  return [ids, d];
};

/** distinct cells per cycle: cycle 1 … 11 */
const cycles = (up: Uint8Array, down: Uint8Array) => {
  const counts: number[] = [];
  // cycle 1: (up[j,c], down[c,j]) laid out j × c
  let cur = new Float64Array(I * H);
  for (let j = 0; j < I; j++) for (let c = 0; c < H; c++) cur[j * H + c] = up[j * H + c] * 126 + down[c * I + j];
  let width = H;
  let distinct1 = new Uint8Array(126 * 126);
  let d1 = 0;
  for (let i = 0; i < cur.length; i++) if (!distinct1[cur[i]]) (distinct1[cur[i]] = 1), d1++;
  counts.push(d1);
  let span = 126 * 126;
  while (width % 2 === 0) {
    const half = width / 2;
    const keys = new Float64Array(I * half);
    for (let j = 0; j < I; j++) for (let c = 0; c < half; c++) keys[j * half + c] = cur[j * width + 2 * c] * span + cur[j * width + 2 * c + 1];
    const [ids, d] = rank(keys);
    counts.push(d);
    cur = ids;
    span = d;
    width = half;
  }
  return counts;
};

const lines = ["layer,cycle,held,distinct_real,distinct_shuffled"];
const sums: { held: number; real: number; shuf: number; fewer: number; same: number; more: number }[] = [];
let gateBad = 0;
const t0 = Date.now();
for (let layer = 0; layer < 28; layer++) {
  const L = `model.layers.${layer}`;
  const up = codesOf(`${L}.mlp.up_proj.weight`);
  const down = codesOf(`${L}.mlp.down_proj.weight`);
  const real = cycles(up, down);
  const shuf = cycles(up, shuffled(down));
  if (real[1] !== measured.get(L)) gateBad++;
  for (let c = 0; c < real.length; c++) {
    const held = I * (H >> c);
    lines.push(`${layer},${c + 1},${held},${real[c]},${shuf[c]}`);
    const s = (sums[c] ??= { held: 0, real: 0, shuf: 0, fewer: 0, same: 0, more: 0 });
    s.held += held;
    s.real += real[c];
    s.shuf += shuf[c];
    if (real[c] < shuf[c]) s.fewer++;
    else if (real[c] === shuf[c]) s.same++;
    else s.more++;
  }
  process.stdout.write(`layer ${layer}: real ${real.join(" ")} · shuffled ${shuf.join(" ")} · ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
}
writeFileSync(`${out}cycle-updown.csv`, lines.join("\n") + "\n");
const summary = [`up×downᵀ cycles, 28 layers, seed ${SEED} · gate vs measure.csv cycle 2: ${28 - gateBad} equal, ${gateBad} different · ${((Date.now() - t0) / 1000).toFixed(0)}s`];
sums.forEach((s, c) =>
  summary.push(`cycle ${String(c + 1).padEnd(2)} held ${s.held} · distinct real ${s.real} · shuffled ${s.shuf} · repeats real ${s.held - s.real} vs shuffled ${s.held - s.shuf} · layers real fewer ${s.fewer}, same ${s.same}, more ${s.more}`),
);
writeFileSync(`${out}cycle-updown.txt`, summary.join("\n") + "\n");
process.stdout.write(summary.join("\n") + "\n");
