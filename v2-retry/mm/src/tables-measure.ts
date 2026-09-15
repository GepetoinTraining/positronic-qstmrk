import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * Measurements (pgarcia: "set up measurements, measure and do, don't think, test"). ±residual basis (126 cells).
 * Every measurement counts DISTINCT cycle-2 cells (two bigrams side by side, 126⁴ possible) against the cells held,
 * on the real arrangement and on a CONTROL: the same cells, positions shuffled (Fisher–Yates, LCG, seed declared).
 * More repeats than the shuffle = the arrangement carries structure.
 *   cols      within a table: columns c, c+1 | c+2, c+3 of the same row
 *   rows      within a table: rows r, r+1 | r+2, r+3 of the same column (control for the 2pow columns)
 *   gate×up   the bigram (gate[j,c], up[j,c]) — same intermediate j, same input c — beside (…, c+1)
 *   q×k       the bigram (q[h·128+d, c], k[⌊h/2⌋·128+d, c]) — query meets its kv group, same dim d, same input c
 *   up×downᵀ  the bigram (up[j,c], down[c,j]) — read from and written to the same hidden c, same j — beside c+1
 *   q×oᵀ      the bigram (q[x,c], o[c,x]) — same head coordinate x, same hidden c — beside c+1
 * For the paired measurements the control shuffles the second table only.
 *   measure.csv  test,table,held,distinct_real,distinct_shuffled
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const SEED = 20260914;
const Q = 126;

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

type T = { shard: string; name: string; rows: number; columns: number; start: number; end: number; base: number };
const tables = new Map<string, T>();
for (const s of ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]) {
  const fd = openSync(dir + s, "r");
  const len = Buffer.alloc(8);
  readSync(fd, len, 0, 8, 0);
  const n = Number(len.readBigUInt64LE(0));
  const head = Buffer.alloc(n);
  readSync(fd, head, 0, n, 8);
  closeSync(fd);
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { shape: number[]; data_offsets: [number, number] }>)) {
    if (name === "__metadata__") continue;
    const [rows, columns] = t.shape.length === 1 ? [1, t.shape[0]] : [t.shape[0], t.shape[1]];
    tables.set(name, { shard: s, name, rows, columns, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
const order = (name: string) => name.replace(/\.(\d+)\./, (_, d) => `.${d.padStart(2, "0")}.`);
const names = [...tables.keys()].sort((a, b) => (order(a) < order(b) ? -1 : 1));

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

const seen = new Uint8Array(Math.ceil((Q * Q * Q * Q) / 8));
/** distinct cycle-2 cells; `quad(k)` gives the four codes of the k-th held cell */
const distinct = (held: number, quad: (k: number, into: Uint8Array) => void) => {
  seen.fill(0);
  const q = new Uint8Array(4);
  let d = 0;
  for (let k = 0; k < held; k++) {
    quad(k, q);
    const i = ((q[0] * Q + q[1]) * Q + q[2]) * Q + q[3];
    const at = i >>> 3;
    const bit = 1 << (i & 7);
    if (!(seen[at] & bit)) {
      seen[at] |= bit;
      d++;
    }
  }
  return d;
};

const rows: string[] = ["test,table,held,distinct_real,distinct_shuffled"];
const sums = new Map<string, { tables: number; held: number; real: number; shuf: number; realFewer: number; same: number; realMore: number }>();
const record = (test: string, table: string, held: number, real: number, shuf: number) => {
  rows.push(`${test},${table},${held},${real},${shuf}`);
  const s = sums.get(test) ?? { tables: 0, held: 0, real: 0, shuf: 0, realFewer: 0, same: 0, realMore: 0 };
  s.tables++;
  s.held += held;
  s.real += real;
  s.shuf += shuf;
  if (real < shuf) s.realFewer++;
  else if (real === shuf) s.same++;
  else s.realMore++;
  sums.set(test, s);
};
const t0 = Date.now();

// within a table: cols and rows
for (const name of names) {
  const t = tables.get(name)!;
  const real = codesOf(name);
  const shuf = shuffled(real);
  const R = t.rows;
  const C = t.columns;
  const colsHeld = R * (C >> 2);
  const byCols = (codes: Uint8Array) => (k: number, q: Uint8Array) => {
    const c = k % (C >> 2);
    const base = ((k - c) / (C >> 2)) * C + 4 * c;
    q[0] = codes[base];
    q[1] = codes[base + 1];
    q[2] = codes[base + 2];
    q[3] = codes[base + 3];
  };
  record("cols", name, colsHeld, distinct(colsHeld, byCols(real)), distinct(colsHeld, byCols(shuf)));
  if (R % 4 === 0) {
    const rowsHeld = (R >> 2) * C;
    const byRows = (codes: Uint8Array) => (k: number, q: Uint8Array) => {
      const c = k % C;
      const base = ((k - c) / C) * 4 * C;
      q[0] = codes[base + c];
      q[1] = codes[base + C + c];
      q[2] = codes[base + 2 * C + c];
      q[3] = codes[base + 3 * C + c];
    };
    record("rows", name, rowsHeld, distinct(rowsHeld, byRows(real)), distinct(rowsHeld, byRows(shuf)));
  }
}
process.stdout.write(`within tables done · ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

// paired: the tables that meet in the matmul
for (let layer = 0; layer < 28; layer++) {
  const L = `model.layers.${layer}`;
  const gate = codesOf(`${L}.mlp.gate_proj.weight`);
  const up = codesOf(`${L}.mlp.up_proj.weight`);
  const down = codesOf(`${L}.mlp.down_proj.weight`);
  const q = codesOf(`${L}.self_attn.q_proj.weight`);
  const k = codesOf(`${L}.self_attn.k_proj.weight`);
  const o = codesOf(`${L}.self_attn.o_proj.weight`);
  const H = 2048;
  const I = 6144;

  // gate×up: (gate[j,c], up[j,c]) beside c+1
  {
    const held = I * (H >> 1);
    const quad = (b: Uint8Array) => (n: number, into: Uint8Array) => {
      const c = n % (H >> 1);
      const at = ((n - c) / (H >> 1)) * H + 2 * c;
      into[0] = gate[at];
      into[1] = b[at];
      into[2] = gate[at + 1];
      into[3] = b[at + 1];
    };
    record("gate×up", L, held, distinct(held, quad(up)), distinct(held, quad(shuffled(up))));
  }
  // q×k: (q[h·128+d, c], k[⌊h/2⌋·128+d, c]) beside c+1
  {
    const held = H * (H >> 1);
    const quad = (b: Uint8Array) => (n: number, into: Uint8Array) => {
      const c = n % (H >> 1);
      const qr = (n - c) / (H >> 1);
      const h = (qr - (qr % 128)) / 128;
      const kr = (h >> 1) * 128 + (qr % 128);
      const qa = qr * H + 2 * c;
      const ka = kr * H + 2 * c;
      into[0] = q[qa];
      into[1] = b[ka];
      into[2] = q[qa + 1];
      into[3] = b[ka + 1];
    };
    record("q×k", L, held, distinct(held, quad(k)), distinct(held, quad(shuffled(k))));
  }
  // up×downᵀ: (up[j,c], down[c,j]) beside c+1
  {
    const held = I * (H >> 1);
    const quad = (b: Uint8Array) => (n: number, into: Uint8Array) => {
      const c = n % (H >> 1);
      const j = (n - c) / (H >> 1);
      into[0] = up[j * H + 2 * c];
      into[1] = b[2 * c * I + j];
      into[2] = up[j * H + 2 * c + 1];
      into[3] = b[(2 * c + 1) * I + j];
    };
    record("up×downᵀ", L, held, distinct(held, quad(down)), distinct(held, quad(shuffled(down))));
  }
  // q×oᵀ: (q[x,c], o[c,x]) beside c+1
  {
    const held = H * (H >> 1);
    const quad = (b: Uint8Array) => (n: number, into: Uint8Array) => {
      const c = n % (H >> 1);
      const x = (n - c) / (H >> 1);
      into[0] = q[x * H + 2 * c];
      into[1] = b[2 * c * H + x];
      into[2] = q[x * H + 2 * c + 1];
      into[3] = b[(2 * c + 1) * H + x];
    };
    record("q×oᵀ", L, held, distinct(held, quad(o)), distinct(held, quad(shuffled(o))));
  }
  process.stdout.write(`layer ${layer} paired · ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

writeFileSync(`${out}measure.csv`, rows.join("\n") + "\n");
const summary = [`seed ${SEED} · ${((Date.now() - t0) / 1000).toFixed(1)}s`];
for (const [test, s] of sums) {
  summary.push(
    `${test.padEnd(10)} tables ${String(s.tables).padEnd(4)} held ${s.held} · distinct real ${s.real} · shuffled ${s.shuf} · repeats real ${s.held - s.real} vs shuffled ${s.held - s.shuf} · tables real fewer ${s.realFewer}, same ${s.same}, more ${s.realMore}`,
  );
}
writeFileSync(`${out}measure.txt`, summary.join("\n") + "\n");
process.stdout.write(summary.join("\n") + "\n");
