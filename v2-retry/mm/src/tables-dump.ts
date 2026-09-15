import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readSync, writeFileSync, writeSync } from "node:fs";
import { dirname } from "node:path";

/**
 * Dump every table of Qwen3-1.7B as CSV (pgarcia):
 *   header   the column numbers; every row starts with its row number (numbering starts at 1: no 0 is produced)
 *   values   as they are stored — each bf16 word's exact value n/2^m — with the WHOLE table multiplied by 2×5 as
 *            long as ANY value in it still has a period: every value comes out a whole number
 * Per table the times multiplied (k) is the largest, over its words, of the twos in 2^m that n does not already
 * carry. Every word's string comes from a LUT of its 32768 magnitudes for that k, so no value is computed twice and
 * no float is used.
 * [B] vectors (norm weights) are written as one row; tables keep the model's own names; output under
 *     v2-retry/tables/qwen3-1.7b/. Usage: tables-dump.ts [name filter regex]
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b/";
const shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
const filter = new RegExp(process.argv[2] ?? ".*");

type T = { shard: string; name: string; shape: number[]; start: number; end: number; base: number };
const tables: T[] = [];
for (const s of shards) {
  const fd = openSync(dir + s, "r");
  const len = Buffer.alloc(8);
  readSync(fd, len, 0, 8, 0);
  const n = Number(len.readBigUInt64LE(0));
  const head = Buffer.alloc(n);
  readSync(fd, head, 0, n, 8);
  closeSync(fd);
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>)) {
    if (name === "__metadata__" || !filter.test(name)) continue;
    tables.push({ shard: s, name, shape: t.shape, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
const order = (name: string) => name.replace(/\.(\d+)\./, (_, d) => `.${d.padStart(2, "0")}.`);
tables.sort((a, b) => (order(a.name) < order(b.name) ? -1 : 1));

/** a magnitude's relation: n / 2^m */
const relation = (mag: number): [bigint, number] => {
  const E = (mag >> 7) & 0xff;
  const f = mag & 0x7f;
  return E > 0 ? [BigInt(128 + f), 134 - E] : [BigInt(f), 133];
};
/** per magnitude: the twos of 2^m not already in n — the times ×(2×5) that word needs before it has no period */
const periodTwos = new Int16Array(0x8000);
for (let mag = 1; mag < 0x8000; mag++) {
  let [n, m] = relation(mag);
  while (m > 0 && (n & 1n) === 0n) {
    n >>= 1n;
    m--;
  }
  periodTwos[mag] = Math.max(0, m);
}
/** the exact decimal of n/2^m multiplied by (2×5)^k */
const decimal = (n: bigint, m: number, k: number) => {
  const N = n * 10n ** BigInt(k);
  if (m <= 0) return String(N << BigInt(-m));
  const P = N * 5n ** BigInt(m); // N/2^m = N·5^m / 10^m
  const scale = 10n ** BigInt(m);
  const whole = P / scale;
  const frac = (P % scale).toString().padStart(m, "0").replace(/0+$/, "");
  return frac ? `${whole}.${frac}` : `${whole}`;
};
const luts = new Map<number, string[]>();
const lutFor = (k: number) => {
  let lut = luts.get(k);
  if (!lut) {
    lut = new Array<string>(0x10000);
    for (let mag = 1; mag < 0x8000; mag++) {
      const [n, m] = relation(mag);
      const s = decimal(n, m, k);
      lut[mag] = s;
      lut[mag | 0x8000] = `-${s}`;
    }
    luts.set(k, lut);
  }
  return lut;
};

const manifest: { name: string; shape: number[]; rows: number; columns: number; times: number; file: string; bytes: number; sha256: string }[] = [];
const t0 = Date.now();
for (const t of tables) {
  const tt = Date.now();
  const fd = openSync(dir + t.shard, "r");
  const bytes = Buffer.alloc(t.end - t.start);
  readSync(fd, bytes, 0, bytes.length, t.base + t.start);
  closeSync(fd);
  const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
  // the whole table is multiplied by 2×5 until no value has anything after a period: k covers, for every word, the
  // twos of its denominator 2^m that its numerator n does not already carry
  let k = 0;
  for (let i = 0; i < words.length; i++) {
    const mag = words[i] & 0x7fff;
    if (mag === 0) throw new Error(`${t.name} holds an exact zero: multiplying by 2×5 never takes it away`);
    if (periodTwos[mag] > k) k = periodTwos[mag];
  }
  const lut = lutFor(k);
  const [rows, columns] = t.shape.length === 1 ? [1, t.shape[0]] : [t.shape[0], t.shape[1]];

  const file = `${out}${t.name.split(".").join("/")}.csv`;
  mkdirSync(dirname(file), { recursive: true });
  const outFd = openSync(file, "w");
  const hash = createHash("sha256");
  let written = 0;
  let chunk: string[] = [];
  let chunkLen = 0;
  const flush = () => {
    const s = chunk.join("");
    const b = Buffer.from(s, "utf8");
    writeSync(outFd, b);
    hash.update(b);
    written += b.length;
    chunk = [];
    chunkLen = 0;
  };
  const push = (s: string) => {
    chunk.push(s);
    chunkLen += s.length;
    if (chunkLen > 8_000_000) flush();
  };
  const header: string[] = [""];
  for (let c = 1; c <= columns; c++) header.push(String(c));
  push(header.join(",") + "\n");
  const row: string[] = new Array(columns + 1);
  for (let r = 0; r < rows; r++) {
    row[0] = String(r + 1);
    const base = r * columns;
    for (let c = 0; c < columns; c++) row[c + 1] = lut[words[base + c]];
    push(row.join(",") + "\n");
  }
  flush();
  closeSync(outFd);
  const sha = hash.digest("hex");
  manifest.push({ name: t.name, shape: t.shape, rows, columns, times: k, file: file.slice(out.length), bytes: written, sha256: sha });
  process.stdout.write(`${t.name.padEnd(48)} ${t.shape.join("x").padEnd(12)} ×(2×5)^${String(k).padEnd(3)} ${(written / 1e6).toFixed(1).padStart(9)} MB  ${((Date.now() - tt) / 1000).toFixed(1)}s\n`);
}

mkdirSync(out, { recursive: true });
const manifestFile = `${out}manifest${process.argv[2] ? "-partial" : ""}.json`;
const body = JSON.stringify({ model: "Qwen3-1.7B", source: dir, rule: "every table multiplied by 2×5 while any value has a period (every value a whole number); header = column numbers; rows numbered; numbering starts at 1", tables: manifest }, null, 1);
writeFileSync(manifestFile, body);
process.stdout.write(`${manifest.length} tables · ${(manifest.reduce((a, m) => a + m.bytes, 0) / 1e9).toFixed(2)} GB · ${((Date.now() - t0) / 1000).toFixed(1)}s · ${manifestFile} sha256 ${createHash("sha256").update(body).digest("hex").slice(0, 8)}…\n`);
