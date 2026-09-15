import { createHash } from "node:crypto";
import { closeSync, openSync, readSync, writeFileSync } from "node:fs";

/**
 * Step before the dump: for every table of Qwen3-1.7B, read each bf16 word where it is stored and report
 *   zeros      words that are exactly zero (±): no multiplying by 2×5 ever takes that 0 away
 *   smallest   the smallest non-zero word, written as a relation n / 2^m (never "0.")
 *   ×(2×5)     how many times the whole table must be multiplied by 2×5 before no value reads "0." —
 *              the least k with (2×5)^k · n ≥ 2^m, decided by crossing integers (BigInt), no floats
 * [B] reading of pgarcia's "while you have ANY 0 inside the tables": any value that reads 0.… (below one in size).
 * Nothing is dumped here.
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
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
    if (name === "__metadata__") continue;
    if (t.dtype !== "BF16") throw new Error(`${name} is ${t.dtype}`);
    tables.push({ shard: s, name, shape: t.shape, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
const order = (name: string) => name.replace(/\.(\d+)\./, (_, d) => `.${d.padStart(2, "0")}.`);
tables.sort((a, b) => (order(a.name) < order(b.name) ? -1 : 1));

/** a word's relation: n / 2^m (n the radial count; subnormal words carry f over 2^133) */
const relation = (w: number): [bigint, number] => {
  const E = (w >> 7) & 0xff;
  const f = w & 0x7f;
  return E > 0 ? [BigInt(128 + f), 134 - E] : [BigInt(f), 133];
};
const timesDecade = (n: bigint, m: number) => {
  if (m <= 0) return 0;
  const target = 1n << BigInt(m);
  let k = 0;
  let v = n;
  while (v < target) {
    v *= 10n;
    k++;
  }
  return k;
};
const rel = ([n, m]: [bigint, number]) => (m > 0 ? `${n}/2^${m}` : `${n}·2^${-m}`);

const lines: string[] = [];
const say = (s: string) => {
  lines.push(s);
};
say("format=v2.tables.scan.v1  Qwen3-1.7B, every table read word by word (bf16: sign bit 15, exponent bits 14..7, mantissa bits 6..0)");
say("[B] 'any 0' read as: any value that reads 0.… ; ×(2×5) = times the whole table is multiplied until none does");
say("table                                            shape          values       zeros   smallest        ×(2×5)  largest");

let totalValues = 0;
let totalZeros = 0;
let specials = 0;
const readBytes = (t: T) => {
  const fd = openSync(dir + t.shard, "r");
  const bytes = Buffer.alloc(t.end - t.start);
  readSync(fd, bytes, 0, bytes.length, t.base + t.start);
  closeSync(fd);
  return bytes;
};
const t0 = Date.now();
let embedBytes: Buffer | null = null;
let tied = "";
for (const t of tables) {
  const bytes = readBytes(t);
  if (t.name === "model.embed_tokens.weight") embedBytes = bytes;
  if (t.name === "lm_head.weight" && embedBytes) tied = bytes.equals(embedBytes) ? "lm_head.weight is byte-identical to model.embed_tokens.weight" : "lm_head.weight differs from model.embed_tokens.weight";
  const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
  let zeros = 0;
  let minKey = Infinity; // smallest non-zero magnitude: exponent then mantissa
  let maxKey = -1;
  let minWord = 0;
  let maxWord = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const mag = w & 0x7fff;
    if (mag === 0) {
      zeros++;
      continue;
    }
    if (mag >= 0x7f80) specials++;
    if (mag < minKey) {
      minKey = mag;
      minWord = w;
    }
    if (mag > maxKey) {
      maxKey = mag;
      maxWord = w;
    }
  }
  totalValues += words.length;
  totalZeros += zeros;
  const small = relation(minWord);
  const k = timesDecade(small[0], small[1]);
  say(`${t.name.padEnd(48)} ${t.shape.join("x").padEnd(14)} ${String(words.length).padEnd(12)} ${String(zeros).padEnd(7)} ${rel(small).padEnd(15)} ${String(k).padEnd(7)} ${rel(relation(maxWord))}`);
}
say(`total: ${tables.length} tables · ${totalValues} values · ${totalZeros} exact zeros · ${specials} inf/nan words`);
say(tied);

const body = lines.join("\n") + "\n";
const receipt = body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;
writeFileSync(new URL("../receipts/tables-scan.txt", import.meta.url), receipt);
const kinds = new Map<string, { n: number; zeros: number; kmin: number; kmax: number }>();
for (const l of lines.slice(3, 3 + tables.length)) {
  const [name, , , zeros, , k] = l.split(/\s+/);
  const kind = name.replace(/\.\d+\./, ".N.");
  const e = kinds.get(kind) ?? { n: 0, zeros: 0, kmin: Infinity, kmax: 0 };
  e.n++;
  e.zeros += Number(zeros);
  e.kmin = Math.min(e.kmin, Number(k));
  e.kmax = Math.max(e.kmax, Number(k));
  kinds.set(kind, e);
}
process.stdout.write(`${lines[0]}\n${lines[1]}\n`);
for (const [kind, e] of kinds) process.stdout.write(`  ${kind.padEnd(48)} ×${String(e.n).padEnd(3)} zeros ${String(e.zeros).padEnd(8)} ×(2×5) from ${e.kmin} to ${e.kmax}\n`);
process.stdout.write(`${lines[lines.length - 2]}\n${lines[lines.length - 1]}\nscan ${((Date.now() - t0) / 1000).toFixed(1)}s · receipts/tables-scan.txt ${receipt.slice(-65, -57)}…\n`);
