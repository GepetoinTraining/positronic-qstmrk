import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readSync, writeFileSync, writeSync } from "node:fs";
import { dirname } from "node:path";

/**
 * The tables of Qwen3-1.7B, deduped and pointerized (pgarcia: not multiplying the whole thing).
 *   slots    every distinct stored magnitude takes the next slot, numbered from 1, in the order it is first met
 *            (tables in order, rows, columns) — Opus-style: a slot, not a name
 *   lut.csv  one line per slot: the stored relation n/2^m, its core (odd part of n × 5^fives), fives, tens —
 *            the whole value is core followed by `tens` zeros, at one scale K for the whole model, where K is the
 *            most twos any met magnitude needs (every value whole, no period)
 *   tables   header = column numbers, rows numbered from 1, cells = ±slot (the sign is the plane)
 * Pass 1 (all tables) assigns slots and K; pass 2 writes the pointer CSVs for the tables matching the filter.
 * [B] table order: names sorted with layer numbers padded; vectors written as one row.
 * Usage: tables-pointers.ts [name filter regex for pass 2]
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-pointers/";
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
    if (name === "__metadata__") continue;
    tables.push({ shard: s, name, shape: t.shape, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
const order = (name: string) => name.replace(/\.(\d+)\./, (_, d) => `.${d.padStart(2, "0")}.`);
tables.sort((a, b) => (order(a.name) < order(b.name) ? -1 : 1));
const wordsOf = (t: T) => {
  const fd = openSync(dir + t.shard, "r");
  const bytes = Buffer.alloc(t.end - t.start);
  readSync(fd, bytes, 0, bytes.length, t.base + t.start);
  closeSync(fd);
  return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
};

/** a magnitude's stored relation n / 2^m */
const relation = (mag: number): [number, number] => {
  const E = (mag >> 7) & 0xff;
  const f = mag & 0x7f;
  return E > 0 ? [128 + f, 134 - E] : [f, 133];
};
const twosNeeded = (mag: number) => {
  let [n, m] = relation(mag);
  while (m > 0 && n % 2 === 0) {
    n /= 2;
    m--;
  }
  return Math.max(0, m);
};

const t0 = Date.now();
// pass 1: slots in first-met order, and the one scale K
const slotOf = new Int32Array(0x8000);
const magOfSlot: number[] = [0];
let K = 0;
for (const t of tables) {
  const words = wordsOf(t);
  for (let i = 0; i < words.length; i++) {
    const mag = words[i] & 0x7fff;
    if (mag === 0) throw new Error(`${t.name} holds an exact zero`);
    if (slotOf[mag] === 0) {
      slotOf[mag] = magOfSlot.length;
      magOfSlot.push(mag);
      K = Math.max(K, twosNeeded(mag));
    }
  }
}

// the LUT: one line per slot, at scale K
mkdirSync(out, { recursive: true });
const lutLines = ["slot,n,m,core,fives,tens"];
for (let slot = 1; slot < magOfSlot.length; slot++) {
  const [n, m] = relation(magOfSlot[slot]);
  let odd = BigInt(n);
  let twos = 0;
  while (odd % 2n === 0n) {
    odd /= 2n;
    twos++;
  }
  // n/2^m · 10^K = odd · 2^twos / 2^m · 2^K · 5^K = odd · 5^(m − twos) · 10^(K − m + twos) when m > twos;
  // when m ≤ twos the value is already whole before any ×(2×5): odd · 2^(twos − m) · 10^K ("fives" then negative: twos)
  const fives = m - twos;
  const tens = K - Math.max(0, fives);
  let core = fives >= 0 ? odd * 5n ** BigInt(fives) : odd * 2n ** BigInt(-fives);
  let zeros = tens;
  while (core % 10n === 0n) {
    core /= 10n;
    zeros++;
  }
  lutLines.push(`${slot},${n},${m},${core},${fives},${zeros}`);
}
const lutBody = lutLines.join("\n") + "\n";
writeFileSync(`${out}lut.csv`, lutBody);
process.stdout.write(`pass 1: ${tables.length} tables · ${magOfSlot.length - 1} slots · K = ${K} · lut.csv sha256 ${createHash("sha256").update(lutBody).digest("hex").slice(0, 8)}… · ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

// pass 2: pointer tables
const manifest: { name: string; shape: number[]; rows: number; columns: number; distinct: number; file: string; bytes: number; sha256: string }[] = [];
const slotText = new Array<string>(0x10000);
for (let mag = 1; mag < 0x8000; mag++) {
  if (slotOf[mag] === 0) continue;
  slotText[mag] = String(slotOf[mag]);
  slotText[mag | 0x8000] = `-${slotOf[mag]}`;
}
for (const t of tables) {
  if (!filter.test(t.name)) continue;
  const tt = Date.now();
  const words = wordsOf(t);
  const [rows, columns] = t.shape.length === 1 ? [1, t.shape[0]] : [t.shape[0], t.shape[1]];
  const seen = new Uint8Array(0x8000);
  let distinct = 0;
  const file = `${out}${t.name.split(".").join("/")}.csv`;
  mkdirSync(dirname(file), { recursive: true });
  const fd = openSync(file, "w");
  const hash = createHash("sha256");
  let written = 0;
  let chunk: string[] = [];
  let chunkLen = 0;
  const flush = () => {
    const b = Buffer.from(chunk.join(""), "utf8");
    writeSync(fd, b);
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
    for (let c = 0; c < columns; c++) {
      const w = words[r * columns + c];
      const mag = w & 0x7fff;
      if (!seen[mag]) {
        seen[mag] = 1;
        distinct++;
      }
      row[c + 1] = slotText[w];
    }
    push(row.join(",") + "\n");
  }
  flush();
  closeSync(fd);
  manifest.push({ name: t.name, shape: t.shape, rows, columns, distinct, file: file.slice(out.length), bytes: written, sha256: hash.digest("hex") });
  process.stdout.write(`${t.name.padEnd(48)} ${t.shape.join("x").padEnd(12)} distinct ${String(distinct).padEnd(6)} ${(written / 1e6).toFixed(1).padStart(9)} MB  ${((Date.now() - tt) / 1000).toFixed(1)}s\n`);
}
const body = JSON.stringify(
  { model: "Qwen3-1.7B", slots: magOfSlot.length - 1, K, lut: "lut.csv", rule: "cells are ±slot into lut.csv; value = core followed by tens zeros, the whole model at one scale (2×5)^K", tables: manifest },
  null,
  1,
);
const manifestFile = `${out}manifest${process.argv[2] ? "-partial" : ""}.json`;
writeFileSync(manifestFile, body);
process.stdout.write(`${manifest.length} pointer tables · ${(manifest.reduce((a, m) => a + m.bytes, 0) / 1e9).toFixed(3)} GB · ${((Date.now() - t0) / 1000).toFixed(1)}s · ${manifestFile}\n`);
