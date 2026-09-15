import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readSync, writeFileSync, writeSync } from "node:fs";
import { ALL, SIDE, draw, eOfWord, lit, wordOfE } from "./cube.ts";

/**
 * Job 1 (pgarcia): the values as a string of minified smaller integers, the minification outside the table values.
 * Every stored magnitude n/2^m is odd·2^twos/2^m = odd/2^e, e = m − twos. Measured: the 4,505 slots are exactly
 * 4,505 distinct (odd, e) pairs, odd one of the 128 odd ints 1…255, e one of 49 values −6…46.
 * e is minified as the first byte type, the cube (src/cube.ts): 3³ cells on/off and 2, the side. The centre lives
 * outside every table and is not chosen: pass 1 meets e's lowest and highest over every weight; the cube reaches ±26
 * from its centre, so the span must be at most 52 and even, and the centre is its middle.
 *   values.csv    per table: header = column numbers, rows numbered, cells ±odd (the sign is the plane)
 *   minify.cube   per table: one little-endian 32-bit word per weight in table order, one cell on, bit 27 the side
 *   cubes.csv     per table: row, then the row as one thing — its + side cube and its − side cube (all cells its
 *                 weights turn on), drawn as x-layers of y-rows of z; line 0 is the whole table
 *   e.csv         one line per e met: its side and cell, slots, and its 5D reading (fives, tens down, twos)
 * Self-check over EVERY weight: the stored word is rebuilt from (value sign, odd, cube word, centre) alone.
 * Usage: tables-minify.ts [name filter regex for which tables are written]
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
const filter = process.argv[2] ? new RegExp(process.argv[2]) : null;

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
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { shape: number[]; data_offsets: [number, number] }>)) {
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

// magnitude → (odd, e)
const oddOf = new Int16Array(0x8000);
const eOf = new Int16Array(0x8000);
for (let mag = 1; mag < 0x8000; mag++) {
  const E = (mag >> 7) & 0xff;
  let odd = E > 0 ? 128 + (mag & 0x7f) : mag & 0x7f;
  let e = E > 0 ? 134 - E : 133;
  while (odd % 2 === 0) {
    odd /= 2;
    e--;
  }
  oddOf[mag] = odd;
  eOf[mag] = e;
}
/** the stored word, from the strings alone: shift odd up until it is a whole bf16 n (128…255), or the subnormal floor */
const rebuild = (minus: boolean, odd: number, e: number) => {
  let n = odd;
  let m = e;
  while (n < 128 && m < 133) {
    n *= 2;
    m++;
  }
  const mag = m === 133 && n < 128 ? n : ((134 - m) << 7) | (n - 128);
  return (minus ? 0x8000 : 0) | mag;
};

class Out {
  readonly fd: number;
  readonly hash = createHash("sha256");
  bytes = 0;
  chunk: Buffer[] = [];
  length = 0;
  constructor(file: string) {
    this.fd = openSync(file, "w");
  }
  push(b: Buffer | string) {
    const buf = typeof b === "string" ? Buffer.from(b, "utf8") : b;
    this.chunk.push(buf);
    this.length += buf.length;
    if (this.length > 8_000_000) this.flush();
  }
  flush() {
    const b = Buffer.concat(this.chunk);
    writeSync(this.fd, b);
    this.hash.update(b);
    this.bytes += b.length;
    this.chunk = [];
    this.length = 0;
  }
  close() {
    this.flush();
    closeSync(this.fd);
    return { bytes: this.bytes, sha256: this.hash.digest("hex") };
  }
}

const t0 = Date.now();
// pass 1: every weight — the magnitudes met, e's lowest and highest
const metMag = new Uint8Array(0x8000);
const eSlots = new Map<number, number>();
let weights = 0;
let eLowest = 999;
let eHighest = -999;
for (const t of tables) {
  const words = wordsOf(t);
  for (let i = 0; i < words.length; i++) {
    const mag = words[i] & 0x7fff;
    if (metMag[mag]) continue;
    metMag[mag] = 1;
    const e = eOf[mag];
    eSlots.set(e, (eSlots.get(e) ?? 0) + 1);
    if (e < eLowest) eLowest = e;
    if (e > eHighest) eHighest = e;
  }
  weights += words.length;
}
const span = eHighest - eLowest;
if (span > 52) throw new Error(`e spans ${eLowest}…${eHighest}: wider than the cube's ±26`);
if (span % 2 !== 0) throw new Error(`e spans ${eLowest}…${eHighest}: no middle, the centre would be a choice`);
const centre = eLowest + span / 2;
process.stdout.write(`pass 1: ${weights} weights · e ${eLowest}…${eHighest} · span ${span} (cube reach 52) · centre ${centre} · ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

// every met magnitude's cube word; the decoded e of every distinct word
const wordOf = new Uint32Array(0x8000);
const eOfCube = new Map<number, number>();
let modelPlus = 0;
let modelMinus = 0;
for (let mag = 1; mag < 0x8000; mag++) {
  if (!metMag[mag]) continue;
  const w = wordOfE(eOf[mag], centre);
  wordOf[mag] = w;
  if (!eOfCube.has(w)) eOfCube.set(w, eOfWord(w, centre));
  if (w & SIDE) modelMinus |= w & ALL;
  else modelPlus |= w & ALL;
}
const valueText = new Array<string>(0x10000);
for (let mag = 1; mag < 0x8000; mag++) {
  if (!metMag[mag]) continue;
  valueText[mag] = String(oddOf[mag]);
  valueText[mag | 0x8000] = `-${oddOf[mag]}`;
}

// pass 2: every weight rebuilt from the strings and the cube alone; the filtered tables written
mkdirSync(out, { recursive: true });
const manifest: object[] = [];
let bad = 0;
for (const t of tables) {
  const words = wordsOf(t);
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const text = valueText[w];
    const minus = text.charCodeAt(0) === 45;
    if (rebuild(minus, Number(minus ? text.slice(1) : text), eOfCube.get(wordOf[w & 0x7fff])!) !== w) bad++;
  }
  if (!filter?.test(t.name)) continue;

  const [rows, columns] = t.shape.length === 1 ? [1, t.shape[0]] : [t.shape[0], t.shape[1]];
  const path = `${out}${t.name.split(".").join("/")}/`;
  mkdirSync(path, { recursive: true });
  const values = new Out(`${path}values.csv`);
  const cubeFile = new Out(`${path}minify.cube`);
  const header: string[] = [""];
  for (let c = 1; c <= columns; c++) header.push(String(c));
  values.push(header.join(",") + "\n");
  const vRow: string[] = new Array(columns + 1);
  const rowCubes: string[] = [];
  let tablePlus = 0;
  let tableMinus = 0;
  for (let r = 0; r < rows; r++) {
    vRow[0] = String(r + 1);
    const cubeRow = new Uint32Array(columns);
    let plus = 0;
    let minus = 0;
    for (let c = 0; c < columns; c++) {
      const w = words[r * columns + c];
      const word = wordOf[w & 0x7fff];
      vRow[c + 1] = valueText[w];
      cubeRow[c] = word;
      if (word & SIDE) minus |= word & ALL;
      else plus |= word & ALL;
    }
    values.push(vRow.join(",") + "\n");
    cubeFile.push(Buffer.from(cubeRow.buffer));
    rowCubes.push(`${r + 1},${draw(plus)},${draw(minus)}`);
    tablePlus |= plus;
    tableMinus |= minus;
  }
  const v = values.close();
  const cb = cubeFile.close();
  const cubesBody = ["row,plus,minus", `0,${draw(tablePlus)},${draw(tableMinus)}`, ...rowCubes].join("\n") + "\n";
  writeFileSync(`${path}cubes.csv`, cubesBody);
  const cs = createHash("sha256").update(cubesBody).digest("hex");
  manifest.push({ name: t.name, shape: t.shape, rows, columns, plus: draw(tablePlus), minus: draw(tableMinus), values: v, cube: cb, cubes: { sha256: cs } });
  process.stdout.write(
    `${t.name}\n  cube + ${draw(tablePlus)} (${lit(tablePlus).length} on) · − ${draw(tableMinus)} (${lit(tableMinus).length} on)\n` +
      `  values ${(v.bytes / 1e6).toFixed(1)} MB (${v.sha256.slice(0, 8)}…) · minify.cube ${(cb.bytes / 1e6).toFixed(1)} MB (${cb.sha256.slice(0, 8)}…) · cubes.csv ${rows + 1} lines (${cs.slice(0, 8)}…)\n`,
  );
}

const eLines = ["e,side,x,y,z,slots,fives,tens_down,twos"];
for (const e of [...eSlots.keys()].sort((a, b) => a - b)) {
  const w = wordOfE(e, centre);
  const [x, y, z] = lit(w & ALL)[0];
  const reading = e >= 0 ? `${e},${e},0` : `0,0,${-e}`;
  eLines.push(`${e},${w & SIDE ? "-" : "+"},${x},${y},${z},${eSlots.get(e)},${reading}`);
}
const eBody = eLines.join("\n") + "\n";
writeFileSync(`${out}e.csv`, eBody);
const rule = `value = ±odd (values.csv); its e outside, as a cube (minify.cube: 27 cells, one on, bit 27 the side) from centre ${centre}; e ≥ 0 → odd·5^e then e decades down; e < 0 → odd·2^−e`;
writeFileSync(`${out}manifest${filter ? "-partial" : ""}.json`, JSON.stringify({ model: "Qwen3-1.7B", rule, centre, bitOrder: "cell (x,y,z) = bit x·9+y·3+z", model_cube: { plus: draw(modelPlus), minus: draw(modelMinus) }, e: "e.csv", tables: manifest }, null, 1));
process.stdout.write(
  `\nmodel cube + ${draw(modelPlus)} (${lit(modelPlus).length} on) · − ${draw(modelMinus)} (${lit(modelMinus).length} on)\n` +
    `weights ${weights} · rebuilt from (value sign, odd, cube word, centre ${centre}): ${weights - bad} same word, ${bad} different\n` +
    `e.csv ${eLines.length - 1} e values (${createHash("sha256").update(eBody).digest("hex").slice(0, 8)}…) · ${((Date.now() - t0) / 1000).toFixed(1)}s\n`,
);
