import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * Cycle the oscillator over a table (pgarcia): the order-preserving move is removing the 2pow column arrangement.
 * Cycle 0: every distinct stored word (±value) is a cell, numbered from 0 in first-met order (rows, columns).
 * Each cycle pairs neighbouring columns (1 with 2, 3 with 4, …) into bigrams; every distinct bigram is a cell of that
 * cycle's atlas (left cell, right cell), first-met order. Then the bigrams are bigrammed, and so on, while the column
 * count still holds a 2. What is left: the odd column count, each row as that many top cells — and the atlases.
 * Nothing is kept per cell but at the top; the table is its atlases and its top cells.
 * Self-check: every row is unfolded from its top cells through the atlases alone and must be the stored row.
 *   <table>/bigram/cycle-<c>.csv  id,left,right    (cycle 0: id,word)
 *   <table>/bigram/top.csv        header = column numbers, rows numbered, cells = top cell ids
 * Usage: tables-bigram.ts <name filter regex>
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
const filter = new RegExp(process.argv[2] ?? "^$");
// basis (pgarcia, Job 1: calculate without the minified portion): "word" = the ±value itself; "residual" = ±residual,
// the handle (minifier, e) removed — cycle 0 cells are then sign·residual, and the unfold checks sign and residual
const basis = process.argv[3] === "residual" ? "residual" : "word";
const residualOfOdd = new Map(
  readFileSync(`${out}odds.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number)).map((r) => [r[0], r[2]] as [number, number]),
);
/** a stored word → its cycle-0 basis code: the word, or sign bit | residual */
const basisOf = (w: number) => {
  if (basis === "word") return w;
  const mag = w & 0x7fff;
  const E = (mag >> 7) & 0xff;
  let odd = E > 0 ? 128 + (mag & 0x7f) : mag & 0x7f;
  while (odd % 2 === 0) odd /= 2;
  return (w & 0x8000) | residualOfOdd.get(odd)!;
};
const SPAN = 2 ** 26;

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
    if (name === "__metadata__" || !filter.test(name)) continue;
    tables.push({ shard: s, name, shape: t.shape, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
const order = (name: string) => name.replace(/\.(\d+)\./, (_, d) => `.${d.padStart(2, "0")}.`);
tables.sort((a, b) => (order(a.name) < order(b.name) ? -1 : 1));

const write = (file: string, lines: string[]) => {
  const body = lines.join("\n") + "\n";
  writeFileSync(file, body);
  return createHash("sha256").update(body).digest("hex").slice(0, 8);
};

for (const t of tables) {
  const t0 = Date.now();
  const fd = openSync(dir + t.shard, "r");
  const bytes = Buffer.alloc(t.end - t.start);
  readSync(fd, bytes, 0, bytes.length, t.base + t.start);
  closeSync(fd);
  const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
  const [rows, columns] = t.shape.length === 1 ? [1, t.shape[0]] : [t.shape[0], t.shape[1]];
  const path = `${out}${t.name.split(".").join("/")}/bigram${basis === "word" ? "" : `-${basis}`}/`;
  mkdirSync(path, { recursive: true });

  // cycle 0: the distinct words
  const idOfWord = new Int32Array(0x10000).fill(-1);
  const wordOfId: number[] = [];
  let cur = new Int32Array(rows * columns);
  for (let i = 0; i < words.length; i++) {
    const w = basisOf(words[i]);
    if (idOfWord[w] < 0) {
      idOfWord[w] = wordOfId.length;
      wordOfId.push(w);
    }
    cur[i] = idOfWord[w];
  }
  const report = [`${t.name} ${t.shape.join("x")} · basis ${basis}`, `  cycle 0: columns ${columns} · cells ${wordOfId.length} (held ${rows * columns})`];
  const shas = [write(`${path}cycle-0.csv`, ["id,word", ...wordOfId.map((w, id) => `${id},${w & 0x8000 ? "-" : ""}${w & 0x7fff}`)])];

  // cycles: pair neighbouring columns while the column count holds a 2
  const lefts: Int32Array[] = [];
  const rights: Int32Array[] = [];
  let cols = columns;
  let cycle = 0;
  while (cols % 2 === 0) {
    cycle++;
    const half = cols / 2;
    const next = new Int32Array(rows * half);
    const idOf = new Map<number, number>();
    const left: number[] = [];
    const right: number[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < half; c++) {
        const a = cur[r * cols + 2 * c];
        const b = cur[r * cols + 2 * c + 1];
        const key = a * SPAN + b;
        let id = idOf.get(key);
        if (id === undefined) {
          id = left.length;
          idOf.set(key, id);
          left.push(a);
          right.push(b);
        }
        next[r * half + c] = id;
      }
    }
    lefts.push(Int32Array.from(left));
    rights.push(Int32Array.from(right));
    const lines = ["id,left,right"];
    for (let id = 0; id < left.length; id++) lines.push(`${id},${left[id]},${right[id]}`);
    shas.push(write(`${path}cycle-${cycle}.csv`, lines));
    report.push(`  cycle ${cycle}: columns ${half} · cells ${left.length} (held ${rows * half})`);
    cur = next;
    cols = half;
  }

  const header: string[] = [""];
  for (let c = 1; c <= cols; c++) header.push(String(c));
  const topLines = [header.join(",")];
  for (let r = 0; r < rows; r++) topLines.push(`${r + 1},${Array.from(cur.subarray(r * cols, (r + 1) * cols)).join(",")}`);
  shas.push(write(`${path}top.csv`, topLines));

  // self-check: unfold every row from its top cells through the atlases alone
  let bad = 0;
  const unfold = (id: number, c: number, into: number[]) => {
    if (c === 0) {
      into.push(wordOfId[id]);
      return;
    }
    unfold(lefts[c - 1][id], c - 1, into);
    unfold(rights[c - 1][id], c - 1, into);
  };
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) unfold(cur[r * cols + c], cycle, row);
    for (let c = 0; c < columns; c++) if (row[c] !== basisOf(words[r * columns + c])) bad++;
  }
  const atlas = wordOfId.length + lefts.reduce((a, l) => a + l.length, 0);
  report.push(
    `  left: ${cols} odd columns × ${rows} rows = ${rows * cols} top cells · atlas cells over all cycles ${atlas} · unfolded: ${rows * columns - bad} same, ${bad} different · ${((Date.now() - t0) / 1000).toFixed(1)}s`,
    `  files: cycle-0…${cycle}.csv, top.csv (${shas.join(" ")})`,
  );
  process.stdout.write(report.join("\n") + "\n");
}
