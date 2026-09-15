import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * Once everything is bigrammed (pgarcia): the cycle-1 atlas of one table (layer 0 k_proj) was already the complete
 * square 126² of ±residual cells. Cycle 2 — bigrams of bigrams — has 126⁴ = 252,047,376 possible cells, more than one
 * table holds, fewer than the model holds (2,031,739,904 / 4). Over every table, every row, columns paired 1-2|3-4:
 * which cycle-1 and cycle-2 cells appear at all, model-wide.
 * ±residual code: 0…62 the residual's place among the 63 on +, 63…125 on −. Machine side: the cell's place in the
 * square/hypersquare is used as a bit index (a presence map, 31.5 MB), not as a value.
 *   bigram-model.txt
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const oddRows = readFileSync(`${out}odds.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number));
const residuals = [...new Set(oddRows.map((r) => r[2]))].sort((a, b) => a - b);
if (residuals.length !== 63) throw new Error(`residuals ${residuals.length}`);
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

const Q = 126;
const pairSeen = new Uint8Array(Q * Q);
const quadSeen = new Uint8Array(Math.ceil((Q * Q * Q * Q) / 8));
let pairCells = 0;
let quadCells = 0;
let pairs = 0;
let quads = 0;
const t0 = Date.now();
const lines: string[] = [];
for (const s of ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]) {
  const fd = openSync(dir + s, "r");
  const len = Buffer.alloc(8);
  readSync(fd, len, 0, 8, 0);
  const hn = Number(len.readBigUInt64LE(0));
  const head = Buffer.alloc(hn);
  readSync(fd, head, 0, hn, 8);
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { shape: number[]; data_offsets: [number, number] }>)) {
    if (name === "__metadata__") continue;
    const bytes = Buffer.alloc(t.data_offsets[1] - t.data_offsets[0]);
    readSync(fd, bytes, 0, bytes.length, 8 + hn + t.data_offsets[0]);
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    const [rows, columns] = t.shape.length === 1 ? [1, t.shape[0]] : [t.shape[0], t.shape[1]];
    if (columns % 4 !== 0) throw new Error(`${name}: ${columns} columns hold no 4`);
    for (let r = 0; r < rows; r++) {
      const base = r * columns;
      for (let c = 0; c < columns; c += 4) {
        const a = codeOfWord[words[base + c]];
        const b = codeOfWord[words[base + c + 1]];
        const x = codeOfWord[words[base + c + 2]];
        const y = codeOfWord[words[base + c + 3]];
        const p = a * Q + b;
        const q = x * Q + y;
        if (!pairSeen[p]) {
          pairSeen[p] = 1;
          pairCells++;
        }
        if (!pairSeen[q]) {
          pairSeen[q] = 1;
          pairCells++;
        }
        const i = p * Q * Q + q;
        const bit = 1 << (i & 7);
        const at = (i - (i & 7)) / 8;
        if (!(quadSeen[at] & bit)) {
          quadSeen[at] |= bit;
          quadCells++;
        }
        pairs += 2;
        quads++;
      }
    }
    lines.push(`${name.padEnd(48)} after: cycle-1 cells ${pairCells} · cycle-2 cells ${quadCells}`);
  }
  closeSync(fd);
}
const summary = [
  `model-wide, ±residual basis (126 cells), columns paired in place:`,
  `cycle 1: ${pairCells} of ${Q * Q} possible cells present (held ${pairs})`,
  `cycle 2: ${quadCells} of ${Q * Q * Q * Q} possible cells present (held ${quads})`,
  `${((Date.now() - t0) / 1000).toFixed(1)}s`,
];
writeFileSync(`${out}bigram-model.txt`, [...summary, "", "per table, running:", ...lines].join("\n") + "\n");
process.stdout.write(summary.join("\n") + "\n" + lines.filter((_, i) => i % 40 === 0 || i === lines.length - 1).join("\n") + "\n");
