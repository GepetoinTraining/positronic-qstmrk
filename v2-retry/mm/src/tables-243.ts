import { createHash } from "node:crypto";
import { closeSync, openSync, readSync, writeFileSync } from "node:fs";

/**
 * The tell (pgarcia): 3⁵, the step sequence, the hyper pyramid. Does every table hold at least one 243?
 * One pass over every weight: per table, how many weights have odd 243, on each plane.
 *   243.csv  table, weights, plus, minus, total
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";

const is243 = new Uint8Array(0x8000);
for (let mag = 1; mag < 0x8000; mag++) {
  const E = (mag >> 7) & 0xff;
  let odd = E > 0 ? 128 + (mag & 0x7f) : mag & 0x7f;
  while (odd % 2 === 0) odd /= 2;
  if (odd === 243) is243[mag] = 1;
}

type Row = { name: string; weights: number; plus: number; minus: number };
const rows: Row[] = [];
for (const s of ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]) {
  const fd = openSync(dir + s, "r");
  const len = Buffer.alloc(8);
  readSync(fd, len, 0, 8, 0);
  const hn = Number(len.readBigUInt64LE(0));
  const head = Buffer.alloc(hn);
  readSync(fd, head, 0, hn, 8);
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { data_offsets: [number, number] }>)) {
    if (name === "__metadata__") continue;
    const bytes = Buffer.alloc(t.data_offsets[1] - t.data_offsets[0]);
    readSync(fd, bytes, 0, bytes.length, 8 + hn + t.data_offsets[0]);
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    let plus = 0;
    let minus = 0;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (!is243[w & 0x7fff]) continue;
      if (w & 0x8000) minus++;
      else plus++;
    }
    rows.push({ name, weights: words.length, plus, minus });
  }
  closeSync(fd);
}
const order = (name: string) => name.replace(/\.(\d+)\./, (_, d) => `.${d.padStart(2, "0")}.`);
rows.sort((a, b) => (order(a.name) < order(b.name) ? -1 : 1));

const lines = ["table,weights,plus,minus,total", ...rows.map((r) => `${r.name},${r.weights},${r.plus},${r.minus},${r.plus + r.minus}`)];
const body = lines.join("\n") + "\n";
writeFileSync(`${out}243.csv`, body);
const none = rows.filter((r) => r.plus + r.minus === 0);
const fewest = [...rows].sort((a, b) => a.plus + a.minus - (b.plus + b.minus)).slice(0, 8);
const all = rows.reduce((a, r) => a + r.plus + r.minus, 0);
const plus = rows.reduce((a, r) => a + r.plus, 0);
process.stdout.write(
  `tables ${rows.length} · with at least one 243: ${rows.length - none.length} · with none: ${none.length}\n` +
    (none.length ? `none:\n${none.map((r) => `  ${r.name} (${r.weights} weights)`).join("\n")}\n` : "") +
    `fewest:\n${fewest.map((r) => `  ${r.name}: ${r.plus + r.minus} (+${r.plus} −${r.minus}) of ${r.weights}`).join("\n")}\n` +
    `total 243s ${all}: + ${plus} · − ${all - plus} · 243.csv sha256 ${createHash("sha256").update(body).digest("hex").slice(0, 8)}…\n`,
);
