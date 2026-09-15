import { createHash } from "node:crypto";
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * How many + and how many − (pgarcia): one pass over every weight, counting its sign per slot. The same int on both
 * signs is a plane. Rolled up to cores through slots.csv.
 *   planes.csv       slot, core_id, band, plus, minus, plane
 *   core-planes.csv  core_id, plus, minus, plane
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const root = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-pointers/";
const slots = readFileSync(`${root}slots.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(","));

// stored magnitude → slot, from each slot's n/2^m
const slotOfMag = new Int32Array(0x8000);
for (const [slot, , , n, m] of slots) {
  const mm = Number(m);
  const nn = Number(n);
  const mag = mm === 133 && nn < 128 ? nn : ((134 - mm) << 7) | (nn - 128);
  slotOfMag[mag] = Number(slot);
}

const plus = new Float64Array(slots.length + 1);
const minus = new Float64Array(slots.length + 1);
let total = 0;
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
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const slot = slotOfMag[w & 0x7fff];
      if (slot === 0) throw new Error(`${name}: a magnitude with no slot`);
      if (w & 0x8000) minus[slot]++;
      else plus[slot]++;
    }
    total += words.length;
  }
  closeSync(fd);
}

const slotLines = ["slot,core_id,band,plus,minus,plane"];
const corePlus = new Map<number, number>();
const coreMinus = new Map<number, number>();
let slotPlanes = 0;
let slotOnlyPlus = 0;
let slotOnlyMinus = 0;
let allPlus = 0;
let allMinus = 0;
for (const [slot, coreId, band] of slots) {
  const p = plus[Number(slot)];
  const q = minus[Number(slot)];
  allPlus += p;
  allMinus += q;
  const plane = p > 0 && q > 0;
  if (plane) slotPlanes++;
  else if (p > 0) slotOnlyPlus++;
  else slotOnlyMinus++;
  slotLines.push(`${slot},${coreId},${band},${p},${q},${plane ? "yes" : "no"}`);
  const c = Number(coreId);
  corePlus.set(c, (corePlus.get(c) ?? 0) + p);
  coreMinus.set(c, (coreMinus.get(c) ?? 0) + q);
}
const coreLines = ["core_id,plus,minus,plane"];
let corePlanes = 0;
let coreOnlyPlus = 0;
let coreOnlyMinus = 0;
for (const [c, p] of [...corePlus].sort((a, b) => a[0] - b[0])) {
  const q = coreMinus.get(c) ?? 0;
  const plane = p > 0 && q > 0;
  if (plane) corePlanes++;
  else if (p > 0) coreOnlyPlus++;
  else coreOnlyMinus++;
  coreLines.push(`${c},${p},${q},${plane ? "yes" : "no"}`);
}
const write = (file: string, lines: string[]) => {
  const body = lines.join("\n") + "\n";
  writeFileSync(`${root}${file}`, body);
  return createHash("sha256").update(body).digest("hex").slice(0, 8);
};
const a = write("planes.csv", slotLines);
const b = write("core-planes.csv", coreLines);
process.stdout.write(`weights ${total}: + ${allPlus} · − ${allMinus}\n`);
process.stdout.write(`slots ${slots.length}: planes (both signs) ${slotPlanes} · only + ${slotOnlyPlus} · only − ${slotOnlyMinus}   planes.csv ${a}…\n`);
process.stdout.write(`cores ${corePlus.size}: planes (both signs) ${corePlanes} · only + ${coreOnlyPlus} · only − ${coreOnlyMinus}   core-planes.csv ${b}…\n`);
const one = slotLines.slice(1).filter((l) => l.endsWith(",no"));
process.stdout.write(`\none-sided slots:\n${one.slice(0, 40).join("\n")}${one.length > 40 ? `\n… ${one.length - 40} more` : ""}\n`);
