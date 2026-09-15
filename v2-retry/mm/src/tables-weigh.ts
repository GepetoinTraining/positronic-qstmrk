import { createHash } from "node:crypto";
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * How much do the weights weigh now (pgarcia) — tensored, not CSV. One pass over every weight, per table: how wide a
 * packed tensor cell must be for each representation we have earned, and what the outside tables (dictionaries) cost.
 * A cell's width is the fewest bits whose count of codes holds every distinct code the table uses.
 *   bf16            the stored word, 16 bits
 *   ±slot           sign + the slot, global LUT (4,505 slots)
 *   ±slot per table sign + the table's own dictionary of the slots it holds
 *   sign·odd·e      sign + odd (128 = 2⁷) + the e the table holds (the cube cell)
 *   sign·res·handle sign + residual (63, in 2⁶) + the handles (minifier, e) the table holds
 * Outside tables are counted at 2 bytes per dictionary entry (a bf16 word per entry names it).
 * Also: whether lm_head is the same table as embed_tokens (sha256 of the stored bytes).
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const residualOfOdd = new Map(
  readFileSync(`${out}odds.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number)).map((r) => [r[0], [r[1], r[2]]] as [number, [number, number]]),
);

// magnitude → odd, e, residual, handle
const oddOf = new Int16Array(0x8000);
const eOf = new Int16Array(0x8000);
const residualOf = new Int16Array(0x8000);
const handleOf = new Int32Array(0x8000);
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
  const [minifier, residual] = residualOfOdd.get(odd) ?? [odd, 1];
  residualOf[mag] = residual;
  handleOf[mag] = minifier * 1000 + (e + 500); // a key, not a value
}
const width = (codes: number) => {
  let w = 0;
  while (2 ** w < codes) w++;
  return w;
};

const layouts = ["bf16", "±slot", "±slot per table", "sign·odd·e", "sign·res·handle"] as const;
const cellBits = new Map<string, number>(layouts.map((l) => [l, 0]));
const outsideBytes = new Map<string, number>(layouts.map((l) => [l, 0]));
const hashes = new Map<string, string>();
const metMag = new Uint8Array(0x8000);
const residualsAll = new Set<number>();
let weights = 0;
let tables = 0;
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
    if (name === "lm_head.weight" || name === "model.embed_tokens.weight") hashes.set(name, createHash("sha256").update(bytes).digest("hex"));
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    const mags = new Set<number>();
    const es = new Set<number>();
    const handles = new Set<number>();
    const seen = new Uint8Array(0x8000);
    for (let i = 0; i < words.length; i++) {
      const mag = words[i] & 0x7fff;
      if (seen[mag]) continue;
      seen[mag] = 1;
      mags.add(mag);
      es.add(eOf[mag]);
      handles.add(handleOf[mag]);
      residualsAll.add(residualOf[mag]);
      metMag[mag] = 1;
    }
    const n = words.length;
    cellBits.set("bf16", cellBits.get("bf16")! + n * 16);
    cellBits.set("±slot per table", cellBits.get("±slot per table")! + n * (1 + width(mags.size)));
    outsideBytes.set("±slot per table", outsideBytes.get("±slot per table")! + mags.size * 2);
    cellBits.set("sign·odd·e", cellBits.get("sign·odd·e")! + n * (1 + 7 + width(es.size)));
    outsideBytes.set("sign·odd·e", outsideBytes.get("sign·odd·e")! + es.size * 2);
    cellBits.set("sign·res·handle", cellBits.get("sign·res·handle")! + n * (1 + 6 + width(handles.size)));
    outsideBytes.set("sign·res·handle", outsideBytes.get("sign·res·handle")! + handles.size * 2);
    weights += n;
    tables++;
  }
  closeSync(fd);
}
let slots = 0;
for (let mag = 1; mag < 0x8000; mag++) slots += metMag[mag];
cellBits.set("±slot", weights * (1 + width(slots)));
outsideBytes.set("±slot", slots * 2);
if (residualsAll.size > 64) throw new Error(`residuals ${residualsAll.size} do not fit 2⁶`);

const gb = (b: number) => (b / 1e9).toFixed(3);
const lines = [`Qwen3-1.7B · ${tables} tables · ${weights} weights · ${slots} slots · ${residualsAll.size} residuals`];
for (const l of layouts) {
  const cells = Math.ceil(cellBits.get(l)! / 8);
  const outside = outsideBytes.get(l)!;
  lines.push(`${l.padEnd(18)} cells ${gb(cells)} GB + outside ${(outside / 1e3).toFixed(1)} KB = ${gb(cells + outside)} GB · ${((cellBits.get(l)! / weights)).toFixed(2)} bits/weight avg`);
}
const tied = hashes.get("lm_head.weight") === hashes.get("model.embed_tokens.weight");
lines.push(`lm_head = embed_tokens byte for byte: ${tied ? "yes" : "no"} (${hashes.get("lm_head.weight")?.slice(0, 8)}… / ${hashes.get("model.embed_tokens.weight")?.slice(0, 8)}…)`);
const body = lines.join("\n") + "\n";
writeFileSync(`${out}weigh.txt`, body);
process.stdout.write(body);
