import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";

/**
 * Oscillated or rotating (pgarcia: only two things do this; random doesn't speak). On the up×downᵀ trace, per layer,
 * ±residual codes (0…62 on +, 63…125 on −), real against down shuffled in place (LCG seed 20260914).
 *   flip       cells (j,c) where up[j,c] and down[c,j] sit on opposite planes            — oscillation between planes
 *   lag L      cells where the cycle-1 cell (up[j,c], down[c,j]) equals the one at c+L, L = 1…16 — oscillation along c
 *   shift s    cells where up[j,c] equals down[(c+s) mod 2048, j], s = 0…2047          — rotation along c
 *              (layers 0, 13, 27; every shift)
 *   osc-rot.txt, rot-<layer>.csv  shift,matches_real,matches_shuffled
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const SEED = 20260914;
const H = 2048;
const I = 6144;
const ROTATE = new Set([0, 13, 27]);

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

type T = { shard: string; start: number; end: number; base: number };
const tables = new Map<string, T>();
for (const s of ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]) {
  const fd = openSync(dir + s, "r");
  const len = Buffer.alloc(8);
  readSync(fd, len, 0, 8, 0);
  const n = Number(len.readBigUInt64LE(0));
  const head = Buffer.alloc(n);
  readSync(fd, head, 0, n, 8);
  closeSync(fd);
  for (const [name, t] of Object.entries(JSON.parse(head.toString("utf8")) as Record<string, { data_offsets: [number, number] }>)) {
    if (name !== "__metadata__") tables.set(name, { shard: s, start: t.data_offsets[0], end: t.data_offsets[1], base: 8 + n });
  }
}
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

/** down[c,j] laid out j × c, so every trace reads along c */
const transposed = (down: Uint8Array) => {
  const t = new Uint8Array(I * H);
  for (let c = 0; c < H; c++) for (let j = 0; j < I; j++) t[j * H + c] = down[c * I + j];
  return t;
};
const measure = (up: Uint8Array, dt: Uint8Array, rotate: boolean) => {
  let flip = 0;
  for (let i = 0; i < up.length; i++) if (up[i] >= 63 !== dt[i] >= 63) flip++;
  const lags: number[] = [];
  for (let L = 1; L <= 16; L++) {
    let m = 0;
    for (let j = 0; j < I; j++) {
      const b = j * H;
      for (let c = 0; c + L < H; c++) if (up[b + c] === up[b + c + L] && dt[b + c] === dt[b + c + L]) m++;
    }
    lags.push(m);
  }
  const shifts = new Float64Array(rotate ? H : 0);
  if (rotate) {
    for (let s = 0; s < H; s++) {
      let m = 0;
      for (let j = 0; j < I; j++) {
        const b = j * H;
        for (let c = 0; c < H - s; c++) if (up[b + c] === dt[b + c + s]) m++;
        for (let c = H - s; c < H; c++) if (up[b + c] === dt[b + c + s - H]) m++;
      }
      shifts[s] = m;
    }
  }
  return { flip, lags, shifts };
};

const summary: string[] = [`oscillated or rotating · up×downᵀ · seed ${SEED}`, "layer · flip real/shuffled (of 12,582,912) · lag 1…16 matches real | shuffled"];
const t0 = Date.now();
for (let layer = 0; layer < 28; layer++) {
  const L = `model.layers.${layer}`;
  const up = codesOf(`${L}.mlp.up_proj.weight`);
  const down = codesOf(`${L}.mlp.down_proj.weight`);
  const rotate = ROTATE.has(layer);
  const real = measure(up, transposed(down), rotate);
  const shuf = measure(up, transposed(shuffled(down)), rotate);
  summary.push(`${layer} · flip ${real.flip}/${shuf.flip} · lags ${real.lags.join(" ")} | ${shuf.lags.join(" ")}`);
  if (rotate) {
    const lines = ["shift,matches_real,matches_shuffled"];
    for (let s = 0; s < H; s++) lines.push(`${s},${real.shifts[s]},${shuf.shifts[s]}`);
    writeFileSync(`${out}rot-${layer}.csv`, lines.join("\n") + "\n");
    const top = [...real.shifts.keys()].sort((a, b) => real.shifts[b] - real.shifts[a]).slice(0, 8);
    const sMax = Math.max(...shuf.shifts);
    const sMin = Math.min(...shuf.shifts);
    const above = [...real.shifts].filter((m) => m > sMax).length;
    summary.push(
      `  rotation layer ${layer}: shift 0 real ${real.shifts[0]} · top shifts ${top.map((s) => `${s}:${real.shifts[s]}`).join(" ")} · real min ${Math.min(...real.shifts)} · shuffled range ${sMin}…${sMax} · real shifts above shuffled max ${above}`,
    );
  }
  process.stdout.write(`${summary.slice(-1 - (rotate ? 1 : 0)).join("\n")} · ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
}
writeFileSync(`${out}osc-rot.txt`, summary.join("\n") + "\n");
