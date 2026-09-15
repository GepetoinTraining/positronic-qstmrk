import { createHash } from "node:crypto";
import { closeSync, openSync, readSync, writeFileSync } from "node:fs";
import { factor, primeLUT } from "./hyper.ts";

/**
 * The odds still minify (pgarcia): minify each odd until it is a prime, or prime², or prime³, … — keep that residual.
 * Minifying goes from the smallest prime up and stops as soon as one prime is left, so the residual is the power of the
 * odd's largest prime and the minifier (odd ÷ residual) is everything below it. It lives outside, like e.
 * [B] 1 has no prime: it stays as it is, residual 1, minifier 1.
 *   odds.csv  odd, minifier, residual, prime, power, slots, weights
 * Weights counted in one pass over every table.
 */

const dir = "D:/positronic-qstmrk/models/qwen3-1.7b/";
const out = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const lut = primeLUT(255);

// every magnitude's odd, and how many weights hold each
const oddOfMag = new Int16Array(0x8000);
for (let mag = 1; mag < 0x8000; mag++) {
  const E = (mag >> 7) & 0xff;
  let odd = E > 0 ? 128 + (mag & 0x7f) : mag & 0x7f;
  while (odd % 2 === 0) odd /= 2;
  oddOfMag[mag] = odd;
}
const weightsOf = new Float64Array(256);
const slotsOf = new Float64Array(256);
const metMag = new Uint8Array(0x8000);
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
      const mag = words[i] & 0x7fff;
      weightsOf[oddOfMag[mag]]++;
      if (!metMag[mag]) {
        metMag[mag] = 1;
        slotsOf[oddOfMag[mag]]++;
      }
    }
    total += words.length;
  }
  closeSync(fd);
}

const lines = ["odd,minifier,residual,prime,power,slots,weights"];
const residuals = new Set<number>();
const minifiers = new Set<number>();
let already = 0;
let minified = 0;
let weightsCheck = 0;
for (let odd = 1; odd < 256; odd += 2) {
  if (!slotsOf[odd]) continue;
  const parts = odd === 1 ? [] : factor(BigInt(odd), lut);
  // minify from the smallest prime up until one prime is left
  const [p, k] = parts.length ? parts[parts.length - 1] : [1n, 0];
  const residual = Number(p) ** k;
  let minifier = 1;
  for (const [q, j] of parts.slice(0, -1)) minifier *= Number(q) ** j;
  if (minifier * residual !== odd) throw new Error(`odd ${odd}: ${minifier}·${residual}`);
  if (minifier === 1) already++;
  else minified++;
  residuals.add(residual);
  minifiers.add(minifier);
  weightsCheck += weightsOf[odd];
  lines.push(`${odd},${minifier},${residual},${p},${k},${slotsOf[odd]},${weightsOf[odd]}`);
}
if (weightsCheck !== total) throw new Error(`weights ${weightsCheck} ≠ ${total}`);
const body = lines.join("\n") + "\n";
writeFileSync(`${out}odds.csv`, body);
const sorted = (s: Set<number>) => [...s].sort((a, b) => a - b).join(" ");
process.stdout.write(
  `odds ${lines.length - 1}: already one prime (or 1) ${already} · minified ${minified} · weights ${total}\n` +
    `residuals ${residuals.size}: ${sorted(residuals)}\n` +
    `minifiers ${minifiers.size}: ${sorted(minifiers)}\n` +
    `odds.csv sha256 ${createHash("sha256").update(body).digest("hex").slice(0, 8)}…\n`,
);
