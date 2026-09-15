import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { hyperLimit, primeLUT } from "./hyper.ts";

/**
 * A count of Pn for each diagonal cycle n^n — how many primes stand up to its limit and up to its whole — without
 * naming or listing a single one (pgarcia: Opus numbers mask them, so memory never runs out). The count keeps only
 * two tables of about √x entries: how many primes stand up to each small value and up to each x/i. Every prime p up to
 * √x crosses off, in both tables at once, the names it reaches (its crossings), never storing which they were.
 * Machine side, flagged: a count is floatland's, doubles hold values below 2^53, and division is floored with a check.
 */

const div = (a: number, b: number) => {
  let q = Math.floor(a / b);
  if (q * b > a) q--;
  else if ((q + 1) * b <= a) q++;
  return q;
};

/** how many primes up to x (x < 2^53) */
export function countPn(x: number): number {
  if (x < 2) return 0;
  const r = Math.floor(Math.sqrt(x));
  const lo = new Float64Array(r + 2); // lo[v]: names 2…v not yet crossed off
  const hi = new Float64Array(r + 2); // hi[i]: names 2…⌊x/i⌋ not yet crossed off
  for (let v = 1; v <= r + 1; v++) lo[v] = v - 1;
  for (let i = 1; i <= r + 1; i++) hi[i] = div(x, i) - 1;
  for (let p = 2; p <= r; p++) {
    if (lo[p] === lo[p - 1]) continue; // p was crossed off: not a prime
    const below = lo[p - 1];
    const pp = p * p;
    const top = Math.min(r, div(x, pp));
    for (let i = 1; i <= top; i++) {
      const d = i * p;
      hi[i] -= (d <= r ? hi[d] : lo[div(x, d)]) - below;
    }
    for (let v = r; v >= pp; v--) lo[v] -= lo[div(v, p)] - below;
  }
  return hi[1];
}

const lines: string[] = [];
const say = (s = "") => {
  lines.push(s);
  process.stdout.write(s + "\n");
};
const t0 = Date.now();

say("format=v2.mm.hyper.pn.v1  a count of Pn for each diagonal cycle n^n, no prime named or listed");

// gate: the count against the LUT
const lut = primeLUT(17_403_400);
const lutCount = (x: number) => {
  let lo = 0;
  let hiIdx = lut.length;
  while (lo < hiIdx) {
    const mid = (lo + hiIdx) >> 1;
    if (lut[mid] <= x) lo = mid + 1;
    else hiIdx = mid;
  }
  return lo;
};
let gate = true;
for (const x of [2, 3, 26, 255, 1000, 3093, 46655, 823415, 1_000_000, 16_777_215, 17_403_391]) {
  const same = countPn(x) === lutCount(x);
  gate &&= same;
  say(`  gate Pn(${x}) = ${countPn(x)} · LUT ${lutCount(x)} · ${same ? "EQUAL" : "DIFFERENT"}`);
}
say(`gate: ${gate ? "EQUAL" : "DIFFERENT"}`);
say();

say("cycle   whole                limit                Pn up to the limit   Pn in the corner   time");
for (let n = 2; n <= 13; n++) {
  const t = Date.now();
  const h = hyperLimit(String(n), n, ["2", "3", "5", "7", "11"].filter((p) => Number(p) < n));
  const inLimit = countPn(Number(h.limit));
  const inWhole = countPn(Number(h.whole));
  say(`${`${n}^${n}`.padEnd(7)} ${String(h.whole).padEnd(20)} ${String(h.limit).padEnd(20)} ${String(inLimit).padEnd(20)} ${String(inWhole - inLimit).padEnd(18)} ${((Date.now() - t) / 1000).toFixed(1)}s`);
}
say();
say(`machine side: counts in doubles (every value below 2^53); total ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const body = lines.filter((l) => !l.startsWith("machine side")).join("\n") + "\n";
const receipt = body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;
writeFileSync(new URL("../receipts/hyper-pn.txt", import.meta.url), receipt);
process.stdout.write(receipt.slice(-72));
