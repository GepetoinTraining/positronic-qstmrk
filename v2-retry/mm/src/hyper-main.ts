import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { DB_PATH } from "./db/store.ts";
import { factor, factorText, hyperLimit, primeLUT } from "./hyper.ts";

/**
 * Cycles up to 13¹³ without seats: the limit of MM[R ×D] for every ring R from 2 to 13 and every dimension D from 1
 * to 13, factored against the LUT. First a gate: the same reading must give every limit the seat-by-seat walks stored.
 */
const lines: string[] = [];
const say = (s = "") => lines.push(s);
const t0 = Date.now();

// the LUT reaches the square root of the largest whole (13¹³), so every limit factors completely
const lut = primeLUT(17_403_400);
const ascending = (top: number) => lut.filter((p) => p < top).map(String);

say("format=v2.mm.hyper.v1  the limit of MM[R ×D] as D slabs of boxes, no seats; factored against the LUT of primes");
say(`LUT: ${lut.length} primes, up to ${lut[lut.length - 1]} (machine side: made by multiplying names into crossings)`);
say();

// the gate against the walks
const db = new DatabaseSync(DB_PATH, { readOnly: true });
const walked = db
  .prepare(
    `SELECT r.walk, r.dimension, r.rail, r.container, t.reading AS limit_reading
     FROM edges e JOIN runs r ON r.id = e.run_id JOIN nodes t ON t.id = e.source
     WHERE e.family = 'limit' AND r.id IN (SELECT MAX(id) FROM runs GROUP BY walk, dimension) ORDER BY r.dimension, r.id`,
  )
  .all() as { walk: string; dimension: number; rail: string; container: string; limit_reading: string }[];
let gate = true;
say("gate: every walked limit read again without seats");
for (const w of walked) {
  const rail = JSON.parse(w.rail) as { label: string; reading: string }[];
  const label = w.container.split(",")[0];
  const ring = rail.find((r) => r.label === label)?.reading ?? label;
  const h = hyperLimit(ring, w.dimension, ascending(Number(ring)));
  const same = String(h.limit) === w.limit_reading;
  gate &&= same;
  say(`  ${w.dimension}D ${w.walk.padEnd(8)} MM[${ring} ×${w.dimension}]  f ${h.fits}  corner ${h.corner}^${w.dimension}  limit ${h.limit}  walked ${w.limit_reading}  ${same ? "EQUAL" : "DIFFERENT"}`);
}
say(`gate: ${gate ? "EQUAL" : "DIFFERENT"}`);
say();

say("the diagonal: cycle n in n dimensions, MM[n ×n]");
for (let n = 2; n <= 13; n++) {
  const h = hyperLimit(String(n), n, ascending(n));
  say(`  ${String(n).padStart(2)}^${String(n).padEnd(2)} whole ${h.whole}  f ${h.fits}  corner ${h.corner}^${n} = ${h.missing}  limit ${h.limit} = ${factorText(factor(h.limit, lut))}`);
}
say();

say("every ring 2…13 in every dimension 1…13: the limit, factored");
for (let n = 2; n <= 13; n++) {
  for (let d = 1; d <= 13; d++) {
    const h = hyperLimit(String(n), d, ascending(n));
    say(`  MM[${n} ×${d}]  limit ${h.limit} = ${factorText(factor(h.limit, lut))}`);
  }
}
say();
say("machine side: BigInt products and accumulation; decimal strings as decade readings; the LUT is a multiplication table of names; nothing from the imaginary side and no relation is kept");

const body = lines.join("\n") + "\n";
const receipt = body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;
writeFileSync(new URL("../receipts/hyper-13.txt", import.meta.url), receipt);
const diagonal = lines.slice(lines.indexOf("the diagonal: cycle n in n dimensions, MM[n ×n]"), lines.indexOf("every ring 2…13 in every dimension 1…13: the limit, factored"));
process.stdout.write([...lines.slice(0, lines.indexOf("gate: " + (gate ? "EQUAL" : "DIFFERENT")) + 1), "", ...diagonal, `(full table in receipts/hyper-13.txt) · ${((Date.now() - t0) / 1000).toFixed(1)}s`, receipt.slice(-72)].join("\n"));
