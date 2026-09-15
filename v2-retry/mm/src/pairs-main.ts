import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { DB_PATH } from "./db/store.ts";

/**
 * The pairs across the seam of 1 in the 2D cells: each quotient on the hulled side (2⁻¹, the top holds the
 * bottom) with its inversion on the relational side, and fractional(i) / fractional(real) for each pair.
 * Machine side, flagged: run names are seat tallies; the quotients are taken arithmetically as a check;
 * the L is the seats of MM[p,p] not held in MM[q,q], tallied.
 */

const db = new DatabaseSync(DB_PATH, { readOnly: true });
const runs = db.prepare("SELECT id, walk, container FROM runs WHERE dimension = 2 AND id IN (SELECT MAX(id) FROM runs WHERE dimension = 2 GROUP BY walk) ORDER BY walk").all() as { id: number; walk: string; container: string }[];

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
const reduce = (a: number, b: number): [number, number] => [a / gcd(a, b), b / gcd(a, b)];
const lOf = (p: number, q: number) => {
  let t = 0;
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) if (!(i < q && j < q)) t++;
  return t;
};

const lines: string[] = [];
const say = (s = "") => lines.push(s);
say("format=v2.mm.pairs.v2  fractional(i) / fractional(real) — each hulled quotient p/q over its inversion q/p; and i×real — the denominator of the bottom crossed with the denominator of the top, over the numerators crossed: pq/qp, a whole (n/n is n), on the imaginary integer side");
say("the bit first: 2 (turn) / 2⁻¹ (relation) → 4 over 1 · L of MM[2,2] outside MM[1,1] = " + lOf(2, 1) + " (O1)");

for (const run of runs) {
  const names = [...new Set((db.prepare("SELECT seats FROM nodes WHERE run_id = ?").all(run.id) as { seats: string }[]).map((n) => (JSON.parse(n.seats) as unknown[]).length))].sort((a, b) => a - b);
  const cells = new Set<string>();
  for (const a of names) for (const b of names) if (a !== b) cells.add(reduce(a, b).join("/"));
  const hulled = [...cells].map((c) => c.split("/").map(Number) as [number, number]).filter(([p, q]) => q !== 1 && p > q).sort((x, y) => x[1] - y[1] || x[0] - y[0]);

  say();
  say(`${run.walk}  MM[${run.container}]  found ${names.join(" ")}  · ${hulled.length} pairs across the seam`);
  say("  i       real    i/real   as squares   a cell here?   L of MM[p,p] outside MM[q,q]   L found?   p,q neighbours?   i×real (top·bottom)   imaginary whole   found?");
  for (const [p, q] of hulled) {
    const [a, b] = reduce(p * p, q * q);
    const here = cells.has(`${a}/${b}`);
    const l = lOf(p, q);
    const whole = p * q; // pq/qp is a whole, n/n is n
    say(`  ${`${p}/${q}`.padEnd(7)} ${`${q}/${p}`.padEnd(7)} ${`${a}/${b}`.padEnd(8)} ${`${p}²/${q}²`.padEnd(12)} ${(here ? "yes" : "—").padEnd(14)} ${String(l).padEnd(30)} ${(names.includes(l) ? "yes" : "—").padEnd(10)} ${(Math.abs(p - q) === 1 ? "yes" : "—").padEnd(17)} ${`${p}·${q}/${q}·${p}`.padEnd(21)} ${`${whole} (${q}×${p})`.padEnd(17)} ${names.includes(whole) ? "yes" : "—"}`);
  }
}

say();
say("machine side: names are seat tallies; the quotient, the squaring and p > q are arithmetic used as the check, not the construction");
const body = lines.join("\n") + "\n";
const receipt = body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;
writeFileSync(new URL("../receipts/pairs-2d.txt", import.meta.url), receipt);
process.stdout.write(receipt);
