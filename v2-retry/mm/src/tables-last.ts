import { createHash } from "node:crypto";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import type { Tick } from "./budget.ts";
import { WalkBudgetExceeded, deadlineTick } from "./budget.ts";
import { walkCycle } from "./cycle.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";
import type { Turn } from "./turns.ts";
import { turnOf, turnText } from "./turns.ts";

/**
 * The last number/number. Cycle to N and keep nothing but the last pair across the seam of each cycle:
 * the largest hulled p/q (p > q, q ≠ 1, reduced) with its inversion q/p, and i×real = pq, the whole on the
 * imaginary integer side. Every cycle is dropped once its last pair is read; only the rail and a running seal
 * are carried. Constant memory across cycles; time is the walk's.
 *
 * With turns (turns.ts), the same cycles are walked once per turn, each turn on its own rail, and one union line
 * per cycle reads the last pair over every turn's names of that cycle. Nothing is stored past the current cycle
 * except the rail per turn and the running seal. With the single turn 1 the output is the v1 format, byte for byte.
 *
 * Machine side, flagged: names are seat tallies; the reduction and p > q are arithmetic used as the check, not the
 * construction (same shadow as pairs-main.ts). No naming, no ledger, no store: the rail grows by arrivals only.
 *
 *   node --experimental-strip-types src/tables-last.ts [N=6] [D=2] [budget_s=900] [--turns=1,c,c2]
 *   receipt: receipts/last-turns-{D}d.txt, sealed
 */

const args = process.argv.slice(2);
const flags = args.filter((a) => a.startsWith("--"));
const pos = args.filter((a) => !a.startsWith("--"));
const N = Number(pos[0] ?? "6");
const D = Number(pos[1] ?? "2");
const BUDGET_S = Number(pos[2] ?? "900");
const labels = (flags.find((f) => f.startsWith("--turns="))?.slice("--turns=".length) ?? "1").split(",");
const turns: Turn[] = labels.map((l) => turnOf(l, D));
const single = turns.length === 1 && labels[0] === "1";

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

/** the last pair across the seam of a cycle: largest q, then largest p, over the names met */
function lastPair(names: Iterable<number>): [number, number] | null {
  let best: [number, number] | null = null;
  const ns = [...new Set(names)];
  for (const a of ns)
    for (const b of ns) {
      if (a <= b) continue; // p > q
      const g = gcd(a, b);
      const p = a / g;
      const q = b / g;
      if (q === 1) continue;
      if (!best || q > best[1] || (q === best[1] && p > best[0])) best = [p, q];
    }
  return best;
}

const receipt = new URL(`../receipts/last-turns-${D}d.txt`, import.meta.url);
mkdirSync(new URL("../receipts/", import.meta.url), { recursive: true });
writeFileSync(receipt, "");
const seal = createHash("sha256");
const say = (s: string) => {
  seal.update(s + "\n");
  appendFileSync(receipt, s + "\n");
  process.stdout.write(s + "\n");
};

if (single) say(`format=v2.mm.last.v1 dimension=${D} cycles=${N}  — the last pair across the seam per cycle: p/q · q/p → pq (i×real, a whole); cycles are dropped, only the rail is carried`);
else {
  say(`format=v2.mm.last.v2 dimension=${D} cycles=${N} turns=${labels.join(",")}  — the last pair across the seam per cycle and per turn: p/q · q/p → pq (i×real, a whole); then the union over the turns; cycles are dropped, only the rail per turn is carried`);
  for (const t of turns) say(`turn ${turnText(t)}`);
}

const pairText = (pair: [number, number] | null) => (pair ? `${pair[0]}/${pair[1]} · ${pair[1]}/${pair[0]} → ${pair[0] * pair[1]}` : "no pair across the seam");
const tag = (t: Turn) => (single ? "" : ` [${t.label}]`);

const rails = new Map<Turn, Ring[]>(turns.map((t) => [t, [bitRing]]));
for (let k = 0; k < N && rails.size > 0; k++) {
  const across = new Set<number>(); // the names of this cycle over every turn, dropped with the cycle
  for (const t of [...rails.keys()]) {
    const rail = rails.get(t)!;
    if (!rail[k]) {
      say(`cycle ${k + 1}${tag(t)}: stopped, the rail has no ring at place ${k}`);
      rails.delete(t);
      continue;
    }
    const tick: Tick = deadlineTick(Date.now() + BUDGET_S * 1000);
    try {
      const c = walkCycle(rail, k, D, { tick, turn: t });
      const names = function* () {
        for (const h of c.held) yield h.thing.region.length;
        for (const a of c.arrivals) yield a.thing.region.length;
        for (const r of c.remet) yield r.thing.region.length;
      };
      const mine = new Set(names());
      for (const n of mine) across.add(n);
      say(`cycle ${k + 1}${tag(t)}: ${pairText(lastPair(mine))}   (rail ${c.railAfter.length} rings)`);
      rails.set(t, c.railAfter);
      // c goes out of scope here: nothing of the cycle is kept
    } catch (e) {
      if (!(e instanceof WalkBudgetExceeded)) throw e;
      say(`cycle ${k + 1}${tag(t)}: stopped gracefully, ${e.message}`);
      rails.delete(t);
    }
  }
  if (!single && rails.size > 0) say(`cycle ${k + 1} [union]: ${pairText(lastPair(across))}   (over ${rails.size} turns)`);
}

const line = `seal sha256 ${seal.digest("hex")}\n`;
appendFileSync(receipt, line);
process.stdout.write(line);
