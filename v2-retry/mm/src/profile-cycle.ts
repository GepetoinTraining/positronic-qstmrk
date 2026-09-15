import { WalkBudgetExceeded, deadlineTick } from "./budget.ts";
import type { Cycle } from "./cycle.ts";
import { walkCycle } from "./cycle.ts";
import type { LedgerEntry } from "./ledger.ts";
import { nameLedger, namedRail } from "./ledger.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";

/**
 * Machine side: rebuild the walks exactly as main.ts does up to naming 4, then walk 2D cycles 5 … `last`, naming
 * after each, timing every stage under a per-cycle budget. Nothing is stored. Usage: profile-cycle.ts <seconds> <last>
 */
const seconds = Number(process.argv[2] ?? "60");
const last = Number(process.argv[3] ?? "13");
const secs = (t: number) => `${((Date.now() - t) / 1000).toFixed(1)}s`;

const walks: Cycle[] = [];
let rail: Ring[] = [bitRing];
let named: { ledger: LedgerEntry[]; rail: Ring[]; known: Ring[] } = { ledger: [], rail, known: [] };
const rename = () => {
  const t = Date.now();
  const ledger = nameLedger(walks);
  named = { ledger, ...namedRail(rail, ledger) };
  return secs(t);
};
const t0 = Date.now();
for (let k = 0; k < 2; k++) rail = walks[walks.push(walkCycle(rail, k, 2)) - 1].railAfter;
rail = walks[walks.push(walkCycle(rail, 0, 3)) - 1].railAfter;
rename();
rail = walks[walks.push(walkCycle(named.rail, 2, 2, { opus: named.ledger, known: named.known })) - 1].railAfter;
rename();
rail = walks[walks.push(walkCycle(named.rail, 1, 3, { opus: named.ledger, known: named.known })) - 1].railAfter;
rename();
rail = walks[walks.push(walkCycle(named.rail, 3, 2, { opus: named.ledger, known: named.known })) - 1].railAfter;
console.log(`rebuilt through 2D cycle 4 in ${secs(t0)}; naming 4 in ${rename()}`);

for (let cycle = 5; cycle <= last; cycle++) {
  const k = cycle - 1;
  if (!named.rail[k]) {
    console.log(`2D cycle ${cycle}: the named rail has no ring at place ${cycle}; stopping`);
    break;
  }
  const start = Date.now();
  const stages: string[] = [];
  let stageStart = start;
  let current = "";
  const onStage = (s: string) => {
    if (current) stages.push(`${current} ${((Date.now() - stageStart) / 1000).toFixed(1)}s`);
    current = s;
    stageStart = Date.now();
  };
  try {
    const c = walkCycle(named.rail, k, 2, { opus: named.ledger, known: named.known, tick: deadlineTick(start + seconds * 1000, onStage) });
    onStage("done");
    walks.push(c);
    rail = c.railAfter;
    const walked = secs(start);
    const naming = rename();
    const primes = named.ledger.filter((e) => e.foundIn === `2D cycle ${cycle}` && e.status === "prime").map((e) => e.name);
    console.log(
      `2D cycle ${cycle} MM[${c.container.coords.map((r) => r.label).join(",")}]: ${c.held.length} held · ${c.arrivals.length} arrivals · ${c.remet.length} re-met · Pq ${c.pq.length} groups · Q* ${c.qstar.length} groups · walked ${walked} · naming ${naming}`,
    );
    console.log(`    stages: ${stages.join(" · ")}`);
    console.log(`    new primes: ${primes.join(" ")}`);
  } catch (e) {
    if (!(e instanceof WalkBudgetExceeded)) throw e;
    onStage("stopped");
    console.log(`2D cycle ${cycle} failed gracefully after ${seconds}s: ${e.message}`);
    console.log(`    stages: ${stages.join(" · ")}`);
    break;
  }
}
