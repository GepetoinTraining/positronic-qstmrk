import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import type { Tick } from "./budget.ts";
import { WalkBudgetExceeded, deadlineTick } from "./budget.ts";
import type { Cycle } from "./cycle.ts";
import { walkCycle } from "./cycle.ts";
import { decadeName } from "./decade.ts";
import { openStore, writeCycle, writeLedger } from "./db/store.ts";
import type { LedgerEntry } from "./ledger.ts";
import { nameLedger, namedRail } from "./ledger.ts";
import { addressed, collapsed, resolution } from "./format.ts";
import { mmText } from "./mm.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";

/** The 1D reading of cycle 1, already sealed by the Rust oscillator (v3). */
const ONE_D = "2{Q*(4/O1, O1/4, O1/2, 2/O1, 1/O1), Pq(4/2, [2/4, 1/2], 1/4), Pn(2)}";

function receiptOf(c: Cycle): string {
  const seat = (s: readonly number[]) => `(${s.join(",")})`;
  const lines: string[] = [];
  const say = (s = "") => lines.push(s);
  say(`format=v2.mm.cycle.v4 dimension=${c.dimension} cycle=${c.cycle}  (order matters; every direction +)`);
  say(`container ${mmText(c.container)}`);
  say("held:");
  for (const { thing, kind } of c.held) say(`  ${thing.label.padEnd(9)} ${kind.padEnd(6)} ${resolution(thing).padEnd(20)} seats ${thing.region.map(seat).join(" ")}`);
  const mets = (title: string, list: typeof c.arrivals) => {
    say(`${title}:`);
    if (list.length === 0) say("  none");
    for (const a of list) say(`  ${a.thing.label.padEnd(4)} ${resolution(a.thing).padEnd(12)} seats ${a.thing.region.map(seat).join(" ")}${a.also.length ? `   also met at ${a.also.length} more region(s)` : ""}`);
  };
  mets("arrivals (new rings)", c.arrivals);
  mets("re-met (rings already on the rail, met here)", c.remet);
  say("collapses (a rail ring this container's matrix is the resolution of):");
  if (c.collapses.length === 0) say("  none");
  for (const x of c.collapses) say(`  ${x.ring.label} ↔ ${x.onto.label} ${resolution(x.onto)}`);
  say();
  const pn = decadeName(c.railAfter); // the rail read in decades, by pairing
  const head = c.container.coords[0].label;
  say(`every address:   ${head}{Q*(${addressed(c.qstar)}), Pq(${addressed(c.pq)}), Pn(${pn})}`);
  if (c.rstar.length) say(`re-met relations: R*(${addressed(c.rstar)})`);
  const down = `${head}{Q*(${collapsed(c.qstar)}), Pq(${collapsed(c.pq)}), Pn(${pn})}`;
  say(`read one lower:  ${down}`);
  if (c.dimension === 2 && c.cycle === 1) say(`gate (cycle 1 read one lower is the sealed 1D reading): ${down === ONE_D ? "EQUAL" : "DIFFERENT"}`);
  say(c.limit ? `limit of the container: ${c.limit.top.label}/${c.limit.bottom.label} = ${resolution(c.limit.top)} over ${resolution(c.limit.bottom)}` : "limit: the cover is held or a matrix");
  say(`rail after: ${c.railAfter.map((r: Ring) => r.label).join(" ")}`);
  return lines.join("\n") + "\n";
}

const db = openStore();
mkdirSync(new URL("../receipts/", import.meta.url), { recursive: true });

const seal = (body: string) => body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;

function run(c: Cycle) {
  const body = receiptOf(c);
  const receipt = seal(body);
  writeFileSync(new URL(`../receipts/cycle${c.cycle}-${c.dimension}d.txt`, import.meta.url), receipt);
  const id = writeCycle(db, c, receipt.slice(-65, -1), receipt);
  process.stdout.write(receipt + `stored as run ${id}\n\n`);
  return c;
}

/** Name the ledger over every walk so far, write its receipt, and return the named rail and the known substitutes. */
function naming(n: number, walks: readonly Cycle[], rail: readonly Ring[], tick?: Tick): { ledger: LedgerEntry[]; rail: Ring[]; known: Ring[] } {
  const ledger = nameLedger(walks, tick);
  const lines = [
    "format=v2.mm.naming.v2  (a ring no crossing has is a prime; one a crossing has is substituted — matrix: a matrix of two or more coordinates held in a walk; pair: the i×real crossing of two things met in one walk, the top's run holding the bottom's)",
    ...ledger.map((e) =>
      e.status === "prime"
        ? `${e.slot.padEnd(4)} → ${e.name.padEnd(6)} prime        found in ${e.foundIn}`
        : `${e.slot.padEnd(4)} → ${e.substitutes.map((s) => `${s.label} (${s.kind}, ${s.walk})`).join(", ")}   substituted   found in ${e.foundIn}`,
    ),
  ];
  const named = namedRail(rail, ledger);
  lines.push(`named rail: ${named.rail.map((r) => r.label).join(" ")}   known substitutes: ${named.known.map((r) => r.label).join(" ")}`);
  const receipt = seal(lines.join("\n") + "\n");
  writeFileSync(new URL(`../receipts/naming-${n}.txt`, import.meta.url), receipt);
  writeLedger(db, ledger);
  process.stdout.write(receipt + "\n");
  return { ledger, ...named };
}

// 2D: two cycles; 3D: the first cycle, the bit on three coordinates, over the rail 2D left
const walks: Cycle[] = [];
let rail: Ring[] = [bitRing];
for (let k = 0; k < 2; k++) rail = run(walks[walks.push(walkCycle(rail, k, 2)) - 1]).railAfter;
rail = run(walks[walks.push(walkCycle(rail, 0, 3)) - 1]).railAfter;

// naming 1, then 2D cycle 3 over the named rail
let named = naming(1, walks, rail);
rail = run(walks[walks.push(walkCycle(named.rail, 2, 2, { opus: named.ledger, known: named.known })) - 1]).railAfter;

// naming 2 resolves cycle 3's arrivals (the pairs across the seam); then 3D cycle 2 over the named rail
named = naming(2, walks, rail);
rail = run(walks[walks.push(walkCycle(named.rail, 1, 3, { opus: named.ledger, known: named.known })) - 1]).railAfter;

// naming 3, then 2D cycle 4
named = naming(3, walks, rail);
rail = run(walks[walks.push(walkCycle(named.rail, 3, 2, { opus: named.ledger, known: named.known })) - 1]).railAfter;

named = naming(4, walks, rail);

// 2D cycles 5 … 13, each named after it. A cycle or a naming past the machine's budget fails gracefully: it writes a
// sealed failure receipt with the stage it reached, nothing is stored as if complete, and the walk stops there.
const BUDGET_S = Number(process.env.MM_BUDGET_S ?? "900");
const LAST = Number(process.env.MM_LAST ?? "13");

function failed(cycle: number, what: string, reason: string) {
  const body = [
    `format=v2.mm.cycle.failed dimension=2 cycle=${cycle}`,
    `${what} stopped gracefully: ${reason}`,
    `budget: ${BUDGET_S}s of wall clock per cycle and per naming (machine side)`,
    `nothing from this ${what} is stored; the rail and ledger stay as naming ${cycle - 1} left them`,
  ].join("\n");
  const receipt = seal(body + "\n");
  writeFileSync(new URL(`../receipts/cycle${cycle}-2d.failed.txt`, import.meta.url), receipt);
  process.stdout.write(receipt + "\n");
}

for (let cycle = 5; cycle <= LAST; cycle++) {
  const k = cycle - 1;
  if (!named.rail[k]) {
    failed(cycle, "walk", "the named rail has no ring at this place");
    break;
  }
  let c: Cycle;
  try {
    c = walkCycle(named.rail, k, 2, { opus: named.ledger, known: named.known, tick: deadlineTick(Date.now() + BUDGET_S * 1000) });
  } catch (e) {
    if (!(e instanceof WalkBudgetExceeded)) throw e;
    failed(cycle, "walk", e.message);
    break;
  }
  walks.push(c);
  try {
    named = naming(cycle, walks, c.railAfter, deadlineTick(Date.now() + BUDGET_S * 1000));
  } catch (e) {
    if (!(e instanceof WalkBudgetExceeded)) throw e;
    walks.pop();
    failed(cycle, "naming", e.message);
    break;
  }
  rail = run(c).railAfter;
}
db.close();
