import { mkdirSync, writeFileSync } from "node:fs";
import type { Tick } from "./budget.ts";
import { WalkBudgetExceeded, deadlineTick } from "./budget.ts";
import { walkCycle } from "./cycle.ts";
import { seal } from "./receipt.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";
import { sidesCycle } from "./sides.ts";
import type { Reading, Turn } from "./turns.ts";
import { readingOf, turnOf, turnText } from "./turns.ts";

/**
 * The frontier test (test mode only: names kept). By cycle k, which integers ≤ hull are NOT met as names — seat
 * tallies of held ∪ arrivals ∪ re-met, cumulative over cycles 1…k — for each turn, and for the union of the turns?
 * The hull is the largest name met. Holes are printed primes | composites. For the holes of the first turn given
 * (the identity, unless the caller says otherwise): which other turn fills each (`filled by`); whether every
 * missing composite has all its prime factors present; whether the hole is reachable at the cell level
 * (reduce(a, b) over names lands on it).
 *
 * Machine side, flagged throughout: names are tallies; primes, factors and the reduction are arithmetic used as the
 * reading of the measurement, not the construction. Nothing here adds a name; the frontier is the measurement.
 *
 *   node --experimental-strip-types src/frontier.ts [D=2] [N=8] [budget_s=900] [--turns=1,-1]      turns in parallel, a rail each
 *   node --experimental-strip-types src/frontier.ts 3 3 900 --turns=1,i:j:k,j:k:i               a track i:j:k is one rail, turns in series
 *   receipt: receipts/frontier-{D}d.txt, sealed; frontier-{D}d-phased.txt when any track has more than one turn
 */

const args = process.argv.slice(2);
const flags = args.filter((a) => a.startsWith("--"));
const pos = args.filter((a) => !a.startsWith("--"));
const D = Number(pos[0] ?? "2");
const N = Number(pos[1] ?? "8");
const BUDGET_S = Number(pos[2] ?? "900");
/** `--engine=sides`: the side-run walk (sides.ts), names only, no seat lattice; gated to reseal this script's receipts */
const SIDES = flags.includes("--engine=sides");
/**
 * A track: one rail, walked through a sequence of turns that cycles — `i:j:k` walks cycle 1 under i, cycle 2 under
 * j, cycle 3 under k, cycle 4 under i again (the turns in series, phased). A single label is a track of one turn.
 */
type Track = { readonly label: string; readonly turns: readonly (Turn | Reading)[] };
const isReading = (x: Turn | Reading): x is Reading => "kind" in x;
const labels = (flags.find((f) => f.startsWith("--turns="))?.slice("--turns=".length) ?? (D === 2 ? "1,-1" : "1,c,c2")).split(",");
const turns: Track[] = labels.map((l) => ({ label: l, turns: l.split(":").map((u) => readingOf(u) ?? turnOf(u, D)) }));
const one = turns[0];
const readingText = (r: Reading) => `${r.label}: ${r.kind === "ring" ? `offset by the run of ring ${r.r}, wrapping` : `offset by one of ${r.r} equal runs the axis is dealt into; breaks when unequal`}`;

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
const isPrime = (n: number) => {
  if (n < 2) return false;
  for (let d = 2; d * d <= n; d++) if (n % d === 0) return false;
  return true;
};
const primeFactors = (n: number) => {
  const out: number[] = [];
  for (let d = 2; d * d <= n; d++)
    while (n % d === 0) {
      out.push(d);
      n /= d;
    }
  if (n > 1) out.push(n);
  return [...new Set(out)];
};
const holesOf = (names: Set<number>) => {
  const max = Math.max(0, ...names);
  const holes: number[] = [];
  for (let n = 1; n <= max; n++) if (!names.has(n)) holes.push(n);
  return { max, holes };
};
const holesText = (holes: number[]) => {
  const p = holes.filter(isPrime);
  const c = holes.filter((h) => !isPrime(h));
  return `${p.length ? p.join(" ") : "(no primes)"} | ${c.length ? c.join(" ") : "(no composites)"}`;
};
/**
 * a hole reached at the cell level: some pair of names reduces to it on either side — a/gcd(a,b) = h means a = h·m
 * and b = m·b' with b' prime to h, so walk the multiples of h and, for each, the multiples of m (same predicate as
 * the pairwise scan, without the square in the names)
 */
const cellReachable = (h: number, names: Set<number>) => {
  const max = Math.max(0, ...names);
  for (let a = h; a <= max; a += h) {
    if (!names.has(a)) continue;
    const m = a / h;
    for (let b = m; b <= max; b += m) if (b !== a && names.has(b) && gcd(h, b / m) === 1) return true;
  }
  return false;
};

const lines: string[] = [];
const say = (s = "") => {
  lines.push(s);
  process.stdout.write(s + "\n");
};

say(`format=v2.mm.frontier.v1 dimension=${D} cycles=${N} turns=${labels.join(",")}  — by cycle k, the integers ≤ hull not met as names (held ∪ arrivals ∪ re-met, cumulative), per turn and for the union; holes as primes | composites; budget ${BUDGET_S}s per walk`);
const entryText = (x: Turn | Reading) => (isReading(x) ? readingText(x) : turnText(x));
for (const t of turns) say(t.turns.length === 1 ? `${isReading(t.turns[0]) ? "reading" : "turn"} ${entryText(t.turns[0])}` : `track ${t.label}: ${t.turns.map(entryText).join("; then ")} — in series on one rail, cycling`);

const rails = new Map<Track, Ring[]>(turns.map((t) => [t, [bitRing]]));
const met = new Map<Track, Set<number>>(turns.map((t) => [t, new Set<number>()]));
const t0 = Date.now();
for (let k = 0; k < N && rails.size > 0; k++) {
  say();
  for (const t of [...rails.keys()]) {
    const rail = rails.get(t)!;
    if (!rail[k]) {
      say(`cycle ${k + 1} [${t.label}]: stopped, the rail has no ring at place ${k}`);
      rails.delete(t);
      continue;
    }
    const tick: Tick = deadlineTick(Date.now() + BUDGET_S * 1000);
    const started = Date.now();
    try {
      const entry = t.turns[k % t.turns.length];
      const names = met.get(t)!;
      let railAfter: Ring[];
      if (SIDES) {
        if (isReading(entry)) throw new Error("refused: the side-run walk does not carry offset readings");
        const c = sidesCycle(rail, k, D, { tick, turn: entry });
        for (const x of c.names) names.add(x);
        railAfter = c.railAfter;
      } else {
        const c = walkCycle(rail, k, D, { tick, ...(isReading(entry) ? { reading: entry } : { turn: entry }) });
        if (c.broke) say(`cycle ${k + 1} [${t.label}]: broke — ${c.broke}`);
        for (const h of c.held) names.add(h.thing.region.length);
        for (const a of c.arrivals) names.add(a.thing.region.length);
        for (const r of c.remet) names.add(r.thing.region.length);
        railAfter = c.railAfter;
      }
      const c = { railAfter };
      rails.set(t, c.railAfter);
      const { max, holes } = holesOf(names);
      say(`cycle ${k + 1} [${t.label}]: max ${String(max).padStart(4)}  holes: ${holesText(holes)}   (${holes.length} holes, ${names.size} names, rail ${c.railAfter.length} rings)`);
      process.stderr.write(`  ${((Date.now() - started) / 1000).toFixed(1)}s\n`); // timing stays out of the sealed body
    } catch (e) {
      if (!(e instanceof WalkBudgetExceeded)) throw e;
      say(`cycle ${k + 1} [${t.label}]: stopped gracefully, ${e.message}`);
      rails.delete(t);
    }
  }
  if (rails.size === 0) break;
  const active = [...rails.keys()];
  const union = new Set<number>();
  for (const t of active) for (const n of met.get(t)!) union.add(n);
  const u = holesOf(union);
  say(`cycle ${k + 1} [union of ${active.map((t) => t.label).join(",")}]: max ${String(u.max).padStart(4)}  holes: ${holesText(u.holes)}   (${u.holes.length} holes, ${union.size} names)`);
  if (rails.has(one)) {
    const mine = met.get(one)!;
    const h1 = holesOf(mine);
    for (const t of active) {
      if (t === one) continue;
      const other = met.get(t)!;
      const same = other.size === mine.size && [...other].every((n) => mine.has(n));
      const filled = h1.holes.filter((h) => other.has(h));
      say(`  filled by ${t.label}: ${same ? `none — the same names as ${one.label}` : filled.length ? filled.join(" ") : "none"}`);
    }
    const composites = h1.holes.filter((h) => !isPrime(h));
    const bad = composites.filter((h) => !primeFactors(h).every((p) => mine.has(p)));
    say(`  [${one.label}] every missing composite has all its prime factors present: ${bad.length ? `NO — ${bad.join(" ")}` : `yes (${composites.length} composites)`}`);
    const unreachable = h1.holes.filter((h) => !cellReachable(h, mine));
    say(`  [${one.label}] holes unreachable at the cell level (reduce(a,b) over names): ${unreachable.length ? unreachable.join(" ") : "none"}`);
  }
}
say();
say("machine side: names are seat tallies; primes, factors and the reduction are arithmetic used as the reading of the measurement, not the construction");
process.stderr.write(`${((Date.now() - t0) / 1000).toFixed(0)}s in all\n`);

mkdirSync(new URL("../receipts/", import.meta.url), { recursive: true });
const receipt = seal(lines.join("\n") + "\n");
const suffix = turns.some((t) => t.turns.some(isReading)) ? "-readings" : turns.some((t) => t.turns.length > 1) ? "-phased" : "";
writeFileSync(new URL(`../receipts/frontier-${D}d${suffix}.txt`, import.meta.url), receipt);
process.stdout.write(receipt.slice(receipt.lastIndexOf("seal sha256")));
