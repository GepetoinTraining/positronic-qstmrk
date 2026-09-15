import { bead, hasSecond, pairsInto } from "./bead.ts";
import type { Bead } from "./bead.ts";
import type { Tick } from "./budget.ts";
import { noTick } from "./budget.ts";
import type { Cycle } from "./cycle.ts";
import { decadeName } from "./decade.ts";
import type { MM } from "./mm.ts";
import { seatsOf } from "./mm.ts";
import type { Region } from "./region.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";

/**
 * How an entry is substituted:
 *   matrix  a matrix of two or more coordinates held in a walk has its run (a crossing on the integer side);
 *   pair    the crossing of a pair across the seam: two things met in one walk, the top's run holding the
 *           bottom's, read i×real (the denominators crossed over the numerators crossed, pq/qp, a whole).
 *           It stands on the imaginary integer side unless a matrix of that dimension also has it.
 */
export type Substitute = { label: string; factors: string[]; dimension: number; walk: string; kind: "matrix" | "pair" };

export type LedgerEntry = {
  slot: string; // O<n>, its OpusNumber slot
  ring: Ring;
  foundIn: string;
  status: "prime" | "substituted";
  name: string; // a prime's name, or the first substitute's label
  substitutes: Substitute[];
};

/** The bottom run has a counterpart seat in the top run for each of its seats. */
const fitsIn = pairsInto;

type Met = { region: Region; name: () => string };
type Crossing = { label: () => string; factors: () => string[]; c: Cycle; kind: "matrix" | "pair" };

/**
 * The run taken off in parts that each pair with `part`: one bead per part, or null when the run does not end on a
 * whole part. A crossing of `part` with the run of parts has the run's pairing — found without building the crossing.
 */
function parts(run: readonly unknown[], part: readonly unknown[]): Bead[] | null {
  const out: Bead[] = [];
  let p = part[Symbol.iterator]();
  if (p.next().done) return null;
  p = part[Symbol.iterator]();
  p.next();
  let inPart = false;
  for (const _ of run) {
    inPart = true;
    if (p.next().done) {
      out.push(bead);
      p = part[Symbol.iterator]();
      p.next();
      inPart = false;
    }
  }
  return inPart ? null : out;
}

/**
 * Naming the primes. Every Opus ring found so far is looked for among the crossings of every walk:
 *   the matrices of two or more coordinates it held, and the pairs across the seam of the things it met;
 *   none has its run   → it stands alone: a prime, named by the reading of its seats;
 *   some crossing has it → it is substituted by those crossings, written with the primes' names.
 * A prime's name is its run read in decades, by pairing (decade.ts). Crossings are never built: an entry's run is
 * taken off in parts of each top's run, and the run of parts is looked for among the things met (machine side: runs
 * are looked up by their decade reading, itself made by pairing).
 */
export function nameLedger(walks: readonly Cycle[], tick: Tick = noTick): LedgerEntry[] {
  const entries: LedgerEntry[] = [];
  for (const c of walks) for (const a of c.arrivals) entries.push({ slot: a.ring.label, ring: a.ring, foundIn: `${c.dimension}D cycle ${c.cycle}`, status: "prime", name: "", substitutes: [] });
  const entryOf = new Map<Ring, LedgerEntry>();
  for (const e of entries) if (!entryOf.has(e.ring)) entryOf.set(e.ring, e);

  const nameOf = (r: Ring) => (r === bitRing ? "2" : entryOf.get(r)?.name || r.label);
  const written = (m: MM) => (m.coords.every((r) => r === bitRing) ? decadeName(seatsOf(m)) : m.coords.length === 1 ? nameOf(m.coords[0]) : `[${m.coords.map(nameOf).join(",")}]`);

  // each walk, read once: its matrices by reading (in held order), and the things it met with their readings
  const walked = walks.map((c) => {
    const matrices = new Map<string, Crossing[]>();
    for (const h of c.held) {
      const m = h.thing.m;
      if (m.coords.length < 2) continue;
      const k = decadeName(h.thing.region);
      const list = matrices.get(k) ?? [];
      list.push({ label: () => written(m), factors: () => m.coords.map(nameOf), c, kind: "matrix" });
      matrices.set(k, list);
    }
    const met: Met[] = [
      ...c.held.filter((h) => h.kind !== "floor").map((h) => ({ region: h.thing.region, name: () => written(h.thing.m) })),
      ...[...c.remet, ...c.arrivals].map((x) => ({ region: x.thing.region, name: () => nameOf(x.ring) })),
    ];
    const readings = met.map((x) => decadeName(x.region));
    const metByReading = new Map<string, number[]>();
    readings.forEach((k, i) => metByReading.set(k, [...(metByReading.get(k) ?? []), i]));
    return { c, matrices, met, readings, metByReading };
  });

  const partsCache = new Map<string, Bead[] | null>();
  const raw = new Map<LedgerEntry, Crossing[]>();
  for (const e of entries) {
    tick("naming: entries", () => `entry ${e.slot}`);
    const er = decadeName(e.ring.seats);
    const subs: Crossing[] = [];
    for (const w of walked) {
      subs.push(...(w.matrices.get(er) ?? []));
      w.met.forEach((top, ti) => {
        tick("naming: crossings", () => `entry ${e.slot}, ${w.c.dimension}D cycle ${w.c.cycle}`);
        if (!hasSecond(top.region)) return;
        const key = `${er}|${w.readings[ti]}`;
        let chunks = partsCache.get(key);
        if (chunks === undefined) {
          chunks = parts(e.ring.seats, top.region);
          partsCache.set(key, chunks);
        }
        if (!chunks || !hasSecond(chunks)) return;
        for (const bi of w.metByReading.get(decadeName(chunks)) ?? []) {
          const bottom = w.met[bi];
          if (bottom === top || !fitsIn(bottom.region, top.region)) continue;
          subs.push({ label: () => `[${top.name()},${bottom.name()}]`, factors: () => [top.name(), bottom.name()], c: w.c, kind: "pair" });
        }
      });
    }
    if (subs.length) {
      e.status = "substituted";
      raw.set(e, subs);
    } else {
      e.name = decadeName(e.ring.seats);
    }
  }

  for (const [e, subs] of raw) {
    for (const x of [...subs.filter((s) => s.kind === "matrix"), ...subs.filter((s) => s.kind === "pair")]) {
      const s: Substitute = { label: x.label(), factors: x.factors(), dimension: x.c.dimension, walk: `${x.c.dimension}D cycle ${x.c.cycle}`, kind: x.kind };
      if (s.factors.includes(e.slot)) continue; // an entry does not cross itself
      if (!e.substitutes.some((t) => t.label === s.label && t.dimension === s.dimension)) e.substitutes.push(s);
    }
    e.name = e.substitutes[0]?.label ?? e.slot;
  }
  return entries;
}

/**
 * The rail with the primes named and the substituted entries taken off it. Substituted runs stay known,
 * so meeting one again is a re-meet of its substitute, not a new arrival.
 */
export function namedRail(rail: readonly Ring[], ledger: readonly LedgerEntry[]): { rail: Ring[]; known: Ring[] } {
  const out: Ring[] = [];
  for (const r of rail) {
    const e = ledger.find((x) => x.ring === r);
    if (!e) out.push(r);
    else if (e.status === "prime") out.push({ label: e.name, seats: r.seats });
  }
  const known = ledger.filter((e) => e.status === "substituted").map((e) => ({ label: e.name, seats: e.ring.seats }));
  return { rail: out, known };
}
