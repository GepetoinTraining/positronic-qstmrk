import { bead, pairsInto, samePairing } from "./bead.ts";
import type { Tick } from "./budget.ts";
import { noTick } from "./budget.ts";
import { decadeName } from "./decade.ts";
import type { MM } from "./mm.ts";
import { mm } from "./mm.ts";
import { matrixLabel } from "./names.ts";
import type { Region } from "./region.ts";
import { holds, isMatrix, place, sameRegion, union } from "./region.ts";
import type { Relation, Thing } from "./relation.ts";
import { group, readings } from "./relation.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";
import type { Site } from "./site.ts";

export type Kind = "whole" | "cube" | "table" | "held" | "floor" | "question" | "remet" | "rail";

/** A ring met in this container: its first region, and every further region it was met at. */
export type Met = { ring: Ring; thing: Thing; also: Region[] };

export type Cycle = {
  dimension: number;
  cycle: number;
  container: MM;
  held: { thing: Thing; kind: Kind }[];
  unordered: [Thing, Thing][];
  arrivals: Met[];
  remet: Met[];
  collapses: { ring: Thing; onto: Thing }[];
  qstar: Relation[][];
  pq: Relation[][];
  rstar: Relation[][];
  limit: Relation | null;
  railAfter: Ring[];
};

export type WalkOptions = {
  /**
   * the Opus ledger so far: a run with one element per arrival already written. An arrival's slot is that run, the
   * arrivals met before it in this walk, and itself, read in decades by pairing. Defaults to the rail's rings other
   * than the bit (the arrivals so far).
   */
  opus?: readonly unknown[];
  /** runs the ledger already knows but that are not rings on the rail (substituted entries) */
  known?: readonly Ring[];
  /** machine side: called inside long loops so a walk past the machine's budget fails gracefully (budget.ts) */
  tick?: Tick;
};

const fitsIn = (r: Ring, container: Ring) => pairsInto(r.seats, container.seats);

/** Coordinate subsets of size s, increasing (every direction +). */
function subsets(n: number, s: number): number[][] {
  if (s === 0) return [[]];
  const out: number[][] = [];
  for (let first = 0; first < n; first++) for (const rest of subsets(n, s - 1)) if (rest.every((a) => a > first)) out.push([first, ...rest]);
  return out;
}

/** Ordered choices of s rings from the rail (order matters). */
function assignments(rings: readonly Ring[], s: number): Ring[][] {
  if (s === 0) return [[]];
  return rings.flatMap((r) => assignments(rings, s - 1).map((rest) => [r, ...rest]));
}

/**
 * One cycle in D dimensions (order matters; every direction +). The container is the k-th ring of the rail
 * taken on every coordinate.
 *   held       the whole; then every matrix of fitting rail rings on every subset of coordinates, largest
 *              first (in 3D: cubes, tables, balls); then the floor.
 *   unions     two held things where neither region holds the other: their union, when it is not a matrix
 *              and pairs with no held region, is either a ring already on the rail met again here (remet)
 *              or a new ring (arrival). A run met again adds an address.
 *   limit      the cover of everything held short of the whole, over the whole.
 *   collapses  a rail ring whose run pairs with a held matrix of this container that is not its own line:
 *              the lower-resolution thing that this container's matrix is the resolution of.
 */
export function walkCycle(rail: readonly Ring[], k: number, dimension: number, options: WalkOptions = {}): Cycle {
  const opus = options.opus ?? rail.filter((r) => r !== bitRing);
  const known = options.known ?? [];
  const tick = options.tick ?? noTick;
  const R = rail[k];
  const all = Array.from({ length: dimension }, (_, a) => a);
  const container = mm(...all.map(() => R));
  const fit = rail.filter((r) => fitsIn(r, R));
  const held: { thing: Thing; kind: Kind }[] = [];
  const add = (rings: Ring[], axes: number[], kind: Kind) => {
    const m = mm(...rings);
    const site: Site = all.map((a) => (axes.includes(a) ? [rings[axes.indexOf(a)]] : []));
    held.push({ thing: { label: matrixLabel(m, axes, dimension), m, site, region: place(container, m, axes) }, kind });
  };

  add(all.map(() => R), all, "whole");
  for (let s = dimension; s >= 1; s--)
    for (const rings of assignments(fit, s))
      for (const axes of subsets(dimension, s)) {
        tick("held", () => `${held.length} held so far`);
        if (s === dimension && rings.every((r) => r === R)) continue;
        add(rings, axes, s === dimension ? (dimension === 3 ? "cube" : "table") : s === 2 ? "table" : "held");
      }
  add([], [], "floor");
  const things = held.map((h) => h.thing);

  const extras: Met[] = []; // arrivals and remets share the slots after the container's coordinates
  const arrivals: Met[] = [];
  const remet: Met[] = [];
  // machine side: runs looked up by their decade reading (itself made by pairing), first met first
  const firstByReading = <T,>(list: readonly T[], run: (x: T) => readonly unknown[]) => {
    const m = new Map<string, T>();
    for (const x of list) {
      const k = decadeName(run(x));
      if (!m.has(k)) m.set(k, x);
    }
    return m;
  };
  const railByReading = firstByReading(rail, (r) => r.seats);
  const knownByReading = firstByReading(known, (r) => r.seats);
  const extrasByReading = new Map<string, Met>();
  const meet = (u: Region) => {
    const reading = decadeName(u);
    const already = extrasByReading.get(reading);
    if (already) {
      if (!sameRegion(already.thing.region, u) && !already.also.some((reg) => sameRegion(reg, u))) already.also.push(u);
      return already;
    }
    const onRail = railByReading.get(reading) ?? knownByReading.get(reading);
    const ring: Ring = onRail ?? { label: `O${decadeName([...opus, ...arrivals, bead])}`, seats: u.map(() => bead) };
    const site: Site = [...all.map(() => []), ...extras.map(() => []), [ring]];
    const met: Met = { ring, thing: { label: ring.label, m: mm(ring), site, region: u }, also: [] };
    extras.push(met);
    extrasByReading.set(reading, met);
    (onRail ? remet : arrivals).push(met);
    return met;
  };

  const heldReadings = new Set(things.map((h) => decadeName(h.region)));
  const unordered: [Thing, Thing][] = [];
  const pqRelations: Relation[] = [];
  things.forEach((a, i) =>
    things.slice(i + 1).forEach((b) => {
      tick("pairs", () => `thing ${i} of ${things.length}, ${extras.length} met`);
      const r = readings(a, b);
      pqRelations.push(...r.relations);
      if (!r.unordered) return;
      unordered.push([a, b]);
      const u = union([a.region, b.region]);
      if (isMatrix(container, u) || heldReadings.has(decadeName(u))) return;
      meet(u);
    }),
  );

  const whole = things[0];
  const cover = union(held.filter((h) => h.kind !== "whole").map((h) => h.thing.region));
  let limit: Relation | null = null;
  if (!sameRegion(cover, whole.region) && !isMatrix(container, cover) && !heldReadings.has(decadeName(cover))) {
    const met = extras.find((x) => sameRegion(x.thing.region, cover)) ?? meet(cover);
    const top = sameRegion(met.thing.region, cover) ? met.thing : { ...met.thing, region: cover };
    limit = { top, bottom: whole };
  }

  tick("limit");
  const collapses: { ring: Thing; onto: Thing }[] = [];
  for (const ring of rail)
    for (const h of held) {
      tick("collapses");
      if (h.kind === "floor" || !samePairing(ring.seats, h.thing.region)) continue;
      if (h.thing.m.coords.length === 1 && h.thing.m.coords[0] === ring) continue;
      const site: Site = [...all.map(() => []), [ring]];
      collapses.push({ ring: { label: ring.label, m: mm(ring), site, region: h.thing.region }, onto: h.thing });
    }

  const relate = (mets: Met[]) => {
    const out: Relation[] = [];
    for (const q of mets)
      for (const h of things) {
        tick("relate");
        out.push(...readings(h, q.thing).relations);
      }
    return group(out, tick);
  };

  return {
    dimension,
    cycle: k + 1,
    container,
    held,
    unordered,
    arrivals,
    remet,
    collapses,
    qstar: relate(arrivals),
    pq: group(pqRelations, tick),
    rstar: relate(remet),
    limit,
    railAfter: [...rail, ...arrivals.map((a) => a.ring)],
  };
}
