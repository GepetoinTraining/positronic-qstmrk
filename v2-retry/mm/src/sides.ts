import type { Bead } from "./bead.ts";
import { bead, pairsInto } from "./bead.ts";
import type { Tick } from "./budget.ts";
import { noTick } from "./budget.ts";
import { decadeName } from "./decade.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";
import type { Turn } from "./turns.ts";
import { identity } from "./turns.ts";

/**
 * The side-run walk: cycle.ts's walk with the seat lattice never built. Names only.
 *
 * Every held thing is a box anchored at a corner of the container, so it is given by one run per axis: a taken
 * coordinate's ring, read forward from the first seat or inverted from the far seat; an untaken axis is the first
 * seat alone. Two boxes hold one another, meet, or cross by their runs axis by axis; the tally of their union is
 * the two boxes less what they share; their union is a matrix when that tally is the product of the shared shadows.
 * The cover of everything held short of the whole is the whole less the corner no held box reaches. The order in
 * which things are held and paired is cycle.ts's order, so arrivals come in the same order and the rail is the same.
 *
 * Machine side, flagged: the tallies are products and differences of run lengths — the shadow of what cycle.ts does
 * by pairing seats. It earns nothing; it is gated against cycle.ts (gate-sides.ts) and reads the same numbers faster.
 * Relations (Q*, Pq, R*), sites and collapses are not built here; draw them with cycle.ts at the cycle you want.
 *
 * Offset readings (turns.ts `Reading`) are not carried here; turns are.
 */

export type SidesCycle = {
  dimension: number;
  cycle: number;
  turn: Turn;
  hull: number;
  heldCount: number;
  /** seat tallies of held ∪ arrivals ∪ re-met */
  names: Set<number>;
  /** arrivals' tallies, in the order met */
  arrivals: number[];
  remet: number[];
  railAfter: Ring[];
};

export type SidesOptions = { opus?: readonly unknown[]; known?: readonly Ring[]; tick?: Tick; turn?: Turn };

/** a box: per container axis, where it starts and how long it runs (addresses are wiring) */
type Box = { lo: Int32Array; len: Int32Array; size: number };

/**
 * A run of `n` beads that is not laid out: it has its length and pairs out bead by bead when iterated, which is all
 * a ring's seats are asked for here (`pairsInto`, `decadeName`, `length`). Machine side: the seat walk would lay the
 * beads out; thousands of arrivals of tens of thousands of beads is the whole heap.
 */
export function run(n: number): readonly Bead[] {
  const r = {
    length: n,
    *[Symbol.iterator]() {
      for (let i = 0; i < n; i++) yield bead;
    },
  };
  return r as unknown as readonly Bead[];
}

/** Coordinate subsets of size s, increasing — cycle.ts's order. */
function subsets(n: number, s: number): number[][] {
  if (s === 0) return [[]];
  const out: number[][] = [];
  for (let first = 0; first < n; first++) for (const rest of subsets(n, s - 1)) if (rest.every((a) => a > first)) out.push([first, ...rest]);
  return out;
}

/** Ordered choices of s rings — cycle.ts's order (the first ring varies slowest). */
function* assignments(rings: readonly Ring[], s: number): Generator<Ring[]> {
  if (s === 0) {
    yield [];
    return;
  }
  for (const r of rings) for (const rest of assignments(rings, s - 1)) yield [r, ...rest];
}

export function sidesCycle(rail: readonly Ring[], k: number, dimension: number, options: SidesOptions = {}): SidesCycle {
  const D = dimension;
  const opus = options.opus ?? rail.filter((r) => r !== bitRing);
  const known = options.known ?? [];
  const tick = options.tick ?? noTick;
  const turn = options.turn ?? identity(D);
  if (turn.perm.length !== D) throw new Error(`refused: turn ${turn.label} has ${turn.perm.length} slots, the walk has ${D}`);
  const R = rail[k];
  const n = R.seats.length;
  const fit = rail.filter((r) => pairsInto(r.seats, R.seats));

  const held: Box[] = [];
  const add = (rings: Ring[], axes: number[]) => {
    const lo = new Int32Array(D);
    const len = new Int32Array(D).fill(1);
    let size = 1;
    axes.forEach((a, kk) => {
      const axis = turn.perm[a];
      const l = rings[kk].seats.length;
      len[axis] = l;
      lo[axis] = turn.sign[a] === -1 ? n - l : 0;
      size *= l;
    });
    held.push({ lo, len, size });
  };
  const all = Array.from({ length: D }, (_, a) => a);
  add(all.map(() => R), all); // the whole
  for (let s = D; s >= 1; s--)
    for (const rings of assignments(fit, s))
      for (const axes of subsets(D, s)) {
        tick("held", () => `${held.length} held so far`);
        if (s === D && rings.every((r) => r === R)) continue;
        add(rings, axes);
      }
  add([], []); // the floor

  const heldReadings = new Set(held.map((b) => b.size));
  const railBySize = new Map<number, Ring>();
  for (const r of [...rail, ...known]) if (!railBySize.has(r.seats.length)) railBySize.set(r.seats.length, r);
  const metSizes = new Set<number>();
  const arrivals: number[] = [];
  const arrivalRings: Ring[] = [];
  const remet: number[] = [];
  const meet = (size: number) => {
    if (metSizes.has(size)) return;
    metSizes.add(size);
    const onRail = railBySize.get(size);
    if (onRail) remet.push(size);
    else {
      arrivals.push(size);
      arrivalRings.push({ label: `O${decadeName([...opus, ...arrivalRings, bead])}`, seats: run(size) });
    }
  };

  const holds = (a: Box, b: Box) => {
    for (let x = 0; x < D; x++) if (b.lo[x] < a.lo[x] || b.lo[x] + b.len[x] > a.lo[x] + a.len[x]) return false;
    return true;
  };
  const N = held.length;
  for (let i = 0; i < N; i++) {
    const a = held[i];
    for (let j = i + 1; j < N; j++) {
      tick("pairs", () => `thing ${i} of ${N}, ${metSizes.size} met`);
      const b = held[j];
      if (holds(a, b) || holds(b, a)) continue;
      // the union's tally, and the product of the shared shadows
      let inter = 1;
      let shadows = 1;
      for (let x = 0; x < D; x++) {
        const lo = Math.max(a.lo[x], b.lo[x]);
        const hi = Math.min(a.lo[x] + a.len[x], b.lo[x] + b.len[x]);
        const shared = Math.max(0, hi - lo);
        inter *= shared;
        shadows *= a.len[x] + b.len[x] - shared;
      }
      const u = a.size + b.size - inter;
      if (u === shadows) continue; // a matrix
      if (heldReadings.has(u)) continue;
      meet(u);
    }
  }

  // the cover of everything held short of the whole: the whole less the corner no held box reaches.
  // A seat with a first-seat coordinate is under a (D−1)-thing; otherwise it is under a full box unless on every
  // axis only the container's own ring reaches it: forward, past the second-largest fit run; inverted, short of it.
  const second = Math.max(0, ...fit.map((r) => r.seats.length).filter((l) => l < n));
  let uncovered = 1;
  for (let a = 0; a < D; a++) uncovered *= turn.sign[a] === -1 ? Math.max(0, n - second - 1) : n - Math.max(second, 1);
  tick("limit");
  if (uncovered > 0 && D >= 2) {
    const cover = Math.pow(n, D) - uncovered;
    if (!heldReadings.has(cover)) meet(cover);
  }

  const names = new Set<number>([...heldReadings, ...arrivals, ...remet]);
  return {
    dimension: D,
    cycle: k + 1,
    turn,
    hull: Math.max(...names),
    heldCount: held.length,
    names,
    arrivals,
    remet,
    railAfter: [...rail, ...arrivalRings],
  };
}
