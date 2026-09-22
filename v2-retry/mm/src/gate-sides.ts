import { walkCycle } from "./cycle.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";
import { sidesCycle } from "./sides.ts";
import { turnOf } from "./turns.ts";

/**
 * The gate for the side-run walk: on the same rail, cycle by cycle, sides.ts must give cycle.ts's names, its
 * arrivals in the same order with the same tallies, its re-met rings, and the same rail after — under the identity
 * and under turns of sign and of permutation. 2D cycles 1–9 (1, i, j, −1), 3D cycles 1–5 (1, −1, i, c, 1+i−j−k).
 *
 *   node --experimental-strip-types src/gate-sides.ts     exit 1 on any difference
 */

let bad = 0;
const same = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const sorted = (s: Iterable<number>) => [...s].sort((x, y) => x - y);

for (const [D, N, labels] of [[2, 9, ["1", "i", "j", "-1"]], [3, 5, ["1", "-1", "i", "c", "1+i-j-k"]]] as const) {
  for (const label of labels) {
    let rail: Ring[] = [bitRing];
    for (let k = 0; k < N; k++) {
      const turn = turnOf(label, D);
      const t0 = Date.now();
      const seat = walkCycle(rail, k, D, { turn });
      const t1 = Date.now();
      const side = sidesCycle(rail, k, D, { turn });
      const t2 = Date.now();
      const seatNames = new Set<number>();
      for (const h of seat.held) seatNames.add(h.thing.region.length);
      for (const a of seat.arrivals) seatNames.add(a.thing.region.length);
      for (const r of seat.remet) seatNames.add(r.thing.region.length);
      const checks: [string, boolean][] = [
        ["names", same(sorted(seatNames), sorted(side.names))],
        ["arrivals in order", same(seat.arrivals.map((a) => a.thing.region.length), side.arrivals)],
        ["re-met", same(sorted(seat.remet.map((r) => r.thing.region.length)), sorted(side.remet))],
        ["rail after", same(seat.railAfter.map((r) => r.seats.length), side.railAfter.map((r) => r.seats.length)) && seat.railAfter.every((r, i) => r.label === side.railAfter[i].label)],
        ["held count", seat.held.length === side.heldCount],
      ];
      const failed = checks.filter(([, ok]) => !ok).map(([name]) => name);
      if (failed.length) bad++;
      process.stdout.write(`${D}D cycle ${k + 1} [${label}]: ${failed.length ? `DIFFERENT (${failed.join(", ")})` : "EQUAL"}   hull ${side.hull}, ${side.names.size} names, ${side.arrivals.length} arrivals   seats ${((t1 - t0) / 1000).toFixed(2)}s, sides ${((t2 - t1) / 1000).toFixed(2)}s\n`);
      rail = seat.railAfter;
    }
  }
}
process.stdout.write(bad ? `gate FAILED: ${bad} cycle(s)\n` : "gate passed: the side-run walk reads the seat walk's numbers\n");
process.exit(bad ? 1 : 0);
