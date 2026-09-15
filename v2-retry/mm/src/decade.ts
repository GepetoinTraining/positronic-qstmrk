import type { Bead } from "./bead.ts";
import { bead, hasFirst, samePairing } from "./bead.ts";
import type { MM } from "./mm.ts";
import { mm, seatsOf } from "./mm.ts";
import type { Region } from "./region.ts";
import { holds, place, union } from "./region.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";

/**
 * Reading a run in decades, by pairing only — no seat is counted.
 *   the digits are runs the construction has earned:
 *     1  the count reading of the bit, 2/2
 *     2  the bit
 *     3  the L of 2₀ and 2₁ (cycle 1's arrival)
 *     4  MM[2,2]
 *     5  the hull of MM[3,3] outside MM[2,2]
 *     6  MM[2,3]
 *     7  the hull of MM[4,4] outside MM[3,3]
 *     8  MM[2,2,2]
 *     9  MM[3,3]
 *   the decade is MM[2,5] (5D: 2 × 5)
 * A run is paired off against the decade as long as the decade pairs into what is left; what is left pairs with one
 * digit. The groups taken off are themselves a run, read the same way one place up. An empty place is the mark of a
 * cycle, 0 — not a produced zero.
 */

type Run = readonly unknown[];
const ringOf = (label: string, run: Run): Ring => ({ label, seats: run.map(() => bead) });
/** the seats of `outer` that `inner` does not hold */
const hull = (outer: MM, inner: MM): Region => {
  const all = seatsOf(outer);
  const held = place(outer, inner, inner.coords.map((_, k) => k));
  return all.filter((s) => !holds(held, [s]));
};

const one: Run = [bitRing.seats[0]];
const two: Run = bitRing.seats;
const four: Run = seatsOf(mm(bitRing, bitRing));
const box22 = mm(bitRing, bitRing);
const three: Run = union([place(box22, mm(bitRing), [0]), place(box22, mm(bitRing), [1])]);
const threeRing = ringOf("3", three);
const fourRing = ringOf("4", four);
const five: Run = hull(mm(threeRing, threeRing), mm(bitRing, bitRing));
const six: Run = seatsOf(mm(bitRing, threeRing));
const seven: Run = hull(mm(fourRing, fourRing), mm(threeRing, threeRing));
const eight: Run = seatsOf(mm(bitRing, bitRing, bitRing));
const nine: Run = seatsOf(mm(threeRing, threeRing));
const decade: Run = seatsOf(mm(bitRing, ringOf("5", five)));

const DIGITS: [string, Run][] = [
  ["1", one],
  ["2", two],
  ["3", three],
  ["4", four],
  ["5", five],
  ["6", six],
  ["7", seven],
  ["8", eight],
  ["9", nine],
];

/**
 * A run written from a decade reading, the other way: each place holds its digit's run, and a place one up holds a
 * decade for every bead of the run below it — built by pairing and joining runs end to end, never by counting.
 */
export function runOfReading(reading: string): Bead[] {
  let run: Bead[] = [];
  for (const ch of reading) {
    const up: Bead[] = [];
    for (const _ of run) for (const _d of decade) up.push(bead); // a decade for every bead of the places so far
    const d = ch === "0" ? [] : DIGITS.find(([name]) => name === ch)?.[1];
    if (d === undefined) throw new Error(`not a decade reading: ${reading}`);
    run = [...up, ...d.map(() => bead)];
  }
  return run;
}

/** the run read in decades (an empty run reads as nothing) */
export function decadeName(run: Run): string {
  // one pass: each element of the run pairs with the next element of the decade; when the decade is spent, a group
  // is taken off and the decade starts again. What is left when the run ends is the part short of a decade.
  const groups: Bead[] = [];
  let rest: unknown[] = [];
  let pattern = decade[Symbol.iterator]();
  let pending = pattern.next();
  for (const x of run) {
    rest.push(x);
    pending = pattern.next();
    if (pending.done) {
      groups.push(bead);
      rest = [];
      pattern = decade[Symbol.iterator]();
      pending = pattern.next();
    }
  }
  const digit = hasFirst(rest) ? DIGITS.find(([, d]) => samePairing(d, rest))?.[0] : "0";
  if (digit === undefined) throw new Error("a remainder short of the decade paired with no digit");
  return hasFirst(groups) ? decadeName(groups) + digit : hasFirst(rest) ? digit : "";
}
