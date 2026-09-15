import { bead } from "./bead.ts";
import type { Ring } from "./ring.ts";

/**
 * 4D as a subset of 3D: MM[a, b, (c, d)]. Three coordinates; the third is two-sided — `showing` is the half
 * you see, `inverted` the half you see when the third is inverted. Two interiors, one seam between them;
 * the sides share no seat, and they need not be equal.
 */
export type TwoSided = { showing: Ring; inverted: Ring };

/** The third coordinate read whole: the showing side's seats, the seam, then the inverted side's seats. */
export function joined(t: TwoSided): Ring {
  return { label: `(${t.showing.label},${t.inverted.label})`, seats: [...t.showing.seats.map(() => bead), ...t.inverted.seats.map(() => bead)] };
}

export type Across = "equal" | "showing-turns" | "inverted-turns" | "apart";

/**
 * How the two sides stand across the seam, by pairing seat for seat:
 *   equal           both run out together;
 *   showing-turns   the showing side has exactly one seat over — neighbours, the turn reading on show;
 *   inverted-turns  the inverted side has exactly one seat over — the same neighbours seen inverted;
 *   apart           more than one seat over on either side.
 */
export function across(t: TwoSided): Across {
  const x = t.showing.seats[Symbol.iterator]();
  const y = t.inverted.seats[Symbol.iterator]();
  for (;;) {
    const p = x.next();
    const q = y.next();
    if (p.done && q.done) return "equal";
    if (p.done || q.done) {
      const rest = p.done ? y : x;
      return rest.next().done ? (p.done ? "inverted-turns" : "showing-turns") : "apart";
    }
  }
}
