import type { Bead } from "./bead.ts";
import { bead, hasSecond, samePairing } from "./bead.ts";

/** A ring: its seats. `label` is for the reader only. */
export type Ring = { readonly label: string; readonly seats: readonly Bead[] };

/** The bit's ring: two seats, given. */
export const bitRing: Ring = { label: "2", seats: [bead, bead] };

/** A ring is a dimension only if it has a second seat. */
export function isDimension(r: Ring): boolean {
  return hasSecond(r.seats);
}

export function sameRing(a: Ring, b: Ring): boolean {
  return samePairing(a.seats, b.seats);
}
