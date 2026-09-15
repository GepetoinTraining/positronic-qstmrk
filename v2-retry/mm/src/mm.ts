import type { Ring } from "./ring.ts";
import { isDimension, sameRing } from "./ring.ts";

/**
 * A manifold matrix MM[r, s, …]: one ring per coordinate. No coordinates is the floor.
 * Its resolution is how many coordinates it has; every coordinate must be a dimension.
 */
export type MM = { readonly coords: readonly Ring[] };

export const floor: MM = { coords: [] };

export function mm(...coords: Ring[]): MM {
  for (const r of coords) {
    if (!isDimension(r)) throw new Error(`refused: ring ${r.label} has no second seat, so it is not a coordinate`);
  }
  return { coords };
}

/** A seat: one seat address per coordinate. Addresses are wiring (JS numbers), never values. */
export type Seat = readonly number[];

export function seatsOf(m: MM): Seat[] {
  let out: Seat[] = [[]];
  for (const r of m.coords) {
    const next: Seat[] = [];
    for (const s of out) r.seats.forEach((_, i) => next.push([...s, i]));
    out = next;
  }
  return out;
}

/** Two matrices held together: their coordinates side by side. */
export function hold(a: MM, b: MM): MM {
  return { coords: [...a.coords, ...b.coords] };
}

/** One matrix: the coordinates pair ring for ring, in order. MM[4,2] is not MM[2,4]. */
export function sameMM(a: MM, b: MM): boolean {
  return a.coords.length === b.coords.length && a.coords.every((r, i) => sameRing(r, b.coords[i]));
}

export function mmText(m: MM): string {
  return `MM[${m.coords.map((r) => r.label).join(",")}]`;
}
