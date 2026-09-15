import type { MM, Seat } from "./mm.ts";
import { seatsOf } from "./mm.ts";

/** A region: seats of a container. Regions are not changed once made. */
export type Region = readonly Seat[];

// machine side: each region keeps a set of its seat addresses, so holding and union do not re-scan seat lists
const keyOf = (s: Seat) => s.join(",");
const keysOf = new WeakMap<Region, Set<string>>();
function keys(r: Region): Set<string> {
  let k = keysOf.get(r);
  if (!k) {
    k = new Set(r.map(keyOf));
    keysOf.set(r, k);
  }
  return k;
}
const containerSeats = new WeakMap<MM, Seat[]>();
function seatsOfContainer(container: MM): Seat[] {
  let s = containerSeats.get(container);
  if (!s) {
    s = seatsOf(container);
    containerSeats.set(container, s);
  }
  return s;
}

/**
 * Place a matrix in a container from the origin: coordinate k of `m` runs along container axis `axes[k]`;
 * every other axis stays at its first seat. Which axes it takes is its orientation — its address.
 */
export function place(container: MM, m: MM, axes: readonly number[]): Region {
  return seatsOfContainer(container).filter((s) =>
    s.every((addr, axis) => {
      const k = axes.indexOf(axis);
      return k < 0 ? addr === 0 : m.coords[k].seats[addr] !== undefined;
    }),
  );
}

/** Every seat of `b` stands in `a`. */
export function holds(a: Region, b: Region): boolean {
  const k = keys(a);
  return b.every((s) => k.has(keyOf(s)));
}

export function sameRegion(a: Region, b: Region): boolean {
  return holds(a, b) && holds(b, a);
}

export function union(regions: readonly Region[]): Region {
  const out: Seat[] = [];
  const seen = new Set<string>();
  for (const r of regions)
    for (const s of r) {
      const k = keyOf(s);
      if (!seen.has(k)) {
        seen.add(k);
        out.push(s);
      }
    }
  keysOf.set(out, seen);
  return out;
}

/** A region is a matrix when it is the product of its own shadows on each axis. */
export function isMatrix(container: MM, region: Region): boolean {
  const shadows = container.coords.map((_, axis) => new Set(region.map((s) => s[axis])));
  const product = seatsOfContainer(container).filter((s) => s.every((addr, axis) => shadows[axis].has(addr)));
  return sameRegion(product, region);
}
