import type { Ring } from "./ring.ts";
import { sameRing } from "./ring.ts";

/**
 * A site: for each coordinate slot, in order, the rings stacked there. Order matters (column, row, …).
 * A slot a thing does not reach is simply empty — nothing marks it. Every direction is + for now.
 */
export type Site = readonly (readonly Ring[])[];

/** Two sites held together: slot by slot, the stacks side by side. */
export function holdSites(a: Site, b: Site): Site {
  const out: Ring[][] = [];
  for (let i = 0; a[i] !== undefined || b[i] !== undefined; i++) out.push([...(a[i] ?? []), ...(b[i] ?? [])]);
  return out;
}

/** One site: every slot, in order, carries the same rings in the same order. */
export function sameSites(a: Site, b: Site): boolean {
  for (let i = 0; a[i] !== undefined || b[i] !== undefined; i++) {
    const x = a[i] ?? [];
    const y = b[i] ?? [];
    if (x.length !== y.length || !x.every((r, k) => sameRing(r, y[k]))) return false;
  }
  return true;
}

export function siteLabels(s: Site): string[][] {
  return s.map((slot) => slot.map((r) => r.label));
}
