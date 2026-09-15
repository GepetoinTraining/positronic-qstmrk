import { hasFirst, hasSecond } from "./bead.ts";
import type { Tick } from "./budget.ts";
import { noTick } from "./budget.ts";
import { decadeName } from "./decade.ts";
import type { Ring } from "./ring.ts";
import type { MM } from "./mm.ts";
import type { Region } from "./region.ts";
import { holds } from "./region.ts";
import type { Site } from "./site.ts";
import { holdSites, sameSites } from "./site.ts";

/** Something on the relational side: its matrix, its ordered site, where it sits, how it is written. */
export type Thing = { readonly label: string; readonly m: MM; readonly site: Site; readonly region: Region };

export type Relation = { readonly top: Thing; readonly bottom: Thing };

/** a/b and c/d name one site when a held with d is, slot for slot and in order, b held with c. */
export function sameSite(p: Relation, q: Relation): boolean {
  return sameSites(holdSites(p.top.site, q.bottom.site), holdSites(p.bottom.site, q.top.site));
}

const isFloor = (t: Thing) => t.m.coords.length === 0;

/**
 * The turn reading first (the thing whose region holds the other's on top), then the relation reading;
 * nothing over the floor. Two things where neither holds the other are returned as `unordered`.
 */
export function readings(a: Thing, b: Thing): { relations: Relation[]; unordered: boolean } {
  let top: Thing, bottom: Thing;
  if (holds(a.region, b.region)) [top, bottom] = [a, b];
  else if (holds(b.region, a.region)) [top, bottom] = [b, a];
  else return { relations: [], unordered: true };
  const relations: Relation[] = [];
  for (const [x, y] of [[top, bottom], [bottom, top]] as const) if (!isFloor(y)) relations.push({ top: x, bottom: y });
  return { relations, unordered: false };
}

// machine side: each ring's pairing class, read once
const ringClass = new WeakMap<Ring, string>();
const classOf = (r: Ring) => {
  let c = ringClass.get(r);
  if (c === undefined) {
    c = decadeName(r.seats);
    ringClass.set(r, c);
  }
  return c;
};
const NEVER = Symbol("never");
const DEEPER = Symbol("deeper");

/**
 * A relation's site, slot by slot, as a key — valid while every stack holds at most one ring. In a slot the top and
 * bottom are balanced (both empty, or rings that pair), or a ring over nothing, or nothing over a ring. Two relations
 * name one site exactly when every slot reads the same; a slot with two different rings never pairs with any slot.
 * Balanced slots at the end are dropped (a slot a site does not reach is empty).
 */
function siteKey(r: Relation): string | typeof NEVER | typeof DEEPER {
  const T = r.top.site;
  const B = r.bottom.site;
  const parts: string[] = [];
  for (let i = 0; T[i] !== undefined || B[i] !== undefined; i++) {
    const t = T[i] ?? [];
    const b = B[i] ?? [];
    if (hasSecond(t) || hasSecond(b)) return DEEPER;
    if (!hasFirst(t) && !hasFirst(b)) parts.push("=");
    else if (hasFirst(t) && hasFirst(b)) {
      if (classOf(t[0]) !== classOf(b[0])) return NEVER;
      parts.push("=");
    } else parts.push(hasFirst(t) ? `+${classOf(t[0])}` : `-${classOf(b[0])}`);
  }
  while (parts[parts.length - 1] === "=") parts.pop();
  return parts.join("|");
}

/** Relations grouped by site, each group where its site first appears. */
export function group(relations: readonly Relation[], tick: Tick = noTick): Relation[][] {
  const groups: Relation[][] = [];
  const keyed = relations.map(siteKey);
  if (keyed.some((k) => k === DEEPER)) {
    // a stack deeper than one ring: compare with every group, as sameSite reads it
    for (const r of relations) {
      tick("group", () => `${groups.length} groups so far`);
      const g = groups.find((g) => sameSite(g[0], r));
      if (g) g.push(r);
      else groups.push([r]);
    }
    return groups;
  }
  const byKey = new Map<string, Relation[]>();
  relations.forEach((r, i) => {
    tick("group", () => `${groups.length} groups so far`);
    const k = keyed[i];
    if (k === NEVER || k === DEEPER) {
      groups.push([r]);
      return;
    }
    const g = byKey.get(k);
    if (g) g.push(r);
    else {
      const fresh = [r];
      groups.push(fresh);
      byKey.set(k, fresh);
    }
  });
  return groups;
}
