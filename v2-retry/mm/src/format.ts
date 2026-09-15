import { hasFirst } from "./bead.ts";
import { decadeName } from "./decade.ts";
import type { Relation, Thing } from "./relation.ts";
import { mmText } from "./mm.ts";

/** The coordinate subscript dropped: what the thing reads as one resolution lower. */
export const flat = (label: string) => label.replace(/[₀-₉]/g, "");

export const relationText = (r: Relation, show: (t: Thing) => string = (t) => t.label) => `${show(r.top)}/${show(r.bottom)}`;

const groupText = (texts: string[]) => (texts.length === 1 ? texts[0] : `[${texts.join(", ")}]`);

/** Groups with every address shown. */
export function addressed(groups: readonly Relation[][]): string {
  return groups.map((g) => groupText(g.map((r) => relationText(r)))).join(", ");
}

/** Groups read one resolution lower: subscripts dropped, repeated readings and repeated groups merged. */
export function collapsed(groups: readonly Relation[][]): string {
  const seen: string[] = [];
  for (const g of groups) {
    const text = groupText([...new Set(g.map((r) => relationText(r, (t) => flat(t.label))))]);
    if (!seen.includes(text)) seen.push(text);
  }
  return seen.join(", ");
}

export function resolution(t: Thing): string {
  return `${mmText(t.m)} (${hasFirst(t.m.coords) ? `${decadeName(t.m.coords)}D` : "floor"})`; // coordinates read in decades, by pairing
}
