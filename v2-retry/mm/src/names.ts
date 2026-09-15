import { decadeName } from "./decade.ts";
import type { MM } from "./mm.ts";
import { seatsOf } from "./mm.ts";
import { bitRing } from "./ring.ts";

const SUB = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];

/**
 * How a matrix is written for the reader. Matrices of the bit alone keep their shadow names (2, 4, 8);
 * anything with an Opus ring is written by ring labels. A thing that does not take every coordinate of its
 * container carries the coordinates it runs on (2₀, 4₀₂).
 */
export function matrixLabel(m: MM, axes: readonly number[], dimension: number): string {
  if (m.coords.length === 0) return "1";
  const bitOnly = m.coords.every((r) => r === bitRing);
  const base = bitOnly ? decadeName(seatsOf(m)) : m.coords.length === 1 ? m.coords[0].label : `[${m.coords.map((r) => r.label).join(",")}]`;
  return axes.length < dimension ? `${base}${axes.map((a) => SUB[a]).join("")}` : base;
}
