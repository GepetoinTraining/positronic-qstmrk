import { createHash } from "node:crypto";
import type { Cycle } from "./cycle.ts";
import { decadeName } from "./decade.ts";
import { addressed, collapsed, resolution } from "./format.ts";
import { mmText } from "./mm.ts";
import type { Ring } from "./ring.ts";
import { isIdentity, turnText } from "./turns.ts";

/** The 1D reading of cycle 1, already sealed by the Rust oscillator (v3). */
const ONE_D = "2{Q*(4/O1, O1/4, O1/2, 2/O1, 1/O1), Pq(4/2, [2/4, 1/2], 1/4), Pn(2)}";

/** a body with its seal: the sha256 of the body on one last line */
export const seal = (body: string) => body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;

/**
 * A cycle's receipt (format v2.mm.cycle.v4). Under the identity turn the header reads as it always has, so the
 * sealed receipts gate the walk byte for byte; under any other turn the header names the turn.
 */
export function receiptOf(c: Cycle): string {
  const seat = (s: readonly number[]) => `(${s.join(",")})`;
  const lines: string[] = [];
  const say = (s = "") => lines.push(s);
  const read = c.reading ? `; reading ${c.reading.label}${c.broke ? " (broke, read forward)" : ""}` : "";
  const orientation = (isIdentity(c.turn) ? "every direction +" : `turn ${turnText(c.turn)}`) + read;
  say(`format=v2.mm.cycle.v4 dimension=${c.dimension} cycle=${c.cycle}  (order matters; ${orientation})`);
  say(`container ${mmText(c.container)}`);
  say("held:");
  for (const { thing, kind } of c.held) say(`  ${thing.label.padEnd(9)} ${kind.padEnd(6)} ${resolution(thing).padEnd(20)} seats ${thing.region.map(seat).join(" ")}`);
  const mets = (title: string, list: typeof c.arrivals) => {
    say(`${title}:`);
    if (list.length === 0) say("  none");
    for (const a of list) say(`  ${a.thing.label.padEnd(4)} ${resolution(a.thing).padEnd(12)} seats ${a.thing.region.map(seat).join(" ")}${a.also.length ? `   also met at ${a.also.length} more region(s)` : ""}`);
  };
  mets("arrivals (new rings)", c.arrivals);
  mets("re-met (rings already on the rail, met here)", c.remet);
  say("collapses (a rail ring this container's matrix is the resolution of):");
  if (c.collapses.length === 0) say("  none");
  for (const x of c.collapses) say(`  ${x.ring.label} ↔ ${x.onto.label} ${resolution(x.onto)}`);
  say();
  const pn = decadeName(c.railAfter); // the rail read in decades, by pairing
  const head = c.container.coords[0].label;
  say(`every address:   ${head}{Q*(${addressed(c.qstar)}), Pq(${addressed(c.pq)}), Pn(${pn})}`);
  if (c.rstar.length) say(`re-met relations: R*(${addressed(c.rstar)})`);
  const down = `${head}{Q*(${collapsed(c.qstar)}), Pq(${collapsed(c.pq)}), Pn(${pn})}`;
  say(`read one lower:  ${down}`);
  if (c.dimension === 2 && c.cycle === 1) say(`gate (cycle 1 read one lower is the sealed 1D reading): ${down === ONE_D ? "EQUAL" : "DIFFERENT"}`);
  say(c.limit ? `limit of the container: ${c.limit.top.label}/${c.limit.bottom.label} = ${resolution(c.limit.top)} over ${resolution(c.limit.bottom)}` : "limit: the cover is held or a matrix");
  say(`rail after: ${c.railAfter.map((r: Ring) => r.label).join(" ")}`);
  return lines.join("\n") + "\n";
}
