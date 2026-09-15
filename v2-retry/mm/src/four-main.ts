import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { pairsInto, samePairing } from "./bead.ts";
import { decadeName } from "./decade.ts";
import type { Cycle } from "./cycle.ts";
import { walkCycle } from "./cycle.ts";
import type { TwoSided } from "./four.ts";
import { across, joined } from "./four.ts";
import { nameLedger, namedRail } from "./ledger.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";

// the named rail, recomputed from the walks (nothing read from the store)
const walks: Cycle[] = [];
let rail: Ring[] = [bitRing];
for (let k = 0; k < 2; k++) {
  const c = walkCycle(rail, k, 2);
  walks.push(c);
  rail = c.railAfter;
}
const w3 = walkCycle(rail, 0, 3);
walks.push(w3);
const named = namedRail(w3.railAfter, nameLedger(walks)).rail;

type Run = { ring: Ring; splits: TwoSided[]; generation: number };
const tally = (r: Ring) => decadeName(r.seats); // a run's name, read in decades by pairing
const lines: string[] = [];
const say = (s = "") => lines.push(s);

say("format=v2.mm.4d.v1  MM[a, b, (c, d)] — the third coordinate two-sided: c showing, d inverted, a seam between, no seat shared");
say(`generation 0 (sides available): the named rail ${named.map((r) => r.label).join(" ")}`);

const runs: Run[] = named.map((ring) => ({ ring, splits: [], generation: 0 }));
for (let g = 1; g <= 2; g++) {
  const sides = runs.map((r) => r.ring);
  for (const showing of sides)
    for (const inverted of sides) {
      const t: TwoSided = { showing, inverted };
      const whole = joined(t);
      const known = runs.find((r) => samePairing(r.ring.seats, whole.seats));
      if (known) {
        if (!known.splits.some((s) => s.showing === showing && s.inverted === inverted)) known.splits.push(t);
      } else runs.push({ ring: { label: tally(whole), seats: whole.seats }, splits: [t], generation: g });
    }
}

say();
say("run   first met   its two-sided splits (showing,inverted)            neighbours across the seam");
for (const r of [...runs].sort((a, b) => (samePairing(a.ring.seats, b.ring.seats) ? 0 : pairsInto(a.ring.seats, b.ring.seats) ? -1 : 1))) {
  if (r.splits.length === 0) continue;
  const splits = r.splits.map((s) => `(${tally(s.showing)},${tally(s.inverted)})`).join(" ");
  const turns = r.splits.filter((s) => across(s) === "showing-turns").map((s) => `${tally(s.showing)}/${tally(s.inverted)}`);
  say(`${tally(r.ring).padEnd(5)} ${(r.generation === 0 ? "rail" : `gen ${r.generation}`).padEnd(10)}  ${splits.padEnd(52)} ${turns.length ? turns.join(", ") + " on show (inverted: " + turns.map((t) => t.split("/").reverse().join("/")).join(", ") + ")" : "—"}`);
}

say();
const land = (a: string, b: string) => {
  const r = runs.find((x) => x.splits.some((s) => tally(s.showing) === a && tally(s.inverted) === b));
  return r ? `${a}/${b} → the two interiors of ${tally(r.ring)} (first met ${r.generation === 0 ? "on the rail" : `in generation ${r.generation}`})` : `${a}/${b} → not reached in two generations`;
};
say("where the neighbour relations go:");
for (const [a, b] of [["3", "2"], ["4", "3"], ["5", "4"], ["6", "5"], ["7", "6"]]) say(`  ${land(a, b)}`);
say();
say("flagged: joining two sides across the seam holds their seat runs end to end on one coordinate — whether that is admitted, or is the addition the house has not produced, is the open declaration");
say("machine side: run names are read in decades by pairing; runs are ordered by which pairs into which; loops are wiring");

const body = lines.join("\n") + "\n";
const receipt = body + `seal sha256 ${createHash("sha256").update(body).digest("hex")}\n`;
writeFileSync(new URL("../receipts/4d-sides.txt", import.meta.url), receipt);
process.stdout.write(receipt);
