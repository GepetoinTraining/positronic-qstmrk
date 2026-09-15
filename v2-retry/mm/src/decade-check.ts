import { decadeName } from "./decade.ts";

/**
 * A machine-side check, not the construction: builds runs with a JS loop and compares the decade reading (made by
 * pairing) with JS's own decimal. It only confirms the reader agrees with floatland's names; nothing reads from it.
 */
const bad: string[] = [];
const run: unknown[] = [];
for (let n = 1; n <= 1200; n++) {
  run.push(0);
  if (decadeName(run) !== String(n)) bad.push(`${n} → ${decadeName(run)}`);
}
console.log(bad.length ? `DIFFERENT at ${bad.slice(0, 10).join(", ")}` : "EQUAL for runs 1…1200");
