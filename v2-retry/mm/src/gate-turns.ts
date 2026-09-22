import { readFileSync } from "node:fs";
import type { Cycle } from "./cycle.ts";
import { walkCycle } from "./cycle.ts";
import { nameLedger, namedRail } from "./ledger.ts";
import { receiptOf, seal } from "./receipt.ts";
import type { Ring } from "./ring.ts";
import { bitRing } from "./ring.ts";
import { identity, turnOf } from "./turns.ts";

/**
 * The gate for turns: the identity turn must reproduce the sealed cycle receipts byte for byte before anything
 * else is read. Walks main.ts's opening sequence without the store — 2D cycles 1–4 and 3D cycles 1–2, with the
 * namings between them — through receipt.ts, and compares each receipt with its file in receipts/. The identity is
 * passed three ways (omitted, identity(D), turnOf("1", D)). Then checks that a turn does act: 2D cycle 2 under -1
 * must differ from the sealed receipt (cycle 1 cannot differ: every held ring there is the container's own).
 *
 *   node --experimental-strip-types src/gate-turns.ts     exit 1 unless every receipt is EQUAL and the turn acts
 */

let bad = 0;
const check = (c: Cycle, name: string) => {
  const got = seal(receiptOf(c));
  const want = readFileSync(new URL(`../receipts/${name}`, import.meta.url), "utf8");
  const ok = got === want;
  if (!ok) bad++;
  process.stdout.write(`${name.padEnd(14)} ${ok ? "EQUAL    " : "DIFFERENT"}   ${got.slice(-65, -1)}\n`);
  return c;
};

const walks: Cycle[] = [];
let rail: Ring[] = [bitRing];
rail = check(walks[walks.push(walkCycle(rail, 0, 2, { turn: identity(2) })) - 1], "cycle1-2d.txt").railAfter;
rail = check(walks[walks.push(walkCycle(rail, 1, 2)) - 1], "cycle2-2d.txt").railAfter;
rail = check(walks[walks.push(walkCycle(rail, 0, 3, { turn: turnOf("1", 3) })) - 1], "cycle1-3d.txt").railAfter;

const naming = (r: readonly Ring[]) => {
  const ledger = nameLedger(walks);
  return { ledger, ...namedRail(r, ledger) };
};
let named = naming(rail);
rail = check(walks[walks.push(walkCycle(named.rail, 2, 2, { opus: named.ledger, known: named.known, turn: turnOf("1", 2) })) - 1], "cycle3-2d.txt").railAfter;
named = naming(rail);
rail = check(walks[walks.push(walkCycle(named.rail, 1, 3, { opus: named.ledger, known: named.known })) - 1], "cycle2-3d.txt").railAfter;
named = naming(rail);
rail = check(walks[walks.push(walkCycle(named.rail, 3, 2, { opus: named.ledger, known: named.known, turn: identity(2) })) - 1], "cycle4-2d.txt").railAfter;

// the turn acts: 2D cycle 2 under -1 is not the sealed cycle 2
const turned = seal(receiptOf(walkCycle(walks[0].railAfter, 1, 2, { turn: turnOf("-1", 2) })));
const acts = turned !== readFileSync(new URL("../receipts/cycle2-2d.txt", import.meta.url), "utf8");
process.stdout.write(`cycle2-2d under -1: ${acts ? "DIFFERENT from the sealed receipt (the turn acts)" : "EQUAL to the sealed receipt (the turn does nothing)"}\n`);
if (!acts) bad++;

process.stdout.write(bad ? `gate FAILED: ${bad} check(s)\n` : "gate passed: the identity turn is the sealed walk, byte for byte\n");
process.exit(bad ? 1 : 0);
