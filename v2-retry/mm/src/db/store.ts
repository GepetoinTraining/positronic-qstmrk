import { mkdirSync, readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import type { Cycle } from "../cycle.ts";
import { decadeName } from "../decade.ts";
import type { LedgerEntry } from "../ledger.ts";
import { mmText } from "../mm.ts";
import { holds } from "../region.ts";
import type { Thing } from "../relation.ts";
import { siteLabels } from "../site.ts";

export const DB_PATH = fileURLToPath(new URL("../../data/mm.sqlite", import.meta.url));
const SCHEMA_VERSION = 5;

export function openStore(): DatabaseSync {
  mkdirSync(fileURLToPath(new URL("../../data/", import.meta.url)), { recursive: true });
  const db = new DatabaseSync(DB_PATH);
  // readers (the viewer) keep reading while a walk writes; a writer waits rather than failing
  db.exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;");
  const version = (db.prepare("PRAGMA user_version").get() as { user_version: number }).user_version;
  if (version !== SCHEMA_VERSION) {
    // the store is rebuilt from the walks; receipts stay on disk
    db.exec("DROP TABLE IF EXISTS edges; DROP TABLE IF EXISTS nodes; DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS ledger;");
    db.exec(`PRAGMA user_version = ${SCHEMA_VERSION}`);
  }
  db.exec(readFileSync(new URL("./schema.sql", import.meta.url), "utf8"));
  return db;
}

/** one transaction: the write holds the store once, briefly, instead of once per row */
function atomically<T>(db: DatabaseSync, work: () => T): T {
  db.exec("BEGIN IMMEDIATE");
  try {
    const out = work();
    db.exec("COMMIT");
    return out;
  } catch (e) {
    db.exec("ROLLBACK");
    throw e;
  }
}

export function writeLedger(db: DatabaseSync, entries: readonly LedgerEntry[]) {
  atomically(db, () => {
    db.exec("DELETE FROM ledger");
    const insert = db.prepare("INSERT INTO ledger (slot, reading, status, name, substitutes, found_in) VALUES (?, ?, ?, ?, ?, ?)");
    for (const e of entries) insert.run(e.slot, decadeName(e.ring.seats), e.status, e.name, JSON.stringify(e.substitutes), e.foundIn);
  });
}

export function writeCycle(db: DatabaseSync, c: Cycle, seal: string, receipt: string): number {
  return atomically(db, () => writeCycleRows(db, c, seal, receipt));
}

function writeCycleRows(db: DatabaseSync, c: Cycle, seal: string, receipt: string): number {
  const run = db
    .prepare("INSERT INTO runs (walk, dimension, container, rail, seal, receipt, created) VALUES (?, ?, ?, ?, ?, ?, ?)")
    .run(
      `cycle${c.cycle}`,
      c.dimension,
      c.container.coords.map((r) => r.label).join(","),
      JSON.stringify(c.railAfter.map((r) => ({ label: r.label, reading: decadeName(r.seats) }))), // each run read in decades, by pairing
      seal,
      receipt,
      new Date().toISOString(),
    );
  const runId = Number(run.lastInsertRowid);

  const all: { thing: Thing; kind: string; also: unknown[] }[] = [
    ...c.held.map((h) => ({ ...h, also: [] })),
    ...c.arrivals.map((a) => ({ thing: a.thing, kind: "question", also: a.also })),
    ...c.remet.map((a) => ({ thing: a.thing, kind: "remet", also: a.also })),
    ...c.collapses.map((x) => ({ thing: x.ring, kind: "rail", also: [] })),
  ];
  if (c.limit && !all.some((x) => x.thing === c.limit!.top)) all.push({ thing: c.limit.top, kind: "remet", also: [] });

  const ids = new Map<Thing, number>();
  const insertNode = db.prepare("INSERT INTO nodes (run_id, label, mm, resolution, kind, site, seats, reading, also, holds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
  for (const { thing, kind, also } of all) {
    const heldHere = all.filter((o) => o.thing !== thing && holds(thing.region, o.thing.region)).length;
    const r = insertNode.run(runId, thing.label, mmText(thing.m), thing.m.coords.length, kind, JSON.stringify(siteLabels(thing.site)), JSON.stringify(thing.region), decadeName(thing.region), JSON.stringify(also), heldHere);
    ids.set(thing, Number(r.lastInsertRowid));
  }

  const insertEdge = db.prepare("INSERT INTO edges (run_id, source, target, family, reading, site_group) VALUES (?, ?, ?, ?, ?, ?)");
  for (const [family, groups] of [["Q*", c.qstar], ["Pq", c.pq], ["R*", c.rstar]] as const) {
    groups.forEach((g, k) => {
      for (const rel of g) {
        const reading = holds(rel.top.region, rel.bottom.region) ? "turn" : "relation";
        insertEdge.run(runId, ids.get(rel.top)!, ids.get(rel.bottom)!, family, reading, `${family}:${k}`);
      }
    });
  }
  if (c.limit) insertEdge.run(runId, ids.get(c.limit.top)!, ids.get(c.limit.bottom)!, "limit", "limit", null);
  for (const [a, b] of c.unordered) insertEdge.run(runId, ids.get(a)!, ids.get(b)!, "unordered", "none", null);
  for (const x of c.collapses) insertEdge.run(runId, ids.get(x.ring)!, ids.get(x.onto)!, "collapse", "resolution", null);
  return runId;
}
