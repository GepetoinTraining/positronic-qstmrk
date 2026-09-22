import { useEffect, useState } from "react";
import type { LedgerRow, RailRing, Run, Walk } from "./api";
import { listRuns, loadLedger, loadRun } from "./api";
import { Cells } from "./Cells";
import { FAMILY_COLOR, Graph } from "./Graph";
import { Ledger } from "./Ledger";
import { RelationSearch, edgeMatcher } from "./RelationSearch";
import { SeatCubes } from "./SeatCubes";
import { SeatGrid } from "./SeatGrid";

type View = "cells" | "graph";

/** One dimension's walks: pick a cycle, see its cells or its graph, search its relations, read its things. */
export function DimensionTab({ dimension }: { dimension: 2 | 3 }) {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [walk, setWalk] = useState<Walk | null>(null);
  const [runMissing, setRunMissing] = useState(false);
  const [view, setView] = useState<View>(dimension === 3 ? "graph" : "cells");
  const [query, setQuery] = useState("");
  const [ledger, setLedger] = useState<LedgerRow[]>([]);

  useEffect(() => {
    loadLedger().then(setLedger);
  }, []);

  useEffect(() => {
    setRuns(null);
    setWalk(null);
    setRunMissing(false);
    listRuns(dimension).then((rs) => {
      setRuns(rs);
      setRunId(rs.length ? rs[rs.length - 1].id : null);
    });
  }, [dimension]);

  useEffect(() => {
    if (runId === null) return;
    setWalk(null);
    setQuery("");
    setRunMissing(false);
    loadRun(runId).then((next) => {
      if (next) return setWalk(next);
      setRunMissing(true);
    });
  }, [runId]);

  if (runs === null) return <section className="tabbody muted">loading…</section>;
  if (runs.length === 0) return <section className="tabbody muted">No {dimension}D walk stored yet. Run <code>npm run walk</code>.</section>;

  const match = walk ? edgeMatcher(walk.nodes, query) : () => false;
  // graceful failure in the viewer: a late cycle can hold more than a browser can draw at once
  const distinctNames = walk ? new Set(walk.nodes.map((n) => n.reading)).size : 0;
  const CELLS_LIMIT = 200;
  const GRAPH_LIMIT = 400;
  const THINGS_LIMIT = 300;
  const SEATS_LIMIT = 400; // seat pictures per thing only for containers up to this many seats
  const tooMany = (what: string, count: number, limit: number) => (
    <p className="muted">
      {what} not drawn: this cycle has {count} (the viewer draws up to {limit}). The relations are still searchable on the right, and the receipt below is complete.
    </p>
  );
  const rail: RailRing[] = walk ? JSON.parse(walk.run.rail) : [];
  const whole = walk?.nodes.find((n) => n.kind === "whole");
  const container: number[][] = whole ? JSON.parse(whole.seats) : [];
  const Seats = dimension === 3 ? SeatCubes : SeatGrid;

  return (
    <section className="tabbody">
      <div className="toolbar">
        <div className="seg" role="group" aria-label="cycle">
          {runs.map((r) => (
            <button key={r.id} className={r.id === runId ? "on" : ""} onClick={() => setRunId(r.id)}>
              {r.walk.replace("cycle", "cycle ")}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="view">
          {(["cells", "graph"] as const).map((v) => (
            <button key={v} className={v === view ? "on" : ""} onClick={() => setView(v)}>
              {v}
            </button>
          ))}
        </div>
        {walk && (
          <span className="muted small">
            container <b className="mono">MM[{walk.run.container}]</b> · rail {rail.map((r) => r.label).join(" ")} · seal {walk.run.seal.slice(0, 12)}…
          </span>
        )}
      </div>

      {runMissing ? (
        <p className="muted">That walk is unavailable in this static build.</p>
      ) : !walk ? (
        <p className="muted">loading…</p>
      ) : (
        <div className={view === "cells" ? "stack" : "split"}>
          <div className="panel">
            {view === "cells" && distinctNames > CELLS_LIMIT ? (
              tooMany("Cells", distinctNames, CELLS_LIMIT)
            ) : view === "graph" && walk.nodes.length > GRAPH_LIMIT ? (
              tooMany("Graph", walk.nodes.length, GRAPH_LIMIT)
            ) : view === "cells" ? (
              <Cells dimension={dimension} cycle={Number(walk.run.walk.replace("cycle", ""))} nodes={walk.nodes} edges={walk.edges} rail={rail} ledger={ledger} selected={query} onPick={setQuery} />
            ) : (
              <>
                <Graph nodes={walk.nodes} edges={walk.edges} lit={(e) => query.trim() === "" || match(e)} selected={query} onPick={setQuery} />
                <ul className="legend">
                  {Object.entries(FAMILY_COLOR).map(([f, c]) => (
                    <li key={f}>
                      <i style={{ background: c }} /> {f}
                    </li>
                  ))}
                  <li className="muted">click a node to search · curve right = turn, left = relation</li>
                </ul>
              </>
            )}
          </div>

          <div className="panel tables">
            <Ledger rows={ledger} onPick={setQuery} />
            <h2>Relations by site</h2>
            <RelationSearch nodes={walk.nodes} edges={walk.edges} query={query} setQuery={setQuery} />

            <h2>Things</h2>
            <table className="data">
              <thead>
                <tr>
                  <th>label</th>
                  <th>matrix</th>
                  <th>res.</th>
                  <th>kind</th>
                  <th>seats</th>
                </tr>
              </thead>
              <tbody>
                {walk.nodes.slice(0, THINGS_LIMIT).map((n) => {
                  const also = JSON.parse(n.also).length;
                  return (
                    <tr key={n.id} className={query === n.label ? "lit pick" : "pick"} onClick={() => setQuery(n.label)}>
                      <td className="mono">{n.label}</td>
                      <td className="mono">{n.mm}</td>
                      <td>{n.resolution === 0 ? "floor" : `${n.resolution}D`}</td>
                      <td className="muted">
                        {n.kind}
                        {also ? ` · +${also} addr.` : ""}
                      </td>
                      <td>
                        {container.length <= SEATS_LIMIT ? (
                          <Seats seats={JSON.parse(n.seats)} container={container} />
                        ) : (
                          <span className="mono muted">{n.reading}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {walk.nodes.length > THINGS_LIMIT && (
              <p className="muted small">
                showing the first {THINGS_LIMIT} of {walk.nodes.length} things
              </p>
            )}
            <details>
              <summary>receipt</summary>
              <pre>{walk.run.receipt}</pre>
            </details>
          </div>
        </div>
      )}
    </section>
  );
}
