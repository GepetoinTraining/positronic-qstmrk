import { useMemo } from "react";
import type { WalkEdge, WalkNode } from "./api";
import { FAMILY_COLOR } from "./Graph";

const LIMIT = 40;

type Row = { key: string; family: WalkEdge["family"]; members: WalkEdge[] };

/** Relations naming one site share a row; limit and unordered edges stand alone. */
function rowsOf(edges: WalkEdge[]): Row[] {
  // machine side: rows looked up by key, kept in the order each key first appears
  const byKey = new Map<string, Row>();
  for (const e of edges) {
    const key = e.site_group ?? `${e.family}#${e.id}`;
    const row = byKey.get(key);
    if (row) row.members.push(e);
    else byKey.set(key, { key, family: e.family, members: [e] });
  }
  return [...byKey.values()];
}

/** Does an edge match the query: a node label exactly (either end), else the relation text. */
export function edgeMatcher(nodes: WalkNode[], query: string) {
  const q = query.trim();
  const label = new Map(nodes.map((n) => [n.id, n.label]));
  const exact = nodes.some((n) => n.label === q);
  return (e: WalkEdge) => {
    if (q === "") return false;
    const s = label.get(e.source)!;
    const t = label.get(e.target)!;
    return exact ? s === q || t === q : `${s}/${t}`.includes(q);
  };
}

type Props = { nodes: WalkNode[]; edges: WalkEdge[]; query: string; setQuery: (q: string) => void };

export function RelationSearch({ nodes, edges, query, setQuery }: Props) {
  const label = useMemo(() => new Map(nodes.map((n) => [n.id, n.label])), [nodes]);
  const rows = useMemo(() => rowsOf(edges), [edges]);
  const match = edgeMatcher(nodes, query);
  const found = query.trim() === "" ? [] : rows.filter((r) => r.members.some(match));
  const shown = found.slice(0, LIMIT);
  const text = (e: WalkEdge) => `${label.get(e.source)}${e.family === "unordered" ? " · " : "/"}${label.get(e.target)}`;

  return (
    <div className="search">
      <div className="searchbar">
        <input type="search" placeholder="click a node or cell, or type a label (e.g. O2, [2,O1], 4/2₀)" value={query} onChange={(ev) => setQuery(ev.target.value)} aria-label="search relations" />
        {query && (
          <button className="clear" onClick={() => setQuery("")}>
            clear
          </button>
        )}
      </div>
      {query.trim() === "" ? (
        <p className="muted small">
          {rows.length} sites and edges in this cycle. Nothing is listed until something is searched.
        </p>
      ) : (
        <>
          <p className="muted small">
            {found.length} of {rows.length} rows match <b className="mono">{query}</b>
            {found.length > LIMIT ? ` · showing the first ${LIMIT}` : ""}
          </p>
          <table className="data">
            <thead>
              <tr>
                <th>family</th>
                <th>site</th>
                <th>relations naming it</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.key}>
                  <td style={{ color: FAMILY_COLOR[r.family] }}>{r.family}</td>
                  <td className="mono muted">{r.key.includes("#") ? "—" : r.key}</td>
                  <td className="mono">
                    {r.members.map((e, i) => (
                      <span key={e.id}>
                        {i > 0 && <span className="muted">  ≡  </span>}
                        <span className={match(e) ? "hit" : ""}>{text(e)}</span>
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
