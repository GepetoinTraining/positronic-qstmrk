import type { WalkEdge, WalkNode } from "./api";

export const FAMILY_COLOR: Record<WalkEdge["family"], string> = {
  "Q*": "#b5651d",
  Pq: "#3d6b8c",
  "R*": "#5f7f35",
  limit: "#a23b2a",
  unordered: "#8c867c",
  collapse: "#6d4a93",
};

const KIND_FILL: Record<WalkNode["kind"], string> = {
  whole: "#2b2a27",
  cube: "#c49a6c",
  table: "#d9b48c",
  held: "#e9c9a4",
  floor: "#1b1a18",
  question: "#a23b2a",
  remet: "#5f7f35",
  rail: "#6d4a93",
};
const DARK = new Set<WalkNode["kind"]>(["whole", "floor", "question", "remet", "rail"]);

const R = 26;
const ROW = 120;
const COL = 110;

const markerId = (f: string) => `arrow-${f.replace("*", "s")}`;

/** Rows by how many other regions a node holds (most at the top); a row's nodes spread across. */
function layout(nodes: WalkNode[]) {
  const levels = [...new Set(nodes.map((n) => n.holds))].sort((a, b) => b - a);
  const widest = Math.max(...levels.map((lv) => nodes.filter((n) => n.holds === lv).length));
  const width = Math.max(560, widest * COL + 80);
  const pos = new Map<number, { x: number; y: number }>();
  levels.forEach((lv, r) => {
    const row = nodes.filter((n) => n.holds === lv);
    row.forEach((n, k) => pos.set(n.id, { x: width / 2 + (k - (row.length - 1) / 2) * COL, y: 50 + r * ROW }));
  });
  return { pos, width, height: 50 + (levels.length - 1) * ROW + 70 };
}

function curve(a: { x: number; y: number }, b: { x: number; y: number }, bend: number) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const s = { x: a.x + ux * R, y: a.y + uy * R };
  const e = { x: b.x - ux * (R + 6), y: b.y - uy * (R + 6) };
  const c = { x: (s.x + e.x) / 2 - uy * bend, y: (s.y + e.y) / 2 + ux * bend };
  return `M${s.x},${s.y} Q${c.x},${c.y} ${e.x},${e.y}`;
}

type Props = {
  nodes: WalkNode[];
  edges: WalkEdge[];
  lit: (e: WalkEdge) => boolean;
  selected: string;
  onPick: (label: string) => void;
};

export function Graph({ nodes, edges, lit, selected, onPick }: Props) {
  const { pos, width, height } = layout(nodes);
  return (
    <svg className="graph" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="nodes and edges of the walk">
      <defs>
        {Object.entries(FAMILY_COLOR).map(([f, c]) => (
          <marker key={f} id={markerId(f)} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={c} />
          </marker>
        ))}
      </defs>
      {edges.map((e) => {
        const a = pos.get(e.source)!;
        const b = pos.get(e.target)!;
        const bend = e.reading === "turn" ? 20 : e.reading === "relation" ? -20 : e.reading === "limit" ? 60 : e.reading === "resolution" ? -60 : 0;
        const on = lit(e);
        return (
          <path
            key={e.id}
            d={curve(a, b, bend)}
            fill="none"
            stroke={FAMILY_COLOR[e.family]}
            strokeWidth={on ? 2 : 1}
            strokeOpacity={on ? 0.9 : 0.08}
            strokeDasharray={e.family === "unordered" ? "3 4" : e.family === "limit" || e.family === "collapse" ? "8 5" : undefined}
            markerEnd={e.family === "unordered" ? undefined : `url(#${markerId(e.family)})`}
          >
            <title>{`${e.family} · ${e.reading}${e.site_group ? ` · site ${e.site_group}` : ""}`}</title>
          </path>
        );
      })}
      {nodes.map((n) => {
        const p = pos.get(n.id)!;
        return (
          <g key={n.id} className="pick" onClick={() => onPick(n.label)}>
            <title>{`${n.label} · ${n.mm} · ${n.kind} · ${n.resolution === 0 ? "floor" : `${n.resolution}D`} · click to search`}</title>
            <circle cx={p.x} cy={p.y} r={R} fill={KIND_FILL[n.kind]} stroke={selected === n.label ? "#a23b2a" : "#1b1a18"} strokeWidth={selected === n.label ? 3 : 1} />
            <text x={p.x} y={p.y + 5} textAnchor="middle" fontSize={n.label.length > 5 ? 11 : 14} fill={DARK.has(n.kind) ? "#f4f1ec" : "#1b1a18"}>
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
