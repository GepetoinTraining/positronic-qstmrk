import type { LedgerRow, RailRing, WalkEdge, WalkNode } from "./api";
import { pairsInto } from "../../src/bead.ts";
import { nodeReading, readName } from "./api";
import { PanZoom } from "./PanZoom";

/**
 * The cells of a cycle: every seat run found, as wholes and relations, placed by what relates to what.
 *   relational side  quotients a/b of found runs whose bottom holds the top (the relation reading); tuples
 *                    naming one site form a fiber (a/a is a whole). Neighbours between relations: |ps − qr| = 1.
 *   hulled side 2⁻¹  quotients whose top holds the bottom (the turn reading, 4/3, 3/2, …) live in the imaginary
 *                    portion, seamed above the grid; each sits over its own inversion, the plane turned over.
 *   integer side     wholes. A whole that a table or cube of the walk (or a ledger substitute) crosses is a
 *                    relation of its factors — 6 is 2×3 — and stands next to those factors, not along a line.
 *                    A whole nothing crosses stands next to the oscillator's place.
 *   oscillator       where 1 was drawn: 1 is not present there; the place is kept for the oscillator.
 *   placement        row = hops from the oscillator's place; column = hops from the seam of identities 1/ring.
 * Names are the construction's decade readings (made by pairing). Which run holds which is read by pairing the
 * runs (seat regions) of the walk's nodes. Shadow, flagged: the viewer still reads names as JS numbers for gcd,
 * cross products and the order of its lists.
 */

/** `real`: some crossing of this whole is a matrix of the tab's own dimension; otherwise it is reached only as a pair (imaginary). */
type Fiber = { p: number; q: number; tuples: [number, number][]; products: number[][]; real: boolean };

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));


function bfs(adj: number[][], sources: number[]) {
  const d: (number | null)[] = adj.map(() => null);
  const queue = [...sources];
  for (const s of sources) d[s] = 0;
  while (queue.length) {
    const u = queue.shift()!;
    for (const v of adj[u]) if (d[v] === null) (d[v] = d[u]! + 1), queue.push(v);
  }
  return d;
}

type Props = { dimension: number; cycle: number; nodes: WalkNode[]; edges: WalkEdge[]; rail: RailRing[]; ledger: LedgerRow[]; selected: string; onPick: (label: string) => void };

/** the cycle of a ledger walk text, "2D cycle 4" → 4 */
const cycleOf = (walk: string) => Number(/cycle (\d+)/.exec(walk)?.[1] ?? 0);

export function Cells({ dimension, cycle, nodes, edges, rail, ledger, selected, onPick }: Props) {
  const names = [...new Set(nodes.map(nodeReading))].sort((a, b) => a - b);
  // each name's run: the seats of a walk node that reads as it
  const runOf = new Map<number, unknown[]>();
  for (const n of nodes) if (!runOf.has(nodeReading(n))) runOf.set(nodeReading(n), JSON.parse(n.seats));
  /** the top holds the bottom: the bottom's run pairs into the top's run */
  const holds = (top: number, bottom: number) => pairsInto(runOf.get(bottom) ?? [], runOf.get(top) ?? []);
  const labelOf = (n: number): string => rail.find((r) => readName(r.reading) === n)?.label ?? nodes.find((x) => nodeReading(x) === n)?.label ?? String(n);
  const limitNode = nodes.find((n) => edges.some((e) => e.family === "limit" && e.source === n.id));
  const limitName = limitNode ? nodeReading(limitNode) : null;

  // ring label → seats, from the rail, the ledger's slots and names, and the bit
  const seatsOf = new Map<string, number>([["2", 2]]);
  for (const r of rail) seatsOf.set(r.label, readName(r.reading));
  for (const l of ledger) {
    seatsOf.set(l.slot, readName(l.reading));
    seatsOf.set(l.name, readName(l.reading));
  }
  for (const n of nodes) if (!seatsOf.has(n.label)) seatsOf.set(n.label, nodeReading(n));

  // crossings: the walk's matrices of two or more coordinates (real), and the ledger's substitutes — a matrix of
  // this dimension is real; a pair across the seam, or a matrix of another dimension, is imaginary here
  const crossings: { whole: number; factors: number[]; real: boolean }[] = [];
  const addCrossing = (whole: number, labels: string[], real: boolean) => {
    const factors = labels.map((l) => seatsOf.get(l));
    if (factors.some((f) => f === undefined)) return;
    const fs = factors as number[];
    const same = crossings.find((c) => c.whole === whole && c.factors.join() === fs.join());
    if (same) same.real ||= real;
    else crossings.push({ whole, factors: fs, real });
  };
  for (const n of nodes) if (n.resolution >= 2) addCrossing(nodeReading(n), n.mm.slice(3, -1).split(","), true);
  for (const l of ledger)
    for (const s of JSON.parse(l.substitutes) as { factors: string[]; kind?: string; dimension: number; walk: string }[])
      addCrossing(readName(l.reading), s.factors, s.kind !== "pair" && s.dimension === dimension && cycleOf(s.walk) <= cycle);

  // fibers: quotients on the relational and hulled sides, wholes (with their crossings) on the integer side
  const fibers: Fiber[] = [];
  const fiberAt = (p: number, q: number) => {
    let f = fibers.find((x) => x.p === p && x.q === q);
    if (!f) fibers.push((f = { p, q, tuples: [], products: [], real: false }));
    return f;
  };
  for (const a of names)
    for (const b of names) {
      const [p, q] = a === b ? [a, 1] : [a / gcd(a, b), b / gcd(a, b)];
      fiberAt(p, q).tuples.push([a, b]);
    }
  for (const c of crossings)
    if (names.includes(c.whole) && c.factors.every((f) => names.includes(f))) {
      const f = fiberAt(c.whole, 1);
      if (!f.products.some((fs) => fs.join() === c.factors.join())) f.products.push(c.factors);
      f.real ||= c.real;
    }

  // the place where 1 was drawn: kept for the oscillator, 1 is not present there
  const slot = fibers.findIndex((f) => f.p === 1 && f.q === 1);
  const index = (p: number, q: number) => fibers.findIndex((f) => f.p === p && f.q === q);
  const adj: number[][] = fibers.map(() => []);
  const links: { i: number; j: number; kind: "neighbour" | "factor" | "slot" }[] = [];
  const link = (i: number, j: number, kind: "neighbour" | "factor" | "slot") => {
    if (i < 0 || j < 0 || i === j || adj[i].includes(j)) return;
    adj[i].push(j);
    adj[j].push(i);
    links.push({ i, j, kind });
  };
  // neighbours, except whole to whole (that is the counting line)
  fibers.forEach((a, i) =>
    fibers.forEach((b, j) => {
      if (j > i && !(a.q === 1 && b.q === 1) && Math.abs(a.p * b.q - a.q * b.p) === 1) link(i, j, "neighbour");
    }),
  );
  // a crossed whole beside its factors; a whole nothing crosses beside the oscillator's place
  fibers.forEach((f, i) => {
    if (f.q !== 1 || i === slot) return;
    if (f.products.length) for (const fs of f.products) for (const x of new Set(fs)) link(index(x, 1), i, "factor");
    else link(slot, i, "slot");
  });

  const seam = rail.map((r) => index(1, readName(r.reading))).filter((i) => i >= 0);
  const d0 = bfs(adj, [slot]);
  const ds = bfs(adj, seam);

  // 4 — the bit crossed with itself, two self-referencing — is held apart in a lane of its own
  const isFour = (f: Fiber) => f.q === 1 && f.products.some((fs) => fs.length === 2 && fs.every((x) => x === 2));
  // the hulled side: the top holds the bottom (the turn reading of a quotient)
  // read from a tuple of found runs on the fiber (tuples naming one site agree), never from p and q
  const isHulled = (f: Fiber) => f.q !== 1 && holds(f.tuples[0][0], f.tuples[0][1]);
  const inversion = (i: number) => index(fibers[i].q, fibers[i].p);
  // the imaginary integer side: a whole reached only as a crossing of a pair across the seam
  const isImaginaryWhole = (f: Fiber) => f.q === 1 && f.products.length > 0 && !f.real;

  // placement: lanes relational (−1) | seam (0) | 4 (2) | integer (1); hulled (3) is mirrored above the seam of 1;
  // imaginary wholes (4) are laid out with the integer side, then turned over above the seam
  const side = (i: number) =>
    i === slot || seam.includes(i) ? 0 : isFour(fibers[i]) ? 2 : isImaginaryWhole(fibers[i]) ? 4 : fibers[i].q === 1 ? 1 : isHulled(fibers[i]) ? 3 : -1;
  const lane = (i: number) => (side(i) === 4 ? 1 : side(i));
  const key = (i: number) => `${lane(i)}:${d0[i] ?? 99}:${lane(i) === 0 ? 0 : ds[i] ?? 9}`;
  const groups = new Map<string, number[]>();
  fibers.forEach((_, i) => {
    if (side(i) !== 3) groups.set(key(i), [...(groups.get(key(i)) ?? []), i]);
  });
  const STEP = 50;
  const ROW = 92;
  const colWidth = new Map<string, number>();
  for (const [k, g] of groups) {
    const [s, , c] = k.split(":");
    const ck = `${s}:${c}`;
    colWidth.set(ck, Math.max(colWidth.get(ck) ?? 0, g.length * STEP + 24));
  }
  const widthOf = (s: number) => [...colWidth].filter(([k]) => Number(k.split(":")[0]) === s).reduce((t, [, w]) => t + w, 0);
  const seamW = Math.max(90, colWidth.get("0:0") ?? 0);
  const left = 20;
  const centre = left + widthOf(-1) + seamW / 2;
  const fourW = widthOf(2) ? widthOf(2) + 40 : 0;
  const width = centre + seamW / 2 + fourW + widthOf(1) + left;
  const rowOf = (i: number) => d0[i] ?? 0;
  const maxRow = Math.max(...fibers.map((_, i) => (side(i) >= 3 ? 0 : rowOf(i))));
  const maxHulledRow = Math.max(0, ...fibers.map((_, i) => (side(i) === 3 ? rowOf(inversion(i)) : side(i) === 4 ? rowOf(i) : 0)));
  const slotY = 60 + maxHulledRow * ROW;
  const height = slotY + (maxRow + 2) * ROW;
  const baseX = (s: number, c: number) => {
    if (s === 0) return centre;
    const inner = [...colWidth].filter(([k]) => Number(k.split(":")[0]) === s && Number(k.split(":")[1]) < c).reduce((t, [, w]) => t + w, 0);
    const own = colWidth.get(`${s}:${c}`)!;
    if (s === 2) return centre + seamW / 2 + 20 + own / 2;
    return s > 0 ? centre + seamW / 2 + fourW + inner + own / 2 : centre - seamW / 2 - inner - own / 2;
  };
  const pos = fibers.map(() => ({ x: 0, y: 0 }));
  for (const [k, g] of groups) {
    const [s, r, c] = k.split(":").map(Number);
    g.forEach((i, n) => {
      pos[i] = { x: baseX(s, c) + (n - (g.length - 1) / 2) * STEP, y: slotY + r * ROW + (g.length > 1 ? (n % 2) * 20 - 10 : 0) };
    });
  }
  // each hulled quotient over its own inversion: the plane turned over across the seam of 1
  fibers.forEach((_, i) => {
    if (side(i) !== 3) return;
    const j = inversion(i);
    pos[i] = { x: pos[j].x, y: slotY - (pos[j].y - slotY) };
  });
  fibers.forEach((_, i) => {
    if (side(i) === 4) pos[i] = { x: pos[i].x, y: slotY - (pos[i].y - slotY) };
  });
  const seamYs = seam.map((i) => pos[i].y);

  const productText = (fs: number[]) => fs.map(labelOf).join("×");
  const fiberLabel = (f: Fiber) => (f.q === 1 ? (f.products.length ? productText(f.products[0]) : labelOf(f.p)) : `${f.p === 1 ? "1" : labelOf(f.p)}/${labelOf(f.q)}`);
  const text = (f: Fiber) => (f.q === 1 ? String(f.p) : `${f.p}/${f.q}`);
  const LINK_STYLE = { neighbour: { stroke: "#8c867c", opacity: 0.25, width: 1 }, factor: { stroke: "#b5651d", opacity: 0.75, width: 1.6 }, slot: { stroke: "#1b1a18", opacity: 0.3, width: 1 } };
  const hulledCount = fibers.filter(isHulled).length;

  return (
    <div>
      <p className="muted small">
        found {names.join(" ")} · seam {rail.map((r) => `1/${r.label}`).join(" ")} · {fibers.length - 1} fibers ({hulledCount} on the hulled side) ·{" "}
        {crossings.filter((c) => names.includes(c.whole)).length} crossings on the integer side · row = hops from the oscillator's place · column = hops from the seam
      </p>
      <PanZoom width={width} height={height} focusX={centre} label="cells placed by relations">
        {/* the imaginary portion, seamed above the grid */}
        <rect x={0} y={0} width={centre - seamW / 2} height={slotY} fill="#6d4a93" fillOpacity={0.05} />
        <text x={12} y={22} fontSize={12} fill="#6d4a93">
          hulled side 2⁻¹ · the top holds the bottom · each over its inversion
        </text>
        <rect x={centre + seamW / 2} y={0} width={width - centre - seamW / 2} height={slotY} fill="#b5651d" fillOpacity={0.05} />
        <text x={width - 12} y={22} fontSize={12} fill="#8a4a14" textAnchor="end">
          imaginary integer side · wholes reached only as i×real of a pair
        </text>
        <line x1={0} x2={width} y1={slotY} y2={slotY} stroke="#6d4a93" strokeWidth={1.5} strokeDasharray="10 6" strokeOpacity={0.6} />
        {links.map(({ i, j, kind }) => (
          <line key={`${i}-${j}`} x1={pos[i].x} y1={pos[i].y} x2={pos[j].x} y2={pos[j].y} stroke={LINK_STYLE[kind].stroke} strokeOpacity={LINK_STYLE[kind].opacity} strokeWidth={LINK_STYLE[kind].width} />
        ))}
        {seamYs.length > 0 && (
          <>
            <line x1={centre} x2={centre} y1={Math.min(...seamYs)} y2={Math.max(...seamYs) + ROW * 0.7} stroke="#1b1a18" strokeWidth={2} strokeDasharray="6 5" />
            <text x={centre} y={Math.max(...seamYs) + ROW * 0.7 + 20} textAnchor="middle" fontSize={16}>
              ∅
            </text>
          </>
        )}
        {slot >= 0 && (
          <g>
            <title>the oscillator's place · 1 is not present here</title>
            <circle cx={pos[slot].x} cy={pos[slot].y} r={22} fill="#f4f1ec" stroke="#1b1a18" strokeWidth={1.5} strokeDasharray="4 3" />
            <text x={pos[slot].x} y={pos[slot].y + 4} textAnchor="middle" fontSize={9} fill="#5a554d">
              oscillator
            </text>
          </g>
        )}
        {fibers.map((f, i) => {
          if (i === slot) return null;
          const onSeam = seam.includes(i);
          const hulled = isHulled(f);
          const imaginary = isImaginaryWhole(f);
          const isRail = f.q === 1 && rail.some((r) => readName(r.reading) === f.p) && f.products.length === 0;
          const isLimit = f.q === 1 && f.p === limitName;
          const four = isFour(f);
          const fill = onSeam ? "#2b2a27" : four ? "#6d4a93" : isLimit ? (imaginary ? "#e3b3a8" : "#a23b2a") : isRail ? "#b5651d" : imaginary ? "#f6ead9" : f.q === 1 ? "#e9c9a4" : hulled ? "#e4dbef" : "#bcd3e2";
          const ink = onSeam || (isLimit && !imaginary) || isRail || four ? "#f4f1ec" : "#1b1a18";
          const lbl = fiberLabel(f);
          const k = f.tuples.length;
          const crossed = f.products.map(productText).join(" · ");
          const picked = selected === lbl;
          return (
            <g key={i} className="pick" onClick={() => onPick(lbl)}>
              <title>{`${text(f)}${hulled ? " · hulled side 2⁻¹" : ""}${imaginary ? " · imaginary integer side" : ""}${crossed ? ` · crossed as ${crossed}` : ""} · fiber ${f.tuples.map(([a, b]) => `${a}/${b}`).join(" ≡ ")} · hops from the oscillator's place: ${d0[i] ?? "-"} · click to search`}</title>
              <circle
                cx={pos[i].x}
                cy={pos[i].y}
                r={18}
                fill={fill}
                stroke={picked ? "#a23b2a" : hulled ? "#6d4a93" : imaginary ? "#b5651d" : "#1b1a18"}
                strokeWidth={picked ? 3 : hulled || imaginary ? 1.2 : 0.8}
                strokeDasharray={(hulled || imaginary) && !picked ? "3 2" : undefined}
              />
              <text x={pos[i].x} y={pos[i].y + 4} textAnchor="middle" fontSize={11} fill={ink}>
                {text(f)}
              </text>
              {lbl !== text(f) && (
                <text x={pos[i].x} y={pos[i].y - 22} textAnchor="middle" fontSize={10} fill={f.products.length ? "#8a4a14" : "#5a554d"}>
                  {f.products.length > 1 ? `${lbl} +${f.products.length - 1}` : lbl}
                </text>
              )}
              {Array.from({ length: k }, (_, b) => (
                <circle key={b} cx={pos[i].x + b * 5 - ((k - 1) * 5) / 2} cy={pos[i].y + 25} r={2} fill="#1b1a18" />
              ))}
            </g>
          );
        })}
      </PanZoom>
      <ul className="legend">
        <li>
          <i className="dot" style={{ background: "#f4f1ec", border: "1px dashed #1b1a18" }} /> the oscillator's place (1 not present)
        </li>
        <li>
          <i className="dot" style={{ background: "#2b2a27" }} /> seam 1/ring
        </li>
        <li>
          <i className="dot" style={{ background: "#b5651d" }} /> ring nothing crosses
        </li>
        <li>
          <i className="dot" style={{ background: "#e9c9a4" }} /> whole crossed by its factors (a×b)
        </li>
        <li>
          <i className="dot" style={{ background: "#6d4a93" }} /> 4, two self-referencing
        </li>
        <li>
          <i className="dot" style={{ background: "#a23b2a" }} /> limit of this cycle
        </li>
        <li>
          <i className="dot" style={{ background: "#bcd3e2" }} /> relational side (the bottom holds the top)
        </li>
        <li>
          <i className="dot" style={{ background: "#e4dbef", border: "1px dashed #6d4a93" }} /> hulled side 2⁻¹ (the top holds the bottom)
        </li>
        <li>
          <i className="dot" style={{ background: "#f6ead9", border: "1px dashed #b5651d" }} /> imaginary integer side (crossed only as a pair, i×real)
        </li>
        <li className="muted">orange lines: factor → whole · grey: neighbours · purple dashes: the seam of 1</li>
      </ul>
    </div>
  );
}
