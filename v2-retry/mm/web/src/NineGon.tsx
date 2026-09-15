import { type MouseEvent, useRef, useState } from "react";

/**
 * The 9-gon that holds the whole thing — pgarcia's 9D, drawn to be corrected.
 *   (1) space carries only ±XYZ, no ijk. We cannot align to true south (the past), so the coordinates are blank and
 *       need a relational anchor: the oscillator.
 *   (2) from the front (pgarcia's drawing): the oscillator is the knot, nine spokes run out, the beak is the doubled
 *       spoke; only the first spokes are lim inf (0.999…), every other link is 1
 *   (3) the 9-gon's angle is 140° (220° on the other side). The beak is a 140° aperture at the oscillator with
 *       nothing connecting there; the 7 edges join the points, each called 2
 *   (4) from every point, the next 9-gon at distance 1: its points are 3, then 4, then 5 … — the web expands by 9-gon,
 *       with no finish anywhere (it is not capped; the machine only stops drawing)
 *   (5) from the 90° angle it is 2 as a flat surface: 2 points, the highest and lowest of the aperture following 140°
 * Drawing numbers and the address readout are description (machine side).
 */

type Pt = [number, number]; // model units, y up; one edge = 1
type View = "front" | "web" | "ninety";
const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;
const slotName = (d: number) => {
  const s = ((Math.round(d) % 360) + 360) % 360;
  return s === 0 ? 360 : s;
};

const R9 = 1 / (2 * Math.sin(rad(20))); // circumradius of a 9-gon with edge 1
const MACHINE_STOPS = 5; // generations the machine draws; the web itself has no finish

type Gon = { beak: Pt; axis: number; gen: number; verts: Pt[] };

/** a 9-gon whose beak vertex sits at `beak`, opening 140° around `axis` */
function gon(beak: Pt, axis: number, gen: number): Gon {
  const c: Pt = [beak[0] + R9 * Math.cos(rad(axis)), beak[1] + R9 * Math.sin(rad(axis))];
  const verts = Array.from({ length: 9 }, (_, j) => [c[0] + R9 * Math.cos(rad(axis + 180 + 40 * j)), c[1] + R9 * Math.sin(rad(axis + 180 + 40 * j))] as Pt);
  return { beak, axis, gen, verts };
}

/** the web: every point of a 9-gon (not its beak) is the beak of the next 9-gon, facing away from where it came from */
function webOf(turn: number, gens: number): Gon[] {
  const out: Gon[] = [];
  let layer = [gon([0, 0], turn, 1)];
  for (let g = 1; g <= gens; g++) {
    out.push(...layer);
    if (g === gens) break;
    const next: Gon[] = [];
    for (const n of layer) {
      const c: Pt = [n.beak[0] + R9 * Math.cos(rad(n.axis)), n.beak[1] + R9 * Math.sin(rad(n.axis))];
      for (let j = 1; j <= 8; j++) {
        const v = n.verts[j];
        next.push(gon(v, deg(Math.atan2(v[1] - c[1], v[0] - c[0])), g + 1));
      }
    }
    layer = next;
  }
  return out;
}

const GEN_COLOR = ["#1b1a18", "#8a4a14", "#6d4a93", "#3d6b39", "#34607f"];

// —— the front view (pgarcia's drawing), in screen units ——
const OSC: Pt = [300, 235];
const STEP = 68;
const LAYERS = 2;
const spokeAngle = (k: number, turn: number) => 90 + turn + 40 * k;
const at = (k: number, layer: number, turn: number): Pt => {
  const a = rad(spokeAngle(k, turn));
  const d = STEP * (1 + layer);
  return [OSC[0] + d * Math.cos(a), OSC[1] - d * Math.sin(a)];
};
const K = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const COLORS = ["#c9853f", "#b5651d", "#a23b2a", "#6d4a93", "#34607f", "#5f86a6", "#3d6b39", "#8fae8b", "#8a4a14"];

export function NineGon() {
  const [view, setView] = useState<View>("web");
  const [turn, setTurn] = useState(90);
  const [gens, setGens] = useState(2);
  const [probe, setProbe] = useState<Pt | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const toSvg = (e: MouseEvent<SVGSVGElement>): Pt | null => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(ctm.inverse());
    return [p.x, p.y];
  };

  const addressText = (dx: number, dy: number) => {
    const sign = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2)}`;
    return `(${sign(dx)}, ${sign(dy)}, +0.00) · slot ${slotName(deg(Math.atan2(dy, dx)))}° · from the anchor`;
  };

  const toolbar = (
    <div className="toolbar">
      <div className="seg" role="group" aria-label="view">
        {(
          [
            ["front", "from the front"],
            ["web", "the 9-gon web (140°)"],
            ["ninety", "at 90°"],
          ] as const
        ).map(([v, label]) => (
          <button
            key={v}
            className={v === view ? "on" : ""}
            onClick={() => {
              setView(v);
              setProbe(null);
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <label className="small" style={{ marginLeft: 12 }}>
        turn about the beak <input type="range" min={0} max={359} value={turn} onChange={(e) => setTurn(Number(e.target.value))} /> <span className="mono">{turn}°</span>
      </label>
      {view !== "front" && (
        <span className="small" style={{ marginLeft: 12 }}>
          expand{" "}
          <button onClick={() => setGens((g) => Math.max(1, g - 1))} disabled={gens <= 1}>
            −
          </button>{" "}
          <span className="mono">{gens}</span>{" "}
          <button onClick={() => setGens((g) => Math.min(MACHINE_STOPS, g + 1))} disabled={gens >= MACHINE_STOPS}>
            +
          </button>{" "}
          <span className="muted">points named {Array.from({ length: gens }, (_, i) => i + 2).join(", ")} …</span>
          {gens >= MACHINE_STOPS && <span className="muted"> · the web has no finish; the machine stops drawing here</span>}
        </span>
      )}
    </div>
  );

  // —— front ——
  if (view === "front") {
    return (
      <div>
        {toolbar}
        <p className="small muted">from the front: the oscillator is the knot, nine spokes run out, the beak is the doubled spoke</p>
        <svg
          ref={svgRef}
          className="graph"
          viewBox="0 0 860 470"
          role="img"
          style={{ cursor: "crosshair" }}
          onClick={(e) => {
            const p = toSvg(e);
            if (p) setProbe(p);
          }}
        >
          {K.filter((k) => k !== 9).map((k) =>
            Array.from({ length: LAYERS }, (_, i) => i + 1).map((layer) => {
              const a = at(k, layer - 1, turn);
              const b = at(k, layer, turn);
              const next = k === 8 ? null : at(k + 1, layer, turn);
              return (
                <g key={`${k}-${layer}`}>
                  <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#3d6b39" strokeOpacity={0.55} />
                  {next && <line x1={b[0]} y1={b[1]} x2={next[0]} y2={next[1]} stroke="#3d6b39" strokeOpacity={0.35} strokeDasharray="4 3" />}
                  <circle cx={b[0]} cy={b[1]} r={3} fill="#3d6b39" fillOpacity={0.6} />
                </g>
              );
            }),
          )}
          {K.map((k) => {
            const p = at(k, 0, turn);
            if (k === 9) {
              const a = rad(spokeAngle(9, turn));
              const nx = -Math.sin(a) * 4;
              const ny = -Math.cos(a) * 4;
              return (
                <g key={k}>
                  <line x1={OSC[0] + nx} y1={OSC[1] + ny} x2={p[0] + nx} y2={p[1] + ny} stroke="#1b1a18" strokeWidth={1.4} />
                  <line x1={OSC[0] - nx} y1={OSC[1] - ny} x2={p[0] - nx} y2={p[1] - ny} stroke="#1b1a18" strokeWidth={1.4} />
                </g>
              );
            }
            return <line key={k} x1={OSC[0]} y1={OSC[1]} x2={p[0]} y2={p[1]} stroke="#1b1a18" strokeWidth={1.4} />;
          })}
          {K.map((k) => {
            const p = at(k, 0, turn);
            const lab = at(k, 0.28, turn);
            return (
              <g key={`p${k}`}>
                <circle cx={p[0]} cy={p[1]} r={6} fill="#f4f1ec" stroke={COLORS[k - 1]} strokeWidth={2} />
                <text x={lab[0]} y={lab[1] + 4} textAnchor="middle" fontSize={11}>
                  {k === 9 ? "9 · beak" : k}
                </text>
              </g>
            );
          })}
          <circle cx={OSC[0]} cy={OSC[1]} r={9} fill="#f4f1ec" stroke="#1b1a18" strokeWidth={2} />
          <circle cx={OSC[0]} cy={OSC[1]} r={4} fill="none" stroke="#1b1a18" strokeWidth={1.2} />
          <text x={OSC[0] + 14} y={OSC[1] + 22} fontSize={11}>
            oscillator
          </text>
          {probe && (
            <g>
              <line x1={OSC[0]} y1={OSC[1]} x2={probe[0]} y2={probe[1]} stroke="#a23b2a" strokeWidth={1.2} />
              <circle cx={probe[0]} cy={probe[1]} r={4} fill="#a23b2a" />
              <text x={probe[0] + 8} y={probe[1] - 8} fontSize={11} fill="#a23b2a">
                {addressText((probe[0] - OSC[0]) / STEP, (OSC[1] - probe[1]) / STEP)}
              </text>
            </g>
          )}
          <text x={20} y={452} fontSize={11} fill="#5a554d">
            every first spoke (black) is 0.999…, lim inf · every web link (green) is 1 · the beak is the doubled spoke, nothing joins it
          </text>
        </svg>
      </div>
    );
  }

  const gonsList = webOf(turn, gens);
  const allPts = gonsList.flatMap((g) => g.verts);
  const xs = allPts.map((p) => p[0]);
  const ys = allPts.map((p) => p[1]);

  // —— at 90°: the flat surface, two points of highest and lowest aperture ——
  if (view === "ninety") {
    const n: Pt = [Math.cos(rad(turn + 90)), Math.sin(rad(turn + 90))];
    const vs = allPts.map((p) => p[0] * n[0] + p[1] * n[1]);
    const lo = Math.min(...vs);
    const hi = Math.max(...vs);
    const first = gonsList[0].verts.map((p) => p[0] * n[0] + p[1] * n[1]);
    const pad = 1.2;
    return (
      <div>
        {toolbar}
        <p className="small muted">at 90°: 2 as a flat surface, 2 points of the highest and lowest aperture following 140°</p>
        <svg className="graph" viewBox={`${-6} ${-hi - pad} ${12} ${hi - lo + 2 * pad}`} role="img" style={{ maxHeight: 470 }}>
          <line x1={0} y1={-hi} x2={0} y2={-lo} stroke="#1b1a18" strokeWidth={2} vectorEffect="non-scaling-stroke" />
          {vs.map((v, i) => (
            <line key={i} x1={-0.15} x2={0.15} y1={-v} y2={-v} stroke="#8c867c" strokeOpacity={0.3} vectorEffect="non-scaling-stroke" />
          ))}
          {[hi, lo].map((v, i) => (
            <g key={i}>
              <circle cx={0} cy={-v} r={0.18} fill="#e9c9a4" stroke="#8a4a14" vectorEffect="non-scaling-stroke" />
              <text x={0.4} y={-v + 0.12} fontSize={0.4}>
                2 · {i === 0 ? "highest" : "lowest"} of the aperture
              </text>
            </g>
          ))}
          <circle cx={0} cy={0} r={0.14} fill="#f4f1ec" stroke="#1b1a18" vectorEffect="non-scaling-stroke" />
          <text x={-0.4} y={0.12} fontSize={0.35} textAnchor="end">
            oscillator (anchor)
          </text>
          <text x={-0.4} y={-Math.max(...first) + 0.12} fontSize={0.3} textAnchor="end" fill="#5a554d">
            first 9-gon reaches here
          </text>
        </svg>
      </div>
    );
  }

  // —— the 9-gon web at 140° ——
  const pad = 1;
  const minX = Math.min(...xs) - pad;
  const maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad;
  const maxY = Math.max(...ys) + pad;
  const size = Math.max(maxX - minX, maxY - minY);
  const labelSize = size / 40;
  const S = (p: Pt): Pt => [p[0], -p[1]];

  return (
    <div>
      {toolbar}
      <p className="small muted">
        the 9-gon at 140°: the beak opens 140° at the oscillator with nothing connecting there (220° on the other side) · every point opens the next 9-gon at 1 · click for an address from the anchor
      </p>
      <svg
        ref={svgRef}
        className="graph"
        viewBox={`${minX} ${-maxY} ${maxX - minX} ${maxY - minY}`}
        role="img"
        style={{ cursor: "crosshair", maxHeight: 520 }}
        onClick={(e) => {
          const p = toSvg(e);
          if (p) setProbe([p[0], -p[1]]);
        }}
      >
        {gonsList
          .slice()
          .reverse()
          .map((g, gi) => {
            const color = GEN_COLOR[(g.gen - 1) % GEN_COLOR.length];
            const op = Math.max(0.25, 1 - (g.gen - 1) * 0.22);
            const v = g.verts;
            const near = (a: Pt, b: Pt, f: number): Pt => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
            return (
              <g key={gi} stroke={color} strokeOpacity={op} fill="none">
                {/* the 7 edges joining the points: the inner part, toward ∅ — lim inf, dotted */}
                {[1, 2, 3, 4, 5, 6, 7].map((j) => {
                  const a = S(v[j]);
                  const b = S(v[j + 1]);
                  return <line key={j} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} vectorEffect="non-scaling-stroke" strokeDasharray="4 3" strokeWidth={g.gen === 1 ? 1.6 : 0.9} />;
                })}
                {/* the beak: two sides from the beak point out to the next points (2 → 3), defined, solid — nothing connecting at the tip */}
                {[1, 8].map((j) => {
                  const a = S(v[j]);
                  const b = S(near(v[j], v[0], 0.9));
                  return <line key={`b${j}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} vectorEffect="non-scaling-stroke" strokeWidth={g.gen === 1 ? 1.8 : 1.1} />;
                })}
                {g.gen <= 2 &&
                  v.slice(1).map((p, j) => {
                    const q = S(p);
                    return (
                      <g key={`p${j}`}>
                        <circle cx={q[0]} cy={q[1]} r={labelSize * 0.45} fill="#f4f1ec" stroke={color} vectorEffect="non-scaling-stroke" />
                        <text x={q[0]} y={q[1] + labelSize * 0.35} fontSize={labelSize} textAnchor="middle" fill={color} stroke="none">
                          {g.gen + 1}
                        </text>
                      </g>
                    );
                  })}
              </g>
            );
          })}

        {/* the first 9-gon's beak aperture: 140°, and 220° on the other side */}
        {(() => {
          const r = 0.5;
          const a1 = rad(turn - 70);
          const a2 = rad(turn + 70);
          const p1 = S([r * Math.cos(a1), r * Math.sin(a1)]);
          const p2 = S([r * Math.cos(a2), r * Math.sin(a2)]);
          const lab = S([0.95 * Math.cos(rad(turn)), 0.95 * Math.sin(rad(turn))]);
          const out = S([0.7 * Math.cos(rad(turn + 180)), 0.7 * Math.sin(rad(turn + 180))]);
          return (
            <g>
              <path d={`M ${p1[0]} ${p1[1]} A ${r} ${r} 0 0 0 ${p2[0]} ${p2[1]}`} fill="none" stroke="#a23b2a" vectorEffect="non-scaling-stroke" />
              <text x={lab[0]} y={lab[1]} fontSize={labelSize} textAnchor="middle" fill="#a23b2a">
                140°
              </text>
              <text x={out[0]} y={out[1]} fontSize={labelSize * 0.9} textAnchor="middle" fill="#5a554d">
                220°
              </text>
            </g>
          );
        })()}

        {/* one tag per kind of edge */}
        {(() => {
          const v = gonsList[0].verts;
          const m = S([(v[4][0] + v[5][0]) / 2, (v[4][1] + v[5][1]) / 2]);
          const second = gonsList[1]?.verts;
          // a beak side of the next 9-gon: from a 2 out to a 3, defined
          const m2 = second ? S([(second[0][0] + second[1][0]) / 2, (second[0][1] + second[1][1]) / 2]) : null;
          return (
            <g>
              <text x={m[0]} y={m[1] - labelSize * 0.4} fontSize={labelSize * 0.8} textAnchor="middle" fill="#1b1a18">
                0.999…
              </text>
              {m2 && (
                <text x={m2[0]} y={m2[1] - labelSize * 0.4} fontSize={labelSize * 0.8} textAnchor="middle" fill="#8a4a14">
                  1
                </text>
              )}
            </g>
          );
        })()}

        <circle cx={0} cy={0} r={labelSize * 0.5} fill="#f4f1ec" stroke="#1b1a18" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        <text x={labelSize * 0.8} y={labelSize * 1.6} fontSize={labelSize} fill="#1b1a18">
          oscillator · anchor · beak (open)
        </text>

        {probe && (
          <g>
            <line x1={0} y1={0} x2={probe[0]} y2={-probe[1]} stroke="#a23b2a" vectorEffect="non-scaling-stroke" />
            <circle cx={probe[0]} cy={-probe[1]} r={labelSize * 0.3} fill="#a23b2a" />
            <text x={probe[0] + labelSize * 0.5} y={-probe[1] - labelSize * 0.5} fontSize={labelSize * 0.9} fill="#a23b2a">
              {addressText(probe[0], probe[1])}
            </text>
          </g>
        )}
      </svg>
      <p className="small muted">
        {gonsList.length} 9-gons drawn over {gens} generation{gens > 1 ? "s" : ""} · solid: from a 2 to a 3, defined (1) · dotted: the inner part toward ∅, lim inf (0.999…) · uncomfortable to look at, as it should be: no alignment to true south
      </p>
    </div>
  );
}
