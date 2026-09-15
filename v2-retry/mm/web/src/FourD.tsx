type P = { x: number; y: number };

const S = 52; // a seat's edge on screen

/** +++ only: first coordinate down-right, second down-left, third up (`h` is the third coordinate's seat height). */
const isoAt = (ox: number, oy: number) => (i: number, j: number, k: number): P => ({ x: ox + (i - j) * S * 0.87, y: oy + (i + j) * S * 0.5 - k });
const poly = (ps: P[]) => ps.map((p) => `${p.x},${p.y}`).join(" ");

/**
 * One 3D object of the 4D container MM[2, 2, (third)]: two seats on each coordinate. Both objects keep every
 * seat; `hull` draws the third coordinate read as 2⁻¹ — the interior seen through the hull, faces open and
 * dashed — instead of the turn that shows.
 */
function Object3D({ fill, hull, third }: { fill: [string, string, string]; hull: boolean; third: string }) {
  const layers = 2;
  const h = S;
  const oy = 30 + layers * h + S * 0.2;
  const iso = isoAt(160, oy);
  const up = Math.min(2.6 * S, oy - 16);
  const cells: [number, number, number][] = [];
  for (let k = 0; k < layers; k++) for (const s of [0, 1, 2]) for (const i of [0, 1]) for (const j of [0, 1]) if (i + j === s) cells.push([i, j, k]);
  const axis = (to: P, text: string) => (
    <g>
      <line x1={iso(0, 0, 0).x} y1={iso(0, 0, 0).y} x2={to.x} y2={to.y} stroke="#a23b2a" strokeWidth={1.3} markerEnd="url(#plus4)" />
      <text x={to.x} y={to.y} dx={5} dy={-4} fontSize={11} fill="#a23b2a">
        {text}
      </text>
    </g>
  );
  return (
    <svg className="graph" viewBox="0 0 320 270" role="img">
      <defs>
        <marker id="plus4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#a23b2a" />
        </marker>
      </defs>
      {cells.map(([i, j, k]) => {
        const lo = k * h;
        const hi = lo + h;
        const top = [iso(i, j, hi), iso(i + 1, j, hi), iso(i + 1, j + 1, hi), iso(i, j + 1, hi)];
        const right = [iso(i + 1, j, lo), iso(i + 1, j + 1, lo), iso(i + 1, j + 1, hi), iso(i + 1, j, hi)];
        const left = [iso(i, j + 1, lo), iso(i + 1, j + 1, lo), iso(i + 1, j + 1, hi), iso(i, j + 1, hi)];
        return (
          <g key={`${i}${j}${k}`} stroke="#1b1a18" strokeWidth={1} fillOpacity={hull ? 0.3 : 1} strokeDasharray={hull ? "4 3" : undefined}>
            <polygon points={poly(right)} fill={fill[0]} />
            <polygon points={poly(left)} fill={fill[1]} />
            <polygon points={poly(top)} fill={fill[2]} />
          </g>
        );
      })}
      {axis(iso(3, 0, 0), "1st +")}
      {axis(iso(0, 3, 0), "2nd +")}
      {axis(iso(0, 0, up), `3rd · ${third}`)}
    </svg>
  );
}

/**
 * The 4D tab. 4D is shown as two 3D objects of one container: the third coordinate is the only place the
 * reading changes. The bit read as 2⁻¹ (the hulled side of the interior, imaginary) gives the relational
 * object; the bit read as a turn (2) gives the operational object. Both keep every seat. Declared only.
 */
export function FourD() {
  return (
    <section className="tabbody">
      <div className="meta">
        <span>
          declared container <b className="mono">MM[2,2,(third)]</b> · 4 is the plane number · the third coordinate is where the reading changes · not walked yet
        </span>
      </div>
      <div className="split">
        <div className="panel">
          <h2>Relational · third coordinate 2⁻¹</h2>
          <Object3D hull third="2⁻¹" fill={["#9fb9cc", "#86a3b8", "#bcd3e2"]} />
          <p className="mono small">MM[2, 2, 2⁻¹] · the hulled side, present and not showing</p>
        </div>
        <div className="panel">
          <h2>Operational · third coordinate 2</h2>
          <Object3D hull={false} third="2" fill={["#d9b48c", "#c49a6c", "#ecd5b8"]} />
          <p className="mono small">MM[2, 2, 2] · the turn, showing</p>
        </div>
      </div>
      <p className="muted small">
        Both objects have every seat. Only the reading of the third coordinate changes: 2⁻¹ on the left, 2 on the right. Declared ahead: 5D (two
        two-sided coordinates and the abscissa, the accumulator) and 6D (every coordinate two-sided, the three planes of 3D).
      </p>
    </section>
  );
}
