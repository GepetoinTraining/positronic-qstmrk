import { useState } from "react";
import type { Label, Line, Poly, V3 } from "./Solid";
import { Solid, cone } from "./Solid";

/**
 * The 7D tab — pgarcia's 7D, drawn to be corrected, not walked. Relational, and rotational by nature.
 *   (1) every structure's ID (1/number) is fibered to one point missing at its centre; that absence is the medium
 *       rotation is evaluated from, and facing its own ID it takes its true north (no alignment in space, so any)
 *   (2) one piece of everything is removed and placed on a line reaching for the oscillator by absence, never arriving
 *   (3) aligned to its ID: {0°, 0°, 0°, ∅, 0°, 0°, 0°}; to move: {(1–360)°, (1–360)°, (1–360)°, ∅, consequence ×3},
 *       looking towards the centre from the centre
 *   (4) an outer cone: the point of view, an imaginary point-to-point connection at 90° to another object's point,
 *       both on outer faces
 *   (5) an inner cone, the first inverted, inside the object: its model of its own movement with 7⁻¹ and its 6
 *       slots, connecting the centre to 7/7 (the counter)
 *   (6) the oscillator, the first transcendental: X² − 1 = 1/x + 1, the circle with a missing centre and the two
 *       points 1/2 and 2, turning through {1, i, −i, −1}; π the second, outside the system
 * The drawing turns with sin/cos (machine side, description only): the construction does not need them.
 */

const N = 3; // seats per coordinate; the centre seat is the absence
const key = (p: readonly number[]) => p.join(",");

/** a structure of N×N×N seats with its centre seat missing */
function structure(origin: V3): V3[] {
  const c = Math.floor(N / 2);
  const out: V3[] = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) if (!(i === c && j === c && k === c)) out.push([origin[0] + i, origin[1] + j, origin[2] + k]);
  return out;
}

/** faces of the seats not shared with another seat of the same structure (so the hole shows) */
function outerFaces(seats: V3[]): V3[][] {
  const present = new Set(seats.map(key));
  const faces: V3[][] = [];
  for (const p of seats)
    for (const axis of [0, 1, 2])
      for (const dir of [0, 1]) {
        const n = [...p];
        n[axis] += dir ? 1 : -1;
        if (present.has(key(n))) continue;
        const [u, v] = [0, 1, 2].filter((k) => k !== axis);
        const corner = (du: number, dv: number): V3 => {
          const q = [...p];
          q[axis] += dir;
          q[u] += du;
          q[v] += dv;
          return q as unknown as V3;
        };
        faces.push([corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]);
      }
  return faces;
}

const rad = (d: number) => (d * Math.PI) / 180;

/** turn about `c` by the three chosen slots, 1st coordinate then 2nd then 3rd (description only) */
function turner(t: [number, number, number], c: V3) {
  const [a, b, g] = t.map(rad);
  return (v: V3): V3 => {
    let [x, y, z] = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
    [y, z] = [y * Math.cos(a) - z * Math.sin(a), y * Math.sin(a) + z * Math.cos(a)];
    [x, z] = [x * Math.cos(b) + z * Math.sin(b), -x * Math.sin(b) + z * Math.cos(b)];
    [x, y] = [x * Math.cos(g) - y * Math.sin(g), x * Math.sin(g) + y * Math.cos(g)];
    return [x + c[0], y + c[1], z + c[2]];
  };
}

const addv = (a: V3, b: V3, k = 1): V3 => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: V3): V3 => {
  const l = Math.hypot(...a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

const QUARTERS = [
  { name: "1", angle: 0 },
  { name: "i", angle: 90 },
  { name: "−i", angle: 270 },
  { name: "−1", angle: 180 },
];

function OscillatorCircle() {
  const [step, setStep] = useState(0);
  const S = 46;
  const cx = 150;
  const cy = 130;
  const q = QUARTERS[step];
  const at = (r: number, deg: number) => ({ x: cx + r * S * Math.cos(rad(deg)), y: cy - r * S * Math.sin(rad(deg)) });
  const two = at(2, q.angle);
  const half = at(0.5, q.angle);
  return (
    <div>
      <svg className="graph" viewBox="0 0 300 260" role="img" style={{ maxWidth: 320 }}>
        <circle cx={cx} cy={cy} r={S} fill="none" stroke="#1b1a18" strokeWidth={1.4} />
        {QUARTERS.map((g) => {
          const p = at(2, g.angle);
          return (
            <text key={g.name} x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fill="#b9b3a8">
              {g.name}
            </text>
          );
        })}
        <line x1={cx} y1={cy} x2={two.x} y2={two.y} stroke="#8c867c" strokeDasharray="3 3" />
        <circle cx={cx} cy={cy} r={5} fill="#f4f1ec" stroke="#1b1a18" strokeDasharray="2 2" />
        <text x={cx + 8} y={cy + 16} fontSize={10} fill="#5a554d">
          ∅ missing centre
        </text>
        <circle cx={half.x} cy={half.y} r={6} fill="#86a3b8" />
        <text x={half.x} y={half.y - 9} textAnchor="middle" fontSize={11}>
          1/2
        </text>
        <circle cx={two.x} cy={two.y} r={6} fill="#c9853f" />
        <text x={two.x} y={two.y - 9} textAnchor="middle" fontSize={11}>
          2
        </text>
        <text x={cx} y={cy + S + 16} textAnchor="middle" fontSize={10} fill="#5a554d">
          radius 1
        </text>
      </svg>
      <div className="toolbar">
        <button onClick={() => setStep((s) => (s + 1) % QUARTERS.length)}>turn</button>
        <span className="small" style={{ marginLeft: 8 }}>
          through {"{"}
          {QUARTERS.map((g, i) => (
            <b key={g.name} style={{ color: i === step ? "#a23b2a" : undefined }}>
              {i ? ", " : ""}
              {g.name}
            </b>
          ))}
          {"}"}
        </span>
      </div>
    </div>
  );
}

export function SevenD() {
  const [theta, setTheta] = useState<[number, number, number]>([0, 0, 0]);
  const aligned = theta.every((t) => t === 0);

  const A0: V3 = [0, 0, 0];
  const B0: V3 = [N + 3, 0, 0];
  const h = N / 2;
  const cA: V3 = [h, h, h];
  const cB: V3 = [B0[0] + h, h, h];
  const turn = turner(theta, cA);

  const polys: Poly[] = [];
  const lines: Line[] = [];
  const labels: Label[] = [];

  for (const f of outerFaces(structure(A0))) polys.push({ pts: f.map(turn), fill: "#c9853f", opacity: 0.18, title: "structure A (moving) · 26 seats and the absence" });
  for (const f of outerFaces(structure(B0))) polys.push({ pts: f, fill: "#6d4a93", opacity: 0.14, title: "structure B · 26 seats and the absence" });

  // the absences at the centres, and the removed pieces on lines reaching for the oscillator
  const osc: V3 = [(cA[0] + cB[0]) / 2, h, N + 5];
  for (const [c, name] of [
    [cA, "A"],
    [cB, "B"],
  ] as const) {
    labels.push({ at: c, text: `∅ · ID 1/27 of ${name}`, color: "#1b1a18" });
    lines.push({ from: c, to: addv(c, sub(osc, c), 0.9), color: "#6d4a93", dashed: true, title: "the removed piece reaching for the oscillator by absence" });
    const piece = addv(c, sub(osc, c), 0.55);
    const s = 0.28;
    polys.push({ pts: [addv(piece, [-s, -s, 0]), addv(piece, [s, -s, 0]), addv(piece, [s, s, 0]), addv(piece, [-s, s, 0])], fill: "#6d4a93", opacity: 0.5, title: `the piece removed from ${name}` });
  }
  labels.push({ at: osc, text: "oscillator · never reached", color: "#6d4a93" });

  // the outer cone (point of view) from A's outer face, and the 90° ray to B's outer face
  const faceA = turn([N, h, h]);
  const outward = norm(sub(faceA, cA));
  const faceB: V3 = [B0[0], h, h];
  polys.push(...cone(faceA, addv(faceA, outward, 1.6), 0.8, "#3d6b39", "outer cone · point of view"));
  lines.push({ from: faceA, to: faceB, color: "#3d6b39", title: aligned ? "point to point at 90° to both faces" : "point to point (A has turned)" });
  labels.push({ at: addv(faceA, sub(faceB, faceA), 0.5), text: aligned ? "90° · point to point" : "point to point", color: "#3d6b39" });

  // the inner cone, inverted and pointing inwards: from the face in to the absence at the centre — the object
  // imagining itself (7⁻¹ and its 6 slots, to 7/7). Its interior only has coordinates with 8D.
  polys.push(...cone(faceA, cA, 0.8, "#a23b2a", "inner cone · inwards · the object imagines itself · 7⁻¹, 6 slots, to 7/7"));
  labels.push({ at: addv(cA, sub(faceA, cA), 0.45), text: "inner cone · inwards · 7⁻¹ → 7/7", color: "#a23b2a" });

  const slot = (i: number) => (
    <label key={i} className="small" style={{ display: "block" }}>
      slot {i + 1}{" "}
      <input type="range" min={0} max={360} value={theta[i]} onChange={(e) => setTheta((t) => t.map((x, k) => (k === i ? Number(e.target.value) : x)) as [number, number, number])} />{" "}
      <span className="mono">{theta[i]}°</span>
    </label>
  );
  const seven = `{${theta[0]}°, ${theta[1]}°, ${theta[2]}°, ∅, ${aligned ? "0°, 0°, 0°" : "(consequence)°, (consequence)°, (consequence)°"}}`;
  const sevenths = [1, 2, 3, 4, 5, 6].map((k) => {
    const digits = String(Math.floor((k * 1e6) / 7)).padStart(6, "0");
    return `${k}/7 = 0.${digits}…`;
  });

  return (
    <section className="tabbody">
      <div className="meta">
        <span>
          7D · relational, rotational by nature · from XYZ to ijk · <b>an attempt to be corrected</b> · not walked · drag to turn the view
        </span>
      </div>
      <div className="split">
        <div className="panel">
          <div className="toolbar" style={{ display: "block" }}>
            <p className="mono small" style={{ margin: "0 0 6px" }}>
              {seven} {aligned ? "· aligned to its ID" : "· looking towards the centre from the centre"}
            </p>
            {[0, 1, 2].map(slot)}
            <button onClick={() => setTheta([0, 0, 0])} disabled={aligned}>
              align to its ID
            </button>
          </div>
          <Solid polys={polys} labels={labels} lines={lines} initial={[0.5, 0.35]} />
        </div>
        <div className="panel">
          <h2>My reading, by number</h2>
          <ol className="small">
            <li>Every structure's ID (1/number) is fibered to one point missing at its centre, ∅. Rotation toward any axis is evaluated from that absence. Facing its own ID, it takes its true north. Space has no alignment, so any alignment is possible.</li>
            <li>Connecting the interior to the oscillator would expand everything. So one piece of everything is removed and placed on a line reaching for the oscillator by absence, never arriving.</li>
            <li>Aligned to its ID: {"{0°, 0°, 0°, ∅, 0°, 0°, 0°}"}. To move: three chosen slots (1–360)°, ∅, then three consequence slots, looking towards the centre from the centre.</li>
            <li>The outer cone is the point of view: an imaginary point-to-point connection at 90° to another object's point, both on outer faces.</li>
            <li>The inner cone is the outer cone inverted, pointing inwards so the object imagines itself. It models the object's movement with 7⁻¹ and its 6 slots, connecting the centre to 7/7 on the relational side, the counter. The inside only has coordinates with 8D.</li>
            <li>Split in 7, there is always one slot more than needed for a finer adjustment. That extra slot traces the movement from α to ω.</li>
            <li>This dimension removes the need for sin, cos, tan and π. π is the second transcendental, outside the system; the first is the oscillator.</li>
          </ol>
          <h2>The oscillator</h2>
          <p className="mono small">X² − 1 = 1/x + 1</p>
          <OscillatorCircle />
          <p className="small">
            Drawn: a missing centre, the circle of radius 1, and the points 2 and 1/2 on one ray. Across that circle each is the other's relation: 2 × 1/2 = 1, the radius. They turn through {"{1, i, −i, −1}"} in your order.
          </p>
          <h2>Checks (arithmetic, shadow)</h2>
          <ul className="small">
            <li>
              X² − 1 = 1/x + 1 holds at x = −1, and where x² = x + 1: x ≈ 1.618 and x ≈ −0.618. At that point both sides equal x itself: the square less one, and the inverse plus one, meet at x. In floatland's vocabulary that x (φ) is algebraic, not transcendental; I read your "transcendental" as "outside the system".
            </li>
            <li>
              Reaching x by x → 1/x + 1 from 1 gives 2, 3/2, 5/3, 8/5, 13/8, 21/13… Each step lands on the other side of it and never arrives. That is the 5D free side cycling to the other side, and these are Fibonacci relations.
            </li>
            <li>
              7⁻¹ in decades has 6 slots: {sevenths.join(" · ")}. Every k/7 turns the same six slots. 7/7 = 0.999999…, the counter.
            </li>
            <li>
              The walks already left out one piece in three limits: 3/4 (2D cycle 1), 8/9 (2D) and 26/27 (3D cycle 2). There the missing piece was a corner, not the centre. The limits 21/25 and 45/49 miss more than one.
            </li>
          </ul>
          <p className="muted small">
            Mine, to be corrected: 3×3×3 structures with the centre seat missing; the ID written 1/27; the cones' sizes and which is apex where; the oscillator placed above; the three consequence slots left unfilled. Not yet understood: how 8 and 9 fall out of 6 here.
          </p>
        </div>
      </div>
    </section>
  );
}
