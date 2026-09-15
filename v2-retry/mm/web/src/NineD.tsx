import { useState } from "react";
import type { Label, Line, Poly, V3 } from "./Solid";
import { NineGon } from "./NineGon";
import { Solid } from "./Solid";

/**
 * The 9D tab — space, pgarcia's 9D, drawn to be corrected, not walked.
 *   (1) empty coordinates everywhere
 *   (2) movement is focus and going: intent and moving
 *   (3) nothing moves if no force has been applied
 *   (4) the force can be self-forcing or not; the model holds both as the same thing — just numbers and their
 *       consequences
 * Only the present consequence is shown, never the sequence of steps: the whole sequence is 10D, not attempted.
 */

const B = 6; // how far the drawn space runs from the origin on each coordinate (description only)
const AXES = [
  { name: "1st", ways: ["left", "right"] },
  { name: "2nd", ways: ["near", "far"] },
  { name: "3rd", ways: ["down", "up"] },
];

type Name = "A" | "B";
type Force = { axis: number; way: -1 | 1 } | null;

const box = (c: V3, h: number): V3[][] => {
  const out: V3[][] = [];
  for (const axis of [0, 1, 2]) {
    const [u, v] = [0, 1, 2].filter((k) => k !== axis);
    for (const dir of [-1, 1]) {
      const corner = (du: number, dv: number): V3 => {
        const q = [...c];
        q[axis] += dir * h;
        q[u] += du * h;
        q[v] += dv * h;
        return q as unknown as V3;
      };
      out.push([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]);
    }
  }
  return out;
};

const coord = (v: readonly number[]) => `(${v.join(", ")})`;

export function NineD() {
  const [pos, setPos] = useState<Record<Name, V3>>({ A: [-3, 0, 0], B: [3, 0, 0] });
  const [focus, setFocus] = useState<Name | null>("A");
  const [force, setForce] = useState<Force>({ axis: 0, way: 1 });
  const [source, setSource] = useState<"self" | "external">("self");
  const [consequence, setConsequence] = useState("nothing has moved");

  const go = () => {
    if (!focus) return setConsequence("no focus, no intent: nothing moves");
    if (!force) return setConsequence(`${focus} is in focus, but no force is applied: nothing moves`);
    const from = pos[focus];
    const to = from.map((x, k) => (k === force.axis ? x + force.way : x)) as unknown as V3;
    setPos((p) => ({ ...p, [focus]: to }));
    setConsequence(`${focus} goes one step ${AXES[force.axis].ways[force.way > 0 ? 1 : 0]} (${source} force): ${coord(from)} → ${coord(to)}`);
  };

  const polys: Poly[] = [];
  const lines: Line[] = [];
  const labels: Label[] = [];

  // empty coordinates everywhere
  for (let x = -B; x <= B; x += 2)
    for (let y = -B; y <= B; y += 2)
      for (let z = -B; z <= B; z += 2)
        for (const k of [0, 1, 2]) {
          const a = [x, y, z];
          const b = [x, y, z];
          a[k] -= 0.12;
          b[k] += 0.12;
          lines.push({ from: a as unknown as V3, to: b as unknown as V3, color: "#8c867c", width: 0.8, opacity: 0.5 });
        }
  AXES.forEach((ax, k) => {
    for (const [i, s] of [
      [0, -1],
      [1, 1],
    ] as const) {
      const at = [0, 0, 0];
      at[k] = s * (B + 1);
      labels.push({ at: at as unknown as V3, text: ax.ways[i], color: "#5a554d" });
    }
  });

  // the two objects, each with its missing centre
  for (const [name, color] of [
    ["A", "#c9853f"],
    ["B", "#6d4a93"],
  ] as const) {
    const c = pos[name];
    const inFocus = focus === name;
    for (const f of box(c, 1.5)) polys.push({ pts: f, fill: color, opacity: inFocus ? 0.35 : 0.15, dashed: !inFocus, title: `object ${name} · centre ${coord(c)}${inFocus ? " · in focus" : ""}` });
    labels.push({ at: c, text: `${name} · ∅ ${coord(c)}`, color: "#1b1a18" });
  }

  // the force on the object in focus
  if (focus && force) {
    const c = pos[focus];
    const from = c.map((x, k) => (k === force.axis ? x + force.way * 1.5 : x)) as unknown as V3;
    const to = c.map((x, k) => (k === force.axis ? x + force.way * 3 : x)) as unknown as V3;
    lines.push({ from, to, color: "#3d6b39", width: 3, title: `${source} force` });
    labels.push({ at: to, text: `force · ${source}`, color: "#3d6b39" });
  }

  const apart = [0, 1, 2].some((k) => Math.abs(pos.A[k] - pos.B[k]) > 3);
  const touching = !apart && [0, 1, 2].some((k) => Math.abs(pos.A[k] - pos.B[k]) === 3);
  const relation = apart ? "apart: each outside the other" : touching ? "touching: faces meet, still outside each other" : "overlapping: one is inside the other";

  return (
    <section className="tabbody">
      <div className="meta">
        <span>
          9D · space · empty coordinates everywhere · movement is focus and going · <b>an attempt to be corrected</b> · not walked · drag to turn the view
        </span>
      </div>
      <div className="split">
        <div className="panel">
          <div className="toolbar" style={{ display: "block" }}>
            <div className="seg" role="group" aria-label="focus" style={{ marginRight: 12 }}>
              {(["A", "B", null] as const).map((n) => (
                <button key={String(n)} className={focus === n ? "on" : ""} onClick={() => setFocus(n)}>
                  {n ? `focus ${n}` : "no focus"}
                </button>
              ))}
            </div>
            <label className="small" style={{ marginRight: 12 }}>
              force{" "}
              <select
                value={force ? `${force.axis}:${force.way}` : "none"}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === "none") return setForce(null);
                  const [a, w] = v.split(":").map(Number);
                  setForce({ axis: a, way: w as -1 | 1 });
                }}
              >
                <option value="none">none</option>
                {AXES.flatMap((ax, k) =>
                  ([-1, 1] as const).map((w) => (
                    <option key={`${k}:${w}`} value={`${k}:${w}`}>
                      {ax.ways[w > 0 ? 1 : 0]} ({ax.name})
                    </option>
                  )),
                )}
              </select>
            </label>
            <div className="seg" role="group" aria-label="source" style={{ marginRight: 12 }}>
              {(["self", "external"] as const).map((s) => (
                <button key={s} className={source === s ? "on" : ""} onClick={() => setSource(s)}>
                  {s === "self" ? "self-forcing" : "external"}
                </button>
              ))}
            </div>
            <button onClick={go}>go</button>
          </div>
          <Solid polys={polys} labels={labels} lines={lines} initial={[0.55, 0.35]} />
        </div>
        <div className="panel">
          <h2>Now</h2>
          <ul className="small">
            <li>
              intent (focus): <b className="mono">{String(focus !== null)}</b>
              {focus ? ` · ${focus}` : ""}
            </li>
            <li>
              force applied: <b className="mono">{String(force !== null)}</b>
              {force ? ` · ${AXES[force.axis].ways[force.way > 0 ? 1 : 0]} · ${source}` : ""}
            </li>
            <li>
              moves on go: <b className="mono">{String(focus !== null && force !== null)}</b>
            </li>
            <li>A and B: {relation}</li>
            <li>consequence: {consequence}</li>
          </ul>
          <h2>My reading, by number</h2>
          <ol className="small">
            <li>9D is space: empty coordinates everywhere.</li>
            <li>Movement is focus and going: intent and moving.</li>
            <li>There is nothing to move if no force has been applied.</li>
            <li>The force can be self-forcing or not. The model holds both as the same thing: the same step, the same consequence. It is just numbers and their consequences.</li>
          </ol>
          <p className="muted small">
            Mine, to be corrected:
          </p>
          <ul className="muted small">
            <li>naming the three ways left/right, near/far, down/up (the 2nd pair is my guess);</li>
            <li>one step per go;</li>
            <li>the empty coordinates drawn every second step;</li>
            <li>only the present consequence kept, since the whole sequence is 10D and not attempted.</li>
          </ul>
        </div>
      </div>
      <div className="panel">
        <h2>The 9-gon that holds the whole thing</h2>
        <NineGon />
        <ol className="small">
          <li>Space carries only ±XYZ, not ijk. The 9-gon is absolute coordinates from true south (past α) of any observer.</li>
          <li>
            Its step sequence is 9 points ending at a triangle beak with nothing connecting them there. From the oscillator it is a 9-sided shape at distance 1, with edges of 0.999… (lim inf). It never closes, so it never folds into a flat polygon: a true 9D structure.
          </li>
          <li>Its beak points at the oscillator. From there it opens out into the integer side, giving every point a coordinate.</li>
          <li>
            Mapped ∅ → 3² points, space comes in. The 9 points fill 3², and ∅ stays outside at the open beak. Nine steps of 0.111… make 0.999…, never 1.
          </li>
          <li>Rotation (ijk) belongs to objects, in 7D, not to space. A point of space has only its ±XYZ address.</li>
          <li>
            From the front (pgarcia's drawing) the oscillator is the knot and nine spokes run out; the beak is the doubled spoke. ONLY the first spokes are lim-inf bound (0.999…). Every other link that makes the web possible is 1, so from here on there is a huge web of distance-1 coordinates, each 1 from another.
          </li>
          <li>
            The 9-gon's angle is 140°, with 220° on the other side. The beak is a 140° aperture at the oscillator with nothing connecting there, and the 7 edges join the points, each called 2. From every point comes the next 9-gon at distance 1: its points are 3, then 4, then 5 …. The web expands by 9-gon and has no finish anywhere.
          </li>
          <li>At 90° it is 2 as a flat surface: 2 points, the highest and lowest of the aperture following 140°.</li>
          <li>Looking at it is not comfortable, and is not supposed to be. We cannot align to true south (the past), so we use a blank coordinate system that requires a relational anchor.</li>
          <li>11D: 2 × 11, 22 … (given, not yet explained).</li>
          <li>
            The 9-gon turns all 360 slots around its beak and fills every single point of space with an address. That is why it is space: D3². Click a point to read its absolute ±XYZ address from the oscillator.
          </li>
          <li>
            Not drawn: 12D = 2²D × 3D, the box we imagine with no border. There is a border: light speed for this model, the cycling. Each cycle expands it, and creation is absurdly faster than anything inside. If everything moves in the same cycle, nothing reaches the edge of the 9-gon, so perception is the 12-gon.
          </li>
        </ol>
        <p className="muted small">
          Mine, to be corrected:
        </p>
        <ul className="muted small">
          <li>the front view after pgarcia's drawing: nine spokes from the knot, the beak as spoke 9 (doubled), no edges between the points;</li>
          <li>the web: unit links drawn outward along each spoke and (dashed) between neighbouring spokes, two layers, with none from the beak spoke;</li>
          <li>each child 9-gon faces away from its parent's middle;</li>
          <li>the machine stops drawing at 5 generations (the web itself has no finish), and labels stop after generation 2;</li>
          <li>the 90° view projects every point across the axis onto one flat line;</li>
          <li>addresses are read from the anchor in edge units (machine side);</li>
          <li>how 2⁷ comes out, which is not drawn now that the angle is corrected;</li>
          <li>the drawing is in the plane, whereas in space the turn fills all of ±XYZ.</li>
        </ul>
      </div>
    </section>
  );
}
