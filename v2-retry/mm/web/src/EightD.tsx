import { useState } from "react";
import type { Label, Line, Poly, V3 } from "./Solid";
import { Solid, cone } from "./Solid";

/**
 * The 8D tab — pgarcia's 8D, drawn to be corrected, not walked.
 *   (1) the inner cone points inwards: the object imagines itself
 *   (2) that needs a boolean function saying which depth of the fractal is in focus
 *   (3) the interior grid of any defined structure, moving in (x, y, z) from the missing centre: the inside has
 *       coordinates too
 *   (4) with that, space (9D) unlocks: inside from outside, up from down, left from right
 * The structure is 3×3×3 seats with the centre missing, at every depth. Coordinates are read from the missing
 * centre, so they run both ways from ∅. Machine side: positions are JS numbers, for the drawing and the readout.
 */

const OFFS: V3[] = [];
for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) if (x || y || z) OFFS.push([x, y, z]);
const key = (v: readonly number[]) => v.join(",");
const PRESENT = new Set(OFFS.map(key));

type Cell = { c: V3; s: number; off: V3 };

/** the 26 seats of the structure around a missing centre `center`, each of edge `s` */
const cellsAt = (center: V3, s: number): Cell[] => OFFS.map((o) => ({ c: [center[0] + o[0] * s, center[1] + o[1] * s, center[2] + o[2] * s], s, off: o }));

/** the faces of those seats not shared with another seat of the same structure, so the missing centre shows */
function faces(cells: Cell[]): V3[][] {
  const out: V3[][] = [];
  for (const cell of cells)
    for (const axis of [0, 1, 2])
      for (const dir of [-1, 1]) {
        const n = [...cell.off];
        n[axis] += dir;
        if (PRESENT.has(key(n))) continue;
        const [u, v] = [0, 1, 2].filter((k) => k !== axis);
        const h = cell.s / 2;
        const corner = (du: number, dv: number): V3 => {
          const q = [...cell.c];
          q[axis] += dir * h;
          q[u] += du * h;
          q[v] += dv * h;
          return q as unknown as V3;
        };
        out.push([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]);
      }
  return out;
}

type Where = { state: "outside" | "missing centre" | "seat"; off: number[] };

/** the boolean function at one depth: is `p` inside the structure around `center` (edge `s`), and in which seat */
function locate(p: V3, center: V3, s: number): Where {
  const q = [0, 1, 2].map((k) => (p[k] - center[k]) / s);
  if (q.some((x) => Math.abs(x) > 1.5)) return { state: "outside", off: [] };
  const off = q.map((x) => Math.max(-1, Math.min(1, Math.round(x))));
  return { state: off.every((x) => x === 0) ? "missing centre" : "seat", off };
}

const OPTIONS = OFFS.map(key);
const coord = (v: readonly number[]) => `(${v.join(", ")})`;
const ninth = (k: number) => (k === 0 ? "0" : `${k}/9`);

export function EightD() {
  const [focus1, setFocus1] = useState(true);
  const [focus2, setFocus2] = useState(false);
  const [seat1, setSeat1] = useState("1,1,1");
  const [seat2, setSeat2] = useState("1,1,1");
  const [probe, setProbe] = useState<[number, number, number]>([10, 6, 8]); // ninths from the missing centre

  const f1 = seat1.split(",").map(Number) as unknown as V3;
  const f2 = seat2.split(",").map(Number) as unknown as V3;
  const C0: V3 = [0, 0, 0];
  const C1: V3 = f1;
  const C2: V3 = [C1[0] + f2[0] / 3, C1[1] + f2[1] / 3, C1[2] + f2[2] / 3];
  const p: V3 = probe.map((k) => k / 9) as unknown as V3;

  const polys: Poly[] = [];
  const lines: Line[] = [];
  const labels: Label[] = [];

  for (const f of faces(cellsAt(C0, 1))) polys.push({ pts: f, fill: "#c9853f", opacity: focus1 ? 0.07 : 0.2, title: "depth 0 · the structure" });
  labels.push({ at: C0, text: "∅ (0, 0, 0)", color: "#1b1a18" });
  if (focus1) {
    for (const f of faces(cellsAt(C1, 1 / 3))) polys.push({ pts: f, fill: "#6d4a93", opacity: focus2 ? 0.12 : 0.3, title: `depth 1 · inside seat ${coord(f1)}` });
    labels.push({ at: [C1[0], C1[1], C1[2] + 0.62], text: `depth 1 · seat ${coord(f1)} · its ∅`, color: "#6d4a93" });
    if (focus2) {
      for (const f of faces(cellsAt(C2, 1 / 9))) polys.push({ pts: f, fill: "#3d6b39", opacity: 0.45, title: `depth 2 · inside ${coord(f1)}·${coord(f2)}` });
      labels.push({ at: [C2[0], C2[1], C2[2] - 0.3], text: `depth 2 · ${coord(f1)}·${coord(f2)}`, color: "#3d6b39" });
    }
  }

  // the inner cone, inwards: from the outer face to the missing centre — the object imagining itself
  polys.push(...cone([1.5, 0, 0], C0, 0.45, "#a23b2a", "inner cone · inwards · the object imagines itself", 0.2));
  labels.push({ at: [1.9, 0, -0.1], text: "inner cone · inwards", color: "#a23b2a" });

  // the probe, moving in (x, y, z) from the missing centre
  const m = 0.05;
  for (const [a, b] of [
    [0, 1],
    [0, 2],
    [1, 2],
  ]) {
    const corner = (da: number, db: number): V3 => {
      const q = [...p];
      q[a] += da * m;
      q[b] += db * m;
      return q as unknown as V3;
    };
    polys.push({ pts: [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)], fill: "#1b1a18", opacity: 0.9, title: "the probe" });
  }
  lines.push({ from: C0, to: p, color: "#1b1a18", dashed: true, title: "from the missing centre to the probe" });
  labels.push({ at: [p[0], p[1], p[2] + 0.18], text: `probe ${coord(probe.map(ninth))}`, color: "#1b1a18" });

  const d0 = locate(p, C0, 1);
  const d1 = locate(p, C1, 1 / 3);
  const d2 = locate(p, C2, 1 / 9);
  const readout = (depth: number, w: Where, focused: boolean, within: string) => (
    <li key={depth}>
      depth {depth}: focus <b className="mono">{String(focused)}</b>
      {focused && (
        <>
          {" "}
          · inside {within}: <b className="mono">{String(w.state === "seat")}</b> · {w.state === "seat" ? `seat ${coord(w.off)}` : w.state}
        </>
      )}
    </li>
  );

  const seatPick = (value: string, set: (v: string) => void) => (
    <select value={value} onChange={(e) => set(e.target.value)}>
      {OPTIONS.map((o) => (
        <option key={o} value={o}>
          ({o})
        </option>
      ))}
    </select>
  );

  return (
    <section className="tabbody">
      <div className="meta">
        <span>
          8D · the inside has coordinates · a boolean for the depth in focus · <b>an attempt to be corrected</b> · not walked · drag to turn the view
        </span>
      </div>
      <div className="split">
        <div className="panel">
          <div className="toolbar" style={{ display: "block" }}>
            <label className="small" style={{ marginRight: 12 }}>
              <input type="checkbox" checked={focus1} onChange={() => setFocus1((x) => !x)} /> focus depth 1, inside seat {seatPick(seat1, setSeat1)}
            </label>
            <label className="small">
              <input type="checkbox" checked={focus2} disabled={!focus1} onChange={() => setFocus2((x) => !x)} /> focus depth 2, inside seat {seatPick(seat2, setSeat2)}
            </label>
            <div style={{ marginTop: 6 }}>
              {(["x", "y", "z"] as const).map((n, i) => (
                <label key={n} className="small" style={{ marginRight: 10 }}>
                  {n}{" "}
                  <input type="range" min={-13} max={13} value={probe[i]} onChange={(e) => setProbe((q) => q.map((v, k) => (k === i ? Number(e.target.value) : v)) as [number, number, number])} />{" "}
                  <span className="mono">{ninth(probe[i])}</span>
                </label>
              ))}
            </div>
          </div>
          <Solid polys={polys} labels={labels} lines={lines} initial={[0.6, 0.4]} />
        </div>
        <div className="panel">
          <h2>The boolean, depth by depth</h2>
          <p className="mono small">probe {coord(probe.map(ninth))} from ∅</p>
          <ul className="small">
            {readout(0, d0, true, "the structure")}
            {readout(1, d1, focus1, `seat ${coord(f1)}`)}
            {readout(2, d2, focus1 && focus2, `${coord(f1)}·${coord(f2)}`)}
          </ul>
          <h2>My reading, by number</h2>
          <ol className="small">
            <li>The inner cone points inwards: the object imagines itself.</li>
            <li>That takes 8D: a boolean function saying which depth of the fractal is in focus.</li>
            <li>It is the interior grid of any defined structure, moving in (x, y, z) from the missing centre, so the inside has coordinates as well.</li>
            <li>With that, space unlocks: inside from outside, up from down, left from right.</li>
          </ol>
          <p className="muted small">
            Mine, to be corrected:
          </p>
          <ul className="muted small">
            <li>at every depth the structure is the same 3×3×3 with its centre missing, and a focused seat holds the next depth;</li>
            <li>coordinates are read from ∅ in both directions;</li>
            <li>the probe steps in ninths;</li>
            <li>each depth has its own "focus" boolean.</li>
          </ul>
          <p className="muted small">Positions are JS numbers (machine side).</p>
        </div>
      </div>
    </section>
  );
}
