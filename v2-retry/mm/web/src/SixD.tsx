import { useState } from "react";
import type { Label, Line, Poly, V3 } from "./Solid";
import { Solid } from "./Solid";

/**
 * The 6D tab: MM[(a,b),(c,d),(e,f)] — every coordinate of 3D two-sided. The container's three planes, each a
 * 4 (MM[2,2], two self-referencing), read two ways: the showing faces meet at the origin corner, the hulled faces
 * (2⁻¹) at the far corner. No signs.
 * Interior (tagged, to show): two 3D objects, each with the coordinates that bound it and its volume; the free
 * space; and rays between the faces of the two objects that can see each other, with their distance.
 * Declared only. Volumes and distances are tallies of seats and steps for the reader (shadow): counting belongs
 * to space, 9D.
 */

const L = 8; // container seats per coordinate, for the drawing

const PLANES = [
  { axis: 0, name: "plane of the 2nd and 3rd coordinates", color: "#86a3b8" },
  { axis: 1, name: "plane of the 1st and 3rd coordinates", color: "#c49a6c" },
  { axis: 2, name: "plane of the 1st and 2nd coordinates", color: "#8fae8b" },
];

type Box = { name: string; lo: V3; hi: V3; color: string };

const OBJECTS: Box[] = [
  { name: "object 1", lo: [1, 1, 1], hi: [3, 4, 4], color: "#c9853f" },
  { name: "object 2", lo: [5, 2, 2], hi: [7, 6, 5], color: "#6d4a93" },
];

const set = (axis: number, level: number, u: number, v: number): V3 => {
  const others = [0, 1, 2].filter((k) => k !== axis);
  const p = [0, 0, 0];
  p[axis] = level;
  p[others[0]] = u;
  p[others[1]] = v;
  return p as unknown as V3;
};

/** the four seats of a container face: the plane at `level` on `axis`, split two by two */
function face(axis: number, level: number): V3[][] {
  const h = L / 2;
  const seats: V3[][] = [];
  for (const u of [0, h]) for (const v of [0, h]) seats.push([set(axis, level, u, v), set(axis, level, u + h, v), set(axis, level, u + h, v + h), set(axis, level, u, v + h)]);
  return seats;
}

function boxFaces(b: Box): V3[][] {
  const out: V3[][] = [];
  for (const axis of [0, 1, 2]) {
    const [p, q] = [0, 1, 2].filter((k) => k !== axis);
    for (const level of [b.lo[axis], b.hi[axis]])
      out.push([set(axis, level, b.lo[p], b.lo[q]), set(axis, level, b.hi[p], b.lo[q]), set(axis, level, b.hi[p], b.hi[q]), set(axis, level, b.lo[p], b.hi[q])]);
  }
  return out;
}

const extent = (b: Box) => [0, 1, 2].map((k) => b.hi[k] - b.lo[k]);
const volume = (b: Box) => extent(b).reduce((t, e) => t * e, 1);
const corner = (v: V3) => `(${v.join(",")})`;

type Facing = { axis: number; near: Box; far: Box; from: number; to: number; rect: [number, number, number, number] };

/** faces that can see each other: apart on one coordinate, overlapping on the other two */
function facings(a: Box, b: Box): Facing[] {
  const out: Facing[] = [];
  for (const axis of [0, 1, 2]) {
    const [p, q] = [0, 1, 2].filter((k) => k !== axis);
    const lo1 = Math.max(a.lo[p], b.lo[p]);
    const hi1 = Math.min(a.hi[p], b.hi[p]);
    const lo2 = Math.max(a.lo[q], b.lo[q]);
    const hi2 = Math.min(a.hi[q], b.hi[q]);
    if (lo1 >= hi1 || lo2 >= hi2) continue;
    if (a.hi[axis] <= b.lo[axis]) out.push({ axis, near: a, far: b, from: a.hi[axis], to: b.lo[axis], rect: [lo1, hi1, lo2, hi2] });
    else if (b.hi[axis] <= a.lo[axis]) out.push({ axis, near: b, far: a, from: b.hi[axis], to: a.lo[axis], rect: [lo1, hi1, lo2, hi2] });
  }
  return out;
}

const AXIS = ["1st", "2nd", "3rd"];

export function SixD() {
  const [side, setSide] = useState<"both" | "showing" | "hulled">("both");
  const [interior, setInterior] = useState(true);
  const polys: Poly[] = [];
  const labels: Label[] = [];
  const lines: Line[] = [];

  for (const p of PLANES) {
    if (side !== "hulled") for (const s of face(p.axis, 0)) polys.push({ pts: s, fill: p.color, opacity: interior ? 0.16 : 0.75, title: `${p.name} · showing` });
    if (side !== "showing") for (const s of face(p.axis, L)) polys.push({ pts: s, fill: p.color, opacity: interior ? 0.06 : 0.18, dashed: true, title: `${p.name} · hulled 2⁻¹` });
  }
  labels.push({ at: [0, 0, 0], text: "origin · showing faces meet" }, { at: [L, L, L], text: "far corner · hulled faces meet", color: "#6d4a93" });

  const [o1, o2] = OBJECTS;
  const seen = facings(o1, o2);
  const free = L * L * L - volume(o1) - volume(o2);

  if (interior) {
    for (const b of OBJECTS) {
      for (const f of boxFaces(b)) polys.push({ pts: f, fill: b.color, opacity: 0.4, title: `${b.name} · ${corner(b.lo)}–${corner(b.hi)} · volume ${volume(b)}` });
      labels.push({ at: [(b.lo[0] + b.hi[0]) / 2, (b.lo[1] + b.hi[1]) / 2, b.hi[2] + 0.5], text: `${b.name} · ${corner(b.lo)}–${corner(b.hi)} · volume ${volume(b)}`, color: b.color });
    }
    for (const f of seen) {
      const [lo1, hi1, lo2, hi2] = f.rect;
      // the parts of the two faces that see each other
      for (const level of [f.from, f.to]) polys.push({ pts: [set(f.axis, level, lo1, lo2), set(f.axis, level, hi1, lo2), set(f.axis, level, hi1, hi2), set(f.axis, level, lo1, hi2)], fill: "#3d6b39", opacity: 0.45, title: "faces that see each other" });
      // one ray per seat of the overlap, face to face
      for (let u = lo1 + 0.5; u < hi1; u++) for (let v = lo2 + 0.5; v < hi2; v++) lines.push({ from: set(f.axis, f.from, u, v), to: set(f.axis, f.to, u, v), title: `ray along the ${AXIS[f.axis]} coordinate · distance ${f.to - f.from}` });
      labels.push({ at: set(f.axis, (f.from + f.to) / 2, hi1, hi2), text: `distance ${f.to - f.from} (${AXIS[f.axis]})`, color: "#3d6b39" });
    }
    labels.push({ at: [L / 2, L, L + 0.6], text: `free space · volume ${free} of ${L * L * L}`, color: "#1b1a18" });
  }

  return (
    <section className="tabbody">
      <div className="meta">
        <span>
          6D <b className="mono">MM[(a,b),(c,d),(e,f)]</b> · every coordinate two-sided · the three planes of 3D, each read two ways · no signs · not walked · drag to turn
        </span>
      </div>
      <div className="split">
        <div className="panel">
          <div className="toolbar">
            <div className="seg" role="group" aria-label="side">
              {(["both", "showing", "hulled"] as const).map((s) => (
                <button key={s} className={s === side ? "on" : ""} onClick={() => setSide(s)}>
                  {s === "hulled" ? "hulled 2⁻¹" : s}
                </button>
              ))}
            </div>
            <label className="small" style={{ marginLeft: 12 }}>
              <input type="checkbox" checked={interior} onChange={() => setInterior((x) => !x)} /> interior
            </label>
          </div>
          <Solid polys={polys} labels={labels} lines={lines} initial={[0.75, 0.5]} />
        </div>
        <div className="panel">
          <h2>Interior · three things</h2>
          <ol className="small">
            {OBJECTS.map((b) => (
              <li key={b.name}>
                <i className="dot" style={{ background: b.color }} /> <b>{b.name}</b>: bounded by {corner(b.lo)} and {corner(b.hi)}. Along the coordinates it runs {AXIS.map((ax, k) => `${ax} ${b.lo[k]}→${b.hi[k]}`).join(", ")}. Volume {volume(b)}.
              </li>
            ))}
            <li>
              <b>Free space</b>: the container less the two objects, volume {free} of {L * L * L}. Faces that see each other are apart on one coordinate and overlap on the other two:{" "}
              {seen.length
                ? seen.map((f) => `${f.near.name} at ${AXIS[f.axis]} ${f.from} sees ${f.far.name} at ${AXIS[f.axis]} ${f.to}, distance ${f.to - f.from}, one ray per seat of the overlap`).join("; ")
                : "none here"}
              .
            </li>
          </ol>
          <h2>The container</h2>
          <ul className="small">
            {PLANES.map((p) => (
              <li key={p.axis}>
                <i className="dot" style={{ background: p.color }} /> {p.name}: a 4 (MM[2,2]), solid where it shows, dashed where it is hulled
              </li>
            ))}
          </ul>
          <p className="small">
            Three planes, each read two ways: 6 = 2×3, the first imaginary whole, i×real of 3/2 · 2/3. Measured [M]: in 3D cycle 1, O5 (6) is two showing planes joined along an edge, at three addresses.
          </p>
          <p className="muted small">
            Flagged: the objects, the container size and their placements are mine, chosen to show the three things. Volumes and distances are tallies (shadow); counting step distance belongs to space, 9D. Calling the far-corner faces "hulled" is my reading of 2⁻¹ on a cube.
          </p>
        </div>
      </div>
    </section>
  );
}
