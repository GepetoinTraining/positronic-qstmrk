import { useEffect, useState } from "react";
import type { LedgerRow } from "./api";
import { loadLedger } from "./api";
import type { Label, Poly, V3 } from "./Solid";
import { Solid } from "./Solid";

/**
 * The 5D tab — pgarcia's display, drawn to be corrected, not walked.
 *   (1) a pentagonal cylinder: five planed sides; the top base +, the free base −(B)
 *   (2) four sides carry a planed cube outward (the 4 planes); the fifth side is free (the hull that is left)
 *   (3) the interior of each side is a pyramid routed to the free base; 72° aperture at the centre
 *   (4) each cube holds the inverse pyramid: the interior pyramid mirrored in the side, its pointy part in the
 *       top — the free base's centre reflected through the side's plane, on the cube's + face
 *   (5) in relation to the pentagon the cube sides rotate +, −, +, −
 *   (6) the free side sits between a + side and a − side: one comes in as the numerator, the other as the
 *       denominator. The free side MUST be 1: they must be equal. When they are not, cycle to the other side.
 * Drawing numbers are description only.
 */

const deg = Math.PI / 180;
const r = 1; // apothem
const a = 2 * r * Math.tan(36 * deg); // a side's width; the cube on it is square, so the cylinder's height is a
const H = a;
const Rc = r / Math.cos(36 * deg);

const at = (angle: number, radius: number, z: number): V3 => [radius * Math.cos(angle), radius * Math.sin(angle), z];
const add = (v: V3, n: V3, k: number): V3 => [v[0] + n[0] * k, v[1] + n[1] * k, v[2] + n[2] * k];
const mid = (...vs: V3[]): V3 => [0, 1, 2].map((k) => vs.reduce((t, v) => t + v[k], 0) / vs.length) as unknown as V3;

type Show = { cylinder: boolean; cubes: boolean; pyramids: boolean; inverse: boolean; archive: boolean };
type Ring = { name: string; reading: string };

const FREE = 4; // the free side; its neighbours are side 4 (index 3) and side 1 (index 0)
const POS = "#c9853f";
const NEG = "#5f86a6";

/** rotation sign of cube side i as the pentagon sees it; `flipped` is the other side */
const signOf = (i: number, flipped: boolean) => ((i % 2 === 0) !== flipped ? "+" : "−");

function build(show: Show, flipped: boolean, num: Ring, den: Ring, balanced: boolean): { polys: Poly[]; labels: Label[] } {
  const polys: Poly[] = [];
  const labels: Label[] = [];
  const phi = (i: number) => (90 + 72 * i) * deg;
  const bottom = [0, 1, 2, 3, 4].map((i) => at(phi(i) - 36 * deg, Rc, 0));
  const top = bottom.map((v): V3 => [v[0], v[1], H]);
  const freeBase: V3 = [0, 0, 0];
  const numSide = [0, 3].find((i) => signOf(i, flipped) === "+")!;

  if (show.cylinder) {
    polys.push({ pts: top, fill: "#ecd5b8", opacity: 0.25, title: "top base +" });
    polys.push({ pts: bottom, fill: "#9fb9cc", opacity: 0.25, dashed: true, title: "free base −(B)" });
    labels.push({ at: [0, 0, H + 0.08], text: "+" }, { at: [0, 0, -0.14], text: "−(B) free base" });
  }

  for (let i = 0; i < 5; i++) {
    const j = (i + 1) % 5;
    const side: V3[] = [bottom[i], bottom[j], top[j], top[i]];
    // side i runs between the corners at phi(i) − 36° and phi(i) + 36°, so it faces out at phi(i)
    const out: V3 = [Math.cos(phi(i)), Math.sin(phi(i)), 0];
    const free = i === FREE;
    const sign = free ? "" : signOf(i, flipped);
    const tint = sign === "+" ? POS : NEG;
    if (show.cylinder) {
      polys.push({ pts: side, fill: free ? (balanced ? "#8fae8b" : "#e3b3a8") : "#e9c9a4", opacity: free ? 0.45 : 0.2, dashed: free, title: free ? "the free side" : `planed side ${i + 1}` });
      labels.push({
        at: add(mid(...side), out, free ? 0.2 : 0.12),
        text: free ? (balanced ? `free = 1` : `free ≠ 1 · ${num.name}/${den.name}`) : `side ${i + 1}`,
        color: free ? (balanced ? "#3d6b39" : "#a23b2a") : "#8a4a14",
      });
    }
    if (show.pyramids) {
      for (let e = 0; e < 4; e++) polys.push({ pts: [side[e], side[(e + 1) % 4], freeBase], fill: free ? "#c9b8de" : "#86a3b8", opacity: 0.3, title: `interior pyramid of side ${i + 1} → free base` });
    }
    if (free) {
      if (show.archive) {
        // the cube coming out of the final face: its top represents the 4 cubes, its bottom is the archive,
        // the ⁻¹ of their relation. Quarters sit toward the cubes they stand for: side[0] is the corner shared
        // with side 4, side[1] the corner shared with side 1; the outer row stands for the far sides 3 and 2.
        const cube = side.map((v) => add(v, out, a));
        const along: V3 = [side[1][0] - side[0][0], side[1][1] - side[0][1], 0];
        const P = (s: number, t: number, z: number): V3 => [side[0][0] + along[0] * s + out[0] * a * t, side[0][1] + along[1] * s + out[1] * a * t, z];
        polys.push({ pts: [cube[0], cube[1], cube[2], cube[3]], fill: "#b9b3a8", opacity: 0.15, title: "archive cube: outer face" });
        polys.push({ pts: [side[0], side[3], cube[3], cube[0]], fill: "#b9b3a8", opacity: 0.12, title: "archive cube: side" });
        polys.push({ pts: [side[1], side[2], cube[2], cube[1]], fill: "#b9b3a8", opacity: 0.12, title: "archive cube: side" });
        const quarters = [
          { s: 0, t: 0, of: 3 },
          { s: 0.5, t: 0, of: 0 },
          { s: 0, t: 0.5, of: 2 },
          { s: 0.5, t: 0.5, of: 1 },
        ];
        for (const q of quarters) {
          const sg = signOf(q.of, flipped);
          const inv = sg === "+" ? "−" : "+";
          const quad = (z: number) => [P(q.s, q.t, z), P(q.s + 0.5, q.t, z), P(q.s + 0.5, q.t + 0.5, z), P(q.s, q.t + 0.5, z)];
          polys.push({ pts: quad(H), fill: sg === "+" ? POS : NEG, opacity: 0.6, title: `top · stands for cube ${q.of + 1} (${sg})` });
          polys.push({ pts: quad(0), fill: inv === "+" ? POS : NEG, opacity: 0.35, dashed: true, title: `archive · cube ${q.of + 1} read ⁻¹ (${sg}⁻¹)` });
          labels.push({ at: P(q.s + 0.25, q.t + 0.25, H + 0.03), text: `${q.of + 1}${sg}`, color: sg === "+" ? "#8a4a14" : "#34607f", facing: "up" });
          labels.push({ at: P(q.s + 0.25, q.t + 0.25, -0.08), text: `${q.of + 1}${sg}⁻¹`, color: "#6d4a93", facing: "down" });
        }
        labels.push({ at: P(0.5, 1.15, H + 0.1), text: `top · the 4 cubes · ${num.name}/${den.name}`, facing: "up" });
        labels.push({ at: P(0.5, 1.15, -0.1), text: `archive · (${num.name}/${den.name})⁻¹ = ${den.name}/${num.name}`, color: "#6d4a93", facing: "down" });
      }
      continue;
    }
    const cube = side.map((v) => add(v, out, a));
    if (show.cubes) {
      const role = i === numSide ? " · numerator" : i === 3 || i === 0 ? " · denominator" : "";
      polys.push({ pts: [cube[0], cube[1], cube[2], cube[3]], fill: tint, opacity: 0.35, title: `cube ${i + 1}: outer face · rotation ${sign}${role}` });
      polys.push({ pts: [side[0], side[1], cube[1], cube[0]], fill: tint, opacity: 0.2, title: `cube ${i + 1}: bottom −` });
      polys.push({ pts: [side[3], side[2], cube[2], cube[3]], fill: tint, opacity: 0.2, title: `cube ${i + 1}: top +` });
      polys.push({ pts: [side[0], side[3], cube[3], cube[0]], fill: tint, opacity: 0.15, title: `cube ${i + 1}: side` });
      polys.push({ pts: [side[1], side[2], cube[2], cube[1]], fill: tint, opacity: 0.15, title: `cube ${i + 1}: side` });
      const face = mid(cube[0], cube[1], cube[2], cube[3]);
      const inText = i === numSide ? ` → ${num.name} (numerator)` : i === 0 || i === 3 ? ` → ${den.name} (denominator)` : "";
      labels.push({ at: add(face, out, 0.15), text: `${sign === "+" ? "↻" : "↺"} ${sign}${inText}`, color: sign === "+" ? "#8a4a14" : "#34607f" });
    }
    if (show.inverse) {
      // the interior pyramid mirrored in its side, its pointy part in the top: the free base's centre reflected
      // through the side's plane, carried up to the + portion
      const apex: V3 = add(add(freeBase, out, 2 * r), [0, 0, 1], H);
      for (let e = 0; e < 4; e++) polys.push({ pts: [side[e], side[(e + 1) % 4], apex], fill: "#6d4a93", opacity: 0.25, title: `inverse pyramid in cube ${i + 1}` });
    }
  }
  return { polys, labels };
}

export function FiveD() {
  const [show, setShow] = useState<Show>({ cylinder: true, cubes: true, pyramids: false, inverse: false, archive: true });
  const [rings, setRings] = useState<Ring[]>([{ name: "2", reading: "2" }]);
  const [numName, setNum] = useState("2");
  const [denName, setDen] = useState("3");
  const [flipped, setFlipped] = useState(false);

  useEffect(() => {
    loadLedger().then((rows: LedgerRow[]) => {
      const primes = rows.filter((l) => l.status === "prime").map((l) => ({ name: l.name, reading: l.reading }));
      setRings([{ name: "2", reading: "2" }, ...primes]);
    });
  }, []);

  const find = (n: string) => rings.find((x) => x.name === n) ?? { name: n, reading: "" };
  // what comes in on the + neighbour is the numerator, on the − neighbour the denominator; cycling swaps which side is +
  const [num, den] = flipped ? [find(denName), find(numName)] : [find(numName), find(denName)];
  const balanced = num.reading === den.reading; // the two runs' decade readings, made by pairing in the construction
  const { polys, labels } = build(show, flipped, num, den, balanced);
  const toggle = (k: keyof Show) => setShow((s) => ({ ...s, [k]: !s[k] }));
  const pick = (value: string, set: (v: string) => void, label: string) => (
    <label className="small" style={{ marginRight: 12 }}>
      {label}{" "}
      <select value={value} onChange={(e) => set(e.target.value)}>
        {rings.map((x) => (
          <option key={x.name} value={x.name}>
            {x.name}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <section className="tabbody">
      <div className="meta">
        <span>
          5D · the accumulator · 4 planes and the 5th, the abscissa · <b>an attempt to be corrected</b> · not walked · drag to turn
        </span>
      </div>
      <div className="split">
        <div className="panel">
          <div className="toolbar">
            {(["cylinder", "cubes", "pyramids", "inverse", "archive"] as const).map((k) => (
              <label key={k} className="small" style={{ marginRight: 12 }}>
                <input type="checkbox" checked={show[k]} onChange={() => toggle(k)} />{" "}
                {k === "inverse" ? "inverse pyramids" : k === "pyramids" ? "interior pyramids" : k === "archive" ? "final-face cube" : k}
              </label>
            ))}
          </div>
          <div className="toolbar">
            {pick(numName, setNum, "comes in on side 1")}
            {pick(denName, setDen, "comes in on side 4")}
            <button className={balanced ? "" : "on"} disabled={balanced} onClick={() => setFlipped((f) => !f)}>
              cycle to the other side
            </button>
            <span className={`small ${balanced ? "" : "muted"}`} style={{ marginLeft: 10 }}>
              free side: <b className="mono">{num.name}/{den.name}</b> {balanced ? "= 1 ✓" : "≠ 1 → cycle"} · reading {flipped ? "from the other side" : "as drawn"}
            </span>
          </div>
          <Solid polys={polys} labels={labels} />
        </div>
        <div className="panel">
          <h2>My reading, by number</h2>
          <ol className="small">
            <li>A pentagonal cylinder, five planed sides. Top base <b>+</b>, free base <b>−(B)</b>.</li>
            <li>Four sides carry a planed cube outward, the 4 planes {"{(+−)xyz}"}. The fifth side is free.</li>
            <li>Inside each side, a pyramid: base the side, apex routed to the free base. Its aperture at the centre is 72°. Not yet placed: your 144°, 144°.</li>
            <li>
              In each cube, the inverse pyramid is the interior pyramid mirrored in the side, with its point at the top. It shares the same base, and its apex is the free base's centre reflected through the side, lifted to the cube's + face. The free side has no cube, so it has no mirror drawn.
            </li>
            <li>
              In relation to the pentagon, the cube sides rotate <b>+, −, +, −</b>: the pentagon sees one as positive, the next as negative.
            </li>
            <li>
              The free side sits between side 1 and side 4, one + and one −. What comes in on the + side is the numerator; what comes in on the − side is the denominator. The free side must be <b>1</b>, so they must be equal. If they are not, cycle to the other side: the signs swap, so numerator and denominator trade places.
            </li>
            <li>
              From the final (free) face comes a cube. Its <b>top</b> represents the 4 cubes: one quarter per cube, with its sign, and the relation {`num/den`}. Its <b>bottom</b> is the archive, the ⁻¹ of that relation: each quarter read ⁻¹, and the relation inverted.
            </li>
          </ol>
          <p className="small">
            Mine, to be corrected:
          </p>
          <ul className="small">
            <li>which neighbour counts as numerator (I chose +);</li>
            <li>that cycling means swapping every sign;</li>
            <li>where each quarter sits: sides 4 and 1 on the inner row, 3 and 2 on the outer row, each toward its cube;</li>
            <li>that ⁻¹ turns each quarter's sign over;</li>
            <li>that the final-face cube holds no inverse pyramid.</li>
          </ul>
          <p className="small">
            Reconciled (pgarcia): n/n read as the count is the 1 the free side must be; read as the integer it is n. 2/2 is the count, and 2 the integer: a 2 on each side.
          </p>
          <p className="small">
            Flagged: the free side where 1 must stand may be the same place as the oscillator's place in the cells, where 1 was drawn. When the two sides are not equal, it cycles back and forth.
          </p>
          <p className="muted small">Equality is read from the runs' decade readings, made by pairing in the construction. Numbers used to draw are description only.</p>
        </div>
      </div>
    </section>
  );
}
