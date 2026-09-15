/**
 * The first byte type (pgarcia): a cube, 3³ = 27 cells each on or off, and 2 — the sign, the side the cube is on.
 * A cell is its (x, y, z), each 0, 1, 2. One side is 27 bits in a machine word; a cube of one weight has exactly one
 * cell on; a cube of many is every cell anything turns on, still one word per side.
 * [B] bit order: cell (x, y, z) is bit x·9 + y·3 + z; the side is bit 27 of a single-weight word.
 */

export const CELLS = 27;
export const ALL = (1 << CELLS) - 1;
export const SIDE = 1 << CELLS;

export const cell = (x: number, y: number, z: number) => 1 << (x * 9 + y * 3 + z);

/** the nine cells of a face: every cell whose `axis` coordinate is k */
export function face(axis: "x" | "y" | "z", k: number): number {
  let m = 0;
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) m |= axis === "x" ? cell(k, a, b) : axis === "y" ? cell(a, k, b) : cell(a, b, k);
  return m;
}

/** the lit cells of a side, as (x, y, z) */
export function lit(cells: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) for (let z = 0; z < 3; z++) if (cells & cell(x, y, z)) out.push([x, y, z]);
  return out;
}

/** a side drawn as three x-layers, each three y-rows of z: 100.000.000/000.010.000/000.000.001 */
export function draw(cells: number): string {
  const layers: string[] = [];
  for (let x = 0; x < 3; x++) {
    const rows: string[] = [];
    for (let y = 0; y < 3; y++) {
      let r = "";
      for (let z = 0; z < 3; z++) r += cells & cell(x, y, z) ? "1" : "0";
      rows.push(r);
    }
    layers.push(rows.join("."));
  }
  return layers.join("/");
}

/** e → its single-weight word: the side, and the one cell at the distance from the centre */
export function wordOfE(e: number, centre: number): number {
  const minus = e < centre;
  let d = minus ? centre - e : e - centre;
  const z = d % 3;
  d = (d - z) / 3;
  const y = d % 3;
  const x = (d - y) / 3;
  if (x > 2) throw new Error(`e ${e} is out of the cube from centre ${centre}`);
  return (minus ? SIDE : 0) | cell(x, y, z);
}

/** a single-weight word → e; throws unless exactly one cell is on */
export function eOfWord(word: number, centre: number): number {
  const on = lit(word & ALL);
  if (on.length !== 1 || word >>> 28 !== 0) throw new Error(`word ${word} is not one cell on one side`);
  const [x, y, z] = on[0];
  const d = (x * 3 + y) * 3 + z;
  return word & SIDE ? centre - d : centre + d;
}
