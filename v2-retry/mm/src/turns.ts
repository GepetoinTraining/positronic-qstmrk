/**
 * A turn: a signed permutation of the coordinate slots. `perm[a]` is the container axis coordinate `a` lands on;
 * `sign[a] = -1` means that coordinate is read inverted — from the far seat of the container's axis instead of the
 * first (the q/p side, the "−" direction). No arithmetic on seats: a turn relabels slots and flips a reading
 * direction, nothing else. The identity is today's walk (order matters; every direction +).
 *
 * The 24 Hurwitz units ±1, ±i, ±j, ±k, ½(±1±i±j±k) written out as turns of three slots — a table, never computed
 * at runtime. [B] the labelling: a unit with positive real part (or, for ±i ±j ±k, positive imaginary part) is its
 * rotation v ↦ q v q̄; its negative is that rotation with every sign flipped, so −1 is the all-inverted turn.
 * As turns, composition closes on these 24 (checked offline); c = ½(1+i+j+k) is the axis-cycle x→y→z, and the
 * turn c² (z→y→x, all +) is the row labelled ½(1−i−j−k) here. Floatland says q³ = −1 for the unit; as turns c³ = 1.
 */
export type Turn = { readonly label: string; readonly perm: readonly number[]; readonly sign: readonly (1 | -1)[] };

const P = 1 as const;
const N = -1 as const;

export const TURNS: readonly Turn[] = [
  { label: "1", perm: [0, 1, 2], sign: [P, P, P] }, //         identity — today's walk
  { label: "-1", perm: [0, 1, 2], sign: [N, N, N] }, //        all inverted — the "−" direction of the whole walk
  { label: "i", perm: [0, 1, 2], sign: [P, N, N] }, //         half-turn about x
  { label: "-i", perm: [0, 1, 2], sign: [N, P, P] },
  { label: "j", perm: [0, 1, 2], sign: [N, P, N] }, //         half-turn about y
  { label: "-j", perm: [0, 1, 2], sign: [P, N, P] },
  { label: "k", perm: [0, 1, 2], sign: [N, N, P] }, //         half-turn about z
  { label: "-k", perm: [0, 1, 2], sign: [P, P, N] },
  { label: "1+i+j+k", perm: [1, 2, 0], sign: [P, P, P] }, //   c = ½(1+i+j+k): x→y→z, the axis-cycle
  { label: "1+i+j-k", perm: [2, 0, 1], sign: [N, P, N] },
  { label: "1+i-j+k", perm: [2, 0, 1], sign: [P, N, N] },
  { label: "1+i-j-k", perm: [1, 2, 0], sign: [N, P, N] },
  { label: "1-i+j+k", perm: [2, 0, 1], sign: [N, N, P] },
  { label: "1-i+j-k", perm: [1, 2, 0], sign: [N, N, P] },
  { label: "1-i-j+k", perm: [1, 2, 0], sign: [P, N, N] },
  { label: "1-i-j-k", perm: [2, 0, 1], sign: [P, P, P] }, //   c²: z→y→x, the axis-cycle the other way
  { label: "-1+i+j+k", perm: [2, 0, 1], sign: [N, N, N] },
  { label: "-1+i+j-k", perm: [1, 2, 0], sign: [N, P, P] },
  { label: "-1+i-j+k", perm: [1, 2, 0], sign: [P, P, N] },
  { label: "-1+i-j-k", perm: [2, 0, 1], sign: [P, P, N] },
  { label: "-1-i+j+k", perm: [1, 2, 0], sign: [P, N, P] },
  { label: "-1-i+j-k", perm: [2, 0, 1], sign: [N, P, P] },
  { label: "-1-i-j+k", perm: [2, 0, 1], sign: [P, N, P] },
  { label: "-1-i-j-k", perm: [1, 2, 0], sign: [N, N, N] },
];

const ALIAS: Record<string, string> = { c: "1+i+j+k", c2: "1-i-j-k", "c²": "1-i-j-k" };

/** the identity turn of a dimension: every slot to itself, every direction + */
export function identity(dimension: number): Turn {
  return { label: "1", perm: Array.from({ length: dimension }, (_, a) => a), sign: Array.from({ length: dimension }, () => P) };
}

/**
 * A three-slot turn read in fewer slots: the first `dimension` slots must land among themselves (a 3-cycle has no
 * 2D reading). In 2D only the signs act; `k` and `-1` read alike there, as do `i` and `-j`, `j` and `-i`.
 */
export function restrict(t: Turn, dimension: number): Turn {
  if (dimension > t.perm.length) {
    // [B] past three slots a turn is padded: the further slots stay put and read +
    const perm = [...t.perm, ...Array.from({ length: dimension - t.perm.length }, (_, a) => t.perm.length + a)];
    const sign = [...t.sign, ...Array.from({ length: dimension - t.sign.length }, () => P)];
    return { label: t.label, perm, sign };
  }
  const perm = t.perm.slice(0, dimension);
  if (perm.some((b) => b >= dimension)) throw new Error(`refused: turn ${t.label} does not close on the first ${dimension} slots`);
  return { label: t.label, perm, sign: t.sign.slice(0, dimension) };
}

/** the turn of a label (aliases c, c2), restricted to the walk's dimension */
export function turnOf(label: string, dimension: number): Turn {
  const name = ALIAS[label] ?? label;
  const t = TURNS.find((u) => u.label === name);
  if (!t) throw new Error(`refused: no turn labelled ${label}`);
  return restrict({ ...t, label }, dimension);
}

export const isIdentity = (t: Turn) => t.perm.every((b, a) => b === a) && t.sign.every((s) => s === P);

export const turnText = (t: Turn) => `${t.label}: perm [${t.perm.join(",")}] sign [${t.sign.map((s) => (s === P ? "+" : "−")).join(",")}]`;

/**
 * A reading of a coordinate from an offset along its ring, wrapping round — the third-turn and the fifth-turn of an
 * axis beside the half-turn (sign −) above. `3` offsets by the run of the ring 3 (always readable when the ring
 * fits). `1/3` offsets by one of three equal runs the axis is dealt into by pairing; when the dealing comes out
 * unequal the reading breaks — the walk says so and reads that cycle forward. [B] both.
 */
export type Reading = { readonly label: string; readonly kind: "ring" | "part"; readonly r: number };

export function readingOf(label: string): Reading | null {
  const part = /^1\/(\d+)$/.exec(label);
  if (part) return { label, kind: "part", r: Number(part[1]) };
  if (/^\d+$/.test(label) && label !== "1") return { label, kind: "ring", r: Number(label) };
  return null;
}

/**
 * The offset, in seats, of a reading on an axis of `n` seats (machine side: a tally, as seat addresses are).
 * A ring reading offsets by its run, wrapped on the axis. A part reading deals the axis round-robin into r runs
 * and offsets by the first; `null` when the runs are unequal (the reading breaks on this axis).
 */
export function offsetOf(reading: Reading, n: number): number | null {
  if (reading.kind === "ring") return reading.r % n;
  const groups: number[] = Array.from({ length: reading.r }, () => 0);
  for (let s = 0; s < n; s++) groups[s % reading.r]++;
  return groups.every((g) => g === groups[0]) ? groups[0] : null;
}
