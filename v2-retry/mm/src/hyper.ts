import { pairsInto, samePairing } from "./bead.ts";
import { decadeName, runOfReading } from "./decade.ts";

/**
 * Walking a container MM[R, R, …, R] of any dimension without its seats (pgarcia: never expose the imaginary ones,
 * don't count, oscillate and multiply, then LUT and factor, keep only what is observed).
 *
 * The limit of a container is the cover of everything held short of the whole. What the cover leaves out is one
 * corner: the seats past f on every axis, where f is the largest ring on the rail that pairs into R short of R (the
 * count reading of the bit, 1, when none does). That corner is a box of w on every axis, w being R's run left over
 * once f is paired off it. The limit is then D slabs, each a box — MM[R ×i, f, w ×(D−1−i)] — multiplied, and the
 * slabs accumulated in decades. No seat is listed and nothing is subtracted.
 *
 * Machine side, flagged: products and the accumulator run on BigInt, and BigInt's decimal string is its decade
 * reading. The LUT of primes is made by multiplying names into crossings: what no crossing reaches is prime.
 */

export type HyperLimit = {
  ring: string; // R
  dimension: number; // D
  fits: string; // f
  corner: string; // w
  whole: bigint; // R^D
  missing: bigint; // w^D, the corner the limit leaves out
  limit: bigint; // the D slabs accumulated
};

const power = (base: bigint, times: number) => {
  let out = 1n;
  for (let i = 0; i < times; i++) out *= base;
  return out;
};

/** the largest ring of `rail` (ascending) that pairs into R short of R; "1" when none does */
export function fitsShort(ring: string, rail: readonly string[]): string {
  const R = runOfReading(ring);
  let best = "1";
  for (const p of rail) {
    const run = runOfReading(p);
    if (pairsInto(run, R) && !samePairing(run, R)) best = p;
  }
  return best;
}

/** R's run left over once f is paired off it */
export function leftOver(ring: string, fits: string): string {
  const R = runOfReading(ring)[Symbol.iterator]();
  for (const _ of runOfReading(fits)) R.next();
  const rest: unknown[] = [];
  for (let x = R.next(); !x.done; x = R.next()) rest.push(x.value);
  return decadeName(rest);
}

export function hyperLimit(ring: string, dimension: number, rail: readonly string[]): HyperLimit {
  const fits = fitsShort(ring, rail);
  const corner = leftOver(ring, fits);
  const [R, f, w] = [BigInt(ring), BigInt(fits), BigInt(corner)];
  let limit = 0n;
  for (let i = 0; i < dimension; i++) limit += power(R, i) * f * power(w, dimension - 1 - i); // one slab, a box
  return { ring, dimension, fits, corner, whole: power(R, dimension), missing: power(w, dimension), limit };
}

/** the LUT: every name up to `top` that no crossing of two smaller names reaches (crossings marked by multiplying) */
export function primeLUT(top: number): number[] {
  const crossed = new Uint8Array(top + 1);
  const primes: number[] = [];
  for (let p = 2; p <= top; p++) {
    if (crossed[p]) continue;
    primes.push(p);
    for (let m = p * p; m <= top; m += p) crossed[m] = 1;
  }
  return primes;
}

/** factor a name against the LUT; a part left over that no LUT prime divides is itself prime (it lies past the LUT's square) */
export function factor(n: bigint, lut: readonly number[]): [bigint, number][] {
  const out: [bigint, number][] = [];
  let rest = n;
  for (const q of lut) {
    const p = BigInt(q);
    if (p * p > rest) break;
    let e = 0;
    while (rest % p === 0n) {
      rest /= p;
      e++;
    }
    if (e) out.push([p, e]);
  }
  if (rest > 1n) out.push([rest, 1]);
  return out;
}

export const factorText = (fs: readonly [bigint, number][]) => (fs.length === 1 && fs[0][1] === 1 ? "prime" : fs.map(([p, e]) => (e > 1 ? `${p}^${e}` : `${p}`)).join(" × "));
