/** A bead: one mark. Nothing here is counted; runs are compared by pairing. */
export type Bead = { readonly kind: "bead" };

export const bead: Bead = { kind: "bead" };

/** Two runs pair out together: the same, without reading either as a quantity. */
export function samePairing(a: readonly unknown[], b: readonly unknown[]): boolean {
  const x = a[Symbol.iterator]();
  const y = b[Symbol.iterator]();
  for (;;) {
    const p = x.next();
    const q = y.next();
    if (p.done && q.done) return true;
    if (p.done || q.done) return false;
  }
}

/** Every element of `a` pairs with an element of `b` before `b` runs out: `a` pairs into `b`. */
export function pairsInto(a: readonly unknown[], b: readonly unknown[]): boolean {
  const x = a[Symbol.iterator]();
  const y = b[Symbol.iterator]();
  for (;;) {
    if (x.next().done) return true;
    if (y.next().done) return false;
  }
}

/** A run has a first element. */
export function hasFirst(a: readonly unknown[]): boolean {
  return !a[Symbol.iterator]().next().done;
}

/** A run has a second element. */
export function hasSecond(a: readonly unknown[]): boolean {
  const x = a[Symbol.iterator]();
  x.next();
  return !x.next().done;
}
