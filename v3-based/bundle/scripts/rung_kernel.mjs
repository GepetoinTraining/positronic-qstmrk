// Viscous CLM rung, typed kernel. State: BigInt64Array, fixed point Q20. No unbounded container.
// Tables built cold, before the loop: viscous keep factor per harmonic, as an integer in Q20.
// Hot loop: integer multiply, add, shift. Every dropped bit is priced into a receipt.
const K = 32, Q = 20n, ONE = 1n << Q;
const DT_SHIFT = 6n;                       // dt = 1/64
function run(nuShift /* nu = 2^-nuShift, or null for inviscid */, tmaxSteps) {
  // cold tables
  const keep = new BigInt64Array(K + 1);   // (1 - nu*dt*k^2) in Q20
  for (let k = 0; k <= K; k++) {
    const kk = BigInt(k * k);
    keep[k] = nuShift === null ? ONE : ONE - ((kk << Q) >> (BigInt(nuShift) + DT_SHIFT));
  }
  const b = new BigInt64Array(K + 1), nb = new BigInt64Array(K + 1);
  b[1] = ONE;
  let receipt = 0n, crossed = null;
  const band = [K / 4, 3 * K / 4];
  const t0 = performance.now();
  for (let n = 1; n <= tmaxSteps; n++) {
    for (let k = 1; k <= K; k++) {
      let s = 0n;
      for (let j = 1; j < k; j++) s += b[j] * b[k - j];          // Q40
      const nl = s >> (Q + 1n + DT_SHIFT);                       // /2, *dt, back to Q20
      receipt += s - (nl << (Q + 1n + DT_SHIFT));                // dropped low bits, priced
      const d = b[k] * keep[k];                                  // Q40
      const dk = d >> Q;
      receipt += d - (dk << Q);
      nb[k] = BigInt.asIntN(64, dk + nl);
      if (nb[k] !== dk + nl) throw new Error(`overflow at k=${k}, n=${n}`);
    }
    b.set(nb);
    if (crossed === null && b[band[0]] > 0n) {
      let all = true;
      for (let k = band[0]; k < band[1]; k++) if (b[k + 1] < b[k]) { all = false; break; }
      if (all) crossed = n;
    }
  }
  const ms = performance.now() - t0;
  // rho at the end, read cold as a rational pair (no division in the loop)
  let lo = null, hi = null;
  for (let k = band[0]; k < band[1]; k++) {
    const r = [b[k + 1], b[k]];
    if (lo === null || r[0] * lo[1] < lo[0] * r[1]) lo = r;
    if (hi === null || r[0] * hi[1] > hi[0] * r[1]) hi = r;
  }
  const f = r => (Number((r[0] * 1000000n) / r[1]) / 1e6).toFixed(6); // print only
  return { crossed, ms, receipt, lo: f(lo), hi: f(hi), steps: tmaxSteps };
}
for (const [label, nu, steps] of [["inviscid", null, 160], ["nu=1/1024", 10, 64], ["nu=1/1024", 10, 160],
                                  ["nu=1/128", 7, 128], ["nu=1/16", 4, 160], ["nu=1/16", 4, 640]]) {
  const r = run(nu, steps);
  const cr = r.crossed === null ? "not reached" : `step ${r.crossed} (t = ${r.crossed}/64)`;
  console.log(`${label.padEnd(10)} t=${steps}/64  rho in [${r.lo}, ${r.hi}]  crossing: ${cr}  ` +
              `time ${r.ms.toFixed(1)} ms (${(r.ms / steps).toFixed(3)} ms/step)  receipt ${r.receipt}`);
}
