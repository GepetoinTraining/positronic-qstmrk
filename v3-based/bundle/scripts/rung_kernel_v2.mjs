// Viscous CLM rung, typed kernel v2. Zero is never a storage value.
//   EMPTY (∅): the harmonic has not been reached yet. Not a number; operations skip it.
//   Every reached value is m * 2^e with m an int in [2^30, 2^31) (sign apart): a per-harmonic rung.
//   A value that is truly 0 can only arise by exact cancellation, and that is logged as an event.
// Tables are built cold. The hot loop is integer multiply / add / shift / compare. Drops are priced.
const K = 32, LO = 1n << 30n, HI = 1n << 31n;
const EMPTY = -(1n << 63n);                           // sentinel: cannot be produced by normalization
let droppedBits = 0, zeroEvents = 0;

function norm(m, e) {                                 // -> [m, e], m in [2^30, 2^31) in magnitude
  if (m === 0n) { zeroEvents++; return [EMPTY, 0]; } // exact cancellation: logged, not stored as 0
  const neg = m < 0n; let a = neg ? -m : m;
  while (a >= HI) { if (a & 1n) droppedBits++; a >>= 1n; e++; }
  while (a < LO) { a <<= 1n; e--; }                   // exact, loses nothing
  return [neg ? -a : a, e];
}
const mul = (ma, ea, mb, eb) => norm(ma * mb, ea + eb);
function add(ma, ea, mb, eb) {
  if (ma === EMPTY) return [mb, eb];
  if (mb === EMPTY) return [ma, ea];
  if (ea < eb) [ma, ea, mb, eb] = [mb, eb, ma, ea];
  const sh = ea - eb;
  if (sh > 62) { droppedBits += 31; return [ma, ea]; } // the small one is wholly below; priced, not zeroed
  const kept = mb >> BigInt(sh);
  if (mb !== kept << BigInt(sh)) droppedBits += sh;
  return norm((ma << 0n) + kept, ea);
}
const ge = (ma, ea, mb, eb) => (ea !== eb ? ea > eb : ma >= mb);  // both positive, normalized

function run(nu /* [num, den] or null */, steps) {
  droppedBits = 0; zeroEvents = 0;
  const M = new BigInt64Array(K + 1).fill(EMPTY), E = new Int32Array(K + 1);
  const KM = new BigInt64Array(K + 1), KE = new Int32Array(K + 1);   // cold table: 1 - nu*dt*k^2
  for (let k = 1; k <= K; k++) {
    // exact rational (den*64 - num*k^2)/(den*64), scaled into [2^30,2^31) with an exponent
    const num = nu ? BigInt(nu[1] * 64 - nu[0] * k * k) : 1n, den = nu ? BigInt(nu[1] * 64) : 1n;
    let [m, e] = norm((num << 62n) / den, -62); KM[k] = m; KE[k] = e;
  }
  [M[1], E[1]] = norm(1n, 0);
  const NM = new BigInt64Array(K + 1), NE = new Int32Array(K + 1);
  let crossed = null;
  for (let n = 1; n <= steps; n++) {
    for (let k = 1; k <= K; k++) {
      let sm = EMPTY, se = 0;
      for (let j = 1; j < k; j++) {
        if (M[j] === EMPTY || M[k - j] === EMPTY) continue;
        const [pm, pe] = mul(M[j], E[j], M[k - j], E[k - j]);
        [sm, se] = add(sm, se, pm, pe);
      }
      if (sm !== EMPTY) se -= 7;                        // * dt/2 = 2^-7, an exponent step
      let dm = EMPTY, de = 0;
      if (M[k] !== EMPTY) [dm, de] = mul(M[k], E[k], KM[k], KE[k]);
      [NM[k], NE[k]] = add(dm, de, sm, se);
    }
    M.set(NM); E.set(NE);
    if (crossed === null) {
      let all = true;
      for (let k = K / 4; k < 3 * K / 4; k++)
        if (M[k] === EMPTY || M[k + 1] === EMPTY || !ge(M[k + 1], E[k + 1], M[k], E[k])) { all = false; break; }
      if (all) crossed = n;
    }
  }
  // cold read of the band ratios (print only)
  const rho = [];
  for (let k = K / 4; k < 3 * K / 4; k++) {
    const sh = E[k + 1] - E[k];
    rho.push(Number((M[k + 1] << 40n) / M[k]) / 2 ** 40 * 2 ** sh);
  }
  return { crossed, lo: Math.min(...rho).toFixed(6), hi: Math.max(...rho).toFixed(6), droppedBits, zeroEvents,
           topMode: `${M[K]}·2^${E[K]}` };
}

const t0 = performance.now();
for (const [label, nu, steps] of [["inviscid", null, 160], ["nu=1/1024", [1, 1024], 64], ["nu=1/1024", [1, 1024], 160],
                                  ["nu=1/128", [1, 128], 128], ["nu=1/16", [1, 16], 160], ["nu=1/16", [1, 16], 640]]) {
  const s = performance.now(); const r = run(nu, steps);
  const cr = r.crossed === null ? "not reached" : `step ${r.crossed} (t=${r.crossed}/64)`;
  console.log(`${label.padEnd(10)} t=${steps}/64  rho [${r.lo}, ${r.hi}]  crossing: ${cr}  ` +
    `${(performance.now() - s).toFixed(0)} ms  dropped bits ${r.droppedBits}  zero events ${r.zeroEvents}`);
}
