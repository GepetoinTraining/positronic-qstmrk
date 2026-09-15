import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

/**
 * Minify the squares and the cubes (pgarcia): every residual and every minifier from odds.csv that is a square or a
 * cube gives up that power, again and again while one is left; the powers given up live outside, in order.
 * [B] a square is tried before a cube; within 1…255 no odd is both (that would take a 6th power, 3⁶ = 729), so the
 * order never decides anything here.
 * [B] 1 is 1² and 1³ and gives up nothing (n/n is n).
 *   roots.csv  odd, minifier, minifier_root, minifier_powers, residual, residual_root, residual_powers, weights
 *              powers are written in the order given up, e.g. 81 → root 3, powers 2.2
 */

const root = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-minified/";
const odds = readFileSync(`${root}odds.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number));

/** the whole r with r^k = v, if there is one */
const rootOf = (v: number, k: number) => {
  for (let r = 2; r ** k <= v; r++) if (r ** k === v) return r;
  return 0;
};
/** give up squares and cubes while one is left */
const minify = (v: number): [number, number[]] => {
  const powers: number[] = [];
  let rest = v;
  for (;;) {
    const s = rest > 1 ? rootOf(rest, 2) : 0;
    if (s) {
      powers.push(2);
      rest = s;
      continue;
    }
    const c = rest > 1 ? rootOf(rest, 3) : 0;
    if (c) {
      powers.push(3);
      rest = c;
      continue;
    }
    return [rest, powers];
  }
};

const lines = ["odd,minifier,minifier_root,minifier_powers,residual,residual_root,residual_powers,weights"];
const residualRoots = new Set<number>();
const minifierRoots = new Set<number>();
const shown: string[] = [];
const notRooted: string[] = [];
for (const [odd, minifier, residual, prime, power, , weights] of odds) {
  const [mr, mp] = minify(minifier);
  const [rr, rp] = minify(residual);
  residualRoots.add(rr);
  minifierRoots.add(mr);
  // the residual's root must be its prime unless a power other than squares and cubes is left
  if (residual > 1 && rr !== prime) notRooted.push(`${residual} = ${prime}^${power} → ${rr}`);
  if (mp.length || rp.length) shown.push(`  ${odd}: minifier ${minifier}${mp.length ? ` → ${mr} (${mp.join(".")})` : ""} · residual ${residual}${rp.length ? ` → ${rr} (${rp.join(".")})` : ""}`);
  lines.push(`${odd},${minifier},${mr},${mp.join(".")},${residual},${rr},${rp.join(".")},${weights}`);
}
const body = lines.join("\n") + "\n";
writeFileSync(`${root}roots.csv`, body);
const sorted = (s: Set<number>) => [...s].sort((a, b) => a - b).join(" ");
process.stdout.write(
  `odds that gave up a square or cube: ${shown.length}\n${shown.join("\n")}\n\n` +
    `residual roots ${residualRoots.size}: ${sorted(residualRoots)}\n` +
    `minifier roots ${minifierRoots.size}: ${sorted(minifierRoots)}\n` +
    `left as a power that is neither square nor cube: ${notRooted.join(" · ") || "none"}\n` +
    `roots.csv sha256 ${createHash("sha256").update(body).digest("hex").slice(0, 8)}…\n`,
);
