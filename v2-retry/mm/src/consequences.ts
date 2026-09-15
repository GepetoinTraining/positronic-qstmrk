import { DatabaseSync } from "node:sqlite";
import { DB_PATH } from "./db/store.ts";

/**
 * Reads the store after a long walk and lays out what each 2D cycle did. The construction's own words come first
 * (containers, arrivals, primes and substitutes as the ledger named them, limits as the receipts wrote them). A
 * machine-side check follows, clearly marked: floatland arithmetic on the ledger's names, only to see where the
 * house's primes agree with floatland's and which of floatland's have not arrived yet.
 */
const db = new DatabaseSync(DB_PATH, { readOnly: true });
const runs = db.prepare("SELECT id, walk, dimension, container, receipt FROM runs WHERE id IN (SELECT MAX(id) FROM runs GROUP BY walk, dimension) ORDER BY dimension, id").all() as {
  id: number;
  walk: string;
  dimension: number;
  container: string;
  receipt: string;
}[];
const ledger = db.prepare("SELECT slot, reading, status, name, found_in FROM ledger ORDER BY id").all() as { slot: string; reading: string; status: string; name: string; found_in: string }[];

console.log("the construction's own words");
for (const r of runs) {
  const where = `${r.dimension}D cycle ${r.walk.replace("cycle", "")}`;
  const kinds = db.prepare("SELECT kind, COUNT(*) AS n FROM nodes WHERE run_id = ? GROUP BY kind").all(r.id) as { kind: string; n: number }[];
  const count = (k: string) => kinds.find((x) => x.kind === k)?.n ?? 0;
  const found = ledger.filter((e) => e.found_in === where);
  const primes = found.filter((e) => e.status === "prime").map((e) => e.name);
  const limit = r.receipt.split("\n").find((l) => l.startsWith("limit"))?.replace(/ = .*/, "") ?? "";
  console.log(
    `  ${where.padEnd(14)} MM[${r.container}]  held ${count("whole") + count("table") + count("cube") + count("held") + count("floor")}  arrivals ${found.length}  re-met ${count("remet")}  primes ${primes.length}  substituted ${found.length - primes.length}  ${limit}`,
  );
  if (primes.length) console.log(`      new primes: ${primes.join(" ")}`);
}

const primesNamed = ledger.filter((e) => e.status === "prime").map((e) => Number(e.name));
const substituted = ledger.filter((e) => e.status === "substituted").map((e) => Number(e.reading));
const isPrime = (n: number) => {
  if (n < 2) return false;
  for (let d = 2; d * d <= n; d++) if (n % d === 0) return false;
  return true;
};
const top = Math.max(...primesNamed, ...substituted);
const named = new Set(primesNamed);
const missing: number[] = [];
for (let n = 2; n <= top; n++) if (isPrime(n) && !named.has(n) && ![2, 3, 5, 7].includes(n)) missing.push(n);

console.log("\nmachine side (floatland arithmetic, a check only)");
console.log(`  ledger primes: ${primesNamed.length}; every one prime in floatland: ${primesNamed.every(isPrime) ? "yes" : `no — ${primesNamed.filter((n) => !isPrime(n)).join(" ")}`}`);
console.log(`  ledger substitutes: ${substituted.length}; every one composite in floatland: ${substituted.every((n) => !isPrime(n)) ? "yes" : `no — ${substituted.filter(isPrime).join(" ")}`}`);
console.log(`  largest reading met: ${top}; floatland primes below it not yet arrived (besides the rail 2 3 5 7): ${missing.length}`);
console.log(`  the first of them: ${missing.slice(0, 40).join(" ")}${missing.length > 40 ? " …" : ""}`);
