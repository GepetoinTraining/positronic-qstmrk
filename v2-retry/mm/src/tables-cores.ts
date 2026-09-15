import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

/**
 * The smaller set (pgarcia: dedup, work on the smaller set, then replicate it to every value).
 * From lut.csv (one line per stored magnitude): a slot's whole value is its core followed by its band of tens, and
 * its core is the odd part of n times its fives. Slots whose cores are the same differ only by band, so the cores
 * dedup further:
 *   cores.csv  core id (from 1, in slot order), core, odd, fives, core ÷ 360 → turns, residual
 *   slots.csv  slot, core id, band (tens), n, m — every slot points at one core and carries its band
 * The pointer tables (±slot) are untouched: slot → core id + band → every weight.
 * Self-check: every slot's core·10^band equals its lut.csv value, and n/2^m·10^K equals it too.
 */

const root = "D:/positronic-qstmrk/v2-retry/tables/qwen3-1.7b-pointers/";
const manifest = JSON.parse(readFileSync(`${root}manifest-partial.json`, "utf8")) as { K: number };
const K = BigInt(manifest.K);
const lut = readFileSync(`${root}lut.csv`, "utf8").trim().split("\n").slice(1).map((l) => l.split(","));

const coreId = new Map<string, number>();
const cores: string[] = ["core_id,core,odd,fives,turns,residual"];
const slots: string[] = ["slot,core_id,band,n,m"];
let bad = 0;
for (const [slot, n, m, core, fives, tens] of lut) {
  let id = coreId.get(core);
  if (id === undefined) {
    id = coreId.size + 1;
    coreId.set(core, id);
    const c = BigInt(core);
    let odd = c;
    let f = 0;
    while (odd % 5n === 0n) {
      odd /= 5n;
      f++;
    }
    cores.push(`${id},${core},${odd},${f},${c / 360n},${c % 360n}`);
  }
  slots.push(`${slot},${id},${tens},${n},${m}`);
  // self-check: the core with its band is the slot's whole value, and it is the stored word at scale K
  const V = BigInt(core) * 10n ** BigInt(tens);
  const mm = BigInt(m);
  const stored = mm >= 0n ? V * (1n << mm) === BigInt(n) * 10n ** K : V === BigInt(n) * (1n << -mm) * 10n ** K;
  if (!stored) bad++;
}
const write = (file: string, lines: string[]) => {
  const body = lines.join("\n") + "\n";
  writeFileSync(`${root}${file}`, body);
  return createHash("sha256").update(body).digest("hex").slice(0, 8);
};
const coresSha = write("cores.csv", cores);
const slotsSha = write("slots.csv", slots);
process.stdout.write(`${lut.length} slots → ${coreId.size} distinct cores · self-check: ${lut.length - bad} exact, ${bad} bad\n`);
process.stdout.write(`cores.csv sha256 ${coresSha}… · slots.csv sha256 ${slotsSha}…\n`);
process.stdout.write(`\n${cores.slice(0, 7).join("\n")}\n…\n\n${slots.slice(0, 7).join("\n")}\n…\n`);
