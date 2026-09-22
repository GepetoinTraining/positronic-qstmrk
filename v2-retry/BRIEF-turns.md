# Brief — the three oscillators (i, j, k) as a turn on `walkCycle`

Repo: `positronic-qstmrk`, `v2-retry/mm`. Written by Claude (Fable 5.1) at pgarcia's request, 22.09.2026, night.
House rules apply: numbers built, not imported; lookups over runtime decisions; every claim tagged [M]/[F]/[B]; sealed receipts decide.

## What was found tonight (reproduce first)

`src/tables-last.ts` (new, in this drop) walks cycles 1…N in 2D with no naming, no ledger, no store, keeps only the last pair across the seam per cycle, and prints one line per cycle. Nine cycles run in ~3 s. [M]

```
cycle 1:   4/3   →     12     cycle 6: 144/143 →  20592
cycle 2:   9/8   →     72     cycle 7: 169/168 →  28392
cycle 3:  25/21  →    525     cycle 8: 256/247 →  63232
cycle 4:  64/63  →   4032     cycle 9: 361/360 → 129960
cycle 5:  49/45  →   2205
```
Every numerator is a square (2² 3² 5² 8² 7² 12² 13² 16² 19²). [M]

Frontier test (names kept for the test only): by cycle k, which integers ≤ hull are **not** met as names (seat tallies of held ∪ arrivals ∪ re-met)?

```
cycle 3: max  25  holes: 23 | 14 18 20 22 24
cycle 4: max  64  holes: 47 53 | 48 60 62
cycle 6: max 144  holes: 83 101 107 113 127 131 | 98 106 110 114 115 118 …
cycle 7: max 169  holes: (no primes) | 115 118 130 141 160 162 166
cycle 8: max 256  holes: 173 227 233 239 251 | 141 166 170 174 185 206 207 …
```
- Holes are a frontier band under the hull, both primes and composites; primes close one or two cycles later (all primes ≤169 present at cycle 7). [M]
- **Every missing composite has all its prime factors present** — zero exceptions over 8 cycles. Crossings do not build composites from primes. [M]
- At the cell level (`reduce(a,b)` over names) only prime holes are unreachable; composites are reachable through a multiple. [M]
- Running the 3D walk (cycles 1–2) over the cycle-4 rail fills **none** of the five holes ≤ 64. Dimension is base, not fiber. [M]

## The posit to test

pgarcia: three oscillators, i then j then k; each fills what the other can't because it uses the other rotation. The walk today is one orientation only ("order matters; every direction +"). The j and k walks are the same walk read through a different turn of the site.

Claude: the i→j→k cycle is the Hurwitz unit ½(1+i+j+k), order 6 — three steps return to the first oscillator sign-flipped (q³ = −1), six close. So the third fill is the first one inverted. [F] — floatland vocabulary; the code must only use the permutation table below.

## What to build

### 1. `turn` on `walkCycle` (`src/cycle.ts`)

```ts
export type Turn = { readonly label: string; readonly perm: readonly number[]; readonly sign: readonly (1 | -1)[] };
walkCycle(rail, k, dimension, { ...options, turn?: Turn })
```
A turn is a signed permutation of the coordinate slots, applied as a lookup when a site is built (`const site: Site = all.map(...)` in `add`) and when `place(container, m, axes)` receives its axes. `perm[a]` is the slot coordinate `a` lands on; `sign[a] = -1` means that coordinate is read inverted (the q/p side, the "−" direction). No arithmetic on seats: a turn relabels slots and flips a reading direction, nothing else. Default turn is the identity (today's walk, bit-identical — gate it against the sealed cycle receipts).

### 2. The table (`src/turns.ts`)

The 24 Hurwitz units as signed permutations of three slots — the rotation group of the cube, written out, not computed. Name the ones that matter:

| label | perm | sign | what it is |
|---|---|---|---|
| `1` | [0,1,2] | [+,+,+] | identity — today's walk |
| `i` | [0,1,2] | [+,−,−] | half-turn about x |
| `j` | [0,1,2] | [−,+,−] | half-turn about y |
| `k` | [0,1,2] | [−,−,+] | half-turn about z |
| `c` | [1,2,0] | [+,+,+] | ½(1+i+j+k): x→y→z, the axis-cycle, scalar ½ |
| `c²` | [2,0,1] | [+,+,+] | its square, z→y→x |
| `−1` | [0,1,2] | [−,−,−] | all inverted — the "−" direction of the whole walk |

Fill the remaining 17 the same way (all signed permutations with an even number of sign flips composed with all permutations of parity matching). [B]: the labelling; the table itself is the group and admits no choice.

In 2D only the sign column acts (a permutation of two equal coordinates changes no name); so 2D has exactly two turns, `1` and `−1`. In 3D all 24 act.

### 3. `tables-last.ts` with turns

Add `--turns 1,c,c2` (default `1`). For each turn run the same cycles 1…N and print the last pair per cycle per turn; then one union line per cycle. Keep the streaming shape: nothing stored past the current cycle except the rail per turn and a running seal. Receipt: `receipts/last-turns-{D}d.txt`, sealed.

### 4. The frontier test as a script (`src/frontier.ts`, test mode only, names kept)

For D=2, turns `1` and `−1`, cycles 1…8: holes ≤ hull per turn and for the union.
For D=3, turns `1`, `c`, `c²`, cycles 1…3 (budget 900 s each; fail gracefully): same.
Print, per cycle: holes for each turn, holes for the union, and `filled by j`, `filled by k` (holes of `1` not in the union). Receipt: `receipts/frontier-{D}d.txt`.

## What decides it

- If the union closes the frontier (holes at 64 go from five to zero across the three turns): pgarcia's posit holds — the oscillators fill each other's holes. [M when sealed]
- If 47 and 53 survive every turn: primes are holes in the fiber as well as the base, and that is the theorem to write next, not a bug to fix.
- Either way the identity turn must reproduce today's sealed receipts byte-for-byte before anything else is read.

## Do not

- No naming pass, no ledger, no SQLite in these scripts.
- No sin/cos/π; no quaternion multiplication at runtime — the 24 entries are a table.
- Do not "fix" holes by adding names. The frontier is the measurement.

## Results (22.09.2026, same night)

Built and run as specified; the sealed receipts are `mm/receipts/frontier-2d.txt`, `frontier-3d.txt`, `last-turns-2d.txt`, `last-turns-3d.txt`, and the gate is `mm/src/gate-turns.ts`. The reading, with every number, is the entry "Turns on walkCycle" at the end of `BUILDER_LOG.md`. Short form: the identity is the sealed walk byte for byte; a pure permutation of slots (c, c²) changes no name; the sign turns fill part of the frontier and never close it.
