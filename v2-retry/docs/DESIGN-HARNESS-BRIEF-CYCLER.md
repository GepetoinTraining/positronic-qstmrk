# DESIGN-HARNESS BRIEF — ADDENDUM IV: THE CYCLER AND THE ATLAS — PRIMITIVES
## "letters across substrate" · the v2 construction's tables pipeline (`positronic-qstmrk/v2-retry`), turned into primitives the site can hold

**What this is.** A companion to `DESIGN-HARNESS-BRIEF.md`, `DESIGN-SURFACE.md`,
`DESIGN-HARNESS-BRIEF-HORIZON.md` and `DESIGN-HARNESS-BRIEF-ATLAS.md`, under the same contract:
**structure, data, states, role behavior, intent only**. Every pixel is the design harness's
(NodeZero's) to decide. "Render distinctly" means *give it a distinct treatment*, never *here is
the treatment*. Hard invariants may not be broken; every `shape` hint may be overridden.

**Where it lives.** Written in `positronic-qstmrk` (`v2-retry/docs/`), because that repository is
the one pushed. `lettersacrosssubstrate` has no remote as of 2026-09-15. Copy it to the LAS root
beside the other briefs when that repo travels.

**What is built** (2026-09-14/15, all in `v2-retry/mm/src/`, each a script with a self-check):

1. **The atlas of a model's values.** Each stored bf16 word is `plane · odd / 2^e`. `e` is written
   as the **cube** byte type (`cube.ts`: 27 cells on/off + the side, from a centre derived from the
   model). The odd splits into a **minifier** (removable, ≤ 27) and a **residual** (what stays: a
   prime, a prime power, 1, or 243). The **handle** is (minifier, e). For Qwen3-1.7B: 4,505 values =
   433 handles × 63 residuals, every pair unique; 2 planes beside them.
   (`tables-minify.ts`, `tables-odds.ts`, `tables-roots.ts`, `tables-243.ts`)
2. **The cycler.** Pair neighbouring cells along an axis whose length holds a 2. Each distinct pair
   is a cell of that cycle's atlas. Repeat until only the odd count is left. Keeps order; every row
   unfolds back exactly. Two bases: the full value, or the residual with the handle removed.
   (`tables-bigram.ts`, `tables-bigram-model.ts`, `tables-cycle-updown.ts`)
3. **Measurements against a control.** Every structural claim is a count on the real arrangement
   beside the same count on the same cells shuffled in place (seed declared). The claim is the gap
   between the two counts, never a statistic. (`tables-measure.ts`, `tables-osc-rot.ts`)

**What is not built:** a library. These are scripts that write CSV/TXT receipts. §6 item 1 names the
API the primitives need. Nothing on the LAS side exists yet.

**The numbers that came out** (Qwen3-1.7B, all [M], all in `v2-retry/BUILDER_LOG.md`):
- e spans −6…46, span 52 = 2·26 = the cube's reach, so centre 20 is forced;
- 128 odds = 2⁷; 63 residuals in 2⁶ with one empty place; minifiers ≤ 27 in 306 of 311 tables,
  and the 5 without a 27 hold a 243;
- cycle 1 on residuals is the complete square 126² model-wide;
- `up × downᵀ` repeats more than its control in 28 of 28 layers at the first pairing, and nothing
  repeats after it;
- the plane relation of `up` and `down` turns with depth: aligned-position matches 207,310 at layer
  0, 186,133 at 13 (inside the control band 183,611…186,915), 146,670 at 27 (the lowest of 2,048
  shifts).

---

## 0. CROSS-CUTTING (binds every cycler/atlas surface, in addition to the atlas brief's §0)

### 0.1 Every number is a string, and exact
Counts are integers as written (`157,239,884`). Residuals, minifiers and e are whole numbers.
Powers are written as powers (`3⁵`, `2⁶`), never expanded into a decimal. No ratio, percentage or
p-value appears unless both counts it came from are beside it. A bar or curve is a picture of
counts; its axis labels are the counts.

### 0.2 The control is always beside the real
A measurement has two numbers, real and shuffled, plus the seed. A surface that shows only the
real count, or only the gap, breaks this. The verdict (§1 `verdict`) is derived from the two counts
and never stored separately.

### 0.3 Structure, not tables
The atlas is shown as what it is: the cube, the residual set, the handle grid, the cycle ladder.
A per-cell table (row × column of values) is allowed only as the **dump view**, opened on request,
never the default. The design harness must not flatten a cube into a list of 27 numbers, or a cycle
atlas into a spreadsheet.

### 0.4 A flag is not an error
Some things are refused on purpose: 243 = 3⁵ is not rooted (neither square nor cube); a centre is
refused when e's span is odd ("the centre would be a choice"); a choice the builder made is labelled
[B]. These render in the refusal register of the atlas brief §0.2 (principled, a menu when
there are options), never as a warning.

### 0.5 Labels are part of the number
[M] measured · [F] floatland reference only · [B] builder's choice. Each has a distinct, quiet
treatment and travels with the number into any line, receipt or picture. An [F] number never
stands alone as a result.

### 0.6 Counts must match their receipt
Every figure comes from a receipt file (CSV/TXT with its sha256). The harness never re-counts,
re-sums or re-sorts beyond the receipt's own order.

---

## 1. VOCABULARY — what the harness binds to

| concept | what it is | data (verbatim shape) | where |
|---|---|---|---|
| **word** | a stored bf16 weight | `u16` | the safetensors shards |
| **plane** | the sign: which side a value sits on | `+` · `−` | bit 15 |
| **odd** | the whole odd number the value carries (1…255) | `number` | `odds.csv` |
| **e** | the power of two the odd is divided by; the minification kept outside the value | `number` (−6…46 for Qwen3-1.7B) | `e.csv` |
| **cube** | the byte type for e: 27 cells on/off, cell (x,y,z) each 0,1,2, plus the side; one lit cell for one weight, many lit for a set | `{ side: "+"\|"−", cells: 27-bit, centre }`, drawn `100.000.000/000.010.000/000.000.001` (x-layers / y-rows . z) | `cube.ts`, `e.csv` (`e,side,x,y,z`), per table `cubes.csv` |
| **face** | the nine cells with one coordinate fixed | `face(axis, k)` | `cube.ts` |
| **centre** | the e at the cube's (0,0,0) on +; derived from the model's e span | `number` (20) | manifest |
| **minifier** | the part of the odd minifying removes, from the smallest prime up (≤ 27) | `number` | `odds.csv` |
| **residual** | what stays: a prime, a prime power, 1, or 243 | `number` + `prime`, `power` | `odds.csv` |
| **root / powers** | a square or cube given up to its root, in order | `81 → 3 (2.2)` | `roots.csv` |
| **the tell** | 243 = 3⁵: the residual no root takes; per-table count and plane split | `table,weights,plus,minus,total` | `243.csv` |
| **handle** | (minifier, e): the removable, controllable part of a value | `"27/14"` | derived |
| **atlas** | every distinct value of a model as handle × residual × plane | 4,505 = 433 × 63 (unique pairs) | `odds.csv`, `e.csv`, `lut.csv` |
| **basis** | what a cycle-0 cell is: the full `word`, or the `residual` (handle removed) | `"word"\|"residual"` | `tables-bigram.ts` arg |
| **cycle** | one pairing of neighbours along a 2-power axis | `{ cycle, columns, cells_distinct, cells_held }` | `bigram/cycle-<c>.csv` (`id,left,right`) |
| **top** | the cells left when the 2-power is gone (the odd count per row) | `row → id[]` | `bigram/top.csv` |
| **unfold** | rebuilding a row from its top cells through the cycle atlases alone | `same`, `different` counts | self-check line |
| **measurement** | one count on real and on control | `{ test, table, held, distinct_real, distinct_shuffled, seed }` | `measure.csv`, `cycle-updown.csv` |
| **verdict** | derived: real repeats more · same · fewer | computed from the two counts | never stored |
| **pairing** | which cells meet: `cols`, `rows`, `gate×up`, `q×k`, `up×downᵀ`, `q×oᵀ` | `string` + the index rule | `tables-measure.ts` header |
| **depth trace** | a count per layer, real and control | `layer → {real, shuffled}` | `osc-rot.txt` (flip), `cycle-updown.csv` |
| **shift scan** | matches at every shift s along an axis, real and control | `shift,matches_real,matches_shuffled` | `rot-<layer>.csv` |
| **receipt** | the script, its output file, the file's sha256 | `{ script, file, sha256 }` | each script's stdout; BUILDER_LOG |

**Intent line for every cycler surface:** a reader can take any number on the page back to a file
and a script, and the picture never says more than the two counts under it.

---

## 2. ROUTE MAP — additions (structure only; the harness chooses where they mount)

| Route | Section | Min role | Gated sub-content | Data |
|---|---|---|---|---|
| `/atlas/models/[model]` *(new)* | a model's atlas: the cube, the residual set, the handle grid, the tell | public | the per-table dump view (owner) | `e.csv`, `odds.csv`, `roots.csv`, `243.csv`, manifest |
| `/atlas/models/[model]/cycler` *(new)* | the cycle ladder per table and model-wide, both bases | public | dump views of cycle atlases (owner) | `bigram/*`, `bigram-model.txt`, `cycle-updown.*` |
| `/atlas/models/[model]/measure` *(new)* | measurements: per pairing, per table; depth traces; shift scans | public | none | `measure.*`, `osc-rot.txt`, `rot-*.csv` |
| `/letters/[id]` | gains the inline **measure line** (§4.7) | as today | as today | the line's receipt by sha |
| `/desk/atlas` | gains the list of receipts per model and any unfold that failed | owner | everything | receipts index |

---

## 3. THE UX WALK — the states a person passes through, in order

| step | who | what they see | exists today | new for design |
|---|---|---|---|---|
| 1 | the builder, in a session | a script runs and prints its counts and sha | the scripts | nothing, no LAS pixel |
| 2 | a reader, on the model's atlas | the cube of e for the whole model (+23 on, −26 on), the 63 residuals with the empty 64th, the 11 minifiers under the 27 ceiling, 243 marked as the tell | none | §4.1–§4.3 |
| 3 | the same reader, opening a table | that table's two cubes (e.g. `q_norm` + empty, − 12 on), its residuals, its 243 count; the dump view behind a door | none | §4.1, §4.4 |
| 4 | the reader, on the cycler | per table, the ladder cycle 0 → top: distinct vs held at each rung; the rung where the atlas is complete (126²) marked declared; unfold same/different | none | §4.5 |
| 5 | the reader, on measurements | per pairing, real beside shuffled with the seed; the table split (fewer / same / more); the depth trace for `up×downᵀ` crossing at layer 13; the shift scans | none | §4.6 |
| 6 | a reader of a letter | a measure line inside prose, its sha, a door to the measurement | none | §4.7 |

---

## 4. SURFACES

### 4.1 The cube

**Intent:** e is seen as a place in three dimensions, not as a number; a set of values is seen as
which cells are lit.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `Cube` | `{ side, cells (27), centre }`; for a set, one cube per side | **one lit** (a weight) · **many lit** (a table, a row, the model) · **empty side** (e.g. every norm has no + cells) · **both sides** | role-flat |
| `CubeCell` | `(x,y,z)`, on/off, the e it stands for, slot count | on · off · **unused code** (−000; e never met, e.g. 42–45) | the e is shown on demand, beside the cell, never instead of it |
| `Face` | nine cells, one coordinate fixed | selected · not | — |

Hard invariants: the cube is never rendered as a list or a number line; the side is part of the
cube, not a colour on a list; the centre is named with its derivation (span 52 = 2·26). `shape`
hint: three 3×3 layers side by side, the reading order x-layer / y-row / z.

### 4.2 The residual set and the minifiers

**Intent:** what can be removed and what cannot are two different kinds of thing.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `ResidualSet` | 63 residuals + the empty place | prime · prime power (with its powers given up) · 1 · **243 (the tell)** · **the empty 64th** (absent by construction, never a gap) | role-flat |
| `MinifierScale` | the 11 minifiers under the 27 ceiling | present · absent in this table · **27 = the byte** (never rooted) | — |
| `HandleGrid` | minifier × e, cells = the residuals each handle carries | full (63) · sparse · tight (27-handles: at most 2) | the grid is a picture of `handle × residual`, not a spreadsheet (§0.3) |

Hard invariants: 243 is never rooted in any view; 27 is never shown as 3³ reduced to 3; the empty
64th is drawn as a place.

### 4.3 The tell

**Intent:** 3⁵ and 3³ as a pair: the residual that cannot be removed and the magnitude that can.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `Tell` | per table: 243 count (+/−), highest minifier | holds 243 · holds 27 · holds both · **neither never occurs** (a measured fact, stated) | role-flat |

### 4.4 The value, composed

**Intent:** one value read as its parts, never as a decimal.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `ValueParts` | plane · residual · minifier · e (cube cell) · the stored word | complete · subnormal (e at the floor) · **e < 0** (twos, not decades) | the decimal reading, if ever shown, is labelled [F] and sits behind the parts |

### 4.5 The cycle ladder

**Intent:** the fractal made visible: each rung is a pairing, and the reader sees where pairs repeat,
where the atlas is complete, and where every cell is new.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `CycleLadder` | rungs cycle 0…top: `columns`, `cells_distinct`, `cells_held`; basis | per rung: **complete** (distinct = the full square: declared, not stored) · **repeating** (distinct < held) · **unique** (distinct = held) | role-flat |
| `Unfold` | same / different counts | **exact** (0 different) · **broken** (any different: the ladder is withdrawn, shown as a refusal) | — |
| `BasisSwitch` | `word` · `residual` | — | switching re-reads a receipt; it never recomputes |

Hard invariants: the two bases are never merged into one ladder; a complete rung says what it is
the square of (126²); the top row count is the odd count left (e.g. 2048 → 1, 6144 → 3).

### 4.6 Measurements

**Intent:** structure is a gap between two counts, and the control is always in view.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `MeasurePair` | `held`, `distinct_real`, `distinct_shuffled`, `seed` | verdict: **real repeats more** · **same** · **real repeats fewer**; each distinct, none good or bad | role-flat |
| `TableSplit` | per pairing: tables fewer / same / more | — | a split like 108/97/106 reads as *no structure*; 28/0/0 reads as *structure*, from the counts, not a label |
| `DepthTrace` | per layer real + control | **agree** (real under control) · **crossing** (real = control, e.g. layer 13) · **flip** (real over control) | — |
| `ShiftScan` | 2,048 shifts, real + control band | **aligned** (s = 0 above the band) · **neutral** · **anti-aligned** (s = 0 under the band) · a shift other than 0 above the band (rotation; none measured yet) | — |

Hard invariants: §0.1, §0.2; the control band is the shuffled range, never a derived interval.

### 4.7 The measure line — inside a letter

**Intent:** a result travels into prose the way a recipe line does (atlas brief §4.1), as a
sentence with an address.

| Component | Data | States | Role behavior |
|---|---|---|---|
| `MeasureLine` *(sibling of `RecipeLine`)* | `up×downᵀ · cycle 2 · 157,239,884 / 157,572,816 · seed 20260914 · 28/28 · <sha8>` | **linked** (sha resolves to a receipt) · **orphan** · **mismatch** (the atlas brief's three states) | as `RecipeLine` |

Hard invariants: the atlas brief §0.3 applies to the sha; the two counts are never collapsed into
a ratio inside the line.

---

## 5. WHAT THE SYSTEM ALREADY HOLDS, AND WHAT IS NEW

| need | exists | new |
|---|---|---|
| an exact certificate | `Receipt` (atlas brief §4.2) | the receipt of a script run: script, file, sha, counts |
| a refusal with a reason | `Refusal`, `<LockedStub>`, the dark door | flags: 243 not rooted, centre refused, [B] choices (§0.4) |
| an inline address in prose | `RecipeLine` | **`MeasureLine`**, the same three states |
| a picture that is a check | `LightBox` + residual caption | **`Cube`**, **`CycleLadder`**, **`DepthTrace`**, **`ShiftScan`**: pictures of counts |
| a set drawn as a place | — | **`Cube` / `Face`**: a new primitive, the byte type as geometry |
| two kinds in one set | `GlossaryTerm` states | **`ResidualSet`** with the empty place, **`MinifierScale`**, **`HandleGrid`** |
| real beside control | — | **`MeasurePair`**, **`TableSplit`**: a new pattern, the control as a permanent sibling |
| a series by depth | `Timeline` | `DepthTrace` states (agree / crossing / flip) |

**Verdict:** design has work. Two primitives are new below the component level: the **cube** (a
27-cell three-layer object with a side) and the **control pair** (every result is two counts and a
seed). The rest composes from the atlas brief's register.

---

## 6. NOT DESIGN — flagged so nothing is assumed designed-away

1. **The scripts are not a library.** The primitives need a pure API before any surface can bind:
   `split(word) → {plane, odd, e}` · `cubeOf(e, centre)` / `eOf(cube, centre)` · `minify(odd) →
   {minifier, residual, prime, power}` · `roots(n) → {root, powers}` · `cycle(cells, width) →
   {atlas, next}` · `unfold(top, atlases)` · `measure(real, control, pairing) → {held, distinct}`,
   with the shuffle and its seed as an argument. Today the same logic is inlined in each
   `tables-*.ts`. Code.
2. **Receipts are not content-addressed yet.** Scripts print sha256 prefixes; there is no receipts
   index, no `inspect(sha)`. The `MeasureLine` needs both (the atlas brief's link step pattern).
3. **Data volume.** Model-wide receipts are small (KB). Per-table dumps (cycle atlases, values,
   cubes) are hundreds of MB for three sample tables and are not in git. The dump views must read
   from the builder's machine or a declared store, never from the page.
4. **The model is external.** Weights are fetched from the Hub; nothing in LAS may carry them
   (`positronic-qstmrk` `.gitignore`, LOG §31: "the ids are the weights").
5. **One model measured.** Every number above is Qwen3-1.7B. A second model is a second set of
   receipts; the centre, the residual count and the ceilings are per model, never assumed.
6. **Open measurements.** The pairings matmul actually makes beyond `up×downᵀ` (attention through
   the softmax, `v × o`), rotation along the intermediate axis (not scanned: 6,144 shifts), and
   whether cycle-2 gaps outside `up×downᵀ` are structure. A surface for these is a dark state until
   receipts exist.

---

## 7. ⟨ink⟩ — Pedro's to decide before the harness starts

1. **Where the cycler lives**: a wing of the atlas mount (one renderer, atlas brief §7.2), or its
   own route under `/atlas/models`. §2 assumes the latter as structure only.
2. **Whether the dump view is public**: §2 gates it to the owner, on the grounds that it is the
   flat view (§0.3).
3. **The 7D schema**: the handle is the kernel, and a 7D schema should give back the sequence. Not
   given yet, so no surface is specified for it.
