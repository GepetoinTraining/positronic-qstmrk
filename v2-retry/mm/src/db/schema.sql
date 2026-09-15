-- The working store for the viewer. The TPB stays the append-only record; this is where walks are read from.
CREATE TABLE IF NOT EXISTS runs (
  id         INTEGER PRIMARY KEY,
  walk       TEXT    NOT NULL,   -- e.g. cycle2
  dimension  INTEGER NOT NULL,   -- the tab it belongs to
  container  TEXT    NOT NULL,   -- ordered coordinates, e.g. "O1,O1"
  rail       TEXT    NOT NULL,   -- JSON: rings on the rail after the cycle, [{label, reading}]
  seal       TEXT    NOT NULL,   -- sha256 of the receipt
  receipt    TEXT    NOT NULL,
  created    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
  id          INTEGER PRIMARY KEY,
  run_id      INTEGER NOT NULL REFERENCES runs(id),
  label       TEXT    NOT NULL,
  mm          TEXT    NOT NULL,  -- MM[…] in coordinate order
  resolution  INTEGER NOT NULL,  -- coordinate tally (0 = floor)
  kind        TEXT    NOT NULL,  -- whole | table | held | floor | question
  site        TEXT    NOT NULL,  -- JSON: ring labels per slot, in order
  seats       TEXT    NOT NULL,  -- JSON: seat addresses in the container
  reading     TEXT    NOT NULL,  -- the region's run read in decades, by pairing
  also       TEXT    NOT NULL,  -- JSON: further regions the same arrival was met at
  holds       INTEGER NOT NULL   -- how many other nodes' regions it holds (layout only)
);

CREATE TABLE IF NOT EXISTS ledger (
  id           INTEGER PRIMARY KEY,
  slot         TEXT    NOT NULL,  -- O<n>
  reading      TEXT    NOT NULL,  -- its run read in decades, by pairing
  status       TEXT    NOT NULL,  -- prime | substituted
  name         TEXT    NOT NULL,
  substitutes  TEXT    NOT NULL,  -- JSON: [{label, dimension, walk}]
  found_in     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
  id          INTEGER PRIMARY KEY,
  run_id      INTEGER NOT NULL REFERENCES runs(id),
  source      INTEGER NOT NULL REFERENCES nodes(id),  -- top of the relation
  target      INTEGER NOT NULL REFERENCES nodes(id),  -- bottom
  family      TEXT    NOT NULL,  -- Q* | Pq | limit | unordered
  reading     TEXT    NOT NULL,  -- turn | relation | limit | none
  site_group  TEXT               -- relations naming one site share it
);
