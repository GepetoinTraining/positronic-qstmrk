export type Run = {
  id: number;
  walk: string;
  dimension: number;
  container: string;
  rail: string; // JSON: [{label, seats}]
  seal: string;
  receipt?: string;
  created: string;
};

/** `reading`: the run read in decades by the construction, by pairing */
export type RailRing = { label: string; reading: string };

export type WalkNode = {
  id: number;
  label: string;
  mm: string;
  resolution: number;
  kind: "whole" | "cube" | "table" | "held" | "floor" | "question" | "remet" | "rail";
  site: string; // JSON: ring labels per slot
  seats: string; // JSON: seat addresses
  reading: string; // the region's run read in decades, by pairing (construction side)
  also: string; // JSON: further regions of the same ring
  holds: number;
};

export type WalkEdge = {
  id: number;
  source: number; // top
  target: number; // bottom
  family: "Q*" | "Pq" | "R*" | "limit" | "unordered" | "collapse";
  reading: "turn" | "relation" | "limit" | "none" | "resolution";
  site_group: string | null;
};

export type Walk = { run: Run; nodes: WalkNode[]; edges: WalkEdge[] };

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\//, "")}`;
const loadJson = async <T>(path: string, fallback: T): Promise<T> => {
  try {
    const r = await fetch(apiUrl(path));
    return r.ok ? ((await r.json()) as T) : fallback; // the store may be mid-rebuild or absent on static hosting
  } catch {
    return fallback;
  }
};

export async function listRuns(dimension: number): Promise<Run[]> {
  return loadJson(`/api/runs?dimension=${dimension}`, []);
}

export async function loadRun(id: number): Promise<Walk | null> {
  return loadJson(`/api/run/${id}`, null);
}

export type LedgerRow = { id: number; slot: string; reading: string; status: "prime" | "substituted"; name: string; substitutes: string; found_in: string };

export async function loadLedger(): Promise<LedgerRow[]> {
  return loadJson("/api/ledger", []);
}

/**
 * The viewer takes the construction's decade readings (made by pairing) and reads them as JS numbers for its own
 * layout. Shadow, flagged: the arithmetic the viewer then does with them (gcd, cross products, list order) is the
 * next shortcut to replace; holding is read by pairing runs (Cells).
 */
export const readName = (reading: string) => Number(reading);
export const nodeReading = (n: WalkNode) => readName(n.reading);
