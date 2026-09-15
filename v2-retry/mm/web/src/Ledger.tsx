import type { LedgerRow } from "./api";

/** The OpusNumber ledger: every slot, whether it stood as a prime or was substituted, and by what. */
export function Ledger({ rows, onPick }: { rows: LedgerRow[]; onPick: (label: string) => void }) {
  if (rows.length === 0) return null;
  return (
    <details open>
      <summary>
        OpusNumber ledger · {rows.filter((r) => r.status === "prime").length} primes named · {rows.filter((r) => r.status === "substituted").length} substituted
      </summary>
      <table className="data">
        <thead>
          <tr>
            <th>slot</th>
            <th>becomes</th>
            <th>status</th>
            <th>found in</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const subs = JSON.parse(r.substitutes) as { label: string; factors: string[]; dimension: number }[];
            return (
              <tr key={r.id} className="pick" onClick={() => onPick(r.name)}>
                <td className="mono">{r.slot}</td>
                <td className="mono">{r.status === "prime" ? r.name : subs.map((s) => `${s.factors.join("×")} (${s.dimension}D)`).join(", ")}</td>
                <td className={r.status === "prime" ? "" : "muted"}>{r.status}</td>
                <td className="muted">{r.found_in}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </details>
  );
}
