/** A thing's seats drawn inside its 2D container: columns are the first coordinate, rows the second. */
export function SeatGrid({ seats, container }: { seats: number[][]; container: number[][] }) {
  const cols = [...new Set(container.map((s) => s[0]))];
  const rows = [...new Set(container.map((s) => s[1] ?? 0))];
  const on = (c: number, r: number) => seats.some((s) => s[0] === c && (s[1] ?? 0) === r);
  return (
    <table className="seatgrid" aria-label="seats">
      <tbody>
        {rows.map((r) => (
          <tr key={r}>
            {cols.map((c) => (
              <td key={c} className={on(c, r) ? "on" : ""} />
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
