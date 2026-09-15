type P = { x: number; y: number };

const S = 10;

/** +++ only: first coordinate down-right, second down-left, third up. */
const iso = (i: number, j: number, k: number): P => ({ x: 30 + (i - j) * S * 0.87, y: 24 + (i + j) * S * 0.5 - k * S });
const poly = (ps: P[]) => ps.map((p) => `${p.x},${p.y}`).join(" ");

/** A thing's seats drawn as cubes inside its 3D container. */
export function SeatCubes({ seats, container }: { seats: number[][]; container: number[][] }) {
  const on = (s: number[]) => seats.some((t) => t[0] === s[0] && t[1] === s[1] && t[2] === s[2]);
  const ordered = [...container].sort((a, b) => a[2] - b[2] || a[0] + a[1] - (b[0] + b[1]));
  return (
    <svg width={60} height={50} viewBox="0 0 60 50" aria-label="seats as cubes">
      {ordered.map(([i, j, k]) => {
        const lit = on([i, j, k]);
        const stroke = lit ? "#1b1a18" : "#d3cbbf";
        const top = [iso(i, j, k + 1), iso(i + 1, j, k + 1), iso(i + 1, j + 1, k + 1), iso(i, j + 1, k + 1)];
        const right = [iso(i + 1, j, k), iso(i + 1, j + 1, k), iso(i + 1, j + 1, k + 1), iso(i + 1, j, k + 1)];
        const left = [iso(i, j + 1, k), iso(i + 1, j + 1, k), iso(i + 1, j + 1, k + 1), iso(i, j + 1, k + 1)];
        return (
          <g key={`${i}${j}${k}`} stroke={stroke} strokeWidth={0.7}>
            <polygon points={poly(right)} fill={lit ? "#b8503c" : "none"} />
            <polygon points={poly(left)} fill={lit ? "#8f3a2a" : "none"} />
            <polygon points={poly(top)} fill={lit ? "#d9735c" : "none"} />
          </g>
        );
      })}
    </svg>
  );
}
