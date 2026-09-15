import { useRef, useState } from "react";

/** A point for the drawing only: description, not geometry. */
export type V3 = readonly [number, number, number];

export type Poly = { pts: readonly V3[]; fill: string; opacity?: number; stroke?: string; dashed?: boolean; title?: string };
/** `facing`: shown only when the drawing is seen from above ("up") or from below ("down") */
export type Label = { at: V3; text: string; color?: string; facing?: "up" | "down" };

/**
 * A small drag-to-turn orthographic view of flat faces (painter's order by depth). The third coordinate is
 * drawn up. Used to display the declared 5D and 6D objects; nothing here is walked.
 */
export type Line = { from: V3; to: V3; color?: string; dashed?: boolean; title?: string; width?: number; opacity?: number };

const addv = (a: V3, b: V3, k = 1): V3 => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
const norm = (a: V3): V3 => {
  const l = Math.hypot(...a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** a cone with its point at `apex`, opening to a circle around `base` (as triangles; description only) */
export function cone(apex: V3, base: V3, radius: number, fill: string, title: string, opacity = 0.22): Poly[] {
  const axis = norm([base[0] - apex[0], base[1] - apex[1], base[2] - apex[2]]);
  const helper: V3 = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const u = norm(cross(axis, helper));
  const w = cross(axis, u);
  const ring = Array.from({ length: 16 }, (_, i) => {
    const t = (i / 16) * Math.PI * 2;
    return addv(addv(base, u, radius * Math.cos(t)), w, radius * Math.sin(t));
  });
  return ring.map((p, i) => ({ pts: [apex, p, ring[(i + 1) % ring.length]], fill, opacity, stroke: fill, title }));
}

export function Solid({
  polys,
  labels = [],
  lines = [],
  width = 520,
  height = 440,
  initial = [0.6, 0.45],
}: {
  polys: Poly[];
  labels?: Label[];
  lines?: Line[];
  width?: number;
  height?: number;
  initial?: [number, number];
}) {
  const [[yaw, pitch], setView] = useState<[number, number]>(initial);
  const drag = useRef<{ x: number; y: number } | null>(null);

  const all = [...polys.flatMap((p) => p.pts), ...lines.flatMap((l) => [l.from, l.to]), ...labels.map((l) => l.at)];
  const c: V3 = [0, 1, 2].map((k) => all.reduce((t, v) => t + v[k], 0) / Math.max(1, all.length)) as unknown as V3;
  const radius = Math.max(1e-6, ...all.map((v) => Math.hypot(v[0] - c[0], v[1] - c[1], v[2] - c[2])));
  // half the view: the farthest point stays inside at every turn
  const scale = (0.5 * Math.min(width, height)) / radius;

  const turn = (v: V3) => {
    const [x0, y0, z0] = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
    const x1 = x0 * Math.cos(yaw) - y0 * Math.sin(yaw);
    const y1 = x0 * Math.sin(yaw) + y0 * Math.cos(yaw);
    const y2 = y1 * Math.cos(pitch) - z0 * Math.sin(pitch);
    const z2 = y1 * Math.sin(pitch) + z0 * Math.cos(pitch);
    return { x: width / 2 + x1 * scale, y: height / 2 - z2 * scale, depth: y2 };
  };

  const drawn = polys
    .map((p) => {
      const q = p.pts.map(turn);
      return { p, q, depth: q.reduce((t, s) => t + s.depth, 0) / q.length };
    })
    .sort((a, b) => b.depth - a.depth);

  return (
    <svg
      className="graph"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      style={{ cursor: "grab", touchAction: "none", userSelect: "none", WebkitUserSelect: "none" }}
      onPointerDown={(e) => {
        e.preventDefault(); // dragging turns the drawing; it does not select text
        drag.current = { x: e.clientX, y: e.clientY };
        (e.target as Element).setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!drag.current) return;
        const dx = e.clientX - drag.current.x;
        const dy = e.clientY - drag.current.y;
        drag.current = { x: e.clientX, y: e.clientY };
        setView(([a, b]) => [a + dx * 0.01, Math.max(-1.5, Math.min(1.5, b + dy * 0.01))]);
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerLeave={() => (drag.current = null)}
    >
      {drawn.map(({ p, q }, i) => (
        <polygon
          key={i}
          points={q.map((s) => `${s.x},${s.y}`).join(" ")}
          fill={p.fill}
          fillOpacity={p.opacity ?? 0.5}
          stroke={p.stroke ?? "#1b1a18"}
          strokeWidth={0.9}
          strokeDasharray={p.dashed ? "4 3" : undefined}
        >
          {p.title && <title>{p.title}</title>}
        </polygon>
      ))}
      {lines.map((l, i) => {
        const a = turn(l.from);
        const b = turn(l.to);
        return (
          <line key={`l${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={l.color ?? "#3d6b39"} strokeWidth={l.width ?? 1.6} strokeOpacity={l.opacity ?? 1} strokeDasharray={l.dashed ? "4 3" : undefined}>
            {l.title && <title>{l.title}</title>}
          </line>
        );
      })}
      {labels.map((l, i) => {
        if ((l.facing === "up" && pitch <= 0) || (l.facing === "down" && pitch >= 0)) return null;
        const s = turn(l.at);
        return (
          <text key={i} x={s.x} y={s.y} fontSize={12} textAnchor="middle" fill={l.color ?? "#a23b2a"} stroke="#f4f1ec" strokeWidth={3} paintOrder="stroke">
            {l.text}
          </text>
        );
      })}
    </svg>
  );
}
