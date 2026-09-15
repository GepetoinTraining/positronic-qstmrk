import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

type T = { x: number; y: number; k: number };

type Props = { width: number; height: number; focusX: number; label: string; children: ReactNode };

/**
 * A canvas you drag to pan and wheel to zoom (around the pointer). Dragging starts only on the background,
 * so anything marked `.pick` keeps its click. Opens centred on `focusX` of the content.
 */
export function PanZoom({ width, height, focusX, label, children }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [t, setT] = useState<T>({ x: 0, y: 20, k: 1 });
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(null);

  const centre = () => {
    const w = ref.current?.clientWidth ?? width;
    setT({ x: w / 2 - focusX, y: 20, k: 1 });
  };

  useEffect(centre, [focusX, width]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const box = el.getBoundingClientRect();
      const mx = e.clientX - box.left;
      const my = e.clientY - box.top;
      setT((p) => {
        const k = Math.min(4, Math.max(0.2, p.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
        return { k, x: mx - ((mx - p.x) * k) / p.k, y: my - ((my - p.y) * k) / p.k };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div className="panzoom-wrap">
      <svg
        ref={ref}
        className={dragging ? "panzoom dragging" : "panzoom"}
        role="img"
        aria-label={label}
        onPointerDown={(e) => {
          if ((e.target as Element).closest(".pick")) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { px: e.clientX, py: e.clientY, x: t.x, y: t.y };
          setDragging(true);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (d) setT((p) => ({ ...p, x: d.x + e.clientX - d.px, y: d.y + e.clientY - d.py }));
        }}
        onPointerUp={() => {
          drag.current = null;
          setDragging(false);
        }}
        onPointerCancel={() => {
          drag.current = null;
          setDragging(false);
        }}
      >
        <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
          <rect width={width} height={height} fill="transparent" />
          {children}
        </g>
      </svg>
      <div className="panzoom-tools">
        <button onClick={centre}>re-centre</button>
        <span className="muted small">drag to pan · wheel to zoom · {Math.round(t.k * 100)}%</span>
      </div>
    </div>
  );
}
