import { useState } from "react";
import { DimensionTab } from "./DimensionTab";
import { EightD } from "./EightD";
import { NineD } from "./NineD";
import { FiveD } from "./FiveD";
import { FourD } from "./FourD";
import { SevenD } from "./SevenD";
import { SixD } from "./SixD";

const TABS = [
  { dim: 2, title: "2D", note: "tables allowed" },
  { dim: 3, title: "3D", note: "cubes allowed" },
  { dim: 4, title: "4D", note: "planes · third coordinate read two ways" },
  { dim: 5, title: "5D", note: "accumulator · pentagonal cylinder" },
  { dim: 6, title: "6D", note: "three planes, each two-sided" },
  { dim: 7, title: "7D", note: "rotation · the missing centre" },
  { dim: 8, title: "8D", note: "the inside has coordinates" },
  { dim: 9, title: "9D", note: "space · focus and going" },
] as const;

type Dim = (typeof TABS)[number]["dim"];

export function App() {
  const [dim, setDim] = useState<Dim>(9);
  return (
    <div className="app">
      <header>
        <h1>Manifold Matrix</h1>
        <p className="sub">walks read from data/mm.sqlite · order matters · every direction +</p>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button key={t.dim} role="tab" aria-selected={dim === t.dim} className={dim === t.dim ? "tab on" : "tab"} onClick={() => setDim(t.dim)}>
              <span className="tab-title">{t.title}</span>
              <span className="tab-note">{t.note}</span>
            </button>
          ))}
        </nav>
      </header>
      <main>{dim === 4 ? <FourD /> : dim === 5 ? <FiveD /> : dim === 6 ? <SixD /> : dim === 7 ? <SevenD /> : dim === 8 ? <EightD /> : dim === 9 ? <NineD /> : <DimensionTab key={dim} dimension={dim} />}</main>
    </div>
  );
}
