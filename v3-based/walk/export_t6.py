# Export B3's closed set from t6_octonion.py's own output (its closed[0]).
# Writes t6_units.bin (240 x 8 signed bytes, doubled coords, units sorted)
# and t6_table.bin (240 x 240 u8: table[a*240+b] = index of units[a]*units[b]).
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "..", "bundle", "scripts", "t6_octonion.py")
exec(open(src).read())                      # runs T6 as registered, prints its lines
S, U, _ = closed[0]
units = sorted(U)
pos = {u: i for i, u in enumerate(units)}
table = bytes(pos[tuple(x // 2 for x in omul2(u, v))] for u in units for v in units)
with open(os.path.join(here, "t6_units.bin"), "wb") as f:
    f.write(bytes(x & 0xFF for u in units for x in u))
with open(os.path.join(here, "t6_table.bin"), "wb") as f:
    f.write(table)
print("exported closed[0]: blocks", sorted(sorted(b) for b in S))
print("units", len(units), "table bytes", len(table))
