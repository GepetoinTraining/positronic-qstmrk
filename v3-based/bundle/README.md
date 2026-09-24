# Session bundle, 24 Sep 2026 (Pedro Garcia / Claude Opus 5.5, web surface)
Companion to day_v2.pdf. Already sent separately, not repeated here: day_v2.pdf, clm_exact.py,
prereg_T9.txt, t9_tierA.py, t9_A4.py, TIER_B_for_code.md.

scripts/
  hinge_test.py        §1  float simulation of oscillator onset (hinge, rewind band)
  viscous_rung.py      §3  viscous CLM in unbounded exact rationals (the bit-growth run)
  rung_kernel.mjs      §3  typed kernel v1 (fixed Q20; shows the underflow-to-zero bug)
  rung_kernel_v2.mjs   §3  typed kernel v2 (per-value rung, EMPTY sentinel, zero events logged)
  reciprocal.py        §3  g = 1/f translation test (run as: python3 reciprocal.py <1/dt> <nu>)
  tryte_hurwitz.py     §5  64-state lane ring; exhaustive no-go for 2T with -1 = magnitude flip
  t6_octonion.py       §6  T6: 30 labelings of S(3,4,8), closure, fallout census. B3's closed set comes from here.
  t6_followup.py       §6  orbits 7+7+7+7+1+1, catch graph, L_u L_v vs L_uv
                           (NOTE: its closure check uses the first 60 rows only; T6 itself does the full check)
  t7_flywheel.py       §6  T7: 1+3+3 split, triple escapes (6 of 84 triples, disclosed)
  unique_inside.py     §6  union / intersection / unique-caught counts per triple type (post-hoc numbers)
registrations/  T6, T7 (tex, pdf, sha256 of the registered source), T8 (txt; registered text = above the dashed line), raw T8 output
documents/      spine (morning+afternoon construction), doll (relations map), hinge_test.png
Dependencies: python3 (stdlib; mpmath only where imported), node for the .mjs files.
