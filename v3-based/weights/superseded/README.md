# Superseded, 27 Sep 2026

These are kept as the record, not as the method. The method is `v3-based/walk/src/bin/geo.rs`; see `v3-based/BUILDER_LOG.md`.

| file | what it did | why it is superseded |
|---|---|---|
| `read15.py` | 15-band read, 256 plane products per projection, redone for 10 regroupings | brute force; the reading measured the random input (about 15 = the band count), not the weights; stopped |
| `rules.py` | layer from rules, exact products carried by BLAS below 2^53 | still multiply-accumulate; libraries for something that needs none |
| `real17.py`, `geo.py` | first real forward and first landing read, numpy | activations rounded to 16-bit positions; checks against a rounded format; libraries |
| `results_geo_rs*.txt`, `results_geo_octet*.txt` | earlier geo.rs runs | carried a reference gate and 16-bit activation rounding, both removed |
