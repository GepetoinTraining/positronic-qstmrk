ARCHIVE — mistaken T12 and T13 (Claude, 25 Sep 2026). Pedro's to delete, the whole folder.
code/     t12.rs, t13.rs and rc.rs (the range coder they shared). Removed from the walk crate.
runs/     every output of both runs, including the void first T12 run, and the two sha files that sealed the
          withdrawn registrations.
weights/  every weight copy Claude made: the 159 tensors fetched by byte range into t10/ (1.4 GB, BF16 LE,
          admitted against v1's MANIFEST sha256), used by T10, T11, T12 and T13. Gitignored, never committed.
          The source safetensors were not touched. t10/fetch.py can fetch them again.
          27 Sep 2026: weights/ is now the local Qwen3-1.7B (layers 0-13, 5 tensors of 14) that walk/src/bin/geo.rs
          reads. Move it before deleting this archive (see ../BUILDER_LOG.md).
