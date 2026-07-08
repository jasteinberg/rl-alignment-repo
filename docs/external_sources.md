# External sources

Third-party repositories and datasets used for reference. We do **not** commit
external code or data into this repo — only this manifest and the fetch scripts.

## Layout conventions
- **Libraries** (transformers, transformer_lens, saelens): conda env `interp`; never cloned.
- **Reusable datasets / models**: canonical copy in `~/models/` (shared store),
  symlinked into each project's `data/` or `models/`. E.g.
  `data/geometry_of_truth -> ~/models/datasets/geometry_of_truth`.
- **Third-party code to read or borrow**: clone into `~/external/<repo>` (shared,
  read-only), record the commit below. If borrowing a function, copy it into
  `utils/` with a comment naming the source repo, commit, and license.
- Record every external source here (URL, commit, why, license).

## Registry

### Geometry of Truth — Marks & Tegmark (2023)
- **Code**: `~/external/geometry-of-truth`
  - source: https://github.com/saprmarks/geometry-of-truth
  - commit: `5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb` (main)
  - why: reference for mass-mean / logistic probes and the truth-direction experiments
  - license: **no LICENSE file present** — reimplement rather than copy verbatim; cite the paper.
- **Datasets**: `~/models/datasets/geometry_of_truth` (symlinked to `data/geometry_of_truth`)
  - fetched via `scripts/fetch_geometry_of_truth.py`
  - files: cities, neg_cities, larger_than, smaller_than, sp_en_trans, neg_sp_en_trans, common_claim_true_false, ...
