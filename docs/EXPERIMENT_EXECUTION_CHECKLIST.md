# Experiment Execution Checklist — WFCRC Research Program (Paper 1)

> **Status:** Frozen operational checklist (MS11). **Version:** 1.0.
> **Date:** 2026-07-18. **Purpose:** the concrete, step-by-step gate run
> immediately before launching **any** experiment cell (a single run) or
> the full E1–E12 campaign — distinct from `docs/EXPERIMENT_PROTOCOL.md`
> (what to run and why) and `docs/PRODUCTION_READINESS_AUDIT.md` (whether
> the project as a whole is ready). This document is a checklist to
> **execute**, not a narrative to read once.
>
> **Before using this checklist for the first time on a given dataset**,
> read `docs/PRODUCTION_READINESS_AUDIT.md`'s per-dataset row — several
> Phase-A datasets are not yet loader-ready (Task 2 of that document); this
> checklist assumes the dataset/model/loader in question already exist and
> checks that they are *correctly* set up, not that they exist at all.

---

## 0 · How to use this checklist

- Run every applicable section **before** the first calibration call of a
  new experiment cell (a specific dataset × family × α × seed
  combination) and **before** the full campaign's first cell.
- A section marked **[per-cell]** must be re-checked for every new cell;
  a section marked **[per-campaign]** is checked once per campaign launch
  (not re-run for every one of the `R=100` resamples).
- Any ✗ in this checklist is a **stop**: do not proceed to calibration
  until it is resolved. Record the resolution in the run's own log
  (§8), never silently.

---

## 1 · Dataset Verification `[per-cell]`

1. ☐ Confirm the dataset's raw archive is extracted at the path its
   `DatasetLoader` expects (`docs/PRODUCTION_READINESS_AUDIT.md` Task 2's
   own per-dataset "Directory layout" column is the reference; do not
   assume a path from memory).
2. ☐ Confirm a concrete `DatasetLoader` for this dataset is registered:
   ```python
   from wfcrc.datasets.registry import DATASETS
   assert "<dataset_name>" in DATASETS
   ```
   If not registered, **stop** — this is a Task-2-identified engineering
   gap (`docs/PRODUCTION_READINESS_AUDIT.md`), not a data problem.
3. ☐ Load the `train`/`calibration`/`test` splits and confirm the A1
   hygiene gate passes (it is enforced automatically at `SplitManifest`
   construction, but confirm no exception was raised/swallowed):
   ```python
   loader = DATASETS["<dataset_name>"](...)
   cal_dataset = loader.load("calibration")
   test_dataset = loader.load("test")
   ```
   A `SplitLeakageError` here is a **stop** — the split-manifest input
   itself is wrong; do not proceed with a rebuilt or "corrected" manifest
   without recording why the original was wrong.
4. ☐ Confirm `len(cal_dataset)`/`len(test_dataset)` match the ids the
   `SplitManifest`/split-generation record for this run declares — a
   silent count mismatch (e.g. a discovery bug silently dropping ids)
   must be caught here, before calibration, not inferred later from an
   oddly-small `n_a`/`n_b`.
5. ☐ For a **new** dataset (first real run against it): perform the
   MS6.3A-equivalent real-data validation pass before trusting it at
   experiment scale — duplicate ids, NaN/Inf values, image/label shape
   match, label-value range, and (per
   `docs/DATASET_SPLIT_POLICY.md` §3.2's own flagged item) non-degenerate
   per-class representation for imbalanced datasets (MSD-Pancreas
   specifically). This is a one-time-per-dataset check, not per-cell —
   but it must have happened at least once before this checklist's
   per-cell steps are trusted.
6. ☐ Confirm the split ratio/unit actually used matches
   `docs/DATASET_SPLIT_POLICY.md` §3's frozen per-dataset policy exactly
   (or, for Kvasir-SEG, that its still-open split-unit question has been
   explicitly resolved and recorded before this run, not worked around
   silently).

## 2 · Checkpoint Verification `[per-cell, cached per model_fingerprint]`

1. ☐ Confirm the checkpoint file exists at the path the `ScoreProvider`
   expects; a missing file must raise `FileNotFoundError` (already
   verified behavior, `docs/PILOT_REPORT.md` §6) — if it does not raise
   cleanly, **stop**, this is a defect, not a data problem.
2. ☐ Compute `model_fingerprint()` and record it in this run's own log —
   two runs sharing a fingerprint should be treated as using the
   byte-identical checkpoint file; two runs of "the same" checkpoint with
   *different* fingerprints (e.g. after any re-save) are **not**
   guaranteed identical weights (`docs/MODEL_POLICY.md` §3's own disclosed
   caveat) and must not be silently treated as interchangeable.
3. ☐ **Checkpoint provenance (mandatory for any externally-sourced
   checkpoint, `docs/MODEL_POLICY.md` §4):**
   ```python
   from wfcrc.models.checkpoint import ... # CheckpointProvenance / assert_no_checkpoint_leakage
   assert_no_checkpoint_leakage(provenance, cal_ids=cal_dataset.ids(), test_ids=test_dataset.ids())
   ```
   A `CheckpointProvenanceError` here is a **stop** — this checkpoint's
   own training data overlaps this run's calibration/test pool; do not
   proceed with this checkpoint for this dataset under any circumstance.
4. ☐ For the MSD-Hippocampus never-trained smoke checkpoint specifically:
   confirm this run is **not** being reported as a scientific result
   (`docs/MODEL_POLICY.md` §2's explicit warning) — if it is, **stop**;
   a real, trained/pretrained checkpoint is required first.
5. ☐ Confirm the model is in inference mode (`model.eval()`, no
   dropout/batchnorm-update path active) and that no optimizer/gradient
   step is reachable anywhere in the call path for this run.

## 3 · Cache Verification `[per-cell]`

1. ☐ Confirm the score cache directory (`Cache(cache_dir)`) is writable
   and, for a **new** experiment cell, is either empty or contains only
   entries this specific `model_fingerprint` produced (a stale cache
   directory shared across different checkpoints without a fingerprint
   check would silently serve wrong scores — the frozen `make_key`
   already keys on `(model_fingerprint, id_)`, so this is enforced by
   construction as long as the *same* cache directory is not reused
   across genuinely different fingerprints in a way that collides keys;
   confirm it is not).
2. ☐ For a **resumed** or re-run cell: confirm cache hits are actually
   occurring for previously-scored ids (a cache-miss-every-time symptom
   indicates a fingerprint or key mismatch, not normal behavior) —
   `docs/PILOT_REPORT.md` §4's own cache-hit-rescoring measurement is the
   template for this check.
3. ☐ **Corruption check:** if a cache entry fails to load
   (`CacheError`), do **not** silently delete and re-cache it as a
   default recovery — first record which key/id was affected and confirm
   the corruption is a filesystem issue, not a symptom of a concurrent
   write (this project's `Cache` is explicitly single-process-only, per
   its own module docstring — never run two processes against the same
   cache directory concurrently).
4. ☐ Confirm the `LossTableBuilder`/`Checkpointer` stage caches
   (`wfcrc.runner.checkpointer.Checkpointer`, if using `ExperimentRunner`)
   are scoped to this run's own `run_dir` — never point two different
   experiment cells at the same `run_dir`.

## 4 · Random Seed Verification `[per-campaign, then per-cell for the resample sub-seed]`

1. ☐ Confirm the **campaign base seed** has been fixed and recorded
   (`docs/PRODUCTION_READINESS_AUDIT.md` Task 3, item 2 — a recommended
   default of `0` is on record pending sign-off; confirm whatever value
   is actually adopted is written down before E1 executes, not chosen ad
   hoc per script invocation).
2. ☐ For each of the `R=100` resamples in a cell, confirm the resample's
   own seed is **derived** from the campaign base seed via
   `wfcrc.utils.seeds.derive_seed` (or an equivalent documented fanout) —
   never a bare, independently-chosen integer per resample.
3. ☐ Confirm no bare `numpy.random.*`/`torch.manual_seed`-outside-of-
   `create_untrained_checkpoint`-style call exists anywhere in the
   run's own code path (project-wide lint policy, `PROJECT_CONTEXT.md`
   §9) — a `ruff`/manual grep for `np.random.seed`, `np.random.rand`,
   bare `random.` calls outside `wfcrc.utils.seeds` is a fast, mechanical
   version of this check.
4. ☐ Confirm determinism directly: run the calibration step twice with
   the identical seed and confirm `lambda_hat`/`empty_flag` match exactly
   (the same check `scripts/pilot_ms10.py`'s `run_one_baseline` performs
   for every baseline) — do this **before** trusting a full `R=100` sweep,
   not only after.

## 5 · Dependency Verification `[per-campaign]`

1. ☐ Confirm the environment matches `requirements/lock.txt` exactly
   (`make install-locked`, or manually diff `pip freeze` against it) —
   this is the artifact that lets a specific past result be reproduced in
   the exact environment that produced it (`docs/reproducibility.md` §5).
2. ☐ Run the full quality gate before the campaign's first real cell:
   ```bash
   ruff check wfcrc tests scripts
   black --check wfcrc tests scripts
   mypy wfcrc scripts
   python -m pytest          # full suite + coverage
   python scripts/reproduce.py   # MS5 golden-file check
   ```
   Any failure here is a **stop** — do not launch an experiment on top of
   a broken quality gate, even if the specific breakage looks unrelated
   to the experiment about to run.
3. ☐ Confirm the dependency closure includes every package the specific
   dataset/model pairing about to run needs (e.g. `nibabel` for MSD,
   `imagecorruptions`/`scikit-image` for Cityscapes-C, `torch` for any
   `ScoreProvider`) — a missing optional dependency should fail at import
   time with a clear `ImportError`, not partway through a long-running
   calibration.
4. ☐ Record the exact git commit (`wfcrc.utils.reproducibility.get_git_commit()`)
   this run's code corresponds to — already captured automatically in
   every `Manifest`, but confirm it is not `None` (a `None` commit means
   the run happened outside a git checkout, or with uncommitted local
   changes that later become unreproducible).

## 6 · Expected Outputs `[per-cell]`

Per `docs/RESULTS_SCHEMA.md`, confirm the run is configured to produce:

1. ☐ `results/E<n>/<run_id>/manifest.json` (or, for a pilot/smoke run,
   the equivalent `results/pilot/<name>/manifest.json` shape) —
   `config_hash`, `seed`, `lambda_hat`, `empty_flag`, `n_a`/`n_b`/
   `b_tilde`/`r_hat_b` (where applicable to the calibrator), `diagnostics`,
   `metrics`, `verification_passed`, `figure_paths`.
2. ☐ For dual-family WF-CRC cells specifically: `figures/g_curve.pdf`+
   `.csv` (via `ExperimentRunner`, if used) — confirm the `.csv` sidecar's
   `source_hash` matches what actually produced the figure (already a
   frozen guarantee of `wfcrc.visualization.base.render_figure`; confirm
   it was not bypassed by a hand-written plotting shortcut).
3. ☐ For an aggregated multi-resample cell: the `docs/RESULTS_SCHEMA.md`
   §2.2 table row shape (`experiment`, `table`, `dataset`, `family`,
   `alpha`, `metric`, `mean`, `ci_lo`, `ci_hi`, `ci_level`, `n_resamples`,
   `stat_*` columns, `config_hash_set`) — confirm every column that
   should be populated is, and every column that should be blank (e.g.
   `stat_*` for a non-comparison row) is blank, not silently `0`/`NaN`
   in a way indistinguishable from a real zero-effect result.

## 7 · Result Validation `[per-cell]`

1. ☐ **For `wfcrc`/`vanilla_crc` cells:** run the frozen `Verifier` and
   confirm `VerificationReport.passed`. A failing check here is the
   project's own STOP-gate — do not checkpoint, aggregate, or report any
   metric from a cell whose verification failed
   (`ExperimentRunner`'s own behavior if used; replicate it manually
   otherwise).
2. ☐ **For every other registered baseline** (no generic `Verifier`
   applies, `docs/PRODUCTION_READINESS_AUDIT.md` Task 3 item 9): confirm
   at minimum — `lambda_hat` is a member of the configured `lambda_grid`;
   the calibration call is deterministic across two identical-seed calls;
   every reported metric is finite (no `NaN`/`inf`); and, where
   applicable, `effective_sizes`'s reported `n_a`/`n_b`/`n_g_*` sum
   consistently with the calibration input's own `n`.
3. ☐ Confirm `realized_marginal_risk` (and, for dual families,
   `realized_worst_case_risk`) were computed against the **held-out test**
   `LossTable` only — never the calibration table, and never a table that
   shares any id with calibration (this is what the A1 gate + this
   checklist's §1 step 3 jointly guarantee; re-confirm here that the
   *specific* table object passed to the metric function is the test one,
   not a copy-paste error passing `cal_loss_table` twice).
4. ☐ Confirm the statistical tests/CIs actually run
   (`docs/EXPERIMENT_PROTOCOL.md` §5) are exactly the ones pre-registered
   for this experiment — no test substitution or parameter change after
   seeing this cell's own numbers (§5.6's own post-hoc-discipline rule).

## 8 · Post-Run Verification `[per-cell, and once more per-campaign at the end]`

1. ☐ Confirm the run's manifest was actually written (not merely
   computed and discarded by a crashed process) and is valid JSON
   round-tripping through `Manifest.from_dict(Manifest.to_dict())`-style
   equality.
2. ☐ Confirm re-running the identical cell (same config, same seed)
   reproduces byte-identical `lambda_hat`/`empty_flag`/metrics — the same
   discipline `scripts/reproduce.py` already exercises for the MS5
   synthetic reference experiment, applied here to a real cell.
3. ☐ Append this cell's result to the campaign manifest
   (`docs/RESULTS_SCHEMA.md` §3.2) — never leave a completed cell
   un-indexed, even if its own `manifest.json` exists on disk.
4. ☐ **At the end of the full campaign:** re-run `python scripts/reproduce.py`
   once more, to confirm the campaign's own work did not somehow regress
   the frozen MS1–MS9 core (it should not have, since no frozen file
   should have been touched — this is a final, mechanical confirmation
   of that invariant, not a new check).
5. ☐ Cross-check every reported table (T1–T7) against
   `docs/RESULTS_SCHEMA.md` §7's aggregation-order rule (max-over-group
   **after** per-resample aggregation, not before) — a script that
   silently computes it the other way produces a plausible-looking but
   wrong number, exactly the failure mode that section warns about.

---

## Connections

`docs/EXPERIMENT_PROTOCOL.md` · `docs/MODEL_POLICY.md` ·
`docs/RESULTS_SCHEMA.md` · `docs/DATASET_SPLIT_POLICY.md` ·
`docs/PRODUCTION_READINESS_AUDIT.md` · `docs/PILOT_REPORT.md` ·
`scripts/pilot_ms10.py` · `PROJECT_CONTEXT.md`
