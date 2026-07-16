# wfcrc

**Worst-case Family Conformal Risk Control** — a reproducible research repository.

Built incrementally against a frozen set of engineering specifications
(Framework Specification, Mathematical Specification v2.1, Algorithm
Specification, Implementation Blueprint, Experiment Blueprint). See
[`docs/`](docs/index.md) for full documentation.

## Milestone status

**MS1-MS5 are complete and frozen.** This repository implements the full
frozen Implementation Blueprint (utils/config through the experiment
runner) short of concrete datasets/models for any real, named dataset —
see below.

- **MS1: core infrastructure.** Deterministic hashing and atomic
  serialization (`utils.io`), numerically stable primitives
  (`utils.numerics`), seeded RNG fanout (`utils.seeds`), structured JSONL
  logging (`utils.logging`), a content-addressed cache (`utils.cache`), git
  commit / environment fingerprinting (`utils.reproducibility`), and a
  strict, typed, hashable, layered configuration system (`config`).
- **MS2: core algorithm.** Bounded monotone task losses
  (`losses`: `FNRLoss`/`FPRLoss`/`MiscoverageLoss`), the four frozen
  ambiguity families (`ambiguity`: `CVaRFamily`/`KLFamily`/
  `FiniteGroupFamily`/`KnownWeightFamily`), and single-split WF-CRC
  calibration (`calibration`: `Splitter`/`ThresholdSearch`/
  `WFCRCCalibrator`).
- **MS3: prediction sets & calibration pipeline.** Nested
  prediction-set constructors (`prediction_sets`: `ThresholdSets`/
  `MorphologicalSets`), the deterministic AS §20 checklist
  (`evaluation.Verifier`), and a thin `LossTable → WFCRCCalibrator →
  optional Verifier` orchestration (`calibration.pipeline`).
- **MS4: datasets contracts, metrics & experiment execution.** Abstract
  data-loading/score-provider contracts plus a concrete loss-table
  assembler (`datasets`: `Dataset`/`DatasetLoader`/
  `ScoreProvider`/`LossTableBuilder`), realized-risk and statistical
  utilities (`evaluation.metrics`: realized worst-case/marginal risk,
  per-group risk, set size, coverage, effective sizes, duality gap,
  bootstrap CIs, one-sided risk test, paired Wilcoxon, Holm correction),
  and a reduced experiment-execution report
  (`evaluation.experiment.run_experiment`).
- **MS5: visualization, experiment runner & reproducibility.**
  Deterministic figure rendering (`visualization`: `plot_g_curve` and the
  paper's F1-F8 figures, each with a byte-reproducible `.pdf`/`.svg` plus a
  `.csv` data sidecar), and a config-driven experiment runner
  (`runner.ExperimentRunner`/`Checkpointer`: calibrate → verify → metrics →
  plot → manifest, with checkpointing, sweeps, and resume). `make reproduce`
  re-runs a fixed-seed reference experiment and diffs it against a
  committed golden file. See
  [CLAIMS_TRACEABILITY.md](CLAIMS_TRACEABILITY.md) for implementation-level
  deviations/gap-fills recorded against the frozen specs, milestone by
  milestone.

**Not yet built (out of scope for MS1-MS5):** concrete datasets/models for
any real, named dataset (Cityscapes, MSD, CIFAR, ...) — no such data or
checkpoints are present in this environment. `ExperimentRunner` and
`run_experiment` both operate on already-built loss tables directly rather
than resolving a dataset/model from configuration; wiring in a real dataset
is the vault's own next milestone (MS6, experiment execution).

## Quick start

```bash
pip install -e ".[dev]"
make test        # pytest (unit suite + coverage)
make lint         # ruff + black --check
make typecheck    # mypy --strict
make reproduce    # re-run the reference experiment, diff against the golden file
```

See [docs/getting-started.md](docs/getting-started.md) for details, and
[docs/configuration.md](docs/configuration.md) /
[docs/reproducibility.md](docs/reproducibility.md) for how the configuration
and reproducibility systems work.

## Requirements

Python 3.12+. See [`pyproject.toml`](pyproject.toml) for dependency ranges
and [`requirements/lock.txt`](requirements/lock.txt) for the exact pinned
environment (`make install-locked`).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, coding standards, and
the pre-PR checklist. See [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE).
