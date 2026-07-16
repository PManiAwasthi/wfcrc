# Contributing

Thanks for your interest in `wfcrc`. This project is built incrementally
against a frozen set of engineering specifications, so a few conventions
here are stricter than a typical repository's.

## Ground rules

- **The specifications are frozen.** The Framework Specification,
  Mathematical Specification, Algorithm Specification, Implementation
  Blueprint, Experiment Blueprint, Engineering Verification Tracker, and
  Claims Traceability Matrix are the source of truth for *what* gets built
  and in what order. They live outside this repository and are read-only
  context. Do not propose redesigning them here; if an implementation
  genuinely can't satisfy a frozen spec, open an issue describing the
  conflict rather than silently deviating.
- **Respect milestone scope.** Each milestone (MS1, MS2, …) has an explicit
  scope. Do not implement mathematics, calibration logic, ambiguity
  families, losses, or experiments ahead of their milestone — see
  [docs/architecture.md](docs/architecture.md) for what MS1 does and
  deliberately does not include.
- **No unrequested redesigns, refactors, or optimizations.** Prefer the
  smallest change that correctly implements the current milestone's spec.

## Getting set up

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

For an exact reproduction of the environment a specific result was
produced in, use the lockfile instead:

```bash
make install-locked
```

See [docs/reproducibility.md](docs/reproducibility.md) for the distinction
between the two.

## Before opening a pull request

```bash
make lint         # ruff check + black --check
make typecheck    # mypy --strict
make test-cov      # pytest with coverage
```

All three must pass. CI runs the same checks (`.github/workflows/ci.yml`)
plus the pre-commit hygiene hooks.

## Code standards

- **Type hints everywhere.** `mypy --strict` is a hard CI gate; every
  public function must be fully annotated.
- **Docstrings on every public function/class/module**, Google style
  (`Args:`/`Returns:`/`Raises:`). Prefer no comments over comments that
  restate what the code already says; a comment should explain a
  non-obvious *why*.
- **Determinism.** No bare `numpy.random.*` calls — obtain RNGs via
  `wfcrc.utils.seeds.rng_for`. No hidden global state beyond what a module
  already documents (e.g. `utils.seeds`' global seed registry).
- **Structured exceptions.** Raise a subclass of
  `wfcrc.exceptions.WFCRCError`, not a bare built-in exception, for any
  intentional failure.
- **Tests.** Every new module needs unit tests; every new branch should be
  covered. The project maintains 100% line and branch coverage on `wfcrc/`
  through MS5 — try not to regress it, though later, more
  experiment-heavy milestones (real dataset/model integration) may
  reasonably relax this bar for integration-style code paths.

## Updating the lockfile

Only regenerate `requirements/lock.txt` after a deliberate, tested
dependency upgrade:

```bash
make lock
```

Commit the updated lockfile in the same change as the dependency bump that
caused it.

## Commit messages

Describe *why* a change was made, not just what changed — the diff already
shows what changed. Keep the subject line under ~70 characters.
