# Architecture

## Design principles

- **Clean architecture / SOLID.** Each module has one responsibility
  (hashing, numerics, seeding, logging, caching, configuration) and depends
  only on what it needs; higher-level modules (config) depend on lower-level
  ones (io, logging), never the reverse.
- **Minimal coupling, high cohesion.** `utils.numerics` and `utils.seeds`
  have no dependency on `config` or each other; `config` depends only on
  `utils.io`.
- **Strong typing.** Every public function is fully type-hinted; `mypy
  --strict` is a CI gate.
- **Deterministic execution.** No hidden global state, no bare
  `numpy.random` calls, canonical serialization for hashing — see
  [Reproducibility](reproducibility.md).
- **Structured exceptions.** Every intentional failure raises a subclass of
  [`WFCRCError`][wfcrc.exceptions.WFCRCError], never a bare built-in
  exception, so calling code can distinguish "wfcrc rejected this input" from
  "something else went wrong."

## MS1 module dependency graph

```text
io ──────────► cache
io ──────────► config ◄──── logging
numerics    seeds     logging      (independent of config; consumed later)
CI ── wraps ──► {all MS1 test suites}
```

`utils.io` is the root utility (canonical hashing + atomic serialization).
`utils.cache` and `wfcrc.config` build on it; `utils.numerics`,
`utils.seeds`, and `utils.logging` are independent leaves that later
milestones (families, calibration, runner) will depend on.

## MS1 package layout

```text
wfcrc/
  wfcrc/
    __init__.py
    _version.py
    constants.py
    exceptions.py
    utils/
      io.py              # C1: hashing, atomic writes, JSON/array serialization
      numerics.py        # C2: logsumexp, clamp, safe_div, quantile
      seeds.py           # C3: deterministic RNG fanout
      logging.py         # C4: structured JSONL run logging
      cache.py           # C5: content-addressed read-through cache
      reproducibility.py # git commit + environment fingerprint
    config/
      schema.py          # C6: typed, immutable Config dataclasses
      loader.py           # C6: layered YAML loading + strict validation
    # ambiguity/, calibration/, datasets/, evaluation/, losses/, models/,
    # prediction_sets/, visualization/ are placeholders for MS2+; MS1 does
    # not implement anything inside them.
  configs/
    default.yaml
  tests/
    unit/utils/  unit/config/
  docs/
  .github/workflows/ci.yml
  Makefile
  pyproject.toml
```

## Blueprint layout vs. actual repository layout

The frozen Implementation Blueprint (§1) specifies a `src/wfcrc/` layout
with subpackages named `data/`, `sets/`, `losses/`, `families/`,
`calibration/`, `verify/`, `metrics/`, `viz/`, `runner/`. The repository's
actual scaffold — created before MS1 and treated as fixed context per the
project brief ("the folder structure already exists... your task is not to
redesign anything") — uses a **flat** layout (`wfcrc/wfcrc/`, no `src/`)
with placeholder directories under some **different names**:

| Blueprint (§1) | Actual repository | Status in MS1 |
|---|---|---|
| `src/wfcrc/data/` | `wfcrc/datasets/` | empty placeholder |
| `src/wfcrc/sets/` | `wfcrc/prediction_sets/` | empty placeholder |
| `src/wfcrc/losses/` | `wfcrc/losses/` | empty placeholder |
| `src/wfcrc/families/` | `wfcrc/ambiguity/` | empty placeholder |
| `src/wfcrc/calibration/` | `wfcrc/calibration/` | empty placeholder |
| `src/wfcrc/metrics/` | `wfcrc/evaluation/` | empty placeholder |
| `src/wfcrc/viz/` | `wfcrc/visualization/` | empty placeholder |
| `src/wfcrc/verify/` | *(no equivalent yet)* | not yet created |
| `src/wfcrc/runner/` | *(no equivalent yet)* | not yet created |
| — | `wfcrc/models/` | empty placeholder, no Blueprint equivalent |
| `src/wfcrc/config/`, `src/wfcrc/utils/` | `wfcrc/config/`, `wfcrc/utils/` | **implemented in MS1**, names match |

**Resolution: this is a documented, deliberate divergence, not a defect to
silently fix later.** MS1 did not rename any of these directories and did
not implement anything inside them (all remain empty, per MS1's
infrastructure-only scope). When a later milestone (MS2 for
`data`/`sets`/`losses`, MS3 for `families`/`calibration`, MS4 for
`verify`/`metrics`) begins implementing one of these subpackages, that
milestone's own implementation spec should confirm the target directory
name against this table rather than assuming the Blueprint's name is
literal. Renaming the placeholder directories to match the Blueprint
exactly would be a safe, low-risk cleanup if done *before* any module
inside them is implemented — but it is out of scope for a documentation
reconciliation and was not judged necessary to resolve here, since an empty
directory carries no behavior to break either way.

## Why `data`/`model`/`sets`/`loss` config sections are generic in MS1

The frozen Implementation Blueprint defines concrete registries
(`SETS`, `LOSSES`, `FAMILIES`) that map config strings to classes, but those
classes do not exist yet — building them is MS2/MS3 scope. MS1's `Config`
schema therefore validates the *shape* every later registry entry must have
(`{name: str, params: dict}`) without committing to which names are valid,
so the config system does not need to change when those registries land.
`family` and `calibration`, by contrast, have concrete fields today because
the frozen Mathematical/Algorithm specifications already fix their meaning
independent of which family/dataset is chosen.
