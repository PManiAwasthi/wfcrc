# Configuration

wfcrc runs are described by a single, strictly validated, immutable
[`Config`][wfcrc.config.schema.Config] object, built by
[`load_config`][wfcrc.config.loader.load_config] from one or more layered
YAML files plus optional CLI overrides.

## Layering

```python
from wfcrc.config import load_config

config = load_config(
    paths=["configs/default.yaml", "configs/dataset_x.yaml"],
    overrides={"calibration.alpha": 0.05},
)
```

Layers are merged left to right — later files win on conflicting keys — and
`overrides` (flat dotted keys) are applied last, on top of every file layer.
Nested sections are deep-merged, so a later layer can override just
`calibration.alpha` without having to repeat `calibration.B`, `.pi`, and
`.lambda_grid`.

## Schema

The top-level sections are `data`, `model`, `sets`, `loss`, `family`,
`calibration`, `runner`, and `seed`. `data`/`model`/`sets`/`loss` are
intentionally generic (`{name, params}`), a shape fixed in MS1 before the
registries that resolve `name` to a concrete implementation existed.
`wfcrc.prediction_sets.SETS` and `wfcrc.losses.LOSSES` (built in MS2/MS3)
now resolve `sets.name`/`loss.name` to a concrete class; `data.name`/
`model.name` still have no registry and no concrete implementation for any
real, named dataset — that requires actual data/checkpoints not present in
this environment (see `CLAIMS_TRACEABILITY.md` §11). `family` and
`calibration` carry the concrete, validated fields the frozen
Mathematical/Algorithm specifications require:

- `calibration.alpha`, `calibration.B`: `0 < alpha < B`.
- `calibration.pi`: `0 < pi < 1`.
- `calibration.lambda_grid`: a strictly increasing, non-empty list of numbers.
- `family.type`: one of `cvar`, `kl`, `finite_group`, `known_weight`; each
  requires exactly one further field (`beta`, `rho`, `masks`, `weights`
  respectively).

Validation is strict: unknown keys, missing required fields, and
numeric-looking strings (e.g. `alpha: "0.1"`) are all rejected with a
[`ConfigError`][wfcrc.exceptions.ConfigError] naming the offending field.

## Provenance hash

```python
config.hash()      # stable hex digest of the resolved configuration
config.to_yaml()    # canonical YAML re-serialization (round-trips through load_config)
config.get("calibration.alpha")  # dotted-path lookup
```

`Config.hash()` is invariant to field/key ordering and reproducible across
processes, so it can be embedded in a run manifest as a provenance
fingerprint (see [Reproducibility](reproducibility.md)).
