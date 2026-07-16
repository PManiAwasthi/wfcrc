"""``make reproduce`` — re-run the reference experiment and diff against the golden file.

Per the Implementation Blueprint (§17) and the MS5 Implementation
Specification (C2 item 5, "acceptance"): "a `make reproduce` target re-runs
a reference experiment and diffs metrics against a stored golden file
within tolerance." No real dataset/model is available in this environment
(`wfcrc.datasets` is ABC-contracts-only, per the MS4 scope decision), so the
"reference experiment" here is a small, fully deterministic *synthetic*
calibration/test `LossTable` pair built from a fixed seed — the same
recipe every time, with no external data dependency. This is the same
scope reduction already applied throughout MS5 (`ExperimentRunner.run`
itself takes already-built `LossTable`s rather than resolving a config-driven
dataset), not a new one.

Usage::

    python scripts/reproduce.py            # compare against the committed golden file
    python scripts/reproduce.py --write-golden   # (re)generate the golden file (dev-only)

Exit code `0` if the fresh run matches the golden file within tolerance,
`1` otherwise (with a diff printed to stderr).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import (
    CalibrationConfig,
    Config,
    DataConfig,
    FamilyConfig,
    LossConfig,
    ModelConfig,
    RunnerConfig,
    SetsConfig,
)
from wfcrc.runner.runner import ExperimentRunner
from wfcrc.utils.io import load_json, save_json

__all__ = ["GOLDEN_PATH", "REFERENCE_SEED", "build_reference_inputs", "compare_to_golden", "main"]

#: Committed golden-file location (repo-relative).
GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "reproduce_golden.json"
)

#: Fixed seed for the reference experiment's synthetic loss tables and calibration split.
REFERENCE_SEED = 0

#: Absolute tolerance for comparing golden vs. fresh numeric fields.
_TOLERANCE = 1e-9


def _synthetic_loss_table(
    n: int, seed: int, lambda_max: float = 0.9, n_lambda: int = 11
) -> LossTable:
    """A small, deterministic, monotone-non-increasing-in-lambda loss table."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 1.0, size=n)
    lambda_grid = np.linspace(0.0, lambda_max, n_lambda)
    values = np.outer(base, (1.0 - lambda_grid))
    return LossTable(values=values, lambda_grid=lambda_grid)


def build_reference_inputs() -> tuple[Config, LossTable, LossTable]:
    """Build the reference experiment's `Config` and calibration/test `LossTable`s.

    `Config.runner.cache_dir` is a fixed, relative string (not tied to any
    per-invocation temp directory) so `Config.hash()` — and therefore
    `Manifest.config_hash` — is itself bit-for-bit reproducible across
    `make reproduce` runs, not just the numeric calibration outputs.

    Returns:
        `(config, cal_loss_table, test_loss_table)`.
    """
    cal_loss_table = _synthetic_loss_table(n=80, seed=REFERENCE_SEED)
    test_loss_table = _synthetic_loss_table(n=60, seed=REFERENCE_SEED + 1)
    lambda_grid = tuple(float(x) for x in cal_loss_table.lambda_grid)

    config = Config(
        data=DataConfig(name="synthetic_reference"),
        model=ModelConfig(name="synthetic_reference"),
        sets=SetsConfig(name="threshold"),
        loss=LossConfig(name="miscoverage"),
        family=FamilyConfig(type="cvar", beta=0.2),
        calibration=CalibrationConfig(alpha=0.3, B=1.0, pi=0.5, lambda_grid=lambda_grid),
        runner=RunnerConfig(cache_dir="cache/reproduce_reference", log_level="INFO"),
        seed=REFERENCE_SEED,
    )
    return config, cal_loss_table, test_loss_table


def compare_to_golden(fresh: dict[str, Any], golden: dict[str, Any]) -> list[str]:
    """Diff a fresh manifest dict against the golden one, within `_TOLERANCE`.

    Args:
        fresh: `Manifest.to_dict()` output from a fresh reference run.
        golden: The previously stored golden manifest dict.

    Returns:
        A list of human-readable mismatch descriptions; empty if the two
        match within tolerance.
    """
    mismatches: list[str] = []
    exact_keys = ("config_hash", "family_type", "empty_flag", "verification_passed")
    numeric_keys = ("lambda_hat", "n_a", "n_b", "b_tilde", "r_hat_b")

    for key in exact_keys:
        if fresh.get(key) != golden.get(key):
            mismatches.append(f"{key}: fresh={fresh.get(key)!r} != golden={golden.get(key)!r}")

    for key in numeric_keys:
        fresh_value, golden_value = fresh.get(key), golden.get(key)
        if fresh_value is None or golden_value is None:
            if fresh_value != golden_value:
                mismatches.append(f"{key}: fresh={fresh_value!r} != golden={golden_value!r}")
            continue
        if abs(float(fresh_value) - float(golden_value)) > _TOLERANCE:
            mismatches.append(
                f"{key}: fresh={fresh_value!r} != golden={golden_value!r} (tol={_TOLERANCE})"
            )

    fresh_metrics = fresh.get("metrics", {})
    golden_metrics = golden.get("metrics", {})
    for key in sorted(set(fresh_metrics) | set(golden_metrics)):
        fresh_value = fresh_metrics.get(key)
        golden_value = golden_metrics.get(key)
        if isinstance(golden_value, dict) or isinstance(fresh_value, dict):
            if fresh_value != golden_value:
                mismatches.append(
                    f"metrics.{key}: fresh={fresh_value!r} != golden={golden_value!r}"
                )
            continue
        if fresh_value is None or golden_value is None:
            if fresh_value != golden_value:
                mismatches.append(
                    f"metrics.{key}: fresh={fresh_value!r} != golden={golden_value!r}"
                )
            continue
        if abs(float(fresh_value) - float(golden_value)) > _TOLERANCE:
            mismatches.append(
                f"metrics.{key}: fresh={fresh_value!r} != golden={golden_value!r} "
                f"(tol={_TOLERANCE})"
            )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the reference experiment and compare (or write) the golden file.

    Args:
        argv: Command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Process exit code: `0` on a match (or a successful `--write-golden`), `1` on mismatch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="(Re)generate the golden file from a fresh reference run instead of comparing.",
    )
    args = parser.parse_args(argv)

    config, cal_loss_table, test_loss_table = build_reference_inputs()
    with tempfile.TemporaryDirectory(prefix="wfcrc_reproduce_") as tmp_dir:
        run_dir = Path(tmp_dir) / "reference_run"
        bundle = ExperimentRunner().run(config, cal_loss_table, test_loss_table, run_dir=run_dir)
        fresh = bundle.manifest.to_dict()

    if args.write_golden:
        save_json(GOLDEN_PATH, fresh)
        print(f"wrote golden file: {GOLDEN_PATH}")
        return 0

    if not GOLDEN_PATH.exists():
        print(
            f"golden file not found: {GOLDEN_PATH} (run with --write-golden first)", file=sys.stderr
        )
        return 1

    golden = load_json(GOLDEN_PATH)
    mismatches = compare_to_golden(fresh, golden)
    if mismatches:
        print("make reproduce: MISMATCH", file=sys.stderr)
        for line in mismatches:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("make reproduce: OK (fresh run matches golden file within tolerance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
