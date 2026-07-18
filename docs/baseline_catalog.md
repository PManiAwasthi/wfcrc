# Baseline Catalog — WFCRC Research Program (Paper 1)

> **Status:** Frozen for first use. **Version:** 1.0. **Date:** 2026-07-18.
> **Purpose (stated explicitly, per its own design brief):** (1) make it
> immediately obvious to reviewers and collaborators exactly which methods
> WF-CRC is being compared against, and (2) force every implemented
> baseline to be traceable to a specific publication, so an accidental
> deviation from the literature is caught by inspection, not discovered
> later. Every row below is a comparator named in the frozen
> `Paper 1 - EXPERIMENT BLUEPRINT.md` §9–10 and reproduced in
> `docs/EXPERIMENT_PROTOCOL.md` §3 / `docs/MODEL_POLICY.md` §1.2.
>
> **`wfcrc` (`wfcrc.baselines.wfcrc_adapter.WFCRCAdapter`) is deliberately
> excluded from this catalog** — it is WF-CRC itself (the method under
> study), adapted to the common `Calibrator` interface so it can be driven
> identically to every comparator below; it is not a comparator baseline.

## How to read the "Citation confidence" column

Because this document's second purpose is catching accidental deviation
from a *specific* publication, it distinguishes what is actually verified
from what is merely named:

- **High** — a well-established, singly-identifiable method; the cited
  paper is the unambiguous, standard reference for it.
- **Vault-only** — the frozen Research Vault names this baseline by
  author/year only (no title, venue, or exact algorithm), and the title
  given here is this document's own best-effort identification, **not**
  independently verified against the original publication during this
  milestone. Treat as provisional until a domain expert confirms it.
- **Unresolved** — no paper is identifiable at all from the frozen vault
  (checked by exhaustive grep, `PROJECT_CONTEXT.md` MS9 section) or from
  the vault's own framing (an explicitly-named "future work" gap). Not
  implemented, and not guessed at.

---

| # | Baseline | Original paper | Citation | Repository class | Implemented? | Used in experiments |
|---|---|---|---|---|---|---|
| 1 | Conformal Risk Control ("vanilla CRC") | Angelopoulos, Bates, Candès, Jordan & Lei, "Conformal Risk Control" | arXiv:2208.02814 (2022) — **High** confidence | `wfcrc.baselines.vanilla_crc.VanillaCRC` (delegates to the frozen `WFCRCCalibrator` + uniform `KnownWeightFamily`, zero duplicated formula) | ✅ Yes | E1, E2, E3, E6, E7 (the marginal-risk reference throughout) |
| 2 | Split conformal prediction / LAC | Sadinle, Lei & Wasserman, "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels" | *Journal of the American Statistical Association* 114(525), 2019 — **High** confidence | `wfcrc.baselines.lac.SplitConformalLAC` (calibration half) + `wfcrc.prediction_sets.classification.ThresholdSets` (set-construction half, already frozen since MS3 and already named "LAC" in its own docstring) | ✅ Yes | E1 (classification generality), E4 (real classification shift) |
| 3 | Temperature scaling | Guo, Pleiss, Sun & Weinberger, "On Calibration of Modern Neural Networks" | ICML 2017 — **High** confidence | `wfcrc.baselines.scaling.fit_temperature` / `.apply_temperature` (score-level utility) + `wfcrc.baselines.scaling.TemperatureScaledLAC` (downstream `Calibrator` wrapper) | ✅ Yes (utility + wrapper; real-logit integration deferred, no concrete model in this repo produces raw logits yet — see `docs/PILOT_REPORT.md` §2) | E11 (calibration quality baseline) |
| 4 | Selective classification / "selective scaling" | Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks" | NeurIPS 2017 — **High** confidence for the paper; **disclosed simplification** in this repo (basic empirical selective-risk criterion only, not the paper's own finite-sample `SGR` binomial-tail bound — see `wfcrc/baselines/scaling.py`'s `fit_selective_threshold` docstring) | `wfcrc.baselines.scaling.fit_selective_threshold` / `.apply_selective_threshold` (standalone utility; no downstream `Calibrator` wrapper — its output *is* the abstention decision, not a further conformal step) | ✅ Yes (simplified criterion, disclosed) | E11 (calibration quality baseline) |
| 5 | Deep ensembles | Lakshminarayanan, Pritzel & Blundell, "Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles" | NeurIPS 2017 — **High** confidence | `wfcrc.baselines.ensembles.aggregate_deep_ensemble_scores` (score-level utility) + `wfcrc.baselines.ensembles.EnsembleAggregatedLAC` (downstream wrapper) | ✅ Yes (aggregation utility + wrapper; real multi-member-inference integration deferred — no ensemble of trained models exists in this repo) | Baseline suite (pre-empts "missing baseline" reviewer criticism, Experiment Blueprint §9–10; not tied to one numbered E# in the Blueprint's own Experiment Catalog) |
| 6 | MC-dropout | Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" | ICML 2016 — **High** confidence | `wfcrc.baselines.ensembles.aggregate_mc_dropout_scores` (identical arithmetic to #5, kept as a separate named function so the two baselines stay distinguishable in results — see module docstring) + `EnsembleAggregatedLAC` | ✅ Yes (aggregation utility + wrapper; real multi-pass-inference integration deferred — the current `HippocampusScoreProvider` has no dropout layer, by design, per its own module docstring's reproducibility rationale) | Baseline suite (same status as #5) |
| 7 | Conditional-coverage conformal prediction ("Gibbs") | Gibbs, Cherian & Candès, "Conformal Prediction With Conditional Guarantees" | arXiv:2305.12616 (2023) — **Vault-only**: the frozen vault names this "Gibbs–Cherian–Candès 2023" with no title; the title above is this document's own best-effort identification, unverified against the original paper in this session | `wfcrc.baselines.group_conditional.GroupConditionalCRC` — **explicitly a disclosed proxy**: implements only the *finite-group* (Mondrian/classwise) specialization of the paper's general covariate-function-class method, which is substantially more complex and not attempted here (see the module's own docstring) | ⚠️ Partially (finite-group specialization only; general method not implemented) | E2 (conditional per-region/organ risk) |
| 8 | Robust conformal prediction (f-divergence ball) | Cauchois & Duchi (exact title/venue not identified) | "2024" — **Vault-only**: the frozen vault (`Theorem Summit - Paper 1 Central Theorem.md`) names this "Cauchois–Duchi 2024" with no title or venue; not independently verified against the original publication in this session — **flagged for PI/domain-expert citation confirmation before this row is cited in a manuscript** | `wfcrc.baselines.robust_fdiv.RobustFDivergenceCP` — reuses the frozen `KLFamily` dual (Algorithm Spec §7) in a pooled (no A/B split), `n`-inflated construction, disclosed as differing from WF-CRC's own single-split architecture in its own module docstring | ✅ Yes (KL/f-divergence variant only) | E3 (synthetic shift), E4 (real shift) |
| 9 | Robust conformal prediction (Lévy–Prokhorov / optimal-transport ball) | Not identified | Frozen `Paper 1 - FRAMEWORK SPECIFICATION.md` itself names this divergence family "future work... open gap," with no specific paper attached anywhere in the vault — **Unresolved** | — | ❌ No — would require extending the frozen ambiguity-family architecture itself (a framework change), explicitly out of every additive milestone's scope through MS10 | E3, E4 (named alongside #8 in the Blueprint's baseline line; not available) |
| 10 | "AA-CRC" (conditional/adaptive conformal risk control) | Not identified | Appears only as a bare, uncited acronym across every vault document that mentions it (confirmed by exhaustive grep, `PROJECT_CONTEXT.md` MS9 section) — **Unresolved** | — | ❌ No — implementing it would mean inventing an algorithm for an unidentified method, which this project's "stop and document, do not invent" rule forbids | E2 (per the Blueprint's own baseline list; not available) |
| 11 | "sem-CRC" (semantic/class-conditional conformal risk control) | Not identified | Same status as #10 — **Unresolved** | — | ❌ No | E2 (per the Blueprint's own baseline list; not available) |
| 12 | Pooled K-fold WF-CRC (architecture negative control) | N/A — an internal ablation this research program defines itself, not an external method (Experiment Blueprint §23, "component-removal" table; confirms frozen Proof Obligation P3) | `Paper 1 - EXPERIMENT BLUEPRINT.md` §23 — **N/A** (not an external citation) | `wfcrc.baselines.negative_controls.PooledKFoldWFCRC` — promoted, formula-for-formula identical, from the MS4 test-only harness `tests/unit/calibration/test_negative_controls.py` | ✅ Yes | E7 (architecture ablation) |
| 13 | Total-`n` inflation WF-CRC (architecture negative control) | N/A — internal ablation (confirms frozen Proof Obligation P4) | `Paper 1 - EXPERIMENT BLUEPRINT.md` §23 — **N/A** | `wfcrc.baselines.negative_controls.TotalNInflationWFCRC` — promoted, formula-for-formula identical, from the same MS4 harness | ✅ Yes | E7 (architecture ablation) |
| 14 | Fixed-η WF-CRC (architecture ablation) | N/A — internal ablation (Experiment Blueprint §23; valid by weak duality for any fixed dual parameter, Math Spec §5) | `Paper 1 - EXPERIMENT BLUEPRINT.md` §23, `Paper 1 - ALGORITHM SPECIFICATION.md` §15 — **N/A** | `wfcrc.baselines.negative_controls.FixedEtaWFCRC` — new in MS9 (no prior implementation, test-only or otherwise, existed before) | ✅ Yes | E7 (architecture ablation) |

---

## Summary

- **11 of 14** named comparator baselines are implemented (behind the
  common `wfcrc.baselines.base.Calibrator` interface where applicable;
  #4/#5/#6 as standalone score-level utilities plus a downstream wrapper,
  per their own genuine architectural scope boundary — see
  `wfcrc/baselines/scaling.py`/`wfcrc/baselines/ensembles.py`'s own
  docstrings).
- **3 of 14** (#9 Lévy–Prokhorov, #10 AA-CRC, #11 sem-CRC) are **not**
  implemented, each for a distinct, disclosed reason: #9 is out of the
  frozen framework's current representational scope by the Framework
  Specification's own admission; #10/#11 have no identifiable algorithm
  anywhere in the frozen vault.
- **2 citations (#7, #8) are flagged "Vault-only"** — their titles are
  this document's own best-effort identification from author/year alone,
  not independently verified against the original publication. These, and
  #9–#11's resolution, are the concrete, actionable items for a domain
  expert / PI review before any of rows #7–#11 is cited in a submitted
  manuscript's related-work section.
- All 10 implemented `Calibrator` baselines (everything except the two
  standalone-utility-only rows #4's own utility half is inside #3/#4/#5/#6's
  wrapper count) were exercised end to end against real MSD Task04_Hippocampus
  data in the MS10 pilot — see `docs/PILOT_REPORT.md` §2.

---

## Connections

`docs/EXPERIMENT_PROTOCOL.md` §3 (Experiment Matrix) · `docs/MODEL_POLICY.md`
§1.2 (Comparator baselines) · `docs/PILOT_REPORT.md` (real-data validation
record) · `wfcrc/baselines/` · `PROJECT_CONTEXT.md` (MS9/MS10 sections) ·
Research Vault: `Paper 1 - EXPERIMENT BLUEPRINT.md` §9–10,
`Theorem Summit - Paper 1 Central Theorem.md`,
`Paper 1 - FRAMEWORK SPECIFICATION.md`,
`Paper 1 - MANUSCRIPT PREPARATION BLUEPRINT.md` §2/§7
