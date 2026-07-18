"""Model registry, checkpoint management, and score providers (MS6.1, MS7).

MS6.1 populated the (then-empty) :data:`MODELS` name-keyed registry
(mirroring :data:`wfcrc.ambiguity.FAMILIES`). MS7 first populates it with a
concrete provider — :mod:`wfcrc.models.checkpoint` (discovery/loading/
fingerprinting) and :mod:`wfcrc.models.scores.hippocampus_segmenter`
(``HippocampusScoreProvider``, the minimum end-to-end vertical-slice model)
— per ``MS6_ARCHITECTURE_SPEC.md`` §3.2, §3.4, §3.5 (reduced scope; see
each module's own docstring for exactly what is/isn't implemented). This
directory existed as an empty placeholder from MS1 through RC1 (no
Blueprint module was ever built here, per ``PROJECT_CONTEXT.md`` §6/§11);
MS6 is what first populated it, MS7 what first made it concrete.
"""

from __future__ import annotations

from wfcrc.models.registry import MODELS

__all__ = ["MODELS"]
