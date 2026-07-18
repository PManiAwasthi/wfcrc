"""Model registry and (future) score-provider/checkpoint infrastructure (MS6).

MS6.1 populates only the :data:`MODELS` name-keyed registry (mirroring
:data:`wfcrc.ambiguity.FAMILIES`); concrete ``ScoreProvider`` implementations
and checkpoint management are later MS6 sub-milestones (see
``MS6_ARCHITECTURE_SPEC.md`` §3.2, §3.4, §3.5). This directory existed as an
empty placeholder from MS1 through RC1 (no Blueprint module was ever built
here, per ``PROJECT_CONTEXT.md`` §6/§11); MS6 is what first populates it.
"""

from __future__ import annotations

from wfcrc.models.registry import MODELS

__all__ = ["MODELS"]
