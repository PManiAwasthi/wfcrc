"""Single-source package version.

Kept as a dedicated module (rather than inline in ``__init__.py``) so build
tooling and ``importlib.metadata`` fallbacks can read it without importing
the rest of the package graph.
"""

from __future__ import annotations

__version__ = "0.1.0"
