"""Backward-compatible config module.

Single source of truth is `backend.config.settings`.
Imports from `config` continue to work via re-export.
"""

from .settings import *  # noqa: F401,F403

