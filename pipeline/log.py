"""Logging that reads like the pipeline's old print() output.

Every stage used to print ``[stage] message``. That format is genuinely good
for a CLI, so it is kept — but routed through ``logging`` so a server can
capture it per job instead of losing it to stdout.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stage = record.name.rsplit(".", 1)[-1]
        return f"[{stage}] {record.getMessage()}"


def setup(level: str | int | None = None) -> None:
    """Install the console handler. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        level = os.environ.get("REELS_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter())
    root = logging.getLogger("pipeline")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"pipeline.{name}")
