"""Append-only JSON Lines log with size-based rotation (trace, API request log).

`logging.handlers.RotatingFileHandler` from the standard library does the rotation; it is used
here as a plain file writer so the line is exactly the JSON record and nothing else.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonlLog:
    """One JSON object per line; rotates to `name.jsonl.1`, `.2`, ... at `max_bytes`."""

    def __init__(self, path: Path, max_bytes: int = 5_000_000, backups: int = 3) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._backups = backups
        self._handler: RotatingFileHandler | None = None

    def _open(self) -> RotatingFileHandler:
        if self._handler is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handler = RotatingFileHandler(
                self._path,
                maxBytes=self._max_bytes,
                backupCount=self._backups,
                encoding="utf-8",
                delay=True,
            )
            self._handler.setFormatter(logging.Formatter("%(message)s"))
        return self._handler

    def write(self, record: dict[str, Any]) -> None:
        """Append one record (non-JSON values are stringified)."""
        line = json.dumps(record, default=str)
        self._open().handle(logging.makeLogRecord({"msg": line, "levelno": logging.INFO}))
