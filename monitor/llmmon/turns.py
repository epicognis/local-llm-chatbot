"""Tok/s comes from the chat server's own per-turn log lines (structlog JSON,
event="turn_complete"), since Ollama's /api/ps reports residency but not
generation throughput. Tails server.log from whatever byte offset is current
each poll, so it's cheap and survives the server restarting/rotating.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TurnInfo:
    model: str
    completion_tokens: int
    elapsed_s: float
    tokens_per_sec: float


class TurnTailer:
    def __init__(self, log_path: str | os.PathLike, maxlen: int = 8) -> None:
        self.log_path = Path(log_path)
        self._file = None
        self._inode_size = None  # (st_ino or st_size fallback) to detect truncation/rotation
        self.recent: deque[TurnInfo] = deque(maxlen=maxlen)
        self.available = False
        self.reason = ""

    def _open(self):
        if not self.log_path.exists():
            self.reason = f"{self.log_path} not found yet (start the server via run.sh first)"
            return None
        if self._file is None:
            try:
                self._file = open(self.log_path, "r", encoding="utf-8", errors="ignore")
                self._file.seek(0, os.SEEK_END)
                self.available = True
            except OSError as exc:
                self.reason = f"can't open {self.log_path}: {exc}"
                return None
        return self._file

    def poll(self) -> None:
        f = self._open()
        if f is None:
            return
        # If the file shrank (server restarted and truncated/rotated), reopen from the top.
        try:
            if self.log_path.stat().st_size < f.tell():
                f.close()
                self._file = None
                f = self._open()
                if f is None:
                    return
                f.seek(0)
        except OSError:
            pass

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "turn_complete":
                self.recent.appendleft(
                    TurnInfo(
                        model=event.get("model", "?"),
                        completion_tokens=event.get("completion_tokens", 0),
                        elapsed_s=event.get("elapsed_s", 0.0),
                        tokens_per_sec=event.get("tokens_per_sec", 0.0),
                    )
                )
