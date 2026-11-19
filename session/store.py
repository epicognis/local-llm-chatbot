"""Server-side session storage (SQLite).

Server-side history is what makes the higher context-management levels possible —
the orchestrator owns the transcript rather than trusting the client to resend it.
SQLite is plenty for a single box; the interface is small enough to swap later.

Stores per session: ordered turns ({role, content, tokens, ts}), the model last
used, a display title, and the Level 2 running summary + compaction state.

Synchronous on purpose: one lock-guarded connection, sub-millisecond writes. A
sync write also can't be interrupted by task cancellation, which is what lets a
disconnected turn still persist its partial reply.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime


def _path_from_url(url: str) -> str:
    """Accept a sqlite:/// URL or a bare path."""
    prefix = "sqlite:///"
    return url[len(prefix):] if url.startswith(prefix) else url


@dataclass
class SessionInfo:
    id: str
    model: str | None
    title: str | None
    summary: str | None
    created_ts: float
    updated_ts: float
    turn_count: int


_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sessions (
        id                TEXT PRIMARY KEY,
        model             TEXT,
        title             TEXT,
        summary           TEXT,
        compaction_state  TEXT,
        created_ts        REAL NOT NULL,
        updated_ts        REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS turns (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL REFERENCES sessions(id),
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        tokens      INTEGER,
        ts          REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id)",
]


def _to_epoch(value) -> float:
    """Old-schema timestamps were ISO strings or SQLite datetime('now') text."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T")).timestamp()
    except ValueError:
        return time.time()


class SessionStore:
    def __init__(self, db_url: str = "sqlite:///./sessions.db") -> None:
        self._path = _path_from_url(db_url)
        # One connection guarded by a lock — simple and correct for a single box.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._migrate_legacy_schema()
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            # Migrate DBs created before these columns existed.
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)")}
            for col in ("model", "title", "compaction_state"):
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT")

    def _migrate_legacy_schema(self) -> None:
        """Convert the original layout — sessions(session_id, name, summary,
        summarized_count, created_at), turns.ts as text — keeping every turn."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)")}
        if "session_id" not in cols:
            return
        conn = self._conn
        conn.execute("BEGIN")
        try:
            conn.execute("ALTER TABLE sessions RENAME TO sessions_legacy")
            conn.execute("ALTER TABLE turns RENAME TO turns_legacy")
            # The renamed table keeps its index, whose name the new schema reuses.
            conn.execute("DROP INDEX IF EXISTS idx_turns_session")
            for stmt in _SCHEMA:
                conn.execute(stmt)
            for s in conn.execute("SELECT * FROM sessions_legacy").fetchall():
                created = _to_epoch(s["created_at"])
                last = conn.execute(
                    "SELECT MAX(ts) AS t FROM turns_legacy WHERE session_id = ?", (s["session_id"],)
                ).fetchone()["t"]
                conn.execute(
                    "INSERT INTO sessions (id, title, summary, created_ts, updated_ts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (s["session_id"], s["name"] or None, s["summary"] or None, created,
                     _to_epoch(last) if last else created),
                )
            for t in conn.execute("SELECT * FROM turns_legacy ORDER BY id").fetchall():
                conn.execute(
                    "INSERT INTO turns (session_id, role, content, tokens, ts) VALUES (?, ?, ?, ?, ?)",
                    (t["session_id"], t["role"], t["content"], t["tokens"], _to_epoch(t["ts"])),
                )
            conn.execute("DROP TABLE turns_legacy")
            conn.execute("DROP TABLE sessions_legacy")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # --- sessions -----------------------------------------------------------

    def new_session(self, model: str | None = None) -> str:
        sid = uuid.uuid4().hex
        self.ensure_session(sid, model)
        return sid

    def _ensure_session_locked(self, session_id: str, model: str | None, now: float) -> None:
        self._conn.execute(
            """INSERT INTO sessions (id, model, summary, created_ts, updated_ts)
               VALUES (?, ?, NULL, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 model = COALESCE(excluded.model, sessions.model),
                 updated_ts = excluded.updated_ts""",
            (session_id, model, now, now),
        )

    def ensure_session(self, session_id: str, model: str | None = None) -> None:
        with self._lock:
            self._ensure_session_locked(session_id, model, time.time())

    def get_session(self, session_id: str) -> SessionInfo | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            count = self._conn.execute(
                "SELECT COUNT(*) AS c FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()["c"]
        return SessionInfo(
            id=row["id"], model=row["model"], title=row["title"], summary=row["summary"],
            created_ts=row["created_ts"], updated_ts=row["updated_ts"], turn_count=count,
        )

    def list_sessions(self, limit: int = 100) -> list[SessionInfo]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.*, COUNT(t.id) AS turn_count
                   FROM sessions s LEFT JOIN turns t ON t.session_id = s.id
                   GROUP BY s.id
                   ORDER BY s.updated_ts DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            SessionInfo(
                id=r["id"], model=r["model"], title=r["title"], summary=r["summary"],
                created_ts=r["created_ts"], updated_ts=r["updated_ts"], turn_count=r["turn_count"],
            )
            for r in rows
        ]

    def rename_session(self, session_id: str, title: str) -> bool:
        """Set a session's display title. Returns False if the session is unknown."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ?, updated_ts = ? WHERE id = ?",
                (title, time.time(), session_id),
            )
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its turns. Returns False if the session is unknown."""
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    # --- turns --------------------------------------------------------------

    def append_turn(self, session_id: str, role: str, content: str, tokens: int | None = None) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, role, content, tokens, ts) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, tokens, now),
            )
            self._conn.execute("UPDATE sessions SET updated_ts = ? WHERE id = ?", (now, session_id))

    def append_exchange(
        self,
        session_id: str,
        model: str | None,
        user_message: str,
        reply: str,
        *,
        user_tokens: int | None = None,
        reply_tokens: int | None = None,
    ) -> None:
        """Persist a user message and its reply atomically, creating the session
        if needed — history always alternates user/assistant."""
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._ensure_session_locked(session_id, model, now)
                self._conn.executemany(
                    "INSERT INTO turns (session_id, role, content, tokens, ts) VALUES (?, ?, ?, ?, ?)",
                    [
                        (session_id, "user", user_message, user_tokens, now),
                        (session_id, "assistant", reply, reply_tokens, now),
                    ],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_turns(self, session_id: str) -> list[dict]:
        """Return prior turns as [{"role","content"}], oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # --- summary (Level 2) --------------------------------------------------

    def set_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))

    def get_summary(self, session_id: str) -> str | None:
        info = self.get_session(session_id)
        return info.summary if info else None

    def set_compaction_state(self, session_id: str, state_json: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET compaction_state = ? WHERE id = ?", (state_json, session_id)
            )

    def get_compaction_state(self, session_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT compaction_state FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row["compaction_state"] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
