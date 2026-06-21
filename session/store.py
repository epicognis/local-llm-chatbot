import aiosqlite
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Turn:
    role: str
    content: str
    tokens: int
    ts: str


class SessionStore:
    def __init__(self, db_url: str):
        self.db_path = db_url.replace("sqlite:///", "")

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    name        TEXT NOT NULL DEFAULT '',
                    summary     TEXT NOT NULL DEFAULT '',
                    summarized_count INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            try:
                await db.execute(
                    "ALTER TABLE sessions ADD COLUMN summarized_count INTEGER NOT NULL DEFAULT 0"
                )
            except aiosqlite.OperationalError:
                pass  # column already exists on a fresh DB
            try:
                await db.execute(
                    "ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT ''"
                )
            except aiosqlite.OperationalError:
                pass  # column already exists on a fresh DB
            await db.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL REFERENCES sessions(session_id),
                    role        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    tokens      INTEGER NOT NULL DEFAULT 0,
                    ts          TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)
            await db.commit()

    async def get_or_create(self, session_id: str) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT session_id, name, summary, summarized_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                await db.execute(
                    "INSERT INTO sessions (session_id) VALUES (?)", (session_id,)
                )
                await db.commit()
                return {"session_id": session_id, "name": "", "summary": "", "summarized_count": 0}
            return {"session_id": row[0], "name": row[1], "summary": row[2], "summarized_count": row[3]}

    async def get_turns(self, session_id: str) -> list[Turn]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT role, content, tokens, ts FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [Turn(role=r[0], content=r[1], tokens=r[2], ts=r[3]) for r in rows]

    async def append_turn(
        self, session_id: str, role: str, content: str, tokens: int = 0
    ):
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO turns (session_id, role, content, tokens, ts) VALUES (?,?,?,?,?)",
                (session_id, role, content, tokens, ts),
            )
            await db.commit()

    async def update_summary(self, session_id: str, summary: str, summarized_count: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sessions SET summary = ?, summarized_count = ? WHERE session_id = ?",
                (summary, summarized_count, session_id),
            )
            await db.commit()

    async def list_sessions(self) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT session_id, name FROM sessions ORDER BY created_at DESC"
            ) as cur:
                rows = await cur.fetchall()
        return [{"session_id": r[0], "name": r[1]} for r in rows]

    async def rename_session(self, session_id: str, name: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sessions SET name = ? WHERE session_id = ?", (name, session_id)
            )
            await db.commit()

    async def delete_session(self, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()
