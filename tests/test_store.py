"""Session store: persistence, ordering, listing, summary."""

from session.store import SessionStore


def _store(tmp_path):
    return SessionStore(f"sqlite:///{tmp_path / 'sessions.db'}")


def test_turns_round_trip_in_order(tmp_path):
    store = _store(tmp_path)
    sid = store.new_session("GPT-OSS 20B")
    store.append_turn(sid, "user", "hello")
    store.append_turn(sid, "assistant", "hi there")
    store.append_turn(sid, "user", "bye")

    turns = store.get_turns(sid)
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert turns[0]["content"] == "hello"


def test_session_listed_with_turn_count(tmp_path):
    store = _store(tmp_path)
    sid = store.new_session("Qwen3 14B")
    store.append_turn(sid, "user", "x")
    sessions = store.list_sessions()
    assert any(s.id == sid and s.turn_count == 1 for s in sessions)


def test_summary_persists(tmp_path):
    store = _store(tmp_path)
    sid = store.new_session()
    store.set_summary(sid, "a running summary")
    assert store.get_summary(sid) == "a running summary"


def test_rename_session(tmp_path):
    store = _store(tmp_path)
    sid = store.new_session("GPT-OSS 20B")
    assert store.rename_session(sid, "VRAM debugging") is True
    assert store.get_session(sid).title == "VRAM debugging"
    # shows up in the listing too
    assert any(s.id == sid and s.title == "VRAM debugging" for s in store.list_sessions())


def test_rename_unknown_session_returns_false(tmp_path):
    store = _store(tmp_path)
    assert store.rename_session("nope", "x") is False


def test_delete_session_removes_it_and_turns(tmp_path):
    store = _store(tmp_path)
    sid = store.new_session()
    store.append_turn(sid, "user", "hi")
    assert store.delete_session(sid) is True
    assert store.get_session(sid) is None
    assert store.get_turns(sid) == []
    assert all(s.id != sid for s in store.list_sessions())


def test_delete_unknown_session_returns_false(tmp_path):
    store = _store(tmp_path)
    assert store.delete_session("nope") is False


def test_survives_reopen(tmp_path):
    db = f"sqlite:///{tmp_path / 'sessions.db'}"
    store = SessionStore(db)
    sid = store.new_session("Llama 3.1 8B")
    store.append_turn(sid, "user", "persist me")
    store.close()

    reopened = SessionStore(db)
    turns = reopened.get_turns(sid)
    assert turns[0]["content"] == "persist me"


def test_append_exchange_is_atomic_and_creates_session(tmp_path):
    store = _store(tmp_path)
    store.append_exchange("fresh", "Gemma 4 12B", "question", "answer",
                          user_tokens=2, reply_tokens=2)
    info = store.get_session("fresh")
    assert info is not None and info.model == "Gemma 4 12B"
    assert store.get_turns("fresh") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_migrates_legacy_schema_keeping_turns(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '', summarized_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, tokens INTEGER NOT NULL DEFAULT 0,
            ts TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE INDEX idx_turns_session ON turns(session_id, id);
        INSERT INTO sessions (session_id, name) VALUES ('old1', 'My old chat');
        INSERT INTO turns (session_id, role, content, ts)
            VALUES ('old1', 'user', 'hi', '2026-11-18T10:00:00+00:00'),
                   ('old1', 'assistant', 'hello', '2026-11-18T10:00:05+00:00');
        """
    )
    conn.commit()
    conn.close()

    store = SessionStore(f"sqlite:///{path}")
    info = store.get_session("old1")
    assert info.title == "My old chat" and info.turn_count == 2
    assert [t["content"] for t in store.get_turns("old1")] == ["hi", "hello"]
    store.append_exchange("old1", "Gemma 4 12B", "again", "yes")  # new schema works
    assert store.get_session("old1").turn_count == 4
