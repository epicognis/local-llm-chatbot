"""Session endpoints — list, create, fetch transcript, rename, delete.

Server-side session state is what makes the higher context levels possible; these
routes let the UI populate its session list and reload a past conversation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions")


class NewSessionRequest(BaseModel):
    model: str | None = None


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@router.get("")
async def list_sessions(request: Request) -> list[dict]:
    return [
        {
            "id": s.id,
            "model": s.model,
            "title": s.title,
            "turn_count": s.turn_count,
            "updated_ts": s.updated_ts,
        }
        for s in request.app.state.store.list_sessions()
    ]


@router.post("")
async def create_session(body: NewSessionRequest, request: Request) -> dict:
    return {"id": request.app.state.store.new_session(body.model)}


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    store = request.app.state.store
    info = store.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "id": info.id,
        "model": info.model,
        "title": info.title,
        "summary": info.summary,
        "turn_count": info.turn_count,
        "turns": store.get_turns(session_id),
    }


@router.patch("/{session_id}")
async def rename_session(session_id: str, body: RenameSessionRequest, request: Request) -> dict:
    title = body.title.strip()
    if not title or not request.app.state.store.rename_session(session_id, title):
        raise HTTPException(status_code=404, detail="session not found")
    return {"id": session_id, "title": title}


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    if not request.app.state.store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"id": session_id, "deleted": True}
