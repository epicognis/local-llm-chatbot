from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from session.store import SessionStore

router = APIRouter()


class RenameRequest(BaseModel):
    name: str


@router.get("/sessions")
async def list_sessions(request: Request):
    store: SessionStore = request.app.state.store
    return await store.list_sessions()


@router.get("/sessions/{session_id}/turns")
async def get_turns(session_id: str, request: Request):
    store: SessionStore = request.app.state.store
    turns = await store.get_turns(session_id)
    return [
        {"role": t.role, "content": t.content, "tokens": t.tokens, "ts": t.ts}
        for t in turns
    ]


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: RenameRequest, request: Request):
    store: SessionStore = request.app.state.store
    await store.rename_session(session_id, body.name)
    return {"session_id": session_id, "name": body.name}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    store: SessionStore = request.app.state.store
    await store.delete_session(session_id)
    return {"deleted": session_id}
