import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "GPT-OSS 20B"
    options: dict = {}


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    orchestrator = request.app.state.orchestrator

    async def event_stream():
        try:
            async for event in orchestrator.chat_stream(
                session_id=body.session_id,
                message=body.message,
                model_name=body.model,
                temperature=body.options.get("temperature"),
                max_tokens=body.options.get("max_tokens"),
            ):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'token', 'text': '[ERROR] ' + str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
