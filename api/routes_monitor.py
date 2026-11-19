"""GET /monitor — point-in-time hardware + Ollama snapshot for the UI panel.

Thin wrapper over the shared ``HardwareMonitor`` on ``app.state`` (which reuses
the ``llmmon`` package in ``monitor/``). The browser polls this on an interval to
drive the real-time monitor panel. Always returns 200 with ``available: false`` + a reason
when sampling can't run, so the panel can render a quiet "unavailable" state
rather than erroring.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/monitor")
async def monitor(request: Request) -> dict:
    return await request.app.state.hw_monitor.snapshot()
