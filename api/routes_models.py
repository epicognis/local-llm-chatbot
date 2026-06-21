from fastapi import APIRouter
from llm.registry import MODEL_REGISTRY

router = APIRouter()


@router.get("/models")
async def list_models():
    return MODEL_REGISTRY
