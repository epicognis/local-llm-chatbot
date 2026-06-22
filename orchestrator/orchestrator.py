import structlog
from llm.backends.base import LLMBackend
from llm.registry import get_model_entry
from orchestrator.context_budget import ContextBudgetAllocator, estimate_tokens
from orchestrator.memory import (
    ContextProvider,
    RawPassthroughProvider,
    SlidingWindowProvider,
    RollingSummaryProvider,
)
from session.store import SessionStore
from metrics.instrument import TurnMetrics
from config.settings import settings

log = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant. "
    "Answer clearly and accurately. "
    "Be concise unless depth is needed."
)


class Orchestrator:
    def __init__(self, backend: LLMBackend, store: SessionStore):
        self.backend = backend
        self.store = store

    def _get_provider(self, model: str) -> ContextProvider:
        level = settings.CONTEXT_LEVEL
        if level == 0:
            return RawPassthroughProvider()
        if level == 1:
            return SlidingWindowProvider(settings.SLIDING_WINDOW_TURNS)
        if level == 2:
            return RollingSummaryProvider(
                backend=self.backend,
                store=self.store,
                summarizer_model=model,
                window_turns=settings.SLIDING_WINDOW_TURNS,
                trigger_tokens=settings.SUMMARY_TRIGGER_TOKENS,
            )
        raise ValueError(f"Unsupported CONTEXT_LEVEL: {level} (only 0, 1, 2 are implemented)")

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Assembles context and streams the model response token by token."""
        session = await self.store.get_or_create(session_id)
        turns = await self.store.get_turns(session_id)

        entry = get_model_entry(model_name)
        model = entry["model"]
        default_ctx = entry["default_ctx"]

        resolved_max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
        resolved_temp = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE

        allocator = ContextBudgetAllocator(
            model_ctx=default_ctx,
            max_tokens=resolved_max_tokens,
            safety_margin=settings.CONTEXT_SAFETY_MARGIN,
        )

        provider = self._get_provider(model)
        context_turns, summary = await provider.prepare(session_id, session, turns)

        messages = allocator.assemble(
            system_prompt=SYSTEM_PROMPT,
            turns=context_turns,
            current_message=message,
            summary=summary,
        )

        log.info(
            "turn_start",
            session_id=session_id,
            model=model,
            context_level=settings.CONTEXT_LEVEL,
            context_messages=len(messages),
        )

        await self.store.append_turn(
            session_id, "user", message, estimate_tokens(message)
        )

        metrics = TurnMetrics(session_id=session_id, model=model)
        response_parts: list[str] = []

        opts = {
            "temperature": resolved_temp,
            "num_predict": resolved_max_tokens,
            "num_ctx": default_ctx,
            "think": False,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        }

        if not await self.backend.is_loaded(model):
            log.info("model_cold_start", session_id=session_id, model=model, num_ctx=default_ctx)
            yield {
                "type": "status",
                "text": f"Loading {model_name} into memory (ctx={default_ctx}) — "
                        f"large context windows can take up to a minute on first use…",
            }

        try:
            async for token in self.backend.chat(messages, model, stream=True, **opts):
                response_parts.append(token)
                metrics.count_token()
                yield {"type": "token", "text": token}
        finally:
            full_response = "".join(response_parts)
            if full_response:
                await self.store.append_turn(
                    session_id,
                    "assistant",
                    full_response,
                    estimate_tokens(full_response),
                )
            metrics.finish()
