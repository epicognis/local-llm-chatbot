import json
import httpx
import structlog
from llm.backends.base import LLMBackend

log = structlog.get_logger()


class OllamaBackend(LLMBackend):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _chat_stream(self, messages: list[dict], model: str, **opts):
        return self._stream_from_ollama(messages, model, **opts)

    async def _stream_from_ollama(self, messages: list[dict], model: str, **opts):
        think = opts.pop("think", None)
        keep_alive = opts.pop("keep_alive", None)
        payload = {"model": model, "messages": messages, "stream": True}
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if opts:
            payload["options"] = opts

        content_emitted = False
        thinking_buffer: list[str] = []

        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = chunk.get("message", {})
                        # Some models (e.g. GPT-OSS) populate both `content` and
                        # `thinking` per chunk — only the former is the visible
                        # answer, so never stream `thinking` live.
                        content = msg.get("content", "")
                        if content:
                            content_emitted = True
                            yield content
                        else:
                            thinking = msg.get("thinking", "")
                            if thinking:
                                thinking_buffer.append(thinking)
                        if chunk.get("done"):
                            log.debug(
                                "ollama_turn_done",
                                model=model,
                                prompt_eval_count=chunk.get("prompt_eval_count"),
                                eval_count=chunk.get("eval_count"),
                            )
                            break
            except httpx.ConnectError:
                yield "[ERROR] Cannot connect to Ollama. Is it running at " + self.base_url + "?"
                return

        # Last resort: some models (e.g. Qwen3 with thinking left enabled) route
        # everything to `thinking` and never populate `content`. Surface it rather
        # than silently returning nothing.
        if not content_emitted and thinking_buffer:
            yield "".join(thinking_buffer)
        elif not content_emitted:
            # Ollama can return a clean "done" stream with zero output if the
            # model failed to finish loading in time (e.g. a large num_ctx cold
            # load timing out) — surface that instead of leaving the UI blank.
            yield "[No response from the model — it may still be loading (large context windows can take a while to warm up). Please try again.]"

    async def _chat_complete(self, messages: list[dict], model: str, **opts) -> str:
        think = opts.pop("think", None)
        keep_alive = opts.pop("keep_alive", None)
        payload = {"model": model, "messages": messages, "stream": False}
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if opts:
            payload["options"] = opts

        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            msg = data["message"]
            return msg.get("content") or msg.get("thinking", "")

    async def is_loaded(self, model: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
            if resp.status_code != 200:
                return True  # don't block chat on a health-check hiccup
            data = resp.json()
            return any(m.get("name") == model for m in data.get("models", []))
        except Exception:
            return True

    async def summarize(self, text: str, model: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation compactly, "
                    "preserving key facts, decisions, and context."
                ),
            },
            {"role": "user", "content": text},
        ]
        return await self._chat_complete(messages, model)
