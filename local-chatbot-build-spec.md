# Local Chatbot — Engineering Build Spec (Handoff)

> **Audience:** an AI coding agent (e.g. running `gpt-oss-20b` locally, or Claude Code) implementing this project from scratch.
> **Style note for the implementer:** Use the exact identifiers, endpoint paths, and field names given here verbatim. Do not abbreviate module names, table names, env vars, or JSON keys. Where a "WRONG → RIGHT" note appears, the WRONG form is a known failure mode — avoid it.
>
> **Status (validated):** The full vertical slice (§11) has been built and driven end-to-end in a real browser against the mock backend and three live Ollama models (Qwen3 14B, GPT-OSS 20B, Gemma 4 12B). Notes tagged *(Verified)* below were confirmed in that process — they are the bugs the first pass of this spec did not anticipate.

---

## 1. Goal

Build a **local, single-box chatbot** that:

1. Serves one or more selectable open-weight LLMs running on a single **RTX 5060 Ti 16GB** GPU.
2. Exposes a **FastAPI** web server with a browser **`/ui`** that takes a user prompt and streams the model's reply.
3. Has an **orchestration layer** that assembles and manages the prompt context within the model's token budget, designed to grow from a trivial passthrough into summarization + retrieval-augmented memory.

This project **generalizes an existing internal NL→SQL engine**. Reuse its proven patterns; do **not** start the architecture from scratch. The pieces that transfer directly: a backend-agnostic LLM shim, a FastAPI + Jinja2 `/ui`, an `LLM_BACKEND` env-var swap, structured logging (structlog), pydantic-settings config, a `MockBackend` for tests, and a multi-turn history pattern. The pieces to **drop**: SQL guardrails, the SQL validator, the Postgres executor, and the CSV/Excel/chart output composer.

---

## 2. Hardware & Runtime Constraints

| Constraint | Detail |
|---|---|
| GPU | RTX 5060 Ti 16GB (Blackwell, sm_120), ~448 GB/s GDDR7 |
| Practical model ceiling | ~20B parameters on a single card. 30B/32B dense models do **not** fit fully in 16GB at 4-bit. |
| Serving runtime | **Ollama** (bundles a recent llama.cpp with CUDA Blackwell support) |
| **Important** | Serving does **NOT** require nightly PyTorch. The sm_120 / nightly-torch headaches documented elsewhere were specific to a separate interpretability project (TransformerLens). Inference via Ollama/llama.cpp works on stable releases. Do not pull nightly torch for this project. |
| Language | Python 3.11, **Python only — no Node.js / npm** |

---

## 3. Model Lineup (Selectable)

Curate these into a registry the `/ui` dropdown reads from. Each maps a friendly name to `(backend, model, default_ctx)`. VRAM and speed figures are approximate community benchmarks for this card — treat as guidance, verify locally.

| Friendly name | Ollama model | Quant / size | Role | Notes |
|---|---|---|---|---|
| **GPT-OSS 20B** (default) | `gpt-oss:20b` | MXFP4, ~12–13 GB | Primary reasoning model | Sparse MoE (~21B total, ~3.6B active). **128K context.** ~80 tok/s on this card because MXFP4 maps natively onto Blackwell tensor cores. The large window relaxes context management — most sessions never need summarization. |
| Qwen3 14B | `qwen3:14b` | Q4_K_M, ~9 GB | Strong all-round dense; code + multilingual | ~33 tok/s @ 16K context. **Native window is 40960** (not a 16K placeholder) — set `default_ctx` from the real window. Leaves room for KV cache. |
| Qwen3 8B | `qwen3:8b` | ~6 GB | Fast tier / routing fallback | Very high prompt-processing throughput; use for trivial turns. |
| Gemma 4 12B | `gemma4:12b` *(verify tag)* | 4-bit, ~7 GB | Family variety + agentic / multimodal | Dense ~11.95B, Apache 2.0, released ~June 2026. **256K context.** Native tool-use / function-calling + a "thinking" reasoning mode (ships with a Gemma Skills Repository) → second contender to GPT-OSS 20B for agentic work. Encoder-free unified multimodal (text/image/audio/video) — unused by a text-only bot but a future input path. MTP drafters give speculative-decoding latency wins. **Caveat:** ~weeks old — confirm the Ollama tag and a stable 4-bit quant before committing. |
| Llama 3.1 8B | `llama3.1:8b` | Q4, ~5 GB | Fast familiar baseline | ~70 tok/s — feels instant. |

**Do NOT** put `qwen3-coder:30b` or any 30B+ model in the single-card registry; they overflow 16GB. For code work, route to Qwen3 14B or GPT-OSS 20B instead. The larger Gemma variants (Gemma 3 27B, Gemma 4 26B MoE) also run ~16–18 GB at 4-bit — too tight once KV cache is added — so stick to the 12B for single-card use.

**Benchmark caveat to encode in any docs:** some online "fit calculators" rate `gpt-oss:20b` at ~30 tok/s / 4K context by assuming Q4_K_M. That is wrong — the model ships natively as MXFP4, which is smaller and faster on Blackwell. Trust card-specific MXFP4 benchmarks.

**`default_ctx` is a *practical* value, not the model's max.** Set each registry `default_ctx` from the model's real window, but cap it to preserve KV-cache headroom on a 16GB card, and **pass it through as Ollama's `num_ctx` per request** (see §6) or it is silently capped at 4096. Verified real windows: GPT-OSS 20B `131072`, Qwen3 14B `40960`, Gemma 4 12B `262144` — but use a practical `default_ctx` such as `32768` for Gemma rather than its 256K max, to keep VRAM sane. *(Verified.)*

---

## 4. Architecture

```
User prompt (browser /ui)
  → FastAPI  (POST /chat, SSE stream)
  → Orchestrator  (assemble context within token budget; optional model routing)
  → LLM Backend shim  (Ollama /api/chat, streaming)
  → token stream back to /ui
```

Stateless HTTP server. Session/context state lives in the orchestration layer (server-side store keyed by `session_id`, see §8).

### Proposed Module Map

```
api/
  main.py               FastAPI app + route registration
  routes_chat.py        POST /chat  (SSE streaming)
  routes_models.py      GET  /models
  routes_meta.py        GET  /health
  templates/ui.html     Two-pane browser UI (model picker + session list left, transcript right)
  static/               CSS / JS for the chat UI

orchestrator/
  orchestrator.py       Per-turn coordinator: route model, assemble context, call backend, stream out
  context_budget.py     Token-budget allocator (see §7)
  memory.py             Rolling summary + (later) retrieval providers
  router.py             Optional tiered model dispatch (cheap model vs strong model)

llm/
  registry.py           Friendly-name → (backend, model, default_ctx) map; backs GET /models
  backends/
    base.py             Abstract shim: chat() + summarize()
    ollama_backend.py   Ollama HTTP /api/chat streaming  (PRIMARY)
    anthropic_backend.py  Optional cloud fallback (reuse from NL→SQL engine if available)
    mock_backend.py     Deterministic responses for tests — NO GPU / NO network

session/
  store.py              Session persistence (SQLite to start; see §8)

config/
  settings.py           pydantic-settings

metrics/
  instrument.py         Per-turn tokens, latency, tokens/sec
```

---

## 5. API Contract

### `POST /chat` — streamed completion (Server-Sent Events)

Request body (JSON):
```json
{
  "session_id": "string",
  "message": "string",
  "model": "GPT-OSS 20B",
  "options": { "temperature": 0.7, "max_tokens": 1024 }
}
```
- `model` is a **friendly name** from the registry (§3), not a raw Ollama tag. The registry resolves it.
- Response: `text/event-stream`. Emit one SSE `data:` event per token/delta, then a terminal event (e.g. `data: [DONE]`).
- **Must support cancellation:** if the client disconnects, abort the upstream Ollama call.

### `GET /models` — registry listing
Returns the curated list so the UI can populate its dropdown:
```json
[ { "name": "GPT-OSS 20B", "model": "gpt-oss:20b", "default_ctx": 131072 }, ... ]
```

### `GET /health` — liveness
Returns `{ "status": "ok" }` plus optionally which models are currently loaded (`ollama ps`).

### `GET /ui` — Jinja2 browser UI
Two-pane layout: left = model picker + session list; right = streaming transcript. Reuse the existing `/ui` Jinja2 scaffold; swap result-table rendering for a streaming chat transcript.

**Starlette 1.x gotcha:** `Jinja2Templates.TemplateResponse` changed its signature — `request` is now the **first positional** argument, not a key inside the context dict. Use `TemplateResponse(request, "ui.html", {...})`, **not** `TemplateResponse("ui.html", {"request": request, ...})`. The old form raises against current Starlette — a one-line fix, but it bites anyone copying an older `/ui` route verbatim. *(Verified.)*

---

## 6. LLM Backend Shim

Keep the abstraction so `llama.cpp`-direct or `vLLM` can drop in later. Minimal contract in `llm/backends/base.py`:

```python
class LLMBackend(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], model: str,
             stream: bool = True, **opts):
        """messages = [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Yields token-delta strings when stream=True; returns full string otherwise."""

    @abstractmethod
    def summarize(self, text: str, model: str) -> str:
        """Condense older conversation turns into a compact summary."""
```

### Ollama backend rules
- Call the **`/api/chat`** endpoint with role-tagged `messages` and `"stream": true`.
  - **WRONG:** `/api/generate` with a hand-rolled prompt template per model.
  - **RIGHT:** `/api/chat` — Ollama applies each model's chat template for you, so you never maintain per-model formatting.
- Parse the streamed JSONL; yield only `message.content` deltas as they arrive (see the thinking-model rule below — do **not** stream `message.thinking`).
- After a turn, read `prompt_eval_count` from the final chunk to **calibrate token counting** (see §7).
- **Pass `num_ctx` explicitly per request** from the registry's `default_ctx`. Ollama defaults `num_ctx` to **4096 regardless of the model's real window**, so without this a model's advertised context (e.g. GPT-OSS's 128K) is silently capped. *(Verified.)*
- Base URL from `OLLAMA_BASE_URL` (default `http://localhost:11434`).

### Reasoning ("thinking") models — output field handling

**The `chat()` contract above is incomplete for reasoning models, and this catches you the hard way.** Reasoning models split their output across **two** fields — `message.content` (the real answer) and `message.thinking` (internal chain-of-thought) — and they do **not** agree on how they use them:

- **Qwen3 14B** routes *all* output to `message.thinking` by default and leaves `message.content` empty. A naive "stream `content`" implementation yields **zero visible tokens** while the model thinks for minutes, and it may never reach a final answer within the token budget.
- **GPT-OSS 20B** populates *both* fields per chunk simultaneously — reasoning in `thinking`, the answer in `content`. A naive "fall back to `thinking` when `content` is empty" fix (which rescues Qwen3) **backfires here**: it streams raw chain-of-thought live, then appends the real answer — leaking the model's internal reasoning into the chat UI.

**Required handling (verified against both models):**
1. Send `"think": false` by default — clean, direct answers with no reasoning-latency tax. *(Reasoning quality drops on hard tasks, so expose `think` as a per-model / per-request option for callers who want it — see below.)*
2. **Only ever stream `message.content` live.** Never stream `message.thinking` to the UI.
3. **Buffer `message.thinking` silently.** Use it **only** as a last-resort fallback if `content` stayed completely empty for the whole turn — so nothing is lost if a future model only ever populates `thinking`.

**If you later expose reasoning** (`think: true`): render `thinking` in a separate, collapsed "reasoning" affordance in the UI (with a "thinking…" indicator), never inline in the answer stream. Treat `think` as a per-model capability flag carried in the registry, not a global setting — Qwen3, GPT-OSS, and Gemma 4 all expose reasoning but format it differently.

---

## 7. Orchestration: Context Budget Allocator

This is the core engineering. The orchestrator's job each turn is to **pack the model's context window** within a token budget, then stream the completion.

### Budget
```
budget = model_default_ctx − options.max_tokens − safety_margin
```
Pick a `safety_margin` (e.g. 512–1024 tokens). Reserve the **anchors** first (system prompt + current user message), then fill the remainder by priority until the budget is exhausted.

### Packing order (highest priority first)
1. **System prompt / persona** — fixed, front-loaded. *(anchor — always present)*
2. **Rolling summary** — older turns compressed into a running summary block. *(managed)*
3. **Retrieved chunks** — top-k semantically relevant past turns. *(managed, optional)*
4. **Recent turns** — verbatim, most recent first. *(managed)*
5. **Current message** — the new user input. *(anchor — always present)*

Overflowing recent turns are **compressed into the rolling summary**, not dropped.

### Progressive levels (build the allocator interface first; each level is just a provider plugged into it)
- **Level 0 — Raw passthrough.** Send full history every turn. Simplest. With GPT-OSS 20B's 128K window you can ride this far. **Start here.**
- **Level 1 — Sliding window.** System prompt + last N turns; drop oldest.
- **Level 2 — Rolling summary.** When history crosses a token threshold, `summarize()` the older turns into a cached running summary; keep recent turns verbatim. Sweet spot for local single-user chat.
- **Level 3 — Retrieval memory.** Embed each turn, store in pgvector, retrieve top-k relevant past turns by semantic similarity instead of pure recency.
- **Level 4 — Hybrid structured memory.** Pinned facts (extracted preferences/entities) + rolling summary + vector recall + recent turns, all assembled by the budget allocator.

### Token counting
- Don't hand-roll per-model tokenizers. Use a cheap heuristic (≈4 chars/token) plus the `safety_margin`, and **calibrate** against the real `prompt_eval_count` Ollama returns. Refuse to exceed the window — a hard guard in the allocator is mandatory at every level.

### Optional model routing (`orchestrator/router.py`)
Cheap classifier or a heuristic (message length, history depth) selects a fast model (`qwen3:8b`) for trivial turns and the strong model (`gpt-oss:20b`) for hard reasoning. Since Ollama keeps multiple models resident (VRAM permitting), routing is just a different `model` per request — the cheapest "smart" feature available.

---

## 8. Session Storage

Move session state **server-side**, keyed by `session_id`. SQLite is plenty for a single box (Postgres if it's already available). Server-side storage is what makes Levels 2–4 possible — the orchestrator owns history rather than trusting the client to resend it.

Store per session: ordered turns `[{role, content, tokens, ts}]`, the cached rolling summary, and (Level 3+) embeddings or a pgvector reference.

---

## 9. Environment Variables

```bash
# Backend selection — anthropic | ollama | mock
LLM_BACKEND=ollama
DEFAULT_MODEL=GPT-OSS 20B        # friendly name from the registry

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
# Per-model context is read from the registry default_ctx; override if needed.

# Generation defaults
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1024
CONTEXT_SAFETY_MARGIN=768

# Context management
CONTEXT_LEVEL=0                  # 0..4, raises as features land
SUMMARY_TRIGGER_TOKENS=0         # 0 = disabled (Level<2)

# Session storage
SESSION_DB_URL=sqlite:///./sessions.db

# Optional cloud fallback
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 10. Design Constraints (carry over from the source project)

- **Python only** — no Node.js / npm.
- Config via **pydantic-settings**; structured logging via **structlog** (JSON).
- Backend selectable via the **`LLM_BACKEND`** env var; keep the shim clean enough that `llama.cpp`-direct or `vLLM` drop in later.
- Keep a **`MockBackend`** so the full test suite runs with **no GPU and no network**. This lets you test the orchestration logic — budget packing, summary triggering, routing, the hard window guard — deterministically. That logic is where the real bugs will live.
- Never log model API keys or credentials.
- Streaming is mandatory for chat UX — a non-streaming reply feels broken.

---

## 11. Build Order (first vertical slice)

The minimal end-to-end that proves the architecture:

1. `llm/backends/base.py` — define the `chat()` / `summarize()` contract.
2. `llm/backends/ollama_backend.py` — implement `chat()` over Ollama `/api/chat` with streaming.
3. `llm/backends/mock_backend.py` — deterministic streaming for tests.
4. `llm/registry.py` — the curated model list (§3).
5. `api/routes_chat.py` — `POST /chat` returning an SSE stream; wire client-disconnect → upstream abort.
6. `api/routes_models.py` + `api/routes_meta.py` — `GET /models`, `GET /health`.
7. `api/templates/ui.html` — model dropdown + streaming transcript.
8. `orchestrator/context_budget.py` — start at **Level 0** with a hard token-budget guard that refuses to exceed the window.
9. `session/store.py` — SQLite session persistence.

Once this slice runs, every later feature — summarizer, pgvector recall, model routing, pinned facts — slots into the allocator one provider at a time **without touching the API or the UI**.

---

## 12. Open Decisions (resolve before/while building)

1. **Session storage:** server-side SQLite (recommended — enables Levels 2–4) vs. client-accumulated history. This spec assumes server-side.
2. **Model picker scope:** curated shortlist (recommended) vs. every pulled Ollama model. This spec assumes a curated registry.
3. **Single-user vs. multi-user:** if multi-user is on the horizon, plan to swap Ollama for vLLM later (better concurrency / NVFP4-MXFP4 serving). The shim makes this a backend change only.

---

## 13. Test Checklist

- [ ] `GET /health` returns ok with no model loaded.
- [ ] `GET /models` returns the curated registry.
- [ ] `POST /chat` streams tokens via SSE end-to-end against `MockBackend` (no GPU).
- [ ] Client disconnect aborts the upstream call.
- [ ] Allocator refuses any assembly that would exceed `budget` (hard guard).
- [ ] Level 0 sends full history; Level 1 truncates correctly; Level 2 triggers `summarize()` at `SUMMARY_TRIGGER_TOKENS`.
- [ ] Token estimate calibrates toward Ollama's reported `prompt_eval_count`.
- [ ] `/ui` model dropdown is populated from `GET /models` and per-request model override works.
- [ ] Against a reasoning model (Qwen3 14B, GPT-OSS 20B): only `message.content` reaches the UI; `message.thinking` is never streamed; an empty-content turn still yields a fallback answer.
- [ ] `num_ctx` is sent per request from `default_ctx`; a long-context turn is not silently capped at 4096.
