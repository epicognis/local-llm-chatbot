# Local Chatbot — Session Summary & Decision Log

> Companion to `local-chatbot-build-spec.md`. This captures the full arc of the design session: what we're building, every decision and its rationale, the technical concepts we worked through, and the build status with the bugs found in practice. The spec is the *what to build*; this is the *why*, plus the reference material behind it.

---

## 1. What We Set Out to Build

A **local, single-box chatbot** running on one **RTX 5060 Ti 16GB** GPU, with three pillars:

1. A selection of popular, capable open-weight LLMs that fit the 16GB card.
2. A **FastAPI** web server with a browser **`/ui`** that takes a prompt and streams the model's reply.
3. An **orchestration layer** that builds and manages prompt context, designed to grow from a trivial passthrough into summarization and retrieval-augmented memory.

**Key strategic decision:** rather than start from scratch, we **generalize an existing internal NL→SQL engine** (documented in `nl-query-engine-knowledge-transfer.md`). That project already contains ~80% of the scaffolding: a backend-agnostic LLM shim, a FastAPI + Jinja2 `/ui`, an `LLM_BACKEND` env-var swap, structlog logging, pydantic-settings config, a `MockBackend` for tests, and a multi-turn history pattern. We keep those, **drop** the SQL-specific layers (guardrails, SQL validator, Postgres executor, CSV/Excel/chart composer), and swap `generate_sql()` for a `chat()` method.

---

## 2. Hardware & Runtime Context

| Item | Detail |
|---|---|
| GPU | RTX 5060 Ti 16GB, Blackwell (sm_120), ~448 GB/s GDDR7 |
| Practical model ceiling | ~20B params on a single card; 30B/32B dense models do not fit at 4-bit |
| Serving runtime | **Ollama** (bundles recent llama.cpp with Blackwell CUDA support) |

**Important correction made early:** the Blackwell/sm_120 + nightly-PyTorch pain documented in the source project was specific to the **interpretability work** (TransformerLens). For **serving** a chatbot through Ollama/llama.cpp, Blackwell is supported on stable releases — no nightly torch needed. That research venv stays quarantined.

---

## 3. Model Selection & Rationale

The card's practical ceiling is ~20B params. Within that envelope we curated a lineup mapping friendly names → `(backend, model, default_ctx)`:

| Model | Size on card | Role | Why |
|---|---|---|---|
| **GPT-OSS 20B** (default) | MXFP4, ~12–13 GB | Primary reasoning model | Sparse MoE (~21B total / ~3.6B active), 128K context, ~80 tok/s — MXFP4 maps natively onto Blackwell tensor cores. |
| **Gemma 4 12B** | 4-bit, ~7 GB | Agentic / multimodal / family variety | Dense ~11.95B, 256K context, native tool-use, multimodal, MTP drafters. (Replaced the original Gemma 3 12B entry.) |
| Qwen3 14B | Q4_K_M, ~9 GB | Strong all-round dense | Code + multilingual; ~33 tok/s; real window 40960. |
| Qwen3 8B | ~6 GB | Fast tier / routing fallback | Very high prompt-processing throughput for trivial turns. |
| Llama 3.1 8B | Q4, ~5 GB | Fast familiar baseline | ~70 tok/s. |

**Excluded (overflow 16GB):** `qwen3-coder:30b` and any 30B+; the larger Gemma variants (Gemma 3 27B, Gemma 4 26B MoE) run ~16–18 GB at 4-bit — too tight once KV cache is added.

**Benchmark trap flagged:** generic "fit calculators" rate `gpt-oss:20b` at ~30 tok/s / 4K by assuming Q4_K_M. That's wrong — it ships natively as MXFP4 (smaller + faster on Blackwell). The ~80 tok/s card-specific figure at 128K is the trustworthy one.

---

## 4. Serving Runtime Decision

Three realistic options, weighed:

- **Ollama (chosen for v1).** Holds multiple models pullable + hot-swappable, keeps weights resident, one HTTP API — directly serves the "selectable model" requirement and maps onto the existing shim. **Use the `/api/chat` endpoint** (role-tagged messages), not `/api/generate`, so Ollama applies each model's chat template and we never hand-maintain per-model formatting.
- **llama.cpp server (later).** Finer control (explicit MXFP4, context/batch tuning) at the cost of managing models yourself. Reach for it when tuning.
- **vLLM (later, if multi-user).** Best concurrency/throughput and NVFP4/MXFP4 path, but heavier; overkill for single-user chat.

The backend shim keeps these interchangeable — swapping runtime is a backend change only.

---

## 5. Architecture (generalized from the NL→SQL engine)

```
User prompt (browser /ui)
  → FastAPI  (POST /chat, SSE stream)
  → Orchestrator  (assemble context within token budget; optional model routing)
  → LLM Backend shim  (Ollama /api/chat, streaming)
  → token stream back to /ui
```

Stateless HTTP server; session/context state lives in the orchestration layer. Module map (`llm/`, `orchestrator/`, `session/`, `api/`, `config/`, `metrics/`) is laid out in spec §4.

### API & UI shape
- `POST /chat` → **SSE streaming** (the main shift from the SQL tool, which returned whole result sets; chat needs live tokens or it feels broken). **Must abort the upstream call on client disconnect.**
- `GET /models` → registry listing so the UI dropdown is dynamic.
- `GET /health`, `GET /ui` (two-pane: model picker + session list / streaming transcript).
- **Session state server-side**, keyed by `session_id` (SQLite to start). This is what makes the higher context-management levels possible — the orchestrator owns history rather than trusting the client to resend it.

---

## 6. Orchestration & Context Management (the core engineering)

The orchestrator's job each turn: **pack the model's context window within a token budget**, then stream.

### Budget allocator
```
budget = model_default_ctx − max_output_tokens − safety_margin
```
Reserve the **anchors** first (system prompt + current message), then fill the remainder by priority until exhausted. Packing order, highest priority first:

1. System prompt / persona — *anchor, always present*
2. Rolling summary — older turns compressed
3. Retrieved chunks — top-k semantically relevant (optional)
4. Recent turns — verbatim, most recent
5. Current message — *anchor, always present*

Overflowing recent turns are **compressed into the rolling summary**, not dropped.

### Progressive levels (build the allocator interface first; each level is just a provider)
- **Level 0 — Raw passthrough** (full history every turn). Where the build currently sits. GPT-OSS 20B's 128K window lets you ride this far.
- **Level 1 — Sliding window** (system + last N turns).
- **Level 2 — Rolling summary** (summarize older turns past a threshold). Sweet spot for local single-user chat.
- **Level 3 — Retrieval memory** (embed turns into pgvector, retrieve by similarity).
- **Level 4 — Hybrid structured memory** (pinned facts + summary + recall + recent).

### Token counting
Don't hand-roll tokenizers — use a heuristic (~4 chars/token) plus safety margin, **calibrated against Ollama's reported `prompt_eval_count`**. A hard guard that refuses to exceed the window is mandatory at every level.

### Model routing
A cheap classifier/heuristic (message length, history depth) picks a fast model (`qwen3:8b`) for trivial turns and the strong model (`gpt-oss:20b`) for hard reasoning. Since Ollama keeps models resident, routing is just a different `model` per request — the cheapest "smart" feature available. **Design note that emerged from the bug work:** the router is also the right place to flip reasoning (`think: true`) on for hard turns, spending reasoning latency only where it's earned.

---

## 7. Concept Deep-Dives

### 7a. What class of model is gpt-oss-20b?
- **Provenance/licensing:** open-weight, Apache 2.0, from OpenAI (Aug 2025).
- **Architecture:** sparse **Mixture-of-Experts** — ~21B total params, ~3.6B active per token.
- **Capability class:** a **reasoning model** (explicit chain-of-thought, adjustable reasoning effort).
- **Format:** ships natively in **MXFP4** (~12–13 GB), which is why it's compact and fast on Blackwell.
- Bigger sibling: `gpt-oss-120b` (117B total / ~5.1B active).
- One-line: an open-weight, MXFP4-native, sparse-MoE reasoning model in the ~20B-total / ~3.6B-active tier.

### 7b. tok/s as a performance metric
- A **token** ≈ ¾ of an English word (~4 chars). tok/s ≠ words/sec, and token counts differ across model families, so tok/s is only cleanly comparable within a family.
- **Two rates hide under "tok/s":**
  - **Prefill (prompt-processing)** — ingests the prompt in parallel; high (hundreds–thousands tok/s). Sets **time-to-first-token (TTFT)** with prompt length.
  - **Decode (generation)** — emits new tokens one at a time (autoregressive); much lower (tens of tok/s). This is what people usually quote.
  - Full latency = TTFT **+** decode rate, not one number.
- **What sets the decode ceiling:** for single-stream local inference, decode is **memory-bandwidth bound**, not compute bound — each token reads the active weights + KV cache from VRAM. So decode tok/s ≈ bandwidth ÷ bytes-read-per-token. This explains why the 5060 Ti's 448 GB/s is the real ceiling, why quantization speeds things up (fewer bytes/weight), and why MoE speeds things up (fewer active weights).
- **Context-dependent:** decode tok/s falls as context grows (KV cache grows and must be re-read), so any figure is meaningless without a stated context length.
- **Batching:** aggregate tok/s across many concurrent requests ≠ per-request tok/s. Ollama optimizes the single stream you feel; vLLM optimizes aggregate — the crux of the single- vs multi-user decision.
- **Perception anchor:** humans read ~5–7 tok/s; past ~10 tok/s text outpaces reading, past ~70 a single reader stops noticing. But for long/agentic outputs where you wait for the whole thing, raw tok/s keeps mattering linearly.

### 7c. Sparse Mixture of Experts (MoE)
- A dense layer's single feed-forward net is replaced by many parallel **experts** plus a small **router** (gating network) that picks a few experts per token; their outputs are combined by weighted sum.
- **"Sparse"** = only a small subset of experts is active per token (e.g. 4 of 32); the rest sit idle.
- **Total vs active params:**
  - **Total** = all experts summed; all must live in VRAM (router may call any). → gpt-oss-20b still loads ~12–13 GB.
  - **Active** = experts that fire per token (~3.6B); determines per-token compute and bandwidth → speed.
- **The trade:** big-model knowledge capacity at small-model speed, paid for with big-model VRAM. A 21B MoE decodes roughly like a ~4B dense model (hence ~80 tok/s vs ~33 for a dense 14B). Sparsity buys speed, **not** a smaller footprint.
- **Clarifications:** "experts" are not human-interpretable specialists — they're learned partitions, routed emergently; training adds a load-balancing term so experts are used evenly.
- This is the exact axis separating the two main models: **gpt-oss-20b = sparse MoE** (fast decode, larger footprint); **Gemma 4 12B = dense** (smaller footprint, no sparsity speedup — it recovers some via MTP speculative-decoding drafters).

---

## 8. Gemma 4 12B — Why It Was Added

Verified as a real ~June 2026 release from Google DeepMind (post-dated the original spec). Swapped in to **replace** the Gemma 3 12B entry because it supersedes it:
- Dense ~11.95B, **Apache 2.0**, ~7 GB at 4-bit — fits the same slot with more VRAM headroom.
- **256K context**, **native agentic tool-use / function-calling**, a dedicated **Gemma Skills Repository**, an explicit **thinking** reasoning mode, and **MTP drafters** (speculative decoding) for latency.
- **Encoder-free "unified" multimodal** (text/image/audio/video) — unused by a text-only bot but a free future input path.
- Benchmarks reportedly **near Google's 26B MoE**.
- **Caveat:** weeks old — confirm the Ollama tag and a stable 4-bit quant before committing (brand-new encoder-free multimodal architectures can lag in runtime support).

### GPT-OSS 20B vs Gemma 4 12B — head-to-head
| Dimension | GPT-OSS 20B | Gemma 4 12B |
|---|---|---|
| Architecture | Sparse MoE (21B/3.6B) | Dense ~11.95B |
| Footprint | ~12–13 GB | ~7 GB |
| Context | 128K | 256K |
| Modality | Text only | Text + image + audio + video |
| Tool use | Strong | Native FC + Skills Repo |
| Speed on card | ~80 tok/s | competitive (dense + MTP); verify |
| Maturity | Aug 2025, rock-solid | ~June 2026, verify quant |
| Edge | Hard text reasoning / code | Long context, multimodal, agent tooling |

**Why the chosen order (GPT-OSS first, then Gemma 4) makes sense:** GPT-OSS gives a stable, well-understood baseline with the strongest raw reasoning, ideal for building/debugging the orchestration layer against a fixed target. Switching to Gemma 4 later is a *capability* change (multimodal, longer context, agent tooling), not a stability gamble — best done once its runtime support settles. Two switch gotchas: tool-call format differs between models (orchestrator's tool layer may need per-model handling), and reasoning controls differ (GPT-OSS's effort knob vs Gemma's thinking mode) — expose both as per-model options.

---

## 9. Build Status (as reported)

**Full vertical slice built and verified end-to-end**, matching spec Build Order §11. All modules present: `llm/backends/{base,ollama_backend,mock_backend}.py`, `llm/registry.py`, `orchestrator/{context_budget,orchestrator}.py`, `session/store.py` (SQLite), `api/{main,routes_chat,routes_models,routes_meta,routes_sessions}.py`, `api/templates/ui.html` + `api/static/chat.css`, `metrics/instrument.py`. Context allocator at **Level 0** (full-history passthrough + hard budget guard).

**Verified:** `/health`, `/models`, `/chat` (SSE), `/sessions`, `/ui` all working; client-disconnect aborts the upstream Ollama call; multi-turn sessions persist in SQLite, survive reload, listed in sidebar; driven live in a real browser (Playwright) against the mock backend and against Ollama with Qwen3 14B, GPT-OSS 20B, and Gemma 4 12B — all three produce correct, clean multi-turn responses through the full stack.

**Currently pulled locally:** GPT-OSS 20B (13 GB), Qwen3 14B (9.3 GB), Gemma 4 12B (7.6 GB). Qwen3 8B and Llama 3.1 8B registered but not yet pulled.

---

## 10. Bugs Found in Practice & Fixes (the spec's first pass missed these)

### Bug 1 — Reasoning models split output across `content` and `thinking`, inconsistently
The `chat()` contract assumed a model just streams `message.content` deltas. In reality:
- **Qwen3 14B** routes all output to `message.thinking` by default, leaving `content` empty → first attempt streamed for 2 minutes with **zero visible tokens** (thinking the whole time, never reaching a final answer in budget).
- **GPT-OSS 20B** populates **both** fields per chunk simultaneously → a naive "fall back to `thinking` if `content` empty" fix (which rescued Qwen3) **backfired**, streaming raw chain-of-thought live then appending the real answer — **leaking internal reasoning into the UI**.

**Fix shipped:** Ollama backend now passes `think: false` by default (clean direct answers, no reasoning-latency tax) and **only ever streams `content` live**; `thinking` is **buffered silently** and used only as a last-resort fallback if `content` stayed completely empty for the whole turn. (Not Ollama-doc-obvious — now documented in spec §6.)

### Bug 2 — Ollama silently caps context at 4096
Ollama defaults `num_ctx` to 4096 regardless of the model's real window. Now passing `num_ctx` **explicitly per request** from the registry's `default_ctx`, otherwise GPT-OSS's 128K window would be silently capped.

### Other deviations corrected
- **Qwen3 14B real context is 40960**, not the 16K placeholder — corrected in the registry.
- **Starlette 1.x** changed `Jinja2Templates.TemplateResponse` — `request` is now positional-first, not a context-dict key. `TemplateResponse(request, "ui.html", {...})`. Bites anyone copying an older `/ui` route.
- **Gemma 3 → Gemma 4** swap (`gemma4:12b`, 7.6 GB). Native ctx 256K but a practical `default_ctx` of 32768 set to keep KV-cache headroom on the 16GB card.

---

## 11. Spec Updates Made This Session

The build spec (`local-chatbot-build-spec.md`) was updated to capture all of the above:
- New §6 subsection **"Reasoning ('thinking') models — output field handling"** with both failure modes and the `think: false` / stream-content-only / buffer-thinking fix, plus forward guidance to treat `think` as a per-model registry flag rendered in a separate collapsed UI affordance.
- `num_ctx`-defaults-to-4096 rule added to §6.
- "`default_ctx` is a practical value, not the max" note + verified windows (GPT-OSS 131072, Qwen3 40960, Gemma 4 262144 → practical 32768) under the registry table.
- Qwen3 14B row corrected to its real 40960 window.
- Starlette `TemplateResponse` positional-first gotcha on the `/ui` route.
- Two new test-checklist items (thinking-field handling; `num_ctx` not silently capped).
- A `(Verified)` status banner marking notes confirmed by the end-to-end run.

---

## 12. Open Decisions & Next Steps

**Open decisions** (from spec §12, still standing):
1. Session storage — server-side SQLite (chosen) vs client-accumulated.
2. Model-picker scope — curated shortlist (chosen) vs every pulled Ollama model.
3. Single- vs multi-user — if multi-user looms, plan a later vLLM swap.

**Natural next steps:**
- **Climb the context ladder:** Level 0 → Level 2 (rolling summary) is the highest-value jump for long sessions; wire the `summarize()` provider into the allocator.
- **Model routing + reasoning gating:** implement `orchestrator/router.py`; let the same "hard turn" signal that picks GPT-OSS also flip `think: true`, so reasoning is spent only where it earns its latency.
- **Per-model capability flags:** formalize `think`, tool-call format, and `default_ctx` as registry-carried per-model attributes (the thinking bugs showed these can't be global).
- **Verify Gemma 4 runtime maturity** before leaning on it as more than a registry entry.
- Later: pgvector retrieval (Level 3), pinned-facts memory (Level 4), metrics-driven cost/throughput forecasting.
