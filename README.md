# Local Chatbot

A local, single-box chatbot: a **FastAPI** server with a streaming browser UI,
selectable open-weight LLMs served by **Ollama** on one RTX 5060 Ti 16GB, and an
**orchestration layer** that packs every turn into the model's context window —
summarizing older conversation when it no longer fits.

- `local-chatbot-build-spec.md` — what to build (API contract, rules, test checklist)
- `session-summary-decision-log.md` — why: every decision, the bugs found in practice

## Quick start

```bash
./run.sh            # Git Bash on Windows (recommended) or WSL/Linux
```

`run.sh` creates a venv on first run (`.venv` on Windows, `.venv-linux` on
WSL/Linux), installs `requirements.txt` when it changes, stops anything still
listening on **port 8080**, starts the server, and prints the URLs:

```
  Chatbot is up
    This machine : http://localhost:8080/ui
    Your network : http://192.168.1.128:8080/ui
```

Ollama must be running with the models pulled — `./pull-models.sh` pulls every
model in the registry. To try the UI with no GPU or Ollama at all, run
`LLM_BACKEND=mock ./run.sh` (deterministic echo replies).

Settings come from environment variables or a `.env` file (see `.env.example`).

## The UI

- **Model picker** with each model's context window; sessions remember their model.
- **Per-turn status line**: waiting → loading model / summarizing → thinking (N
  tokens) → generating (~N tok · X tok/s) → final stats (first token, tokens,
  decode tok/s). A watchdog flags a stall; **Stop** cancels the turn (the partial
  reply is kept). Errors appear under the reply and are never saved to history.
- **Monitor panel** (polled every 1.5 s): GPU utilization, VRAM, power, temperature,
  CPU and system RAM, each resident model's VRAM/RAM split, and a warning when a
  model claims GPU memory the device isn't actually using (it fell back to CPU).
  **Purge GPU** unloads every model (weights stay on disk).
- **LLM Config panel**: Thinking mode, Keep model loaded, Force CPU (for GPU-vs-CPU
  comparisons), Auto-load on select, context size, max tokens, temperature,
  top-p, top-k.
- Sessions: rename, delete, reopen. On a phone the sidebar becomes a drawer (☰).

## Models (`llm/registry.py`)

| Model | Tag | Context | Reasoning |
|---|---|---|---|
| Gemma 4 12B (default) | `gemma4:12b` | 32768 | optional |
| GPT-OSS 20B | `gpt-oss:20b` | 65536 | always on, minimum `"low"` |
| Llama 3.1 8B | `llama3.1:8b` | 32768 | — |
| Qwen3 14B | `qwen3:14b` | 16384 | optional |
| Qwen3 8B | `qwen3:8b` | 32768 | optional |

Each context window is the largest measured to fit **100% on the GPU** on this
card (GPT-OSS at 131072 spilled to CPU and ran at ~47 tok/s instead of ~86). Raise
it per turn from the LLM Config panel. Only one model is kept resident at a time.

## Context management

`CONTEXT_LEVEL` selects the strategy; the hard window guard applies at every level.

- **0** — full history every turn; refuses (doesn't trim) once it no longer fits.
- **1** — sliding window: as many recent turns as fit.
- **2** (default) — **recursive summarization** (`recursive_summarization/`, a
  stdlib-only library). While history fits, every turn is passed verbatim with zero
  extra model calls. Once it doesn't, older turns are compressed into a running
  summary (injected as a system message) while recent turns stay verbatim. Steady
  state uses a cheap incremental `fold()`; a full recompaction every
  `SUMMARY_RECOMPACT_CADENCE` folds resets drift. State is persisted per session.
- **3–4** (retrieval / hybrid memory) — not yet implemented.

Token counts use a ~4 chars/token heuristic calibrated per model against Ollama's
reported `prompt_eval_count` (calibration can only make the guard stricter).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/ui` | Browser chat UI (`/` redirects here) |
| POST | `/chat` | Streamed reply (SSE): `status`, `delta`, `thinking`, `stats`, `error`, then `[DONE]` |
| GET | `/models` | Curated registry for the dropdown |
| POST | `/load` · `/unload` | Eager-load a model (purging others) · unload all |
| GET | `/monitor` | Hardware + Ollama snapshot for the monitor panel |
| GET | `/health` | Liveness + Ollama version and resident models |
| GET/POST/PATCH/DELETE | `/sessions[/{id}]` | List, create, read, rename, delete sessions |

## Layout

```
api/                     FastAPI app, routes, UI template + static (chat.css, lissajous.js)
config/settings.py       pydantic-settings
llm/registry.py          friendly name -> tag, context window, reasoning controls
llm/backends/            base shim, Ollama (/api/chat streaming), mock
orchestrator/            per-turn coordinator, context budget, Level 2 wiring
recursive_summarization/ Level 2 library + its contract tests
session/store.py         SQLite sessions + turns (+ Level 2 state)
metrics/                 per-turn metrics, hardware-monitor bridge
monitor/                 standalone terminal hardware monitor (./run_monitor.sh)
tests/                   pytest suite — runs on the mock backend, no GPU/network
```

## Tests

```bash
.venv/Scripts/python -m pytest      # Git Bash
.venv-linux/bin/python -m pytest    # WSL / Linux
```

## Terminal monitor

`./run_monitor.sh` runs the standalone hardware dashboard (its own venv). It also
tails `server.log` for each turn's tokens/sec.
