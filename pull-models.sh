#!/usr/bin/env bash
#
# pull-models.sh — download every model in the registry into Ollama.
#
# Single source of truth: the tags come from llm/registry.py, so this stays in
# sync with what the UI offers. Prefers the `ollama` CLI when present (nicer
# progress); otherwise falls back to the HTTP pull API. Continues past failures
# and prints a summary at the end.
#
#   ./pull-models.sh                                  # OLLAMA_BASE_URL or localhost
#   OLLAMA_BASE_URL=http://192.168.1.50:11434 ./pull-models.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# Use the project venv's interpreter if run.sh has created one.
PY=""
for cand in .venv/Scripts/python.exe .venv-linux/bin/python .venv/bin/python; do
  [ -x "$cand" ] && { PY="$cand"; break; }
done
if [ -z "$PY" ]; then
  if [ -n "${MSYSTEM:-}" ]; then PY=python; else PY=python3; fi
fi

# llm/registry.py is stdlib-only, so any Python 3.10+ can read it.
mapfile -t TAGS < <("$PY" -c "from llm.registry import Registry; print('\n'.join(s.model for s in Registry().list()))" | tr -d '\r')
if [ "${#TAGS[@]}" -eq 0 ]; then
  echo "[pull-models] could not read tags from llm/registry.py" >&2
  exit 1
fi

if ! curl -sf -o /dev/null "$BASE_URL/api/tags"; then
  echo "[pull-models] ERROR: no Ollama daemon reachable at $BASE_URL" >&2
  echo "  start Ollama, or set OLLAMA_BASE_URL to where it runs." >&2
  exit 2
fi

echo "[pull-models] target: $BASE_URL"
echo "[pull-models] models: ${TAGS[*]}"

OK=() FAIL=()
for tag in "${TAGS[@]}"; do
  echo "----------------------------------------------------------------"
  echo "[pull-models] pulling $tag ..."
  if command -v ollama >/dev/null 2>&1; then
    if OLLAMA_HOST="$BASE_URL" ollama pull "$tag"; then OK+=("$tag"); else FAIL+=("$tag"); fi
  else
    # HTTP fallback: stream status lines from the pull API.
    if curl -sf -X POST "$BASE_URL/api/pull" -H 'Content-Type: application/json' \
         -d "{\"model\":\"$tag\",\"stream\":true}" \
       | "$PY" -c "import sys, json
last = ''
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except ValueError:
        continue
    if o.get('error'):
        print('    ERROR:', o['error']); sys.exit(1)
    s = o.get('status', '')
    if s != last:
        print('   ', s); last = s"; then
      OK+=("$tag")
    else
      FAIL+=("$tag")
    fi
  fi
done

echo "================================================================"
echo "[pull-models] done. ok=${#OK[@]}  failed=${#FAIL[@]}"
[ "${#OK[@]}" -gt 0 ] && echo "  ok:     ${OK[*]}"
[ "${#FAIL[@]}" -gt 0 ] && echo "  FAILED: ${FAIL[*]}"
[ "${#FAIL[@]}" -eq 0 ]
