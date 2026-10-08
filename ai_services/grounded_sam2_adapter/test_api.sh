#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-8090}"
PROMPT="${3:-banana}"
PYTHON_BIN="${PYTHON:-python}"
PAYLOAD="$("$PYTHON_BIN" -c 'import json, sys; print(json.dumps({"text_prompt": sys.argv[1], "include_mask_rle": False, "save_debug_image": True}))' "$PROMPT")"
curl --fail-with-body -sS "http://${HOST}:${PORT}/health"
echo
curl --fail-with-body -sS -X POST "http://${HOST}:${PORT}/api/v1/detect_segment" \
  -H 'Content-Type: application/json' \
  --data "$PAYLOAD" | "$PYTHON_BIN" -m json.tool
