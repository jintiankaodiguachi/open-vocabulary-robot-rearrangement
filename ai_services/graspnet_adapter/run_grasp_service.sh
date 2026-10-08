#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${GRASP_CONFIG:-${SCRIPT_DIR}/config.yaml}}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="${PWD}/${CONFIG_PATH}"; fi
RUNTIME_DIR="${GRASP_RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" grasp_service.py \
  --config "$CONFIG_PATH" \
  --runtime-dir "$RUNTIME_DIR" \
  --host "${GRASP_HOST:-0.0.0.0}" \
  --port "${GRASP_PORT:-8091}" "$@"
