#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${RK_CAMERA_CONFIG:-${SCRIPT_DIR}/config.yaml}}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="${PWD}/${CONFIG_PATH}"; fi
cd "$SCRIPT_DIR"
exec "${PYTHON:-python3}" app.py --config "$CONFIG_PATH" "$@"
