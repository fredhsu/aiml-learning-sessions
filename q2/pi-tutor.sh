#!/usr/bin/env bash
# Launch Pi as the persistent robot-learning curriculum tutor.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
session_id="${PI_TUTOR_SESSION_ID:-robot-learning-tutor}"

if ! command -v pi >/dev/null 2>&1; then
  echo "error: pi is not on PATH" >&2
  exit 127
fi

if [[ ! -f "$repo_dir/AGENTS.md" ]]; then
  echo "error: run this script from the curriculum repository; required AGENTS.md is missing" >&2
  exit 1
fi

cd "$repo_dir"

exec pi \
  --approve \
  --session-id "$session_id" \
  "$@"
