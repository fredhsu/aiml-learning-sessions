#!/usr/bin/env bash
# Launch Pi as the persistent robot-learning curriculum tutor.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
system_prompt="$repo_dir/curriculum-system-prompt.md"
session_id="${PI_TUTOR_SESSION_ID:-robot-learning-tutor}"

if ! command -v pi >/dev/null 2>&1; then
  echo "error: pi is not on PATH" >&2
  exit 127
fi

if [[ ! -f "$system_prompt" || ! -f "$repo_dir/AGENTS.md" ]]; then
  echo "error: run this script from the curriculum repository; required prompt files are missing" >&2
  exit 1
fi

cd "$repo_dir"

exec pi \
  --approve \
  --append-system-prompt "$(<"$system_prompt")" \
  --session-id "$session_id" \
  "$@"
