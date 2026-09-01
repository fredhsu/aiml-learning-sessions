#!/usr/bin/env bash
# Launch Pi as the persistent photography curriculum tutor.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
session_id="${PI_TUTOR_SESSION_ID:-photography-tutor}"

if ! command -v pi >/dev/null 2>&1; then
  echo "error: pi is not on PATH" >&2
  exit 127
fi

if [[ ! -f "$repo_dir/AGENTS.md" ]]; then
  echo "error: run this script from the curriculum repository; required AGENTS.md is missing" >&2
  exit 1
fi

if ! command -v exiftool >/dev/null 2>&1; then
  echo "note: exiftool is not installed. Raw files cannot be verified without it." >&2
  echo "      JPEG verification still works via Pillow. Install with:" >&2
  echo "      sudo pacman -S perl-image-exiftool" >&2
fi

cd "$repo_dir"

exec pi \
  --approve \
  --session-id "$session_id" \
  "$@"
