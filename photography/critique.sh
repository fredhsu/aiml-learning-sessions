#!/usr/bin/env bash
# Send frames to the tutor for a fixed-rubric critique.
#
# The rubric requires your predicted scores to exist BEFORE critique. This
# script does not check that for you -- it is your commitment, not the tool's.
#
# Usage: ./critique.sh shoots/2026-08-30-riverside/keepers/*.jpg
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
session_id="${PI_TUTOR_SESSION_ID:-photography-critique}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <image> [image...]" >&2
  exit 64
fi

cd "$repo_dir"

args=()
for f in "$@"; do
  [[ -f "$f" ]] || { echo "error: no such file: $f" >&2; exit 1; }
  args+=("@$f")
done

exec pi --approve --session-id "$session_id" \
  "@rubrics/image-critique-rubric.md" "${args[@]}" \
  "Critique these frames against the fixed rubric. Score blind: do not ask for my shot intent until after you have scored every dimension, then ask for it and report intent achievement separately. Complete every mandatory per-frame and per-set field, including the evaluator-drift check."
