#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dinfer_dir="${OLED_DINFER_REPO:-$(dirname "${repo_dir}")/dInfer}"
expected_revision="1ffeb961cd258bede74fcf5ca8a416ae6d57b18f"

if [[ ! -f "${dinfer_dir}/python/dinfer/__init__.py" ]]; then
  echo "dInfer checkout not found at ${dinfer_dir}" >&2
  exit 1
fi

if git -C "${dinfer_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  actual_revision="$(git -C "${dinfer_dir}" rev-parse HEAD)"
  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    echo "dInfer revision mismatch: expected ${expected_revision}, got ${actual_revision}" >&2
    exit 1
  fi
elif [[ "${OLED_DINFER_SKIP_REVISION_CHECK:-0}" != "1" ]]; then
  echo "dInfer checkout has no Git metadata, so its revision cannot be verified." >&2
  echo "Use the pinned Git checkout or set OLED_DINFER_SKIP_REVISION_CHECK=1 after manual review." >&2
  exit 1
fi

python -m pip install -e "${repo_dir}[dev,server]"
printf 'export OLED_DINFER_PATH=%q\n' "${dinfer_dir}/python"
