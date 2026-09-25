#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dinfer_dir="${OLED_DINFER_REPO:-$(dirname "${repo_dir}")/dInfer}"

export OLED_DINFER_PATH="${OLED_DINFER_PATH:-${dinfer_dir}/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec oled-moe benchmark \
  --model "${OLED_MODEL_PATH:-/workspace/models/LLaDA2mini}" \
  --device cuda:0 \
  --gen-length "${OLED_GEN_LENGTH:-32}" \
  --block-length "${OLED_BLOCK_LENGTH:-16}" \
  --cache-size "${OLED_CACHE_SIZE:-100}" \
  --warmup-runs "${OLED_WARMUP_RUNS:-0}" \
  --runs "${OLED_RUNS:-1}" \
  "$@"
