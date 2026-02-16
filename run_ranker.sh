#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-$SCRIPT_DIR/.env.openrouter}"

if [[ -f "$OPENROUTER_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$OPENROUTER_ENV_FILE"
  set +a
fi

DATA_ROOT="$SCRIPT_DIR/data/new_data"
VOLUMES_SPEC="all"
INPUT_GLOB="*.pdf"
PROCESSING_MODE="image"
WORKSPACE_ROOT="data/workspaces"
DATASET_TAG_PREFIX="standardworks_epstein_files"
DATASET_SOURCE_LABEL="Epstein-Files GitHub (raw PDFs)"
DATASET_SOURCE_URL="https://github.com/yung-megafone/Epstein-Files"
DATASET_METADATA_FILE="$SCRIPT_DIR/data/dataset_profiles/standardworks_epstein_files.json"
GIT_OUTPUT_ROOT="contrib/fta"
TRACK_CHUNKS_IN_GIT=1

ENDPOINT="http://localhost:5555/v1"
API_FORMAT="openai"
MODEL="qwen/qwen3-vl-30b"
PROVIDER="local"
OPENROUTER_MODEL="qwen/qwen3-vl-30b-a3b-instruct"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
OPENROUTER_REFERER="${OPENROUTER_REFERER:-}"
OPENROUTER_TITLE="${OPENROUTER_TITLE:-Epstein File Ranker}"
API_KEY=""
HTTP_REFERER=""
X_TITLE=""
MODEL_EXPLICIT=0
ENDPOINT_EXPLICIT=0
MAX_PARALLEL_REQUESTS=4
PARALLEL_SCHEDULING="batch"
PARALLEL_SCHEDULING_EXPLICIT=0
IMAGE_PREFETCH=0
IMAGE_PREFETCH_EXPLICIT=0
IMAGE_MAX_PAGES=128
PDF_PAGES_PER_IMAGE=4
PDF_PART_PAGES=128
IMAGE_RENDER_DPI=120
IMAGE_DETAIL="low"
IMAGE_OUTPUT_FORMAT="jpeg"
IMAGE_JPEG_QUALITY=75
IMAGE_MAX_SIDE=1024
DEBUG_IMAGE_DIR=""
MAX_OUTPUT_TOKENS=900
TEMPERATURE=0.0
SLEEP_SECONDS=0
CHUNK_SIZE=1000
MAX_ROWS=""

RESUME=1
SKIP_MISSING=1
DRY_RUN=0
KEEP_TEMP=0

EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  ./run_ranker.sh [options] [-- <extra gpt_ranker.py flags>]

Core options:
  --volumes SPEC             Volumes to run: 1,2,5-7 or all (default: all)
  --data-root PATH           Dataset root containing VOL00001.. folders
  --workspace-root PATH      Workspace root for isolated outputs/checkpoints
  --dataset-tag-prefix NAME  Prefix for dataset tag (suffix _vol00001 is added)
  --git-output-root PATH     Root for Git-tracked chunk outputs (default: contrib/fta)
  --workspace-chunks         Keep chunks inside workspace instead of contrib/ (not tracked)
  --glob PATTERN             File glob inside each volume (default: *.pdf)
  --processing-mode MODE     auto | text | image (default: image)

Model/runtime options:
  --provider NAME            local | openrouter (default: local)
  --endpoint URL             Model endpoint base URL (default: http://localhost:5555/v1)
  --api-format FORMAT        auto | openai | chat (default: openai)
  --model ID                 Model id (default: qwen/qwen3-vl-30b)
  --api-key KEY              API key for hosted providers
  --http-referer URL         HTTP-Referer header (OpenRouter recommendation)
  --x-title NAME             X-Title header (OpenRouter recommendation)
  --openrouter-model ID      OpenRouter model id (default: qwen/qwen3-vl-30b-a3b-instruct)
  --openrouter-env-file PATH OpenRouter env file (default: .env.openrouter)
  --openrouter-api-key KEY   OpenRouter key (or set OPENROUTER_API_KEY)
  --openrouter-referer URL   OpenRouter referer override (or OPENROUTER_REFERER)
  --openrouter-title NAME    OpenRouter title override (or OPENROUTER_TITLE)
  --parallel N               Max parallel requests (default: 4)
  --parallel-scheduling MODE auto | window | batch (default: batch)
  --image-prefetch N         Extra queued image tasks beyond --parallel (default: 0)
  --image-max-pages N        Max rendered PDF pages per document (default: 128)
  --pdf-pages-per-image N    Pack N PDF pages into one tiled image block (default: 4)
  --pdf-part-pages N         Split each PDF into N-page parts (default: 128)
  --image-render-dpi N       PDF render DPI (default: 120)
  --image-detail MODE        auto | low | high (default: low)
  --image-output-format FMT  png | jpeg (default: jpeg)
  --image-jpeg-quality N     JPEG quality for prepared images (default: 75)
  --image-max-side N         Downscale long side to this px (0 disables, default: 1024)
  --debug-image-dir PATH     Save intermediate rendered/packed images + timing JSONs
  --max-output-tokens N      Max completion tokens per request (default: 900)
  --temperature FLOAT        Sampling temperature (default: 0.0)
  --sleep SECONDS            Delay between submissions (default: 0)
  --chunk-size N             Output chunk size (default: 1000)
  --max-rows N               Limit rows for smoke test per volume

Control options:
  --resume                   Resume from prior progress (default)
  --no-resume                Start fresh per selected workspace/tag
  --skip-missing             Skip volumes not yet downloaded (default)
  --strict-missing           Exit if a requested volume directory is missing
  --keep-temp                Keep temporary run directory (debugging only)
  --dry-run                  Print commands without running
  -h, --help                 Show this help

Examples:
  ./run_ranker.sh --volumes 1
  ./run_ranker.sh --provider openrouter --openrouter-api-key sk-... --volumes 1 --parallel 2
  ./run_ranker.sh --volumes 1,2,6-8 --parallel 4 --dry-run
  ./run_ranker.sh --volumes all --strict-missing
  ./run_ranker.sh --volumes 1 -- --reasoning-effort low --sleep 0.5
USAGE
}

rebuild_git_manifest() {
  "$PYTHON_BIN" - "$GIT_OUTPUT_ROOT" <<'PY'
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
pattern = str(root / "VOL*" / "epstein_ranked_*.jsonl")
chunks = []
rows_processed = 0

for path in sorted(glob.glob(pattern)):
    name = os.path.basename(path)
    match = re.match(r"epstein_ranked_(\d+)_(\d+)\.jsonl$", name)
    if not match:
        continue
    start = int(match.group(1))
    end = int(match.group(2))
    rel = Path(path).as_posix()
    if rel.startswith("./"):
        rel = rel[2:]
    chunks.append({"start_row": start, "end_row": end, "json": rel})
    try:
        with open(path, "r", encoding="utf-8") as handle:
            rows_processed += sum(1 for _ in handle)
    except OSError:
        pass

manifest = {
    "metadata": {
        "total_dataset_rows": "unknown",
        "rows_processed": rows_processed,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "chunks": chunks,
}

root.mkdir(parents=True, exist_ok=True)
manifest_path = root / "chunks.json"
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
PY
}

list_existing_volumes() {
  find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'VOL*' -print 2>/dev/null \
    | sed -E 's#.*/VOL0*([0-9]+)$#\1#' \
    | grep -E '^[0-9]+$' \
    | sort -n -u
}

expand_volumes_spec() {
  local spec="$1"

  if [[ "$spec" == "all" ]]; then
    list_existing_volumes
    return
  fi

  local token start end n
  IFS=',' read -r -a tokens <<< "$spec"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -z "$token" ]] && continue

    if [[ "$token" =~ ^([0-9]{1,5})-([0-9]{1,5})$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( start > end )); then
        echo "Invalid volume range '$token' (start > end)." >&2
        return 1
      fi
      for ((n = start; n <= end; n++)); do
        echo "$n"
      done
    elif [[ "$token" =~ ^[0-9]{1,5}$ ]]; then
      echo "$token"
    else
      echo "Invalid volume token '$token'. Use numbers, ranges, or 'all'." >&2
      return 1
    fi
  done | sort -n -u
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volumes)
      VOLUMES_SPEC="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --workspace-root)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --dataset-tag-prefix)
      DATASET_TAG_PREFIX="$2"
      shift 2
      ;;
    --git-output-root)
      GIT_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --workspace-chunks)
      TRACK_CHUNKS_IN_GIT=0
      shift
      ;;
    --dataset-source-label)
      DATASET_SOURCE_LABEL="$2"
      shift 2
      ;;
    --dataset-source-url)
      DATASET_SOURCE_URL="$2"
      shift 2
      ;;
    --dataset-metadata-file)
      DATASET_METADATA_FILE="$2"
      shift 2
      ;;
    --glob)
      INPUT_GLOB="$2"
      shift 2
      ;;
    --processing-mode)
      PROCESSING_MODE="$2"
      shift 2
      ;;
    --endpoint)
      ENDPOINT="$2"
      ENDPOINT_EXPLICIT=1
      shift 2
      ;;
    --api-format)
      API_FORMAT="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      MODEL_EXPLICIT=1
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --http-referer)
      HTTP_REFERER="$2"
      shift 2
      ;;
    --x-title)
      X_TITLE="$2"
      shift 2
      ;;
    --openrouter-model)
      OPENROUTER_MODEL="$2"
      shift 2
      ;;
    --openrouter-env-file)
      OPENROUTER_ENV_FILE="$2"
      if [[ -f "$OPENROUTER_ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$OPENROUTER_ENV_FILE"
        set +a
      fi
      OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
      OPENROUTER_REFERER="${OPENROUTER_REFERER:-}"
      OPENROUTER_TITLE="${OPENROUTER_TITLE:-Epstein File Ranker}"
      shift 2
      ;;
    --openrouter-api-key)
      OPENROUTER_API_KEY="$2"
      shift 2
      ;;
    --openrouter-referer)
      OPENROUTER_REFERER="$2"
      shift 2
      ;;
    --openrouter-title)
      OPENROUTER_TITLE="$2"
      shift 2
      ;;
    --parallel)
      MAX_PARALLEL_REQUESTS="$2"
      shift 2
      ;;
    --parallel-scheduling)
      PARALLEL_SCHEDULING="$2"
      PARALLEL_SCHEDULING_EXPLICIT=1
      shift 2
      ;;
    --image-prefetch)
      IMAGE_PREFETCH="$2"
      IMAGE_PREFETCH_EXPLICIT=1
      shift 2
      ;;
    --image-max-pages)
      IMAGE_MAX_PAGES="$2"
      shift 2
      ;;
    --pdf-pages-per-image)
      PDF_PAGES_PER_IMAGE="$2"
      shift 2
      ;;
    --pdf-part-pages)
      PDF_PART_PAGES="$2"
      shift 2
      ;;
    --image-render-dpi)
      IMAGE_RENDER_DPI="$2"
      shift 2
      ;;
    --image-detail)
      IMAGE_DETAIL="$2"
      shift 2
      ;;
    --image-output-format)
      IMAGE_OUTPUT_FORMAT="$2"
      shift 2
      ;;
    --image-jpeg-quality)
      IMAGE_JPEG_QUALITY="$2"
      shift 2
      ;;
    --image-max-side)
      IMAGE_MAX_SIDE="$2"
      shift 2
      ;;
    --debug-image-dir)
      DEBUG_IMAGE_DIR="$2"
      shift 2
      ;;
    --max-output-tokens)
      MAX_OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --sleep)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --max-rows)
      MAX_ROWS="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    --skip-missing)
      SKIP_MISSING=1
      shift
      ;;
    --strict-missing)
      SKIP_MISSING=0
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: $DATA_ROOT" >&2
  exit 1
fi

# Use a per-run temp directory and clean it by default to avoid disk buildup.
TMP_ROOT="$WORKSPACE_ROOT/.tmp"
mkdir -p "$TMP_ROOT"
RUN_TMP_DIR="$(mktemp -d "$TMP_ROOT/ranker_tmp.XXXXXX")"
export TMPDIR="$RUN_TMP_DIR"
cleanup_temp() {
  if (( KEEP_TEMP )); then
    echo "Keeping temp dir: $RUN_TMP_DIR"
    return
  fi
  rm -rf "$RUN_TMP_DIR" 2>/dev/null || true
}
trap cleanup_temp EXIT

if [[ "$PROVIDER" == "openrouter" ]]; then
  API_FORMAT="openai"
  if (( ENDPOINT_EXPLICIT == 0 )); then
    ENDPOINT="https://openrouter.ai/api/v1"
  fi
  if (( MODEL_EXPLICIT == 0 )); then
    MODEL="$OPENROUTER_MODEL"
  fi
  if [[ -z "$API_KEY" ]]; then
    API_KEY="$OPENROUTER_API_KEY"
  fi
  if [[ -z "$HTTP_REFERER" ]]; then
    HTTP_REFERER="$OPENROUTER_REFERER"
  fi
  if [[ -z "$X_TITLE" ]]; then
    X_TITLE="$OPENROUTER_TITLE"
  fi
  if [[ -z "$API_KEY" ]] && (( ! DRY_RUN )); then
    echo "OpenRouter provider selected but no API key found." >&2
    echo "Use --openrouter-api-key, --api-key, or set OPENROUTER_API_KEY." >&2
    exit 1
  fi

  # Hosted inference benefits from keeping request slots full while preparing the next jobs.
  if (( PARALLEL_SCHEDULING_EXPLICIT == 0 )); then
    PARALLEL_SCHEDULING="window"
  fi
  if (( IMAGE_PREFETCH_EXPLICIT == 0 )); then
    IMAGE_PREFETCH="$MAX_PARALLEL_REQUESTS"
  fi
fi

VOLUMES=()
while IFS= read -r vol; do
  [[ -n "$vol" ]] && VOLUMES+=("$vol")
done < <(expand_volumes_spec "$VOLUMES_SPEC")
if [[ ${#VOLUMES[@]} -eq 0 ]]; then
  echo "No volumes selected. Check --volumes and available data under $DATA_ROOT." >&2
  exit 1
fi

echo "Selected volumes: ${VOLUMES[*]}"
echo "Data root: $DATA_ROOT"
echo "Workspace root: $WORKSPACE_ROOT"
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Endpoint: $ENDPOINT"

for vol in "${VOLUMES[@]}"; do
  VOL_DIR="$(printf "%s/VOL%05d" "$DATA_ROOT" "$vol")"
  VOL_NAME="$(printf "VOL%05d" "$vol")"
  DATASET_TAG="$(printf "%s_vol%05d" "$DATASET_TAG_PREFIX" "$vol")"
  GIT_CHUNK_DIR="$GIT_OUTPUT_ROOT/$VOL_NAME"
  GIT_CHUNK_MANIFEST="$GIT_CHUNK_DIR/chunks.json"

  if [[ ! -d "$VOL_DIR" ]]; then
    if (( SKIP_MISSING )); then
      echo "[skip] VOL$(printf "%05d" "$vol") missing at $VOL_DIR"
      continue
    fi
    echo "Missing requested volume directory: $VOL_DIR" >&2
    exit 1
  fi

  CMD=(
    "$PYTHON_BIN" gpt_ranker.py
    --input "$VOL_DIR"
    --input-glob "$INPUT_GLOB"
    --processing-mode "$PROCESSING_MODE"
    --dataset-workspace-root "$WORKSPACE_ROOT"
    --dataset-tag "$DATASET_TAG"
    --dataset-source-label "$DATASET_SOURCE_LABEL"
    --dataset-source-url "$DATASET_SOURCE_URL"
    --endpoint "$ENDPOINT"
    --api-format "$API_FORMAT"
    --model "$MODEL"
    --max-parallel-requests "$MAX_PARALLEL_REQUESTS"
    --parallel-scheduling "$PARALLEL_SCHEDULING"
    --image-prefetch "$IMAGE_PREFETCH"
    --image-max-pages "$IMAGE_MAX_PAGES"
    --pdf-pages-per-image "$PDF_PAGES_PER_IMAGE"
    --pdf-part-pages "$PDF_PART_PAGES"
    --image-render-dpi "$IMAGE_RENDER_DPI"
    --image-detail "$IMAGE_DETAIL"
    --image-output-format "$IMAGE_OUTPUT_FORMAT"
    --image-jpeg-quality "$IMAGE_JPEG_QUALITY"
    --image-max-side "$IMAGE_MAX_SIDE"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --temperature "$TEMPERATURE"
    --sleep "$SLEEP_SECONDS"
    --chunk-size "$CHUNK_SIZE"
  )
  if [[ -n "$DEBUG_IMAGE_DIR" ]]; then
    CMD+=(--debug-image-dir "$DEBUG_IMAGE_DIR")
  fi

  if (( TRACK_CHUNKS_IN_GIT )); then
    CMD+=(--chunk-dir "$GIT_CHUNK_DIR" --chunk-manifest "$GIT_CHUNK_MANIFEST")
  fi

  if [[ -f "$DATASET_METADATA_FILE" ]]; then
    CMD+=(--dataset-metadata-file "$DATASET_METADATA_FILE")
  fi
  if [[ -n "$API_KEY" ]]; then
    CMD+=(--api-key "$API_KEY")
  fi
  if [[ -n "$HTTP_REFERER" ]]; then
    CMD+=(--http-referer "$HTTP_REFERER")
  fi
  if [[ -n "$X_TITLE" ]]; then
    CMD+=(--x-title "$X_TITLE")
  fi
  if (( RESUME )); then
    CMD+=(--resume)
  fi
  if [[ -n "$MAX_ROWS" ]]; then
    CMD+=(--max-rows "$MAX_ROWS")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  if (( DRY_RUN )); then
    REDACTED_CMD=()
    SENSITIVE_NEXT=0
    for arg in "${CMD[@]}"; do
      if (( SENSITIVE_NEXT )); then
        REDACTED_CMD+=("***REDACTED***")
        SENSITIVE_NEXT=0
        continue
      fi
      REDACTED_CMD+=("$arg")
      if [[ "$arg" == "--api-key" ]]; then
        SENSITIVE_NEXT=1
      fi
    done
    printf '[dry-run] '
    printf '%q ' "${REDACTED_CMD[@]}"
    printf '\n'
    continue
  fi

  echo "[run] VOL$(printf "%05d" "$vol") -> dataset-tag=$DATASET_TAG"
  "${CMD[@]}"
done

if (( TRACK_CHUNKS_IN_GIT )) && (( ! DRY_RUN )); then
  rebuild_git_manifest
  echo "[done] Updated Git-tracked manifest: $GIT_OUTPUT_ROOT/chunks.json"
fi
