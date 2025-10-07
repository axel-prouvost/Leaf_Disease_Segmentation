#!/usr/bin/env sh

set -euo pipefail

# Resolve script and repo root directories
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# ============================================================
# User configuration (edit these defaults to your environment)
# ============================================================
# Path to annotations JSON file
JSON_PATH="$REPO_ROOT/01_labels/annotations_49.ordered.json"
# Path to folder containing .h5 prediction files
H5_FOLDER="$REPO_ROOT/LAB/results/10/h5_predictions"
# Output directory for evaluation results
OUTPUT_DIR="$REPO_ROOT/LAB/results/10/prediction_analysis"
# Minimum cluster size to keep (0 = no filtering)
MIN_CLUSTER_SIZE="0"
# Toggle features
NO_CHART="false"      # true/false
NO_SUMMARY="false"    # true/false
SKIP_GRAPH="false"    # true/false
# Skip heavy pixelwise CSV generation
NO_PIXELWISE="true"   # true/false
# Explicit CSV for graph step (leave empty to auto-detect in OUTPUT_DIR)
CSV_PATH=""

print_usage() {
  cat <<EOF
Usage: ./02_scripts/run_evaluation.sh [--json <annotations.json>] [--h5-folder <path_to_h5_folder>] [options]

If not provided via flags, the script uses the values configured at the top of this file.

Options:
  --output DIR             Output directory for evaluation results
  --min-cluster-size N     Minimum cluster size to keep (default: 0 = no filtering)
  --no-chart               Disable chart generation during evaluation
  --no-summary             Disable batch summary generation
  --skip-graph             Skip running graph/regression step
  --no-pixelwise           Skip generating pixelwise_predictions.csv (lighter, recommended)
  --csv PATH               Explicit CSV path for graph step (overrides auto-detect)
  -h, --help               Show this help and exit

This script runs two steps:
  1) Evaluate: $SCRIPT_DIR/evaluate_folder_with_json.py
  2) Graph:    $SCRIPT_DIR/graph_from_evaluation.py (auto-picks CSV from the evaluation output)
EOF
}

# Parse args (overrides configuration above)
while [ $# -gt 0 ]; do
  case "$1" in
    --json)
      JSON_PATH=${2:-}
      shift 2
      ;;
    --h5-folder)
      H5_FOLDER=${2:-}
      shift 2
      ;;
    --output)
      OUTPUT_DIR=${2:-}
      shift 2
      ;;
    --min-cluster-size)
      MIN_CLUSTER_SIZE=${2:-}
      shift 2
      ;;
    --no-chart)
      NO_CHART="true"
      shift 1
      ;;
    --no-summary)
      NO_SUMMARY="true"
      shift 1
      ;;
    --skip-graph)
      SKIP_GRAPH="true"
      shift 1
      ;;
    --no-pixelwise)
      NO_PIXELWISE="true"
      shift 1
      ;;
    --csv)
      CSV_PATH=${2:-}
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

# Validate required paths (from config block or CLI)
if [ -z "$JSON_PATH" ] || [ -z "$H5_FOLDER" ]; then
  echo "Error: JSON_PATH and H5_FOLDER are not set." >&2
  echo "Edit the configuration at the top of this file or pass --json / --h5-folder." >&2
  print_usage
  exit 1
fi

if [ ! -f "$JSON_PATH" ]; then
  echo "Error: JSON file not found: $JSON_PATH" >&2
  exit 1
fi

if [ ! -d "$H5_FOLDER" ]; then
  echo "Error: H5 folder not found: $H5_FOLDER" >&2
  exit 1
fi

# Choose Python interpreter (prefer repo venv)
if [ -x "$REPO_ROOT/venv/Scripts/python.exe" ]; then
  PY="$REPO_ROOT/venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
elif command -v py >/dev/null 2>&1; then
  PY=py
else
  echo "Error: No Python interpreter found. Ensure Python is installed or create the venv." >&2
  exit 1
fi

EVAL_SCRIPT="$SCRIPT_DIR/evaluate_folder_with_json.py"
GRAPH_SCRIPT="$SCRIPT_DIR/graph_from_evaluation.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "Error: Evaluation script not found: $EVAL_SCRIPT" >&2
  exit 1
fi

if [ ! -f "$GRAPH_SCRIPT" ] && [ "$SKIP_GRAPH" != "true" ]; then
  echo "Error: Graph script not found: $GRAPH_SCRIPT" >&2
  exit 1
fi

echo "==> Running evaluation"

# If using Windows python.exe under Git Bash, convert MSYS paths to Windows
IS_WIN_PY=0
case "$PY" in
  *.exe) IS_WIN_PY=1 ;;
esac

JSON_ARG="$JSON_PATH"
H5_ARG="$H5_FOLDER"
OUT_ARG="$OUTPUT_DIR"
EVAL_SCRIPT_ARG="$EVAL_SCRIPT"
GRAPH_SCRIPT_ARG="$GRAPH_SCRIPT"
if [ $IS_WIN_PY -eq 1 ] && command -v cygpath >/dev/null 2>&1; then
  JSON_ARG=$(cygpath -w "$JSON_PATH")
  H5_ARG=$(cygpath -w "$H5_FOLDER")
  OUT_ARG=$(cygpath -w "$OUTPUT_DIR")
  EVAL_SCRIPT_ARG=$(cygpath -w "$EVAL_SCRIPT")
  GRAPH_SCRIPT_ARG=$(cygpath -w "$GRAPH_SCRIPT")
fi

"$PY" "$EVAL_SCRIPT_ARG" "$JSON_ARG" "$H5_ARG" --output "$OUT_ARG" --min-cluster-size "$MIN_CLUSTER_SIZE" $( [ "$NO_CHART" = "true" ] && printf %s "--no-chart" ) $( [ "$NO_SUMMARY" = "true" ] && printf %s "--no-summary" ) $( [ "$NO_PIXELWISE" = "true" ] && printf %s "--no-pixelwise" )

if [ "$SKIP_GRAPH" = "true" ]; then
  echo "==> Skipping graph step as requested"
  exit 0
fi

# Determine CSV for graph step (test using POSIX path if needed)
DETECT_DIR="$OUTPUT_DIR"
if [ $IS_WIN_PY -eq 1 ] && command -v cygpath >/dev/null 2>&1; then
  # Ensure shell checks use POSIX path
  DETECT_DIR=$(cygpath -u "$OUT_ARG")
fi

if [ -z "$CSV_PATH" ]; then
  if [ -f "$DETECT_DIR/per_leaf_pixel_counts.csv" ]; then
    CSV_PATH="$DETECT_DIR/per_leaf_pixel_counts.csv"
  elif [ -f "$DETECT_DIR/pixelwise_predictions.csv" ]; then
    CSV_PATH="$DETECT_DIR/pixelwise_predictions.csv"
  else
    echo "Error: Could not find CSV output in '$DETECT_DIR'." >&2
    echo "       Expected one of: per_leaf_pixel_counts.csv or pixelwise_predictions.csv" >&2
    echo "       You can also pass --csv <path> to specify it explicitly." >&2
    exit 1
  fi
fi

# Clean up legacy duplicate graph from older runs
if [ -f "$DETECT_DIR/surface_regression_analysis_non_uniform.png" ]; then
  rm -f "$DETECT_DIR/surface_regression_analysis_non_uniform.png" || true
fi

# For Windows python, convert CSV path to Windows form
CSV_ARG="$CSV_PATH"
if [ $IS_WIN_PY -eq 1 ] && command -v cygpath >/dev/null 2>&1; then
  CSV_ARG=$(cygpath -w "$CSV_PATH")
fi

echo "==> Running graph/regression on: $CSV_PATH"

# For Windows python, convert OUT dir to Windows path too
OUT_DIR_ARG="$OUT_ARG"
if [ $IS_WIN_PY -eq 1 ] && command -v cygpath >/dev/null 2>&1; then
  OUT_DIR_ARG=$(cygpath -w "$DETECT_DIR")
fi

"$PY" "$GRAPH_SCRIPT_ARG" --csv "$CSV_ARG" --out-dir "$OUT_DIR_ARG"

echo "==> Done"


