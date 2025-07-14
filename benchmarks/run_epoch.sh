#!/usr/bin/env bash
set -euo pipefail

# ---------- config (edit here on version bumps) ----------
PY="benchmarks/orchard_bench_v0.8.py"      # bump when you rename the Python harness
VENV="$HOME/projects/orchard/orchard-env"  # absolute path to virtual‑env
DATA_DEFAULT="data/wikitext-2/wiki.train.tokens"
RUN_DIR="runs"                              # committed artefacts live here
PM_OPTS="--samplers cpu_power,gpu_power,thermal -i 1000"
# --------------------------------------------------------

# 1 activate deterministic Python environment
if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
else
  echo "run_epoch.sh: virtual‑env not found at $VENV" >&2
  exit 3
fi

mkdir -p "$RUN_DIR"

# 2 parse CLI — forward all args to Python but trap --tag / --data / --config for filenames

tag=""                     # required
data_path="$DATA_DEFAULT"    # optional override
py_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="$2"; py_args+=("$1" "$2"); shift 2 ;;
    --tag=*)
      tag="${1#*=}"; py_args+=("$1"); shift ;;
    --data)
      data_path="$2"; py_args+=("$1" "$2"); shift 2 ;;
    --data=*)
      data_path="${1#*=}"; py_args+=("$1"); shift ;;
    --config|--config=*)
      # pass through to Python so YAML override works
      if [[ "$1" == --config ]]; then
        py_args+=("$1" "$2"); shift 2
      else
        py_args+=("$1"); shift
      fi ;;
    *)
      py_args+=("$1"); shift ;;
  esac
done

# ------ quick-check preset --------------------------------------------------
if [[ "${QUICK_CHECK:-0}" == "1" ]]; then
  py_args+=( --bs 4 --seq 256 --steps 40 )
  export PROGRESS_EVERY=5          # more frequent ticker
  tag="${tag}_qc"
  echo ">> QUICK-CHECK mode: bs=4 seq=256 steps=40 (≈15-min run)" >&2
fi
# ---------------------------------------------------------------------------


if [[ -z "$tag" ]]; then
  echo "run_epoch.sh: --tag is required" >&2
  exit 2
fi

# 3 construct run-scoped filenames
_ts=$(date +%Y%m%d-%H%M%S)
LOG_FILE_ALL="$RUN_DIR/${_ts}_${tag}_all.log"
LOG_FILE_ERRORS="$RUN_DIR/${_ts}_${tag}_errors.log"
LOG_FILE_RUN="$RUN_DIR/${_ts}_${tag}_run.log"
POW_FILE="$RUN_DIR/${_ts}_${tag}.power"

# 4 launch powermetrics (requires sudo or NOPASSWD entry)
POW_FILE=""  # No power logging by default on PC
if [[ "$(uname)" == "Darwin" ]]; then
  sudo powermetrics $PM_OPTS -o "$POW_FILE" &
  PM_PID=$!
  trap 'kill $PM_PID 2>/dev/null || true' EXIT INT TERM
  export POW_FILE
fi

export POW_FILE  # consumed by orchard_bench_v0.8.py

# 5 run benchmark, split output for logging and real-time display
# Launch Python, split all output:
# - All output to $LOG_FILE_ALL
# - Only error lines to $LOG_FILE_ERRORS
# - Only progress ("step" lines) to $LOG_FILE_RUN
# - Errors and progress lines also echoed to terminal

python3 "$PY" --data "$data_path" --tag "$tag" "${py_args[@]}" 2>&1 \
| tee "$LOG_FILE_ALL" \
| awk -v runfile="$LOG_FILE_RUN" -v errfile="$LOG_FILE_ERRORS" -v flush=10 '
  function now() { return systime(); }
  BEGIN { lastflush = now(); }
  {
    # Save every line to correct log file if it matches
    if ($0 ~ /^\[orchard\]\[.*\] step/) {
      print $0 | "tee -a \"" runfile "\"";
      print $0;      # Print to terminal
    }
    if ($0 ~ /ERROR|Traceback|RuntimeError/) {
      print $0 | "tee -a \"" errfile "\"";
      print $0;      # Print to terminal
    }
    # Periodically flush log files to disk
    if (now() - lastflush > flush) {
      close(runfile);
      close(errfile);
      lastflush = now();
    }
  }
'
exit_code=${PIPESTATUS[0]}
