#!/usr/bin/env bash
# ==============================================================================
# F2 Word2Vec (W2V1 / W2V2) Training Status & Throughput Monitor
# Supports: CBOW, Skip-Gram, NNLM, Negative Sampling, Hierarchical Softmax, Phrases
# Multi-Process Aware: Monitors all active training processes sequentially
#
# Usage:
#   tools/w2v_status.sh                 # Current status & throughput for all active jobs
#   tools/w2v_status.sh 3               # Custom sampling duration (e.g. 3 seconds)
#   tools/w2v_status.sh [PID]           # Monitor specific process PID
#   tools/w2v_status.sh -d              # View with per-shard replica details
#   tools/w2v_status.sh -w              # Watch mode (updates every 5 seconds)
#   tools/w2v_status.sh -0              # Report without live sampling
# ==============================================================================

set -euo pipefail

SHOW_DETAIL=0
SAMPLE_SECS=5
TARGET_PID=""
PASSTHROUGH_ARGS=()
IS_WATCH=0

# Parse arguments safely preserving options for watch mode
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--watch)
      IS_WATCH=1
      shift
      ;;
    -d|--detail)
      SHOW_DETAIL=1
      PASSTHROUGH_ARGS+=("-d")
      shift
      ;;
    -s|--sample)
      SAMPLE_SECS="${2:-5}"
      PASSTHROUGH_ARGS+=("-s" "$SAMPLE_SECS")
      shift 2
      ;;
    -0|--no-sample)
      SAMPLE_SECS=0
      PASSTHROUGH_ARGS+=("-0")
      shift
      ;;
    -p|--pid)
      TARGET_PID="$2"
      PASSTHROUGH_ARGS+=("-p" "$2")
      shift 2
      ;;
    [0-9]*)
      if [ -z "$TARGET_PID" ] && [ "$1" -gt 60 ]; then
        TARGET_PID="$1"
        PASSTHROUGH_ARGS+=("$1")
      elif [ -z "$TARGET_PID" ] && [ -d "/proc/$1" ] && (grep -qiE "repro|w2v" "/proc/$1/cmdline" 2>/dev/null); then
        TARGET_PID="$1"
        PASSTHROUGH_ARGS+=("$1")
      else
        SAMPLE_SECS="$1"
        PASSTHROUGH_ARGS+=("$1")
      fi
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$IS_WATCH" -eq 1 ]; then
  exec watch -n 5 "$0" "${PASSTHROUGH_ARGS[@]}"
fi

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git rev-parse --show-toplevel)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# 1. Discover target training process(es)
if [ -n "$TARGET_PID" ]; then
  if [ ! -d "/proc/$TARGET_PID" ]; then
    echo "========================================================================"
    echo " Error: Process PID $TARGET_PID does not exist."
    echo "========================================================================"
    exit 1
  fi
  MONITOR_PIDS=("$TARGET_PID")
else
  # Discover candidate worker processes
  CANDIDATE_PIDS=($(pgrep -f "repro run f2" 2>/dev/null || true))
  MONITOR_PIDS=()
  for p in "${CANDIDATE_PIDS[@]}"; do
    if [ "$p" -ne "$$" ] && [ -d "/proc/$p" ]; then
      comm=$(cat "/proc/$p/comm" 2>/dev/null || true)
      if [ "$comm" = "repro" ] || [[ "$comm" =~ python ]]; then
        MONITOR_PIDS+=("$p")
      fi
    fi
  done

  # Fallback if comm was different but cmdline has repro run f2
  if [ ${#MONITOR_PIDS[@]} -eq 0 ]; then
    for p in "${CANDIDATE_PIDS[@]}"; do
      if [ "$p" -ne "$$" ] && [ -d "/proc/$p" ]; then
        MONITOR_PIDS+=("$p")
      fi
    done
  fi

  if [ ${#MONITOR_PIDS[@]} -eq 0 ]; then
    echo "========================================================================"
    echo " No active F2 W2V training process ('repro run f2') was found."
    echo "========================================================================"
    exit 0
  fi
fi

# Sort PIDs by start time (oldest running process first)
if [ ${#MONITOR_PIDS[@]} -gt 1 ]; then
  sorted_pids=()
  while IFS= read -r pid_entry; do
    [ -n "$pid_entry" ] && sorted_pids+=("$pid_entry")
  done < <(for p in "${MONITOR_PIDS[@]}"; do
    etime=$( (ps -p "$p" -o etimes= 2>/dev/null || echo 0) | tr -d ' ' )
    echo "$etime $p"
  done | sort -nr -k1,1 | awk '{print $2}')
  MONITOR_PIDS=("${sorted_pids[@]}")
fi

# Helper: Collect FD positions for a given process and corpus file
collect_pos_for_pid() {
  local pid="$1"
  local cf="$2"
  local fds="$3"
  local total_pos=0
  local pos_list=""
  for fd in $fds; do
    local pos
    pos=$(awk '$1 == "pos:" {print $2}' "/proc/$pid/fdinfo/$fd" 2>/dev/null || echo 0)
    total_pos=$((total_pos + pos))
    pos_list="$pos_list $fd:$pos"
  done
  echo "$total_pos|$pos_list"
}

# Helper: Resolve corresponding staging directory for a process
resolve_staging_dir() {
  local proc_start_ts="$1"
  local cmd_args="$2"

  local target_dim=""
  if [[ "$cmd_args" =~ -d([0-9]+) ]]; then
    target_dim="${BASH_REMATCH[1]}"
  fi

  local target_suite=""
  if [[ "$cmd_args" =~ w2v2 ]]; then
    target_suite="w2v2"
  elif [[ "$cmd_args" =~ w2v1 ]]; then
    target_suite="w2v1"
  fi

  local best_dir=""
  local best_score=999999999

  for dir in "$REPO_ROOT"/.staging/exp/f2/*/tracked/*; do
    [ -d "$dir" ] || continue

    if [ -n "$target_suite" ] && [[ ! "$dir" =~ /$target_suite/ ]]; then
      continue
    fi

    local d_birth
    d_birth=$(stat -c %W "$dir" 2>/dev/null || echo 0)
    if ! [[ "$d_birth" =~ ^[0-9]+$ ]] || [ "$d_birth" -le 0 ]; then
      d_birth=$(stat -c %Y "$dir" 2>/dev/null || echo 0)
    fi

    local time_diff=$(( d_birth > proc_start_ts ? d_birth - proc_start_ts : proc_start_ts - d_birth ))

    local penalty=0
    if [ -f "$dir/result.json" ]; then
      penalty=100000000
    fi

    if [ -n "$target_dim" ]; then
      local cand_dim
      cand_dim=$(grep -A 5 '"input_embeddings.npy"' "$dir"/checkpoints/generations/*/manifest.json 2>/dev/null | grep -o "[0-9]\+" | tail -n 1 || echo "")
      if [ -n "$cand_dim" ] && [ "$cand_dim" != "$target_dim" ]; then
        penalty=$((penalty + 500000000))
      fi
    fi

    local score=$((time_diff + penalty))
    if [ "$score" -lt "$best_score" ]; then
      best_score="$score"
      best_dir="$dir"
    fi
  done

  if [ "$best_score" -lt 100000000 ] && [ -n "$best_dir" ]; then
    echo "$best_dir"
  fi
}

# Associative storage for per-PID metadata
declare -A P_START_TS P_ETIMES P_CMD P_SUITE P_MODEL P_OBJ P_TOT_EPOCHS
declare -A P_CORPUS_FILE P_CORPUS_SIZE P_FDS P_NUM_SHARDS
declare -A P_RUN_ID P_COMP_EPOCHS P_COMP_TOKENS P_LAST_EPOCH_MTIME P_CURR_EPOCH P_CURR_EPOCH_START P_EPOCH_START_EST P_SLOT
declare -A P_USE_CACHE P_T_START P_POS_START P_T_END P_POS_END P_POS_LIST

NOW_INT=$(date +%s)
T_MEASURE_1=$(date +%s.%N 2>/dev/null || date +%s)
ANY_NEED_SAMPLE=0

# Phase 1: Inspect metadata & collect initial positions
for PID in "${MONITOR_PIDS[@]}"; do
  PROC_ETIMES=$( (ps -p "$PID" -o etimes= 2>/dev/null || echo 0) | tr -d ' ' )
  if [[ "$PROC_ETIMES" =~ ^[0-9]+$ ]] && [ "$PROC_ETIMES" -gt 0 ]; then
    PROC_START_TS=$((NOW_INT - PROC_ETIMES))
  else
    PROC_START_TS=$(stat -c %Y "/proc/$PID" 2>/dev/null || echo "$NOW_INT")
    PROC_ETIMES=$((NOW_INT - PROC_START_TS))
    if [ "$PROC_ETIMES" -lt 0 ]; then PROC_ETIMES=0; fi
  fi
  P_START_TS[$PID]="$PROC_START_TS"
  P_ETIMES[$PID]="$PROC_ETIMES"

  CMD_ARGS=$( (ps -p "$PID" -o args= 2>/dev/null || echo "repro run f2") | sed -e 's/.*repro/repro/' )
  P_CMD[$PID]="$CMD_ARGS"

  SUITE="w2v"
  if echo "$CMD_ARGS" | grep -q "w2v2"; then
    SUITE="w2v2 (Phrases / Negative Sampling)"
  elif echo "$CMD_ARGS" | grep -q "w2v1"; then
    SUITE="w2v1 (Word Relationships)"
  fi
  P_SUITE[$PID]="$SUITE"

  MODEL="Standard"
  if echo "$CMD_ARGS" | grep -qi "cbow"; then
    MODEL="CBOW"
  elif echo "$CMD_ARGS" | grep -qi "skipgram"; then
    MODEL="Skip-Gram"
  elif echo "$CMD_ARGS" | grep -qi "nnlm"; then
    MODEL="Feedforward NNLM"
  fi
  P_MODEL[$PID]="$MODEL"

  OBJECTIVE="Hierarchical Softmax"
  if echo "$CMD_ARGS" | grep -qiE "neg|ns|negative"; then
    OBJECTIVE="Negative Sampling"
  fi
  P_OBJ[$PID]="$OBJECTIVE"

  TOTAL_EPOCHS=1
  if [ -n "$MODEL" ] && [[ "$MODEL" =~ NNLM ]]; then
    TOTAL_EPOCHS=3
  fi

  CORPUS_FILE=$(ls -l "/proc/$PID/fd" 2>/dev/null | awk '{print $NF}' | grep -E '\.(txt|bin)$' | sort | uniq -c | sort -nr | head -n 1 | awk '{print $2}' || true)
  P_USE_CACHE[$PID]=0
  P_CORPUS_FILE[$PID]="$CORPUS_FILE"

  if [ -z "$CORPUS_FILE" ] || [ ! -f "$CORPUS_FILE" ]; then
    P_CORPUS_SIZE[$PID]=0
    P_FDS[$PID]=""
    P_NUM_SHARDS[$PID]=0
    P_RUN_ID[$PID]="unknown"
    P_COMP_EPOCHS[$PID]=0
    P_COMP_TOKENS[$PID]=0
    P_LAST_EPOCH_MTIME[$PID]=0
    P_TOT_EPOCHS[$PID]="$TOTAL_EPOCHS"
    P_CURR_EPOCH[$PID]=1
    P_CURR_EPOCH_START[$PID]="$PROC_START_TS"
    P_EPOCH_START_EST[$PID]=0
    P_SLOT[$PID]="unknown"
    P_T_START[$PID]="$T_MEASURE_1"
    P_POS_START[$PID]=0
    P_T_END[$PID]="$T_MEASURE_1"
    P_POS_END[$PID]=0
    P_POS_LIST[$PID]=""
    continue
  fi

  CORPUS_SIZE=$(stat -c %s "$CORPUS_FILE" 2>/dev/null || stat -f %z "$CORPUS_FILE" 2>/dev/null)
  P_CORPUS_SIZE[$PID]="$CORPUS_SIZE"

  FDS=$(ls -l "/proc/$PID/fd" 2>/dev/null | awk -v cf="$CORPUS_FILE" '$NF == cf {print $9}')
  NUM_SHARDS=$(echo "$FDS" | wc -w)
  P_FDS[$PID]="$FDS"
  P_NUM_SHARDS[$PID]="$NUM_SHARDS"

  # Resolve Staging directory and Checkpoints
  ACTIVE_RUN_ID="unknown"
  COMPLETED_EPOCHS=0
  COMPLETED_TOKENS=0
  LAST_EPOCH_MTIME=0
  VARIANT_SLOT="unknown"

  MATCHED_DIR=$(resolve_staging_dir "$PROC_START_TS" "$CMD_ARGS")
  if [ -n "$MATCHED_DIR" ] && [ -d "$MATCHED_DIR" ]; then
    ACTIVE_RUN_ID=$(basename "$MATCHED_DIR")
    LATEST_JSON="$MATCHED_DIR/checkpoints/latest.json"
    if [ -f "$LATEST_JSON" ]; then
      COMPLETED_EPOCHS=$(awk -F': ' '/"epoch"/ {gsub(/[^0-9]/,"",$2); print $2}' "$LATEST_JSON" 2>/dev/null || echo 0)
      COMPLETED_TOKENS=$(awk -F': ' '/"update"/ {gsub(/[^0-9]/,"",$2); print $2}' "$LATEST_JSON" 2>/dev/null || echo 0)
      LAST_EPOCH_MTIME=$(stat -c %Y "$LATEST_JSON" 2>/dev/null || echo 0)
    fi
  fi

  # Check tmux log for slot and epoch ground truth
  PROC_TTY=$( (ps -p "$PID" -o tty= 2>/dev/null || echo "") | tr -d ' ' )
  if [ -n "$PROC_TTY" ] && [ "$PROC_TTY" != "?" ] && command -v tmux >/dev/null 2>&1; then
    TARGET_PANE=$(tmux list-panes -a -F "#{pane_tty} #{session_name}:#{window_index}.#{pane_index}" 2>/dev/null | awk -v tty="/dev/$PROC_TTY" '$1 == tty {print $2}' | head -n 1 || true)
    if [ -n "$TARGET_PANE" ]; then
      PANE_LOG=$(tmux capture-pane -t "$TARGET_PANE" -p -S -500 2>/dev/null || true)
      SLOT_LINE=$(echo "$PANE_LOG" | grep -E "preparing .* slot=" | tail -n 1 || true)
      if [ -n "$SLOT_LINE" ]; then
        VARIANT_SLOT=$(echo "$SLOT_LINE" | sed -E 's/.*slot=([^ ]+).*/\1/' || echo "unknown")
        PREP_TOTAL=$(echo "$SLOT_LINE" | sed -E 's/.*epochs=([0-9]+).*/\1/' || echo "")
        PREP_COMPLETED=$(echo "$SLOT_LINE" | sed -E 's/.*completed=([0-9]+).*/\1/' || echo 0)
        if [[ "$PREP_TOTAL" =~ ^[0-9]+$ ]] && [ "$PREP_TOTAL" -gt 0 ]; then
          TOTAL_EPOCHS="$PREP_TOTAL"
        fi
        if [[ "$PREP_COMPLETED" =~ ^[0-9]+$ ]]; then
          COMPLETED_EPOCHS="$PREP_COMPLETED"
        fi
      fi
      # Only parse epoch lines that appeared AFTER the current preparing slot boundary
      CURRENT_RUN_LOG=$(echo "$PANE_LOG" | awk '/preparing .* slot=/ {buf=""} {buf = buf "\n" $0} END {print buf}')
      LAST_EPOCH_LINE=$(echo "$CURRENT_RUN_LOG" | grep -E "epoch=[0-9]+/[0-9]+" | tail -n 1 || true)
      if [ -n "$LAST_EPOCH_LINE" ]; then
        LOG_COMPLETED=$(echo "$LAST_EPOCH_LINE" | sed -E 's/.*epoch=([0-9]+)\/([0-9]+).*/\1/' || echo 0)
        LOG_TOTAL=$(echo "$LAST_EPOCH_LINE" | sed -E 's/.*epoch=([0-9]+)\/([0-9]+).*/\2/' || echo 1)
        if [[ "$LOG_COMPLETED" =~ ^[0-9]+$ ]] && [ "$LOG_COMPLETED" -gt "$COMPLETED_EPOCHS" ]; then
          COMPLETED_EPOCHS="$LOG_COMPLETED"
        fi
        if [[ "$LOG_TOTAL" =~ ^[0-9]+$ ]] && [ "$LOG_TOTAL" -gt 0 ]; then
          TOTAL_EPOCHS="$LOG_TOTAL"
        fi
      fi
    fi
  fi

  if ! [[ "$COMPLETED_EPOCHS" =~ ^[0-9]+$ ]]; then COMPLETED_EPOCHS=0; fi
  if ! [[ "$COMPLETED_TOKENS" =~ ^[0-9]+$ ]]; then COMPLETED_TOKENS=0; fi
  if ! [[ "$TOTAL_EPOCHS" =~ ^[0-9]+$ ]] || [ "$TOTAL_EPOCHS" -le 0 ]; then TOTAL_EPOCHS=1; fi

  CURRENT_EPOCH=$((COMPLETED_EPOCHS + 1))
  if [ "$CURRENT_EPOCH" -gt "$TOTAL_EPOCHS" ]; then
    CURRENT_EPOCH="$TOTAL_EPOCHS"
  fi

  if [ "$COMPLETED_EPOCHS" -gt 0 ] && [ "$LAST_EPOCH_MTIME" -gt 0 ]; then
    CURRENT_EPOCH_START="$LAST_EPOCH_MTIME"
    EPOCH_START_IS_ESTIMATED=1
  else
    CURRENT_EPOCH_START="$PROC_START_TS"
    EPOCH_START_IS_ESTIMATED=0
  fi

  P_RUN_ID[$PID]="$ACTIVE_RUN_ID"
  P_COMP_EPOCHS[$PID]="$COMPLETED_EPOCHS"
  P_COMP_TOKENS[$PID]="$COMPLETED_TOKENS"
  P_LAST_EPOCH_MTIME[$PID]="$LAST_EPOCH_MTIME"
  P_TOT_EPOCHS[$PID]="$TOTAL_EPOCHS"
  P_CURR_EPOCH[$PID]="$CURRENT_EPOCH"
  P_CURR_EPOCH_START[$PID]="$CURRENT_EPOCH_START"
  P_EPOCH_START_EST[$PID]="$EPOCH_START_IS_ESTIMATED"
  P_SLOT[$PID]="$VARIANT_SLOT"

  # Measure initial position
  RES_1=$(collect_pos_for_pid "$PID" "$CORPUS_FILE" "$FDS")
  TOTAL_POS_1="${RES_1%%|*}"
  POS_LIST_1="${RES_1#*|}"

  # Cache inspection
  CACHE_FILE="/tmp/.w2v_status_${UID}_${PID}.cache"
  USE_CACHE=0
  CACHE_TIME=0
  CACHE_POS=0

  if [ -f "$CACHE_FILE" ]; then
    read -r C_START C_CORPUS C_EPOCH C_TIME C_POS < "$CACHE_FILE" 2>/dev/null || true
    if [ -n "${C_TIME:-}" ] && [ -n "${C_POS:-}" ]; then
      if [ "${C_START:-0}" = "$PROC_START_TS" ] && [ "${C_CORPUS:-}" = "$CORPUS_FILE" ] && [ "${C_EPOCH:-0}" = "$COMPLETED_EPOCHS" ]; then
        DIFF_TIME=$(awk -v t1="$C_TIME" -v t2="$T_MEASURE_1" 'BEGIN {print (t2 - t1)}' 2>/dev/null || echo 0)
        if awk -v dt="$DIFF_TIME" -v cp="$C_POS" -v p1="$TOTAL_POS_1" 'BEGIN {exit !(dt >= 1.0 && dt <= 30.0 && p1 >= cp)}' 2>/dev/null; then
          USE_CACHE=1
          CACHE_TIME="$C_TIME"
          CACHE_POS="$C_POS"
        fi
      fi
    fi
  fi

  P_USE_CACHE[$PID]="$USE_CACHE"
  if [ "$USE_CACHE" -eq 1 ]; then
    P_T_START[$PID]="$CACHE_TIME"
    P_POS_START[$PID]="$CACHE_POS"
    P_T_END[$PID]="$T_MEASURE_1"
    P_POS_END[$PID]="$TOTAL_POS_1"
    P_POS_LIST[$PID]="$POS_LIST_1"
  else
    ANY_NEED_SAMPLE=1
    P_T_START[$PID]="$T_MEASURE_1"
    P_POS_START[$PID]="$TOTAL_POS_1"
    P_POS_LIST[$PID]="$POS_LIST_1"
  fi
done

# Phase 2: Live sampling if needed
if [ "$SAMPLE_SECS" -gt 0 ] && [ "$ANY_NEED_SAMPLE" -eq 1 ]; then
  if [ -t 1 ]; then
    count=${#MONITOR_PIDS[@]}
    if [ "$count" -gt 1 ]; then
      msg="Sampling real-time throughput for $count processes (${SAMPLE_SECS}s)"
    else
      msg="Sampling real-time throughput (${SAMPLE_SECS}s)"
    fi
    for ((i=1; i<=SAMPLE_SECS; i++)); do
      printf " \033[36m*\033[0m %s... [%d/%d]\r" "$msg" "$i" "$SAMPLE_SECS"
      sleep 1
    done
    printf "\r\033[K"
  else
    sleep "$SAMPLE_SECS"
  fi

  T_MEASURE_2=$(date +%s.%N 2>/dev/null || date +%s)
  for PID in "${MONITOR_PIDS[@]}"; do
    if [ "${P_USE_CACHE[$PID]}" -eq 0 ] && [ -n "${P_CORPUS_FILE[$PID]}" ]; then
      RES_2=$(collect_pos_for_pid "$PID" "${P_CORPUS_FILE[$PID]}" "${P_FDS[$PID]}")
      TOTAL_POS_2="${RES_2%%|*}"
      POS_LIST_2="${RES_2#*|}"

      P_T_END[$PID]="$T_MEASURE_2"
      P_POS_END[$PID]="$TOTAL_POS_2"
      P_POS_LIST[$PID]="$POS_LIST_2"

      # Save state to cache
      CACHE_FILE="/tmp/.w2v_status_${UID}_${PID}.cache"
      echo "${P_START_TS[$PID]} ${P_CORPUS_FILE[$PID]} ${P_COMP_EPOCHS[$PID]} $T_MEASURE_2 $TOTAL_POS_2" > "$CACHE_FILE" 2>/dev/null || true
    fi
  done
else
  # No sampling or all cached
  for PID in "${MONITOR_PIDS[@]}"; do
    if [ "${P_USE_CACHE[$PID]}" -eq 0 ] && [ -n "${P_CORPUS_FILE[$PID]}" ]; then
      P_T_END[$PID]="${P_T_START[$PID]}"
      P_POS_END[$PID]="${P_POS_START[$PID]}"
    fi
  done
fi

# Phase 3: Sequential Reporting
PROC_TOTAL=${#MONITOR_PIDS[@]}
PROC_IDX=1

for PID in "${MONITOR_PIDS[@]}"; do
  CORPUS_FILE="${P_CORPUS_FILE[$PID]}"
  if [ -z "$CORPUS_FILE" ] || [ ! -f "$CORPUS_FILE" ]; then
    printf "========================================================================\n"
    if [ "$PROC_TOTAL" -gt 1 ]; then
      printf " [Process %d/%d] F2 Word2Vec Training Monitor (PID: %d)\n" "$PROC_IDX" "$PROC_TOTAL" "$PID"
    else
      printf " F2 Word2Vec Training Monitor (PID: %d)\n" "$PID"
    fi
    printf "========================================================================\n"
    printf " Command:             %s\n" "${P_CMD[$PID]}"
    printf " PID:                 %d\n" "$PID"
    printf " Process Elapsed:     %.1f hours (%d seconds)\n" "$(( ${P_ETIMES[$PID]} / 3600 ))" "${P_ETIMES[$PID]}"
    printf " Status:              No active training corpus FD found.\n"
    printf "                      (Process may be preparing datasets, phrases, or evaluating)\n"
    printf "========================================================================\n"
  else
    awk -v pid="$PID" \
        -v cmd_args="${P_CMD[$PID]}" \
        -v suite="${P_SUITE[$PID]}" \
        -v model="${P_MODEL[$PID]}" \
        -v objective="${P_OBJ[$PID]}" \
        -v num_shards="${P_NUM_SHARDS[$PID]}" \
        -v corpus_file="${P_CORPUS_FILE[$PID]}" \
        -v corpus_size="${P_CORPUS_SIZE[$PID]}" \
        -v proc_etimes="${P_ETIMES[$PID]}" \
        -v proc_start_ts="${P_START_TS[$PID]}" \
        -v current_epoch_start="${P_CURR_EPOCH_START[$PID]}" \
        -v epoch_start_est="${P_EPOCH_START_EST[$PID]}" \
        -v current_epoch="${P_CURR_EPOCH[$PID]}" \
        -v completed_epochs="${P_COMP_EPOCHS[$PID]}" \
        -v total_epochs="${P_TOT_EPOCHS[$PID]}" \
        -v completed_tokens="${P_COMP_TOKENS[$PID]}" \
        -v run_id="${P_RUN_ID[$PID]}" \
        -v slot_id="${P_SLOT[$PID]}" \
        -v total_pos="${P_POS_END[$PID]}" \
        -v pos_start="${P_POS_START[$PID]}" \
        -v t_start="${P_T_START[$PID]}" \
        -v t_end="${P_T_END[$PID]}" \
        -v pos_list="${P_POS_LIST[$PID]}" \
        -v show_detail="$SHOW_DETAIL" \
        -v proc_idx="$PROC_IDX" \
        -v proc_total="$PROC_TOTAL" '
    BEGIN {
      now = systime()
      training_sec = proc_etimes
      if (training_sec <= 0) training_sec = 1

      total_elapsed_sec = now - proc_start_ts
      if (total_elapsed_sec <= 0) total_elapsed_sec = 1

      epoch_elapsed_sec = now - current_epoch_start
      if (epoch_elapsed_sec <= 0) epoch_elapsed_sec = 1

      start_base = (num_shards - 1) / 2.0 * corpus_size
      done_bytes = total_pos - start_base
      if (done_bytes < 0) done_bytes = 0
      if (done_bytes > corpus_size) done_bytes = corpus_size

      remain_bytes = corpus_size - done_bytes
      pct = (done_bytes / corpus_size) * 100.0

      bytes_per_token = 5.353556
      epoch_done_tokens_est = done_bytes / bytes_per_token
      epoch_total_tokens_est = corpus_size / bytes_per_token

      all_total_bytes = total_epochs * corpus_size
      all_done_bytes = (completed_epochs * corpus_size) + done_bytes
      all_pct = (all_done_bytes / all_total_bytes) * 100.0
      all_remain_bytes = all_total_bytes - all_done_bytes
      if (all_remain_bytes < 0) all_remain_bytes = 0
      all_done_tokens = completed_tokens + epoch_done_tokens_est

      # Instantaneous Speed
      delta_time = t_end - t_start
      delta_bytes = total_pos - pos_start
      has_instant = (delta_time >= 0.5 && delta_bytes >= 0)

      if (has_instant && delta_time > 0) {
        instant_speed_bytes = delta_bytes / delta_time
        instant_speed_tokens = (delta_bytes / bytes_per_token) / delta_time
      } else {
        instant_speed_bytes = 0
        instant_speed_tokens = 0
      }

      # Cumulative Speeds
      epoch_speed_bytes = (epoch_elapsed_sec > 0) ? (done_bytes / epoch_elapsed_sec) : 0
      epoch_speed_tokens = (epoch_elapsed_sec > 0) ? (epoch_done_tokens_est / epoch_elapsed_sec) : 0

      overall_speed_bytes = (total_elapsed_sec > 0) ? (all_done_bytes / total_elapsed_sec) : 0
      overall_speed_tokens = (total_elapsed_sec > 0) ? (all_done_tokens / total_elapsed_sec) : 0

      # Primary effective speed for ETA (prefer instant if running, fallback to epoch avg)
      if (has_instant && instant_speed_bytes > 0) {
        effective_speed_bytes = instant_speed_bytes
        effective_speed_tokens = instant_speed_tokens
        speed_mode = sprintf("last %.1fs instant", delta_time)
      } else if (epoch_speed_bytes > 0) {
        effective_speed_bytes = epoch_speed_bytes
        effective_speed_tokens = epoch_speed_tokens
        speed_mode = "current epoch avg"
      } else {
        effective_speed_bytes = 0
        effective_speed_tokens = 0
        speed_mode = "paused/unknown"
      }

      printf "========================================================================\n"
      if (proc_total > 1) {
        printf " [Process %d/%d] F2 Word2Vec Training Monitor (PID: %d)\n", proc_idx, proc_total, pid
      } else {
        printf " F2 Word2Vec Training Monitor\n"
      }
      printf "========================================================================\n"
      printf " Suite / Model:       %s | %s (%s)\n", suite, model, objective
      printf " Command:             %s\n", cmd_args
      if (slot_id != "unknown") {
        printf " Variant Slot:        %s\n", slot_id
      }
      if (run_id != "unknown") {
        printf " Run ID (MLflow):     %s\n", run_id
      }
      printf " PID:                 %d (Workers: %d parallel threads)\n", pid, num_shards
      printf " Process Elapsed:     %.1f hours (%d seconds)\n", training_sec / 3600.0, training_sec
      printf " Corpus File:         %s\n", corpus_file
      printf " Corpus Size:         %.2f GB (%d bytes)\n", corpus_size / (1024^3), corpus_size
      printf "------------------------------------------------------------------------\n"
      printf " [Progress: Epoch %d / %d]\n", current_epoch, total_epochs
      printf " Current Epoch FD:    %.2f GB / %.2f GB (%.2f%%) [estimated]\n", done_bytes / (1024^3), corpus_size / (1024^3), pct
      printf " Current Epoch Est:   ~%.1fM / ~%.1fM tokens [estimated]\n", epoch_done_tokens_est / 1e6, epoch_total_tokens_est / 1e6
      if (completed_epochs > 0) {
        printf " Completed Tokens:    %.1fM tokens (Ground Truth from checkpoint)\n", completed_tokens / 1e6
        printf " Total Run Progress:  %.2f GB / %.2f GB (%.2f%% across %d epochs) [estimated]\n", all_done_bytes / (1024^3), all_total_bytes / (1024^3), all_pct, total_epochs
      }
      printf "------------------------------------------------------------------------\n"
      printf " [Throughput]\n"
      if (has_instant && instant_speed_tokens > 0) {
        printf " Current Speed:       %.1f tokens/sec (last %.1fs instant, ~%.1f KB/s)\n", instant_speed_tokens, delta_time, instant_speed_bytes / 1024.0
      } else {
        printf " Current Speed:       0.0 tokens/sec (Paused / Sync / Checkpointing)\n"
      }

      if (completed_epochs > 0) {
        printf " Current Epoch Avg:   %.1f tokens/sec (~%.1f KB/s over %.1fh) [estimated]\n", epoch_speed_tokens, epoch_speed_bytes / 1024.0, epoch_elapsed_sec / 3600.0
        printf " Total Run Avg:       %.1f tokens/sec (across %d epochs over %.1fh)\n", overall_speed_tokens, completed_epochs + 1, total_elapsed_sec / 3600.0
      } else {
        printf " Epoch 1 Avg Speed:   %.1f tokens/sec (~%.1f KB/s over %.1fh)\n", epoch_speed_tokens, epoch_speed_bytes / 1024.0, epoch_elapsed_sec / 3600.0
      }

      printf "------------------------------------------------------------------------\n"
      printf " [Estimated Completion Time (ETA)]\n"

      if (effective_speed_bytes > 0) {
        epoch_remain_sec = remain_bytes / effective_speed_bytes
        full_remain_sec = all_remain_bytes / effective_speed_bytes

        curr_ep = completed_epochs + 1
        if (curr_ep > total_epochs) curr_ep = total_epochs

        cmd_ep = sprintf("date -d @%.0f \"+%%Y-%%m-%%d %%H:%%M:%%S\"", now + epoch_remain_sec)
        cmd_ep | getline eta_ep; close(cmd_ep)
        printf " Current Epoch (%d/%d): in %.2f hours (~%d mins)   -> %s [%s]\n", curr_ep, total_epochs, epoch_remain_sec / 3600.0, epoch_remain_sec / 60, eta_ep, speed_mode

        if (total_epochs > 1) {
          cmd_full = sprintf("date -d @%.0f \"+%%Y-%%m-%%d %%H:%%M:%%S\"", now + full_remain_sec)
          cmd_full | getline eta_full; close(cmd_full)
          printf " Full Run (%d epochs): in %.2f hours (~%.1f days)   -> %s [%s]\n", total_epochs, full_remain_sec / 3600.0, full_remain_sec / 86400.0, eta_full, speed_mode
        }
      } else {
        printf " Current Epoch Finish: [Paused / Speed 0 - Unable to calculate ETA]\n"
        if (total_epochs > 1) {
          printf " Full Run Finish:      [Paused / Speed 0 - Unable to calculate ETA]\n"
        }
      }

      if (show_detail == 1) {
        printf "------------------------------------------------------------------------\n"
        printf " [Shard Details (%d Replicas - FD Position Observation)]\n", num_shards
        printf " %-6s | %-5s | %-12s | %-12s | %-6s\n", "Shard", "FD", "Start Byte", "Current Pos", "Done %"
        printf "-------+-------+--------------+--------------+--------\n"
        n = split(pos_list, pairs, " ")
        shard_len = corpus_size / num_shards
        for (i = 1; i <= n; i++) {
          split(pairs[i], kv, ":")
          fd_num = kv[1]
          cur_pos = kv[2]
          shard_idx = i - 1
          s_start = int(shard_idx * shard_len)
          s_done = cur_pos - s_start
          s_pct = (s_done / shard_len) * 100.0
          if (s_pct < 0) s_pct = 0
          if (s_pct > 100) s_pct = 100
          printf " %5d | %5s | %12d | %12d | %5.1f%%\n", shard_idx, fd_num, s_start, cur_pos, s_pct
        }
      }
      printf "========================================================================\n"
    }
    '
  fi

  if [ "$PROC_IDX" -lt "$PROC_TOTAL" ]; then
    echo ""
  fi
  PROC_IDX=$((PROC_IDX + 1))
done
