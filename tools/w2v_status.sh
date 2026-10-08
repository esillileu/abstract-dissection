#!/usr/bin/env bash
# ==============================================================================
# F2 Word2Vec (W2V1 / W2V2) Training Status & Throughput Monitor
# Supports: CBOW, Skip-Gram, NNLM, Negative Sampling, Hierarchical Softmax, Phrases
#
# Usage:
#   tools/w2v_status.sh                 # Current status summary & real-time throughput
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
    [0-9]*)
      if [ -z "$TARGET_PID" ] && [ -d "/proc/$1" ]; then
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

# 1. Locate and validate target training process
if [ -n "$TARGET_PID" ]; then
  PID="$TARGET_PID"
  if [ ! -d "/proc/$PID" ]; then
    echo "========================================================================"
    echo " Error: Process PID $PID does not exist."
    echo "========================================================================"
    exit 1
  fi
else
  PIDS=($(pgrep -f "python.*repro run f2" 2>/dev/null || true))
  if [ ${#PIDS[@]} -eq 0 ]; then
    PIDS=($(pgrep -f "repro run f2" 2>/dev/null || true))
  fi

  if [ ${#PIDS[@]} -eq 0 ]; then
    echo "========================================================================"
    echo " No active F2 W2V training process ('repro run f2') was found."
    echo "========================================================================"
    exit 0
  fi
  PID="${PIDS[0]}"
fi

# 2. Process Lifetime & Start Timestamp
PROC_ETIMES=$( (ps -p "$PID" -o etimes= 2>/dev/null || echo 0) | tr -d ' ' )
NOW_INT=$(date +%s)
if [[ "$PROC_ETIMES" =~ ^[0-9]+$ ]] && [ "$PROC_ETIMES" -gt 0 ]; then
  PROC_START_TS=$((NOW_INT - PROC_ETIMES))
else
  PROC_START_TS=$(stat -c %Y "/proc/$PID" 2>/dev/null || echo "$NOW_INT")
  PROC_ETIMES=$((NOW_INT - PROC_START_TS))
  if [ "$PROC_ETIMES" -lt 0 ]; then PROC_ETIMES=0; fi
fi

# 3. Locate active corpus file
CORPUS_FILE=$(ls -l "/proc/$PID/fd" 2>/dev/null | awk '{print $NF}' | grep -E '\.(txt|bin)$' | sort | uniq -c | sort -nr | head -n 1 | awk '{print $2}' || true)

if [ -z "$CORPUS_FILE" ] || [ ! -f "$CORPUS_FILE" ]; then
  echo "========================================================================"
  echo " Process PID $PID is running, but no active training corpus FD found."
  echo " (The process may be preparing datasets, phrases, or evaluating)."
  echo "========================================================================"
  exit 0
fi

CORPUS_SIZE=$(stat -c %s "$CORPUS_FILE" 2>/dev/null || stat -f %z "$CORPUS_FILE" 2>/dev/null)

# 4. Collect worker file descriptors referencing this corpus
FDS=$(ls -l "/proc/$PID/fd" 2>/dev/null | awk -v cf="$CORPUS_FILE" '$NF == cf {print $9}')
NUM_SHARDS=$(echo "$FDS" | wc -w)

if [ "$NUM_SHARDS" -eq 0 ]; then
  echo "========================================================================"
  echo " Process PID $PID has no active open descriptors for $CORPUS_FILE."
  echo "========================================================================"
  exit 0
fi

collect_pos_entries() {
  for fd in $FDS; do
    pos=$(awk '$1 == "pos:" {print $2}' "/proc/$PID/fdinfo/$fd" 2>/dev/null || echo 0)
    echo "$pos $fd"
  done | sort -k1,1n
}

# 5. Extract command line arguments and detect suite / model characteristics
CMD_ARGS=$( (ps -p "$PID" -o args= 2>/dev/null || echo "repro run f2") | sed -e 's/.*repro/repro/' )

SUITE="w2v"
if echo "$CMD_ARGS" | grep -q "w2v2"; then
  SUITE="w2v2 (Phrases / Negative Sampling)"
elif echo "$CMD_ARGS" | grep -q "w2v1"; then
  SUITE="w2v1 (Word Relationships)"
fi

MODEL="Standard"
if echo "$CMD_ARGS" | grep -qi "cbow"; then
  MODEL="CBOW"
elif echo "$CMD_ARGS" | grep -qi "skipgram"; then
  MODEL="Skip-Gram"
elif echo "$CMD_ARGS" | grep -qi "nnlm"; then
  MODEL="Feedforward NNLM"
fi

OBJECTIVE="Hierarchical Softmax"
if echo "$CMD_ARGS" | grep -qiE "neg|ns|negative"; then
  OBJECTIVE="Negative Sampling"
fi

TOTAL_EPOCHS=1
if [ -n "$MODEL" ] && [[ "$MODEL" =~ NNLM ]]; then
  TOTAL_EPOCHS=3
fi

# 6. Locate Ground Truth Run Staging & Checkpoint Metadata
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_RUN_ID="unknown"
COMPLETED_EPOCHS=0
COMPLETED_TOKENS=0
LAST_EPOCH_MTIME=0
HAS_GROUND_TRUTH=0
VARIANT_SLOT="unknown"

# Search active staging directory under .staging/exp/f2/*/tracked/
# Find the newest tracked directory created/modified after process start
CANDIDATE_RUN_DIRS=$(ls -ldt "$REPO_ROOT"/.staging/exp/f2/*/tracked/* 2>/dev/null | awk '{print $NF}' || true)

for dir in $CANDIDATE_RUN_DIRS; do
  if [ -d "$dir" ]; then
    d_mtime=$(stat -c %Y "$dir" 2>/dev/null || echo 0)
    if [ "$d_mtime" -ge "$((PROC_START_TS - 120))" ]; then
      ACTIVE_RUN_ID=$(basename "$dir")
      # Check latest checkpoint
      LATEST_JSON="$dir/checkpoints/latest.json"
      if [ -f "$LATEST_JSON" ]; then
        COMPLETED_EPOCHS=$(awk -F': ' '/"epoch"/ {gsub(/[^0-9]/,"",$2); print $2}' "$LATEST_JSON" 2>/dev/null || echo 0)
        COMPLETED_TOKENS=$(awk -F': ' '/"update"/ {gsub(/[^0-9]/,"",$2); print $2}' "$LATEST_JSON" 2>/dev/null || echo 0)
        LAST_EPOCH_MTIME=$(stat -c %Y "$LATEST_JSON" 2>/dev/null || echo 0)
        HAS_GROUND_TRUTH=1
      fi
      break
    fi
  fi
done

# Check tmux log as additional ground truth verification if available
PROC_TTY=$( (ps -p "$PID" -o tty= 2>/dev/null || echo "") | tr -d ' ' )
if [ -n "$PROC_TTY" ] && [ "$PROC_TTY" != "?" ] && command -v tmux >/dev/null 2>&1; then
  TARGET_PANE=$(tmux list-panes -a -F "#{pane_tty} #{session_name}:#{window_index}.#{pane_index}" 2>/dev/null | awk -v tty="/dev/$PROC_TTY" '$1 == tty {print $2}' | head -n 1 || true)
  if [ -n "$TARGET_PANE" ]; then
    PANE_LOG=$(tmux capture-pane -t "$TARGET_PANE" -p -S -500 2>/dev/null || true)
    LAST_EPOCH_LINE=$(echo "$PANE_LOG" | grep -E "epoch=[0-9]+/[0-9]+" | tail -n 1 || true)
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
    SLOT_LINE=$(echo "$PANE_LOG" | grep -E "preparing .* slot=" | tail -n 1 || true)
    if [ -n "$SLOT_LINE" ]; then
      VARIANT_SLOT=$(echo "$SLOT_LINE" | sed -E 's/.*slot=([^ ]+).*/\1/' || echo "unknown")
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

# Epoch start time calculation:
# Epoch 1 start is process start time.
# Subsequent epochs use latest.json mtime as an approximation ([estimated]).
if [ "$COMPLETED_EPOCHS" -gt 0 ] && [ "$LAST_EPOCH_MTIME" -gt 0 ]; then
  CURRENT_EPOCH_START="$LAST_EPOCH_MTIME"
  EPOCH_START_IS_ESTIMATED=1
else
  CURRENT_EPOCH_START="$PROC_START_TS"
  EPOCH_START_IS_ESTIMATED=0
fi

# 7. Initial FD Measurement
T1=$(date +%s.%N 2>/dev/null || date +%s)
SORTED_ENTRIES_1=$(collect_pos_entries)

TOTAL_POS_1=0
POS_LIST_1=""
while read -r pos fd; do
  TOTAL_POS_1=$((TOTAL_POS_1 + pos))
  POS_LIST_1="$POS_LIST_1 $fd:$pos"
done <<< "$SORTED_ENTRIES_1"

# 8. Cache Validation & Inspection
# Cache key includes PID, PROC_START_TS, CORPUS_FILE, and COMPLETED_EPOCHS to invalidate across restarts/epochs
CACHE_FILE="/tmp/.w2v_status_${UID}_${PID}.cache"
USE_CACHE=0
CACHE_TIME=0
CACHE_POS=0

if [ -f "$CACHE_FILE" ]; then
  read -r C_START C_CORPUS C_EPOCH C_TIME C_POS < "$CACHE_FILE" 2>/dev/null || true
  if [ -n "${C_TIME:-}" ] && [ -n "${C_POS:-}" ]; then
    if [ "${C_START:-0}" = "$PROC_START_TS" ] && [ "${C_CORPUS:-}" = "$CORPUS_FILE" ] && [ "${C_EPOCH:-0}" = "$COMPLETED_EPOCHS" ]; then
      DIFF_TIME=$(awk -v t1="$C_TIME" -v t2="$T1" 'BEGIN {print (t2 - t1)}' 2>/dev/null || echo 0)
      if awk -v dt="$DIFF_TIME" -v cp="$C_POS" -v p1="$TOTAL_POS_1" 'BEGIN {exit !(dt >= 1.0 && dt <= 30.0 && p1 >= cp)}' 2>/dev/null; then
        USE_CACHE=1
        CACHE_TIME="$C_TIME"
        CACHE_POS="$C_POS"
      fi
    fi
  fi
fi

# 9. Print Immediate Top Summary
awk -v pid="$PID" \
    -v cmd_args="$CMD_ARGS" \
    -v suite="$SUITE" \
    -v model="$MODEL" \
    -v objective="$OBJECTIVE" \
    -v num_shards="$NUM_SHARDS" \
    -v corpus_file="$CORPUS_FILE" \
    -v corpus_size="$CORPUS_SIZE" \
    -v proc_etimes="$PROC_ETIMES" \
    -v current_epoch="$CURRENT_EPOCH" \
    -v completed_epochs="$COMPLETED_EPOCHS" \
    -v total_epochs="$TOTAL_EPOCHS" \
    -v completed_tokens="$COMPLETED_TOKENS" \
    -v epoch_start_est="$EPOCH_START_IS_ESTIMATED" \
    -v run_id="$ACTIVE_RUN_ID" \
    -v slot_id="$VARIANT_SLOT" \
    -v total_pos="$TOTAL_POS_1" '
BEGIN {
  now = systime()
  training_sec = proc_etimes
  if (training_sec <= 0) training_sec = 1

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

  printf "========================================================================\n"
  printf " F2 Word2Vec Training Monitor\n"
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
}
'

# 10. Sample Real-Time Delta or Use Valid Cache
if [ "$USE_CACHE" -eq 1 ]; then
  T_START="$CACHE_TIME"
  POS_START="$CACHE_POS"
  T_END="$T1"
  POS_END="$TOTAL_POS_1"
  POS_LIST="$POS_LIST_1"
elif [ "$SAMPLE_SECS" -gt 0 ]; then
  if [ -t 1 ]; then
    for ((i=1; i<=SAMPLE_SECS; i++)); do
      printf " \033[36m*\033[0m Sampling real-time throughput (%ds)... [%d/%d]\r" "$SAMPLE_SECS" "$i" "$SAMPLE_SECS"
      sleep 1
    done
    printf "\r\033[K"
  else
    sleep "$SAMPLE_SECS"
  fi

  T_END=$(date +%s.%N 2>/dev/null || date +%s)
  SORTED_ENTRIES_2=$(collect_pos_entries)
  POS_END=0
  POS_LIST_2=""
  while read -r pos fd; do
    POS_END=$((POS_END + pos))
    POS_LIST_2="$POS_LIST_2 $fd:$pos"
  done <<< "$SORTED_ENTRIES_2"

  T_START="$T1"
  POS_START="$TOTAL_POS_1"
  POS_LIST="$POS_LIST_2"
else
  T_START="$T1"
  POS_START="$TOTAL_POS_1"
  T_END="$T1"
  POS_END="$TOTAL_POS_1"
  POS_LIST="$POS_LIST_1"
fi

# Save state to cache with complete validation key
echo "$PROC_START_TS $CORPUS_FILE $COMPLETED_EPOCHS $T_END $POS_END" > "$CACHE_FILE" 2>/dev/null || true

# 11. Throughput & ETA Evaluation
awk -v num_shards="$NUM_SHARDS" \
    -v corpus_size="$CORPUS_SIZE" \
    -v proc_start_ts="$PROC_START_TS" \
    -v current_epoch_start="$CURRENT_EPOCH_START" \
    -v epoch_start_est="$EPOCH_START_IS_ESTIMATED" \
    -v completed_epochs="$COMPLETED_EPOCHS" \
    -v total_epochs="$TOTAL_EPOCHS" \
    -v completed_tokens="$COMPLETED_TOKENS" \
    -v total_pos="$POS_END" \
    -v pos_start="$POS_START" \
    -v t_start="$T_START" \
    -v t_end="$T_END" \
    -v model="$MODEL" \
    -v pos_list="$POS_LIST" \
    -v show_detail="$SHOW_DETAIL" '
BEGIN {
  now = systime()
  total_elapsed_sec = now - proc_start_ts
  if (total_elapsed_sec <= 0) total_elapsed_sec = 1

  epoch_elapsed_sec = now - current_epoch_start
  if (epoch_elapsed_sec <= 0) epoch_elapsed_sec = 1

  start_base = (num_shards - 1) / 2.0 * corpus_size
  done_bytes = total_pos - start_base
  if (done_bytes < 0) done_bytes = 0
  if (done_bytes > corpus_size) done_bytes = corpus_size

  remain_bytes = corpus_size - done_bytes
  bytes_per_token = 5.353556
  epoch_done_tokens_est = done_bytes / bytes_per_token

  all_total_bytes = total_epochs * corpus_size
  all_done_bytes = (completed_epochs * corpus_size) + done_bytes
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
