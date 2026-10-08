#!/usr/bin/env bash
# ==============================================================================
# F2 Word2Vec (W2V1 / W2V2) Training Status & Throughput Monitor
# Supports: CBOW, Skip-Gram, NNLM, Negative Sampling, Hierarchical Softmax, Phrases
#
# Usage:
#   tools/w2v_status.sh           # View current status summary & throughput
#   tools/w2v_status.sh -d        # View with per-shard replica details
#   tools/w2v_status.sh -w        # Watch mode (updates every 5 seconds)
# ==============================================================================

set -euo pipefail

# Handle watch mode
if [ "${1:-}" = "-w" ] || [ "${1:-}" = "--watch" ]; then
  shift
  exec watch -n 5 "$0" "$@"
fi

SHOW_DETAIL=0
if [ "${1:-}" = "-d" ] || [ "${1:-}" = "--detail" ]; then
  SHOW_DETAIL=1
fi

# 1. Locate active python repro run process
PID=$(pgrep -f "python.*repro run f2" | head -n 1 || true)
if [ -z "$PID" ]; then
  PID=$(pgrep -f "repro run f2" | head -n 1 || true)
fi

if [ -z "$PID" ]; then
  echo "========================================================================"
  echo " No active F2 W2V training process ('repro run f2') was found."
  echo "========================================================================"
  exit 0
fi

# 2. Locate active corpus file
# In w2v1 and w2v2, the active training corpus (materialized/*.txt or phrases.txt)
# is opened concurrently by all worker threads. We detect the most referenced text file.
CORPUS_FILE=$(ls -l "/proc/$PID/fd" 2>/dev/null | awk '{print $NF}' | grep -E '\.(txt|bin)$' | sort | uniq -c | sort -nr | head -n 1 | awk '{print $2}' || true)

if [ -z "$CORPUS_FILE" ] || [ ! -f "$CORPUS_FILE" ]; then
  echo "========================================================================"
  echo " Process PID $PID is running, but no active training corpus FD found."
  echo " (The process may be preparing datasets, phrases, or evaluating)."
  echo "========================================================================"
  exit 0
fi

CORPUS_SIZE=$(stat -c %s "$CORPUS_FILE" 2>/dev/null || stat -f %z "$CORPUS_FILE" 2>/dev/null)
CORPUS_MTIME=$(stat -c %Y "$CORPUS_FILE" 2>/dev/null || stat -f %m "$CORPUS_FILE" 2>/dev/null)

# 3. Collect all worker file descriptors referencing this corpus
FDS=$(ls -l "/proc/$PID/fd" 2>/dev/null | awk -v cf="$CORPUS_FILE" '$NF == cf {print $9}')
NUM_SHARDS=$(echo "$FDS" | wc -w)

if [ "$NUM_SHARDS" -eq 0 ]; then
  echo "========================================================================"
  echo " Process PID $PID has no active open descriptors for $CORPUS_FILE."
  echo "========================================================================"
  exit 0
fi

# Sort worker descriptors by byte offset
SORTED_ENTRIES=$(
  for fd in $FDS; do
    pos=$(awk '$1 == "pos:" {print $2}' "/proc/$PID/fdinfo/$fd" 2>/dev/null || echo 0)
    echo "$pos $fd"
  done | sort -k1,1n
)

TOTAL_POS=0
POS_LIST=""
while read -r pos fd; do
  TOTAL_POS=$((TOTAL_POS + pos))
  POS_LIST="$POS_LIST $fd:$pos"
done <<< "$SORTED_ENTRIES"

# 4. Extract command line arguments and detect suite / model characteristics
CMD_ARGS=$(ps -p "$PID" -o args= 2>/dev/null | sed -e 's/.*repro/repro/' || echo "repro run f2")

# Suite detection
SUITE="w2v"
if echo "$CMD_ARGS" | grep -q "w2v2"; then
  SUITE="w2v2 (Phrases / Negative Sampling)"
elif echo "$CMD_ARGS" | grep -q "w2v1"; then
  SUITE="w2v1 (Word Relationships)"
fi

# Model detection
MODEL="Standard"
if echo "$CMD_ARGS" | grep -qi "cbow"; then
  MODEL="CBOW"
elif echo "$CMD_ARGS" | grep -qi "skipgram"; then
  MODEL="Skip-Gram"
elif echo "$CMD_ARGS" | grep -qi "nnlm"; then
  MODEL="Feedforward NNLM"
fi

# Objective detection
OBJECTIVE="Hierarchical Softmax"
if echo "$CMD_ARGS" | grep -qiE "neg|ns|negative"; then
  OBJECTIVE="Negative Sampling"
fi

# 5. Calculate statistics and print report via awk
awk -v pid="$PID" \
    -v cmd_args="$CMD_ARGS" \
    -v suite="$SUITE" \
    -v model="$MODEL" \
    -v objective="$OBJECTIVE" \
    -v num_shards="$NUM_SHARDS" \
    -v corpus_file="$CORPUS_FILE" \
    -v corpus_size="$CORPUS_SIZE" \
    -v corpus_mtime="$CORPUS_MTIME" \
    -v total_pos="$TOTAL_POS" \
    -v pos_list="$POS_LIST" \
    -v show_detail="$SHOW_DETAIL" '
BEGIN {
  now = systime()
  training_sec = now - corpus_mtime
  if (training_sec <= 0) training_sec = 1

  # Each shard starts at (i * corpus_size / num_shards)
  # Sum of starts = ((num_shards - 1) / 2) * corpus_size
  start_base = (num_shards - 1) / 2.0 * corpus_size
  done_bytes = total_pos - start_base
  if (done_bytes < 0) done_bytes = 0
  if (done_bytes > corpus_size) done_bytes = corpus_size

  remain_bytes = corpus_size - done_bytes
  pct = (done_bytes / corpus_size) * 100.0

  # English corpus average token density: ~5.3535 bytes / word
  bytes_per_token = 5.353556
  tokens_total = corpus_size / bytes_per_token
  done_tokens = done_bytes / bytes_per_token
  remain_tokens = remain_bytes / bytes_per_token

  speed_bytes = done_bytes / training_sec
  speed_tokens = done_tokens / training_sec

  epoch_total_sec = corpus_size / speed_bytes
  epoch_remain_sec = remain_bytes / speed_bytes

  # Multi-epoch estimates
  # 1 Epoch (CBOW / Skip-Gram standard), 3 Epochs (NNLM), 5 Epochs (typical)
  remain_3ep_sec = epoch_remain_sec + 2 * epoch_total_sec
  remain_5ep_sec = epoch_remain_sec + 4 * epoch_total_sec

  printf "========================================================================\n"
  printf " F2 Word2Vec Training Monitor\n"
  printf "========================================================================\n"
  printf " Suite / Model:       %s | %s (%s)\n", suite, model, objective
  printf " Command:             %s\n", cmd_args
  printf " PID:                 %d (Workers: %d parallel threads)\n", pid, num_shards
  printf " Elapsed Time:        %.1f hours (%d seconds)\n", training_sec / 3600.0, training_sec
  printf " Corpus File:         %s\n", corpus_file
  printf " Corpus Size:         %.2f GB (%d bytes, ~%.1fM tokens)\n", corpus_size / (1024^3), corpus_size, tokens_total / 1e6
  printf "------------------------------------------------------------------------\n"
  printf " [Current Epoch Progress]\n"
  printf " Processed Bytes:     %.2f GB / %.2f GB (%.2f%%)\n", done_bytes / (1024^3), corpus_size / (1024^3), pct
  printf " Processed Tokens:    %.1fM / ~%.1fM tokens\n", done_tokens / 1e6, tokens_total / 1e6
  printf " Remaining in Epoch:  %.2f MB (~%.1fM tokens)\n", remain_bytes / (1024^2), remain_tokens / 1e6
  printf "------------------------------------------------------------------------\n"
  printf " [Throughput]\n"
  printf " Byte Speed:          %.1f KB/s (%.1f bytes/s)\n", speed_bytes / 1024.0, speed_bytes
  printf " Token Speed:         %.1f tokens/sec\n", speed_tokens
  printf " Time Per 1 Epoch:    %.2f hours (~%d mins)\n", epoch_total_sec / 3600.0, epoch_total_sec / 60
  printf "------------------------------------------------------------------------\n"
  printf " [Estimated Completion Time (ETA)]\n"

  cmd = "date -d @" int(now + epoch_remain_sec) " \"+%Y-%m-%d %H:%M:%S\""
  cmd | getline eta_ep1; close(cmd)
  printf " Current Epoch Finish: in %.2f hours (~%d mins)   -> %s\n", epoch_remain_sec / 3600.0, epoch_remain_sec / 60, eta_ep1

  if (model ~ /NNLM/) {
    cmd = "date -d @" int(now + remain_3ep_sec) " \"+%Y-%m-%d %H:%M:%S\""
    cmd | getline eta_3ep; close(cmd)
    printf " Full Run (3 epochs): in %.2f hours (~%.1f days)   -> %s\n", remain_3ep_sec / 3600.0, remain_3ep_sec / 86400.0, eta_3ep
  } else {
    cmd = "date -d @" int(now + remain_3ep_sec) " \"+%Y-%m-%d %H:%M:%S\""
    cmd | getline eta_3ep; close(cmd)
    printf " If 3 Epochs Run:     in %.2f hours (~%.1f days)   -> %s\n", remain_3ep_sec / 3600.0, remain_3ep_sec / 86400.0, eta_3ep
  }

  if (show_detail == 1) {
    printf "------------------------------------------------------------------------\n"
    printf " [Shard Details (%d Replicas)]\n", num_shards
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
