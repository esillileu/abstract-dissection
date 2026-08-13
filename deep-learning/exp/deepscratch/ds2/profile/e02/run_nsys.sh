#!/usr/bin/env bash
set -euo pipefail

profile_root="exp/deepscratch/ds2/profile/e02/results/nsys"
measured_updates="${MEASURED_UPDATES:-100}"
phase_updates="${PHASE_UPDATES:-5}"
mkdir -p "${profile_root}"

conditions=(
  original-cbow-onehot-fs
  original-cbow-fs
  original-cbow-ns
  original-skipgram-onehot-fs
  original-skipgram-fs
  original-skipgram-ns
  implemented-cbow-onehot-fs
  implemented-cbow-fs
  implemented-cbow-ns
  implemented-cbow-fused-ns
  implemented-skipgram-onehot-fs
  implemented-skipgram-fs
  implemented-skipgram-ns
  implemented-skipgram-fused-ns
)
if (( $# > 0 )); then
  conditions=("$@")
fi

for condition in "${conditions[@]}"; do
  attempt=1
  until nsys profile \
    --force-overwrite=true \
    --sample=none \
    --trace=cuda,nvtx,osrt \
    --output="${profile_root}/${condition}" \
    uv run python -m exp.deepscratch.ds2.profile.e02.update \
      --condition "${condition}" \
      --device cuda:0 \
      --stage detail \
      --warmup-updates 5 \
      --measured-updates "${measured_updates}" \
      --phase-updates "${phase_updates}" \
      --repetitions 1 \
      --output "${profile_root}/${condition}.json"
  do
    if (( attempt >= 3 )); then
      echo "nsys profile failed after ${attempt} attempts: ${condition}" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "retrying nsys profile (${attempt}/3): ${condition}" >&2
  done
  nsys stats \
    --force-export=true \
    --report cuda_api_sum \
    "${profile_root}/${condition}.nsys-rep" >/dev/null
done

uv run python -m exp.deepscratch.ds2.profile.e02.summarize_nsys
