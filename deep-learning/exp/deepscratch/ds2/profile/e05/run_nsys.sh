#!/usr/bin/env bash
set -euo pipefail

profile_root="${EXP_CACHE_ROOT:-.cache/experiments}/deepscratch/ds2/e05/implemented/profile/nsys"
payload_root="$(mktemp -d /tmp/e05-nsys-payload.XXXXXX)"
nsys_bin="${NSYS_BIN:-$(command -v nsys)}"
mkdir -p "${profile_root}" "${payload_root}"

stages=(baseline phase1 phase2 phase3)
if (( $# > 0 )); then
  stages=("$@")
fi

for stage in "${stages[@]}"; do
  attempt=1
  until "${nsys_bin}" profile \
    --force-overwrite=true \
    --sample=none \
    --trace=cuda,nvtx,osrt \
    --output="${profile_root}/${stage}" \
    uv run python -m exp profile ds2 -e 05 \
      --stage "${stage}" \
      --trace \
      --update-warmup 2 \
      --measured-updates 5 \
      --update-repetitions 1 \
      --output-dir "${payload_root}/${stage}"
  do
    if (( attempt >= 3 )); then
      echo "Nsight failed after ${attempt} attempts: ${stage}" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "retrying Nsight (${attempt}/3): ${stage}" >&2
  done
  "${nsys_bin}" stats \
    --force-export=true \
    --report cuda_api_sum \
    "${profile_root}/${stage}.nsys-rep" >/dev/null
done

uv run python -m exp.deepscratch.ds2.profile.e05.summarize_nsys
