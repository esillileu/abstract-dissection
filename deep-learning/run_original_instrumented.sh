#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Runner schema v2 records timing.json and parameter_manifest.json. Existing
# schema-v1 caches are invalidated automatically.
uv run python -m exp ds1 run --original -e 01-07

# DS2 uses the existing resumable CuPy runner script, with the expensive PTB
# Word2Vec trials deliberately scheduled last.
"$repo_root/run_ds2_original_cupy.sh"
