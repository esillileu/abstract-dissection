#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Run the shorter original trials first. Valid instrumented CuPy caches are
# reused, so this command can be safely rerun after an interruption. E08
# follows its required E07 attention checkpoint.
uv run python -m exp ds2 run --original -e 01,03,04,06,07,08

# The two full PTB Word2Vec trials are the longest-running originals.
uv run python -m exp ds2 run --original -e 02
