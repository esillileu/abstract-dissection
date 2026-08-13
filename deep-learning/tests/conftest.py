from __future__ import annotations

import sys
from pathlib import Path


# The repository-level ``exp`` migration workspace is intentionally not part of
# the installable ``src`` wheel, but tracking integration tests exercise both.
sys.path.insert(0, str(Path(__file__).parents[1]))
