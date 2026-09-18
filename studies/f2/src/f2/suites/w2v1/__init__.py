"""W2V1 reconstruction suite composition."""

from pathlib import Path

from f2.suites.protocol import SuiteCatalog

DEFINITION = SuiteCatalog("w2v1", Path(__file__).parent).build_definition()

__all__ = ["DEFINITION"]
