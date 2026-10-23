"""W2V2 word and phrase reconstruction suite composition."""

from pathlib import Path

from f2.suites.protocol import SuiteCatalog

DEFINITION = SuiteCatalog("w2v2", Path(__file__).parent).build_definition()

__all__ = ["DEFINITION"]
