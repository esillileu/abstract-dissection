"""Native catalog definitions for DeepScratch volume 2."""

from pathlib import Path

from exp.domain import DomainDefinition


ROOT = Path("exp/deepscratch/ds2")

IMPLEMENTED = DomainDefinition(
    name="ds2",
    config_root=ROOT / "config/implemented",
    spec_module="exp.deepscratch.ds2.implemented.spec",
    executor_module="exp.deepscratch.ds2.implemented.executor",
    analysis_module="exp.deepscratch.ds2.analysis.render",
)

ORIGINAL = DomainDefinition(
    name="ds2_original",
    config_root=ROOT / "config/original",
    spec_module="exp.deepscratch.ds2.original.spec",
    executor_module="exp.deepscratch.ds2.original.executor",
    analysis_module="exp.deepscratch.ds2.original.results",
)
