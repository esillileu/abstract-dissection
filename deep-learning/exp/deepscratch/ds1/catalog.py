"""Native catalog definitions for DeepScratch volume 1."""

from pathlib import Path

from exp.domain import DomainDefinition


ROOT = Path("exp/deepscratch/ds1")

IMPLEMENTED = DomainDefinition(
    name="ds1",
    config_root=ROOT / "config/implemented",
    spec_module="exp.deepscratch.ds1.implemented.spec",
    executor_module="exp.deepscratch.ds1.implemented.executor",
    analysis_module="exp.deepscratch.ds1.analysis.render",
)

ORIGINAL = DomainDefinition(
    name="ds1_original",
    config_root=ROOT / "config/original",
    spec_module="exp.deepscratch.ds1.original.spec",
    executor_module="exp.deepscratch.ds1.original.executor",
    analysis_module="exp.deepscratch.ds1.original.results",
)
