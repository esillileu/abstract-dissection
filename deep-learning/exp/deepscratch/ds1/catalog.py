"""Native catalog definitions for DeepScratch volume 1."""

from pathlib import Path

from exp.framework.execution import ExecutionDefinition


ROOT = Path("exp/deepscratch/ds1")

IMPLEMENTED = ExecutionDefinition(
    name="deepscratch.ds1.implemented",
    config_root=ROOT / "config/implemented",
    spec_module="exp.deepscratch.ds1.implemented.spec",
    executor_module="exp.deepscratch.ds1.implemented.executor",
    domain="deepscratch",
    suite="ds1",
    variant="implemented",
)

ORIGINAL = ExecutionDefinition(
    name="deepscratch.ds1.original",
    config_root=ROOT / "config/original",
    spec_module="exp.deepscratch.ds1.original.spec",
    executor_module="exp.deepscratch.ds1.original.executor",
    domain="deepscratch",
    suite="ds1",
    variant="original",
)
