"""Native catalog definitions for DeepScratch volume 1."""

from pathlib import Path

from exp.framework.execution import ExecutionDefinition


ROOT = Path("exp/deepscratch/ds1")
CHECKPOINT_SOURCE_RESOLVER = "exp.deepscratch.legacy"

IMPLEMENTED = ExecutionDefinition(
    name="deepscratch.ds1.implemented",
    config_root=ROOT / "config/implemented",
    spec_module="exp.deepscratch.ds1.implemented.spec",
    executor_module="exp.deepscratch.ds1.implemented.executor",
    checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
    domain="deepscratch",
    suite="ds1",
    variant="implemented",
)

ORIGINAL = ExecutionDefinition(
    name="deepscratch.ds1.original",
    config_root=ROOT / "config/original",
    spec_module="exp.deepscratch.ds1.original.spec",
    executor_module="exp.deepscratch.ds1.original.executor",
    checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
    domain="deepscratch",
    suite="ds1",
    variant="original",
)
