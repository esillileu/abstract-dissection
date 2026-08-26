"""Native catalog definitions for DLFS volume 1 (DS1)."""

from pathlib import Path

from repro_core.execution import ExecutionDefinition

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SOURCE_RESOLVER = "repro_mlflow.checkpoint_source"

IMPLEMENTED = ExecutionDefinition(
    name="deepscratch.ds1.implemented",
    config_root=ROOT / "config/implemented",
    spec_module="dlfs.ds1.implemented.spec",
    executor_module="dlfs.ds1.implemented.executor",
    checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
    domain="deepscratch",
    suite="ds1",
    variant="implemented",
)

ORIGINAL = ExecutionDefinition(
    name="deepscratch.ds1.original",
    config_root=ROOT / "config/original",
    spec_module="dlfs.ds1.original.spec",
    executor_module="dlfs.ds1.original.executor",
    checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
    domain="deepscratch",
    suite="ds1",
    variant="original",
)
