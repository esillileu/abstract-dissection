"""Native catalog definitions for DLFS volume 2 (DS2)."""

from pathlib import Path

from repro_core.execution import ExecutionDefinition

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SOURCE_RESOLVER = "repro_mlflow.checkpoint_source"

IMPLEMENTED = ExecutionDefinition(
    name="deepscratch.ds2.implemented",
    config_root=ROOT / "config/implemented",
    spec_module="dlfs.ds2.implemented.spec",
    executor_module="dlfs.ds2.implemented.executor",
    checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
    domain="deepscratch",
    suite="ds2",
    variant="implemented",
    all_excluded_kinds=("performance_profile",),
)

ORIGINAL = ExecutionDefinition(
    name="deepscratch.ds2.original",
    config_root=ROOT / "config/original",
    spec_module="dlfs.ds2.original.spec",
    executor_module="dlfs.ds2.original.executor",
    checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
    domain="deepscratch",
    suite="ds2",
    variant="original",
)
