"""Adapter for implemented DS2 SchemaV1 runs."""

from exp.deepscratch.identity import Variant
from exp.framework.results import MlflowResultStore


def load_native_result(client, run_id, declarations):
    specs = _metric_specs(declarations)
    result = MlflowResultStore(client).load(
        run_id, metric_specs=specs, include_artifacts=False
    )
    if result.schema_version != 1:
        raise ValueError(
            f"canonical run {run_id} has unsupported schema version "
            f"{result.schema_version}"
        )
    return result


def _metric_specs(declarations):
    return list(dict.fromkeys(
        (metric_id, declaration.unit, declaration.split, declaration.axis)
        for declaration in declarations
        for metric_id in declaration.native_ids(Variant.IMPLEMENTED)
    ))
