from pathlib import Path

from mlprosection.experiment.registry import register_executor

from exp.deepscratch.original_runtime.promoted_executor import execute


@register_executor("deepscratch.ds1.original")
class DS1OriginalExecutor:
    def run(self, config, context):
        return execute(
            config,
            context,
            domain="deepscratch.ds1.original",
            source_root=Path(__file__).parent / "source",
        )
