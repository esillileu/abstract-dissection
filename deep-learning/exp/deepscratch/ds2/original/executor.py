from pathlib import Path

from mlprosection.experiment.registry import register_executor

from exp.deepscratch.original_runtime.promoted_executor import execute


@register_executor("ds2_original")
class DS2OriginalExecutor:
    def run(self, config, context):
        return execute(
            config,
            context,
            domain="ds2_original",
            source_root=Path(__file__).parent / "source",
        )
