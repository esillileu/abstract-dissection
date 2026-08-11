from pathlib import Path

from mlprosection.experiment.registry import register_executor

from exp.original.promoted_executor import execute


@register_executor("ds1_original")
class DS1OriginalExecutor:
    def run(self, config, context):
        return execute(config, context, domain="ds1_original", source_root=Path(__file__).parent / "src")
