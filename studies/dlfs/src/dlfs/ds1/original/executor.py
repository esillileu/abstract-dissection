from pathlib import Path

from dlfs.original_runtime.promoted_executor import execute
from repro_core.context.paths import RuntimePaths
from repro_core.registry import register_executor


@register_executor("deepscratch.ds1.original")
class DS1OriginalExecutor:
    def run(self, config, context):
        ref_root = RuntimePaths.from_environment().reference("dlfs1-book") / "source"
        if not ref_root.exists():
            ref_root = Path(__file__).parent / "source"
        return execute(
            config,
            context,
            domain="deepscratch.ds1.original",
            source_root=ref_root,
        )
