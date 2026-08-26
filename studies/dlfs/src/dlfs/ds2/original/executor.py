from pathlib import Path

from dlfs.original_runtime.promoted_executor import execute
from repro_core.context.paths import RuntimePaths


class DS2OriginalExecutor:
    def run(self, config, context):
        ref_root = RuntimePaths.from_environment().reference("dlfs2-book") / "source"
        if not ref_root.exists():
            ref_root = Path(__file__).parent / "source"
        return execute(
            config,
            context,
            domain="deepscratch.ds2.original",
            source_root=ref_root,
        )


_EXECUTORS = {
    "deepscratch.ds2.original": DS2OriginalExecutor(),
}


def get_executor(kind: str):
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown DS2 original experiment kind: {kind}") from exc
