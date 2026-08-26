from dlfs.original_runtime.promoted_spec import parse


def parse_run_spec(path, *, atomic_run_id=None, overrides=None):
    return parse(
        path,
        domain="deepscratch.ds2.original",
        atomic_run_id=atomic_run_id,
        overrides=overrides,
    )
