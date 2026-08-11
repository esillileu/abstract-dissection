from exp.original.promoted_spec import parse


def parse_run_spec(path, *, atomic_run_id=None, overrides=None):
    return parse(path, domain="ds2_original", atomic_run_id=atomic_run_id, overrides=overrides)
