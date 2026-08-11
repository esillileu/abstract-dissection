from exp.original.promoted_analyze import analyze as _analyze


def analyze(**kwargs):
    return _analyze(domain="ds1_original", **kwargs)
