"""Markdown report renderer for e02 fused negative-sampling validation."""

from __future__ import annotations

from .metrics import _max_error


def render_report(result: dict[str, object]) -> str:
    protocol = result["protocol"]
    lines = [
        "# e02 fused negative-sampling validation",
        "",
        "## Result",
        "",
        "The ordinary and fused Word2Vec paths were compared in lockstep.",
        "Both paths used the same initial parameters, batches, negative",
        "candidate IDs, and Adam optimizer state.",
        "",
        "| device | model | updates | max loss abs. error | max gradient combined error | max parameter combined error | max Adam-state combined error | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for device, cases in result["comparisons"].items():
        for kind, case in cases.items():
            rows = case["updates"]
            lines.append(
                "| {device} | {kind} | {updates} | {loss:.3e} | {gradient:.3e} | "
                "{parameter:.3e} | {state:.3e} | {status} |".format(
                    device=device,
                    kind=kind,
                    updates=len(rows),
                    loss=_max_error(rows, "loss_error", "max_absolute"),
                    gradient=_max_error(rows, "gradient_errors", "required_atol_rtol"),
                    parameter=_max_error(
                        rows, "parameter_errors", "required_atol_rtol"
                    ),
                    state=_max_error(
                        rows, "optimizer_state_errors", "required_atol_rtol"
                    ),
                    status="PASS" if case["passed"] else "FAIL",
                )
            )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- dtype: `{protocol['dtype']}`",
            f"- updates per model: `{protocol['updates']}`",
            f"- negative samples: `{protocol['negative_samples']}`",
            f"- optimizer: `{protocol['optimizer']}`",
            f"- same negative candidates: `{protocol['same_negative_candidates']}`",
            f"- loss ceiling: `{protocol['loss_ceiling']}`",
            f"- gradient ceiling: `{protocol['gradient_ceiling']}`",
            f"- parameter/optimizer-state ceiling: `{protocol['parameter_and_optimizer_ceiling']}`",
            "- CUDA comparisons synchronize before reading gradients and parameters.",
            "",
            f"Overall status: **{'PASS' if result['passed'] else 'FAIL'}**",
            "",
            "The complete per-update values are stored in the adjacent JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)
