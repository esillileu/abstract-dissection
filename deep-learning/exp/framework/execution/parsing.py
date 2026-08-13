"""Pure parsers used by all experiment domains."""

from __future__ import annotations

import re

import yaml


def deep_merge(
    base: dict[str, object], override: dict[str, object]
) -> dict[str, object]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def parse_overrides(values: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be KEY=VALUE: {value}")
        key, raw = value.split("=", 1)
        if not key or any(not part for part in key.split(".")):
            raise ValueError(f"override key must use dotted names: {key}")
        cursor = result
        for part in key.split(".")[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"conflicting override path: {key}")
            cursor = child
        leaf = key.split(".")[-1]
        if leaf in cursor and isinstance(cursor[leaf], dict):
            raise ValueError(f"conflicting override path: {key}")
        cursor[leaf] = yaml.safe_load(raw)
    return result


def parse_seed_values(value: str | None, *, available: list[int]) -> list[int] | None:
    if value is None:
        return None
    values: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError("seed value must not be empty")
        try:
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError(f"seed range must be ascending: {item}")
                values.extend(range(start, end + 1))
            else:
                values.append(int(item))
        except ValueError as exc:
            if str(exc).startswith("seed range"):
                raise
            raise ValueError(f"invalid seed selection: {item}") from exc
    selected = list(dict.fromkeys(values))
    invalid = [seed for seed in selected if seed not in available]
    if invalid:
        valid = ", ".join(str(seed) for seed in available)
        raise ValueError(
            f"seed values are not in the selected seed set ({valid}): {invalid}"
        )
    return selected


def parse_experiment_ids(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        for raw_item in value.split(","):
            item = raw_item.strip().lower()
            match = re.fullmatch(r"e?(\d+)(?:-e?(\d+))?", item)
            if match is None:
                raise ValueError(f"invalid experiment selection: {item}")
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if start > end:
                raise ValueError(f"experiment range must be ascending: {item}")
            selected.extend(f"e{number:02d}" for number in range(start, end + 1))
    return list(dict.fromkeys(selected))


def parse_atomic_run_ids(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        for raw_item in value.split(","):
            item = raw_item.strip()
            if not item:
                raise ValueError("atomic run ID must not be empty")
            selected.append(item)
    return list(dict.fromkeys(selected))
