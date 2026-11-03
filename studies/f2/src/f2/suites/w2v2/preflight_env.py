"""System environment and memory estimation helpers for preflight."""

from __future__ import annotations

from pathlib import Path

import psutil


def resolve_source(config: dict, paths, override: Path | None = None) -> Path:
    """Inject the configured dataset root; absolute overrides are explicit inputs."""
    candidate = override or Path(str(config["corpus"]["path"]))
    return (
        candidate
        if candidate.is_absolute()
        else paths.dataset("f2") / "corpus" / candidate
    )


def available_memory(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
) -> int:
    """Use the smallest remaining host/current-cgroup/ancestor budget."""
    available = psutil.virtual_memory().available
    relative = next(
        (
            line.split("::", 1)[1]
            for line in proc_cgroup.read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if relative is None:
        raise ValueError("cgroup v2 memory accounting is required")
    cgroup_root = cgroup_root.resolve()
    leaf = (cgroup_root / relative.lstrip("/")).resolve()
    if not leaf.is_relative_to(cgroup_root) or not leaf.is_dir():
        leaf = cgroup_root
    while True:
        maximum, current = leaf / "memory.max", leaf / "memory.current"
        if maximum.is_file() and current.is_file():
            value = maximum.read_text().strip()
            if value != "max":
                available = min(
                    available, max(0, int(value) - int(current.read_text()))
                )
        if leaf == cgroup_root:
            break
        leaf = leaf.parent
    return available


__all__ = ["available_memory", "resolve_source"]
