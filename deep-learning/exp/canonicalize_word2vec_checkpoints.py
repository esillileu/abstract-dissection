"""Canonicalize Word2Vec artifacts so the model owns both embedding matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MigrationReport:
    artifact_roots: int
    checkpoint_generations: int
    moved_output_weights: int
    renamed_optimizer_states: int
    updated_config_digests: int
    parameter_manifests: int
    mirrors: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as file:
        temporary = Path(file.name)
        np.savez(file, **arrays)
    os.replace(temporary, path)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parameter_entry(name: str, data: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "requires_grad": True,
        "numel": int(data.size),
        "final_mean": float(data.mean()),
        "final_std": float(data.std()),
        "final_min": float(data.min()),
        "final_max": float(data.max()),
        "final_digest": hashlib.sha256(data.tobytes()).hexdigest(),
    }


def _is_word2vec_root(root: Path) -> bool:
    condition_path = root / "config/condition.json"
    if not condition_path.is_file():
        return False
    condition = _read_json(condition_path)
    return str(condition.get("atomic_run_id", "")).startswith("W2V-")


def _rename_optimizer_state(path: Path, *, apply: bool) -> bool:
    if not path.is_file():
        raise FileNotFoundError(f"missing optimizer state: {path}")
    with path.open("rb") as file:
        state = pickle.load(file)
    changed = False
    for value in state.values():
        if not isinstance(value, dict) or "objective.W_out" not in value:
            continue
        if "model.W_out" in value:
            raise ValueError(f"duplicate W_out optimizer state in {path}")
        if apply:
            value["model.W_out"] = value.pop("objective.W_out")
        changed = True
    if changed and apply:
        _atomic_bytes(path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    return changed


def _canonicalize_generation(
    path: Path,
    *,
    config_digest: str | None,
    apply: bool,
) -> tuple[bool, bool, bool]:
    model_path = path / "model_parameters.npz"
    objective_path = path / "objective_parameters.npz"
    model = _read_npz(model_path)
    objective = _read_npz(objective_path)
    moved = False
    if "W_out" not in model:
        if set(objective) != {"W_out"} or "W_in" not in model:
            raise ValueError(f"unexpected Word2Vec parameter layout: {path}")
        if apply:
            model["W_out"] = objective["W_out"]
            _write_npz(model_path, model)
            _write_npz(objective_path, {})
        moved = True
    elif objective:
        raise ValueError(f"canonical Word2Vec objective has parameters: {path}")
    renamed = _rename_optimizer_state(
        path / "optimizer_state.pkl",
        apply=apply,
    )
    manifest_path = path / "manifest.json"
    manifest = (
        _read_json(manifest_path)
        if config_digest is not None
        else None
    )
    updated_digest = bool(
        manifest is not None
        and manifest.get("config_digest") != config_digest
    )
    if updated_digest and apply:
        assert manifest is not None
        manifest["config_digest"] = config_digest
        _write_json(manifest_path, manifest)
    return moved, renamed, updated_digest


def _current_config_digest(root: Path) -> str | None:
    condition = _read_json(root / "config/condition.json")
    training = condition.get("training")
    if not isinstance(training, dict) or "entrypoint" not in training:
        return None
    seed_path = root / "config/seed.json"
    if not seed_path.is_file():
        return None
    entrypoint = Path(str(training["entrypoint"]))
    if not entrypoint.is_file():
        return None
    from exp.deepscratch.ds2.implemented.executor import _config_digest
    from exp.deepscratch.ds2.implemented.spec import parse_run_spec

    config = parse_run_spec(
        entrypoint,
        atomic_run_id=str(condition["atomic_run_id"]),
    ).to_executor_config()
    config["seed"] = int(_read_json(seed_path)["master"])
    return _config_digest(config)


def _update_parameter_manifest(
    root: Path,
    generation: Path,
    *,
    apply: bool,
) -> bool:
    path = root / "model/parameter_manifest.json"
    entries = _read_json(path)
    names = [entry.get("name") for entry in entries]
    if "W_out" in names:
        return False
    if names != ["W_in"]:
        raise ValueError(f"unexpected Word2Vec parameter manifest: {path}")
    model = _read_npz(generation / "model_parameters.npz")
    if "W_out" not in model:
        if not apply:
            objective = _read_npz(generation / "objective_parameters.npz")
            output_weight = objective["W_out"]
        else:
            raise ValueError(f"migrated checkpoint has no W_out: {generation}")
    else:
        output_weight = model["W_out"]
    entries.append(_parameter_entry("W_out", output_weight))
    if apply:
        _write_json(path, entries)
    return True


def _update_checkpoint_metadata(root: Path, *, apply: bool) -> None:
    role_digests: dict[str, str] = {}
    for pointer_path in sorted((root / "checkpoints").glob("*.json")):
        if pointer_path.name == "checkpoint_manifest.json":
            continue
        pointer = _read_json(pointer_path)
        generation = root / "checkpoints" / str(pointer["path"])
        digest = _path_digest(generation)
        role = str(pointer["role"])
        pointer["sha256"] = digest
        role_digests[role] = digest
        if apply:
            _write_json(pointer_path, pointer)

    manifest_path = root / "checkpoints/checkpoint_manifest.json"
    manifest = _read_json(manifest_path)
    for role in ("latest", "best"):
        value = manifest.get(role)
        if value is not None and role in role_digests:
            value["digest"] = role_digests[role]
    if manifest.get("final") is not None and "latest" in role_digests:
        manifest["final"]["digest"] = role_digests["latest"]
    if apply:
        _write_json(manifest_path, manifest)

    csv_path = root / "checkpoints.csv"
    if not csv_path.is_file():
        return
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        role = "best" if row.get("kind") in {"best", "selected"} else "latest"
        if "sha256" in row and role in role_digests:
            row["sha256"] = role_digests[role]
    if apply:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=csv_path.parent,
            delete=False,
        ) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, csv_path)


def _copy_canonical_metadata(root: Path, mirror: Path, *, apply: bool) -> None:
    for relative in (
        "model/parameter_manifest.json",
        "checkpoints/checkpoint_manifest.json",
        "checkpoints.csv",
    ):
        source = root / relative
        target = mirror / relative
        if source.is_file() and target.is_file() and apply:
            _atomic_bytes(target, source.read_bytes())


def canonicalize(
    artifact_root: Path,
    mirror_root: Path,
    *,
    apply: bool,
) -> MigrationReport:
    roots = sorted(
        path.parent.parent
        for path in artifact_root.glob("**/config/condition.json")
        if _is_word2vec_root(path.parent.parent)
        and (path.parent.parent / "checkpoints/generations").is_dir()
    )
    generations = 0
    moved = 0
    renamed = 0
    updated_config_digests = 0
    manifests = 0
    mirrors = 0
    for root in roots:
        config_digest = _current_config_digest(root)
        checkpoint_generations = sorted(
            path
            for path in (root / "checkpoints/generations").iterdir()
            if path.is_dir()
        )
        if not checkpoint_generations:
            continue
        for generation in checkpoint_generations:
            did_move, did_rename, did_update_digest = _canonicalize_generation(
                generation,
                config_digest=config_digest,
                apply=apply,
            )
            generations += 1
            moved += int(did_move)
            renamed += int(did_rename)
            updated_config_digests += int(did_update_digest)
        latest_pointer = _read_json(root / "checkpoints/latest.json")
        latest = root / "checkpoints" / str(latest_pointer["path"])
        manifests += int(
            _update_parameter_manifest(root, latest, apply=apply)
        )
        if apply:
            _update_checkpoint_metadata(root, apply=True)
        resolved = _read_json(root / "config/resolved.json")
        run_key = resolved.get("run_key")
        mirror = mirror_root / str(run_key)
        if run_key and mirror.is_dir():
            _copy_canonical_metadata(root, mirror, apply=apply)
            mirrors += 1
    return MigrationReport(
        artifact_roots=len(roots),
        checkpoint_generations=generations,
        moved_output_weights=moved,
        renamed_optimizer_states=renamed,
        updated_config_digests=updated_config_digests,
        parameter_manifests=manifests,
        mirrors=mirrors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("infra/mlflow/data/artifacts"),
    )
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=Path("exp/ds2/results/mlflow_artifacts"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    report = canonicalize(
        args.artifact_root,
        args.mirror_root,
        apply=args.apply,
    )
    payload = {
        "applied": args.apply,
        **report.__dict__,
    }
    print(json.dumps(payload, sort_keys=True))
    if args.apply:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        report_path = (
            Path("infra/mlflow/data/maintenance-reports")
            / f"{timestamp}-word2vec-checkpoint-canonicalization.json"
        )
        _write_json(report_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
