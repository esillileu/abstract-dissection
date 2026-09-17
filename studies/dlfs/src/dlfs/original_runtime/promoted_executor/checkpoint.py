from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def _promote_final_checkpoint(
    config: dict[str, object], context, output: Path, final: dict[str, float]
) -> Path | None:
    """Publish an original parameter archive through the canonical checkpoint API."""
    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, dict) or not checkpoint.get("save_final", False):
        return None
    source = output / "checkpoint.npz"
    if not source.is_file():
        raise ValueError("checkpoint.save_final requires raw/checkpoint.npz")
    root = Path(str(context.metadata["checkpoint_root"]))
    root.mkdir(parents=True, exist_ok=True)
    target = root / "final.npz"
    with tempfile.NamedTemporaryFile(dir=root, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        import numpy as np

        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        # Original Word2Vec archives call the analysis-ready embedding matrix
        # ``word_vectors`` and retain the exact upstream parameter sequence as
        # ``param_###``.  Keep both and expose the canonical model key.
        if "W_in" not in arrays and "word_vectors" in arrays:
            arrays["W_in"] = arrays["word_vectors"]
        np.savez(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    pointer = {
        "schema_version": 2,
        "role": "latest",
        "path": target.name,
        "sha256": digest,
        "epoch": int(final.get("final/system/completed_epochs", 0)),
        "update": int(final.get("final/system/total_updates", 0)),
    }
    temporary_pointer = root / ".latest.json.tmp"
    temporary_pointer.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_pointer, root / "latest.json")
    return target


def _dependency_root(config: dict[str, object], output: Path) -> Path:
    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, dict) or not checkpoint.get("source_path"):
        return output.parent
    source = Path(str(checkpoint["source_path"]))
    root = output / "dependency"
    target = root / "data" / "e07" / "dlfs2.ch08.date.attention-seq2seq-reverse"
    target.mkdir(parents=True, exist_ok=True)
    candidate = source / "checkpoint.npz" if source.is_dir() else source
    if not candidate.is_file():
        raise ValueError(f"e08 source artifact is missing checkpoint.npz: {source}")
    # The old e08 adapter only requires the archive and a valid-cache marker is
    # intentionally bypassed by its promoted-domain branch.
    (target / "SOURCE_PATH").write_text(str(candidate), encoding="utf-8")
    return root


def _checkpoint_parameters(root: Path):
    path = root / "checkpoint.npz"
    if not path.is_file():
        return ()
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return tuple(
            (name.removeprefix("param__"), archive[name].copy())
            for name in archive.files
            if name.startswith("param_")
        )
