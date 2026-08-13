from __future__ import annotations

import json
import pickle

import numpy as np

from exp.deepscratch.ds2.implemented.checkpoint_migration import canonicalize
from mlprosection.nn.model.architecture import CBOW
from mlprosection.nn.objective import SoftmaxWithLoss
from mlprosection.optim.SGD import Adam


def test_canonicalize_moves_word2vec_output_parameters(tmp_path):
    artifact_root = tmp_path / "artifacts"
    root = artifact_root / "10/run/artifacts"
    generation = root / "checkpoints/generations/latest-epoch-0001"
    generation.mkdir(parents=True)
    (root / "config").mkdir()
    (root / "model").mkdir()
    np.savez(generation / "model_parameters.npz", W_in=np.ones((3, 2)))
    np.savez(generation / "objective_parameters.npz", W_out=np.full((3, 2), 2.0))
    with (generation / "optimizer_state.pkl").open("wb") as file:
        pickle.dump(
            {
                "type": "Adam",
                "m": {
                    "model.W_in": np.ones((3, 2)),
                    "objective.W_out": np.full((3, 2), 3.0),
                },
            },
            file,
        )
    (generation / "manifest.json").write_text(
        json.dumps({"schema_version": 2}),
        encoding="utf-8",
    )
    (root / "config/condition.json").write_text(
        json.dumps({"atomic_run_id": "W2V-TOY-CBOW-FULL"}),
        encoding="utf-8",
    )
    (root / "config/resolved.json").write_text(
        json.dumps({"run_key": "key"}),
        encoding="utf-8",
    )
    (root / "model/parameter_manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": "W_in",
                    "numel": 6,
                }
            ]
        ),
        encoding="utf-8",
    )
    pointer = {
        "schema_version": 2,
        "role": "latest",
        "path": "generations/latest-epoch-0001",
        "sha256": "old",
    }
    (root / "checkpoints/latest.json").write_text(
        json.dumps(pointer),
        encoding="utf-8",
    )
    (root / "checkpoints/checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "latest": {"digest": "old"},
                "final": {"digest": "old"},
                "best": None,
            }
        ),
        encoding="utf-8",
    )
    (root / "checkpoints.csv").write_text(
        "kind,sha256\nlatest,old\n",
        encoding="utf-8",
    )

    report = canonicalize(
        artifact_root,
        tmp_path / "mirrors",
        apply=True,
    )

    assert report.moved_output_weights == 1
    with np.load(generation / "model_parameters.npz") as model:
        assert model.files == ["W_in", "W_out"]
        np.testing.assert_array_equal(model["W_out"], np.full((3, 2), 2.0))
    with np.load(generation / "objective_parameters.npz") as objective:
        assert objective.files == []
    with (generation / "optimizer_state.pkl").open("rb") as file:
        state = pickle.load(file)
    assert "model.W_out" in state["m"]
    assert "objective.W_out" not in state["m"]
    model = CBOW(3, 2, backend="cpu")
    objective = SoftmaxWithLoss(backend="cpu")
    model.load_params_npz(generation / "model_parameters.npz")
    objective.load_params_npz(generation / "objective_parameters.npz")
    optimizer = Adam(
        [
            (f"model.{name}", parameter)
            for name, parameter in model.named_parameters()
        ]
    )
    optimizer.load_state_dict(state)
    np.testing.assert_array_equal(
        optimizer.m["model.W_out"],
        np.full((3, 2), 3.0),
    )
    manifest = json.loads(
        (root / "model/parameter_manifest.json").read_text(encoding="utf-8")
    )
    assert [(entry["name"], entry["numel"]) for entry in manifest] == [
        ("W_in", 6),
        ("W_out", 6),
    ]
    pointer = json.loads(
        (root / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    assert pointer["sha256"] != "old"


def test_canonicalize_dry_run_does_not_change_files(tmp_path):
    artifact_root = tmp_path / "artifacts"
    root = artifact_root / "10/run/artifacts"
    generation = root / "checkpoints/generations/latest-epoch-0001"
    generation.mkdir(parents=True)
    (root / "config").mkdir()
    (root / "model").mkdir()
    np.savez(generation / "model_parameters.npz", W_in=np.ones((2, 1)))
    np.savez(generation / "objective_parameters.npz", W_out=np.ones((2, 1)))
    with (generation / "optimizer_state.pkl").open("wb") as file:
        pickle.dump({"type": "SGD"}, file)
    (root / "config/condition.json").write_text(
        json.dumps({"atomic_run_id": "W2V-X"}),
        encoding="utf-8",
    )
    (root / "config/resolved.json").write_text(
        json.dumps({}),
        encoding="utf-8",
    )
    (root / "model/parameter_manifest.json").write_text(
        json.dumps([{"name": "W_in"}]),
        encoding="utf-8",
    )
    (root / "checkpoints/latest.json").write_text(
        json.dumps({"path": "generations/latest-epoch-0001"}),
        encoding="utf-8",
    )
    before = (generation / "model_parameters.npz").read_bytes()

    report = canonicalize(
        artifact_root,
        tmp_path / "mirrors",
        apply=False,
    )

    assert report.moved_output_weights == 1
    assert (generation / "model_parameters.npz").read_bytes() == before
