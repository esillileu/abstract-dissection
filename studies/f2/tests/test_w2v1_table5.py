from __future__ import annotations

from pathlib import Path

from f2.catalog.manifest import read_manifest
from f2.definition import DEFINITION
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner


def test_table5_plan_and_catalog_slots() -> None:
    definition = DEFINITION.get_suite("w2v1")
    plans = Planner(definition).build(
        RunSelection(experiment_ids=("04",)), RunOptions()
    )
    assert len(plans) == 18
    catalog = read_manifest(Path("studies/f2/catalog/w2v.json"))
    slots = {slot["planned_run_slot_id"]: slot for slot in catalog["planned_run_slots"]}
    for plan in plans:
        spec = definition.load_run_spec(
            plan.path, atomic_run_id=plan.atomic_run_id, overrides={}
        ).with_seed(plan.seed)
        corpus, condition = plan.atomic_run_id.split("--")
        architecture = condition.split("-", 1)[0]
        config = spec.to_executor_config()
        identity = config["identity"]
        assert identity["planned_run_slot_id"] in slots
        slot = slots[identity["planned_run_slot_id"]]
        assert slot["seed"] == plan.seed
        assert slot["atomic_run_id"] == condition
        assert slot["plan_experiment_id"].endswith(identity["experiment_spec_id"])
        assert identity["study"] == "table5"
        assert identity["execution_plan_id"] == f"w2v1-table5-{corpus}-r1"
        assert config["vocabulary"]["max_lexical_words"] == 1_000_000
        training = config["training"]
        assert training["model_kind"] == (
            "cbow" if architecture == "cbow" else "skip_gram"
        )
        assert training["window_radius"] == (4 if architecture == "cbow" else 10)
        assert training["context_policy"] == (
            "fixed" if architecture == "cbow" else "dynamic"
        )
        assert training["epochs"] == 1
        assert training["objective_kind"] == "hierarchical_softmax"
        assert training["initial_learning_rate"] == 0.025
        assert training["subsampling_threshold"] == 0.0


def test_table5_smoke_execution(tmp_path: Path) -> None:
    import csv

    from f2.suites.w2v.artifacts import load_checkpoint, load_lookup_artifact
    from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
    from repro_core.context import ExperimentContext, RuntimePaths
    from repro_core.execution.runner import run_config

    definition = DEFINITION.get_suite("w2v1")
    fixture_corpus = Path(__file__).parent / "fixtures/w2v1-corpus.txt"
    fixture_questions = Path(__file__).parent / "fixtures/questions-words.txt"
    questions = parse_analogy_questions(fixture_questions.read_bytes().splitlines())

    root = tmp_path / "repo"
    root.mkdir()
    paths = RuntimePaths(
        repo_root=Path(__file__).resolve().parents[3],
        data_root=root / "data",
        artifacts_root=root / "artifacts",
        cache_root=root / "cache",
        staging_root=root / "staging",
        references_root=root / "references",
        studies_root=root / "studies",
    )

    for atomic_run_id, dim, _model in [
        ("wmt--cbow-d300-w783m-ep1", 300, "cbow"),
        ("wmt--skipgram-d600-w783m-ep1", 600, "skip_gram"),
    ]:
        spec = definition.load_run_spec(
            "studies/f2/src/f2/suites/w2v1/config/e04_table5.yaml",
            atomic_run_id=atomic_run_id,
            overrides={
                "corpus": {"path": str(fixture_corpus)},
                "vocabulary": {
                    "initial_capacity": 16,
                    "hash_capacity": 128,
                    "min_count": 1,
                    "max_lexical_words": 1000,
                },
                "training": {
                    "embedding_dimension": dim,
                    "epochs": 1,
                    "thread_count": 1,
                    "learning_rate_update_interval": 5,
                    "observation_interval": 1,
                },
            },
        ).with_seed(1)

        result = run_config(
            spec.to_executor_config(),
            ExperimentContext(paths=paths),
            executor_module=definition.executor_module,
        )

        ckpt = load_checkpoint(result.checkpoint)
        assert ckpt.completed_epochs == 1

        obs_file = result.root / "metrics/observations.csv"
        with open(obs_file, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lrs = [float(r["learning_rate"]) for r in rows]
        assert lrs[0] <= 0.025
        assert lrs[-1] < lrs[0]

        lookup = load_lookup_artifact(result.root / "lookup")
        assert lookup.embeddings.shape[1] == dim
        eval_res = evaluate_analogies(
            lookup, questions, vocabulary_limit=None, batch_size=16
        )
        assert eval_res.overall.valid_count > 0
