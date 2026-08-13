"""Original ch06 BetterRnnlm recipe."""

from pathlib import Path

from exp.deepscratch.original_runtime.measurement import OriginalMeasurements
from exp.deepscratch.original_runtime.runtime_context import (
    array_module,
    budget,
    master_seed,
    reset_runtime,
    set_runtime,
)

from .common import (
    COMMON_SOURCES,
    Trial,
    checkpoint_arrays,
    importlib,
    np,
    save_csv,
    save_npz,
    source_imports,
    to_device,
)


def evaluate_pretrained(
    worktree: Path,
    checkpoint: Path,
    *,
    selected_device: str = "cuda:0",
    batch_size: int = 10,
    time_size: int = 35,
) -> float:
    """Evaluate an upstream BetterRnnlm pickle on the complete PTB test split."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BetterRnnlm checkpoint not found: {checkpoint}")
    if batch_size <= 0 or time_size <= 0:
        raise ValueError("batch_size and time_size must be positive")

    tokens = set_runtime(seed=1, selected_device=selected_device, config={})
    try:
        with source_imports(worktree, gpu=True):
            ptb = importlib.import_module("dataset.ptb")
            util = importlib.import_module("common.util")
            model_cls = importlib.import_module("ch06.better_rnnlm").BetterRnnlm

            corpus_test, word_to_id, _ = ptb.load_data("test")
            corpus_test = to_device(util, corpus_test)
            model = model_cls(len(word_to_id), 650, 650, 0.5)
            model.load_params(str(checkpoint))
            model.reset_state()
            return float(
                util.eval_perplexity(
                    model,
                    corpus_test,
                    batch_size=batch_size,
                    time_size=time_size,
                )
            )
    finally:
        reset_runtime(tokens)


def run(worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree, gpu=True):
        cp = array_module()
        ptb = importlib.import_module("dataset.ptb")
        util = importlib.import_module("common.util")
        model_cls = importlib.import_module("ch06.better_rnnlm").BetterRnnlm
        trainer_cls = importlib.import_module("common.trainer").RnnlmTrainer
        optimizer_cls = importlib.import_module("common.optimizer").SGD

        corpus, word_to_id, _ = ptb.load_data("train")
        corpus_valid, _, _ = ptb.load_data("val")
        corpus_test, _, _ = ptb.load_data("test")
        np.random.seed(master_seed())
        cp.random.seed(master_seed())
        corpus = to_device(util, corpus)
        corpus_valid = to_device(util, corpus_valid)
        corpus_test = to_device(util, corpus_test)

        model = model_cls(len(word_to_id), 650, 650, 0.5)
        optimizer = optimizer_cls(20.0)
        trainer = trainer_cls(model, optimizer)
        measurements = OriginalMeasurements(output)
        rows = []
        best_valid = float("inf")
        best_params = None
        epochs = budget("max_epochs", 40)

        for epoch in range(epochs):
            with measurements.training():
                trainer.fit(
                    corpus[:-1],
                    corpus[1:],
                    1,
                    20,
                    35,
                    0.25,
                    eval_interval=20,
                )
            model.reset_state()
            valid_ppl = float(util.eval_perplexity(model, corpus_valid))
            improved = valid_ppl < best_valid
            if improved:
                best_valid = valid_ppl
                best_params = checkpoint_arrays(model.params)
            else:
                optimizer.lr /= 4.0
            rows.append(
                {
                    "epoch": epoch + 1,
                    "phase": "epoch",
                    "split": "valid",
                    "perplexity": valid_ppl,
                    "learning_rate": float(optimizer.lr),
                    "is_best": improved,
                }
            )
            model.reset_state()

        model.reset_state()
        test_ppl = float(util.eval_perplexity(model, corpus_test))
        rows.append(
            {
                "epoch": epochs + 1,
                "phase": "terminal",
                "split": "test",
                "perplexity": test_ppl,
                "learning_rate": float(optimizer.lr),
                "is_best": False,
            }
        )
        measurements.save(model.params)
        params = best_params or checkpoint_arrays(model.params)

    save_csv(output / "metrics.csv", rows)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = (
    Trial(
        "dlfs2.ch06.ptb-better-rnnlm",
        "cupy",
        {
            "model": "BetterRnnlm",
            "epochs": 40,
            "wordvec_size": 650,
            "hidden_size": 650,
            "dropout": 0.5,
            "batch_size": 20,
            "time_size": 35,
            "learning_rate": 20.0,
            "max_grad": 0.25,
            "validation_decay": 4.0,
        },
        COMMON_SOURCES
        + ("ch06/better_rnnlm.py", "ch06/train_better_rnnlm.py"),
        run,
    ),
)
