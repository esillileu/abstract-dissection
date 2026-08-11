"""Original ch05 small-corpus SimpleRnnlm."""

from pathlib import Path

from exp.original.measurement import OriginalMeasurements
from exp.original.runtime_context import array_module, budget, master_seed

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, install_b2_compatibility_aliases, np, save_csv, save_npz, source_imports
from .common import to_device


def run(worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree, gpu=True):
        cp = array_module()
        ptb = importlib.import_module("dataset.ptb")
        util = importlib.import_module("common.util")
        install_b2_compatibility_aliases()
        model_cls = importlib.import_module("ch05.simple_rnnlm").SimpleRnnlm
        trainer_cls = importlib.import_module("common.trainer").RnnlmTrainer
        optimizer_cls = importlib.import_module("common.optimizer").SGD
        corpus, _, _ = ptb.load_data("train")
        corpus = to_device(util, corpus[:1000])
        np.random.seed(master_seed())
        cp.random.seed(master_seed())
        model = model_cls(int(cp.max(corpus).item()) + 1, 100, 100)
        trainer = trainer_cls(model, optimizer_cls(0.1))
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.fit(corpus[:-1], corpus[1:], budget("max_epochs", 100), 10, 5)
        measurements.save(model.params)
        rows = [
            {"plot_index": index, "perplexity": value, "eval_interval": 20}
            for index, value in enumerate(trainer.ppl_list)
        ]
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = (
    Trial(
        "dlfs2.ch05.ptb-small-rnnlm",
        "cupy",
        {"epochs": 100, "batch_size": 10, "time_size": 5, "learning_rate": 0.1},
        COMMON_SOURCES + ("ch05/simple_rnnlm.py", "ch05/train.py"),
        run,
    ),
)
