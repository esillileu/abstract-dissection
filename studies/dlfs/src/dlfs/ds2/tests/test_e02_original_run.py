import numpy as np

from dlfs.ds2.original.run import e02
from dlfs.ds2.original.run.common import source_imports
from repro_core.context.paths import RuntimePaths

BOOK_ROOT = (
    RuntimePaths.from_environment().reference("dlfs2-book") / "source"
).resolve()


def test_original_e02_registers_embedding_and_onehot_full_softmax_trials() -> None:
    assert [trial.trial_id for trial in e02.TRIALS] == [
        "dlfs2.ch04.ptb-cbow-negative-sampling",
        "ext.ds2.ptb-cbow-full-softmax",
        "ext.ds2.ptb-cbow-onehot-full-softmax",
        "dlfs2.ch04.ptb-skipgram-negative-sampling",
        "ext.ds2.ptb-skipgram-full-softmax",
        "ext.ds2.ptb-skipgram-onehot-full-softmax",
    ]
    cbow_ns = e02.TRIALS[0]
    assert cbow_ns.conditions == {
        "model": "CBOW",
        "epochs": 10,
        "batch_size": 100,
        "window": 5,
    }
    assert cbow_ns.source_files == (
        *e02.COMMON_SOURCES,
        "ch04/cbow.py",
        "ch04/skip_gram.py",
        "ch04/negative_sampling_layer.py",
        "ch04/train.py",
    )
    assert e02.TRIALS[2].conditions["input_representation"] == "one_hot"
    assert e02.TRIALS[5].conditions["input_representation"] == "one_hot"


def test_original_e02_full_softmax_adaptations_forward_and_backward() -> None:
    contexts = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    target = np.array([2, 3, 4], dtype=np.int32)

    with source_imports(BOOK_ROOT):
        for kind in ("cbow", "skipgram"):
            for one_hot in (False, True):
                model = e02.build_full_softmax_model(
                    kind,
                    5,
                    3,
                    1,
                    one_hot=one_hot,
                )
                loss = model.forward(contexts, target)
                model.backward()

                assert np.isfinite(loss)
                assert model.word_vecs.shape == (5, 3)
                assert model.params[-1].shape == (3, 5)
                assert all(
                    param.shape == grad.shape
                    for param, grad in zip(
                        model.params,
                        model.grads,
                        strict=True,
                    )
                )
