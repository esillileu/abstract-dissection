from __future__ import annotations

from deepscratch.core import Tensor

from ..records import DS2Records


def _decode_ids(values, id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char[int(value)] for value in values)


def _record_seq_predictions(
    records: DS2Records,
    model,
    questions,
    answers,
    char_to_id,
    id_to_char,
    backend,
    recording: dict[str, object],
    *,
    epoch: int,
    predictions=None,
) -> None:
    config = recording.get("predictions")
    if not isinstance(config, dict):
        return
    if str(config.get("split", "test")) != "test":
        raise ValueError("seq2seq predictions currently support split: test")
    count = min(int(config.get("count", 10)), len(questions))
    start_id = char_to_id["_"]
    was_training = bool(getattr(model, "training", True))
    model.train(False)
    try:
        for example_id in range(count):
            expected = [int(value) for value in answers[example_id][1:]]
            if predictions is None:
                question = Tensor(
                    backend.xp.asarray(
                        questions[example_id : example_id + 1],
                        dtype=backend.xp.int64,
                    ),
                    backend=backend,
                )
                predicted = model.generate(question, start_id, len(expected))
            else:
                predicted = [int(value) for value in predictions[example_id]]
            records.add_prediction(
                {
                    "epoch": epoch,
                    "example_id": example_id,
                    "source": _decode_ids(questions[example_id], id_to_char),
                    "target": _decode_ids(expected, id_to_char),
                    "prediction": _decode_ids(predicted, id_to_char),
                    "exact_match": int(predicted == expected),
                    "token_correct": sum(
                        left == right
                        for left, right in zip(predicted, expected, strict=True)
                    ),
                    "token_count": len(expected),
                }
            )
    finally:
        model.train(was_training)
