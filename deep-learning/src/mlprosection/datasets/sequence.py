# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy


id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name='addition.txt', seed=1984, *, split_algorithm="default_rng"):
    global id_to_char, char_to_id
    id_to_char, char_to_id = {}, {}
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions, answers = [], []

    for line in open(file_path, 'r'):
        idx = line.find('_')
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    # 어휘 사전 생성
    for i in range(len(questions)):
        q, a = questions[i], answers[i]
        _update_vocab(q)
        _update_vocab(a)

    # 넘파이 배열 생성
    x = numpy.zeros((len(questions), len(questions[0])), dtype=numpy.int32)
    t = numpy.zeros((len(questions), len(answers[0])), dtype=numpy.int32)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # 뒤섞기
    indices = numpy.arange(len(x))
    if split_algorithm == "default_rng":
        numpy.random.default_rng(seed).shuffle(indices)
    elif split_algorithm == "legacy_numpy_randomstate":
        # The book uses ``np.random.seed(seed); np.random.shuffle(indices)``.
        # A private RandomState preserves that MT19937 permutation without
        # mutating the process-global RNG used for model initialization.
        numpy.random.RandomState(seed).shuffle(indices)
    else:
        raise ValueError(f"unknown sequence split algorithm: {split_algorithm}")
    x = x[indices]
    t = t[indices]

    # 검증 데이터셋으로 10% 할당
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char


def load_sequence(
    file_name: str,
    *,
    seed: int,
    split_algorithm: str = "default_rng",
):
    """Return a deterministic character-level split and its vocabulary."""
    (x_train, t_train), (x_test, t_test) = load_data(
        file_name,
        seed=seed,
        split_algorithm=split_algorithm,
    )
    char_to_id_value, id_to_char_value = get_vocab()
    return {
        "train": (x_train, t_train), "test": (x_test, t_test),
        "char_to_id": dict(char_to_id_value), "id_to_char": dict(id_to_char_value),
    }
