"""Explicit preparation of corpus and benchmark resources, outside analysis."""

from __future__ import annotations

import glob
import hashlib
import subprocess
import urllib.request
from pathlib import Path

from repro_io.checksum import sha256_file
from repro_io.http import SerialDownloader

from f2.common.paths import get_benchmark_data_dir, get_corpus_data_dir
from f2.suites.w2v.evaluation import parse_analogy_questions
from repro_core.context import RuntimePaths

QUESTIONS_WORDS_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/20c129af10659f7c50e86e3be406df663beff438/questions-words.txt"


QUESTIONS_WORDS_SHA256 = (
    "8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36"
)


def ensure_questions_words(target_path: Path) -> Path:
    target = Path(target_path)
    if target.is_file():
        return target
    if target.name != "questions-words.txt":
        raise ValueError(f"word analogy questions do not exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    try:
        with urllib.request.urlopen(QUESTIONS_WORDS_URL) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != QUESTIONS_WORDS_SHA256:
            raise ValueError(
                f"downloaded questions-words.txt checksum mismatch: {digest} != {QUESTIONS_WORDS_SHA256}"
            )
        target.write_bytes(data)
        return target
    except Exception as err:
        raise ValueError(f"word analogy questions do not exist: {target}") from err


MSR_SYNTACTIC_URL = "https://raw.githubusercontent.com/ResponsiblyAI/responsibly/715c13ff7cf19de9d42f92d2e8c4de697dd4c638/responsibly/we/data/benchmark/MSR-syntax.txt"


MSR_SYNTACTIC_SHA256 = (
    "ac90a8f492d19fd892cc0e27c5f63dcd9a520b0044b91acd99e521668cc1ebc5"
)


def ensure_msr_syntactic(target: Path) -> Path:
    """Verify a pinned MSR mirror; this is not the Sentence Completion dataset."""
    target = Path(target)
    if not target.is_file():
        if target.name != "MSR-syntax.txt":
            raise ValueError(f"MSR syntactic analogy questions do not exist: {target}")
        SerialDownloader().download(
            MSR_SYNTACTIC_URL,
            target,
            expected_sha256=MSR_SYNTACTIC_SHA256,
            expected_length=231634,
        )
    if sha256_file(target) != MSR_SYNTACTIC_SHA256:
        raise ValueError(f"MSR syntactic analogy checksum mismatch: {target}")
    questions = parse_analogy_questions(target.read_bytes().splitlines())
    if len(questions) != 8000 or {question.category for question in questions} != {
        "all"
    }:
        raise ValueError(
            "MSR syntactic mirror must contain 8,000 questions in its all section"
        )
    return target


MSR_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/XingxingZhang/td-treelstm/"
    "5f11c4f4e22362a476b9900f87e65f238b891dcb/msr_scripts/Holmes.lm_format.questions.txt"
)


MSR_ANSWERS_URL = (
    "https://raw.githubusercontent.com/XingxingZhang/td-treelstm/"
    "5f11c4f4e22362a476b9900f87e65f238b891dcb/msr_scripts/Holmes.lm_format.answers.txt"
)


MSR_CSV_URL = (
    "https://raw.githubusercontent.com/pochih/Sentence-Completion/"
    "346ff7f4106a452dc92d6e1f35fe1f0da854c800/data/testing_data.csv"
)


MSR_QUESTIONS_SHA256 = (
    "659b752b219e64b748f54333d0019a2878645ae430c8fb9d3254c2e561d220dc"
)


MSR_ANSWERS_SHA256 = "c9dec1e8c9168cb67260dad36ebd2041d12776c0f36cbfc295d9b8f08a699fad"


MSR_CSV_SHA256 = "f3a76019be32c5714aec7fef9763f87d7587d23421e38156c8bb16f4e40bfe0e"


HOLMES_REPO_URL = "https://github.com/pochih/Sentence-Completion.git"


HOLMES_REPO_COMMIT = "346ff7f4106a452dc92d6e1f35fe1f0da854c800"


HOLMES_CORPUS_SHA256 = (
    "2d37eb27d9de873da4f6419740230e23cb9f75b91cc67b456d1e1a3fc34b680f"
)


HOLMES_RAW_BOOKS_COUNT = 522


HOLMES_RAW_BYTES = 236110188


def _download_verified(url: str, expected_sha256: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == expected_sha256:
            return target

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"downloaded file from {url} checksum mismatch: {digest} != {expected_sha256}"
        )
    target.write_bytes(data)
    return target


def ensure_msr_benchmark_data(
    benchmark_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Ensure the canonical 1,040 questions and gold answers exist with verified checksums."""
    paths = RuntimePaths.from_environment()
    base_dir = (
        Path(benchmark_dir)
        if benchmark_dir
        else get_benchmark_data_dir(paths) / "msr_sentence_completion"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    q_path = base_dir / "Holmes.lm_format.questions.txt"
    a_path = base_dir / "Holmes.lm_format.answers.txt"

    _download_verified(MSR_QUESTIONS_URL, MSR_QUESTIONS_SHA256, q_path)
    _download_verified(MSR_ANSWERS_URL, MSR_ANSWERS_SHA256, a_path)

    return q_path, a_path


def ensure_holmes_corpus(corpus_path: Path | None = None) -> Path:
    """Ensure the verified 522-book Holmes training corpus exists at corpus_path."""
    paths = RuntimePaths.from_environment()
    target = (
        Path(corpus_path) if corpus_path else get_corpus_data_dir(paths) / "holmes.txt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        h = hashlib.sha256()
        with target.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        if h.hexdigest() == HOLMES_CORPUS_SHA256:
            return target

    # Check cache directory for repo clone
    cache_dir = paths.staging_root / "f2" / "pochih_repo"
    data_dir = cache_dir / "data" / "Holmes_Training_Data"
    if not data_dir.is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                HOLMES_REPO_URL,
                str(cache_dir),
            ],
            check=True,
            capture_output=True,
        )

    book_files = sorted(glob.glob(str(data_dir / "*.TXT")))
    if len(book_files) != HOLMES_RAW_BOOKS_COUNT:
        raise ValueError(
            f"expected {HOLMES_RAW_BOOKS_COUNT} Gutenberg book files; found {len(book_files)}"
        )

    h = hashlib.sha256()
    with target.open("wb") as out:
        for fpath in book_files:
            with open(fpath, "rb") as inp:
                chunk = inp.read()
                out.write(chunk)
                h.update(chunk)

    digest = h.hexdigest()
    if digest != HOLMES_CORPUS_SHA256:
        raise ValueError(
            f"concatenated Holmes corpus checksum mismatch: {digest} != {HOLMES_CORPUS_SHA256}"
        )

    return target
