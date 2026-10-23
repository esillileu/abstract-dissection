"""Offline corpus-specific and paired Table 3 analysis of saved evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from repro_core.context import RuntimePaths

from .corpus_investigation import CORPORA, csv_rows, truth
from .corpus_report import FEATURES, table
from .diagnostic import SEEDS
from .relations import write_csv

OUTCOMES = ("cbow_accuracy", "skipgram_accuracy", "gap")
LOG_FEATURES = {
    "context_diversity_per_occurrence",
    "shard_rate_cv",
    "direct_per_occurrence",
}


def key(row):
    return row["category"], row["word_a"].lower(), row["word_b"].lower()


def correlation(x, y, controls, weights):
    """Weighted partial correlations, residualizing all columns together."""
    root = np.sqrt(weights)[:, None]
    design = controls * root
    values = np.column_stack((x, y)) * root
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    a, b = residual[:, : x.shape[1]], residual[:, x.shape[1] :]
    denominator = np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)[None, :])
    return np.divide(
        a.T @ b,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-10,
    )


def percentile_interval(values):
    """Leave unidentified coefficients missing instead of reporting a zero effect."""
    flat = values.reshape(len(values), -1)
    result = np.full((2, flat.shape[1]), np.nan)
    valid = np.isfinite(flat).any(axis=0)
    result[:, valid] = np.nanquantile(flat[:, valid], (0.025, 0.975), axis=0)
    return result.reshape((2, *values.shape[1:]))


def features(rows):
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    for j, feature in enumerate(FEATURES):
        if feature in LOG_FEATURES:
            x[:, j] = np.log1p(x[:, j])
    return x


def nuisance(rows):
    logf = np.log1p(np.array([float(r["min_frequency"]) for r in rows]))
    logf = (logf - logf.mean()) / max(logf.std(), 1e-12)
    imbalance = np.array(
        [
            abs(np.log((float(r["frequency_a"]) + 1) / (float(r["frequency_b"]) + 1)))
            for r in rows
        ]
    )
    categories = sorted({r["category"] for r in rows})
    return np.column_stack(
        (
            np.ones(len(rows)),
            logf,
            logf**2,
            imbalance,
            *[np.array([r["category"] == c for r in rows]) for c in categories[1:]],
        )
    )


def load_questions(cache):
    """Keep all seed outcomes and the actual source and target word identities."""
    questions = {}
    for r in csv_rows(cache / "question_outcomes.csv"):
        if not truth(r["included"]):
            continue
        qid = int(r["question_id"])
        q = questions.setdefault(
            qid,
            dict(
                category=r["category"],
                words=tuple(r[f"word_{c}"].lower() for c in "abcd"),
                common=truth(r["common_subset"]),
                scores=np.full((2, 3), np.nan),
            ),
        )
        q["scores"][int(r["model"] == "skipgram"), SEEDS.index(int(r["seed"]))] = (
            truth(r["correct"]) * 100
        )
    if any(not np.isfinite(q["scores"]).all() for q in questions.values()):
        raise ValueError("Missing model/seed question outcomes")
    return questions


def resamples(rows, questions, rng, repetitions):
    """Shared word/pair multipliers on questions; aggregate to unique target pairs.

    Equal target-pair weighting is the estimand. Question weights perturb both
    target-pair mass and its conditional success, preserving source dependence.
    Multipliers are shared across categories and corpora; seeds are paired.
    """
    positions = {key(r): i for i, r in enumerate(rows)}
    selected = [
        (qid, q)
        for qid, q in sorted(questions.items())
        if (q["category"], *q["words"][2:]) in positions
    ]
    target = np.array([positions[q["category"], *q["words"][2:]] for _, q in selected])
    words = sorted({w for _, q in selected for w in q["words"]})
    word_ids = {w: i for i, w in enumerate(words)}
    pairs = sorted(
        {q["words"][start : start + 2] for _, q in selected for start in (0, 2)}
    )
    pair_ids = {p: i for i, p in enumerate(pairs)}
    wi = [[word_ids[w] for w in sorted(set(q["words"]))] for _, q in selected]
    pi = [
        [pair_ids[p] for p in sorted({q["words"][:2], q["words"][2:]})]
        for _, q in selected
    ]
    # Padding uses an extra multiplier fixed at one (repeated words count once).
    windex = np.full((len(selected), 4), len(words), dtype=int)
    pindex = np.full((len(selected), 2), len(pairs), dtype=int)
    for i, (w, p) in enumerate(zip(wi, pi, strict=True)):
        windex[i, : len(w)] = w
        pindex[i, : len(p)] = p
    count = np.bincount(target, minlength=len(rows))
    if (count == 0).any():
        raise ValueError("Feature pairs missing question outcomes")
    scores = np.array([q["scores"] for _, q in selected])

    def aggregate(weight, seeds):
        mass = np.bincount(target, weights=weight, minlength=len(rows))
        y = np.column_stack(
            [
                np.bincount(
                    target,
                    weights=weight * scores[:, m, seeds].mean(axis=1),
                    minlength=len(rows),
                )
                / mass
                for m in (0, 1)
            ]
        )
        return np.column_stack((y, y[:, 1] - y[:, 0])), mass / count

    yield aggregate(np.ones(len(selected)), np.arange(3))
    for _ in range(repetitions):
        wm = np.r_[rng.exponential(size=len(words)), 1]
        pm = np.r_[rng.exponential(size=len(pairs)), 1]
        weight = wm[windex].prod(1) * pm[pindex].prod(1)
        yield aggregate(weight, rng.integers(0, 3, 3))


def analyze(label, rows, questions, repetitions, other=None):
    """Analyze all categories independently and pooled, without outcome screening."""
    rng = np.random.default_rng(20261011)
    x = features(rows)
    controls = nuisance(rows)
    if other is not None:
        x = features(other) - x
        # Baseline and changes in frequency/imbalance are both nuisance variables.
        controls = np.column_stack(
            (controls, nuisance(other)[:, 1:4] - controls[:, 1:4])
        )
    scopes = {"overall": np.arange(len(rows))}
    scopes.update(
        {
            c: np.array([i for i, r in enumerate(rows) if r["category"] == c])
            for c in sorted({r["category"] for r in rows})
        }
    )
    scopes = {c: ids for c, ids in scopes.items() if len(ids) >= 10}
    estimates = defaultdict(list)
    means = defaultdict(list)
    samples = resamples(rows, questions, rng, repetitions)
    other_samples = (
        resamples(other, questions.other, np.random.default_rng(20261011), repetitions)
        if other is not None
        else None
    )
    point_y = None
    for y, weight in samples:
        if other_samples is not None:
            y = next(other_samples)[0] - y
        if point_y is None:
            point_y = y
        for scope, ids in scopes.items():
            for adjustment in (
                "unadjusted",
                "frequency_category",
                "exposure_sensitivity",
            ):
                c = (
                    np.ones((len(ids), 1))
                    if adjustment == "unadjusted"
                    else controls[ids]
                )
                if adjustment == "exposure_sensitivity":
                    # Context amount and spelling coverage can confound relation proxies.
                    extra = np.array(
                        [
                            [
                                np.log1p(float(rows[i]["context_mass_a"])),
                                np.log1p(float(rows[i]["context_mass_b"])),
                                float(rows[i]["exact_spelling_share"]),
                            ]
                            for i in ids
                        ]
                    )
                    if other is not None:
                        delta = (
                            np.array(
                                [
                                    [
                                        np.log1p(float(other[i]["context_mass_a"])),
                                        np.log1p(float(other[i]["context_mass_b"])),
                                        float(other[i]["exact_spelling_share"]),
                                    ]
                                    for i in ids
                                ]
                            )
                            - extra
                        )
                        extra = np.column_stack((extra, delta))
                    c = np.column_stack((c, extra))
                estimates[scope, adjustment].append(
                    correlation(x[ids], y[ids], c, weight[ids])
                )
            means[scope].append(np.average(y[ids], weights=weight[ids], axis=0))
    result, summaries, collinearity = [], [], []
    for (scope, adjustment), values in estimates.items():
        values = np.array(values)
        low, high = percentile_interval(values[1:])
        ids = scopes[scope]
        for j, f in enumerate(FEATURES):
            for m, outcome in enumerate(OUTCOMES):
                result.append(
                    dict(
                        analysis=label,
                        scope=scope,
                        window=rows[0]["window"],
                        adjustment=adjustment,
                        feature=f,
                        outcome=outcome,
                        pairs=len(ids),
                        partial_r=values[0, j, m],
                        ci_low=low[j, m],
                        ci_high=high[j, m],
                    )
                )
        if adjustment == "frequency_category":
            residual = (
                x[ids]
                - controls[ids] @ np.linalg.lstsq(controls[ids], x[ids], rcond=None)[0]
            )
            sd = residual.std(0)
            for a, b in itertools.combinations(range(len(FEATURES)), 2):
                r = (
                    np.corrcoef(residual[:, a], residual[:, b])[0, 1]
                    if sd[a] > 1e-10 and sd[b] > 1e-10
                    else np.nan
                )
                if np.isfinite(r) and abs(r) >= 0.7:
                    collinearity.append(
                        dict(
                            analysis=label,
                            scope=scope,
                            window=rows[0]["window"],
                            feature_a=FEATURES[a],
                            feature_b=FEATURES[b],
                            residual_r=r,
                        )
                    )
    for scope, values in means.items():
        values = np.array(values)
        low, high = np.quantile(values[1:], (0.025, 0.975), axis=0)
        for m, outcome in enumerate(OUTCOMES):
            summaries.append(
                dict(
                    analysis=label,
                    scope=scope,
                    window=rows[0]["window"],
                    outcome=outcome,
                    pairs=len(scopes[scope]),
                    mean_percent=values[0, m],
                    ci_low=low[m],
                    ci_high=high[m],
                )
            )
    pairs = [
        dict(
            analysis=label,
            category=r["category"],
            window=r["window"],
            word_a=r["word_a"],
            word_b=r["word_b"],
            **{o: point_y[i, m] for m, o in enumerate(OUTCOMES)},
            **{f: x[i, j] for j, f in enumerate(FEATURES)},
        )
        for i, r in enumerate(rows)
    ]
    return result, summaries, collinearity, pairs


class PairedQuestions(dict):
    """Identical question IDs on both sides of a corpus comparison."""

    other: dict


def failure_patterns(pairs):
    records = []
    for corpus in CORPORA:
        rows = [r for r in pairs if r["analysis"] == corpus and r["window"] == "fixed4"]
        for category in ("overall", *sorted({r["category"] for r in rows})):
            selected = (
                rows
                if category == "overall"
                else [r for r in rows if r["category"] == category]
            )
            records.append(
                dict(
                    corpus=corpus,
                    category=category,
                    pairs=len(selected),
                    cbow_zero_pairs=sum(r["cbow_accuracy"] == 0 for r in selected),
                    both_zero_pairs=sum(
                        r["cbow_accuracy"] == r["skipgram_accuracy"] == 0
                        for r in selected
                    ),
                    cbow_zero_sg_positive_pairs=sum(
                        r["cbow_accuracy"] == 0 < r["skipgram_accuracy"]
                        for r in selected
                    ),
                    sg_below_cbow_pairs=sum(r["gap"] < 0 for r in selected),
                )
            )
    return records


def report(output, links, summaries, pairs, distributions, category_changes):
    lines = [
        "# W2V1 Table 3: 코퍼스별 독립 분석과 동일 관계쌍 비교",
        "",
        "기존 통계·질문별 예측만 사용했다. 재학습, 학습 조건 변경, 새로운 코퍼스 스캔은 수행하지 않았다. 모든 관계쌍 분석은 서로 다른 directed target pair에 동일 가중치를 준다. 원래 Table 3의 질문 가중 정확도는 `category_accuracy_by_seed.csv`에 있으며 이 보고서의 pair 평균과 구별해야 한다.",
        "",
        "## 방법과 불확실성",
        "",
        "CBOW accuracy, SG accuracy, SG-CBOW gap을 별개 종속변수로 분석했다. fixed4와 dynamic10_expected를 각각 평가하고 모든 category 내부 분석 및 category 통제 pooled 분석을 저장했다. 기본 통제는 log 최소 빈도, 제곱항, endpoint 빈도 불균형, category이다. 추가 민감도 분석은 endpoint 문맥 총량과 정확 표기 비율을 통제한다. 지표 자체를 통제변수로 넣으면 해당 지표의 효과는 식별되지 않으므로 NaN으로 보고한다.",
        "",
        "95% 구간은 공유 단어 및 source/target relation pair에 동일한 exponential multiplier를 부여하고, 같은 질문의 CBOW/SG 및 corpus 비교에 같은 가중치와 seed 재표집을 적용하는 multi-membership multiplier bootstrap이다. 질문별 가중 성과를 target pair로 다시 집계하고 매번 nuisance 회귀를 적합한다. 단어·pair 중복 및 source question 의존성을 반영하는 탐색적 민감도 구간이며, 소수 seed(3), corpus overlap, 고차 의존성에 대해 보장된 검정 구간은 아니다. 모든 지표를 탐색했으며 다중검정 보정이나 인과적 주장을 하지 않는다. 동일 부호·0을 벗어난 구간도 확증적 발견으로 해석하지 않는다.",
        "",
        "cross-corpus 분석은 세 corpus 모두에서 포함된 **동일 question ID**만 사용하며 source pair와 target pair도 동일하다. 표본을 각각 독립적으로 고르지 않았다. 변화는 오른쪽-왼쪽이다. feature log1p 변환은 context_diversity_per_occurrence, shard_rate_cv, direct_per_occurrence에만 적용한다. 원래 척도 변화는 별도 CSV에 저장한다.",
        "",
        "## Phase 1: corpus-specific analysis",
        "",
    ]
    for corpus in CORPORA:
        lines += [
            f"### {corpus.upper()}",
            "",
            "Category별 pair 평균 정확도와 공유 단어·관계쌍 bootstrap 구간:",
            "",
        ]
        lines += table(
            [
                r
                for r in summaries
                if r["analysis"] == corpus and r["window"] == "fixed4"
            ],
            ["scope", "outcome", "pairs", "mean_percent", "ci_low", "ci_high"],
        )
        lines += [
            "",
            "실패 유형: CBOW 0은 세 seed와 모든 해당 질문에서 정답이 없는 pair이다. SG>0은 일부 성공을 의미하며 항상 성공을 뜻하지 않는다.",
            "",
        ]
        lines += table(
            [r for r in failure_patterns(pairs) if r["corpus"] == corpus],
            [
                "category",
                "pairs",
                "cbow_zero_pairs",
                "both_zero_pairs",
                "cbow_zero_sg_positive_pairs",
                "sg_below_cbow_pairs",
            ],
        )
        candidates = [
            r
            for r in links
            if r["analysis"] == corpus
            and r["scope"] == "overall"
            and r["window"] == "fixed4"
            and r["adjustment"] == "frequency_category"
            and r["outcome"] == "cbow_accuracy"
            and np.isfinite(r["partial_r"])
        ]
        ranked = sorted(candidates, key=lambda r: -abs(r["partial_r"]))[:6]
        lines += ["", "가장 큰 빈도/category 통제 CBOW 연관과 SG·gap 대조:", ""]
        chosen = [
            r
            for r in links
            if r["analysis"] == corpus
            and r["scope"] == "overall"
            and r["window"] == "fixed4"
            and r["adjustment"] == "frequency_category"
            and r["feature"] in {r["feature"] for r in ranked}
        ]
        lines += table(chosen, ["feature", "outcome", "partial_r", "ci_low", "ci_high"])
        lines += ["", "독립적 결론 및 내부 반대 증거:", ""]
        for r in ranked[:3]:
            internal = [
                v
                for v in links
                if v["analysis"] == corpus
                and v["scope"] != "overall"
                and v["window"] == "fixed4"
                and v["adjustment"] == "frequency_category"
                and v["feature"] == r["feature"]
                and v["outcome"] == "cbow_accuracy"
                and np.isfinite(v["partial_r"])
            ]
            same = sum(
                np.sign(v["partial_r"]) == np.sign(r["partial_r"]) for v in internal
            )
            opposite = [
                v["scope"]
                for v in internal
                if np.sign(v["partial_r"]) != np.sign(r["partial_r"])
            ]
            sensitivity = next(
                v
                for v in links
                if v["analysis"] == corpus
                and v["scope"] == "overall"
                and v["window"] == "fixed4"
                and v["adjustment"] == "exposure_sensitivity"
                and v["feature"] == r["feature"]
                and v["outcome"] == "cbow_accuracy"
            )
            interval = (
                "구간은 0을 포함한다"
                if r["ci_low"] <= 0 <= r["ci_high"]
                else "구간은 0을 벗어난다"
            )
            lines += [
                f"- `{r['feature']}`: 낮은 CBOW 정확도는 {'작은' if r['partial_r'] > 0 else '큰'} 값과 연관된다 (r={r['partial_r']:+.3f}, [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]; {interval}). 내부 category 부호 일치 {same}/{len(internal)}, 반대 category: {', '.join(opposite) or '없음'}. 문맥 총량/표기 추가 통제 r={sensitivity['partial_r']:+.3f}. 이 corpus의 후보 설명으로 유지하며 다른 corpus로 일반화하지 않는다."
            ]
        failures = sorted(
            [r for r in pairs if r["analysis"] == corpus and r["window"] == "fixed4"],
            key=lambda r: (r["cbow_accuracy"], -r["gap"]),
        )[:12]
        lines += [
            "",
            "실패 pair 예시(낮은 CBOW, 동률이면 큰 gap; 전체 pair는 CSV):",
            "",
        ]
        lines += table(
            failures,
            [
                "category",
                "word_a",
                "word_b",
                "cbow_accuracy",
                "skipgram_accuracy",
                "gap",
            ],
        )
        d = next(r for r in distributions if r["corpus"] == corpus)
        lines += [
            "",
            f"Corpus distribution: exact duplicate line 비율 {float(d['exact_duplicate_line_fraction']):.4f}, normalized duplicate {float(d['normalized_duplicate_line_fraction']):.4f}, OOV token {float(d['oov_token_fraction']):.4f}, 평균 line 길이 {float(d['mean_raw_line_length']):.2f}. Corpus 전체의 상수 특성은 corpus 내부 pair 성과의 회귀 설명변수가 될 수 없다. Shard concentration/rate CV는 pair별 분포 proxy로 따로 분석했다.",
            "",
        ]
    lines += ["## Phase 2: Cross-corpus paired analysis", ""]
    for left, right in itertools.combinations(CORPORA, 2):
        label = f"{left}->{right}"
        lines += [f"### {label}", "", "동일 target pair의 성과 변화:", ""]
        lines += table(
            [
                r
                for r in summaries
                if r["analysis"] == label and r["window"] == "fixed4"
            ],
            ["scope", "outcome", "pairs", "mean_percent", "ci_low", "ci_high"],
        )
        best = sorted(
            [
                r
                for r in links
                if r["analysis"] == label
                and r["scope"] == "overall"
                and r["window"] == "fixed4"
                and r["adjustment"] == "frequency_category"
                and r["outcome"] == "cbow_accuracy"
                and np.isfinite(r["partial_r"])
            ],
            key=lambda r: -abs(r["partial_r"]),
        )[:5]
        lines += [
            "",
            "feature 변화와 성과 변화의 연관; SG와 gap은 선택성을 확인하는 대조:",
            "",
        ]
        lines += table(
            [
                r
                for r in links
                if r["analysis"] == label
                and r["scope"] == "overall"
                and r["window"] == "fixed4"
                and r["adjustment"] == "frequency_category"
                and r["feature"] in {r["feature"] for r in best}
            ],
            ["feature", "outcome", "partial_r", "ci_low", "ci_high"],
        )
        for category in ("city-in-state", "capital-world"):
            selected = sorted(
                [
                    r
                    for r in links
                    if r["analysis"] == label
                    and r["scope"] == category
                    and r["window"] == "fixed4"
                    and r["adjustment"] == "frequency_category"
                    and r["outcome"] == "cbow_accuracy"
                    and np.isfinite(r["partial_r"])
                ],
                key=lambda r: -abs(r["partial_r"]),
            )[:3]
            lines += [
                "",
                f"{category}: 내부 pair에서 가장 큰 변화 연관(반대 부호도 보존):",
                "",
            ]
            lines += table(
                [
                    r
                    for r in links
                    if r["analysis"] == label
                    and r["scope"] == category
                    and r["window"] == "fixed4"
                    and r["adjustment"] == "frequency_category"
                    and r["feature"] in {r["feature"] for r in selected}
                ],
                ["feature", "outcome", "partial_r", "ci_low", "ci_high"],
            )
    lines += [
        "",
        "## Phase 3: 종합과 잔여 현상",
        "",
        "각 corpus의 후보 특성은 위 독립 결론을 따른다. 공통 현상은 다음 부호 일치 표로 제한하며, corpus별 크기·category 반전·SG 대조가 다르면 별개 설명을 유지한다.",
        "",
    ]
    common = []
    for feature in FEATURES:
        rs = [
            next(
                r
                for r in links
                if r["analysis"] == c
                and r["scope"] == "overall"
                and r["window"] == "fixed4"
                and r["adjustment"] == "frequency_category"
                and r["feature"] == feature
                and r["outcome"] == "cbow_accuracy"
            )
            for c in CORPORA
        ]
        common.append(
            dict(
                feature=feature,
                **{c: r["partial_r"] for c, r in zip(CORPORA, rs, strict=True)},
                same_sign=len({np.sign(r["partial_r"]) for r in rs}) == 1,
                intervals_excluding_zero=sum(
                    r["ci_low"] > 0 or r["ci_high"] < 0 for r in rs
                ),
            )
        )
    lines += table(
        common, ["feature", *CORPORA, "same_sign", "intervals_excluding_zero"]
    )
    lines += [
        "",
        "세 corpus를 독립적으로 계산한 결과 가장 큰 controlled CBOW 연관은 모두 target endpoint 문맥 분포의 JS divergence(음의 연관)와 overlap(양의 연관)이다. LM1B r=-0.498/+0.485, WMT -0.513/+0.489, UMBC -0.378/+0.369이며 두 window 및 문맥 총량 추가 통제에서도 유지된다. 이것은 '관계 endpoint의 문맥이 서로 비슷할수록 실패한다'는 단순 혼동 설명에 대한 반대 증거다. Endpoint 간 관계 정합성과 category-wide center 구별력은 다른 측정이다.",
        "",
        "LM1B에서는 JS 음의 부호가 14/14 category, overlap 양의 부호가 13/14에서 나타난다. WMT는 둘 다 14/14이다. UMBC에서는 JS 13/14, overlap 14/14이며 pooled 크기는 작다. LM1B의 shared_distinctive_mass는 city-in-state에서 부호가 반대다. 개별 corpus의 가장 일관된 후보는 이 endpoint context 정합성 지표군이지만 SG에서도 같은 부호와 비슷하거나 더 큰 연관이 있어 CBOW 저성능에 선택적인 원인은 확인되지 않았다.",
        "",
        "LM1B/WMT의 single_context_center_information_bits와 sampled conditional_proxy_discriminability는 오히려 CBOW 정확도와 음의 연관이다. UMBC에서는 이 보조 설명의 구간이 0을 포함한다. 낮은 center 구별력이 CBOW 실패를 유발한다는 방향은 저장된 proxy로 일관되게 지지되지 않는다. Dedup exposure와 relation_pattern_log2_lift의 pooled 구간도 세 corpus 모두에서 0을 포함하므로 중복이나 직접 관계 template 부족을 주된 설명으로 순위화할 근거가 약하다.",
        "",
        "동일 pair 변화에서는 LM1B->UMBC와 WMT->UMBC의 overlap 증가/JS 감소 및 shared distinctive mass 증가가 CBOW 성과 증가와 함께 변한다. SG에서도 같은 연관이 있고 해당 gap 연관 구간은 0을 포함한다. LM1B->WMT는 가장 큰 변화 연관도 0을 포함하므로 동일 수준의 근거가 없다.",
        "",
        "### city-in-state와 capital-world의 상반된 변화",
        "",
        "LM1B->UMBC의 CBOW 변화는 city-in-state +12.87 pp, capital-world -5.72 pp이고 WMT->UMBC는 +12.95/-3.45 pp이다. SG는 각각 +7.74/-26.83 pp, +8.97/-26.42 pp이다. UMBC에서 capital-world의 SG 하락이 더 크다는 점은 'CBOW만 악화'라는 설명에 반한다.",
        "",
        "원래 척도의 category 평균 feature 변화(동일 pair 가중):",
        "",
    ]
    lines += table(
        [
            r
            for r in category_changes
            if r["window"] == "fixed4"
            and r["category"] in ("city-in-state", "capital-world")
            and r["feature"]
            in (
                "context_overlap",
                "context_js_bits",
                "shared_distinctive_mass",
                "dedup_exposure_fraction",
                "direct_per_occurrence",
                "shard_rate_cv",
            )
        ],
        ["analysis", "category", "feature", "pairs", "mean_raw_change"],
    )
    lines += [
        "",
        "UMBC의 capital-world는 overlap과 shared distinctive mass 감소, JS 증가가 함께 나타나 성과 하락과 정합적이다. city-in-state는 overlap과 shared distinctive mass가 증가하지만 LM1B 대비 JS도 증가한다. 따라서 JS만으로 두 category의 평균 방향을 설명할 수 없다. 두 category 모두 direct exposure가 감소하고 shard CV는 크게 증가하는데 성과 방향은 반대여서 이 두 feature의 단순 평균 변화 역시 충분한 설명이 아니다. WMT 대비 UMBC에서 중복 제거 노출은 두 category 모두 증가하므로 중복만으로 방향 차이를 설명하기 어렵다.",
        "",
        "category 내부 paired 분석에서는 대부분의 feature 구간이 0을 포함한다. WMT->UMBC capital-world의 JS 변화는 fixed4 r=-0.379, [-0.670,-0.007]로 약한 탐색적 근거가 있지만 category·window·지표 선택과 bootstrap Monte Carlo 변동에 민감하다. city-in-state의 가장 큰 fixed4 연관(예: LM1B->UMBC proxy r=-0.365, WMT->UMBC shared distinctive mass r=+0.351)은 0을 포함한다. 평균 변화와 일부 pair 연관이 맞아도 상반된 변화가 충분히 설명되었다고 결론내릴 수 없다.",
        "",
        "- `collinearity.csv`은 nuisance 제거 후 |r|≥0.7인 지표쌍을 모두 보존한다. 이런 지표는 독립적인 설명의 개수로 세지 않는다. context_overlap과 JS 등은 같은 context 분포에서 파생되며, conditional proxy는 실제 CBOW 예측력이 아니다.",
        "- city-in-state와 capital-world의 평균 변화가 반대여도 평균 특성 변화만으로 설명이 성립하지 않는다. 각 category 내부 동일 pair 변화의 연관·구간·SG/gap 대조가 뒷받침해야 한다. `category_feature_changes.csv`의 평균 변화와 위 내부 연관이 일치하지 않으면 생태적 평균 설명에 대한 반대 증거다.",
        "- SG에서 같은 방향 연관이 나타나면 CBOW에 선택적인 설명으로 부르지 않는다. gap 변화 연관은 두 모델의 차이와의 연관이며 실제 objective, loss, gradient 또는 optimizer 메커니즘을 측정한 결과가 아니다.",
        "- 저장된 feature는 target pair endpoint의 특성이다. source pair 특성은 설명모형에 직접 넣지 않았다(공유 source 의존성은 bootstrap에 반영). 모든 관련 교란변수가 통제되었다는 주장은 불가능하다.",
        "- KMV context diversity와 128-template sampled NB proxy의 측정 오차, sense/alias 미주석, shard의 비문서 단위, 실제 training RNG/worker 경계 미재현, 원 논문 corpus 부재는 잔여 한계다. 새 scan으로도 의미 관계 주석 또는 인과 식별이 자동으로 해결되지 않는다.",
        "- 반대 category 부호, 넓은 구간, 추가 통제에 따른 변화, SG와 공유하는 연관, window별 차이는 모두 유지한다. 세 corpus에 단일 원인을 강제로 부여하지 않는다.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(output / "cross_corpus_consistency.csv", common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    if args.repetitions < 100:
        parser.error("At least 100 bootstrap repetitions required")
    paths = RuntimePaths.from_environment()
    root = paths.analysis_output("f2", "w2v1") / "corpus-investigation"
    output = root / "corpus-comparison"
    output.mkdir(parents=True, exist_ok=True)
    inputs, data, questions = {}, {}, {}
    for c in CORPORA:
        feature_path = root / c / "pair_features_with_accuracy.csv"
        question_path = (
            paths.cache_root
            / "f2/w2v1/corpus-investigation"
            / c
            / "question_outcomes.csv"
        )
        for p in (feature_path, question_path):
            inputs[str(p.relative_to(paths.repo_root))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
        data[c] = csv_rows(feature_path)
        questions[c] = load_questions(question_path.parent)
    common_ids = set.intersection(
        *(set(qid for qid, q in questions[c].items() if q["common"]) for c in CORPORA)
    )
    for qid in common_ids:
        if (
            len(
                {
                    (questions[c][qid]["category"], questions[c][qid]["words"])
                    for c in CORPORA
                }
            )
            != 1
        ):
            raise ValueError("Question identity differs across corpora")
    links, summaries, collinearity, pair_results, raw_changes, category_changes = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with threadpool_limits(limits=1):
        for window in ("fixed4", "dynamic10_expected"):
            for c in CORPORA:
                rows = [r for r in data[c] if r["window"] == window]
                print(f"Independent {c} {window}: {len(rows)} pairs", flush=True)
                result = analyze(c, rows, questions[c], args.repetitions)
                for destination, values in zip(
                    (links, summaries, collinearity, pair_results), result, strict=True
                ):
                    destination.extend(values)
            for left, right in itertools.combinations(CORPORA, 2):
                lm = {key(r): r for r in data[left] if r["window"] == window}
                rm = {key(r): r for r in data[right] if r["window"] == window}
                valid = {
                    (q["category"], *q["words"][2:])
                    for qid, q in questions[left].items()
                    if qid in common_ids
                }
                keys = sorted(set(lm) & set(rm) & valid)
                lrows, rrows = [lm[k] for k in keys], [rm[k] for k in keys]
                qs = PairedQuestions({qid: questions[left][qid] for qid in common_ids})
                qs.other = {qid: questions[right][qid] for qid in common_ids}
                label = f"{left}->{right}"
                print(f"Paired {label} {window}: {len(keys)} pairs", flush=True)
                result = analyze(label, lrows, qs, args.repetitions, other=rrows)
                for destination, values in zip(
                    (links, summaries, collinearity, pair_results), result, strict=True
                ):
                    destination.extend(values)
                for l, r in zip(lrows, rrows, strict=True):
                    raw_changes.append(
                        dict(
                            analysis=label,
                            category=l["category"],
                            window=window,
                            word_a=l["word_a"],
                            word_b=l["word_b"],
                            **{f: float(r[f]) - float(l[f]) for f in FEATURES},
                        )
                    )
                for category in sorted({r["category"] for r in lrows}):
                    subset = [
                        r
                        for r in raw_changes
                        if r["analysis"] == label
                        and r["window"] == window
                        and r["category"] == category
                    ]
                    for f in FEATURES:
                        category_changes.append(
                            dict(
                                analysis=label,
                                category=category,
                                window=window,
                                feature=f,
                                pairs=len(subset),
                                mean_raw_change=float(np.mean([r[f] for r in subset])),
                            )
                        )
    for name, rows in (
        ("associations", links),
        ("accuracy", summaries),
        ("collinearity", collinearity),
        ("pair_results", pair_results),
        ("pair_feature_changes", raw_changes),
        ("category_feature_changes", category_changes),
        ("failure_patterns", failure_patterns(pair_results)),
    ):
        write_csv(output / f"{name}.csv", rows)
    report(
        output,
        links,
        summaries,
        pair_results,
        csv_rows(root / "corpus_distribution.csv"),
        category_changes,
    )
    write_csv(
        output / "input_provenance.csv",
        [dict(path=p, sha256=h) for p, h in inputs.items()],
    )
    cache_output = paths.cache_root / "f2/w2v1/corpus-comparison"
    cache_output.mkdir(parents=True, exist_ok=True)
    (cache_output / "manifest.json").write_text(
        json.dumps(
            dict(
                inputs=inputs,
                bootstrap_repetitions=args.repetitions,
                bootstrap_seed=20261011,
                common_questions=len(common_ids),
                training_changed=False,
                corpus_scanned=False,
            ),
            indent=2,
        )
    )
    print(output / "summary.md", flush=True)


if __name__ == "__main__":
    main()
