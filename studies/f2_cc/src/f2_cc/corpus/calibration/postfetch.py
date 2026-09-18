"""Post-fetch classifier threshold sweeps and stratified cross-validation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .models import PostFetchOperatingPoint


def evaluate_postfetch_sweep(
    audited_items: list[dict[str, Any]],
) -> list[PostFetchOperatingPoint]:
    """Sweep post-fetch news_score threshold across full audited sample."""
    tot_gold_doc_w = sum(
        r["total_weight"] for r in audited_items if r["gold_class"] == 1
    )
    tot_gold_word_w = sum(
        r["total_weight"] * r["word_count_gold"]
        for r in audited_items
        if r["gold_class"] == 1
    )
    tot_gold_byte_w = sum(
        r["total_weight"] * float(r["downloaded_bytes"])
        for r in audited_items
        if r["gold_class"] == 1
    )

    all_extracted_words = sum(
        r["total_weight"] * float(r["word_count"])
        for r in audited_items
        if int(r.get("extraction_success", 0)) == 1
        and int(r.get("is_english", 0)) == 1
        and int(r.get("is_valid", 0)) == 1
    )

    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    points: list[PostFetchOperatingPoint] = []

    for tau in thresholds:
        pred_items = [
            r
            for r in audited_items
            if int(r.get("is_english", 0)) == 1
            and int(r.get("is_valid", 0)) == 1
            and float(r.get("news_score", 0.0)) >= tau
        ]

        tp_doc = sum(r["total_weight"] for r in pred_items if r["gold_class"] == 1)
        fp_doc = sum(r["total_weight"] for r in pred_items if r["gold_class"] == 0)

        tp_word = sum(
            r["total_weight"] * r["word_count_gold"]
            for r in pred_items
            if r["gold_class"] == 1
        )
        fp_word = sum(
            r["total_weight"] * float(r["word_count"])
            for r in pred_items
            if r["gold_class"] == 0
        )

        tp_byte = sum(
            r["total_weight"] * float(r["downloaded_bytes"])
            for r in pred_items
            if r["gold_class"] == 1
        )
        fp_byte = sum(
            r["total_weight"] * float(r["downloaded_bytes"])
            for r in pred_items
            if r["gold_class"] == 0
        )

        doc_ppv = tp_doc / (tp_doc + fp_doc) if (tp_doc + fp_doc) > 0 else 0.0
        doc_rec = tp_doc / tot_gold_doc_w if tot_gold_doc_w > 0 else 0.0

        word_ppv = tp_word / (tp_word + fp_word) if (tp_word + fp_word) > 0 else 0.0
        word_rec = tp_word / tot_gold_word_w if tot_gold_word_w > 0 else 0.0

        byte_ppv = tp_byte / (tp_byte + fp_byte) if (tp_byte + fp_byte) > 0 else 0.0
        byte_rec = tp_byte / tot_gold_byte_w if tot_gold_byte_w > 0 else 0.0

        retained_corpus_words = sum(
            r["total_weight"] * float(r["word_count"]) for r in pred_items
        )
        storage_saved = (
            1.0 - (retained_corpus_words / all_extracted_words)
            if all_extracted_words > 0
            else 0.0
        )

        points.append(
            PostFetchOperatingPoint(
                threshold=tau,
                doc_precision=doc_ppv,
                doc_recall=doc_rec,
                word_precision=word_ppv,
                word_recall=word_rec,
                byte_precision=byte_ppv,
                byte_recall=byte_rec,
                storage_savings_pct=storage_saved,
                is_cross_validated=False,
            )
        )
    return points


def evaluate_cross_validated_postfetch(
    audited_items: list[dict[str, Any]],
    target_recalls: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate out-of-fold cross-validated generalization at target operating points."""
    if target_recalls is None:
        target_recalls = [0.95, 0.97, 0.98, 1.00]

    random.seed(42)
    strata_buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in audited_items:
        k = (int(r["audit_stratum"]), int(r["gold_class"]))
        strata_buckets.setdefault(k, []).append(r)

    for k in strata_buckets:
        random.shuffle(strata_buckets[k])

    folds: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for _k, items in strata_buckets.items():
        for i, item in enumerate(items):
            folds[i % 5].append(item)

    results: list[dict[str, Any]] = []
    for target_rec in target_recalls:
        oof_preds: list[tuple[dict[str, Any], bool, float]] = []

        for fold_idx in range(5):
            train_set = [
                item for i, f in enumerate(folds) if i != fold_idx for item in f
            ]
            test_set = folds[fold_idx]

            train_gold_w = sum(
                r["total_weight"] * r["word_count_gold"]
                for r in train_set
                if r["gold_class"] == 1
            )

            best_tau = 0.0
            for cand_tau in np.linspace(0.0, 3.0, 121):
                pred_train = [
                    r
                    for r in train_set
                    if int(r.get("is_english", 0)) == 1
                    and int(r.get("is_valid", 0)) == 1
                    and float(r.get("news_score", 0.0)) >= cand_tau
                ]
                train_tp = sum(
                    r["total_weight"] * r["word_count_gold"]
                    for r in pred_train
                    if r["gold_class"] == 1
                )
                train_rec = train_tp / train_gold_w if train_gold_w > 0 else 0.0
                if train_rec >= target_rec:
                    best_tau = float(cand_tau)

            for r in test_set:
                accepted = (
                    int(r.get("is_english", 0)) == 1
                    and int(r.get("is_valid", 0)) == 1
                    and float(r.get("news_score", 0.0)) >= best_tau
                )
                oof_preds.append((r, accepted, best_tau))

        tot_gold_w = sum(
            r["total_weight"] * r["word_count_gold"]
            for r, _, _ in oof_preds
            if r["gold_class"] == 1
        )
        tp_word = sum(
            r["total_weight"] * r["word_count_gold"]
            for r, acc, _ in oof_preds
            if acc and r["gold_class"] == 1
        )
        fp_word = sum(
            r["total_weight"] * float(r["word_count"])
            for r, acc, _ in oof_preds
            if acc and r["gold_class"] == 0
        )

        tot_gold_doc = sum(
            r["total_weight"] for r, _, _ in oof_preds if r["gold_class"] == 1
        )
        tp_doc = sum(
            r["total_weight"] for r, acc, _ in oof_preds if acc and r["gold_class"] == 1
        )
        fp_doc = sum(
            r["total_weight"] for r, acc, _ in oof_preds if acc and r["gold_class"] == 0
        )

        w_rec = tp_word / tot_gold_w if tot_gold_w > 0 else 0.0
        w_ppv = tp_word / (tp_word + fp_word) if (tp_word + fp_word) > 0 else 0.0
        d_rec = tp_doc / tot_gold_doc if tot_gold_doc > 0 else 0.0
        d_ppv = tp_doc / (tp_doc + fp_doc) if (tp_doc + fp_doc) > 0 else 0.0

        all_ext = sum(
            r["total_weight"] * float(r["word_count"])
            for r, _, _ in oof_preds
            if int(r.get("extraction_success", 0)) == 1
            and int(r.get("is_english", 0)) == 1
            and int(r.get("is_valid", 0)) == 1
        )
        retained_w = sum(
            r["total_weight"] * float(r["word_count"]) for r, acc, _ in oof_preds if acc
        )
        stor_saved = 1.0 - (retained_w / all_ext) if all_ext > 0 else 0.0

        results.append(
            {
                "target_recall": target_rec,
                "oof_word_recall": w_rec,
                "oof_word_precision": w_ppv,
                "oof_doc_recall": d_rec,
                "oof_doc_precision": d_ppv,
                "storage_savings_pct": stor_saved,
            }
        )
    return results


__all__ = [
    "evaluate_cross_validated_postfetch",
    "evaluate_postfetch_sweep",
]
