#!/usr/bin/env python3
"""Train a text-only robustness fallback regressor on the public augmented set.

Mirrors the repo's train_uncertainty_regressor.py protocol (StratifiedGroupKFold
grouped by problem text, out-of-fold threshold selection by balanced accuracy,
refit on all rows, joblib export) but uses problem-text statistics instead of
generation uncertainty features, so it applies to any model_id without loading
a model. Training data: aimo-interp/augmented-sample-math-agg@f972ced (public).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

FEATURE_NAMES: tuple[str, ...] = (
    "text_char_len",
    "text_word_count",
    "text_avg_word_len",
    "text_line_count",
    "text_digit_frac",
    "text_number_count",
    "text_max_number_log10",
    "text_total_number_log10",
    "text_operator_count",
    "text_paren_count",
    "text_comma_count",
    "text_question_count",
    "text_superlative_flag",
    "text_frac_unique_words",
)

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[A-Za-z]+")
SUPERLATIVE_RE = re.compile(
    r"\b(greatest|largest|smallest|minimum|maximum|fewest|most|least|longest|shortest)\b",
    re.IGNORECASE,
)
OPERATORS = "+-*/=^"

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "text_fallback_regressor"


def extract_text_features(text: str) -> dict[str, float]:
    chars = len(text)
    words = WORD_RE.findall(text)
    word_count = len(words)
    numbers = [float(m.group(0)) for m in NUMBER_RE.finditer(text)]
    log_magnitudes = [math.log10(abs(n)) if n > 0 else 0.0 for n in numbers]
    digits = sum(c.isdigit() for c in text)
    return {
        "text_char_len": float(chars),
        "text_word_count": float(word_count),
        "text_avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "text_line_count": float(text.count("\n") + 1),
        "text_digit_frac": digits / chars if chars else 0.0,
        "text_number_count": float(len(numbers)),
        "text_max_number_log10": max(log_magnitudes) if log_magnitudes else 0.0,
        "text_total_number_log10": float(sum(log_magnitudes)),
        "text_operator_count": float(sum(text.count(op) for op in OPERATORS)),
        "text_paren_count": float(text.count("(") + text.count(")")),
        "text_comma_count": float(text.count(",")),
        "text_question_count": float(text.count("?")),
        "text_superlative_flag": float(bool(SUPERLATIVE_RE.search(text))),
        "text_frac_unique_words": (len({w.lower() for w in words}) / word_count) if word_count else 0.0,
    }


def candidates() -> dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=3, min_samples_leaf=8, random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2, max_features=0.7, random_state=42
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2, max_features=0.7, random_state=42
        ),
    }


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Threshold maximizing OOF balanced accuracy for predict robust == score < t."""
    candidate_ts = np.unique(np.concatenate(([scores.min() - 1e-9], scores, [scores.max() + 1e-9])))
    best = (-1.0, -1.0, float("nan"))
    for t in candidate_ts:
        pred = scores < t
        bal = balanced_accuracy_score(labels, pred)
        acc = accuracy_score(labels, pred)
        if (bal, acc) > (best[0], best[1]):
            best = (bal, acc, float(t))
    return best[2], best[0], best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    frame = frame.drop_duplicates(subset=["original_problem", "model_is_robust", "absolute_accuracy_decay"])
    features = pd.DataFrame.from_records(
        [extract_text_features(t) for t in frame["original_problem"]],
        columns=list(FEATURE_NAMES),
    )
    target = frame["absolute_accuracy_decay"].to_numpy(dtype=float)
    labels = frame["model_is_robust"].to_numpy(dtype=bool)
    groups = frame["original_problem"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    report: dict[str, object] = {"rows": len(frame), "unique_problems": int(pd.unique(groups).size)}
    results = []
    oof_store: dict[str, np.ndarray] = {}
    for name, estimator in candidates().items():
        oof = np.zeros(len(frame), dtype=float)
        for train_idx, val_idx in splitter.split(features, labels, groups):
            model = clone(estimator).fit(features.iloc[train_idx], target[train_idx])
            oof[val_idx] = model.predict(features.iloc[val_idx])
        threshold, bal, acc = best_threshold(oof, labels)
        mae = mean_absolute_error(target, oof)
        results.append({"candidate": name, "threshold": threshold, "balanced_accuracy": bal, "accuracy": acc, "mae": mae})
        oof_store[name] = oof
        print(f"{name}: threshold={threshold:.4f} balanced_acc={bal:.4f} acc={acc:.4f} mae={mae:.4f}")

    winner = max(results, key=lambda r: (r["balanced_accuracy"], r["accuracy"]))
    final_model = clone(candidates()[winner["candidate"]]).fit(features, target)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "feature_names": list(FEATURE_NAMES),
        "estimator": final_model,
        "decision_threshold": float(winner["threshold"]),
        "selected_params": {"family": winner["candidate"]},
        "cv_results": {"candidates": results, "winner": winner["candidate"]},
        "training_provenance": {
            "dataset": "aimo-interp/augmented-sample-math-agg",
            "revision": "f972ced0705096f8d7ca7fac30825900b8b7fb6a",
            "split": "validation",
            "rows": len(frame),
            "target": "absolute_accuracy_decay",
            "label": "model_is_robust",
        },
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    report["winner"] = winner
    report["candidates"] = results
    print(f"winner={winner['candidate']} -> {args.output}")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
