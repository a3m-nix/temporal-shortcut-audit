#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
24-temporal_audit-thresholdfree-multibudget-translation-posthoc-v1.py

Post-hoc strengthening analysis for the temporal-shortcut manuscript.

NO RETRAINING is performed.

This script adds three robustness blocks requested for the revised manuscript:

A. Primary controlled-intervention threshold-free metrics
   -------------------------------------------------------
   Uses the saved lambda=0 and lambda=1 sample-level common-q scores from the
   360-run controlled temporal intervention.
   Reports:
     - ROC-AUC
     - PR-AUC / Average Precision
     - paired lambda0 - lambda1 differences
     - Nadeau-Bengio corrected repeated-CV tests
     - Holm correction across the 12 dataset-model comparisons separately
       within each metric family.

B. Audit-informed A(q+t) vs B(q) multi-budget operating points
   ------------------------------------------------------------
   Uses already-saved calibration-normal q scores and test q scores from
   temporal_audit_temporal_upgrade_e2e_v1_2.
   For each run and condition, thresholds are recalculated from its own
   calibration-normal scores at:
     - P95
     - P99
     - P99.9
   Reports FNR, FPR, ROC-AUC, PR-AUC and B-A differences.
   These are descriptive/post-hoc supporting sensitivity results.

C. Global translation control: FPR as the informative endpoint
   ------------------------------------------------------------
   Uses the existing fixed-weight "global_plus_one_fit_span" intervention for
   A_q_t and reports original vs shifted:
     - FNR
     - FPR
   under:
     - the original fixed threshold
     - the shifted-calibration recalibrated threshold

Evidence class
--------------
POSTHOC_MECHANISTIC_STRENGTHENING.
The frozen primary evidence remains the lambda=0 vs lambda=1 non-epoch FNR
controlled intervention with Nadeau-Bengio/Holm-12 inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt


# =============================================================================
# PATHS / DESIGN
# =============================================================================

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
LEGACY_ROOT = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"

SNN_ROOT = (
    LEGACY_ROOT
    / "controlled_temporal_intervention"
    / "snn_3datasets_30fold_v1"
)

AE3_ROOT = (
    LEGACY_ROOT
    / "controlled_temporal_intervention"
    / "ae3_3datasets_30fold_v1"
)

AUDIT_ROOT = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2"

DEFAULT_OUT = (
    BASE
    / "temporal_audit_posthoc_strengthening_v1"
    / "06_thresholdfree_multibudget_translation"
)

DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]
MODELS = ["CNN_AE", "Dense_AE", "LSTM_AE", "SNN_AE"]
SEEDS = [0, 1, 2]
FOLDS = list(range(1, 11))

DATASET_LABEL = {
    "edge_iiotset": "Edge-IIoTset",
    "ciciot2023": "CICIoT2023",
    "ton_iot": "ToN-IoT",
}

MODEL_LABEL = {
    "CNN_AE": "CNN-AE",
    "Dense_AE": "Dense-AE",
    "LSTM_AE": "LSTM-AE",
    "SNN_AE": "SNN-AE",
}

BUDGETS = [95.0, 99.0, 99.9]

NB_TEST_TRAIN_RATIO = 1.0 / 9.0
ALPHA = 0.05

EVIDENCE_CLASS = "POSTHOC_MECHANISTIC_STRENGTHENING"
SCRIPT_VERSION = "1.0.0"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(m, dtype=float)
    running_max = 0.0

    for i, pv in enumerate(ranked):
        v = (m - i) * pv
        running_max = max(running_max, v)
        adjusted_ranked[i] = min(1.0, running_max)

    out = np.empty(m, dtype=float)
    out[order] = adjusted_ranked
    return out


def nb_corrected_test(
    diff: np.ndarray,
    ratio: float = NB_TEST_TRAIN_RATIO,
    alpha: float = ALPHA,
) -> Dict[str, float]:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) < 2:
        raise ValueError("Need at least two finite paired differences")

    n = len(d)
    mean_diff = float(np.mean(d))
    median_diff = float(np.median(d))
    sd = float(np.std(d, ddof=1))
    se = float(np.sqrt(((1.0 / n) + ratio) * (sd ** 2)))
    df = n - 1

    if se == 0:
        if mean_diff == 0:
            t_stat = 0.0
            p = 1.0
        else:
            t_stat = np.inf if mean_diff > 0 else -np.inf
            p = 0.0
        lo = mean_diff
        hi = mean_diff
    else:
        t_stat = float(mean_diff / se)
        p = float(2.0 * student_t.sf(abs(t_stat), df))
        tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df))
        lo = float(mean_diff - tcrit * se)
        hi = float(mean_diff + tcrit * se)

    try:
        if np.allclose(d, 0):
            w_stat = 0.0
            w_p = 1.0
        else:
            w = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
            w_stat = float(w.statistic)
            w_p = float(w.pvalue)
    except Exception:
        w_stat = np.nan
        w_p = np.nan

    return {
        "n_paired_units": int(n),
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "sd_difference": sd,
        "corrected_se": se,
        "nb_t_stat": t_stat,
        "nb_df": int(df),
        "nb_p_raw": p,
        "corrected_ci95_low": lo,
        "corrected_ci95_high": hi,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_raw": w_p,
    }


def binary_operating_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(score, dtype=float)

    if len(y) != len(s):
        raise ValueError("y/score length mismatch")

    pred = (s > float(threshold)).astype(np.int8)

    attack = y == 1
    normal = y == 0

    n_attack = int(np.sum(attack))
    n_normal = int(np.sum(normal))

    fn = int(np.sum(attack & (pred == 0)))
    tp = int(np.sum(attack & (pred == 1)))
    fp = int(np.sum(normal & (pred == 1)))
    tn = int(np.sum(normal & (pred == 0)))

    fnr = float(fn / n_attack) if n_attack else np.nan
    fpr = float(fp / n_normal) if n_normal else np.nan
    recall = float(tp / n_attack) if n_attack else np.nan

    return {
        "n": int(len(y)),
        "n_attack": n_attack,
        "n_normal": n_normal,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "fnr": fnr,
        "fpr": fpr,
        "recall": recall,
    }


def threshold_free_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(score, dtype=float)

    if len(np.unique(y)) != 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}

    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }


def percentile_threshold(cal_scores: np.ndarray, percentile: float) -> float:
    x = np.asarray(cal_scores, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("No finite calibration scores")
    return float(np.percentile(x, float(percentile)))


# =============================================================================
# A. CONTROLLED INTERVENTION: THRESHOLD-FREE METRICS
# =============================================================================

def discover_controlled_runs() -> pd.DataFrame:
    rows = []

    for root, source_block in [(SNN_ROOT, "SNN"), (AE3_ROOT, "AE3")]:
        runs_dir = root / "runs"
        if not runs_dir.exists():
            raise FileNotFoundError(runs_dir)

        for marker in sorted(runs_dir.rglob("RUN_COMPLETE.json")):
            run_dir = marker.parent
            manifest_path = run_dir / "run_manifest.json"
            sample_path = run_dir / "sample_level_intervention.csv.gz"

            if not manifest_path.exists() or not sample_path.exists():
                continue

            obj = json.loads(manifest_path.read_text())

            rows.append({
                "dataset": str(obj["dataset"]),
                "model": str(obj["model"]),
                "seed": int(obj["seed"]),
                "fold": int(obj["fold"]),
                "source_block": source_block,
                "run_dir": str(run_dir),
                "sample_level_path": str(sample_path),
            })

    df = pd.DataFrame(rows)

    expected = 3 * 4 * 3 * 10
    if len(df) != expected:
        raise RuntimeError(f"Expected {expected} controlled runs, found {len(df)}")

    check = (
        df.groupby(["dataset", "model"])
        .size()
        .reset_index(name="n")
    )

    if len(check) != 12 or not (check["n"] == 30).all():
        raise RuntimeError(
            "Controlled run inventory mismatch:\n"
            + check.to_string(index=False)
        )

    return df.sort_values(["dataset", "model", "seed", "fold"]).reset_index(drop=True)


def controlled_thresholdfree(run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []

    for _, rr in run_df.iterrows():
        p = Path(rr["sample_level_path"])
        df = pd.read_csv(p)

        required = {
            "source_row_index",
            "y_true",
            "lambda",
            "non_epoch_score",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{p}: missing {missing}")

        for lam in [0.0, 1.0]:
            g = df[np.isclose(df["lambda"].astype(float), lam)].copy()

            if g["source_row_index"].duplicated().any():
                raise RuntimeError(f"{p}: duplicate source rows at lambda={lam}")

            tf = threshold_free_metrics(
                g["y_true"].to_numpy(dtype=np.int8),
                g["non_epoch_score"].to_numpy(dtype=float),
            )

            metric_rows.append({
                "dataset": rr["dataset"],
                "model": rr["model"],
                "seed": int(rr["seed"]),
                "fold": int(rr["fold"]),
                "lambda": lam,
                "n_test": int(len(g)),
                **tf,
            })

    metrics = pd.DataFrame(metric_rows)

    stat_rows = []

    for metric in ["roc_auc", "pr_auc"]:
        for (ds, model), g in metrics.groupby(["dataset", "model"]):
            wide = g.pivot(
                index=["seed", "fold"],
                columns="lambda",
                values=metric,
            )

            if set(wide.columns) != {0.0, 1.0} or len(wide) != 30:
                raise RuntimeError(
                    f"{ds} {model} {metric}: incomplete lambda pair"
                )

            # Same direction as primary endpoint:
            # lambda0 (time neutralized) minus lambda1 (original time).
            diff = (
                wide[0.0].to_numpy(dtype=float)
                - wide[1.0].to_numpy(dtype=float)
            )

            s = nb_corrected_test(diff)

            stat_rows.append({
                "metric": metric,
                "dataset": ds,
                "model": model,
                "lambda0_mean": float(wide[0.0].mean()),
                "lambda1_mean": float(wide[1.0].mean()),
                **s,
            })

    stats = pd.DataFrame(stat_rows)

    # Holm within each metric family across 12 dataset-model comparisons.
    stats["p_holm12"] = np.nan
    stats["significant_holm12_0_05"] = False

    for metric, idx in stats.groupby("metric").groups.items():
        ii = list(idx)
        adj = holm_adjust(stats.loc[ii, "nb_p_raw"].to_numpy(dtype=float))
        stats.loc[ii, "p_holm12"] = adj
        stats.loc[ii, "significant_holm12_0_05"] = adj < ALPHA

    return metrics, stats


# =============================================================================
# B. AUDIT-INFORMED A vs B MULTI-BUDGET ANALYSIS
# =============================================================================

def audit_run_dir(
    ds: str,
    condition: str,
    model: str,
    seed: int,
) -> Path:
    return (
        AUDIT_ROOT
        / "runs"
        / ds
        / condition
        / model
        / f"seed_{seed:02d}"
    )


def load_audit_run(
    ds: str,
    condition: str,
    model: str,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    run_dir = audit_run_dir(ds, condition, model, seed)

    marker = run_dir / "RUN_COMPLETE.json"
    cal_path = run_dir / "calibration_scores.npz"
    test_path = run_dir / "test_predictions.csv.gz"

    for p in [marker, cal_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with np.load(cal_path, allow_pickle=False) as z:
        if "q_score" not in z.files:
            raise RuntimeError(f"{cal_path}: q_score missing")
        cal_q = z["q_score"].astype(np.float64)

    test = pd.read_csv(test_path)

    required = {
        "source_row_index",
        "y_true",
        "q_score",
    }
    missing = sorted(required - set(test.columns))
    if missing:
        raise RuntimeError(f"{test_path}: missing {missing}")

    if test["source_row_index"].duplicated().any():
        raise RuntimeError(f"{test_path}: duplicate source_row_index")

    return cal_q, test


def audit_multibudget() -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []

    for ds in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                # Fail closed: A and B must evaluate exactly the same test rows.
                cal_a, test_a = load_audit_run(ds, "A_q_t", model, seed)
                cal_b, test_b = load_audit_run(ds, "B_q", model, seed)

                if not np.array_equal(
                    test_a["source_row_index"].to_numpy(dtype=np.int64),
                    test_b["source_row_index"].to_numpy(dtype=np.int64),
                ):
                    raise RuntimeError(
                        f"{ds} {model} seed={seed}: A/B test source-row mismatch"
                    )

                if not np.array_equal(
                    test_a["y_true"].to_numpy(dtype=np.int8),
                    test_b["y_true"].to_numpy(dtype=np.int8),
                ):
                    raise RuntimeError(
                        f"{ds} {model} seed={seed}: A/B y_true mismatch"
                    )

                for condition, cal_q, test in [
                    ("A_q_t", cal_a, test_a),
                    ("B_q", cal_b, test_b),
                ]:
                    y = test["y_true"].to_numpy(dtype=np.int8)
                    score = test["q_score"].to_numpy(dtype=float)
                    tf = threshold_free_metrics(y, score)

                    for percentile in BUDGETS:
                        th = percentile_threshold(cal_q, percentile)
                        op = binary_operating_metrics(y, score, th)

                        # Approximate count of calibration observations in the
                        # nominal upper tail. This is descriptive only.
                        tail_expected = len(cal_q) * (100.0 - percentile) / 100.0

                        run_rows.append({
                            "dataset": ds,
                            "model": model,
                            "seed": int(seed),
                            "condition": condition,
                            "threshold_percentile": float(percentile),
                            "calibration_n": int(len(cal_q)),
                            "nominal_calibration_upper_tail_count": float(tail_expected),
                            "threshold": th,
                            "roc_auc": tf["roc_auc"],
                            "pr_auc": tf["pr_auc"],
                            **op,
                        })

    run_df = pd.DataFrame(run_rows)

    # Run-balanced summaries across three model seeds.
    summary_rows = []

    for keys, g in run_df.groupby(
        ["dataset", "model", "condition", "threshold_percentile"]
    ):
        ds, model, condition, percentile = keys

        row = {
            "dataset": ds,
            "model": model,
            "condition": condition,
            "threshold_percentile": float(percentile),
            "n_model_seeds": int(g["seed"].nunique()),
            "calibration_n": int(g["calibration_n"].iloc[0]),
            "nominal_calibration_upper_tail_count":
                float(g["nominal_calibration_upper_tail_count"].iloc[0]),
        }

        for col in ["fnr", "fpr", "roc_auc", "pr_auc"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = float(g[col].std(ddof=1))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    return run_df, summary


def audit_multibudget_contrasts(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive paired B - A differences across the 3 matched model seeds.
    No inferential p-values are promoted because n=3 seeds is too small for
    strong manuscript-level inference.
    """
    rows = []

    metrics = ["fnr", "fpr", "roc_auc", "pr_auc"]

    for (ds, model, percentile), g in run_df.groupby(
        ["dataset", "model", "threshold_percentile"]
    ):
        for metric in metrics:
            wide = g.pivot(
                index="seed",
                columns="condition",
                values=metric,
            )

            if set(wide.columns) != {"A_q_t", "B_q"} or len(wide) != 3:
                raise RuntimeError(
                    f"{ds} {model} P{percentile} {metric}: incomplete A/B seed pairing"
                )

            diff = (
                wide["B_q"].to_numpy(dtype=float)
                - wide["A_q_t"].to_numpy(dtype=float)
            )

            rows.append({
                "dataset": ds,
                "model": model,
                "threshold_percentile": float(percentile),
                "metric": metric,
                "n_paired_seeds": int(len(diff)),
                "A_mean": float(wide["A_q_t"].mean()),
                "B_mean": float(wide["B_q"].mean()),
                "B_minus_A_mean": float(np.mean(diff)),
                "B_minus_A_median": float(np.median(diff)),
                "B_minus_A_min": float(np.min(diff)),
                "B_minus_A_max": float(np.max(diff)),
            })

    return pd.DataFrame(rows)


# =============================================================================
# C. TRANSLATION CONTROL: DELTA FPR
# =============================================================================

def translation_control() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    scheme = "global_plus_one_fit_span"
    condition = "A_q_t"

    for ds in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                original_path = (
                    AUDIT_ROOT
                    / "runs"
                    / ds
                    / condition
                    / model
                    / f"seed_{seed:02d}"
                    / "test_predictions.csv.gz"
                )

                shifted_path = (
                    AUDIT_ROOT
                    / "interventions"
                    / ds
                    / condition
                    / model
                    / f"seed_{seed:02d}"
                    / scheme
                    / "test_predictions_shifted.csv.gz"
                )

                metrics_path = (
                    AUDIT_ROOT
                    / "interventions"
                    / ds
                    / condition
                    / model
                    / f"seed_{seed:02d}"
                    / scheme
                    / "metrics.json"
                )

                for p in [original_path, shifted_path, metrics_path]:
                    if not p.exists():
                        raise FileNotFoundError(p)

                original = pd.read_csv(original_path)
                shifted = pd.read_csv(shifted_path)

                required_orig = {
                    "source_row_index",
                    "y_true",
                    "q_prediction",
                    "q_score",
                }
                required_shift = {
                    "source_row_index",
                    "y_true",
                    "original_q_prediction",
                    "shifted_fixed_prediction",
                    "shifted_recalibrated_prediction",
                    "shifted_q_score",
                }

                mo = sorted(required_orig - set(original.columns))
                ms = sorted(required_shift - set(shifted.columns))

                if mo:
                    raise RuntimeError(f"{original_path}: missing {mo}")
                if ms:
                    raise RuntimeError(f"{shifted_path}: missing {ms}")

                if not np.array_equal(
                    original["source_row_index"].to_numpy(dtype=np.int64),
                    shifted["source_row_index"].to_numpy(dtype=np.int64),
                ):
                    raise RuntimeError(
                        f"{ds} {model} seed={seed}: translation source-row mismatch"
                    )

                if not np.array_equal(
                    original["y_true"].to_numpy(dtype=np.int8),
                    shifted["y_true"].to_numpy(dtype=np.int8),
                ):
                    raise RuntimeError(
                        f"{ds} {model} seed={seed}: translation y_true mismatch"
                    )

                y = original["y_true"].to_numpy(dtype=np.int8)

                tracks = [
                    ("original", original["q_prediction"].to_numpy(dtype=np.int8)),
                    ("shifted_fixed", shifted["shifted_fixed_prediction"].to_numpy(dtype=np.int8)),
                    ("shifted_recalibrated", shifted["shifted_recalibrated_prediction"].to_numpy(dtype=np.int8)),
                ]

                track_metrics = {}

                for track, pred in tracks:
                    attack = y == 1
                    normal = y == 0

                    fnr = float(np.mean(pred[attack] == 0)) if np.any(attack) else np.nan
                    fpr = float(np.mean(pred[normal] == 1)) if np.any(normal) else np.nan

                    track_metrics[track] = {
                        "fnr": fnr,
                        "fpr": fpr,
                    }

                result = json.loads(metrics_path.read_text())

                # Cross-check against saved metrics.
                fixed_saved = result["fixed_threshold_metrics"]
                recal_saved = result["recalibrated_threshold_metrics"]

                if abs(track_metrics["shifted_fixed"]["fnr"] - float(fixed_saved["fnr"])) > 1e-12:
                    raise RuntimeError(f"{metrics_path}: fixed FNR mismatch")
                if abs(track_metrics["shifted_fixed"]["fpr"] - float(fixed_saved["fpr"])) > 1e-12:
                    raise RuntimeError(f"{metrics_path}: fixed FPR mismatch")
                if abs(track_metrics["shifted_recalibrated"]["fnr"] - float(recal_saved["fnr"])) > 1e-12:
                    raise RuntimeError(f"{metrics_path}: recal FNR mismatch")
                if abs(track_metrics["shifted_recalibrated"]["fpr"] - float(recal_saved["fpr"])) > 1e-12:
                    raise RuntimeError(f"{metrics_path}: recal FPR mismatch")

                rows.append({
                    "dataset": ds,
                    "model": model,
                    "seed": int(seed),
                    "condition": condition,
                    "scheme": scheme,

                    "original_fnr": track_metrics["original"]["fnr"],
                    "fixed_shifted_fnr": track_metrics["shifted_fixed"]["fnr"],
                    "recal_shifted_fnr": track_metrics["shifted_recalibrated"]["fnr"],

                    "original_fpr": track_metrics["original"]["fpr"],
                    "fixed_shifted_fpr": track_metrics["shifted_fixed"]["fpr"],
                    "recal_shifted_fpr": track_metrics["shifted_recalibrated"]["fpr"],

                    "delta_fnr_fixed_minus_original":
                        track_metrics["shifted_fixed"]["fnr"]
                        - track_metrics["original"]["fnr"],

                    "delta_fnr_recal_minus_original":
                        track_metrics["shifted_recalibrated"]["fnr"]
                        - track_metrics["original"]["fnr"],

                    "delta_fpr_fixed_minus_original":
                        track_metrics["shifted_fixed"]["fpr"]
                        - track_metrics["original"]["fpr"],

                    "delta_fpr_recal_minus_original":
                        track_metrics["shifted_recalibrated"]["fpr"]
                        - track_metrics["original"]["fpr"],
                })

    run_df = pd.DataFrame(rows)

    summary_rows = []

    for (ds, model), g in run_df.groupby(["dataset", "model"]):
        row = {
            "dataset": ds,
            "model": model,
            "n_model_seeds": int(g["seed"].nunique()),
        }

        for col in [
            "original_fnr",
            "fixed_shifted_fnr",
            "recal_shifted_fnr",
            "delta_fnr_fixed_minus_original",
            "delta_fnr_recal_minus_original",
            "original_fpr",
            "fixed_shifted_fpr",
            "recal_shifted_fpr",
            "delta_fpr_fixed_minus_original",
            "delta_fpr_recal_minus_original",
        ]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = float(g[col].std(ddof=1))

        summary_rows.append(row)

    return run_df, pd.DataFrame(summary_rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_multibudget_fnr(
    summary: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = summary[summary["dataset"] == ds].copy()

        x = np.arange(len(MODELS))
        width = 0.12

        offsets = {
            ("A_q_t", 95.0): -2.5 * width,
            ("B_q",   95.0): -1.5 * width,
            ("A_q_t", 99.0): -0.5 * width,
            ("B_q",   99.0):  0.5 * width,
            ("A_q_t", 99.9):  1.5 * width,
            ("B_q",   99.9):  2.5 * width,
        }

        for condition in ["A_q_t", "B_q"]:
            for percentile in BUDGETS:
                vals = []
                for model in MODELS:
                    g = sub[
                        (sub["model"] == model)
                        & (sub["condition"] == condition)
                        & np.isclose(sub["threshold_percentile"], percentile)
                    ]
                    vals.append(
                        100.0 * float(g.iloc[0]["fnr_mean"])
                        if len(g) else np.nan
                    )

                label = f"{condition.replace('_q_t',' = q+t').replace('_q',' = q')} | P{percentile:g}"

                ax.bar(
                    x + offsets[(condition, percentile)],
                    vals,
                    width=width,
                    label=label,
                )

        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("FNR (%)" if ds == DATASETS[0] else "")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()

    fig.savefig(fig_dir / "fig_multibudget_A_vs_B_fnr.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_multibudget_A_vs_B_fnr.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_multibudget_A_vs_B_fnr.svg", bbox_inches="tight")
    plt.close(fig)


def plot_translation_fpr(
    summary: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = summary[summary["dataset"] == ds].copy()

        x = np.arange(len(MODELS))
        width = 0.24

        orig = []
        fixed = []
        recal = []

        for model in MODELS:
            g = sub[sub["model"] == model]
            if not len(g):
                orig.append(np.nan)
                fixed.append(np.nan)
                recal.append(np.nan)
                continue
            r = g.iloc[0]
            orig.append(100.0 * float(r["original_fpr_mean"]))
            fixed.append(100.0 * float(r["fixed_shifted_fpr_mean"]))
            recal.append(100.0 * float(r["recal_shifted_fpr_mean"]))

        ax.bar(x - width, orig, width=width, label="Original")
        ax.bar(x, fixed, width=width, label="Shifted | fixed threshold")
        ax.bar(x + width, recal, width=width, label="Shifted | recalibrated")

        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("FPR (%)" if ds == DATASETS[0] else "")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()

    fig.savefig(fig_dir / "fig_translation_control_fpr.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_translation_control_fpr.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_translation_control_fpr.svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.output_root
    tables = out / "tables"
    figures = out / "figures"
    metadata = out / "metadata"

    for p in [out, tables, figures, metadata]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("THRESHOLD-FREE + MULTI-BUDGET + TRANSLATION-FPR ANALYSIS START")
    print("Evidence class :", EVIDENCE_CLASS)
    print("Controlled SNN :", SNN_ROOT)
    print("Controlled AE3 :", AE3_ROOT)
    print("Audit-informed :", AUDIT_ROOT)
    print("Output root    :", out)
    print("=" * 120)

    # -------------------------------------------------------------------------
    # A. Controlled threshold-free metrics
    # -------------------------------------------------------------------------
    controlled_inventory = discover_controlled_runs()
    controlled_metrics, controlled_stats = controlled_thresholdfree(controlled_inventory)

    controlled_inventory.to_csv(
        tables / "controlled_run_inventory.csv",
        index=False,
    )
    controlled_metrics.to_csv(
        tables / "controlled_thresholdfree_run_metrics.csv",
        index=False,
    )
    controlled_stats.to_csv(
        tables / "controlled_thresholdfree_statistics_holm12.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # B. A vs B multi-budget
    # -------------------------------------------------------------------------
    budget_runs, budget_summary = audit_multibudget()
    budget_contrasts = audit_multibudget_contrasts(budget_runs)

    budget_runs.to_csv(
        tables / "audit_informed_multibudget_run_metrics.csv",
        index=False,
    )
    budget_summary.to_csv(
        tables / "audit_informed_multibudget_summary.csv",
        index=False,
    )
    budget_contrasts.to_csv(
        tables / "audit_informed_multibudget_B_minus_A_contrasts.csv",
        index=False,
    )

    # Explicit P99.9 stability warning table.
    p999_warning = (
        budget_summary[
            np.isclose(budget_summary["threshold_percentile"], 99.9)
        ][
            [
                "dataset",
                "condition",
                "calibration_n",
                "nominal_calibration_upper_tail_count",
            ]
        ]
        .drop_duplicates()
        .sort_values(["dataset", "condition"])
        .reset_index(drop=True)
    )

    p999_warning["interpretation"] = np.where(
        p999_warning["nominal_calibration_upper_tail_count"] < 5,
        "EXTREME_TAIL_SENSITIVITY_ONLY",
        "TAIL_SENSITIVITY",
    )

    p999_warning.to_csv(
        tables / "p999_calibration_tail_support.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # C. Translation FPR
    # -------------------------------------------------------------------------
    translation_runs, translation_summary = translation_control()

    translation_runs.to_csv(
        tables / "translation_global_plus_one_span_run_metrics.csv",
        index=False,
    )
    translation_summary.to_csv(
        tables / "translation_global_plus_one_span_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Manuscript compact summary
    # -------------------------------------------------------------------------
    # Controlled threshold-free differences in percentage-point-like scale for
    # readability (AUC/AP are proportions, so *100 gives points).
    controlled_manuscript = controlled_stats.copy()
    controlled_manuscript["lambda0_mean_pct"] = 100.0 * controlled_manuscript["lambda0_mean"]
    controlled_manuscript["lambda1_mean_pct"] = 100.0 * controlled_manuscript["lambda1_mean"]
    controlled_manuscript["mean_difference_points"] = 100.0 * controlled_manuscript["mean_difference"]
    controlled_manuscript["ci95_low_points"] = 100.0 * controlled_manuscript["corrected_ci95_low"]
    controlled_manuscript["ci95_high_points"] = 100.0 * controlled_manuscript["corrected_ci95_high"]

    controlled_manuscript.to_csv(
        tables / "manuscript_thresholdfree_summary.csv",
        index=False,
    )

    # P95/P99/P99.9 FNR/FPR pivot for A and B.
    manuscript_budget = budget_summary[
        [
            "dataset",
            "model",
            "condition",
            "threshold_percentile",
            "calibration_n",
            "nominal_calibration_upper_tail_count",
            "fnr_mean",
            "fpr_mean",
            "roc_auc_mean",
            "pr_auc_mean",
        ]
    ].copy()

    manuscript_budget.to_csv(
        tables / "manuscript_multibudget_summary.csv",
        index=False,
    )

    translation_summary.to_csv(
        tables / "manuscript_translation_fpr_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_multibudget_fnr(budget_summary, figures)
    plot_translation_fpr(translation_summary, figures)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    manifest = {
        "script_version": SCRIPT_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "analysis_only": True,
        "training_performed": False,
        "block_A_controlled_thresholdfree": {
            "scores": "saved non_epoch_score at lambda=0 and lambda=1",
            "metrics": ["ROC-AUC", "PR-AUC/Average Precision"],
            "difference_direction": "lambda0_minus_lambda1",
            "primary_test": "Nadeau-Bengio corrected repeated-CV t-test",
            "multiple_testing": "Holm across 12 dataset-model comparisons separately per metric family",
        },
        "block_B_audit_multibudget": {
            "conditions": ["A_q_t", "B_q"],
            "threshold_percentiles": BUDGETS,
            "threshold_source": "each condition's saved calibration_normal q_score distribution",
            "test_scores": "saved q_score on identical test rows",
            "metrics": ["FNR", "FPR", "ROC-AUC", "PR-AUC"],
            "inference_status": "descriptive paired 3-seed sensitivity; not promoted to strong inferential claim",
            "p999_note": (
                "P99.9 is retained only as extreme-tail sensitivity. "
                "Interpret according to the calibration upper-tail count table."
            ),
        },
        "block_C_translation": {
            "condition": "A_q_t",
            "scheme": "global_plus_one_fit_span",
            "weights": "fixed",
            "reports": [
                "original FNR/FPR",
                "shifted FNR/FPR with original fixed threshold",
                "shifted FNR/FPR with recalibrated shifted-calibration threshold",
            ],
            "purpose": (
                "avoid interpreting near-zero FNR change as selectivity when "
                "the original attack FNR is already at a floor"
            ),
        },
        "note": (
            "These post-hoc robustness analyses strengthen interpretation and "
            "do not replace the frozen primary controlled-intervention endpoint."
        ),
    }

    save_json(metadata / "analysis_manifest.json", manifest)

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CONTROLLED THRESHOLD-FREE METRICS: lambda0 - lambda1")
    print("=" * 120)

    cols = [
        "metric",
        "dataset",
        "model",
        "lambda0_mean",
        "lambda1_mean",
        "mean_difference",
        "corrected_ci95_low",
        "corrected_ci95_high",
        "nb_p_raw",
        "p_holm12",
        "significant_holm12_0_05",
    ]
    print(controlled_stats[cols].to_string(index=False))

    print("\n" + "=" * 120)
    print("AUDIT-INFORMED MULTI-BUDGET B - A CONTRASTS")
    print("=" * 120)

    # Focus terminal output on FNR/FPR; AUC/AP are in CSV.
    print(
        budget_contrasts[
            budget_contrasts["metric"].isin(["fnr", "fpr"])
        ].to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("P99.9 CALIBRATION TAIL SUPPORT")
    print("=" * 120)
    print(p999_warning.to_string(index=False))

    print("\n" + "=" * 120)
    print("GLOBAL +1 FIT-SPAN TRANSLATION: FNR/FPR SUMMARY")
    print("=" * 120)

    tcols = [
        "dataset",
        "model",
        "original_fnr_mean",
        "fixed_shifted_fnr_mean",
        "recal_shifted_fnr_mean",
        "delta_fnr_fixed_minus_original_mean",
        "delta_fnr_recal_minus_original_mean",
        "original_fpr_mean",
        "fixed_shifted_fpr_mean",
        "recal_shifted_fpr_mean",
        "delta_fpr_fixed_minus_original_mean",
        "delta_fpr_recal_minus_original_mean",
    ]
    print(translation_summary[tcols].to_string(index=False))

    # -------------------------------------------------------------------------
    # Checksums
    # -------------------------------------------------------------------------
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.txt"
    )

    with (out / "SHA256SUMS.txt").open("w") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(out)}\n")

    print("\n" + "=" * 120)
    print("THRESHOLD-FREE + MULTI-BUDGET + TRANSLATION-FPR ANALYSIS COMPLETE")
    print("=" * 120)
    print("Output root:", out)
    print("Controlled threshold-free:",
          tables / "manuscript_thresholdfree_summary.csv")
    print("Multi-budget:",
          tables / "manuscript_multibudget_summary.csv")
    print("Translation FPR:",
          tables / "manuscript_translation_fpr_summary.csv")
    print("P99.9 support:",
          tables / "p999_calibration_tail_support.csv")
    print("Checksums:", out / "SHA256SUMS.txt")


if __name__ == "__main__":
    main()
