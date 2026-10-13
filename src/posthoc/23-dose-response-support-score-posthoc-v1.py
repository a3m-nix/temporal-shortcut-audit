#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
23-temporal_audit-dose-response-support-score-posthoc-v1.py

Post-hoc mechanistic strengthening analysis for the controlled temporal
intervention.

This script DOES NOT retrain models.

It uses the saved controlled-intervention artifacts from:
- SNN-AE:  snn_3datasets_30fold_v1
- Dense/LSTM/CNN-AE: ae3_3datasets_30fold_v1

Analyses
--------
A. FNR dose-response across lambda:
   lambda = [0, .001, .0025, .005, .01, .025, .05, .10, .25, .50, 1]
   Primary plotted endpoint: non-epoch FNR.
   Intermediate lambdas are descriptive mechanistic evidence only.

B. Relationship between original timestamp distance outside the fold-specific
   fit-normal support and the common non-epoch reconstruction score s_q:
   - primary scope: attack packets at lambda=1
   - per-run Spearman rho
   - balanced run-level decile summaries
   - score-to-threshold ratio included for cross-run comparability

Evidence class
--------------
POSTHOC_MECHANISTIC_STRENGTHENING.
This analysis does not replace the frozen primary lambda=0 vs lambda=1
Nadeau-Bengio/Holm inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
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

INTEGRATED_ROOT = (
    LEGACY_ROOT
    / "controlled_temporal_intervention"
    / "integrated_4models_holm12_v1"
)

DEFAULT_OUT = (
    BASE
    / "temporal_audit_posthoc_strengthening_v1"
    / "02_dose_response_support_score"
)

DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]
MODELS = ["CNN_AE", "Dense_AE", "LSTM_AE", "SNN_AE"]

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

EXPECTED_LAMBDAS = [
    0.0, 0.001, 0.0025, 0.005, 0.01, 0.025,
    0.05, 0.10, 0.25, 0.50, 1.00,
]

EXPECTED_RUNS = 3 * 10 * 3 * 4  # dataset * seed * fold * model
EXPECTED_RUN_LAMBDA_ROWS = EXPECTED_RUNS * len(EXPECTED_LAMBDAS)

EVIDENCE_CLASS = "POSTHOC_MECHANISTIC_STRENGTHENING"
SCRIPT_VERSION = "1.0.0"


# =============================================================================
# HELPERS
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


def finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a[np.isfinite(a)]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan, np.nan, int(len(x))
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, int(len(x))
    rho, p = spearmanr(x, y)
    return float(rho), float(p), int(len(x))


def run_identity_from_manifest(run_dir: Path) -> Dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    obj = json.loads(manifest_path.read_text())

    required = ["dataset", "model", "seed", "fold"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise RuntimeError(f"{manifest_path}: missing {missing}")

    return {
        "dataset": str(obj["dataset"]),
        "model": str(obj["model"]),
        "seed": int(obj["seed"]),
        "fold": int(obj["fold"]),
        "run_dir": str(run_dir),
    }


def discover_runs() -> pd.DataFrame:
    rows = []

    for root, source_block in [
        (SNN_ROOT, "SNN"),
        (AE3_ROOT, "AE3"),
    ]:
        runs_dir = root / "runs"
        if not runs_dir.exists():
            raise FileNotFoundError(runs_dir)

        for marker in sorted(runs_dir.rglob("RUN_COMPLETE.json")):
            run_dir = marker.parent
            ident = run_identity_from_manifest(run_dir)
            ident["source_block"] = source_block
            ident["run_complete_path"] = str(marker)
            ident["run_complete_sha256"] = sha256_file(marker)
            rows.append(ident)

    run_df = pd.DataFrame(rows)

    if len(run_df) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} completed runs, found {len(run_df)}"
        )

    counts = (
        run_df.groupby(["dataset", "model"])
        .size()
        .reset_index(name="n")
    )

    if len(counts) != 12 or not (counts["n"] == 30).all():
        raise RuntimeError(
            "Expected exactly 30 seed-fold runs for each dataset-model pair.\n"
            + counts.to_string(index=False)
        )

    if set(run_df["dataset"]) != set(DATASETS):
        raise RuntimeError(f"Dataset mismatch: {sorted(run_df['dataset'].unique())}")

    if set(run_df["model"]) != set(MODELS):
        raise RuntimeError(f"Model mismatch: {sorted(run_df['model'].unique())}")

    return run_df.sort_values(
        ["dataset", "model", "seed", "fold"]
    ).reset_index(drop=True)


# =============================================================================
# DOSE RESPONSE
# =============================================================================

def collect_lambda_metrics(run_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for _, rr in run_df.iterrows():
        run_dir = Path(rr["run_dir"])
        p = run_dir / "lambda_summary.csv"
        if not p.exists():
            raise FileNotFoundError(p)

        df = pd.read_csv(p)

        required = {
            "lambda",
            "non_epoch_fnr",
            "non_epoch_recall",
            "non_epoch_fpr",
            "non_epoch_auc",
            "non_epoch_f1",
            "attack_non_epoch_score_mean",
            "attack_non_epoch_score_median",
            "normal_non_epoch_score_mean",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{p}: missing columns {missing}")

        for k in ["dataset", "model", "seed", "fold"]:
            if k not in df.columns:
                df[k] = rr[k]

        # Fail closed on identity.
        if not (df["dataset"].astype(str) == str(rr["dataset"])).all():
            raise RuntimeError(f"{p}: dataset identity mismatch")
        if not (df["model"].astype(str) == str(rr["model"])).all():
            raise RuntimeError(f"{p}: model identity mismatch")
        if not (df["seed"].astype(int) == int(rr["seed"])).all():
            raise RuntimeError(f"{p}: seed identity mismatch")
        if not (df["fold"].astype(int) == int(rr["fold"])).all():
            raise RuntimeError(f"{p}: fold identity mismatch")

        lambdas = sorted(df["lambda"].astype(float).tolist())
        expected = sorted(EXPECTED_LAMBDAS)
        if len(lambdas) != len(expected) or not np.allclose(lambdas, expected, atol=1e-12, rtol=0):
            raise RuntimeError(
                f"{p}: lambda grid mismatch\n"
                f"found={lambdas}\nexpected={expected}"
            )

        df["source_lambda_summary"] = str(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if len(out) != EXPECTED_RUN_LAMBDA_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUN_LAMBDA_ROWS} run-lambda rows, found {len(out)}"
        )

    return out


def summarize_dose_response(fold_lambda: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    monotonic_rows = []

    # Run-balanced summary for each lambda.
    for (ds, model, lam), g in fold_lambda.groupby(["dataset", "model", "lambda"]):
        vals = g["non_epoch_fnr"].to_numpy(dtype=float)
        summary_rows.append({
            "dataset": ds,
            "model": model,
            "lambda": float(lam),
            "n_seed_fold_units": int(len(vals)),
            "non_epoch_fnr_mean": float(np.mean(vals)),
            "non_epoch_fnr_sd": float(np.std(vals, ddof=1)),
            "non_epoch_fnr_median": float(np.median(vals)),
            "non_epoch_fnr_q25": float(np.quantile(vals, 0.25)),
            "non_epoch_fnr_q75": float(np.quantile(vals, 0.75)),
            "non_epoch_fpr_mean": float(g["non_epoch_fpr"].mean()),
            "non_epoch_auc_mean": float(g["non_epoch_auc"].mean()),
            "non_epoch_f1_mean": float(g["non_epoch_f1"].mean()),
            "attack_non_epoch_score_mean": float(g["attack_non_epoch_score_mean"].mean()),
            "normal_non_epoch_score_mean": float(g["normal_non_epoch_score_mean"].mean()),
        })

    # Within-run monotonicity descriptor.
    for (ds, model, seed, fold), g in fold_lambda.groupby(
        ["dataset", "model", "seed", "fold"]
    ):
        gg = g.sort_values("lambda")
        rho, p, n = safe_spearman(
            gg["lambda"].to_numpy(dtype=float),
            gg["non_epoch_fnr"].to_numpy(dtype=float),
        )

        fnr0 = float(gg.loc[np.isclose(gg["lambda"], 0.0), "non_epoch_fnr"].iloc[0])
        fnr1 = float(gg.loc[np.isclose(gg["lambda"], 1.0), "non_epoch_fnr"].iloc[0])

        diffs = np.diff(gg["non_epoch_fnr"].to_numpy(dtype=float))
        nondecreasing_steps = int(np.sum(diffs >= -1e-12))
        nonincreasing_steps = int(np.sum(diffs <= 1e-12))

        monotonic_rows.append({
            "dataset": ds,
            "model": model,
            "seed": int(seed),
            "fold": int(fold),
            "n_lambda": int(n),
            "spearman_rho_lambda_vs_fnr": rho,
            "spearman_p_raw": p,
            "fnr_lambda0": fnr0,
            "fnr_lambda1": fnr1,
            "delta_fnr_lambda0_minus_lambda1": fnr0 - fnr1,
            "nondecreasing_steps_of_10": nondecreasing_steps,
            "nonincreasing_steps_of_10": nonincreasing_steps,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(monotonic_rows)


# =============================================================================
# SUPPORT DISTANCE -> s_q
# =============================================================================

def collect_support_score_relation(
    run_df: pd.DataFrame,
    quantile_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spearman_rows = []
    bin_rows = []

    for _, rr in run_df.iterrows():
        run_dir = Path(rr["run_dir"])

        sample_path = run_dir / "sample_level_intervention.csv.gz"
        risk_path = run_dir / "sample_temporal_risk.csv.gz"
        thresholds_path = run_dir / "thresholds.json"

        for p in [sample_path, risk_path, thresholds_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        sample = pd.read_csv(sample_path)
        risk = pd.read_csv(risk_path)
        thresholds = json.loads(thresholds_path.read_text())

        required_sample = {
            "source_row_index",
            "y_true",
            "lambda",
            "non_epoch_score",
        }
        required_risk = {
            "source_row_index",
            "y_true",
            "scaled_extrapolation_distance",
            "normalized_temporal_distance",
            "scaled_epoch_original",
        }

        ms = sorted(required_sample - set(sample.columns))
        mr = sorted(required_risk - set(risk.columns))
        if ms:
            raise RuntimeError(f"{sample_path}: missing {ms}")
        if mr:
            raise RuntimeError(f"{risk_path}: missing {mr}")

        qth = float(thresholds["non_epoch_threshold"])
        if not np.isfinite(qth) or qth <= 0:
            raise RuntimeError(f"{thresholds_path}: invalid non_epoch_threshold={qth}")

        # Original timestamp endpoint only.
        s1 = sample[np.isclose(sample["lambda"].astype(float), 1.0)].copy()

        if s1["source_row_index"].duplicated().any():
            raise RuntimeError(f"{sample_path}: duplicate source_row_index at lambda=1")
        if risk["source_row_index"].duplicated().any():
            raise RuntimeError(f"{risk_path}: duplicate source_row_index")

        merged = s1.merge(
            risk[
                [
                    "source_row_index",
                    "y_true",
                    "scaled_extrapolation_distance",
                    "normalized_temporal_distance",
                    "scaled_epoch_original",
                ]
            ],
            on="source_row_index",
            how="inner",
            suffixes=("_sample", "_risk"),
            validate="one_to_one",
        )

        if len(merged) != len(s1) or len(merged) != len(risk):
            raise RuntimeError(
                f"{run_dir}: sample/risk join mismatch "
                f"sample={len(s1)} risk={len(risk)} merged={len(merged)}"
            )

        if not np.array_equal(
            merged["y_true_sample"].to_numpy(dtype=np.int8),
            merged["y_true_risk"].to_numpy(dtype=np.int8),
        ):
            raise RuntimeError(f"{run_dir}: y_true mismatch after join")

        merged["score_to_threshold"] = (
            merged["non_epoch_score"].astype(float) / qth
        )

        for scope, sub in [
            ("all_test", merged),
            ("attack", merged[merged["y_true_sample"].astype(int) == 1]),
            ("normal", merged[merged["y_true_sample"].astype(int) == 0]),
        ]:
            rho, p, n = safe_spearman(
                sub["scaled_extrapolation_distance"].to_numpy(dtype=float),
                sub["non_epoch_score"].to_numpy(dtype=float),
            )
            rho_norm, p_norm, _ = safe_spearman(
                sub["normalized_temporal_distance"].to_numpy(dtype=float),
                sub["non_epoch_score"].to_numpy(dtype=float),
            )

            spearman_rows.append({
                "dataset": rr["dataset"],
                "model": rr["model"],
                "seed": int(rr["seed"]),
                "fold": int(rr["fold"]),
                "scope": scope,
                "n": int(n),
                "spearman_rho_scaled_support_distance_vs_s_q": rho,
                "spearman_p_raw_scaled_support_distance_vs_s_q": p,
                "spearman_rho_normalized_temporal_distance_vs_s_q": rho_norm,
                "spearman_p_raw_normalized_temporal_distance_vs_s_q": p_norm,
                "non_epoch_threshold": qth,
            })

        # Balanced binned relation for attack packets.
        attack = merged[merged["y_true_sample"].astype(int) == 1].copy()
        attack = attack[
            np.isfinite(attack["scaled_extrapolation_distance"].astype(float))
            & np.isfinite(attack["non_epoch_score"].astype(float))
        ].copy()

        if len(attack) < quantile_bins:
            continue

        # Rank-first qcut makes exactly balanced quantile groups even when
        # support distances contain ties; the x-coordinate remains the actual
        # mean support distance of that bin.
        ranks = attack["scaled_extrapolation_distance"].rank(
            method="first"
        )
        attack["distance_quantile_bin"] = pd.qcut(
            ranks,
            q=quantile_bins,
            labels=False,
            duplicates="drop",
        )

        for b, g in attack.groupby("distance_quantile_bin"):
            bin_rows.append({
                "dataset": rr["dataset"],
                "model": rr["model"],
                "seed": int(rr["seed"]),
                "fold": int(rr["fold"]),
                "distance_quantile_bin": int(b) + 1,
                "n_attack": int(len(g)),
                "support_distance_mean": float(g["scaled_extrapolation_distance"].mean()),
                "support_distance_median": float(g["scaled_extrapolation_distance"].median()),
                "normalized_temporal_distance_mean": float(g["normalized_temporal_distance"].mean()),
                "s_q_mean": float(g["non_epoch_score"].mean()),
                "s_q_median": float(g["non_epoch_score"].median()),
                "score_to_threshold_mean": float(g["score_to_threshold"].mean()),
                "score_to_threshold_median": float(g["score_to_threshold"].median()),
            })

    return pd.DataFrame(spearman_rows), pd.DataFrame(bin_rows)


def summarize_support_score(
    spearman_df: pd.DataFrame,
    bin_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Primary manuscript summary: attack-only, run-level rhos.
    attack = spearman_df[spearman_df["scope"] == "attack"].copy()
    summary_rows = []

    for (ds, model), g in attack.groupby(["dataset", "model"]):
        rho = g["spearman_rho_scaled_support_distance_vs_s_q"].to_numpy(dtype=float)
        rho = rho[np.isfinite(rho)]

        summary_rows.append({
            "dataset": ds,
            "model": model,
            "n_seed_fold_units": int(len(g)),
            "n_finite_rho": int(len(rho)),
            "rho_mean": float(np.mean(rho)) if len(rho) else np.nan,
            "rho_median": float(np.median(rho)) if len(rho) else np.nan,
            "rho_q25": float(np.quantile(rho, 0.25)) if len(rho) else np.nan,
            "rho_q75": float(np.quantile(rho, 0.75)) if len(rho) else np.nan,
            "rho_min": float(np.min(rho)) if len(rho) else np.nan,
            "rho_max": float(np.max(rho)) if len(rho) else np.nan,
            "positive_rho_fraction": float(np.mean(rho > 0)) if len(rho) else np.nan,
            "rho_ge_0_3_fraction": float(np.mean(rho >= 0.3)) if len(rho) else np.nan,
        })

    rho_summary = pd.DataFrame(summary_rows)

    # Run-balanced bin summary.
    bin_summary_rows = []
    for (ds, model, b), g in bin_df.groupby(
        ["dataset", "model", "distance_quantile_bin"]
    ):
        bin_summary_rows.append({
            "dataset": ds,
            "model": model,
            "distance_quantile_bin": int(b),
            "n_seed_fold_units": int(len(g)),
            "support_distance_mean_across_runs": float(g["support_distance_mean"].mean()),
            "support_distance_median_across_runs": float(g["support_distance_median"].median()),
            "s_q_mean_across_runs": float(g["s_q_mean"].mean()),
            "s_q_median_across_runs": float(g["s_q_median"].median()),
            "score_to_threshold_mean_across_runs": float(g["score_to_threshold_mean"].mean()),
            "score_to_threshold_sd_across_runs": float(g["score_to_threshold_mean"].std(ddof=1)),
            "score_to_threshold_median_across_runs": float(g["score_to_threshold_median"].median()),
        })

    bin_summary = pd.DataFrame(bin_summary_rows)
    return rho_summary, bin_summary


# =============================================================================
# FIGURES
# =============================================================================

def plot_dose_response(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = summary[summary["dataset"] == ds].copy()

        for model in MODELS:
            g = sub[sub["model"] == model].sort_values("lambda")
            x = g["lambda"].to_numpy(dtype=float)
            y = 100.0 * g["non_epoch_fnr_mean"].to_numpy(dtype=float)
            sd = 100.0 * g["non_epoch_fnr_sd"].to_numpy(dtype=float)

            ax.plot(
                x, y,
                marker="o",
                markersize=4.5,
                linewidth=1.5,
                label=MODEL_LABEL[model],
            )
            # Descriptive ±1 SD only; not an inferential CI.
            ax.fill_between(
                x,
                y - sd,
                y + sd,
                alpha=0.10,
            )

        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xlabel(r"$\lambda$", fontsize=10)
        ax.set_xscale("symlog", linthresh=0.001, linscale=0.8)
        ax.set_xticks(EXPECTED_LAMBDAS)
        ax.set_xticklabels(
            ["0", ".001", ".0025", ".005", ".01", ".025", ".05", ".1", ".25", ".5", "1"],
            rotation=45,
            ha="right",
            fontsize=7.5,
        )
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Non-epoch FNR (%)", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_primary_lambda_fnr_dose_response.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_primary_lambda_fnr_dose_response.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_primary_lambda_fnr_dose_response.svg", bbox_inches="tight")
    plt.close(fig)


def plot_support_score(bin_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = bin_summary[bin_summary["dataset"] == ds].copy()

        for model in MODELS:
            g = sub[sub["model"] == model].sort_values("distance_quantile_bin")
            x = g["support_distance_mean_across_runs"].to_numpy(dtype=float)
            y = g["score_to_threshold_mean_across_runs"].to_numpy(dtype=float)

            # log1p x makes extreme out-of-support distances readable while
            # keeping zero exactly at zero.
            ax.plot(
                np.log1p(x),
                y,
                marker="o",
                markersize=4.5,
                linewidth=1.5,
                label=MODEL_LABEL[model],
            )

        ax.axhline(1.0, linewidth=1.0)
        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xlabel(r"$\log(1+d_{support})$", fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$s_q$ / P99 calibration threshold", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_support_distance_vs_non_epoch_score.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_support_distance_vs_non_epoch_score.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_support_distance_vs_non_epoch_score.svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quantile-bins", type=int, default=10)
    args = ap.parse_args()

    out = args.output_root
    tables = out / "tables"
    figures = out / "figures"
    metadata = out / "metadata"

    for p in [out, tables, figures, metadata]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("DOSE-RESPONSE + SUPPORT-SCORE ANALYSIS START")
    print("Evidence class :", EVIDENCE_CLASS)
    print("SNN root       :", SNN_ROOT)
    print("AE3 root       :", AE3_ROOT)
    print("Output root    :", out)
    print("=" * 118)

    run_df = discover_runs()
    run_df.to_csv(tables / "completed_run_inventory.csv", index=False)

    print(f"\nCompleted controlled-intervention runs: {len(run_df)} / {EXPECTED_RUNS}")

    # -------------------------------------------------------------------------
    # A. Dose response
    # -------------------------------------------------------------------------
    fold_lambda = collect_lambda_metrics(run_df)
    dose_summary, monotonic_run = summarize_dose_response(fold_lambda)

    fold_lambda.to_csv(tables / "fold_lambda_metrics_4models.csv", index=False)
    dose_summary.to_csv(tables / "dose_response_summary.csv", index=False)
    monotonic_run.to_csv(tables / "dose_response_run_monotonicity.csv", index=False)

    # Compare summary against already frozen integrated lambda table when present.
    integrated_lambda_path = (
        INTEGRATED_ROOT / "tables" / "integrated_dataset_lambda_summary_4models.csv"
    )
    concordance = {"integrated_lambda_table_present": integrated_lambda_path.exists()}

    if integrated_lambda_path.exists():
        frozen = pd.read_csv(integrated_lambda_path)
        keys = ["dataset", "model", "lambda"]
        check = dose_summary.merge(
            frozen[
                keys + ["non_epoch_fnr_mean"]
            ],
            on=keys,
            how="outer",
            suffixes=("_recomputed", "_frozen"),
            indicator=True,
        )
        if not (check["_merge"] == "both").all():
            raise RuntimeError("Recomputed/frozen lambda summary key mismatch")
        maxdiff = float(
            np.max(
                np.abs(
                    check["non_epoch_fnr_mean_recomputed"].to_numpy(dtype=float)
                    - check["non_epoch_fnr_mean_frozen"].to_numpy(dtype=float)
                )
            )
        )
        concordance["integrated_lambda_table_path"] = str(integrated_lambda_path)
        concordance["integrated_lambda_table_sha256"] = sha256_file(integrated_lambda_path)
        concordance["max_abs_non_epoch_fnr_mean_difference"] = maxdiff
        if maxdiff > 1e-12:
            raise RuntimeError(
                f"Dose-response recomputation differs from frozen integrated table: maxdiff={maxdiff}"
            )

    save_json(metadata / "lambda_summary_concordance.json", concordance)

    # -------------------------------------------------------------------------
    # B. Support distance -> q-score
    # -------------------------------------------------------------------------
    spearman_df, bin_df = collect_support_score_relation(
        run_df,
        quantile_bins=int(args.quantile_bins),
    )
    rho_summary, bin_summary = summarize_support_score(spearman_df, bin_df)

    spearman_df.to_csv(tables / "support_distance_s_q_run_spearman.csv", index=False)
    rho_summary.to_csv(tables / "support_distance_s_q_summary.csv", index=False)
    bin_df.to_csv(tables / "support_distance_s_q_run_bins.csv", index=False)
    bin_summary.to_csv(tables / "support_distance_s_q_bin_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Manuscript compact tables
    # -------------------------------------------------------------------------
    endpoint = dose_summary[
        dose_summary["lambda"].isin([0.0, 1.0])
    ].copy()

    endpoint_wide = endpoint.pivot(
        index=["dataset", "model"],
        columns="lambda",
        values="non_epoch_fnr_mean",
    ).reset_index()

    endpoint_wide = endpoint_wide.rename(
        columns={
            0.0: "fnr_lambda0_mean",
            1.0: "fnr_lambda1_mean",
        }
    )

    endpoint_wide["delta_fnr_lambda0_minus_lambda1_pp"] = (
        100.0
        * (
            endpoint_wide["fnr_lambda0_mean"]
            - endpoint_wide["fnr_lambda1_mean"]
        )
    )

    manuscript = endpoint_wide.merge(
        rho_summary,
        on=["dataset", "model"],
        how="left",
        validate="one_to_one",
    )

    manuscript.to_csv(
        tables / "manuscript_dose_response_support_score_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_dose_response(dose_summary, figures)
    plot_support_score(bin_summary, figures)

    # -------------------------------------------------------------------------
    # Metadata / hashes
    # -------------------------------------------------------------------------
    manifest = {
        "script_version": SCRIPT_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "analysis_only": True,
        "training_performed": False,
        "controlled_intervention_runs_expected": EXPECTED_RUNS,
        "controlled_intervention_runs_found": int(len(run_df)),
        "lambda_grid": EXPECTED_LAMBDAS,
        "primary_endpoint_for_curve": "non_epoch_fnr",
        "lambda_curve_role": (
            "descriptive mechanistic intervention-response curve; "
            "primary statistical inference remains lambda=0 vs lambda=1 "
            "Nadeau-Bengio corrected repeated-CV test with Holm-12"
        ),
        "support_distance_source": "saved sample_temporal_risk.csv.gz from each controlled-intervention run",
        "score_source": "saved lambda=1 non_epoch_score from sample_level_intervention.csv.gz",
        "support_score_primary_scope": "attack packets",
        "support_score_primary_statistic": (
            "within-run Spearman rho between scaled extrapolation distance "
            "and common non-epoch reconstruction score s_q"
        ),
        "support_score_plot": (
            "run-balanced attack quantile bins; y normalized by each run's "
            "P99 non-epoch calibration threshold for cross-run comparability"
        ),
        "quantile_bins": int(args.quantile_bins),
        "note": (
            "This post-hoc analysis strengthens mechanism characterization "
            "and does not replace the frozen primary controlled-intervention evidence."
        ),
    }
    save_json(metadata / "analysis_manifest.json", manifest)

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 118)
    print("PRIMARY DOSE-RESPONSE ENDPOINT + SUPPORT-DISTANCE SUMMARY")
    print("=" * 118)

    display_cols = [
        "dataset",
        "model",
        "fnr_lambda0_mean",
        "fnr_lambda1_mean",
        "delta_fnr_lambda0_minus_lambda1_pp",
        "rho_median",
        "rho_q25",
        "rho_q75",
        "positive_rho_fraction",
    ]

    print(manuscript[display_cols].to_string(index=False))

    print("\n" + "=" * 118)
    print("MEAN NON-EPOCH FNR (%) BY LAMBDA")
    print("=" * 118)

    d = dose_summary.copy()
    d["fnr_pct"] = 100.0 * d["non_epoch_fnr_mean"]
    pivot = d.pivot_table(
        index=["dataset", "model"],
        columns="lambda",
        values="fnr_pct",
    )
    print(pivot.round(3).to_string())

    print("\n" + "=" * 118)
    print("ATTACK SUPPORT-DISTANCE vs s_q: RUN-LEVEL SPEARMAN SUMMARY")
    print("=" * 118)
    print(rho_summary.to_string(index=False))

    # Hash all outputs
    artifact_files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    with (out / "SHA256SUMS.txt").open("w") as f:
        for p in artifact_files:
            f.write(f"{sha256_file(p)}  {p.relative_to(out)}\n")

    print("\n" + "=" * 118)
    print("DOSE-RESPONSE + SUPPORT-SCORE ANALYSIS COMPLETE")
    print("=" * 118)
    print("Output root:", out)
    print("Main summary:", tables / "manuscript_dose_response_support_score_summary.csv")
    print("Dose response:", tables / "dose_response_summary.csv")
    print("Spearman summary:", tables / "support_distance_s_q_summary.csv")
    print("Figures:", figures)
    print("Checksums:", out / "SHA256SUMS.txt")


if __name__ == "__main__":
    main()
