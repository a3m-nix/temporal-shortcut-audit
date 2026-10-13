#!/usr/bin/env python3
"""
Integrated Holm-12 statistical audit for the controlled temporal intervention.

Combines:
- SNN-AE: 3 dataset comparisons
- Dense-AE/LSTM-AE/CNN-AE: 9 dataset-model comparisons

Primary endpoint:
    non_epoch_fnr_lambda0_minus_lambda1

Primary inference:
    Nadeau-Bengio corrected repeated-CV t-test

Multiplicity:
    Holm correction across all 12 dataset-model comparisons

This script performs analysis only. It does not retrain any model.
"""

from pathlib import Path
import os
import hashlib
import json
import shutil

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ["TEMPORAL_AUDIT_ROOT"] + "/"
    "0-temporal-feature-audit-v4-allminmax-3seed10fold"
)

SNN_ROOT = (
    ROOT
    / "controlled_temporal_intervention"
    / "snn_3datasets_30fold_v1"
)

AE3_ROOT = (
    ROOT
    / "controlled_temporal_intervention"
    / "ae3_3datasets_30fold_v1"
)

OUT = (
    ROOT
    / "controlled_temporal_intervention"
    / "integrated_4models_holm12_v1"
)

SNN_STATS = SNN_ROOT / "tables" / "endpoint_non_epoch_statistics.csv"
AE3_STATS = AE3_ROOT / "tables" / "endpoint_non_epoch_statistics.csv"

SNN_LAMBDA = SNN_ROOT / "tables" / "dataset_lambda_summary.csv"
AE3_LAMBDA = AE3_ROOT / "tables" / "dataset_lambda_summary.csv"

EXPECTED_MODELS = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
EXPECTED_DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]
PRIMARY_ENDPOINT = "non_epoch_fnr_lambda0_minus_lambda1"
ALPHA = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def holm_adjust(p_values):
    """
    Holm step-down adjusted p-values.
    Returns multiplicity-adjusted p-values in original row order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    if m == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(m, dtype=float)

    running_max = 0.0
    for i, pv in enumerate(ranked):
        adjusted = (m - i) * pv
        running_max = max(running_max, adjusted)
        adjusted_ranked[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

for p in [SNN_STATS, AE3_STATS, SNN_LAMBDA, AE3_LAMBDA]:
    if not p.exists():
        raise FileNotFoundError(p)

if OUT.exists():
    raise FileExistsError(
        f"Integrated audit output already exists: {OUT}\n"
        "Refusing to overwrite frozen evidence."
    )

(OUT / "tables").mkdir(parents=True, exist_ok=False)
(OUT / "metadata").mkdir(parents=True, exist_ok=False)


# ---------------------------------------------------------------------------
# Load and normalize endpoint statistics
# ---------------------------------------------------------------------------

snn = pd.read_csv(SNN_STATS)
ae3 = pd.read_csv(AE3_STATS)

if "model" not in snn.columns:
    snn.insert(1, "model", "SNN_AE")

required_cols = [
    "dataset",
    "model",
    "endpoint",
    "n_paired_units",
    "lambda0_mean",
    "lambda1_mean",
    "mean_difference",
    "median_difference",
    "sd_difference",
    "corrected_se",
    "corrected_ci95_low",
    "corrected_ci95_high",
    "nb_t_stat",
    "nb_df",
    "nb_p_raw",
    "wilcoxon_stat",
    "wilcoxon_p_raw",
]

for name, df in [("SNN", snn), ("AE3", ae3)]:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} statistics missing columns: {missing}")

snn = snn[required_cols].copy()
ae3 = ae3[required_cols].copy()

combined = pd.concat([snn, ae3], ignore_index=True)

# Exact design checks.
if len(combined) != 12:
    raise RuntimeError(f"Expected 12 comparisons, found {len(combined)}")

if set(combined["dataset"]) != set(EXPECTED_DATASETS):
    raise RuntimeError(
        f"Dataset mismatch: {sorted(set(combined['dataset']))}"
    )

if set(combined["model"]) != set(EXPECTED_MODELS):
    raise RuntimeError(
        f"Model mismatch: {sorted(set(combined['model']))}"
    )

if not (combined["endpoint"] == PRIMARY_ENDPOINT).all():
    raise RuntimeError(
        "Not all rows use the frozen primary endpoint"
    )

if not (combined["n_paired_units"] == 30).all():
    raise RuntimeError(
        "Expected exactly 30 paired seed-fold units per comparison"
    )

counts = (
    combined.groupby(["dataset", "model"])
    .size()
    .reset_index(name="n")
)

if not (counts["n"] == 1).all() or len(counts) != 12:
    raise RuntimeError(
        "Expected exactly one endpoint row per dataset-model pair"
    )


# ---------------------------------------------------------------------------
# Holm across all 12 primary comparisons
# ---------------------------------------------------------------------------

combined["nb_p_holm12"] = holm_adjust(
    combined["nb_p_raw"].to_numpy(dtype=float)
)

combined["primary_significant_holm12"] = (
    combined["nb_p_holm12"] < ALPHA
)

combined["mean_difference_pp"] = (
    100.0 * combined["mean_difference"]
)
combined["ci95_low_pp"] = (
    100.0 * combined["corrected_ci95_low"]
)
combined["ci95_high_pp"] = (
    100.0 * combined["corrected_ci95_high"]
)

model_order = {m: i for i, m in enumerate(EXPECTED_MODELS)}
dataset_order = {d: i for i, d in enumerate(EXPECTED_DATASETS)}

combined["_model_order"] = combined["model"].map(model_order)
combined["_dataset_order"] = combined["dataset"].map(dataset_order)

combined = combined.sort_values(
    ["_dataset_order", "_model_order"]
).drop(
    columns=["_model_order", "_dataset_order"]
).reset_index(drop=True)

combined.to_csv(
    OUT / "tables" / "integrated_endpoint_non_epoch_statistics_holm12.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Compact manuscript-facing result matrix
# ---------------------------------------------------------------------------

matrix = combined[
    [
        "dataset",
        "model",
        "lambda0_mean",
        "lambda1_mean",
        "mean_difference",
        "mean_difference_pp",
        "ci95_low_pp",
        "ci95_high_pp",
        "nb_p_raw",
        "nb_p_holm12",
        "primary_significant_holm12",
    ]
].copy()

matrix.to_csv(
    OUT / "tables" / "integrated_endpoint_result_matrix.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Integrate lambda-response summaries for all four architectures
# ---------------------------------------------------------------------------

snn_lambda = pd.read_csv(SNN_LAMBDA)
ae3_lambda = pd.read_csv(AE3_LAMBDA)

if "model" not in snn_lambda.columns:
    snn_lambda.insert(1, "model", "SNN_AE")

lambda_combined = pd.concat(
    [snn_lambda, ae3_lambda],
    ignore_index=True,
)

if set(lambda_combined["model"]) != set(EXPECTED_MODELS):
    raise RuntimeError("Lambda-summary model set mismatch")

if set(lambda_combined["dataset"]) != set(EXPECTED_DATASETS):
    raise RuntimeError("Lambda-summary dataset set mismatch")

lambda_combined.to_csv(
    OUT / "tables" / "integrated_dataset_lambda_summary_4models.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Cross-architecture dataset summary
# ---------------------------------------------------------------------------

dataset_summary_rows = []

for dataset in EXPECTED_DATASETS:
    g = combined[combined["dataset"] == dataset]

    dataset_summary_rows.append(
        {
            "dataset": dataset,
            "n_models": int(len(g)),
            "mean_delta_fnr_pp_across_models":
                float(g["mean_difference_pp"].mean()),
            "min_delta_fnr_pp":
                float(g["mean_difference_pp"].min()),
            "max_delta_fnr_pp":
                float(g["mean_difference_pp"].max()),
            "n_significant_holm12":
                int(g["primary_significant_holm12"].sum()),
            "all_models_significant_holm12":
                bool(g["primary_significant_holm12"].all()),
            "all_corrected_ci_exclude_zero":
                bool(
                    (
                        (g["corrected_ci95_low"] > 0)
                        | (g["corrected_ci95_high"] < 0)
                    ).all()
                ),
        }
    )

dataset_summary = pd.DataFrame(dataset_summary_rows)

dataset_summary.to_csv(
    OUT / "tables" / "integrated_dataset_level_cross_architecture_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Audit metadata
# ---------------------------------------------------------------------------

source_files = {
    "snn_endpoint_statistics": {
        "path": str(SNN_STATS),
        "sha256": sha256_file(SNN_STATS),
    },
    "ae3_endpoint_statistics": {
        "path": str(AE3_STATS),
        "sha256": sha256_file(AE3_STATS),
    },
    "snn_lambda_summary": {
        "path": str(SNN_LAMBDA),
        "sha256": sha256_file(SNN_LAMBDA),
    },
    "ae3_lambda_summary": {
        "path": str(AE3_LAMBDA),
        "sha256": sha256_file(AE3_LAMBDA),
    },
}

protocol = {
    "status": "FROZEN_INTEGRATED_STATISTICAL_AUDIT",
    "analysis_only": True,
    "training_performed": False,
    "primary_endpoint": PRIMARY_ENDPOINT,
    "models": EXPECTED_MODELS,
    "datasets": EXPECTED_DATASETS,
    "comparisons": 12,
    "paired_units_per_comparison": 30,
    "primary_test": "Nadeau-Bengio corrected repeated-CV t-test",
    "multiple_testing": "Holm adjustment across all 12 dataset-model comparisons",
    "alpha": ALPHA,
    "secondary_test": "Wilcoxon signed-rank retained as robustness evidence only",
    "effect_direction": (
        "positive mean_difference means non-epoch FNR is higher at lambda=0 "
        "than at lambda=1"
    ),
    "source_files": source_files,
}

save_json(
    OUT / "metadata" / "integrated_holm12_protocol.json",
    protocol,
)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

lines = []
lines.append("# Integrated controlled temporal intervention audit")
lines.append("")
lines.append("Primary endpoint: non-epoch FNR at λ=0 minus λ=1.")
lines.append(
    "Multiplicity: Holm correction across 12 dataset-model comparisons."
)
lines.append("")
lines.append("## Cross-architecture endpoint results")
lines.append("")

for dataset in EXPECTED_DATASETS:
    lines.append(f"### {dataset}")
    g = combined[combined["dataset"] == dataset]

    for _, r in g.iterrows():
        lines.append(
            f"- {r['model']}: "
            f"ΔFNR={r['mean_difference_pp']:.2f} pp; "
            f"corrected 95% CI "
            f"[{r['ci95_low_pp']:.2f}, {r['ci95_high_pp']:.2f}] pp; "
            f"Holm-12 p={r['nb_p_holm12']:.6g}; "
            f"significant={bool(r['primary_significant_holm12'])}"
        )
    lines.append("")

lines.append("## Dataset-level cross-architecture pattern")
lines.append("")

for _, r in dataset_summary.iterrows():
    lines.append(
        f"- {r['dataset']}: "
        f"{int(r['n_significant_holm12'])}/{int(r['n_models'])} models significant "
        f"after Holm-12; mean ΔFNR across models="
        f"{r['mean_delta_fnr_pp_across_models']:.2f} pp "
        f"(range {r['min_delta_fnr_pp']:.2f} to "
        f"{r['max_delta_fnr_pp']:.2f} pp)."
    )

lines.append("")
lines.append(
    "Interpretation boundary: these results establish controlled "
    "within-model decision dependence on absolute capture time under the "
    "tested reconstruction-based architectures. They do not establish "
    "deployment generalization or causal behavior outside the evaluated "
    "datasets and protocol."
)

(OUT / "RESULTS_README.md").write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Checksums + completion marker
# ---------------------------------------------------------------------------

artifact_files = sorted(
    p for p in OUT.rglob("*")
    if p.is_file()
)

checksum_lines = []
for p in artifact_files:
    if p.name == "SHA256SUMS.txt":
        continue
    checksum_lines.append(
        f"{sha256_file(p)}  {p.relative_to(OUT)}"
    )

(OUT / "SHA256SUMS.txt").write_text(
    "\n".join(checksum_lines) + "\n",
    encoding="utf-8",
)

save_json(
    OUT / "RUN_COMPLETE.json",
    {
        "status": "PASS",
        "analysis": "integrated_4models_holm12",
        "training_performed": False,
        "n_comparisons": 12,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "holm_family_size": 12,
        "all_expected_dataset_model_pairs_present": True,
        "output_root": str(OUT),
    },
)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

print("=" * 120)
print("INTEGRATED HOLM-12 AUDIT COMPLETE")
print("=" * 120)

print(
    combined[
        [
            "dataset",
            "model",
            "mean_difference_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "nb_p_raw",
            "nb_p_holm12",
            "primary_significant_holm12",
        ]
    ].to_string(index=False)
)

print("\nDATASET-LEVEL SUMMARY")
print(dataset_summary.to_string(index=False))

print("\nOutput:", OUT)
