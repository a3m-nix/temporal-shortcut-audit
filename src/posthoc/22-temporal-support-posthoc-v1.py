#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
22-temporal_audit-temporal-support-posthoc-v1.py

Post-hoc temporal-support audit for the TEMPORAL_AUDIT temporal-shortcut manuscript.

Purpose
-------
1) Reconstruct the exact scaled frame.time_epoch used by A_q_t from the saved
   MinMaxScaler fitted on fit_normal.
2) Report the distribution of scaled time (t_tilde) for:
      - fit_normal
      - earlystop_normal
      - calibration_normal
      - test_normal
      - attack
3) Report the fraction below 0, within [0,1], and above 1.
4) Quantify distance outside fit-normal support.
5) Verify acquisition order using Spearman correlation between source_row_index
   and frame.time_epoch.
6) Quantify normal-vs-attack overlap of scaled timestamp distributions.
7) Save reproducible figure-source tables, figures, manifest, and SHA-256 hashes.

Evidence class
--------------
POSTHOC_MECHANISTIC_STRENGTHENING
This analysis does not replace the frozen primary evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
SOURCE_ROOT = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2"
DEFAULT_OUT = BASE / "temporal_audit_posthoc_strengthening_v1" / "01_temporal_support"

DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]

DATASET_LABEL = {
    "edge_iiotset": "Edge-IIoTset",
    "ciciot2023": "CICIoT2023",
    "ton_iot": "ToN-IoT",
}

PARTITION_ORDER = [
    "fit_normal",
    "earlystop_normal",
    "calibration_normal",
    "test_normal",
    "attack",
]

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


def finite_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a[np.isfinite(a)]


def percentile_summary(x: np.ndarray) -> Dict[str, float]:
    x = finite_array(x)
    if len(x) == 0:
        return {
            "n": 0,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "sd": np.nan,
        }
    qs = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "median": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def support_distance(t_scaled: np.ndarray) -> np.ndarray:
    """
    Distance to fit-normal MinMax support [0,1].
      0 when inside support,
      -t when below 0,
      t-1 when above 1.
    """
    t = np.asarray(t_scaled, dtype=np.float64)
    return np.maximum.reduce([
        np.zeros_like(t),
        -t,
        t - 1.0,
    ])


def partition_name(row_split: str, y: int) -> str | None:
    s = str(row_split)
    yy = int(y)
    if s == "fit_normal":
        return "fit_normal"
    if s == "earlystop_normal":
        return "earlystop_normal"
    if s == "calibration_normal":
        return "calibration_normal"
    if s == "test" and yy == 0:
        return "test_normal"
    if s == "test" and yy == 1:
        return "attack"
    return None


def empirical_overlap_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 200,
) -> Tuple[float, int, float, float]:
    """
    Histogram-based overlap coefficient:
        OVL = integral min(f_x, f_y) dt
    using common equal-width bins.

    Returns:
      overlap, bins_used, lower_bound, upper_bound
    """
    x = finite_array(x)
    y = finite_array(y)

    if len(x) == 0 or len(y) == 0:
        return np.nan, 0, np.nan, np.nan

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan, 0, lo, hi

    if math.isclose(lo, hi):
        return 1.0, 1, lo, hi

    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    widths = np.diff(edges)
    ovl = float(np.sum(np.minimum(hx, hy) * widths))
    # Numerical protection
    ovl = min(1.0, max(0.0, ovl))
    return ovl, bins, lo, hi


def monotonic_violation_rate(
    source_row_index: np.ndarray,
    epoch: np.ndarray,
) -> Dict[str, float]:
    """
    Sort by source_row_index and quantify adjacent epoch decreases.
    This is descriptive only; Spearman rho is the primary requested statistic.
    """
    idx = np.asarray(source_row_index, dtype=np.float64)
    t = np.asarray(epoch, dtype=np.float64)

    mask = np.isfinite(idx) & np.isfinite(t)
    idx = idx[mask]
    t = t[mask]

    if len(idx) < 2:
        return {
            "adjacent_pairs": max(0, len(idx) - 1),
            "epoch_decrease_pairs": 0,
            "epoch_equal_pairs": 0,
            "epoch_increase_pairs": 0,
            "violation_rate": np.nan,
        }

    order = np.argsort(idx, kind="mergesort")
    ts = t[order]
    d = np.diff(ts)

    dec = int(np.sum(d < 0))
    eq = int(np.sum(d == 0))
    inc = int(np.sum(d > 0))
    n = int(len(d))

    return {
        "adjacent_pairs": n,
        "epoch_decrease_pairs": dec,
        "epoch_equal_pairs": eq,
        "epoch_increase_pairs": inc,
        "violation_rate": float(dec / n) if n else np.nan,
    }


# =============================================================================
# CORE
# =============================================================================

def load_exact_scaled_timestamp(ds: str) -> Tuple[pd.DataFrame, Dict]:
    prepared_path = SOURCE_ROOT / "prepared_data" / f"{ds}_prepared.parquet"
    scaler_path = SOURCE_ROOT / "metadata" / "scalers" / ds / "A_q_t" / "scaler.pkl"
    manifest_path = SOURCE_ROOT / "metadata" / "scalers" / ds / "A_q_t" / "scaler_manifest.json"

    for p in [prepared_path, scaler_path, manifest_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    prepared = pd.read_parquet(prepared_path)
    manifest = json.loads(manifest_path.read_text())
    scaler = joblib.load(scaler_path)

    features = list(manifest.get("features", []))
    if "frame.time_epoch" not in features:
        raise RuntimeError(f"{ds}: frame.time_epoch absent from A_q_t feature contract")

    required = {"source_row_index", "frame.time_epoch", "split", "y"}
    missing = sorted(required - set(prepared.columns))
    if missing:
        raise RuntimeError(f"{ds}: missing prepared columns: {missing}")

    raw = prepared[features].to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(raw)):
        bad = np.argwhere(~np.isfinite(raw))
        raise RuntimeError(f"{ds}: nonfinite raw A_q_t matrix; first bad={bad[:10].tolist()}")

    scaled = scaler.transform(raw).astype(np.float64)
    j = features.index("frame.time_epoch")

    out = prepared.copy()
    out["t_scaled"] = scaled[:, j]
    out["support_distance"] = support_distance(out["t_scaled"].to_numpy())

    out["audit_partition"] = [
        partition_name(s, y)
        for s, y in zip(out["split"], out["y"])
    ]

    # Only rows belonging to the intended analysis partitions.
    out = out[out["audit_partition"].notna()].copy().reset_index(drop=True)

    # Sanity: fit_normal timestamp should map into [0,1] up to tiny FP tolerance.
    fit_t = out.loc[out["audit_partition"] == "fit_normal", "t_scaled"].to_numpy(dtype=np.float64)
    tol = 1e-9
    if len(fit_t) == 0:
        raise RuntimeError(f"{ds}: no fit_normal rows")
    if np.nanmin(fit_t) < -tol or np.nanmax(fit_t) > 1.0 + tol:
        raise RuntimeError(
            f"{ds}: fit_normal t_scaled outside [0,1]: "
            f"min={np.nanmin(fit_t)} max={np.nanmax(fit_t)}"
        )

    meta = {
        "dataset": ds,
        "prepared_path": str(prepared_path),
        "prepared_sha256": sha256_file(prepared_path),
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "scaler_manifest_path": str(manifest_path),
        "scaler_manifest_sha256": sha256_file(manifest_path),
        "features": features,
        "timestamp_feature_index": j,
        "timestamp_data_min_fit_normal": float(scaler.data_min_[j]),
        "timestamp_data_max_fit_normal": float(scaler.data_max_[j]),
        "timestamp_data_range_fit_normal": float(scaler.data_range_[j]),
        "timestamp_scale_parameter": float(scaler.scale_[j]),
        "timestamp_min_parameter": float(scaler.min_[j]),
    }
    return out, meta


def analyze_dataset(ds: str, out_dir: Path) -> Dict[str, pd.DataFrame]:
    df, meta = load_exact_scaled_timestamp(ds)

    # -------------------------------------------------------------------------
    # 1) Partition summaries
    # -------------------------------------------------------------------------
    partition_rows = []

    for part in PARTITION_ORDER:
        sub = df[df["audit_partition"] == part].copy()
        t = sub["t_scaled"].to_numpy(dtype=np.float64)
        d = sub["support_distance"].to_numpy(dtype=np.float64)
        raw_epoch = sub["frame.time_epoch"].to_numpy(dtype=np.float64)

        s = percentile_summary(t)
        dsupport = percentile_summary(d)
        raw_s = percentile_summary(raw_epoch)

        n = len(sub)
        below = int(np.sum(t < 0))
        within = int(np.sum((t >= 0) & (t <= 1)))
        above = int(np.sum(t > 1))
        outside = below + above

        row = {
            "dataset": ds,
            "dataset_label": DATASET_LABEL[ds],
            "partition": part,
            "n": int(n),

            "t_scaled_min": s["min"],
            "t_scaled_q01": s["q01"],
            "t_scaled_q05": s["q05"],
            "t_scaled_q25": s["q25"],
            "t_scaled_median": s["median"],
            "t_scaled_q75": s["q75"],
            "t_scaled_q95": s["q95"],
            "t_scaled_q99": s["q99"],
            "t_scaled_max": s["max"],
            "t_scaled_mean": s["mean"],
            "t_scaled_sd": s["sd"],

            "below_support_n": below,
            "within_support_n": within,
            "above_support_n": above,
            "outside_support_n": outside,
            "below_support_pct": 100.0 * below / n if n else np.nan,
            "within_support_pct": 100.0 * within / n if n else np.nan,
            "above_support_pct": 100.0 * above / n if n else np.nan,
            "outside_support_pct": 100.0 * outside / n if n else np.nan,

            "support_distance_mean": dsupport["mean"],
            "support_distance_median": dsupport["median"],
            "support_distance_q95": dsupport["q95"],
            "support_distance_q99": dsupport["q99"],
            "support_distance_max": dsupport["max"],

            "raw_epoch_min": raw_s["min"],
            "raw_epoch_median": raw_s["median"],
            "raw_epoch_max": raw_s["max"],
        }
        partition_rows.append(row)

    partition_df = pd.DataFrame(partition_rows)

    # -------------------------------------------------------------------------
    # 2) Acquisition-order monotonicity / Spearman
    # -------------------------------------------------------------------------
    spearman_rows = []

    scopes = [
        ("all_used", df),
        ("normal_all", df[df["y"].astype(int) == 0]),
        ("attack_all", df[df["y"].astype(int) == 1]),
        ("test_normal", df[df["audit_partition"] == "test_normal"]),
        ("test_attack", df[df["audit_partition"] == "attack"]),
    ]

    for scope, sub in scopes:
        idx = sub["source_row_index"].to_numpy(dtype=np.float64)
        epoch = sub["frame.time_epoch"].to_numpy(dtype=np.float64)
        mask = np.isfinite(idx) & np.isfinite(epoch)

        if np.sum(mask) >= 3:
            rho, p = spearmanr(idx[mask], epoch[mask])
            mv = monotonic_violation_rate(idx[mask], epoch[mask])
        else:
            rho, p = np.nan, np.nan
            mv = monotonic_violation_rate(idx[mask], epoch[mask])

        spearman_rows.append({
            "dataset": ds,
            "dataset_label": DATASET_LABEL[ds],
            "scope": scope,
            "n": int(np.sum(mask)),
            "spearman_rho_source_row_vs_epoch": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p_raw": float(p) if np.isfinite(p) else np.nan,
            **mv,
        })

    spearman_df = pd.DataFrame(spearman_rows)

    # -------------------------------------------------------------------------
    # 3) Normal-vs-attack timestamp overlap
    # -------------------------------------------------------------------------
    overlap_rows = []

    # Primary comparison: all normal rows versus attack test rows.
    # Secondary comparison: test-normal versus attack.
    overlap_specs = [
        (
            "all_normal_vs_attack",
            df[df["y"].astype(int) == 0]["t_scaled"].to_numpy(dtype=np.float64),
            df[df["audit_partition"] == "attack"]["t_scaled"].to_numpy(dtype=np.float64),
        ),
        (
            "test_normal_vs_attack",
            df[df["audit_partition"] == "test_normal"]["t_scaled"].to_numpy(dtype=np.float64),
            df[df["audit_partition"] == "attack"]["t_scaled"].to_numpy(dtype=np.float64),
        ),
    ]

    for scope, normal_t, attack_t in overlap_specs:
        for bins in [100, 200, 500]:
            ovl, bins_used, lo, hi = empirical_overlap_coefficient(normal_t, attack_t, bins=bins)
            overlap_rows.append({
                "dataset": ds,
                "dataset_label": DATASET_LABEL[ds],
                "scope": scope,
                "bins_requested": int(bins),
                "bins_used": int(bins_used),
                "normal_n": int(len(normal_t)),
                "attack_n": int(len(attack_t)),
                "overlap_coefficient": ovl,
                "common_min_t_scaled": lo,
                "common_max_t_scaled": hi,
            })

    overlap_df = pd.DataFrame(overlap_rows)

    # -------------------------------------------------------------------------
    # 4) Row-level figure source
    # -------------------------------------------------------------------------
    row_source_cols = [
        "sample_local_index",
        "source_row_index",
        "frame.time_epoch",
        "split",
        "audit_partition",
        "y",
        "t_scaled",
        "support_distance",
    ]
    existing_cols = [c for c in row_source_cols if c in df.columns]
    row_source = df[existing_cols].copy()
    row_source.insert(0, "dataset", ds)
    row_source.insert(1, "dataset_label", DATASET_LABEL[ds])

    # -------------------------------------------------------------------------
    # Save dataset-local outputs
    # -------------------------------------------------------------------------
    ds_dir = out_dir / "by_dataset" / ds
    ds_dir.mkdir(parents=True, exist_ok=True)

    partition_df.to_csv(ds_dir / "temporal_support_partition_summary.csv", index=False)
    spearman_df.to_csv(ds_dir / "source_row_epoch_spearman.csv", index=False)
    overlap_df.to_csv(ds_dir / "timestamp_overlap_coefficient.csv", index=False)
    row_source.to_parquet(ds_dir / "temporal_support_row_source.parquet", index=False)
    save_json(ds_dir / "source_metadata.json", meta)

    return {
        "partition": partition_df,
        "spearman": spearman_df,
        "overlap": overlap_df,
        "rows": row_source,
    }


# =============================================================================
# FIGURES
# =============================================================================

def make_figures(
    partition_all: pd.DataFrame,
    rows_all: pd.DataFrame,
    overlap_all: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure A: out-of-support share by partition
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        sub = partition_all[partition_all["dataset"] == ds].copy()
        sub["partition"] = pd.Categorical(
            sub["partition"],
            categories=PARTITION_ORDER,
            ordered=True,
        )
        sub = sub.sort_values("partition")

        x = np.arange(len(sub))
        y = sub["outside_support_pct"].to_numpy(dtype=float)

        ax.bar(x, y)
        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Fit", "ES", "Cal", "Test-N", "Attack"],
            rotation=20,
            ha="right",
            fontsize=9,
        )
        ax.set_xlabel("Partition", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Outside fit-normal timestamp support (%)", fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_temporal_support_outside_share.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_temporal_support_outside_share.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_temporal_support_outside_share.svg", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure B: scaled-time distributions, test-normal vs attack
    # Use clipped display only for visualization; source data remain unmodified.
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = rows_all[rows_all["dataset"] == ds].copy()
        n = sub[sub["audit_partition"] == "test_normal"]["t_scaled"].to_numpy(dtype=float)
        a = sub[sub["audit_partition"] == "attack"]["t_scaled"].to_numpy(dtype=float)

        combined = np.concatenate([n, a]) if len(n) and len(a) else np.array([0.0, 1.0])
        lo = float(np.nanquantile(combined, 0.005))
        hi = float(np.nanquantile(combined, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
            lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
        if math.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5

        bins = np.linspace(lo, hi, 80)
        ax.hist(n, bins=bins, density=True, alpha=0.45, label="Test normal")
        ax.hist(a, bins=bins, density=True, alpha=0.45, label="Attack")
        ax.axvline(0.0, linewidth=1.0)
        ax.axvline(1.0, linewidth=1.0)
        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xlabel(r"Scaled $frame.time\_epoch$ ($\tilde{t}$)", fontsize=9)
        ax.grid(axis="y", alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density", fontsize=10)
    axes[2].legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_temporal_support_test_normal_vs_attack.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_temporal_support_test_normal_vs_attack.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_temporal_support_test_normal_vs_attack.svg", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure C: OVL robustness across histogram resolutions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    xpos = np.arange(len(DATASETS))
    width = 0.23
    for i, bins in enumerate([100, 200, 500]):
        vals = []
        for ds in DATASETS:
            s = overlap_all[
                (overlap_all["dataset"] == ds)
                & (overlap_all["scope"] == "test_normal_vs_attack")
                & (overlap_all["bins_requested"] == bins)
            ]
            vals.append(float(s.iloc[0]["overlap_coefficient"]) if len(s) else np.nan)
        ax.bar(xpos + (i - 1) * width, vals, width=width, label=f"{bins} bins")

    ax.set_xticks(xpos)
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Timestamp overlap coefficient", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    fig.savefig(fig_dir / "fig_temporal_overlap_coefficient.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_temporal_overlap_coefficient.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_temporal_overlap_coefficient.svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for the post-hoc temporal-support audit.",
    )
    args = ap.parse_args()

    out_dir: Path = args.output_root
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    meta_dir = out_dir / "metadata"

    for p in [out_dir, tables_dir, figs_dir, meta_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("TEMPORAL SUPPORT AUDIT START")
    print("Evidence class :", EVIDENCE_CLASS)
    print("Source root    :", SOURCE_ROOT)
    print("Output root    :", out_dir)
    print("=" * 110)

    partition_frames = []
    spearman_frames = []
    overlap_frames = []
    row_frames = []

    for ds in DATASETS:
        print(f"\n[DATASET] {ds}")
        result = analyze_dataset(ds, out_dir)
        partition_frames.append(result["partition"])
        spearman_frames.append(result["spearman"])
        overlap_frames.append(result["overlap"])
        row_frames.append(result["rows"])

    partition_all = pd.concat(partition_frames, ignore_index=True)
    spearman_all = pd.concat(spearman_frames, ignore_index=True)
    overlap_all = pd.concat(overlap_frames, ignore_index=True)
    rows_all = pd.concat(row_frames, ignore_index=True)

    partition_all.to_csv(tables_dir / "temporal_support_partition_summary.csv", index=False)
    spearman_all.to_csv(tables_dir / "source_row_epoch_spearman.csv", index=False)
    overlap_all.to_csv(tables_dir / "timestamp_overlap_coefficient.csv", index=False)
    rows_all.to_parquet(tables_dir / "temporal_support_row_source.parquet", index=False)

    # Compact manuscript-oriented summary.
    summary_rows = []
    for ds in DATASETS:
        p = partition_all[partition_all["dataset"] == ds].set_index("partition")
        s = spearman_all[
            (spearman_all["dataset"] == ds) & (spearman_all["scope"] == "all_used")
        ].iloc[0]
        o = overlap_all[
            (overlap_all["dataset"] == ds)
            & (overlap_all["scope"] == "test_normal_vs_attack")
            & (overlap_all["bins_requested"] == 200)
        ].iloc[0]

        summary_rows.append({
            "dataset": ds,
            "fit_t_scaled_min": float(p.loc["fit_normal", "t_scaled_min"]),
            "fit_t_scaled_max": float(p.loc["fit_normal", "t_scaled_max"]),
            "test_normal_outside_support_pct": float(p.loc["test_normal", "outside_support_pct"]),
            "attack_outside_support_pct": float(p.loc["attack", "outside_support_pct"]),
            "attack_above_support_pct": float(p.loc["attack", "above_support_pct"]),
            "attack_below_support_pct": float(p.loc["attack", "below_support_pct"]),
            "attack_support_distance_median": float(p.loc["attack", "support_distance_median"]),
            "attack_support_distance_q95": float(p.loc["attack", "support_distance_q95"]),
            "spearman_rho_source_row_vs_epoch_all_used": float(s["spearman_rho_source_row_vs_epoch"]),
            "spearman_p_raw_all_used": float(s["spearman_p_raw"]),
            "monotonic_violation_rate_all_used": float(s["violation_rate"]),
            "test_normal_vs_attack_overlap_coefficient_200bins": float(o["overlap_coefficient"]),
        })

    manuscript_summary = pd.DataFrame(summary_rows)
    manuscript_summary.to_csv(tables_dir / "manuscript_temporal_support_summary.csv", index=False)

    make_figures(partition_all, rows_all, overlap_all, figs_dir)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "analysis_status": "POSTHOC_EXPLORATORY_MECHANISTIC_STRENGTHENING",
        "source_root": str(SOURCE_ROOT),
        "output_root": str(out_dir),
        "datasets": DATASETS,
        "timestamp_condition_replayed": "A_q_t",
        "timestamp_feature": "frame.time_epoch",
        "scaler_rule": "saved A_q_t MinMaxScaler fitted on fit_normal only",
        "support_definition_scaled": "[0,1]",
        "partition_definition": {
            "fit_normal": "split == fit_normal",
            "earlystop_normal": "split == earlystop_normal",
            "calibration_normal": "split == calibration_normal",
            "test_normal": "split == test and y == 0",
            "attack": "split == test and y == 1",
        },
        "overlap_primary": "histogram-based overlap coefficient for test_normal vs attack; 200-bin value used in compact summary",
        "overlap_sensitivity_bins": [100, 200, 500],
        "primary_order_check": "Spearman rho(source_row_index, frame.time_epoch)",
        "note": (
            "This post-hoc analysis strengthens mechanism characterization and does not replace "
            "the frozen primary controlled-intervention evidence."
        ),
    }
    save_json(meta_dir / "analysis_manifest.json", manifest)

    # SHA256 sums
    files_for_hash = sorted([
        p for p in out_dir.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.txt"
    ])
    with (out_dir / "SHA256SUMS.txt").open("w") as f:
        for p in files_for_hash:
            f.write(f"{sha256_file(p)}  {p.relative_to(out_dir)}\n")

    # Console summary
    print("\n" + "=" * 110)
    print("MANUSCRIPT TEMPORAL SUPPORT SUMMARY")
    print("=" * 110)
    print(manuscript_summary.to_string(index=False))

    print("\n" + "=" * 110)
    print("PARTITION SUPPORT SUMMARY")
    print("=" * 110)
    cols = [
        "dataset", "partition", "n",
        "t_scaled_median",
        "outside_support_pct",
        "above_support_pct",
        "below_support_pct",
        "support_distance_median",
        "support_distance_q95",
    ]
    print(partition_all[cols].to_string(index=False))

    print("\n" + "=" * 110)
    print("SOURCE ROW vs EPOCH SPEARMAN")
    print("=" * 110)
    print(
        spearman_all[
            [
                "dataset", "scope", "n",
                "spearman_rho_source_row_vs_epoch",
                "spearman_p_raw",
                "violation_rate",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("TEMPORAL SUPPORT AUDIT COMPLETE")
    print("=" * 110)
    print("Output root:", out_dir)
    print("Main summary:", tables_dir / "manuscript_temporal_support_summary.csv")
    print("Partition table:", tables_dir / "temporal_support_partition_summary.csv")
    print("Spearman table:", tables_dir / "source_row_epoch_spearman.csv")
    print("Overlap table:", tables_dir / "timestamp_overlap_coefficient.csv")
    print("Checksums:", out_dir / "SHA256SUMS.txt")


if __name__ == "__main__":
    main()
