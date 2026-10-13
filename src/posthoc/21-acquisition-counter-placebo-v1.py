#!/usr/bin/env python3
"""
21-temporal_audit-acquisition-counter-placebo-v1.py

Post-hoc mechanistic placebo experiment for the TEMPORAL_AUDIT temporal-shortcut study.

Scientific question
-------------------
Does the effect attributed to absolute frame.time_epoch reflect timestamp semantics,
or can an acquisition-order counter reproduce the same detector behaviour? A second
negative-control arm adds an in-support deterministic pseudo-random feature to test
whether merely adding one extra input dimension can reproduce the effect.

Conditions
----------
Reference conditions are NOT retrained here; they are loaded from the frozen
temporal_audit_temporal_upgrade_e2e_v1_2 evidence block:

  A = q + frame.time_epoch     existing absolute-time reference
  B = q                        existing packet-only reference

Two new conditions are trained:

  E = q + acquisition_counter source_row_index used as a chronology counter
  N = q + in_support_noise    deterministic pseudo-random negative placebo

Important fairness rule
-----------------------
E and N use the exact model seed that the existing A_q_t run used for the same
(dataset, architecture, nominal seed). Since A, E, and N have the same input
width and feature-slot layout (q plus one final cue), this pairs initialization
and stochastic training as closely as possible without retraining A.

The primary score remains the q-only reconstruction score. The counter/noise cue
can affect the reconstruction but does not enter the anomaly score directly.

Evidence status
---------------
POSTHOC_MECHANISTIC_PLACEBO_AFTER_PRIMARY_ANALYSES
This script does not overwrite the historical v3 evidence or the prospective
v1.2 audit-informed validation evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")

import joblib
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# FROZEN PATH / HASH CONTRACT
# =============================================================================

SCRIPT_VERSION = "1.0.0"
EVIDENCE_CLASS = "POSTHOC_MECHANISTIC_PLACEBO_AFTER_PRIMARY_ANALYSES"

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
LEGACY_ROOT = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
BASE_PIPELINE = LEGACY_ROOT / "20-end-to-end-audit-informed-v1.2.py"
SOURCE_ROOT = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2"
DEFAULT_ROOT = BASE / "temporal_audit_acquisition_counter_placebo_v1"

EXPECTED = {
    "20-end-to-end-audit-informed-v1.2.py":
        "18ec1e3ba134c5aa0db31017b00ae4eb2c0758d055dbd2c47cb7721aee07ad0b",
    "04-confirmatory-allminmax-3seed-10fold.py":
        "71aa55026845bbe40d112c7dc0b36b3a4bc00d8f76860389245c25080d6fb21f",
    "03-model-epoch-audit.py":
        "cac1a5cca1a91190967248e592a113a88b3fa71bc8dc5851d841e900005a25a2",
    "source_feature_contract_sha256":
        "17d1823b82b89fdfdb4dd5d4b2392ceb8b69f442d5e80b0050a09a47db8fa77b",
    "source_protocol_freeze_sha256":
        "f575a865b1e798b0412dfc0d1948cb2aa483dd86521ec6a148c3375a7cd31afe",
    "source_protocol_hash":
        "6921766ff0f11d825b5f0dc36fc915ad6873a627daa312b53fe11bb2ca41e45c",
}

NEW_CONDITIONS = {
    "E_q_acquisition_counter": {
        "extra_feature": "acquisition_counter",
        "role": "acquisition_order_placebo",
    },
    "N_q_in_support_noise": {
        "extra_feature": "in_support_noise",
        "role": "negative_noise_placebo",
    },
}

REFERENCE_CONDITIONS = ["A_q_t", "B_q"]
PRIMARY_CONTRASTS = {
    # delta = FNR(right) - FNR(left), matching pipeline v1.2 convention.
    "B_minus_A_timestamp_value": ("A_q_t", "B_q"),
    "B_minus_E_acquisition_counter_value": ("E_q_acquisition_counter", "B_q"),
    "B_minus_N_noise_placebo_value": ("N_q_in_support_noise", "B_q"),
    "E_minus_A_counter_vs_timestamp": ("A_q_t", "E_q_acquisition_counter"),
    "N_minus_B_noise_vs_packet_only": ("B_q", "N_q_in_support_noise"),
}

NOISE_ALGORITHM = "sha256_uniform_v1_with_fit_anchors_0_1"
ACQUISITION_COUNTER_DEFINITION = "raw source_row_index from prepared parent-table mapping"


# =============================================================================
# SMALL LOCAL HELPERS
# =============================================================================

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "||".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def hash_to_unit(*parts: Any) -> float:
    raw = "||".join(str(x) for x in parts).encode("utf-8")
    n = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return n / float(2**64 - 1)


def load_base_pipeline():
    if not BASE_PIPELINE.exists():
        raise FileNotFoundError(BASE_PIPELINE)
    spec = importlib.util.spec_from_file_location("temporal_audit_base_v12_for_placebo", BASE_PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ensure_source_hashes(base) -> None:
    checks = {
        BASE_PIPELINE: EXPECTED["20-end-to-end-audit-informed-v1.2.py"],
        base.LEGACY_FINAL_SCRIPT: EXPECTED["04-confirmatory-allminmax-3seed-10fold.py"],
        base.LEGACY_BASE_SCRIPT: EXPECTED["03-model-epoch-audit.py"],
        SOURCE_ROOT / "metadata" / "feature_contract.json": EXPECTED["source_feature_contract_sha256"],
        SOURCE_ROOT / "metadata" / "protocol_freeze.json": EXPECTED["source_protocol_freeze_sha256"],
    }
    failures = []
    for p, expected in checks.items():
        if not p.exists():
            failures.append(f"MISSING {p}")
            continue
        got = sha256_file(p)
        if got != expected:
            failures.append(f"HASH {p}: expected={expected} got={got}")
    if failures:
        raise RuntimeError("Frozen source contract failed:\n" + "\n".join(failures))

    freeze = json.loads((SOURCE_ROOT / "metadata" / "protocol_freeze.json").read_text())
    if freeze.get("protocol_hash") != EXPECTED["source_protocol_hash"]:
        raise RuntimeError(
            f"Source protocol hash changed: {freeze.get('protocol_hash')} != {EXPECTED['source_protocol_hash']}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--stage", choices=["preflight", "train", "analyze", "package", "all"], default="all")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--bootstrap-iterations", type=int, default=20000)
    return ap.parse_args()


def build_dirs(base, root: Path) -> dict[str, Path]:
    dirs = base.ensure_dirs(root)
    # Do not duplicate the prepared datasets. Symlinks are created in preflight.
    return dirs


def source_prepared_path(ds: str) -> Path:
    return SOURCE_ROOT / "prepared_data" / f"{ds}_prepared.parquet"


def output_prepared_link(dirs: dict[str, Path], ds: str) -> Path:
    return dirs["prepared"] / f"{ds}_prepared.parquet"


def augmented_prepared(base, dirs: dict[str, Path], ds: str) -> pd.DataFrame:
    p = output_prepared_link(dirs, ds)
    if not p.exists():
        raise RuntimeError(f"Prepared-data link missing: {p}; run preflight first")
    df = pd.read_parquet(p)

    required = {
        "source_row_index", "frame.time_epoch", "split", "y", "sample_local_index",
        "analysis_unit_id", "analysis_unit_kind", "capture_id", "capture_path",
        "raw_capture_id", "raw_capture_path", "flow_id", "label_type",
    }
    miss = sorted(required - set(df.columns))
    if miss:
        raise RuntimeError(f"{ds}: prepared dataset missing required columns: {miss}")

    # Acquisition counter: exact absolute parent-table row index used by v1.2 provenance.
    source_rows = pd.to_numeric(df["source_row_index"], errors="raise").to_numpy(dtype=np.int64)
    df["acquisition_counter"] = source_rows.astype(np.float64)

    # Negative placebo: cryptographic-hash pseudo-random values, deterministic per row.
    # Two fit-normal rows selected by an independent anchor hash are set to exact 0/1,
    # guaranteeing that every later random value lies inside the fit-normal support.
    noise = np.asarray(
        [hash_to_unit("noise", NOISE_ALGORITHM, ds, int(r)) for r in source_rows],
        dtype=np.float64,
    )
    fit_idx = np.flatnonzero(df["split"].astype(str).to_numpy() == "fit_normal")
    if len(fit_idx) < 2:
        raise RuntimeError(f"{ds}: fewer than two fit_normal rows")
    anchor_keys = np.asarray(
        [hash_to_unit("anchor", NOISE_ALGORITHM, ds, int(source_rows[i])) for i in fit_idx],
        dtype=np.float64,
    )
    order = np.argsort(anchor_keys, kind="mergesort")
    noise[fit_idx[order[0]]] = 0.0
    noise[fit_idx[order[1]]] = 1.0
    if np.min(noise[fit_idx]) != 0.0 or np.max(noise[fit_idx]) != 1.0:
        raise RuntimeError(f"{ds}: failed to anchor fit-normal noise support to [0,1]")
    if np.min(noise) < 0.0 or np.max(noise) > 1.0:
        raise RuntimeError(f"{ds}: placebo noise escaped [0,1]")
    df["in_support_noise"] = noise
    return df


def new_specs(q_features: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "E_q_acquisition_counter": {
            "features": list(q_features) + ["acquisition_counter"],
            "preprocess": "precision_safe",
            "role": NEW_CONDITIONS["E_q_acquisition_counter"]["role"],
        },
        "N_q_in_support_noise": {
            "features": list(q_features) + ["in_support_noise"],
            "preprocess": "precision_safe",
            "role": NEW_CONDITIONS["N_q_in_support_noise"]["role"],
        },
    }


def source_run_dir(ds: str, condition: str, model: str, seed: int) -> Path:
    return SOURCE_ROOT / "runs" / ds / condition / model / f"seed_{int(seed):02d}"


def new_run_dir(dirs: dict[str, Path], ds: str, condition: str, model: str, seed: int) -> Path:
    return dirs["runs"] / ds / condition / model / f"seed_{int(seed):02d}"


def flatten_metrics(m: dict[str, Any]) -> dict[str, Any]:
    out = {
        "dataset": m["dataset"], "condition": m["condition"], "model": m["model"],
        "seed": int(m["seed"]), "model_seed": int(m["model_seed"]),
        "input_dim": int(m["input_dim"]), "q_dim": int(m["q_dim"]),
        "params": int(m["params"]), "epochs_used": int(m["epochs_used"]),
        "q_threshold": float(m["q_threshold_p99_calibration_normal"]),
        "full_threshold": float(m["full_threshold_p99_calibration_normal"]),
    }
    out.update({f"q_{k}": v for k, v in m["q_score_metrics"].items()})
    out.update({f"full_{k}": v for k, v in m["full_score_metrics"].items()})
    return out


# =============================================================================
# PROTOCOL FREEZE / PREFLIGHT
# =============================================================================

def construct_protocol(base, args: argparse.Namespace, models: list[str], seeds: list[int]) -> dict[str, Any]:
    prepared_hashes = {
        ds: sha256_file(source_prepared_path(ds)) for ds in base.DATASET_ORDER
    }
    source_freeze = json.loads((SOURCE_ROOT / "metadata" / "protocol_freeze.json").read_text())
    protocol = {
        "script_version": SCRIPT_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "scientific_question": (
            "whether acquisition order can reproduce the absolute-timestamp reconstruction effect, "
            "with an in-support random feature as a negative placebo"
        ),
        "source_evidence_root": str(SOURCE_ROOT),
        "source_protocol_hash": source_freeze["protocol_hash"],
        "source_pipeline_sha256": sha256_file(BASE_PIPELINE),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "datasets": list(base.DATASET_ORDER),
        "models": list(models),
        "nominal_seeds": list(seeds),
        "reference_conditions": REFERENCE_CONDITIONS,
        "new_conditions": NEW_CONDITIONS,
        "primary_contrasts": PRIMARY_CONTRASTS,
        "acquisition_counter_definition": ACQUISITION_COUNTER_DEFINITION,
        "noise_algorithm": NOISE_ALGORITHM,
        "feature_slot_rule": "q followed by one cue; identical width/order pattern to A_q_t",
        "initialization_pairing": (
            "E and N deliberately use the A_q_t condition index when calling the frozen model_seed() "
            "function, yielding the exact A model_seed for each dataset/model/nominal seed"
        ),
        "training": {
            "learning_rate": float(base.LEARNING_RATE),
            "max_epochs": int(base.MAX_EPOCHS),
            "patience": int(base.PATIENCE),
            "batch_size": int(base.BATCH_SIZE),
            "normal_only": True,
            "scaler": "MinMaxScaler",
            "scaler_fit": "fit_normal only",
            "early_stopping_data": "earlystop_normal only",
            "threshold_data": "calibration_normal only",
            "operating_percentile": float(base.OPERATING_PERCENTILE),
            "primary_score": "q-only mean squared reconstruction error",
        },
        "statistics": {
            "unit": "analysis_unit_id",
            "seed_handling": "mean analysis-unit FNR across the three paired model seeds before inference",
            "effect_ci": "analysis-unit bootstrap mean difference",
            "primary_p_value": "paired analysis-unit sign-flip test",
            "secondary_test": "Wilcoxon signed-rank when >=5 units",
            "multiplicity": "Holm across 12 dataset-model tests separately within each planned contrast",
            "bootstrap_iterations": int(args.bootstrap_iterations),
        },
        "prepared_sha256": prepared_hashes,
        "claim_boundary": [
            "post-hoc mechanistic placebo after primary analyses",
            "does not replace frozen historical v3 evidence",
            "does not replace prospective v1.2 validation evidence",
            "acquisition counter is a dataset-order cue, not a semantic network feature",
            "noise is a negative control, not a deployable feature",
            "model seeds are paired stochastic replications, not independent environments",
        ],
    }
    protocol_hash = stable_json_hash(protocol)
    return {"protocol": protocol, "protocol_hash": protocol_hash}


def preflight(base, args: argparse.Namespace, root: Path, dirs: dict[str, Path], models: list[str], seeds: list[int]) -> str:
    ensure_source_hashes(base)
    freeze_obj = construct_protocol(base, args, models, seeds)
    protocol_hash = freeze_obj["protocol_hash"]
    freeze_path = dirs["metadata"] / "protocol_freeze.json"
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text())
        if old.get("protocol_hash") != protocol_hash:
            raise RuntimeError(
                f"Output root contains different protocol {old.get('protocol_hash')} != {protocol_hash}; use new --root"
            )
    else:
        base.save_json(freeze_path, freeze_obj)

    # Archive exact source code/contracts used.
    archive = dirs["source_archive"]
    archive.mkdir(parents=True, exist_ok=True)
    for p in [
        BASE_PIPELINE,
        base.LEGACY_FINAL_SCRIPT,
        base.LEGACY_BASE_SCRIPT,
        SOURCE_ROOT / "metadata" / "feature_contract.json",
        SOURCE_ROOT / "metadata" / "protocol_freeze.json",
        Path(__file__).resolve(),
    ]:
        dst = archive / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    # Link, never rewrite, the exact prepared datasets from v1.2.
    for ds in base.DATASET_ORDER:
        src = source_prepared_path(ds).resolve()
        dst = output_prepared_link(dirs, ds)
        if dst.is_symlink():
            if dst.resolve() != src:
                raise RuntimeError(f"{ds}: existing prepared-data symlink points elsewhere: {dst.resolve()}")
        elif dst.exists():
            if sha256_file(dst) != sha256_file(src):
                raise RuntimeError(f"{ds}: existing prepared copy does not match source")
        else:
            dst.symlink_to(src)

    legacy = base.load_legacy_driver()
    q_legacy, _ = base.resolve_condition_specs(legacy)
    fc = json.loads((SOURCE_ROOT / "metadata" / "feature_contract.json").read_text())
    q_features = list(fc["q_features"])
    if q_features != list(q_legacy):
        raise RuntimeError("q feature contract differs from frozen legacy-derived q")

    # Validate source A/B run completeness and the seed pairing contract.
    a_idx = base.SAFE_CONDITION_NAMES.index("A_q_t")
    source_freeze = json.loads((SOURCE_ROOT / "metadata" / "protocol_freeze.json").read_text())
    source_protocol_hash = source_freeze["protocol_hash"]
    pairing_rows = []
    for di, ds in enumerate(base.DATASET_ORDER):
        for model in models:
            mi = base.MODELS.index(model)
            for seed in seeds:
                expected_mseed = base.model_seed(di, a_idx, mi, int(seed))
                for cond in REFERENCE_CONDITIONS:
                    rd = source_run_dir(ds, cond, model, seed)
                    if not base.run_artifacts_valid(rd, source_protocol_hash):
                        raise RuntimeError(f"Invalid/missing reference run: {rd}")
                am = base.load_json(source_run_dir(ds, "A_q_t", model, seed) / "metrics.json")
                observed = int(am["model_seed"])
                if observed != expected_mseed:
                    raise RuntimeError(
                        f"A seed contract mismatch {ds} {model} seed={seed}: observed={observed} expected={expected_mseed}"
                    )
                pairing_rows.append({
                    "dataset": ds, "model": model, "seed": int(seed),
                    "a_model_seed": observed, "planned_E_model_seed": expected_mseed,
                    "planned_N_model_seed": expected_mseed,
                })
    pd.DataFrame(pairing_rows).to_csv(dirs["metadata"] / "paired_seed_contract.csv", index=False)

    # Generate feature diagnostics now, before any new model outcome is viewed.
    diag_rows = []
    vector_rows = []
    for ds in base.DATASET_ORDER:
        df = augmented_prepared(base, dirs, ds)
        fit = df[df["split"].astype(str) == "fit_normal"]
        fit_counter_min = float(fit["acquisition_counter"].min())
        fit_counter_max = float(fit["acquisition_counter"].max())
        fit_noise_min = float(fit["in_support_noise"].min())
        fit_noise_max = float(fit["in_support_noise"].max())
        for split, g in df.groupby(df["split"].astype(str), sort=False):
            counter = g["acquisition_counter"].to_numpy(dtype=float)
            noise = g["in_support_noise"].to_numpy(dtype=float)
            epoch = pd.to_numeric(g["frame.time_epoch"], errors="raise").to_numpy(dtype=float)
            y = g["y"].to_numpy(dtype=int)
            rho_ce = stats.spearmanr(counter, epoch).statistic if len(g) > 1 else np.nan
            rho_ne = stats.spearmanr(noise, epoch).statistic if len(g) > 1 else np.nan
            rho_nc = stats.spearmanr(noise, counter).statistic if len(g) > 1 else np.nan
            rho_ny = stats.spearmanr(noise, y).statistic if len(np.unique(y)) > 1 else np.nan
            diag_rows.append({
                "dataset": ds, "split": split, "n": len(g),
                "counter_min": float(np.min(counter)), "counter_max": float(np.max(counter)),
                "counter_outside_fit_support_pct": float(100*np.mean((counter < fit_counter_min) | (counter > fit_counter_max))),
                "noise_min": float(np.min(noise)), "noise_max": float(np.max(noise)),
                "noise_outside_fit_support_pct": float(100*np.mean((noise < fit_noise_min) | (noise > fit_noise_max))),
                "spearman_counter_epoch": float(rho_ce) if np.isfinite(rho_ce) else np.nan,
                "spearman_noise_epoch": float(rho_ne) if np.isfinite(rho_ne) else np.nan,
                "spearman_noise_counter": float(rho_nc) if np.isfinite(rho_nc) else np.nan,
                "spearman_noise_y": float(rho_ny) if np.isfinite(rho_ny) else np.nan,
            })
        # Persist just the generated controls keyed by immutable row identity.
        vector_rows.append(df[[
            "sample_local_index", "source_row_index", "split", "y",
            "acquisition_counter", "in_support_noise"
        ]].assign(dataset=ds))
    pd.DataFrame(diag_rows).to_csv(dirs["tables"] / "placebo_feature_diagnostics.csv", index=False)
    pd.concat(vector_rows, ignore_index=True).to_parquet(
        dirs["metadata"] / "placebo_feature_vectors.parquet", index=False
    )

    base.save_json(dirs["metadata"] / "PREFLIGHT_COMPLETE.json", {
        "status": "COMPLETED", "protocol_hash": protocol_hash,
        "expected_new_fits": len(base.DATASET_ORDER) * len(models) * len(seeds) * len(NEW_CONDITIONS),
        "new_conditions": list(NEW_CONDITIONS),
    })
    base.log(root, f"PREFLIGHT COMPLETE protocol_hash={protocol_hash}")
    return protocol_hash


def require_preflight(base, dirs: dict[str, Path], expected_hash: str | None = None) -> str:
    p = dirs["metadata"] / "protocol_freeze.json"
    m = dirs["metadata"] / "PREFLIGHT_COMPLETE.json"
    if not p.exists() or not m.exists():
        raise RuntimeError("Preflight not complete")
    obj = json.loads(p.read_text())
    ph = obj["protocol_hash"]
    if expected_hash is not None and ph != expected_hash:
        raise RuntimeError(f"Protocol hash mismatch {ph} != {expected_hash}")
    return ph


# =============================================================================
# TRAINING — ONLY E AND N
# =============================================================================

def validate_q_scaler_against_A(base, dirs: dict[str, Path], ds: str, condition: str, spec: dict[str, Any], q_features: list[str]) -> dict[str, Any]:
    new_sc = joblib.load(base.scaler_dir(dirs, ds, condition) / "scaler.pkl")
    a_sc = joblib.load(SOURCE_ROOT / "metadata" / "scalers" / ds / "A_q_t" / "scaler.pkl")
    a_features = q_features + ["frame.time_epoch"]
    new_features = list(spec["features"])
    maxima = {}
    for attr in ["data_min_", "data_max_", "scale_", "min_"]:
        av = getattr(a_sc, attr)
        nv = getattr(new_sc, attr)
        diffs = []
        for q in q_features:
            ia = a_features.index(q)
            inew = new_features.index(q)
            diffs.append(abs(float(av[ia]) - float(nv[inew])))
        maxima[attr] = float(max(diffs) if diffs else 0.0)
    max_all = max(maxima.values())
    if max_all > 1e-12:
        raise RuntimeError(f"{ds} {condition}: q-scaler mismatch vs A_q_t: {maxima}")
    return {"dataset": ds, "condition": condition, **{f"max_abs_{k}": v for k, v in maxima.items()}, "max_abs_all": max_all}


def train(base, args: argparse.Namespace, root: Path, dirs: dict[str, Path], models: list[str], seeds: list[int]) -> None:
    protocol_hash = require_preflight(base, dirs)
    legacy = base.load_legacy_driver()
    q_legacy, _ = base.resolve_condition_specs(legacy)
    q_features = list(q_legacy)
    specs = new_specs(q_features)
    a_idx = base.SAFE_CONDITION_NAMES.index("A_q_t")

    rows = []
    scaler_qc = []
    expected = len(base.DATASET_ORDER) * len(models) * len(seeds) * len(specs)
    for di, ds in enumerate(base.DATASET_ORDER):
        prepared = augmented_prepared(base, dirs, ds)
        for condition, spec in specs.items():
            scaler = base.fit_or_load_scaler(dirs, ds, condition, spec, prepared, protocol_hash)
            scaler_qc.append(validate_q_scaler_against_A(base, dirs, ds, condition, spec, q_features))
            for model in models:
                mi = base.MODELS.index(model)
                for seed in seeds:
                    # Critical: use A_q_t condition index to reproduce A's model seed exactly.
                    metrics = base._run_one_neural(
                        args, root, dirs, protocol_hash, legacy,
                        ds, di, condition, a_idx, spec, model, mi, int(seed),
                        prepared, scaler, q_features,
                    )
                    # Fail closed if paired initialization was not achieved.
                    am = base.load_json(source_run_dir(ds, "A_q_t", model, seed) / "metrics.json")
                    if int(metrics["model_seed"]) != int(am["model_seed"]):
                        raise RuntimeError(
                            f"Paired model seed failed {ds} {condition} {model} seed={seed}: "
                            f"new={metrics['model_seed']} A={am['model_seed']}"
                        )
                    rows.append(flatten_metrics(metrics))
        del prepared
        gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "placebo_run_metrics.csv", index=False)
    pd.DataFrame(scaler_qc).to_csv(dirs["tables"] / "q_scaler_consistency_vs_A.csv", index=False)
    if len(out) != expected:
        raise RuntimeError(f"New-run aggregation has {len(out)} rows, expected {expected}")
    base.save_json(dirs["metadata"] / "TRAIN_COMPLETE.json", {
        "status": "COMPLETED", "protocol_hash": protocol_hash,
        "completed_new_fits": len(out), "expected_new_fits": expected,
    })
    base.log(root, f"TRAIN COMPLETE new_fits={len(out)}")


# =============================================================================
# ANALYSIS — REFERENCES A/B + NEW E/N
# =============================================================================

def load_prediction(base, dirs: dict[str, Path], ds: str, condition: str, model: str, seed: int) -> pd.DataFrame:
    if condition in REFERENCE_CONDITIONS:
        p = source_run_dir(ds, condition, model, seed) / "test_predictions.csv.gz"
    else:
        p = new_run_dir(dirs, ds, condition, model, seed) / "test_predictions.csv.gz"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def load_metrics(base, dirs: dict[str, Path], ds: str, condition: str, model: str, seed: int) -> dict[str, Any]:
    if condition in REFERENCE_CONDITIONS:
        p = source_run_dir(ds, condition, model, seed) / "metrics.json"
    else:
        p = new_run_dir(dirs, ds, condition, model, seed) / "metrics.json"
    return base.load_json(p)


def paired_unit_rows(base, dirs: dict[str, Path], contrast: str, left: str, right: str, models: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_cols = ["source_row_index", "analysis_unit_id", "y_true"]
    for ds in base.DATASET_ORDER:
        for model in models:
            for seed in seeds:
                l = load_prediction(base, dirs, ds, left, model, seed)
                r = load_prediction(base, dirs, ds, right, model, seed)
                for c in identity_cols:
                    if c not in l.columns or c not in r.columns:
                        raise RuntimeError(f"Missing identity column {c} in {ds} {model} seed={seed} {left}/{right}")
                lm = l[["source_row_index", "analysis_unit_id", "y_true", "q_prediction", "q_score"]].copy()
                rm = r[["source_row_index", "analysis_unit_id", "y_true", "q_prediction", "q_score"]].copy()
                z = lm.merge(rm, on="source_row_index", how="inner", suffixes=("_left", "_right"), validate="one_to_one")
                if len(z) != len(l) or len(z) != len(r):
                    raise RuntimeError(f"Test identity length mismatch {contrast} {ds} {model} seed={seed}")
                if not np.array_equal(z["y_true_left"].to_numpy(), z["y_true_right"].to_numpy()):
                    raise RuntimeError(f"y identity mismatch {contrast} {ds} {model} seed={seed}")
                if not np.array_equal(z["analysis_unit_id_left"].astype(str).to_numpy(), z["analysis_unit_id_right"].astype(str).to_numpy()):
                    raise RuntimeError(f"analysis-unit identity mismatch {contrast} {ds} {model} seed={seed}")
                atk = z[z["y_true_left"].to_numpy(dtype=int) == 1].copy()
                atk["analysis_unit_id"] = atk["analysis_unit_id_left"].astype(str)
                for aid, g in atk.groupby("analysis_unit_id", sort=False):
                    fnr_l = float(np.mean(g["q_prediction_left"].to_numpy(dtype=int) == 0))
                    fnr_r = float(np.mean(g["q_prediction_right"].to_numpy(dtype=int) == 0))
                    rows.append({
                        "contrast": contrast, "dataset": ds, "model": model, "seed": int(seed),
                        "analysis_unit_id": str(aid), "attack_n": int(len(g)),
                        "fnr_left": fnr_l, "fnr_right": fnr_r,
                        "delta_fnr_right_minus_left": fnr_r - fnr_l,
                        "mean_q_score_left": float(np.mean(g["q_score_left"])),
                        "mean_q_score_right": float(np.mean(g["q_score_right"])),
                        "delta_mean_q_score_right_minus_left": float(np.mean(g["q_score_right"]) - np.mean(g["q_score_left"])),
                        "left_condition": left, "right_condition": right,
                    })
    return rows


def analyze(base, args: argparse.Namespace, root: Path, dirs: dict[str, Path], models: list[str], seeds: list[int]) -> None:
    protocol_hash = require_preflight(base, dirs)
    tmark = dirs["metadata"] / "TRAIN_COMPLETE.json"
    if not tmark.exists():
        raise RuntimeError("Training not complete")

    # Run-level condition table A/B/E/N.
    run_rows = []
    all_conditions = ["A_q_t", "B_q", "E_q_acquisition_counter", "N_q_in_support_noise"]
    for ds in base.DATASET_ORDER:
        for condition in all_conditions:
            for model in models:
                for seed in seeds:
                    m = load_metrics(base, dirs, ds, condition, model, seed)
                    run_rows.append(flatten_metrics(m))
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(dirs["tables"] / "combined_ABEN_run_metrics.csv", index=False)

    metric_summary = run_df.groupby(["dataset", "condition", "model"], as_index=False).agg(
        seeds=("seed", "nunique"),
        q_fnr_mean=("q_fnr", "mean"), q_fnr_sd=("q_fnr", "std"),
        q_fpr_mean=("q_fpr", "mean"), q_fpr_sd=("q_fpr", "std"),
        q_roc_auc_mean=("q_roc_auc", "mean"), q_pr_auc_mean=("q_pr_auc", "mean"),
        q_f1_mean=("q_f1", "mean"), q_balanced_accuracy_mean=("q_balanced_accuracy", "mean"),
    )
    metric_summary.to_csv(dirs["tables"] / "combined_ABEN_metric_summary.csv", index=False)

    # Compact mechanistic comparison table.
    mech_rows = []
    for ds in base.DATASET_ORDER:
        for model in models:
            sub = metric_summary[(metric_summary.dataset == ds) & (metric_summary.model == model)].set_index("condition")
            need = set(all_conditions)
            if not need.issubset(sub.index):
                raise RuntimeError(f"Missing condition in metric summary {ds} {model}: {need-set(sub.index)}")
            A = float(sub.loc["A_q_t", "q_fnr_mean"])
            B = float(sub.loc["B_q", "q_fnr_mean"])
            E = float(sub.loc["E_q_acquisition_counter", "q_fnr_mean"])
            N = float(sub.loc["N_q_in_support_noise", "q_fnr_mean"])
            timestamp_value = B - A
            counter_value = B - E
            noise_value = B - N
            mech_rows.append({
                "dataset": ds, "model": model,
                "A_timestamp_fnr": A, "B_q_fnr": B, "E_counter_fnr": E, "N_noise_fnr": N,
                "B_minus_A_timestamp_value": timestamp_value,
                "B_minus_E_counter_value": counter_value,
                "B_minus_N_noise_value": noise_value,
                "E_minus_A_counter_timestamp_gap": E - A,
                "N_minus_B_noise_packet_gap": N - B,
                "counter_to_timestamp_value_ratio": (
                    counter_value / timestamp_value if abs(timestamp_value) > 1e-12 else np.nan
                ),
            })
    mech = pd.DataFrame(mech_rows)
    mech.to_csv(dirs["tables"] / "mechanistic_placebo_summary.csv", index=False)

    # Analysis-unit paired contrasts.
    seed_rows = []
    for cname, (left, right) in PRIMARY_CONTRASTS.items():
        seed_rows.extend(paired_unit_rows(base, dirs, cname, left, right, models, seeds))
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(dirs["tables"] / "placebo_analysis_unit_seed_level.csv", index=False)

    unit_df = seed_df.groupby(
        ["contrast", "dataset", "model", "analysis_unit_id", "left_condition", "right_condition"],
        as_index=False,
    ).agg(
        seeds=("seed", "nunique"),
        attack_n_mean=("attack_n", "mean"),
        fnr_left=("fnr_left", "mean"),
        fnr_right=("fnr_right", "mean"),
        delta_fnr_right_minus_left=("delta_fnr_right_minus_left", "mean"),
        mean_q_score_left=("mean_q_score_left", "mean"),
        mean_q_score_right=("mean_q_score_right", "mean"),
        delta_mean_q_score_right_minus_left=("delta_mean_q_score_right_minus_left", "mean"),
    )
    unit_df.to_csv(dirs["tables"] / "placebo_analysis_unit_per_unit.csv", index=False)

    stat_rows = []
    for keys, g in unit_df.groupby(
        ["contrast", "dataset", "model", "left_condition", "right_condition"], sort=False
    ):
        contrast, ds, model, left, right = keys
        diff = g["delta_fnr_right_minus_left"].to_numpy(dtype=float)
        qdiff = g["delta_mean_q_score_right_minus_left"].to_numpy(dtype=float)
        ci_lo, ci_hi = base.bootstrap_mean_ci(
            diff, iterations=int(args.bootstrap_iterations),
            seed=stable_seed("boot", contrast, ds, model)
        )
        p_sf, sf_method, sf_n = base.signflip_pvalue(
            diff, seed=stable_seed("signflip", contrast, ds, model)
        )
        p_w = base.wilcoxon_safe(diff)
        qci_lo, qci_hi = base.bootstrap_mean_ci(
            qdiff, iterations=int(args.bootstrap_iterations),
            seed=stable_seed("qboot", contrast, ds, model)
        )
        stat_rows.append({
            "contrast": contrast, "dataset": ds, "model": model,
            "left_condition": left, "right_condition": right,
            "analysis_units": int(len(g)), "seeds_per_unit_min": int(g["seeds"].min()),
            "mean_fnr_left": float(np.mean(g["fnr_left"])),
            "mean_fnr_right": float(np.mean(g["fnr_right"])),
            "mean_delta_fnr_right_minus_left": float(np.mean(diff)),
            "median_delta_fnr_right_minus_left": float(np.median(diff)),
            "bootstrap_ci95_low": ci_lo, "bootstrap_ci95_high": ci_hi,
            "signflip_p_raw": p_sf, "signflip_method": sf_method, "signflip_n": sf_n,
            "wilcoxon_p_raw": p_w,
            "mean_delta_q_score_right_minus_left": float(np.mean(qdiff)),
            "q_score_bootstrap_ci95_low": qci_lo, "q_score_bootstrap_ci95_high": qci_hi,
        })
    stats_df = pd.DataFrame(stat_rows)
    # Holm separately within each planned contrast across dataset-model comparisons.
    stats_df["signflip_p_holm12"] = np.nan
    stats_df["wilcoxon_p_holm12"] = np.nan
    for contrast, idx in stats_df.groupby("contrast").groups.items():
        ii = list(idx)
        stats_df.loc[ii, "signflip_p_holm12"] = base.holm_adjust(stats_df.loc[ii, "signflip_p_raw"].to_numpy(dtype=float))
        stats_df.loc[ii, "wilcoxon_p_holm12"] = base.holm_adjust(stats_df.loc[ii, "wilcoxon_p_raw"].to_numpy(dtype=float))
    stats_df["signflip_significant_holm12"] = stats_df["signflip_p_holm12"] < 0.05
    stats_df.to_csv(dirs["tables"] / "placebo_contrast_statistics.csv", index=False)

    # Cross-pair descriptive agreement: does acquisition counter reproduce the timestamp pattern?
    wide = mech[[
        "dataset", "model", "B_minus_A_timestamp_value", "B_minus_E_counter_value", "B_minus_N_noise_value"
    ]].copy()
    if len(wide) >= 3:
        rho_counter = stats.spearmanr(wide["B_minus_A_timestamp_value"], wide["B_minus_E_counter_value"])
        rho_noise = stats.spearmanr(wide["B_minus_A_timestamp_value"], wide["B_minus_N_noise_value"])
    else:
        rho_counter = rho_noise = type("X", (), {"statistic": np.nan, "pvalue": np.nan})()
    base.save_json(dirs["tables"] / "cross_architecture_pattern_agreement.json", {
        "n_dataset_model_pairs": int(len(wide)),
        "spearman_timestamp_vs_counter_effect": float(rho_counter.statistic),
        "p_timestamp_vs_counter_effect": float(rho_counter.pvalue),
        "spearman_timestamp_vs_noise_effect": float(rho_noise.statistic),
        "p_timestamp_vs_noise_effect": float(rho_noise.pvalue),
        "note": "descriptive cross-pair pattern agreement; not an independent-environment inferential test",
    })

    base.save_json(dirs["metadata"] / "ANALYZE_COMPLETE.json", {
        "status": "COMPLETED", "protocol_hash": protocol_hash,
        "run_rows": int(len(run_df)), "analysis_unit_seed_rows": int(len(seed_df)),
        "contrast_rows": int(len(stats_df)),
    })

    print("\n" + "="*120)
    print("MECHANISTIC PLACEBO SUMMARY (FNR, fractions; multiply by 100 for percentage points)")
    print("="*120)
    print(mech.to_string(index=False))
    print("\n" + "="*120)
    print("PLANNED CONTRAST STATISTICS")
    print("="*120)
    cols = [
        "contrast", "dataset", "model", "mean_delta_fnr_right_minus_left",
        "bootstrap_ci95_low", "bootstrap_ci95_high", "signflip_p_holm12"
    ]
    print(stats_df[cols].to_string(index=False))
    base.log(root, "ANALYZE COMPLETE")


# =============================================================================
# PACKAGE / CHECKSUMS
# =============================================================================

def package(base, root: Path, dirs: dict[str, Path]) -> None:
    protocol_hash = require_preflight(base, dirs)
    if not (dirs["metadata"] / "ANALYZE_COMPLETE.json").exists():
        raise RuntimeError("Analysis not complete")

    # Logs remain mutable until the process exits, so they are intentionally excluded
    # from the immutable checksum inventory. Record this boundary explicitly.
    base.save_json(dirs["metadata"] / "PACKAGE_COMPLETE.json", {
        "status": "COMPLETED", "protocol_hash": protocol_hash,
        "checksum_scope": "all files except logs/*, SHA256SUMS.txt, and artifact_inventory.csv self-entry",
    })

    def immutable_files():
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if rel.parts and rel.parts[0] == "logs":
                continue
            if p.name in {"SHA256SUMS.txt", "artifact_inventory.csv"}:
                continue
            yield p

    entries = [(str(p.relative_to(root)), sha256_file(p), p.stat().st_size) for p in immutable_files()]
    inv_path = dirs["review"] / "artifact_inventory.csv"
    pd.DataFrame(entries, columns=["relative_path", "sha256", "size_bytes"]).to_csv(inv_path, index=False)

    # SHA256SUMS covers the immutable evidence plus the inventory itself.
    lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in immutable_files()]
    lines.append(f"{sha256_file(inv_path)}  {inv_path.relative_to(root)}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    base.log(root, f"PACKAGE COMPLETE immutable_files={len(lines)}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    base = load_base_pipeline()
    ensure_source_hashes(base)

    source_freeze = json.loads((SOURCE_ROOT / "metadata" / "protocol_freeze.json").read_text())
    source_models = list(source_freeze["protocol"]["models"])
    source_seeds = [int(x) for x in source_freeze["protocol"]["model_seeds"]]
    models = list(args.models) if args.models is not None else source_models
    seeds = [int(x) for x in (args.seeds if args.seeds is not None else source_seeds)]

    unknown_models = [m for m in models if m not in source_models or m not in base.MODELS]
    if unknown_models:
        raise RuntimeError(f"Unknown/non-frozen model(s): {unknown_models}")
    unknown_seeds = [s for s in seeds if s not in source_seeds]
    if unknown_seeds:
        raise RuntimeError(f"Seed(s) outside frozen source protocol: {unknown_seeds}")

    dirs = build_dirs(base, root)
    base.log(root, f"PLACEBO PIPELINE START stage={args.stage} root={root}")

    if args.stage in {"preflight", "all"}:
        preflight(base, args, root, dirs, models, seeds)
    if args.stage in {"train", "all"}:
        train(base, args, root, dirs, models, seeds)
    if args.stage in {"analyze", "all"}:
        analyze(base, args, root, dirs, models, seeds)
    if args.stage in {"package", "all"}:
        package(base, root, dirs)

    base.log(root, f"PLACEBO PIPELINE END stage={args.stage}")


if __name__ == "__main__":
    main()
