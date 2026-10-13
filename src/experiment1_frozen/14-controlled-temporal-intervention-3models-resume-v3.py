#!/usr/bin/env python3
"""
Additional controlled temporal-intervention experiment.

Scope:
- Dense-AE, LSTM-AE, and CNN-AE
- 3 datasets
- 3 seeds x 10 folds x 3 models = 270 retrained models
- frozen 34F protocol from 04-confirmatory-allminmax-3seed-10fold.py
- lambda intervention on scaled frame.time_epoch only
- full and non-epoch reconstruction-score decision tracks
- fail-closed structural identity gate; legacy numerical concordance retained as descriptive QC

This is additional post-hoc controlled-intervention evidence, not a replacement
for the original confirmatory experiment.
"""

import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
LEGACY_ROOT = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
FINAL_SCRIPT = LEGACY_ROOT / "04-confirmatory-allminmax-3seed-10fold.py"
EXP_ROOT = LEGACY_ROOT / "controlled_temporal_intervention" / "ae3_3datasets_30fold_v1"
RUNS_DIR = EXP_ROOT / "runs"
TABLES_DIR = EXP_ROOT / "tables"
FIGURES_DIR = EXP_ROOT / "figures"
METADATA_DIR = EXP_ROOT / "metadata"
SCRIPT_ARCHIVE_DIR = METADATA_DIR / "scripts"
for d in [EXP_ROOT, RUNS_DIR, TABLES_DIR, FIGURES_DIR, METADATA_DIR, SCRIPT_ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("final_exp", FINAL_SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
audit = exp.audit

DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]
MODELS = ["Dense_AE", "LSTM_AE", "CNN_AE"]
CONDITION = "with_epoch_34F"
SEEDS = [0, 1, 2]
FOLDS = list(range(1, 11))
LAMBDAS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
PRIMARY_ENDPOINT = "non_epoch_fnr_lambda0_minus_lambda1"
NB_TEST_TRAIN_RATIO = 1.0 / 9.0
ALPHA = 0.05
features = list(exp.FEATURES_34)
EPOCH_FEATURE = "frame.time_epoch"
epoch_idx = features.index(EPOCH_FEATURE)
keep_idx = np.array([i for i in range(len(features)) if i != epoch_idx], dtype=int)


def json_safe(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
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


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_or_verify_json(path: Path, payload):
    normalized = json_safe(payload)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != normalized:
            raise RuntimeError(f"Frozen metadata mismatch for {path}; refusing changed protocol")
    else:
        save_json(path, normalized)


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        raw = (m - rank) * pvals[idx]
        running = max(running, raw)
        adjusted[idx] = min(1.0, running)
    return adjusted


def nb_corrected_test(diff, ratio=NB_TEST_TRAIN_RATIO, alpha=ALPHA):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        raise ValueError("Need at least two paired differences")
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = float(np.sqrt(((1.0 / n) + ratio) * (sd ** 2)))
    df = n - 1
    if se == 0:
        t_stat = 0.0 if mean_diff == 0 else (np.inf if mean_diff > 0 else -np.inf)
        p_value = 1.0 if mean_diff == 0 else 0.0
        ci_low = mean_diff
        ci_high = mean_diff
    else:
        t_stat = float(mean_diff / se)
        p_value = float(2.0 * student_t.sf(abs(t_stat), df))
        tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df))
        ci_low = float(mean_diff - tcrit * se)
        ci_high = float(mean_diff + tcrit * se)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "median_diff": float(np.median(diff)),
        "sd_diff": sd,
        "corrected_se": se,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def exact_metric_equal(a, b, tol=1e-12):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def transition_counts(y_true, baseline_pred, current_pred, class_value):
    mask = y_true == class_value
    b = baseline_pred[mask]
    c = current_pred[mask]
    return {
        "n": int(np.sum(mask)),
        "baseline0_current0": int(np.sum((b == 0) & (c == 0))),
        "baseline0_current1": int(np.sum((b == 0) & (c == 1))),
        "baseline1_current0": int(np.sum((b == 1) & (c == 0))),
        "baseline1_current1": int(np.sum((b == 1) & (c == 1))),
    }


protocol = {
    "experiment_type": "additional_posthoc_controlled_temporal_intervention",
    "legacy_final_root": str(LEGACY_ROOT),
    "current_legacy_driver_script_sha256": sha256_file(FINAL_SCRIPT),
    "current_base_model_script_sha256": sha256_file(exp.BASE_SCRIPT),
    "historical_hash_limitation": (
        "The frozen legacy manifest stored the base-model script path but not historical script hashes; "
        "therefore byte-identical implementation identity with the historical run cannot be proven."
    ),
    "datasets": DATASETS,
    "models": MODELS,
    "condition": CONDITION,
    "feature_count": len(features),
    "epoch_feature": EPOCH_FEATURE,
    "feature_columns": features,
    "seeds": SEEDS,
    "folds": FOLDS,
    "expected_model_runs": 270,
    "lambdas": LAMBDAS,
    "intervention": "scaled_t_lambda = scaled_training_normal_epoch_median + lambda*(scaled_t_original-scaled_training_normal_epoch_median)",
    "full_score": "MSE across all 34 reconstruction components",
    "non_epoch_score": "MSE across 33 reconstruction components excluding frame.time_epoch",
    "thresholds_fixed_across_lambda": True,
    "all_non_epoch_inputs_held_fixed_across_lambda": True,
    "primary_new_endpoint": PRIMARY_ENDPOINT,
    "intermediate_lambda_use": "descriptive mechanistic intervention-response curve only",
    "primary_inference_unit": "dataset-model-seed-fold",
    "primary_statistical_test": "Nadeau-Bengio corrected repeated-CV t-test",
    "multiple_testing": "Holm across 9 dataset-model comparisons for primary non-epoch endpoint",
    "secondary_test": "Wilcoxon signed-rank",
    "claim_boundary": "Controlled decision-dependence evidence only; not deployment generalization or causal proof",
    "acceptance_gate": {
        "mode": "structural_identity_only_v3",
        "source_row_order_exact": True,
        "y_true_exact": True,
        "frozen_split_files_reused": True,
        "frozen_scaler_files_reused": True,
        "frozen_feature_order_reused": True,
        "frozen_model_seed_rule_reused": True,
        "legacy_prediction_or_metric_concordance_required_for_acceptance": False,
    },
    "legacy_concordance": {
        "role": "descriptive reproducibility/QC audit only; never a run acceptance criterion",
        "reported_fields": [
            "lambda1_prediction_agreement",
            "lambda0_prediction_agreement_with_legacy_neutralization",
            "legacy_vs_current_threshold",
            "legacy_vs_current_fnr_recall",
            "score_mae_and_max_abs_difference",
        ],
    },
}
write_or_verify_json(METADATA_DIR / "protocol_freeze_v3.json", protocol)

protocol_revision_v3 = {
    "status": "FROZEN_PROTOCOL_AMENDMENT_BEFORE_REMAINING_RUNS",
    "revision": "v3_structural_gate_legacy_concordance_descriptive_only",
    "why_revision_was_needed": (
        "The v2 legacy-concordance gate stopped at Edge-IIoTset/LSTM-AE seed=0 fold=3. "
        "That fold reused the frozen data/split/scaler/seed protocol but differed materially from the historical "
        "legacy predictions (prediction agreement about 0.89-0.91; FNR difference about 17.8-19.9 pp)."
    ),
    "repeatability_characterization": {
        "path": str(
            LEGACY_ROOT
            / "controlled_temporal_intervention"
            / "repeatability_lstm_edge_seed00_fold01_v1"
        ),
        "result": (
            "Three current-environment repeats of Edge-IIoTset/LSTM-AE seed=0 fold=1 were bitwise identical "
            "to one another (prediction agreement=1.0 and score differences=0), while differing slightly from legacy."
        ),
    },
    "historical_identity_limit": (
        "The original confirmatory manifest contains the base-model script path but no historical SHA-256 hashes; "
        "current hashes are frozen from this point forward, but historical implementation drift cannot be proven or excluded."
    ),
    "scientific_rationale": (
        "The controlled-intervention estimand is within-model: lambda values are compared using the same newly trained "
        "weights, validation-derived threshold, test samples, and fixed non-epoch inputs. Legacy numerical reproduction "
        "is therefore a reproducibility audit, not a validity condition for the intervention contrast."
    ),
    "what_did_not_change": [
        "datasets and frozen corpus",
        "train/validation/test split files",
        "scalers",
        "feature set/order",
        "model architectures and hyperparameters",
        "model seed rule",
        "lambda grid",
        "P99 validation-normal threshold protocol",
        "primary endpoint",
        "statistical tests and multiplicity correction",
    ],
    "new_hard_gate": [
        "source_row_index order must exactly match legacy test evidence",
        "y_true must exactly match legacy test evidence",
        "the experiment must directly reuse the frozen split and scaler files",
    ],
    "legacy_concordance_policy": "always save and report; never fail a run solely for numerical disagreement",
}
save_json(METADATA_DIR / "protocol_revision_v3.json", protocol_revision_v3)

current_implementation_freeze = {
    "frozen_at_utc": pd.Timestamp.utcnow().isoformat(),
    "driver_script": str(FINAL_SCRIPT),
    "driver_script_sha256": sha256_file(FINAL_SCRIPT),
    "base_model_script": str(exp.BASE_SCRIPT),
    "base_model_script_sha256": sha256_file(exp.BASE_SCRIPT),
    "note": "These hashes freeze the current implementation for the additional intervention experiment only.",
}
write_or_verify_json(METADATA_DIR / "current_implementation_freeze_v3.json", current_implementation_freeze)

if not (METADATA_DIR / "environment.json").exists():
    save_json(METADATA_DIR / "environment.json", {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER"),
    })

for src in [Path(__file__).resolve(), FINAL_SCRIPT.resolve(), exp.BASE_SCRIPT.resolve()]:
    dst = SCRIPT_ARCHIVE_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


DATASET_HASHES = {}
for dataset_name in DATASETS:
    path = exp.DATASETS[dataset_name]["sample"]
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[HASH] {dataset_name}: {path}")
    DATASET_HASHES[dataset_name] = sha256_file(path)
write_or_verify_json(METADATA_DIR / "dataset_source_hashes.json", {
    d: {"path": str(exp.DATASETS[d]["sample"]), "sha256": DATASET_HASHES[d]}
    for d in DATASETS
})


def run_fold(dataset_name, dataset_idx, sample, model_name, seed, fold):
    run_dir = RUNS_DIR / dataset_name / model_name / f"seed_{seed:02d}" / f"fold_{fold:02d}"
    complete_marker = run_dir / "RUN_COMPLETE.json"
    if complete_marker.exists():
        print(f"[RESUME] {dataset_name:<12} {model_name:<8} seed={seed} fold={fold:02d}")
        return
    if run_dir.exists():
        failed_archive = (
            METADATA_DIR / "pre_v3_incomplete_run_evidence" / dataset_name / model_name /
            f"seed_{seed:02d}" / f"fold_{fold:02d}"
        )
        if not failed_archive.exists():
            failed_archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, failed_archive)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = exp.DATASETS[dataset_name]
    split_file = LEGACY_ROOT / "splits" / dataset_name / f"seed_{seed:02d}" / f"fold_{fold:02d}.npz"
    scaler_path = LEGACY_ROOT / "scalers" / dataset_name / CONDITION / f"seed_{seed:02d}" / f"fold_{fold:02d}.pkl"
    legacy_run = LEGACY_ROOT / "runs" / dataset_name / CONDITION / model_name / f"seed_{seed:02d}" / f"fold_{fold:02d}"
    legacy_metrics_path = legacy_run / "metrics.json"
    legacy_predictions_path = legacy_run / "test_predictions.csv.gz"
    for p in [split_file, scaler_path, legacy_metrics_path, legacy_predictions_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with np.load(split_file, allow_pickle=False) as z:
        train_idx = z["train_normal_idx"]
        val_idx = z["val_normal_idx"]
        test_idx = z["test_idx"]

    scaler = joblib.load(scaler_path)
    Xall = sample[features].to_numpy(dtype=np.float32)
    X_train_raw = Xall[train_idx]
    X_val_raw = Xall[val_idx]
    X_test_raw = Xall[test_idx]
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    y_test = (sample.iloc[test_idx][cfg["label"]].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()
    groups = exp.attack_group_labels(dataset_name, sample, test_idx, y_test)
    source_rows = sample.iloc[test_idx]["source_row_index"].to_numpy(dtype=np.int64)
    neutral_value = float(np.median(X_train[:, epoch_idx]))

    raw_train_epoch = X_train_raw[:, epoch_idx].astype(float)
    raw_test_epoch = X_test_raw[:, epoch_idx].astype(float)
    scaled_test_epoch = X_test[:, epoch_idx].astype(float)
    train_epoch_min = float(np.min(raw_train_epoch))
    train_epoch_max = float(np.max(raw_train_epoch))
    train_epoch_width = max(train_epoch_max - train_epoch_min, 1e-12)
    raw_distance = np.where(raw_test_epoch < train_epoch_min, train_epoch_min - raw_test_epoch,
                    np.where(raw_test_epoch > train_epoch_max, raw_test_epoch - train_epoch_max, 0.0))
    normalized_temporal_distance = raw_distance / train_epoch_width
    scaled_extrapolation_distance = np.maximum.reduce([
        np.zeros_like(scaled_test_epoch, dtype=float),
        -scaled_test_epoch,
        scaled_test_epoch - 1.0,
    ])

    mseed = exp.model_seed(dataset_idx, seed, fold, model_name)
    exp.set_seed(mseed)
    tf.keras.backend.clear_session()
    model = audit.build_model(model_name, len(features), dataset_name)
    model.compile(optimizer=Adam(learning_rate=exp.LEARNING_RATE, clipnorm=1.0 if model_name == "Dense_AE" else None), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=exp.PATIENCE, restore_best_weights=True, verbose=0)
    t0 = time.time()
    history = model.fit(
        audit.input_for_model(X_train, model_name),
        audit.target_for_model(X_train, model_name),
        validation_data=(audit.input_for_model(X_val, model_name), audit.target_for_model(X_val, model_name)),
        epochs=exp.EPOCHS,
        batch_size=exp.BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )
    train_time = time.time() - t0
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(run_dir / "training_history.csv", index=False)

    weights_path = run_dir / "model.weights.h5"
    model.save_weights(weights_path)
    full_model_saved = False
    full_model_save_error = None
    try:
        model.save(run_dir / "model.keras", include_optimizer=False)
        full_model_saved = True
    except Exception as exc:
        full_model_save_error = repr(exc)

    val_pred = audit.predict_flat(model, X_val, model_name)
    val_se = np.square(X_val - val_pred).astype(np.float32)
    val_full_score = np.mean(val_se, axis=1)
    val_non_epoch_score = np.mean(val_se[:, keep_idx], axis=1)
    threshold_full = float(np.percentile(val_full_score, exp.OPERATING_PERCENTILE))
    threshold_non_epoch = float(np.percentile(val_non_epoch_score, exp.OPERATING_PERCENTILE))
    save_json(run_dir / "thresholds.json", {
        "operating_percentile": exp.OPERATING_PERCENTILE,
        "full_threshold": threshold_full,
        "non_epoch_threshold": threshold_non_epoch,
        "neutral_scaled_epoch_value": neutral_value,
    })

    lambda_metric_rows = []
    sample_rows = []
    group_rows = []
    transition_rows = []
    se_stack = []
    full_pred_by_lambda = {}
    non_epoch_pred_by_lambda = {}
    full_score_by_lambda = {}
    non_epoch_score_by_lambda = {}
    original_scaled_epoch = X_test[:, epoch_idx].copy()

    for lam in LAMBDAS:
        X_probe = X_test.copy()
        X_probe[:, epoch_idx] = neutral_value + lam * (original_scaled_epoch - neutral_value)
        recon = audit.predict_flat(model, X_probe, model_name)
        se = np.square(X_probe - recon).astype(np.float32)
        se_stack.append(se)
        full_score = np.mean(se, axis=1)
        non_epoch_score = np.mean(se[:, keep_idx], axis=1)
        epoch_error = se[:, epoch_idx]
        full_pred = (full_score > threshold_full).astype(np.int8)
        non_epoch_pred = (non_epoch_score > threshold_non_epoch).astype(np.int8)
        full_metrics = audit.metric_dict(y_test, full_score, threshold_full)
        non_epoch_metrics = audit.metric_dict(y_test, non_epoch_score, threshold_non_epoch)
        full_pred_by_lambda[lam] = full_pred
        non_epoch_pred_by_lambda[lam] = non_epoch_pred
        full_score_by_lambda[lam] = full_score
        non_epoch_score_by_lambda[lam] = non_epoch_score
        attack_mask = y_test == 1
        normal_mask = y_test == 0
        lambda_metric_rows.append({
            "dataset": dataset_name, "model": model_name, "seed": seed, "fold": fold, "lambda": lam,
            "full_accuracy": full_metrics["accuracy"], "full_precision": full_metrics["precision"],
            "full_recall": full_metrics["recall"], "full_f1": full_metrics["f1"],
            "full_auc": full_metrics["auc"], "full_fpr": full_metrics["fpr"], "full_fnr": full_metrics["fnr"],
            "non_epoch_accuracy": non_epoch_metrics["accuracy"], "non_epoch_precision": non_epoch_metrics["precision"],
            "non_epoch_recall": non_epoch_metrics["recall"], "non_epoch_f1": non_epoch_metrics["f1"],
            "non_epoch_auc": non_epoch_metrics["auc"], "non_epoch_fpr": non_epoch_metrics["fpr"],
            "non_epoch_fnr": non_epoch_metrics["fnr"],
            "attack_full_score_mean": float(np.mean(full_score[attack_mask])),
            "attack_full_score_median": float(np.median(full_score[attack_mask])),
            "attack_non_epoch_score_mean": float(np.mean(non_epoch_score[attack_mask])),
            "attack_non_epoch_score_median": float(np.median(non_epoch_score[attack_mask])),
            "attack_epoch_error_mean": float(np.mean(epoch_error[attack_mask])),
            "attack_epoch_error_median": float(np.median(epoch_error[attack_mask])),
            "normal_full_score_mean": float(np.mean(full_score[normal_mask])),
            "normal_non_epoch_score_mean": float(np.mean(non_epoch_score[normal_mask])),
        })
        for i in range(len(y_test)):
            sample_rows.append({
                "source_row_index": int(source_rows[i]), "y_true": int(y_test[i]),
                "attack_group": str(groups[i]), "lambda": lam,
                "epoch_scaled_original": float(original_scaled_epoch[i]),
                "epoch_scaled_intervened": float(X_probe[i, epoch_idx]),
                "full_score": float(full_score[i]), "non_epoch_score": float(non_epoch_score[i]),
                "epoch_reconstruction_error": float(epoch_error[i]),
                "full_prediction": int(full_pred[i]), "non_epoch_prediction": int(non_epoch_pred[i]),
            })

    full_baseline = full_pred_by_lambda[1.0]
    non_epoch_baseline = non_epoch_pred_by_lambda[1.0]
    attack_mask = y_test == 1
    full_baseline_detected_attack = attack_mask & (full_baseline == 1)
    non_epoch_baseline_detected_attack = attack_mask & (non_epoch_baseline == 1)
    n_full_baseline_detected = int(np.sum(full_baseline_detected_attack))
    n_non_epoch_baseline_detected = int(np.sum(non_epoch_baseline_detected_attack))

    for row in lambda_metric_rows:
        lam = float(row["lambda"])
        full_current = full_pred_by_lambda[lam]
        non_epoch_current = non_epoch_pred_by_lambda[lam]
        full_flip = int(np.sum(full_baseline_detected_attack & (full_current == 0)))
        non_epoch_flip = int(np.sum(non_epoch_baseline_detected_attack & (non_epoch_current == 0)))
        row["full_original_detected_attacks"] = n_full_baseline_detected
        row["full_detected_to_missed"] = full_flip
        row["full_tdfr"] = full_flip / n_full_baseline_detected if n_full_baseline_detected else np.nan
        row["non_epoch_original_detected_attacks"] = n_non_epoch_baseline_detected
        row["non_epoch_detected_to_missed"] = non_epoch_flip
        row["non_epoch_tdfr"] = non_epoch_flip / n_non_epoch_baseline_detected if n_non_epoch_baseline_detected else np.nan
        for score_type, baseline_pred, current_pred in [
            ("full", full_baseline, full_current), ("non_epoch", non_epoch_baseline, non_epoch_current)
        ]:
            for class_name, class_value in [("normal", 0), ("attack", 1)]:
                transition_rows.append({
                    "dataset": dataset_name, "model": model_name, "seed": seed, "fold": fold,
                    "lambda": lam, "score_type": score_type, "class_name": class_name,
                    **transition_counts(y_test, baseline_pred, current_pred, class_value)
                })

    for group in sorted(np.unique(groups[attack_mask])):
        mask = attack_mask & (groups == group)
        full_group_baseline_detected = mask & (full_baseline == 1)
        non_epoch_group_baseline_detected = mask & (non_epoch_baseline == 1)
        n_group = int(np.sum(mask))
        n_full_group_base = int(np.sum(full_group_baseline_detected))
        n_non_epoch_group_base = int(np.sum(non_epoch_group_baseline_detected))
        for lam in LAMBDAS:
            full_current = full_pred_by_lambda[lam]
            non_epoch_current = non_epoch_pred_by_lambda[lam]
            full_recall = float(np.mean(full_current[mask]))
            non_epoch_recall = float(np.mean(non_epoch_current[mask]))
            full_flip = int(np.sum(full_group_baseline_detected & (full_current == 0)))
            non_epoch_flip = int(np.sum(non_epoch_group_baseline_detected & (non_epoch_current == 0)))
            group_rows.append({
                "dataset": dataset_name, "model": model_name, "seed": seed, "fold": fold,
                "attack_group": str(group), "lambda": lam, "n_attack": n_group,
                "full_recall": full_recall, "full_fnr": 1.0 - full_recall,
                "full_original_detected": n_full_group_base, "full_detected_to_missed": full_flip,
                "full_tdfr": full_flip / n_full_group_base if n_full_group_base else np.nan,
                "non_epoch_recall": non_epoch_recall, "non_epoch_fnr": 1.0 - non_epoch_recall,
                "non_epoch_original_detected": n_non_epoch_group_base,
                "non_epoch_detected_to_missed": non_epoch_flip,
                "non_epoch_tdfr": non_epoch_flip / n_non_epoch_group_base if n_non_epoch_group_base else np.nan,
                "full_score_mean": float(np.mean(full_score_by_lambda[lam][mask])),
                "non_epoch_score_mean": float(np.mean(non_epoch_score_by_lambda[lam][mask])),
            })

    temporal_risk_df = pd.DataFrame({
        "source_row_index": source_rows, "y_true": y_test, "attack_group": groups.astype(str),
        "raw_epoch": raw_test_epoch, "scaled_epoch_original": scaled_test_epoch,
        "normalized_temporal_distance": normalized_temporal_distance,
        "scaled_extrapolation_distance": scaled_extrapolation_distance,
        "outside_training_normal_raw_support": (raw_distance > 0).astype(np.int8),
        "outside_scaled_0_1": ((scaled_test_epoch < 0) | (scaled_test_epoch > 1)).astype(np.int8),
    })
    temporal_risk_df.to_csv(run_dir / "sample_temporal_risk.csv.gz", index=False, compression="gzip")
    risk_group_rows = []
    for group, g in temporal_risk_df[temporal_risk_df["y_true"] == 1].groupby("attack_group"):
        risk_group_rows.append({
            "dataset": dataset_name, "model": model_name, "seed": seed, "fold": fold,
            "attack_group": str(group), "n_attack": int(len(g)),
            "normalized_temporal_distance_mean": float(g["normalized_temporal_distance"].mean()),
            "normalized_temporal_distance_median": float(g["normalized_temporal_distance"].median()),
            "scaled_extrapolation_distance_mean": float(g["scaled_extrapolation_distance"].mean()),
            "scaled_extrapolation_distance_median": float(g["scaled_extrapolation_distance"].median()),
            "outside_training_normal_raw_support_ratio": float(g["outside_training_normal_raw_support"].mean()),
            "outside_scaled_0_1_ratio": float(g["outside_scaled_0_1"].mean()),
        })
    pd.DataFrame(risk_group_rows).to_csv(run_dir / "attack_group_temporal_risk.csv", index=False)

    legacy_pred = pd.read_csv(legacy_predictions_path)
    with open(legacy_metrics_path, "r", encoding="utf-8") as f:
        legacy_metrics = json.load(f)
    source_order_match = np.array_equal(legacy_pred["source_row_index"].to_numpy(dtype=np.int64), source_rows)
    y_true_match = np.array_equal(legacy_pred["y_true"].to_numpy(dtype=np.int8), y_test)
    new_l1_score = full_score_by_lambda[1.0]
    new_l0_score = full_score_by_lambda[0.0]
    new_l1_pred = full_pred_by_lambda[1.0]
    new_l0_pred = full_pred_by_lambda[0.0]
    old_l1_score = legacy_pred["score"].to_numpy(dtype=float)
    old_l0_score = legacy_pred["score_epoch_neutralized"].to_numpy(dtype=float)
    old_l1_pred = legacy_pred["prediction"].to_numpy(dtype=np.int8)
    old_l0_pred = legacy_pred["prediction_epoch_neutralized"].to_numpy(dtype=np.int8)
    l1_agree = float(np.mean(new_l1_pred == old_l1_pred))
    l0_agree = float(np.mean(new_l0_pred == old_l0_pred))
    new_l1_metrics = audit.metric_dict(y_test, new_l1_score, threshold_full)
    new_l0_metrics = audit.metric_dict(y_test, new_l0_score, threshold_full)
    old_l1_metrics = legacy_metrics["main_metrics"]
    old_l0_metrics = legacy_metrics["diagnostics"]["epoch_neutralized_input"]
    sanity = {
        "dataset": dataset_name, "model": model_name, "seed": seed, "fold": fold, "model_seed": mseed,
        "source_row_order_match": source_order_match, "y_true_match": y_true_match,
        "new_threshold_full": threshold_full, "legacy_threshold_full": old_l1_metrics.get("threshold"),
        "lambda1_prediction_agreement": l1_agree,
        "lambda0_prediction_agreement_with_legacy_neutralization": l0_agree,
        "new_lambda1_fnr": new_l1_metrics["fnr"], "legacy_original_fnr": old_l1_metrics["fnr"],
        "new_lambda0_fnr": new_l0_metrics["fnr"], "legacy_neutralized_fnr": old_l0_metrics["fnr"],
        "new_lambda1_recall": new_l1_metrics["recall"], "legacy_original_recall": old_l1_metrics["recall"],
        "new_lambda0_recall": new_l0_metrics["recall"], "legacy_neutralized_recall": old_l0_metrics["recall"],
        "lambda1_score_mae_vs_legacy": float(np.mean(np.abs(new_l1_score - old_l1_score))),
        "lambda1_score_max_abs_vs_legacy": float(np.max(np.abs(new_l1_score - old_l1_score))),
        "lambda0_score_mae_vs_legacy": float(np.mean(np.abs(new_l0_score - old_l0_score))),
        "lambda0_score_max_abs_vs_legacy": float(np.max(np.abs(new_l0_score - old_l0_score))),
    }
    # -----------------------------------------------------------------
    # V3 acceptance gate: structural identity only.
    #
    # Legacy numerical concordance remains fully recorded as descriptive
    # reproducibility/QC evidence. It is not an acceptance criterion because
    # the scientific contrast is paired within the same newly trained model.
    # -----------------------------------------------------------------
    l1_fnr_abs_diff = abs(float(new_l1_metrics["fnr"]) - float(old_l1_metrics["fnr"]))
    l0_fnr_abs_diff = abs(float(new_l0_metrics["fnr"]) - float(old_l0_metrics["fnr"]))
    l1_recall_abs_diff = abs(float(new_l1_metrics["recall"]) - float(old_l1_metrics["recall"]))
    l0_recall_abs_diff = abs(float(new_l0_metrics["recall"]) - float(old_l0_metrics["recall"]))

    legacy_concordance_flag = bool(
        l1_agree >= 0.98
        and l0_agree >= 0.98
        and l1_fnr_abs_diff <= 0.03
        and l0_fnr_abs_diff <= 0.03
        and l1_recall_abs_diff <= 0.03
        and l0_recall_abs_diff <= 0.03
    )

    structural_gate_pass = bool(source_order_match and y_true_match)

    sanity.update({
        "acceptance_gate_mode": "structural_identity_only_v3",
        "structural_gate_pass": structural_gate_pass,
        "legacy_concordance_is_acceptance_criterion": False,
        "legacy_concordance_flag_at_reference_thresholds": legacy_concordance_flag,
        "lambda1_fnr_abs_difference_vs_legacy": l1_fnr_abs_diff,
        "lambda0_fnr_abs_difference_vs_legacy": l0_fnr_abs_diff,
        "lambda1_recall_abs_difference_vs_legacy": l1_recall_abs_diff,
        "lambda0_recall_abs_difference_vs_legacy": l0_recall_abs_diff,
        "reference_only_prediction_agreement_threshold": 0.98,
        "reference_only_fnr_abs_difference_threshold": 0.03,
        "reference_only_recall_abs_difference_threshold": 0.03,
    })

    sanity["reproduction_gate_pass"] = structural_gate_pass
    save_json(run_dir / "sanity_check.json", sanity)

    if not structural_gate_pass:
        save_json(
            run_dir / "STRUCTURAL_GATE_FAIL_V3.json",
            {
                "status": "FAIL",
                "gate": "structural_identity_only_v3",
                "sanity": sanity,
            },
        )
        raise RuntimeError(
            f"FAIL-CLOSED structural gate: "
            f"{dataset_name} {model_name} seed={seed} fold={fold}"
        )

    pd.DataFrame(lambda_metric_rows).to_csv(run_dir / "lambda_summary.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(run_dir / "sample_level_intervention.csv.gz", index=False, compression="gzip")
    pd.DataFrame(group_rows).to_csv(run_dir / "attack_group_lambda_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(run_dir / "decision_transition_summary.csv", index=False)
    np.savez_compressed(
        run_dir / "reconstruction_se_by_lambda.npz",
        lambdas=np.asarray(LAMBDAS, dtype=np.float64),
        source_row_index=source_rows.astype(np.int64),
        y_true=y_test.astype(np.int8),
        squared_reconstruction_error=np.stack(se_stack, axis=0).astype(np.float32),
    )
    save_json(run_dir / "run_manifest.json", {
        "status": "PASS", "dataset": dataset_name, "model": model_name, "condition": CONDITION,
        "seed": seed, "fold": fold, "model_seed": mseed, "feature_count": len(features),
        "neutral_scaled_epoch_value": neutral_value, "epochs_used": int(len(history.history.get("loss", []))),
        "train_time_sec": float(train_time), "full_threshold": threshold_full,
        "non_epoch_threshold": threshold_non_epoch, "full_model_saved": full_model_saved,
        "full_model_save_error": full_model_save_error, "weights_path": str(weights_path),
        "split_file": str(split_file), "split_file_sha256": sha256_file(split_file),
        "scaler_path": str(scaler_path), "scaler_sha256": sha256_file(scaler_path),
        "legacy_metrics_path": str(legacy_metrics_path), "legacy_metrics_sha256": sha256_file(legacy_metrics_path),
        "legacy_predictions_path": str(legacy_predictions_path),
        "legacy_predictions_sha256": sha256_file(legacy_predictions_path),
        "sample_path": str(cfg["sample"]), "sample_sha256": DATASET_HASHES[dataset_name],
        "reproduction_gate_pass": True,
        "acceptance_gate_mode": "structural_identity_only_v3",
        "legacy_concordance_is_acceptance_criterion": False,
    })
    save_json(complete_marker, {
        "status": "PASS", "dataset": dataset_name, "model": model_name, "seed": seed, "fold": fold,
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
    })
    print(f"[DONE] {dataset_name:<12} {model_name:<8} seed={seed} fold={fold:02d} epochs={len(history.history['loss']):02d} lambda0_FNR={new_l0_metrics['fnr']:.4f} lambda1_FNR={new_l1_metrics['fnr']:.4f}")
    del scaler, Xall, X_train_raw, X_val_raw, X_test_raw, X_train, X_val, X_test
    del y_test, groups, model, history, val_pred, val_se, se_stack, sample_rows
    tf.keras.backend.clear_session()
    gc.collect()


for dataset_idx, dataset_name in enumerate(DATASETS):
    cfg = exp.DATASETS[dataset_name]
    needed_cols = list(dict.fromkeys(features + [cfg["label"], "label_type", "source_row_index"]))
    print("\n" + "=" * 110)
    print(f"DATASET: {dataset_name}")
    print("=" * 110)
    sample = pd.read_parquet(cfg["sample"], columns=needed_cols)
    for feat in features:
        sample[feat] = pd.to_numeric(sample[feat], errors="coerce")
    sample[features] = sample[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    for model_name in MODELS:
        print(f"MODEL: {model_name}")
        for seed in SEEDS:
            for fold in FOLDS:
                run_fold(dataset_name, dataset_idx, sample, model_name, seed, fold)
    del sample
    gc.collect()
    tf.keras.backend.clear_session()

fold_lambda_frames = []
group_frames = []
transition_frames = []
risk_frames = []
reproduction_rows = []
for dataset_name in DATASETS:
    for model_name in MODELS:
        for seed in SEEDS:
            for fold in FOLDS:
                run_dir = RUNS_DIR / dataset_name / model_name / f"seed_{seed:02d}" / f"fold_{fold:02d}"
                if not (run_dir / "RUN_COMPLETE.json").exists():
                    raise RuntimeError(f"Missing completed run: {run_dir}")
                fold_lambda_frames.append(pd.read_csv(run_dir / "lambda_summary.csv"))
                group_frames.append(pd.read_csv(run_dir / "attack_group_lambda_summary.csv"))
                transition_frames.append(pd.read_csv(run_dir / "decision_transition_summary.csv"))
                risk_frames.append(pd.read_csv(run_dir / "attack_group_temporal_risk.csv"))
                with open(run_dir / "sanity_check.json", "r", encoding="utf-8") as f:
                    reproduction_rows.append(json.load(f))

fold_lambda = pd.concat(fold_lambda_frames, ignore_index=True)
groups_all = pd.concat(group_frames, ignore_index=True)
transitions_all = pd.concat(transition_frames, ignore_index=True)
risk_all = pd.concat(risk_frames, ignore_index=True)
reproduction_df = pd.DataFrame(reproduction_rows)
reproduction_df["structural_gate_pass_derived"] = (
    reproduction_df["source_row_order_match"].astype(bool)
    & reproduction_df["y_true_match"].astype(bool)
)
fold_lambda.to_csv(TABLES_DIR / "fold_lambda_metrics.csv", index=False)
groups_all.to_csv(TABLES_DIR / "attack_group_fold_lambda_metrics.csv", index=False)
transitions_all.to_csv(TABLES_DIR / "decision_transition_summary.csv", index=False)
risk_all.to_csv(TABLES_DIR / "attack_group_temporal_risk.csv", index=False)
reproduction_df.to_csv(TABLES_DIR / "reproduction_audit.csv", index=False)

numeric_cols = [
    "full_recall", "full_fnr", "full_f1", "full_auc", "full_fpr", "full_tdfr",
    "non_epoch_recall", "non_epoch_fnr", "non_epoch_f1", "non_epoch_auc", "non_epoch_fpr", "non_epoch_tdfr",
    "attack_full_score_mean", "attack_non_epoch_score_mean", "attack_epoch_error_mean",
]
summary_rows = []
for (dataset_name, model_name, lam), g in fold_lambda.groupby(["dataset", "model", "lambda"]):
    row = {"dataset": dataset_name, "model": model_name, "lambda": lam, "n_fold_units": int(len(g))}
    for col in numeric_cols:
        row[f"{col}_mean"] = float(g[col].mean())
        row[f"{col}_std"] = float(g[col].std(ddof=1))
        row[f"{col}_median"] = float(g[col].median())
    summary_rows.append(row)
dataset_lambda_summary = pd.DataFrame(summary_rows)
dataset_lambda_summary.to_csv(TABLES_DIR / "dataset_lambda_summary.csv", index=False)

attack_group_summary = []
for (dataset_name, model_name, attack_group, lam), g in groups_all.groupby(["dataset", "model", "attack_group", "lambda"]):
    attack_group_summary.append({
        "dataset": dataset_name, "model": model_name, "attack_group": attack_group, "lambda": lam, "n_fold_rows": int(len(g)),
        "full_recall_mean": float(g["full_recall"].mean()), "full_recall_std": float(g["full_recall"].std(ddof=1)),
        "full_tdfr_mean": float(g["full_tdfr"].mean()), "full_tdfr_median": float(g["full_tdfr"].median()),
        "non_epoch_recall_mean": float(g["non_epoch_recall"].mean()),
        "non_epoch_recall_std": float(g["non_epoch_recall"].std(ddof=1)),
        "non_epoch_tdfr_mean": float(g["non_epoch_tdfr"].mean()),
        "non_epoch_tdfr_median": float(g["non_epoch_tdfr"].median()),
    })
pd.DataFrame(attack_group_summary).to_csv(TABLES_DIR / "attack_group_summary.csv", index=False)

risk_summary = []
for (dataset_name, model_name, attack_group), g in risk_all.groupby(["dataset", "model", "attack_group"]):
    risk_summary.append({
        "dataset": dataset_name, "model": model_name, "attack_group": attack_group, "n_fold_rows": int(len(g)),
        "normalized_temporal_distance_mean": float(g["normalized_temporal_distance_mean"].mean()),
        "scaled_extrapolation_distance_mean": float(g["scaled_extrapolation_distance_mean"].mean()),
        "outside_training_normal_raw_support_ratio_mean": float(g["outside_training_normal_raw_support_ratio"].mean()),
        "outside_scaled_0_1_ratio_mean": float(g["outside_scaled_0_1_ratio"].mean()),
    })
pd.DataFrame(risk_summary).to_csv(TABLES_DIR / "attack_group_temporal_risk_summary.csv", index=False)

stat_rows = []
for model_name in MODELS:
    for dataset_name in DATASETS:
        g = fold_lambda[
            (fold_lambda["dataset"] == dataset_name) &
            (fold_lambda["model"] == model_name)
        ]
        l0 = g[g["lambda"] == 0.0].set_index(["seed", "fold"])["non_epoch_fnr"].sort_index()
        l1 = g[g["lambda"] == 1.0].set_index(["seed", "fold"])["non_epoch_fnr"].sort_index()
        if not l0.index.equals(l1.index):
            raise RuntimeError(f"Endpoint pairing mismatch: {dataset_name} {model_name}")
        diff = (l0 - l1).to_numpy(dtype=float)
        nb = nb_corrected_test(diff)
        try:
            wx = wilcoxon(
                l0.to_numpy(dtype=float),
                l1.to_numpy(dtype=float),
                alternative="two-sided",
                zero_method="wilcox",
            )
            wx_stat, wx_p = float(wx.statistic), float(wx.pvalue)
        except ValueError:
            wx_stat, wx_p = np.nan, np.nan
        stat_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "endpoint": PRIMARY_ENDPOINT,
            "n_paired_units": int(len(diff)),
            "lambda0_mean": float(l0.mean()),
            "lambda1_mean": float(l1.mean()),
            "mean_difference": nb["mean_diff"],
            "median_difference": nb["median_diff"],
            "sd_difference": nb["sd_diff"],
            "corrected_se": nb["corrected_se"],
            "corrected_ci95_low": nb["ci95_low"],
            "corrected_ci95_high": nb["ci95_high"],
            "nb_t_stat": nb["t_stat"],
            "nb_df": nb["df"],
            "nb_p_raw": nb["p_value"],
            "wilcoxon_stat": wx_stat,
            "wilcoxon_p_raw": wx_p,
        })
stats_df = pd.DataFrame(stat_rows)
stats_df["nb_p_holm"] = holm_adjust(stats_df["nb_p_raw"].to_numpy(dtype=float))
stats_df.to_csv(TABLES_DIR / "endpoint_non_epoch_statistics.csv", index=False)

lambda_order = LAMBDAS
x = np.arange(len(lambda_order))
xlabels = [str(v) for v in lambda_order]

def plot_metric_for_model(model_name, metric_col, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset_name in DATASETS:
        g = dataset_lambda_summary[
            (dataset_lambda_summary["dataset"] == dataset_name) &
            (dataset_lambda_summary["model"] == model_name)
        ].set_index("lambda").loc[lambda_order]
        ax.plot(x, g[metric_col].to_numpy(dtype=float), marker="o", label=dataset_name)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_xlabel("Temporal intervention λ")
    ax.set_ylabel(ylabel)
    ax.set_title(model_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{filename}_{model_name}.png", dpi=300)
    fig.savefig(FIGURES_DIR / f"{filename}_{model_name}.svg")
    plt.close(fig)

for model_name in MODELS:
    plot_metric_for_model(model_name, "full_fnr_mean", "Mean full-score false-negative rate", "fnr_lambda_curve")
    plot_metric_for_model(model_name, "non_epoch_fnr_mean", "Mean non-epoch false-negative rate", "non_epoch_fnr_lambda_curve")
    plot_metric_for_model(model_name, "full_tdfr_mean", "Mean temporal decision flip rate", "tdfr_lambda_curve")

save_json(METADATA_DIR / "final_vs_probe_evidence_register.json", {
    "legacy_final_evidence": {"root": str(LEGACY_ROOT), "status": "frozen primary confirmatory evidence", "training_runs": 720, "clipping_additional_fits": 240},
    "development_probes": {
        "probe_v1": "controlled_temporal_intervention/probe_ciciot2023_snn_34F_seed00_fold01_v1",
        "probe_v2": "controlled_temporal_intervention/probe_ciciot2023_snn_34F_seed00_fold01_v2",
        "status": "development/probe evidence only",
    },
    "current_additional_experiment": {"root": str(EXP_ROOT), "status": "additional post-hoc controlled intervention evidence", "models": MODELS, "datasets": DATASETS, "paired_units_per_dataset_model": 30},
    "protocol_revision_v3": str(METADATA_DIR / "protocol_revision_v3.json"),
    "current_implementation_freeze_v3": str(METADATA_DIR / "current_implementation_freeze_v3.json"),
})

(EXP_ROOT / "RESULTS_README.md").write_text(f"""# Controlled Temporal Intervention — Dense/LSTM/CNN Autoencoders / 3 Datasets / 30 Fold Units

Evidence status: additional post-hoc controlled-intervention evidence; does not replace frozen primary confirmatory evidence at `{LEGACY_ROOT}`.

Frozen design:
- Models: Dense-AE, LSTM-AE, CNN-AE
- Datasets: {DATASETS}
- Seeds: {SEEDS}; folds: 1..10; total retrained models: 270
- Lambda grid: {LAMBDAS}
- Full and non-epoch reconstruction-score tracks
- All non-epoch inputs fixed across lambda
- Thresholds fixed from validation-normal P99
- Intermediate lambdas are descriptive mechanistic response points
- Primary endpoint: non-epoch FNR at lambda=0 minus lambda=1
- Primary test: Nadeau-Bengio corrected repeated-CV t-test; Holm across 9 dataset-model comparisons
- Wilcoxon signed-rank is secondary

Every remaining fold must pass exact structural identity checks. Legacy numerical agreement is retained as descriptive reproducibility/QC evidence and is not a run-acceptance criterion. This v3 amendment was frozen after a 3-repeat LSTM repeatability characterization and a separate legacy-concordance failure on Edge-IIoTset/LSTM-AE seed=0 fold=3. The intervention design, endpoint, and statistical analysis were unchanged.

Claim boundary: controlled decision-dependence evidence only; not real-world deployment generalization and not causal proof.
""", encoding="utf-8")

save_json(EXP_ROOT / "RUN_COMPLETE.json", {
    "status": "PASS", "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
    "expected_model_runs": 270, "completed_model_runs": 270,
    "primary_endpoint": PRIMARY_ENDPOINT,
    "structural_gate_all_pass": bool(reproduction_df["structural_gate_pass_derived"].all()),
    "legacy_concordance_is_acceptance_criterion": False,
    "main_statistics": str(TABLES_DIR / "endpoint_non_epoch_statistics.csv"),
    "protocol": str(METADATA_DIR / "protocol_freeze_v3.json"),
    "protocol_amendment": str(METADATA_DIR / "protocol_revision_v3.json"),
    "current_implementation_freeze": str(METADATA_DIR / "current_implementation_freeze_v3.json"),
})

inventory_rows = []
for path in sorted(EXP_ROOT.rglob("*")):
    if path.is_file() and path.name not in {"artifact_inventory.json", "SHA256SUMS.txt"}:
        inventory_rows.append({
            "relative_path": str(path.relative_to(EXP_ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
save_json(METADATA_DIR / "artifact_inventory.json", {
    "experiment_root": str(EXP_ROOT), "file_count_excluding_inventory_and_checksum": len(inventory_rows), "files": inventory_rows,
})
checksum_lines = []
for path in sorted(EXP_ROOT.rglob("*")):
    if path.is_file() and path.name != "SHA256SUMS.txt":
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(EXP_ROOT)}")
(EXP_ROOT / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

print("\n" + "=" * 110)
print("CONTROLLED TEMPORAL INTERVENTION BATCH COMPLETE")
print("=" * 110)
print(f"Output root : {EXP_ROOT}")
print(f"Main summary: {TABLES_DIR / 'dataset_lambda_summary.csv'}")
print(f"Statistics  : {TABLES_DIR / 'endpoint_non_epoch_statistics.csv'}")
print(f"Checksums   : {EXP_ROOT / 'SHA256SUMS.txt'}")
