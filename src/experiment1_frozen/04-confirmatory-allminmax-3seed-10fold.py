from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import rankdata
import sklearn
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
SOURCE_V2 = BASE / "0-temporal-feature-audit-v2-3datasets"
BASE_SCRIPT = BASE / "0-temporal-feature-audit" / "03-model-epoch-audit.py"
ROOT = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(BASE_SCRIPT)
if not SOURCE_V2.exists():
    raise FileNotFoundError(SOURCE_V2)

DIRS = {
    "runs": ROOT / "runs",
    "splits": ROOT / "splits",
    "scalers": ROOT / "scalers",
    "tables": ROOT / "tables",
    "figures": ROOT / "figures",
    "metadata": ROOT / "metadata",
    "evidence": ROOT / "dataset_audit_evidence",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# IMPORT VALIDATED MODEL/AUDIT IMPLEMENTATION
# =============================================================================

spec = importlib.util.spec_from_file_location("audit_base", BASE_SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

# =============================================================================
# CONFIRMATORY DESIGN
# =============================================================================

DATASETS = {
    "edge_iiotset": {
        "sample": SOURCE_V2 / "materialized_samples" / "edge_iiotset_audit_sample.parquet",
        "label": "label",
        "normal": "Normal",
        "scaler": "minmax",
        "snn": {
            "threshold": 0.5480,
            "decay": 0.7011,
            "surrogate_scale": 13.4029,
        },
    },
    "ciciot2023": {
        "sample": SOURCE_V2 / "materialized_samples" / "ciciot2023_audit_sample.parquet",
        "label": "label",
        "normal": "Benign",
        "scaler": "minmax",
        "snn": {
            "threshold": 0.3584,
            "decay": 0.8315,
            "surrogate_scale": 6.9200,
        },
    },
    "ton_iot": {
        "sample": SOURCE_V2 / "materialized_samples" / "ton_iot_audit_sample.parquet",
        "label": "label",
        "normal": "Normal",
        "scaler": "minmax",
        # Fixed canonical LIF configuration; identical for 34F and 33F.
        "snn": {
            "threshold": 0.5,
            "decay": 0.7,
            "surrogate_scale": 10.0,
        },
    },
}

FEATURES_33 = list(audit.FINAL_33)
FEATURES_34 = list(audit.FEATURES_34)
CONDITIONS = {
    "with_epoch_34F": FEATURES_34,
    "without_epoch_33F": FEATURES_33,
}
MODELS = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
SEEDS = [0, 1, 2]
N_FOLDS = 10
VAL_RATIO_WITHIN_TRAIN_NORMAL = 0.20
CORPUS_SEED = 4242

# Same controlled optimization protocol for paired feature audit.
LEARNING_RATE = 5e-4
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
LATENT_DIM = 32
TIME_STEPS = 3
OPERATING_PERCENTILE = 99
SPIKE_SAMPLE_MAX = 10000

# Saving 720 neural-network files is unnecessary for inference/statistics and can
# consume large storage. All folds save predictions, histories, scalers, metrics,
# thresholds and diagnostics, which are sufficient to reproduce reported results.
SAVE_MODELS = False

# Statistical analysis.
INFERENTIAL_METRICS = ["fnr", "f1", "auc", "fpr"]
PRIMARY_METRIC = "fnr"
BOOTSTRAP_ITERATIONS = 20000
ALPHA = 0.05
NB_TEST_TRAIN_RATIO = (1 / N_FOLDS) / ((N_FOLDS - 1) / N_FOLDS)  # 1/9

# Tell imported helpers which configuration to use.
audit.DATASETS = {
    name: {
        "label": cfg["label"],
        "normal": cfg["normal"],
        "scaler": cfg["scaler"],
        "snn": cfg["snn"],
    }
    for name, cfg in DATASETS.items()
}
audit.LEARNING_RATE = LEARNING_RATE
audit.EPOCHS = EPOCHS
audit.PATIENCE = PATIENCE
audit.BATCH_SIZE = BATCH_SIZE
audit.LATENT_DIM = LATENT_DIM
audit.TIME_STEPS = TIME_STEPS
audit.OPERATING_PERCENTILE = OPERATING_PERCENTILE
audit.SPIKE_SAMPLE_MAX = SPIKE_SAMPLE_MAX

# =============================================================================
# GENERIC HELPERS
# =============================================================================

def json_safe(x):
    if x is None:
        return None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def model_seed(dataset_idx: int, seed: int, fold: int, model_name: str) -> int:
    model_offset = {
        "SNN_AE": 101,
        "Dense_AE": 202,
        "LSTM_AE": 303,
        "CNN_AE": 404,
    }[model_name]
    return 100000 * (dataset_idx + 1) + 10000 * (seed + 1) + 100 * fold + model_offset


def safe_auc(y, score):
    try:
        return float(sklearn.metrics.roc_auc_score(y, score))
    except Exception:
        return float("nan")

# =============================================================================
# CONFIRMATORY CORPUS
# =============================================================================

def ton_stratum_mapping() -> dict[int, str]:
    path = SOURCE_V2 / "metadata" / "ton_iot_selected_attack_strata.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return {
        int(r.source_row_index): f"DDoS_stratum_{int(r.temporal_stratum):02d}"
        for r in df.itertuples(index=False)
    }


def build_balanced_confirmatory_corpus(dataset_name: str, sample: pd.DataFrame):
    cfg = DATASETS[dataset_name]
    y = (sample[cfg["label"]].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()

    attack_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    if len(normal_idx) < len(attack_idx):
        raise RuntimeError(
            f"{dataset_name}: need at least {len(attack_idx):,} normal rows, "
            f"available {len(normal_idx):,}."
        )

    rng = np.random.RandomState(CORPUS_SEED)
    selected_normal = rng.choice(normal_idx, size=len(attack_idx), replace=False)
    corpus_idx = np.concatenate([selected_normal, attack_idx]).astype(np.int64)
    rng.shuffle(corpus_idx)

    corpus_y = y[corpus_idx]
    if dataset_name == "ton_iot":
        mapping = ton_stratum_mapping()
        strata = []
        for idx in corpus_idx:
            if y[idx] == 0:
                strata.append("NORMAL")
            else:
                src = int(sample.iloc[idx]["source_row_index"])
                if src not in mapping:
                    raise RuntimeError(f"Missing ToN temporal stratum for source row {src}")
                strata.append(mapping[src])
        strata = np.asarray(strata, dtype=object)
    else:
        label_type = sample["label_type"].astype(str).to_numpy()
        strata = np.asarray([
            "NORMAL" if y[idx] == 0 else label_type[idx]
            for idx in corpus_idx
        ], dtype=object)

    counts = pd.Series(strata).value_counts().sort_index()
    if int(counts.min()) < N_FOLDS:
        raise RuntimeError(
            f"{dataset_name}: smallest stratum has {counts.min()} rows, "
            f"which is less than {N_FOLDS} folds."
        )

    corpus_dir = DIRS["metadata"] / "confirmatory_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        corpus_dir / f"{dataset_name}_corpus_indices.npz",
        sample_local_indices=corpus_idx,
        y=corpus_y,
        strata=strata.astype(str),
    )

    corpus_manifest = pd.DataFrame({
        "sample_local_index": corpus_idx,
        "source_row_index": sample.iloc[corpus_idx]["source_row_index"].to_numpy(dtype=np.int64),
        "y": corpus_y,
        "stratum": strata,
    })
    corpus_manifest.to_csv(corpus_dir / f"{dataset_name}_corpus_manifest.csv.gz", index=False, compression="gzip")

    counts.rename_axis("stratum").reset_index(name="count").to_csv(
        corpus_dir / f"{dataset_name}_stratum_counts.csv", index=False
    )

    save_json(
        corpus_dir / f"{dataset_name}_summary.json",
        {
            "dataset": dataset_name,
            "sample_file": cfg["sample"],
            "sample_file_sha256": sha256_file(cfg["sample"]),
            "corpus_seed": CORPUS_SEED,
            "attack_rows": int(np.sum(corpus_y == 1)),
            "normal_rows": int(np.sum(corpus_y == 0)),
            "total_rows": int(len(corpus_idx)),
            "balanced": bool(np.sum(corpus_y == 1) == np.sum(corpus_y == 0)),
            "stratum_counts": counts.to_dict(),
        },
    )

    return corpus_idx, corpus_y, strata

# =============================================================================
# FOLD GENERATION
# =============================================================================

def make_and_save_folds(dataset_name: str, sample: pd.DataFrame, corpus_idx, corpus_y, strata):
    all_folds = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        dummy_x = np.zeros((len(corpus_idx), 1), dtype=np.int8)
        for fold, (trainval_pos, test_pos) in enumerate(skf.split(dummy_x, strata), start=1):
            trainval_idx = corpus_idx[trainval_pos]
            test_idx = corpus_idx[test_pos]

            y_trainval = (sample.iloc[trainval_idx][DATASETS[dataset_name]["label"]].astype(str)
                          != DATASETS[dataset_name]["normal"]).astype(np.int8).to_numpy()

            normal_pool = trainval_idx[y_trainval == 0].copy()
            rng = np.random.RandomState(900000 + seed * 1000 + fold)
            rng.shuffle(normal_pool)

            split = int(round((1.0 - VAL_RATIO_WITHIN_TRAIN_NORMAL) * len(normal_pool)))
            train_normal_idx = normal_pool[:split]
            val_normal_idx = normal_pool[split:]

            if len(train_normal_idx) == 0 or len(val_normal_idx) == 0:
                raise RuntimeError(f"{dataset_name} seed={seed} fold={fold}: empty normal train/val split")

            y_test = (sample.iloc[test_idx][DATASETS[dataset_name]["label"]].astype(str)
                      != DATASETS[dataset_name]["normal"]).astype(np.int8).to_numpy()

            fold_dir = DIRS["splits"] / dataset_name / f"seed_{seed:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fold_dir / f"fold_{fold:02d}.npz",
                train_normal_idx=train_normal_idx.astype(np.int64),
                val_normal_idx=val_normal_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
            )

            all_folds.append({
                "dataset": dataset_name,
                "seed": seed,
                "fold": fold,
                "train_normal_n": len(train_normal_idx),
                "val_normal_n": len(val_normal_idx),
                "test_n": len(test_idx),
                "test_normal_n": int(np.sum(y_test == 0)),
                "test_attack_n": int(np.sum(y_test == 1)),
            })

    return pd.DataFrame(all_folds)

# =============================================================================
# GROUP LABELS FOR SECURITY-SPECIFIC ANALYSIS
# =============================================================================

def attack_group_labels(dataset_name: str, sample: pd.DataFrame, test_idx: np.ndarray, y_test: np.ndarray):
    if dataset_name == "ton_iot":
        mapping = ton_stratum_mapping()
        groups = []
        for local_idx, yy in zip(test_idx, y_test):
            if yy == 0:
                groups.append("NORMAL")
            else:
                src = int(sample.iloc[local_idx]["source_row_index"])
                groups.append(mapping[src])
        return np.asarray(groups, dtype=object)

    labels = sample.iloc[test_idx]["label_type"].astype(str).to_numpy()
    return np.asarray(["NORMAL" if yy == 0 else label for yy, label in zip(y_test, labels)], dtype=object)

# =============================================================================
# ONE MODEL/FOLD RUN
# =============================================================================

def run_model_fold(
    dataset_name: str,
    dataset_idx: int,
    condition_name: str,
    feature_cols: list[str],
    model_name: str,
    seed: int,
    fold: int,
    sample: pd.DataFrame,
    train_normal_idx: np.ndarray,
    val_normal_idx: np.ndarray,
    test_idx: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_path: Path,
):
    run_dir = (
        DIRS["runs"] / dataset_name / condition_name / model_name /
        f"seed_{seed:02d}" / f"fold_{fold:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists():
        print(f"[RESUME] {dataset_name} {condition_name} {model_name} seed={seed} fold={fold}")
        return

    cfg = DATASETS[dataset_name]
    y_test = (sample.iloc[test_idx][cfg["label"]].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()
    groups = attack_group_labels(dataset_name, sample, test_idx, y_test)
    source_rows = sample.iloc[test_idx]["source_row_index"].to_numpy(dtype=np.int64)

    mseed = model_seed(dataset_idx, seed, fold, model_name)
    set_seed(mseed)
    tf.keras.backend.clear_session()

    model = audit.build_model(model_name, len(feature_cols), dataset_name)
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0 if model_name in {"SNN_AE", "Dense_AE"} else None,
    )
    model.compile(optimizer=optimizer, loss="mse")

    es = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    t0 = time.time()
    history = model.fit(
        audit.input_for_model(X_train, model_name),
        audit.target_for_model(X_train, model_name),
        validation_data=(
            audit.input_for_model(X_val, model_name),
            audit.target_for_model(X_val, model_name),
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )
    train_time = time.time() - t0

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(run_dir / "training_history.csv", index=False)

    val_pred = audit.predict_flat(model, X_val, model_name)
    test_pred = audit.predict_flat(model, X_test, model_name)

    val_se = np.square(X_val - val_pred)
    test_se = np.square(X_test - test_pred)
    val_score = np.mean(val_se, axis=1)
    test_score = np.mean(test_se, axis=1)
    threshold = float(np.percentile(val_score, OPERATING_PERCENTILE))

    metrics = audit.metric_dict(y_test, test_score, threshold)
    pred_binary = (test_score > threshold).astype(np.int8)

    # Per-feature reconstruction evidence.
    feat_rows = []
    for j, feat in enumerate(feature_cols):
        normal_err = test_se[y_test == 0, j]
        attack_err = test_se[y_test == 1, j]
        feat_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "feature": feat,
            "normal_mse_mean": float(np.mean(normal_err)),
            "normal_mse_median": float(np.median(normal_err)),
            "attack_mse_mean": float(np.mean(attack_err)),
            "attack_mse_median": float(np.median(attack_err)),
        })
    pd.DataFrame(feat_rows).to_csv(run_dir / "feature_reconstruction_summary.csv", index=False)

    # Per-attack subtype/temporal-stratum results.
    group_rows = []
    for group in sorted(np.unique(groups[y_test == 1])):
        mask = (y_test == 1) & (groups == group)
        recall = float(np.mean(pred_binary[mask]))
        group_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "attack_group": group,
            "n": int(np.sum(mask)),
            "detected": int(np.sum(pred_binary[mask] == 1)),
            "missed": int(np.sum(pred_binary[mask] == 0)),
            "recall": recall,
            "fnr": 1.0 - recall,
            "score_mean": float(np.mean(test_score[mask])),
            "score_median": float(np.median(test_score[mask])),
        })
    pd.DataFrame(group_rows).to_csv(run_dir / "group_detection.csv", index=False)

    prediction_df = pd.DataFrame({
        "source_row_index": source_rows,
        "y_true": y_test,
        "attack_group": groups,
        "score": test_score,
        "prediction": pred_binary,
    })

    diagnostics = {}

    if "frame.time_epoch" in feature_cols:
        epoch_idx = feature_cols.index("frame.time_epoch")
        keep = [i for i in range(len(feature_cols)) if i != epoch_idx]

        # Diagnostic A: remove only epoch error component from anomaly score.
        val_score_no_epoch = np.mean(val_se[:, keep], axis=1)
        test_score_no_epoch = np.mean(test_se[:, keep], axis=1)
        threshold_no_epoch = float(np.percentile(val_score_no_epoch, OPERATING_PERCENTILE))
        metrics_no_epoch = audit.metric_dict(y_test, test_score_no_epoch, threshold_no_epoch)

        # Diagnostic B: neutralize epoch input, no retraining; retain original 34F threshold.
        neutral_value = float(np.median(X_train[:, epoch_idx]))
        X_test_neutral = X_test.copy()
        X_test_neutral[:, epoch_idx] = neutral_value
        neutral_pred = audit.predict_flat(model, X_test_neutral, model_name)
        neutral_score = np.mean(np.square(X_test_neutral - neutral_pred), axis=1)
        metrics_neutral = audit.metric_dict(y_test, neutral_score, threshold)
        pred_neutral = (neutral_score > threshold).astype(np.int8)

        attack_se = test_se[y_test == 1]
        normal_se = test_se[y_test == 0]
        attack_fraction = float(np.sum(attack_se[:, epoch_idx]) / max(np.sum(attack_se), 1e-12))
        normal_fraction = float(np.sum(normal_se[:, epoch_idx]) / max(np.sum(normal_se), 1e-12))

        diagnostics.update({
            "epoch_scaled_train_normal_median": neutral_value,
            "score_excluding_epoch_component": metrics_no_epoch,
            "epoch_neutralized_input": metrics_neutral,
            "epoch_reconstruction_contribution_attack": attack_fraction,
            "epoch_reconstruction_contribution_normal": normal_fraction,
        })

        prediction_df["score_excluding_epoch_component"] = test_score_no_epoch
        prediction_df["score_epoch_neutralized"] = neutral_score
        prediction_df["prediction_epoch_neutralized"] = pred_neutral

        if model_name == "SNN_AE":
            diagnostics["snn_spike_original"] = audit.snn_spike_stats(model, X_test, y_test)
            diagnostics["snn_spike_epoch_neutralized"] = audit.snn_spike_stats(model, X_test_neutral, y_test)

        del X_test_neutral, neutral_pred

    elif model_name == "SNN_AE":
        diagnostics["snn_spike_original"] = audit.snn_spike_stats(model, X_test, y_test)

    prediction_df.to_csv(run_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    if SAVE_MODELS:
        model.save(run_dir / "model.keras")

    save_json(
        metrics_path,
        {
            "dataset": dataset_name,
            "condition": condition_name,
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "model_seed": mseed,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "params": int(model.count_params()),
            "scaler_path": scaler_path,
            "epochs_used": len(history.history.get("loss", [])),
            "train_time_sec": train_time,
            "operating_percentile": OPERATING_PERCENTILE,
            "main_metrics": metrics,
            "diagnostics": diagnostics,
        },
    )

    print(
        f"[DONE] {dataset_name:<12} {condition_name:<17} {model_name:<8} "
        f"seed={seed} fold={fold:02d} F1={metrics['f1']:.4f} "
        f"R={metrics['recall']:.4f} FNR={metrics['fnr']:.4f} AUC={metrics['auc']:.4f}"
    )

    del model, history, val_pred, test_pred, val_se, test_se, prediction_df
    tf.keras.backend.clear_session()
    gc.collect()

# =============================================================================
# RUN ALL 720 TRAININGS
# =============================================================================

def run_confirmatory_experiment():
    split_summaries = []

    for dataset_idx, dataset_name in enumerate(DATASETS):
        cfg = DATASETS[dataset_name]
        sample_path = cfg["sample"]
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)

        print("\n" + "=" * 110)
        print(f"DATASET: {dataset_name}")
        print("=" * 110)

        needed_cols = list(dict.fromkeys(FEATURES_34 + [cfg["label"], "label_type", "source_row_index"]))
        sample = pd.read_parquet(sample_path, columns=needed_cols)
        for feat in FEATURES_34:
            sample[feat] = pd.to_numeric(sample[feat], errors="coerce")
        sample[FEATURES_34] = sample[FEATURES_34].replace([np.inf, -np.inf], np.nan).fillna(0)

        corpus_idx, corpus_y, strata = build_balanced_confirmatory_corpus(dataset_name, sample)
        fold_df = make_and_save_folds(dataset_name, sample, corpus_idx, corpus_y, strata)
        split_summaries.append(fold_df)

        for seed in SEEDS:
            for fold in range(1, N_FOLDS + 1):
                split_file = DIRS["splits"] / dataset_name / f"seed_{seed:02d}" / f"fold_{fold:02d}.npz"
                with np.load(split_file, allow_pickle=False) as z:
                    train_idx = z["train_normal_idx"]
                    val_idx = z["val_normal_idx"]
                    test_idx = z["test_idx"]

                for condition_name, features in CONDITIONS.items():
                    scaler_dir = DIRS["scalers"] / dataset_name / condition_name / f"seed_{seed:02d}"
                    scaler_dir.mkdir(parents=True, exist_ok=True)
                    scaler_path = scaler_dir / f"fold_{fold:02d}.pkl"

                    Xall = sample[features].to_numpy(dtype=np.float32)
                    X_train_raw = Xall[train_idx]
                    X_val_raw = Xall[val_idx]
                    X_test_raw = Xall[test_idx]

                    scaler = audit.make_scaler(dataset_name, len(X_train_raw))
                    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
                    X_val = scaler.transform(X_val_raw).astype(np.float32)
                    X_test = scaler.transform(X_test_raw).astype(np.float32)

                    if not scaler_path.exists():
                        joblib.dump(scaler, scaler_path)

                    for model_name in MODELS:
                        run_model_fold(
                            dataset_name=dataset_name,
                            dataset_idx=dataset_idx,
                            condition_name=condition_name,
                            feature_cols=features,
                            model_name=model_name,
                            seed=seed,
                            fold=fold,
                            sample=sample,
                            train_normal_idx=train_idx,
                            val_normal_idx=val_idx,
                            test_idx=test_idx,
                            X_train=X_train,
                            X_val=X_val,
                            X_test=X_test,
                            scaler_path=scaler_path,
                        )

                    del scaler, Xall, X_train_raw, X_val_raw, X_test_raw, X_train, X_val, X_test
                    gc.collect()

        del sample, corpus_idx, corpus_y, strata
        tf.keras.backend.clear_session()
        gc.collect()

    pd.concat(split_summaries, ignore_index=True).to_csv(DIRS["tables"] / "split_summary.csv", index=False)

# =============================================================================
# AGGREGATION
# =============================================================================

def collect_results():
    metric_rows = []
    perturb_rows = []
    spike_rows = []
    epoch_contrib_rows = []
    group_frames = []
    feature_frames = []

    for dataset in DATASETS:
        for condition in CONDITIONS:
            for model in MODELS:
                for seed in SEEDS:
                    for fold in range(1, N_FOLDS + 1):
                        run_dir = (
                            DIRS["runs"] / dataset / condition / model /
                            f"seed_{seed:02d}" / f"fold_{fold:02d}"
                        )
                        path = run_dir / "metrics.json"
                        if not path.exists():
                            raise RuntimeError(f"Missing completed run: {path}")

                        with path.open("r", encoding="utf-8") as f:
                            p = json.load(f)
                        m = p["main_metrics"]
                        metric_rows.append({
                            "dataset": dataset,
                            "condition": condition,
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "feature_count": p["feature_count"],
                            "params": p["params"],
                            "epochs_used": p["epochs_used"],
                            "train_time_sec": p["train_time_sec"],
                            **m,
                        })

                        diag = p.get("diagnostics", {})
                        if "score_excluding_epoch_component" in diag:
                            perturb_rows.append({
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "fold": fold,
                                "diagnostic": "34F_score_excluding_epoch_component",
                                **diag["score_excluding_epoch_component"],
                            })
                        if "epoch_neutralized_input" in diag:
                            perturb_rows.append({
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "fold": fold,
                                "diagnostic": "34F_epoch_neutralized_input",
                                **diag["epoch_neutralized_input"],
                            })
                        if "epoch_reconstruction_contribution_attack" in diag:
                            epoch_contrib_rows.append({
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "fold": fold,
                                "attack_epoch_error_fraction": diag["epoch_reconstruction_contribution_attack"],
                                "normal_epoch_error_fraction": diag["epoch_reconstruction_contribution_normal"],
                            })
                        for key in ["snn_spike_original", "snn_spike_epoch_neutralized"]:
                            if key in diag:
                                spike_rows.append({
                                    "dataset": dataset,
                                    "condition": condition,
                                    "model": model,
                                    "seed": seed,
                                    "fold": fold,
                                    "spike_condition": key,
                                    **diag[key],
                                })

                        gp = run_dir / "group_detection.csv"
                        if gp.exists():
                            group_frames.append(pd.read_csv(gp))
                        fp = run_dir / "feature_reconstruction_summary.csv"
                        if fp.exists():
                            feature_frames.append(pd.read_csv(fp))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(DIRS["tables"] / "fold_metrics.csv", index=False)
    pd.DataFrame(perturb_rows).to_csv(DIRS["tables"] / "perturbation_fold_metrics.csv", index=False)
    pd.DataFrame(spike_rows).to_csv(DIRS["tables"] / "snn_spike_fold_metrics.csv", index=False)
    pd.DataFrame(epoch_contrib_rows).to_csv(DIRS["tables"] / "epoch_reconstruction_contribution_fold.csv", index=False)

    groups = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    groups.to_csv(DIRS["tables"] / "group_detection_fold.csv", index=False)
    features.to_csv(DIRS["tables"] / "feature_reconstruction_fold.csv", index=False)

    return metrics, pd.DataFrame(perturb_rows), pd.DataFrame(spike_rows), groups, features


def summarize_metrics(metrics: pd.DataFrame):
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr", "threshold"]
    rows = []
    for keys, g in metrics.groupby(["dataset", "condition", "model"]):
        row = dict(zip(["dataset", "condition", "model"], keys))
        row["n_evaluations"] = len(g)
        for col in metric_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_sd"] = g[col].std(ddof=1)
            row[f"{col}_median"] = g[col].median()
            row[f"{col}_min"] = g[col].min()
            row[f"{col}_max"] = g[col].max()
        row["epochs_used_mean"] = g["epochs_used"].mean()
        row["train_time_sec_mean"] = g["train_time_sec"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DIRS["tables"] / "summary_metrics.csv", index=False)
    return out

# =============================================================================
# STATISTICS
# =============================================================================

def holm_adjust(pvalues: np.ndarray):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_rank_biserial(diff: np.ndarray):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    denom = w_plus + w_minus
    return 0.0 if denom == 0 else (w_plus - w_minus) / denom


def paired_bootstrap_ci(diff: np.ndarray, iterations: int, seed: int):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    rng = np.random.RandomState(seed)
    n = len(diff)
    if n == 0:
        return np.nan, np.nan
    samples = rng.choice(diff, size=(iterations, n), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def nadeau_bengio_test(diff: np.ndarray):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    correction = (1.0 / n) + NB_TEST_TRAIN_RATIO
    se = math.sqrt(correction * (sd ** 2)) if sd > 0 else 0.0
    df = n - 1

    if se == 0:
        t_stat = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0 else 0.0
        ci_low = mean
        ci_high = mean
    else:
        t_stat = mean / se
        p = float(2.0 * stats.t.sf(abs(t_stat), df=df))
        crit = float(stats.t.ppf(1.0 - ALPHA / 2.0, df=df))
        ci_low = mean - crit * se
        ci_high = mean + crit * se

    return {
        "n": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "nb_correction_factor": correction,
        "nb_standard_error": se,
        "nb_t": t_stat,
        "nb_df": df,
        "nb_p_raw": p,
        "nb_ci95_low": ci_low,
        "nb_ci95_high": ci_high,
    }


def run_statistics(metrics: pd.DataFrame):
    # Paired rows are matched exactly by dataset/model/seed/fold.
    a = metrics[metrics["condition"] == "with_epoch_34F"].copy()
    b = metrics[metrics["condition"] == "without_epoch_33F"].copy()
    merged = a.merge(
        b,
        on=["dataset", "model", "seed", "fold"],
        suffixes=("_34F", "_33F"),
        validate="one_to_one",
    )

    delta_rows = []
    for _, r in merged.iterrows():
        base = {"dataset": r.dataset, "model": r.model, "seed": r.seed, "fold": r.fold}
        for metric in ["accuracy", "precision", "recall", "fnr", "f1", "auc", "fpr"]:
            v34 = float(r[f"{metric}_34F"])
            v33 = float(r[f"{metric}_33F"])
            delta_rows.append({**base, "metric": metric, "value_34F": v34, "value_33F": v33, "delta_33F_minus_34F": v33 - v34})
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(DIRS["tables"] / "paired_deltas.csv", index=False)

    test_rows = []
    for metric in INFERENTIAL_METRICS:
        dm = deltas[deltas["metric"] == metric]
        for (dataset, model), g in dm.groupby(["dataset", "model"]):
            diff = g["delta_33F_minus_34F"].to_numpy(dtype=float)
            nb = nadeau_bengio_test(diff)
            boot_low, boot_high = paired_bootstrap_ci(
                diff,
                iterations=BOOTSTRAP_ITERATIONS,
                seed=700000 + INFERENTIAL_METRICS.index(metric) * 1000 + list(DATASETS).index(dataset) * 100 + MODELS.index(model),
            )

            if np.allclose(diff, 0):
                wil_stat, wil_p = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", correction=False, method="auto")
                    wil_stat, wil_p = float(res.statistic), float(res.pvalue)
                except ValueError:
                    wil_stat, wil_p = 0.0, 1.0

            sd = float(np.std(diff, ddof=1))
            cohen_dz = 0.0 if sd == 0 else float(np.mean(diff) / sd)

            g34 = g["value_34F"].to_numpy(dtype=float)
            g33 = g["value_33F"].to_numpy(dtype=float)

            test_rows.append({
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "primary_metric": metric == PRIMARY_METRIC,
                "mean_34F": float(np.mean(g34)),
                "sd_34F": float(np.std(g34, ddof=1)),
                "mean_33F": float(np.mean(g33)),
                "sd_33F": float(np.std(g33, ddof=1)),
                **nb,
                "bootstrap_ci95_low_descriptive": float(boot_low),
                "bootstrap_ci95_high_descriptive": float(boot_high),
                "wilcoxon_W": wil_stat,
                "wilcoxon_p_raw": wil_p,
                "cohen_dz": cohen_dz,
                "rank_biserial": paired_rank_biserial(diff),
            })

    tests = pd.DataFrame(test_rows)

    # Holm adjustment within each metric family across the 12 dataset-model comparisons.
    adjusted_frames = []
    for metric, g in tests.groupby("metric", sort=False):
        g = g.copy()
        g["nb_p_holm"] = holm_adjust(g["nb_p_raw"].to_numpy())
        g["wilcoxon_p_holm"] = holm_adjust(g["wilcoxon_p_raw"].to_numpy())
        g["nb_significant_holm_0_05"] = g["nb_p_holm"] < 0.05
        g["wilcoxon_significant_holm_0_05"] = g["wilcoxon_p_holm"] < 0.05
        adjusted_frames.append(g)

    tests = pd.concat(adjusted_frames, ignore_index=True)
    tests.to_csv(DIRS["tables"] / "statistical_tests.csv", index=False)

    primary = tests[tests["metric"] == PRIMARY_METRIC].copy()
    primary.to_csv(DIRS["tables"] / "statistical_tests_primary_fnr.csv", index=False)

    return deltas, tests

# =============================================================================
# DIAGNOSTIC SUMMARIES
# =============================================================================

def summarize_diagnostics(perturb: pd.DataFrame, spikes: pd.DataFrame, groups: pd.DataFrame, features: pd.DataFrame):
    if not perturb.empty:
        rows = []
        for keys, g in perturb.groupby(["dataset", "model", "diagnostic"]):
            row = dict(zip(["dataset", "model", "diagnostic"], keys))
            row["n_evaluations"] = len(g)
            for col in ["recall", "f1", "auc", "fpr", "fnr"]:
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_sd"] = g[col].std(ddof=1)
            rows.append(row)
        pd.DataFrame(rows).to_csv(DIRS["tables"] / "perturbation_summary.csv", index=False)

    if not spikes.empty:
        rows = []
        for keys, g in spikes.groupby(["dataset", "condition", "spike_condition"]):
            row = dict(zip(["dataset", "condition", "spike_condition"], keys))
            row["n_evaluations"] = len(g)
            for col in ["spike_rate_normal", "spike_rate_attack", "spike_rate_delta_attack_minus_normal", "sparsity_normal", "sparsity_attack"]:
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_sd"] = g[col].std(ddof=1)
            rows.append(row)
        pd.DataFrame(rows).to_csv(DIRS["tables"] / "snn_spike_summary.csv", index=False)

    if not groups.empty:
        rows = []
        for keys, g in groups.groupby(["dataset", "condition", "model", "attack_group"]):
            row = dict(zip(["dataset", "condition", "model", "attack_group"], keys))
            row["n_fold_evaluations"] = len(g)
            row["recall_mean"] = g["recall"].mean()
            row["recall_sd"] = g["recall"].std(ddof=1)
            row["fnr_mean"] = g["fnr"].mean()
            row["fnr_sd"] = g["fnr"].std(ddof=1)
            row["score_mean"] = g["score_mean"].mean()
            rows.append(row)
        pd.DataFrame(rows).to_csv(DIRS["tables"] / "group_detection_summary.csv", index=False)

    if not features.empty:
        rows = []
        for keys, g in features.groupby(["dataset", "condition", "model", "feature"]):
            row = dict(zip(["dataset", "condition", "model", "feature"], keys))
            row["normal_mse_mean"] = g["normal_mse_mean"].mean()
            row["normal_mse_sd"] = g["normal_mse_mean"].std(ddof=1)
            row["attack_mse_mean"] = g["attack_mse_mean"].mean()
            row["attack_mse_sd"] = g["attack_mse_mean"].std(ddof=1)
            rows.append(row)
        pd.DataFrame(rows).to_csv(DIRS["tables"] / "feature_reconstruction_summary.csv", index=False)

# =============================================================================
# FIGURES
# =============================================================================

def plot_corrected_delta_ci(tests: pd.DataFrame, metric: str, filename: str, ylabel: str):
    src = tests[tests["metric"] == metric].copy()
    src["label"] = src["dataset"] + " | " + src["model"]
    src.to_csv(DIRS["figures"] / f"{filename}_source.csv", index=False)

    y = np.arange(len(src))
    x = src["mean_delta"].to_numpy()
    lower = x - src["nb_ci95_low"].to_numpy()
    upper = src["nb_ci95_high"].to_numpy() - x

    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(src))))
    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(src["label"])
    ax.set_xlabel(ylabel)
    ax.set_title(f"Paired 33F−34F effect with Nadeau–Bengio corrected 95% CI: {metric.upper()}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(DIRS["figures"] / f"{filename}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_perturbation(metrics: pd.DataFrame, perturb: pd.DataFrame):
    base34 = metrics[metrics["condition"] == "with_epoch_34F"][
        ["dataset", "model", "seed", "fold", "f1"]
    ].copy()
    base34["scenario"] = "34F original"

    base33 = metrics[metrics["condition"] == "without_epoch_33F"][
        ["dataset", "model", "seed", "fold", "f1"]
    ].copy()
    base33["scenario"] = "33F retrained"

    p = perturb[["dataset", "model", "seed", "fold", "diagnostic", "f1"]].copy()
    p["scenario"] = p["diagnostic"].map({
        "34F_score_excluding_epoch_component": "34F score excludes epoch",
        "34F_epoch_neutralized_input": "34F epoch neutralized",
    })
    combined = pd.concat([
        base34[["dataset", "model", "seed", "fold", "scenario", "f1"]],
        base33[["dataset", "model", "seed", "fold", "scenario", "f1"]],
        p[["dataset", "model", "seed", "fold", "scenario", "f1"]],
    ], ignore_index=True)

    summary = combined.groupby(["dataset", "model", "scenario"])["f1"].agg(mean="mean", std="std").reset_index()
    summary.to_csv(DIRS["figures"] / "perturbation_f1_source.csv", index=False)

    scenarios = ["34F original", "34F score excludes epoch", "34F epoch neutralized", "33F retrained"]
    labels = [f"{d}\n{m}" for d in DATASETS for m in MODELS]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, scenario in enumerate(scenarios):
        vals = []
        errs = []
        for d in DATASETS:
            for m in MODELS:
                row = summary[(summary["dataset"] == d) & (summary["model"] == m) & (summary["scenario"] == scenario)]
                vals.append(float(row["mean"].iloc[0]))
                errs.append(float(row["std"].iloc[0]))
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=2, label=scenario)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Inference perturbation and retrained 33F comparison (mean ± SD, 30 evaluations)")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "perturbation_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(DIRS["figures"] / "perturbation_f1.svg", bbox_inches="tight")
    plt.close(fig)


def plot_snn_spikes(spikes: pd.DataFrame):
    if spikes.empty:
        return
    summary = spikes.groupby(["dataset", "condition", "spike_condition"])[
        "spike_rate_delta_attack_minus_normal"
    ].agg(mean="mean", std="std").reset_index()
    summary.to_csv(DIRS["figures"] / "snn_spike_separation_source.csv", index=False)

    labels = []
    original = []
    neutral = []
    for d in DATASETS:
        labels.append(d)
        row_orig = summary[(summary.dataset == d) & (summary.condition == "with_epoch_34F") & (summary.spike_condition == "snn_spike_original")]
        row_neut = summary[(summary.dataset == d) & (summary.condition == "with_epoch_34F") & (summary.spike_condition == "snn_spike_epoch_neutralized")]
        original.append(float(row_orig["mean"].iloc[0]))
        neutral.append(float(row_neut["mean"].iloc[0]))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, original, width, label="34F original")
    ax.bar(x + width/2, neutral, width, label="34F epoch neutralized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spike-rate separation (attack − normal)")
    ax.set_title("SNN internal response to epoch neutralization")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "snn_spike_separation.png", dpi=300, bbox_inches="tight")
    fig.savefig(DIRS["figures"] / "snn_spike_separation.svg", bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# PROVENANCE + REPORT
# =============================================================================

def copy_dataset_evidence():
    candidates = [
        SOURCE_V2 / "dataset_audit",
        SOURCE_V2 / "metadata" / "ton_iot_temporal_sampling_equal_count.csv",
        SOURCE_V2 / "metadata" / "ton_iot_selected_attack_strata.csv",
    ]
    for src in candidates:
        if not src.exists():
            continue
        if src.is_dir():
            dst = DIRS["evidence"] / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, DIRS["evidence"] / src.name)


def write_manifest():
    save_json(DIRS["metadata"] / "CONFIRMATORY_EXPERIMENT_MANIFEST.json", {
        "experiment": "Temporal shortcut confirmatory study",
        "design": "3 seeds x 10 stratified folds",
        "datasets": list(DATASETS.keys()),
        "models": MODELS,
        "conditions": list(CONDITIONS.keys()),
        "expected_training_runs": len(DATASETS) * len(MODELS) * len(CONDITIONS) * len(SEEDS) * N_FOLDS,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "balanced_confirmatory_corpus": True,
        "corpus_seed": CORPUS_SEED,
        "normal_training_only": True,
        "validation_normal_ratio": VAL_RATIO_WITHIN_TRAIN_NORMAL,
        "threshold": f"P{OPERATING_PERCENTILE} validation-normal reconstruction error",
        "scalers": {d: DATASETS[d]["scaler"] for d in DATASETS},
        "feature_conditions": {
            "34F": FEATURES_34,
            "33F": FEATURES_33,
        },
        "optimization": {
            "learning_rate": LEARNING_RATE,
            "epochs_max": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "time_steps": TIME_STEPS,
        },
        "statistics": {
            "primary_metric": PRIMARY_METRIC,
            "secondary_metrics": [m for m in INFERENTIAL_METRICS if m != PRIMARY_METRIC],
            "paired_unit": "dataset-model-seed-fold",
            "primary_test": "Nadeau-Bengio corrected resampled t-test",
            "test_train_ratio": NB_TEST_TRAIN_RATIO,
            "robustness_test": "Wilcoxon signed-rank",
            "multiple_testing": "Holm correction separately within each metric family (12 dataset-model comparisons)",
            "effect_sizes": ["Cohen dz", "paired rank-biserial correlation"],
            "bootstrap": f"paired percentile bootstrap, {BOOTSTRAP_ITERATIONS} iterations; descriptive sensitivity CI",
            "corrected_ci": "Nadeau-Bengio corrected 95% CI used as primary inferential interval",
        },
        "inference_diagnostics_34F": [
            "reconstruction score excluding epoch component",
            "epoch-neutralized input without retraining",
        ],
        "snn_internal_diagnostics": ["spike rate normal", "spike rate attack", "spike separation", "sparsity"],
        "save_models": SAVE_MODELS,
        "source_v2": SOURCE_V2,
        "base_model_script": BASE_SCRIPT,
    })

    save_json(DIRS["metadata"] / "environment.json", {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    })


def write_readme(summary: pd.DataFrame, tests: pd.DataFrame):
    lines = [
        "# Confirmatory Temporal Shortcut Study",
        "",
        "Design: 3 seeds × 10 stratified folds; paired 34F vs 33F evaluation.",
        "",
        "Primary security endpoint: FNR. F1, AUC and FPR are secondary inferential outcomes.",
        "",
        "Nadeau–Bengio corrected resampled t-test is the primary inferential test; Wilcoxon signed-rank is a robustness check. Holm correction is applied within each metric family.",
        "",
        "## Primary FNR statistical results",
        "",
    ]
    primary = tests[tests.metric == PRIMARY_METRIC].copy()
    cols = [
        "dataset", "model", "mean_34F", "mean_33F", "mean_delta",
        "nb_ci95_low", "nb_ci95_high", "nb_p_holm", "wilcoxon_p_holm",
        "cohen_dz", "rank_biserial",
    ]
    lines.append("```text")
    lines.append(primary[cols].to_string(index=False))
    lines.append("```")
    lines.extend([
        "",
        "## Main files",
        "",
        "- tables/fold_metrics.csv",
        "- tables/summary_metrics.csv",
        "- tables/paired_deltas.csv",
        "- tables/statistical_tests.csv",
        "- tables/statistical_tests_primary_fnr.csv",
        "- tables/perturbation_fold_metrics.csv",
        "- tables/perturbation_summary.csv",
        "- tables/snn_spike_fold_metrics.csv",
        "- tables/snn_spike_summary.csv",
        "- tables/group_detection_fold.csv",
        "- tables/group_detection_summary.csv",
        "- figures/delta_fnr_corrected_ci.png",
        "- figures/delta_f1_corrected_ci.png",
        "- figures/perturbation_f1.png",
        "- figures/snn_spike_separation.png",
        "",
        "All per-run histories, predictions, thresholds, reconstruction summaries and diagnostics are retained under runs/.",
        "",
    ])
    (ROOT / "RESULTS_README.md").write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("TEMPORAL SHORTCUT CONFIRMATORY EXPERIMENT — 3 SEEDS × 10 FOLDS")
    print("=" * 110)
    print(f"Output root : {ROOT}")
    print(f"Runs        : {len(DATASETS)} datasets × {len(MODELS)} models × 2 conditions × 3 seeds × 10 folds = 720 trainings")
    print("Primary test: Nadeau–Bengio corrected resampled t-test")
    print("Robustness  : Wilcoxon signed-rank + Holm correction")
    print()

    copy_dataset_evidence()
    write_manifest()

    run_confirmatory_experiment()

    metrics, perturb, spikes, groups, features = collect_results()
    summary = summarize_metrics(metrics)
    deltas, tests = run_statistics(metrics)
    summarize_diagnostics(perturb, spikes, groups, features)

    plot_corrected_delta_ci(tests, "fnr", "delta_fnr_corrected_ci", "ΔFNR = 33F − 34F")
    plot_corrected_delta_ci(tests, "f1", "delta_f1_corrected_ci", "ΔF1 = 33F − 34F")
    plot_perturbation(metrics, perturb)
    plot_snn_spikes(spikes)

    write_readme(summary, tests)

    save_json(ROOT / "RUN_COMPLETE.json", {
        "status": "PASS",
        "completed_training_runs": int(len(metrics)),
        "expected_training_runs": 720,
        "paired_evaluations_per_dataset_model": len(SEEDS) * N_FOLDS,
        "primary_metric": PRIMARY_METRIC,
        "main_statistics": DIRS["tables"] / "statistical_tests.csv",
        "main_fold_metrics": DIRS["tables"] / "fold_metrics.csv",
    })

    print("\n" + "=" * 110)
    print("CONFIRMATORY EXPERIMENT COMPLETE")
    print("=" * 110)
    print("Main results:")
    print(DIRS["tables"] / "summary_metrics.csv")
    print(DIRS["tables"] / "statistical_tests.csv")
    print(DIRS["tables"] / "statistical_tests_primary_fnr.csv")
    print(ROOT / "RESULTS_README.md")


if __name__ == "__main__":
    main()
