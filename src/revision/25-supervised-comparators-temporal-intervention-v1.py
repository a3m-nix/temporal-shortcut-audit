#!/usr/bin/env python3
"""
25-supervised-comparators-temporal-intervention-v1.py

Revision experiment (co-promotor comment 2, with support for comment 3).

PART A  Four supervised classifiers under the SAME frozen corpus and folds as Experiment I
        (0-temporal-feature-audit-v4-allminmax-3seed10fold: 3 seeds x 10 folds).
        Models (fixed, pre-declared hyperparameters; no tuning):
          XGB : gradient-boosted trees (axis-aligned splits, boosting)
          RF  : random forest (bagging + feature subsampling, max_features="sqrt")
          LR  : logistic regression (linear decision function)
          MLP : supervised 32-32 dense network (Adam 5e-4, batch 256, 30 epochs),
                matched to the Dense-AE optimisation settings of the paper
        Conditions:
          S34_with_epoch     : FEATURES_34 (33 packet features + frame.time_epoch)
          S33_without_epoch  : FEATURES_33 (retrained without frame.time_epoch)
          S1_epoch_only      : frame.time_epoch only (supervised time-only check)
        Perturbations on trained models (weights and threshold fixed):
          lambda          : t(lambda) = m + lambda*(t - m), m = median scaled epoch of train_normal
                            (identical rule to Experiment I, Section 3.5); S34 and S1
          epoch_permuted  : test-set permutation of the epoch column (model-agnostic importance); S34
        Roles (identical to Experiment I):
          train       = train_normal_idx + all trainval attacks (supervised labels)
          calibration = val_normal_idx (P99 threshold only; never used for fitting)
          test        = test_idx
        Scaler: MinMax fitted on train_normal only (parity with Experiment I).

PART B  Capture-time structure per dataset from the Experiment II split manifest
        (descriptive; no model fitting).

Evidence class: REVISION_SUPERVISED_COMPARATOR_POSTHOC. It does not replace the frozen
primary reconstruction evidence.

Usage (one dataset per process, then aggregate):
  export TEMPORAL_AUDIT_ROOT=$TEMPORAL_AUDIT_ROOT
  STAGE=train RUN_DATASETS=edge_iiotset VALIDATION_MODE=1 python 25-...py
  STAGE=train RUN_DATASETS=ciciot2023   VALIDATION_MODE=1 python 25-...py
  STAGE=train RUN_DATASETS=ton_iot      VALIDATION_MODE=1 python 25-...py
  STAGE=aggregate                       VALIDATION_MODE=1 python 25-...py
  (VALIDATION_MODE=1: 1 seed x 2 folds timing probe; VALIDATION_MODE=0: 3 seeds x 10 folds)
"""

# =============================================================================
# START: THREAD LIMITS (must be set before numpy/sklearn are imported)
# =============================================================================
import os

N_JOBS = int(os.environ.get("N_JOBS", "4"))   # threads per model; keep small to avoid contention
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_JOBS))
# =============================================================================
# END: THREAD LIMITS
# =============================================================================


# =============================================================================
# START: IMPORTS
# =============================================================================
import gc
import hashlib
import json
import math
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost is missing: pip install xgboost") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# =============================================================================
# END: IMPORTS
# =============================================================================


# =============================================================================
# START: CONFIGURATION (edit here if needed)
# =============================================================================
BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
SOURCE_V2 = BASE / "0-temporal-feature-audit-v2-3datasets"              # materialized samples (read-only)
FROZEN_V4 = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"    # Experiment I corpus + folds (read-only)
REPO = Path(os.environ.get("TEMPORAL_AUDIT_REPO", "<REPO_ROOT>")).resolve()
SPLIT_MANIFEST_ALL = REPO / "manifests" / "experiment2_audit_informed" / "split_manifest_all.csv.gz"

ROOT = BASE / "temporal_audit_revision_supervised_comparators_v1"         # isolated output folder

VALIDATION_MODE = os.environ.get("VALIDATION_MODE", "1") == "1"          # default = timing probe
STAGE = os.environ.get("STAGE", "train")                                  # "train" or "aggregate"
RUN_DATASETS = [d.strip() for d in os.environ.get(
    "RUN_DATASETS", "edge_iiotset,ciciot2023,ton_iot").split(",") if d.strip()]
RUN_MODELS = [m.strip() for m in os.environ.get("RUN_MODELS", "XGB,RF,LR,MLP").split(",") if m.strip()]

DATASETS = {
    "edge_iiotset": {"label": "label", "normal": "Normal"},
    "ciciot2023":   {"label": "label", "normal": "Benign"},
    "ton_iot":      {"label": "label", "normal": "Normal"},
}
DATASET_DISPLAY = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023", "ton_iot": "ToN-IoT"}

# Feature lists copied verbatim from 03-model-epoch-audit.py (FINAL_33 / FEATURES_34).
FEATURES_33 = [
    "frame.len", "ip.version", "ip.hdr_len", "ip.dsfield.dscp", "ip.len", "ip.id",
    "ip.frag_offset", "ip.flags.df", "ip.flags.mf", "ip.ttl", "ip.proto",
    "tcp.srcport", "tcp.dstport", "tcp.seq", "tcp.ack", "tcp.hdr_len", "tcp.len",
    "tcp.window_size", "tcp.flags.fin", "tcp.flags.syn", "tcp.flags.reset",
    "tcp.flags.push", "tcp.flags.ack", "tcp.flags.urg", "tcp.flags.ns",
    "tcp.flags.cwr", "udp.srcport", "udp.dstport", "udp.length", "icmp.type",
    "icmp.code", "data.len", "inter_arrival_time",
]
FEATURES_34 = ["frame.len", "frame.time_epoch", *FEATURES_33[1:]]
EPOCH = "frame.time_epoch"
assert len(FEATURES_33) == 33 and len(FEATURES_34) == 34 and EPOCH not in FEATURES_33

CONDITIONS = {
    "S34_with_epoch": FEATURES_34,
    "S33_without_epoch": FEATURES_33,
    "S1_epoch_only": [EPOCH],
}
LAMBDA_CONDITIONS = {"S34_with_epoch", "S1_epoch_only"}
PERMUTATION_CONDITIONS = {"S34_with_epoch"}

LAMBDAS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0]  # Section 3.5 grid

SEEDS = [0] if VALIDATION_MODE else [0, 1, 2]
FOLDS = [1, 2] if VALIDATION_MODE else list(range(1, 11))
N_FOLDS_DESIGN = 10
NB_TEST_TRAIN_RATIO = (1 / N_FOLDS_DESIGN) / ((N_FOLDS_DESIGN - 1) / N_FOLDS_DESIGN)  # 1/9, as script 04
ALPHA = 0.05
OPERATING_PERCENTILE = 99        # primary threshold: P99 of calibration-normal scores (paper rule)
FIXED_PROB_THRESHOLD = 0.5       # secondary, conventional supervised threshold

MODEL_SEED_OFFSET = {"XGB": 505, "RF": 606, "LR": 707, "MLP": 808}  # distinct from AE offsets 101-404
MODEL_DISPLAY = {"XGB": "XGBoost", "RF": "Random Forest", "LR": "Logistic Regression", "MLP": "MLP"}
MODEL_COLOR = {"XGB": "#1b9e77", "RF": "#d95f02", "LR": "#7570b3", "MLP": "#e7298a"}  # ColorBrewer Dark2


def build_model(name: str, seed: int):
    """Fixed, pre-declared hyperparameters (no tuning)."""
    if name == "XGB":
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=1.0,
            colsample_bytree=1.0, min_child_weight=1.0, reg_lambda=1.0,
            tree_method="hist", objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=N_JOBS)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_features="sqrt", max_depth=None, min_samples_leaf=1,
            bootstrap=True, random_state=seed, n_jobs=N_JOBS)
    if name == "LR":
        return LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=seed)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(32, 32), activation="relu", solver="adam",
            learning_rate_init=5e-4, batch_size=256, max_iter=30, alpha=1e-4,
            early_stopping=False, shuffle=True, random_state=seed)
    raise ValueError(name)


MODEL_PARAMS_DOC = {
    "XGB": "n_estimators=300, max_depth=6, learning_rate=0.1, tree_method=hist",
    "RF": "n_estimators=300, max_features=sqrt, max_depth=None, bootstrap=True",
    "LR": "lbfgs, C=1.0, max_iter=1000",
    "MLP": "hidden=(32,32), relu, adam lr=5e-4, batch=256, max_iter=30 epochs, alpha=1e-4, no early stopping",
}

DIRS = {k: ROOT / k for k in ["runs", "tables", "figures", "metadata"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
# =============================================================================
# END: CONFIGURATION
# =============================================================================


# =============================================================================
# START: GENERIC HELPERS
# =============================================================================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def json_safe(x):
    if x is None:
        return None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, np.integer):
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


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def model_seed(dataset_idx: int, seed: int, fold: int, model: str) -> int:
    return 100000 * (dataset_idx + 1) + 10000 * (seed + 1) + 100 * fold + MODEL_SEED_OFFSET[model]


def mem_report(tag: str):
    import resource
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        avail = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]
        avail_mb = int(avail) / 1024
    except Exception:
        avail_mb = float("nan")
    log(f"[MEM] {tag}: process peak={peak_mb:,.0f} MB | system available={avail_mb:,.0f} MB")


def metric_dict(y_true, scores, threshold) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    pred = (scores > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = np.nan
    return {"threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fnr": fn / (fn + tp) if (fn + tp) else np.nan,
            "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
            "recall": rec, "precision": prec, "f1": f1, "auc": float(auc), "pr_auc": float(pr_auc)}
# =============================================================================
# END: GENERIC HELPERS
# =============================================================================


# =============================================================================
# START: FROZEN INPUT LOADING (sample, corpus, folds, ToN strata)
# =============================================================================
def sample_path(ds: str) -> Path:
    return SOURCE_V2 / "materialized_samples" / f"{ds}_audit_sample.parquet"


def corpus_path(ds: str) -> Path:
    return FROZEN_V4 / "metadata" / "confirmatory_corpus" / f"{ds}_corpus_indices.npz"


def load_sample(ds: str) -> pd.DataFrame:
    p = sample_path(ds)
    if not p.exists():
        raise FileNotFoundError(p)
    cfg = DATASETS[ds]
    cols = list(dict.fromkeys(FEATURES_34 + [cfg["label"], "label_type", "source_row_index"]))
    s = pd.read_parquet(p, columns=cols)
    for c in FEATURES_34:                               # same sanitisation as script 04
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s[FEATURES_34] = s[FEATURES_34].replace([np.inf, -np.inf], np.nan).fillna(0)
    return s


def load_corpus(ds: str):
    p = corpus_path(ds)
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        return z["sample_local_indices"].astype(np.int64), z["y"].astype(np.int8)


def load_fold(ds: str, seed: int, fold: int):
    p = FROZEN_V4 / "splits" / ds / f"seed_{seed:02d}" / f"fold_{fold:02d}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        return z["train_normal_idx"], z["val_normal_idx"], z["test_idx"], p


def ton_stratum_mapping() -> dict[int, str]:
    df = pd.read_csv(SOURCE_V2 / "metadata" / "ton_iot_selected_attack_strata.csv")
    return {int(r.source_row_index): f"DDoS_stratum_{int(r.temporal_stratum):02d}"
            for r in df.itertuples(index=False)}


def attack_groups(ds, sample, idx, y, ton_map):
    if ds == "ton_iot":
        src = sample["source_row_index"].to_numpy()[idx]
        return np.asarray(["NORMAL" if yy == 0 else ton_map[int(s)] for s, yy in zip(src, y)], dtype=object)
    lt = sample["label_type"].astype(str).to_numpy()[idx]
    return np.asarray(["NORMAL" if yy == 0 else l for l, yy in zip(lt, y)], dtype=object)
# =============================================================================
# END: FROZEN INPUT LOADING
# =============================================================================


# =============================================================================
# START: PART A - ONE MODEL / CONDITION / SEED / FOLD RUN (checkpointed)
# =============================================================================
def run_one(ds, ds_idx, model_name, condition, features, seed, fold,
            X_train, y_train, X_val, X_test, y_test, train_normal_mask,
            groups_test, source_rows_test, fold_path):
    run_dir = DIRS["runs"] / ds / model_name / condition / f"seed_{seed:02d}" / f"fold_{fold:02d}"
    done = run_dir / "metrics.json"
    if done.exists():
        log(f"[RESUME] {ds} {model_name} {condition} seed={seed} fold={fold}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    mseed = model_seed(ds_idx, seed, fold, model_name)
    clf = build_model(model_name, mseed)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train, y_train)
    convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    train_time = time.time() - t0

    def score(X):
        return clf.predict_proba(X)[:, 1]

    val_score = score(X_val)                       # calibration normals only
    test_score = score(X_test)
    thr_p99 = float(np.percentile(val_score, OPERATING_PERCENTILE))

    rows = []

    def add(perturbation, lam, s):
        for rule, thr in (("P99_calibration_normal", thr_p99), ("fixed_0.5", FIXED_PROB_THRESHOLD)):
            rows.append({"dataset": ds, "model": model_name, "condition": condition, "seed": seed,
                         "fold": fold, "perturbation": perturbation, "lambda": lam,
                         "threshold_rule": rule, **metric_dict(y_test, s, thr)})

    preds = {"source_row_index": source_rows_test, "y_true": y_test,
             "attack_group": groups_test, "score_original": test_score}
    epoch_median_scaled = None

    if condition in LAMBDA_CONDITIONS:
        e = features.index(EPOCH)
        epoch_median_scaled = float(np.median(X_train[train_normal_mask, e]))
        for lam in LAMBDAS:
            if lam == 1.0:
                s = test_score
            else:
                Xi = X_test.copy()
                Xi[:, e] = epoch_median_scaled + lam * (X_test[:, e] - epoch_median_scaled)
                s = score(Xi)
                del Xi
            add("lambda", lam, s)
            if lam == 0.0:
                preds["score_lambda0"] = s
    else:
        add("lambda", 1.0, test_score)             # original input (no intervention)

    if condition in PERMUTATION_CONDITIONS:
        e = features.index(EPOCH)
        rng = np.random.RandomState(mseed)
        Xp = X_test.copy()
        Xp[:, e] = X_test[rng.permutation(len(X_test)), e]
        s = score(Xp)
        add("epoch_permuted", np.nan, s)
        preds["score_epoch_permuted"] = s
        del Xp

    pd.DataFrame(rows).to_csv(run_dir / "metrics_long.csv", index=False)

    grp = []
    for key in [k for k in ("score_original", "score_lambda0") if k in preds]:
        pb = (preds[key] > thr_p99).astype(np.int8)
        for g in sorted(np.unique(groups_test[y_test == 1])):
            mk = (y_test == 1) & (groups_test == g)
            grp.append({"dataset": ds, "model": model_name, "condition": condition, "seed": seed,
                        "fold": fold, "score": key, "attack_group": g, "n": int(mk.sum()),
                        "fnr": float(1.0 - pb[mk].mean())})
    pd.DataFrame(grp).to_csv(run_dir / "group_detection.csv", index=False)
    pd.DataFrame(preds).to_csv(run_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    save_json(done, {
        "dataset": ds, "model": model_name, "condition": condition, "seed": seed, "fold": fold,
        "model_seed": mseed, "model_params": MODEL_PARAMS_DOC[model_name], "features": features,
        "n_train": int(len(y_train)), "n_train_attack": int(y_train.sum()),
        "n_calibration_normal": int(len(X_val)), "n_test": int(len(y_test)),
        "threshold_p99_calibration_normal": thr_p99,
        "epoch_median_scaled_train_normal": epoch_median_scaled,
        "convergence_warning": convergence_warning,
        "train_time_sec": train_time, "fold_file": fold_path, "n_jobs": N_JOBS,
    })
    fnr0 = next(r["fnr"] for r in rows if r["threshold_rule"] == "P99_calibration_normal" and r["lambda"] == 1.0)
    log(f"[DONE] {ds:<12} {model_name:<3} {condition:<17} s={seed} f={fold:02d} "
        f"FNR(P99,orig)={fnr0:.4f} t={train_time:.1f}s{' [conv-warn]' if convergence_warning else ''}")
    del clf
    gc.collect()
# =============================================================================
# END: PART A - ONE RUN
# =============================================================================


# =============================================================================
# START: PART A - DRIVER (one dataset per process)
# =============================================================================
def run_part_a():
    for ds in RUN_DATASETS:
        if ds not in DATASETS:
            raise ValueError(f"Unknown dataset: {ds}")
    for m in RUN_MODELS:
        if m not in MODEL_SEED_OFFSET:
            raise ValueError(f"Unknown model: {m}")

    for ds in RUN_DATASETS:
        ds_idx = list(DATASETS).index(ds)          # fixed index -> identical seeds across processes
        log("=" * 90)
        log(f"DATASET {ds} | models={RUN_MODELS}")
        mem_report(f"{ds} before load")
        sample = load_sample(ds)
        cfg = DATASETS[ds]
        y_all = (sample[cfg["label"]].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()
        src_all = sample["source_row_index"].to_numpy(dtype=np.int64)
        corpus_idx, corpus_y = load_corpus(ds)
        if not np.array_equal(y_all[corpus_idx], corpus_y):
            raise RuntimeError(f"{ds}: corpus labels do not match sample labels")
        ton_map = ton_stratum_mapping() if ds == "ton_iot" else None
        Xfull = {c: sample[f].to_numpy(dtype=np.float32) for c, f in CONDITIONS.items()}
        save_json(DIRS["metadata"] / f"inputs_{ds}.json", {
            "dataset": ds, "sample_sha256": sha256_file(sample_path(ds)),
            "corpus_sha256": sha256_file(corpus_path(ds)), "n_jobs": N_JOBS,
            "validation_mode": VALIDATION_MODE, "models": RUN_MODELS})
        mem_report(f"{ds} after load")

        for seed in SEEDS:
            for fold in FOLDS:
                tr_n, val_n, test_idx, fpath = load_fold(ds, seed, fold)
                trainval_attack = np.setdiff1d(corpus_idx[corpus_y == 1], test_idx)
                train_idx = np.concatenate([tr_n, trainval_attack]).astype(np.int64)
                # Integrity checks: disjoint roles, full corpus coverage, pure-normal roles.
                if np.intersect1d(train_idx, np.concatenate([test_idx, val_n])).size \
                        or np.intersect1d(val_n, test_idx).size:
                    raise RuntimeError(f"{ds} s{seed} f{fold}: role overlap")
                if not np.array_equal(np.sort(np.concatenate([train_idx, val_n, test_idx])), np.sort(corpus_idx)):
                    raise RuntimeError(f"{ds} s{seed} f{fold}: roles do not cover the frozen corpus")
                if y_all[tr_n].any() or y_all[val_n].any():
                    raise RuntimeError(f"{ds} s{seed} f{fold}: attack inside a normal role")

                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
                train_normal_mask = (y_train == 0)
                groups_test = attack_groups(ds, sample, test_idx, y_test, ton_map)

                for cond, feats in CONDITIONS.items():
                    X = Xfull[cond]
                    scaler = MinMaxScaler().fit(X[tr_n])           # train_normal only
                    X_train = scaler.transform(X[train_idx]).astype(np.float32)
                    X_val = scaler.transform(X[val_n]).astype(np.float32)
                    X_test = scaler.transform(X[test_idx]).astype(np.float32)
                    for m in RUN_MODELS:
                        run_one(ds, ds_idx, m, cond, feats, seed, fold, X_train, y_train,
                                X_val, X_test, y_test, train_normal_mask, groups_test,
                                src_all[test_idx], fpath)
                    del scaler, X_train, X_val, X_test
                    gc.collect()
        del sample, y_all, src_all, corpus_idx, corpus_y, Xfull
        gc.collect()
        mem_report(f"{ds} finished")
# =============================================================================
# END: PART A - DRIVER
# =============================================================================


# =============================================================================
# START: PART A - AGGREGATION AND STATISTICS
# =============================================================================
def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def nadeau_bengio(diff):
    # Same formula as script 04 (lines 858-892).
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return dict(n=0, mean_delta=np.nan, sd_delta=np.nan, nb_t=np.nan, nb_p_raw=np.nan,
                    nb_ci95_low=np.nan, nb_ci95_high=np.nan)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = math.sqrt(((1.0 / n) + NB_TEST_TRAIN_RATIO) * sd ** 2) if sd > 0 else 0.0
    if se == 0:
        return dict(n=n, mean_delta=mean, sd_delta=sd, nb_t=np.nan,
                    nb_p_raw=1.0 if mean == 0 else 0.0, nb_ci95_low=mean, nb_ci95_high=mean)
    t = mean / se
    crit = float(stats.t.ppf(1 - ALPHA / 2, df=n - 1))
    return dict(n=n, mean_delta=mean, sd_delta=sd, nb_t=t,
                nb_p_raw=float(2.0 * stats.t.sf(abs(t), df=n - 1)),
                nb_ci95_low=mean - crit * se, nb_ci95_high=mean + crit * se)


def wilcoxon_p(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def collect(pattern: str) -> pd.DataFrame:
    files = sorted(DIRS["runs"].rglob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def aggregate_part_a():
    met = collect("metrics_long.csv")
    grp = collect("group_detection.csv")
    meta = pd.DataFrame([json.load(open(f)) for f in sorted(DIRS["runs"].rglob("metrics.json"))])
    met.to_csv(DIRS["tables"] / "sup_run_metrics_long.csv", index=False)
    grp.to_csv(DIRS["tables"] / "sup_group_detection_long.csv", index=False)
    meta.drop(columns=["features"], errors="ignore").to_csv(DIRS["tables"] / "sup_run_metadata.csv", index=False)

    keys = ["dataset", "model", "condition", "threshold_rule", "perturbation", "lambda"]
    summ = (met.groupby(keys, dropna=False)[["fnr", "fpr", "f1", "auc", "pr_auc"]]
               .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).rstrip("_") for c in summ.columns]
    summ.to_csv(DIRS["tables"] / "sup_summary_mean_sd.csv", index=False)

    idx = ["seed", "fold"]
    tests = []
    for rule in ("P99_calibration_normal", "fixed_0.5"):
        for m in met["model"].unique():
            for ds in DATASETS:
                d = met[(met.dataset == ds) & (met.model == m) & (met.threshold_rule == rule)]
                s34 = d[d.condition == "S34_with_epoch"]
                l1 = s34[(s34.perturbation == "lambda") & (s34["lambda"] == 1.0)].set_index(idx)
                l0 = s34[(s34.perturbation == "lambda") & (s34["lambda"] == 0.0)].set_index(idx)
                pm = s34[s34.perturbation == "epoch_permuted"].set_index(idx)
                s33 = d[d.condition == "S33_without_epoch"].set_index(idx)
                contrasts = {
                    "C1_lambda0_minus_lambda1_S34": (l0, l1),     # primary: fixed-weight intervention
                    "C2_S33_minus_S34": (s33, l1),                # retraining without epoch
                    "C3_permuted_minus_original_S34": (pm, l1),   # model-agnostic epoch importance
                }
                for cname, (a, b) in contrasts.items():
                    common = a.index.intersection(b.index)
                    if len(common) == 0:
                        continue
                    for metric in ("fnr", "fpr", "auc"):
                        diff = (a.loc[common, metric] - b.loc[common, metric]).to_numpy()
                        tests.append({"threshold_rule": rule, "model": m, "dataset": ds,
                                      "contrast": cname, "metric": metric,
                                      **nadeau_bengio(diff), "wilcoxon_p": wilcoxon_p(diff)})
    tests = pd.DataFrame(tests)
    # Holm across the 12 model x dataset comparisons within each (threshold, contrast, metric),
    # mirroring the 12-comparison family of the primary reconstruction analysis.
    tests["holm_p"] = np.nan
    for _, ix in tests.groupby(["threshold_rule", "contrast", "metric"]).groups.items():
        tests.loc[ix, "holm_p"] = holm_adjust(tests.loc[ix, "nb_p_raw"].to_numpy())
    tests.to_csv(DIRS["tables"] / "sup_paired_tests.csv", index=False)

    g = grp[grp.condition == "S34_with_epoch"]
    if not g.empty:
        (g.groupby(["dataset", "model", "attack_group", "score"])["fnr"].mean()
          .unstack("score").reset_index()
          .to_csv(DIRS["tables"] / "sup_group_fnr_S34_original_vs_lambda0.csv", index=False))

    conv = meta.groupby(["dataset", "model", "condition"])["convergence_warning"].mean().reset_index()
    conv.to_csv(DIRS["tables"] / "sup_convergence_warning_rate.csv", index=False)
    return met, tests
# =============================================================================
# END: PART A - AGGREGATION AND STATISTICS
# =============================================================================


# =============================================================================
# START: PART A - FIGURES (read from saved tables only)
# =============================================================================
def figures_part_a():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    summ = pd.read_csv(DIRS["tables"] / "sup_summary_mean_sd.csv")
    models = [m for m in MODEL_SEED_OFFSET if m in set(summ.model)]
    base = summ[(summ.threshold_rule == "P99_calibration_normal")]

    # Figure 1: lambda response of S34 models (FNR), one panel per dataset.
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in models:
            s = base[(base.dataset == ds) & (base.model == m) & (base.condition == "S34_with_epoch")
                     & (base.perturbation == "lambda")].sort_values("lambda")
            if s.empty:
                continue
            x = np.where(s["lambda"] == 0, 5e-4, s["lambda"])
            ax.plot(x, 100 * s["fnr_mean"], marker="o", ms=2.5, lw=1,
                    color=MODEL_COLOR[m], label=MODEL_DISPLAY[m])
        ax.set_xscale("log")
        ax.set_title(DATASET_DISPLAY[ds])
        ax.set_xlabel("λ (λ=0 at 5e-4)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Attack FNR (%)")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "sup_lambda_response_fnr.png", dpi=600)
    plt.close(fig)

    # Figure 2: FNR for original S34, S34 at lambda=0, and retrained S33.
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    w = 0.26
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(models))
        def val(m, cond, pert, lam):
            s = base[(base.dataset == ds) & (base.model == m) & (base.condition == cond)
                     & (base.perturbation == pert) & np.isclose(base["lambda"].fillna(-1), lam)]
            return 100 * float(s["fnr_mean"].iloc[0]) if len(s) else np.nan
        orig = [val(m, "S34_with_epoch", "lambda", 1.0) for m in models]
        neut = [val(m, "S34_with_epoch", "lambda", 0.0) for m in models]
        s33 = [val(m, "S33_without_epoch", "lambda", 1.0) for m in models]
        ax.bar(x - w, orig, w, color="#4393c3", label="S34 original")
        ax.bar(x, neut, w, color="#d6604d", label="S34 λ=0")
        ax.bar(x + w, s33, w, color="#878787", label="S33 retrained")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(DATASET_DISPLAY[ds])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Attack FNR (%)")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "sup_fnr_original_lambda0_s33.png", dpi=600)
    plt.close(fig)
# =============================================================================
# END: PART A - FIGURES
# =============================================================================


# =============================================================================
# START: PART B - CAPTURE-TIME STRUCTURE (descriptive, comment 3)
# =============================================================================
def run_part_b():
    if not SPLIT_MANIFEST_ALL.exists():
        log(f"[PART B SKIPPED] missing {SPLIT_MANIFEST_ALL}")
        return None
    cols = ["dataset", "y", "raw_capture_id", "raw_capture_block_certified", "frame.time_epoch"]
    df = pd.read_csv(SPLIT_MANIFEST_ALL, usecols=cols)
    df["cls"] = np.where(df["y"] == 1, "attack", "normal")

    def utc(v):
        return pd.to_datetime(v, unit="s", utc=True).strftime("%Y-%m-%d %H:%M")

    cap = (df.groupby(["dataset", "cls", "raw_capture_id"])
             .agg(n=("frame.time_epoch", "size"), epoch_min=("frame.time_epoch", "min"),
                  epoch_max=("frame.time_epoch", "max"),
                  certified_fraction=("raw_capture_block_certified", "mean")).reset_index())
    cap["duration_h"] = (cap.epoch_max - cap.epoch_min) / 3600
    cap["start_utc"] = cap.epoch_min.map(utc)
    cap["end_utc"] = cap.epoch_max.map(utc)
    cap.to_csv(DIRS["tables"] / "capture_time_structure.csv", index=False)

    rows = []
    for ds, g in df.groupby("dataset"):
        n = g[g.cls == "normal"]["frame.time_epoch"]
        a = g[g.cls == "attack"]["frame.time_epoch"]
        rows.append({
            "dataset": ds,
            "normal_start_utc": utc(n.min()), "normal_end_utc": utc(n.max()),
            "normal_span_h": (n.max() - n.min()) / 3600,
            "attack_start_utc": utc(a.min()), "attack_end_utc": utc(a.max()),
            "attack_span_h": (a.max() - a.min()) / 3600,
            "attack_inside_normal_calendar_range": float(((a >= n.min()) & (a <= n.max())).mean()),
            "normal_inside_attack_calendar_range": float(((n >= a.min()) & (n <= a.max())).mean()),
            "attack_before_all_normal": float((a < n.min()).mean()),
            "attack_after_all_normal": float((a > n.max()).mean()),
            "normal_captures": int(g[g.cls == "normal"].raw_capture_id.nunique()),
            "attack_captures": int(g[g.cls == "attack"].raw_capture_id.nunique()),
            "raw_capture_certified_fraction": float(g.raw_capture_block_certified.mean()),
        })
    win = pd.DataFrame(rows)
    win.to_csv(DIRS["tables"] / "class_time_windows.csv", index=False)
    return win
# =============================================================================
# END: PART B
# =============================================================================


# =============================================================================
# START: MAIN
# =============================================================================
def main():
    t0 = time.time()
    log(f"STAGE={STAGE} VALIDATION_MODE={VALIDATION_MODE} seeds={SEEDS} folds={FOLDS} "
        f"n_jobs={N_JOBS} models={RUN_MODELS}")
    log(f"Output: {ROOT}")

    if STAGE == "train":
        run_part_a()
        log(f"TRAIN STAGE FINISHED for {RUN_DATASETS} in {(time.time() - t0) / 60:.1f} min")
        return
    if STAGE != "aggregate":
        raise ValueError("STAGE must be 'train' or 'aggregate'")

    missing = [d for d in DATASETS if not (DIRS["metadata"] / f"inputs_{d}.json").exists()]
    if missing:
        raise RuntimeError(f"Train stage not finished for: {missing}")
    inputs = [json.load(open(p)) for p in sorted(DIRS["metadata"].glob("inputs_*.json"))]
    met, tests = aggregate_part_a()
    figures_part_a()
    win = run_part_b()

    save_json(DIRS["metadata"] / "run_manifest.json", {
        "script": Path(__file__).name, "script_sha256": sha256_file(Path(__file__)),
        "evidence_class": "REVISION_SUPERVISED_COMPARATOR_POSTHOC",
        "validation_mode": VALIDATION_MODE, "seeds": SEEDS, "folds": FOLDS,
        "lambdas": LAMBDAS, "conditions": CONDITIONS, "models": MODEL_PARAMS_DOC,
        "roles": {"train": "train_normal_idx + trainval attacks (frozen v4 folds)",
                  "calibration": "val_normal_idx (P99 threshold only)",
                  "test": "test_idx (frozen v4 folds)",
                  "scaler": "MinMax fitted on train_normal only"},
        "intervention": "t(lambda) = m + lambda*(t - m); m = median scaled epoch of train_normal; weights and threshold fixed",
        "permutation": "test-set permutation of the epoch column, RandomState(model_seed); S34 only",
        "statistics": "Nadeau-Bengio corrected paired t (test/train=1/9); Holm across 12 model x dataset "
                      "comparisons per (threshold, contrast, metric); Wilcoxon secondary",
        "inputs": inputs, "split_manifest_part_b": str(SPLIT_MANIFEST_ALL),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgb.__version__},
        "runtime_sec_aggregate": time.time() - t0,
    })

    pd.set_option("display.width", 220)
    p = tests[(tests.threshold_rule == "P99_calibration_normal") & (tests.metric == "fnr")]
    print("\n=== PAIRED TESTS: FNR, P99 calibration-normal threshold (delta in proportion) ===")
    print(p[["model", "dataset", "contrast", "n", "mean_delta", "nb_ci95_low", "nb_ci95_high",
             "holm_p", "wilcoxon_p"]].sort_values(["contrast", "model", "dataset"])
          .round(4).to_string(index=False))
    s = met[(met.threshold_rule == "P99_calibration_normal") & (met.perturbation == "lambda")
            & (met["lambda"].isin([0.0, 1.0]))]
    print("\n=== MEAN FNR / FPR / AUC (P99) ===")
    print(s.groupby(["dataset", "model", "condition", "lambda"])[["fnr", "fpr", "auc"]]
           .mean().round(4).to_string())
    if win is not None:
        print("\n=== CLASS TIME WINDOWS (Part B) ===")
        print(win.round(3).to_string(index=False))
    log(f"AGGREGATE FINISHED in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
# =============================================================================
# END: MAIN
# =============================================================================
