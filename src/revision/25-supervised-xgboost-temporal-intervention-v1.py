#!/usr/bin/env python3
"""
25-supervised-xgboost-temporal-intervention-v1.py

Revision experiment (co-promotor comment 2 + comment 3 support).

PART A  Supervised XGBoost under the SAME frozen corpus and folds as Experiment I
        (0-temporal-feature-audit-v4-allminmax-3seed10fold: 3 seeds x 10 folds).
        Conditions:
          S34_with_epoch     : FEATURES_34 (33 packet features + frame.time_epoch)
          S33_without_epoch  : FEATURES_33 (retrained without frame.time_epoch)
          S1_epoch_only      : frame.time_epoch only (supervised time-only sanity check)
        Fixed-weight timestamp intervention on trained S34 / S1 models:
          t(lambda) = m + lambda * (t - m), m = median scaled epoch of train_normal
          (identical rule to Experiment I, Section 3.5).
        Roles (identical to Experiment I):
          train      = train_normal_idx  + all trainval attacks (supervised labels)
          calibration= val_normal_idx    (P99 threshold only, never used for fitting)
          test       = test_idx
        Scaler: MinMax fitted on train_normal only (parity with Experiment I).

PART B  Capture-time structure per dataset from the Experiment II split manifest
        (descriptive support for comment 3; no model fitting).

Evidence class: REVISION_SUPERVISED_COMPARATOR (post-hoc, labeled as such).
It does not replace the frozen primary reconstruction evidence.

Run (tmux, log tee'd):
  export TEMPORAL_AUDIT_ROOT=$TEMPORAL_AUDIT_ROOT
  VALIDATION_MODE=1 python 25-...py   # timing probe: 1 seed x 2 folds
  VALIDATION_MODE=0 python 25-...py   # full run: 3 seeds x 10 folds
"""

# =============================================================================
# START: IMPORTS
# =============================================================================
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
except ImportError as e:  # explicit message instead of silent failure
    raise SystemExit(
        "xgboost is not installed in this environment. Install with:\n"
        "  pip install xgboost"
    ) from e

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

# Frozen inputs (read-only).
SOURCE_V2 = BASE / "0-temporal-feature-audit-v2-3datasets"            # materialized samples
FROZEN_V4 = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"  # Experiment I corpus + folds

# Experiment II split manifest (Part B). Repository copy is used.
REPO = Path(os.environ.get(
    "TEMPORAL_AUDIT_REPO", "<REPO_ROOT>"
)).expanduser().resolve()
SPLIT_MANIFEST_ALL = REPO / "manifests" / "experiment2_audit_informed" / "split_manifest_all.csv.gz"

# Isolated output folder (never overwrites earlier experiments).
ROOT = BASE / "temporal_audit_revision_supervised_xgboost_v1"

VALIDATION_MODE = os.environ.get("VALIDATION_MODE", "1") == "1"   # default = timing probe
N_JOBS = int(os.environ.get("N_JOBS", "16"))                        # server has 30 cores

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
assert len(FEATURES_33) == 33 and len(FEATURES_34) == 34
assert EPOCH not in FEATURES_33

CONDITIONS = {
    "S34_with_epoch": FEATURES_34,
    "S33_without_epoch": FEATURES_33,
    "S1_epoch_only": [EPOCH],
}
INTERVENTION_CONDITIONS = {"S34_with_epoch", "S1_epoch_only"}

# Lambda grid identical to Section 3.5 of the manuscript.
LAMBDAS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0]

SEEDS = [0] if VALIDATION_MODE else [0, 1, 2]
FOLDS = [1, 2] if VALIDATION_MODE else list(range(1, 11))
N_FOLDS_DESIGN = 10                                      # used by Nadeau-Bengio correction
NB_TEST_TRAIN_RATIO = (1 / N_FOLDS_DESIGN) / ((N_FOLDS_DESIGN - 1) / N_FOLDS_DESIGN)  # 1/9, as script 04
ALPHA = 0.05

OPERATING_PERCENTILE = 99      # primary threshold rule (calibration-normal P99), as in the paper
FIXED_PROB_THRESHOLD = 0.5     # secondary, conventional supervised threshold

# Fixed XGBoost hyperparameters (no tuning, no early stopping -> no extra split roles).
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1.0,
    reg_lambda=1.0,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
)
XGB_SEED_OFFSET = 505  # distinct from AE offsets 101/202/303/404 in script 04

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


def model_seed(dataset_idx: int, seed: int, fold: int) -> int:
    return 100000 * (dataset_idx + 1) + 10000 * (seed + 1) + 100 * fold + XGB_SEED_OFFSET


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
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fnr": fn / (fn + tp) if (fn + tp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "recall": rec, "precision": prec, "f1": f1,
        "auc": float(auc), "pr_auc": float(pr_auc),
    }
# =============================================================================
# END: GENERIC HELPERS
# =============================================================================


# =============================================================================
# START: FROZEN INPUT LOADING (sample, corpus, folds, ToN strata)
# =============================================================================
def load_sample(dataset: str) -> pd.DataFrame:
    path = SOURCE_V2 / "materialized_samples" / f"{dataset}_audit_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    cfg = DATASETS[dataset]
    cols = list(dict.fromkeys(FEATURES_34 + [cfg["label"], "label_type", "source_row_index"]))
    s = pd.read_parquet(path, columns=cols)
    # Same numeric sanitisation as script 04 (lines 638-640).
    for c in FEATURES_34:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s[FEATURES_34] = s[FEATURES_34].replace([np.inf, -np.inf], np.nan).fillna(0)
    return s


def load_corpus(dataset: str):
    path = FROZEN_V4 / "metadata" / "confirmatory_corpus" / f"{dataset}_corpus_indices.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        return z["sample_local_indices"].astype(np.int64), z["y"].astype(np.int8)


def load_fold(dataset: str, seed: int, fold: int):
    path = FROZEN_V4 / "splits" / dataset / f"seed_{seed:02d}" / f"fold_{fold:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        return z["train_normal_idx"], z["val_normal_idx"], z["test_idx"], path


def ton_stratum_mapping() -> dict[int, str]:
    path = SOURCE_V2 / "metadata" / "ton_iot_selected_attack_strata.csv"
    df = pd.read_csv(path)
    return {int(r.source_row_index): f"DDoS_stratum_{int(r.temporal_stratum):02d}"
            for r in df.itertuples(index=False)}


def attack_groups(dataset: str, sample: pd.DataFrame, idx: np.ndarray, y: np.ndarray, ton_map):
    if dataset == "ton_iot":
        src = sample["source_row_index"].to_numpy()[idx]
        return np.asarray(["NORMAL" if yy == 0 else ton_map[int(s)] for s, yy in zip(src, y)], dtype=object)
    lt = sample["label_type"].astype(str).to_numpy()[idx]
    return np.asarray(["NORMAL" if yy == 0 else l for l, yy in zip(lt, y)], dtype=object)
# =============================================================================
# END: FROZEN INPUT LOADING
# =============================================================================


# =============================================================================
# START: PART A - ONE CONDITION / SEED / FOLD RUN (checkpointed)
# =============================================================================
def run_one(dataset, dataset_idx, condition, features, seed, fold,
            sample, y_all, train_idx, val_idx, test_idx, groups_test, fold_path):
    run_dir = DIRS["runs"] / dataset / condition / f"seed_{seed:02d}" / f"fold_{fold:02d}"
    done = run_dir / "metrics.json"
    if done.exists():
        log(f"[RESUME] {dataset} {condition} seed={seed} fold={fold}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    X = sample[features].to_numpy(dtype=np.float64)
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    train_normal_mask = (y_train == 0)

    # Scaler fitted on train_normal only (parity with Experiment I).
    scaler = MinMaxScaler()
    scaler.fit(X[train_idx][train_normal_mask])
    X_train = scaler.transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    mseed = model_seed(dataset_idx, seed, fold)
    clf = xgb.XGBClassifier(**XGB_PARAMS, random_state=mseed, n_jobs=N_JOBS)
    t0 = time.time()
    clf.fit(X_train, y_train, verbose=False)
    train_time = time.time() - t0

    val_score = clf.predict_proba(X_val)[:, 1]          # calibration normals only
    test_score = clf.predict_proba(X_test)[:, 1]
    thr_p99 = float(np.percentile(val_score, OPERATING_PERCENTILE))

    rows = []
    def add_rows(lam, scores):
        for rule, thr in (("P99_calibration_normal", thr_p99), ("fixed_0.5", FIXED_PROB_THRESHOLD)):
            m = metric_dict(y_test, scores, thr)
            rows.append({"dataset": dataset, "condition": condition, "seed": seed, "fold": fold,
                         "lambda": lam, "threshold_rule": rule, **m})

    preds = {"y_true": y_test, "attack_group": groups_test,
             "source_row_index": sample["source_row_index"].to_numpy()[test_idx],
             "score_original": test_score}

    epoch_median_scaled = None
    if condition in INTERVENTION_CONDITIONS:
        e = features.index(EPOCH)
        epoch_median_scaled = float(np.median(X_train[train_normal_mask, e]))
        for lam in LAMBDAS:
            Xi = X_test.copy()
            Xi[:, e] = epoch_median_scaled + lam * (X_test[:, e] - epoch_median_scaled)
            s_lam = test_score if lam == 1.0 else clf.predict_proba(Xi)[:, 1]
            add_rows(lam, s_lam)
            if lam == 0.0:
                preds["score_lambda0"] = s_lam
            del Xi
    else:
        add_rows(1.0, test_score)   # no intervention: report as lambda=1 (original input)

    pd.DataFrame(rows).to_csv(run_dir / "metrics_long.csv", index=False)

    # Per-attack-group detection at the primary threshold (original and lambda=0).
    grp_rows = []
    for key in [k for k in ("score_original", "score_lambda0") if k in preds]:
        pb = (preds[key] > thr_p99).astype(np.int8)
        for g in sorted(np.unique(groups_test[y_test == 1])):
            mk = (y_test == 1) & (groups_test == g)
            grp_rows.append({"dataset": dataset, "condition": condition, "seed": seed, "fold": fold,
                             "score": key, "attack_group": g, "n": int(mk.sum()),
                             "fnr": float(1.0 - pb[mk].mean())})
    pd.DataFrame(grp_rows).to_csv(run_dir / "group_detection.csv", index=False)
    pd.DataFrame(preds).to_csv(run_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    # Feature importance (total gain and split count) - share and rank of the epoch feature.
    booster = clf.get_booster()
    fmap = {f"f{i}": n for i, n in enumerate(features)}
    imp = []
    for itype in ("total_gain", "weight"):
        sc = {fmap[k]: v for k, v in booster.get_score(importance_type=itype).items()}
        total = sum(sc.values()) or 1.0
        ranked = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (feat, val) in enumerate(ranked, start=1):
            imp.append({"dataset": dataset, "condition": condition, "seed": seed, "fold": fold,
                        "importance_type": itype, "feature": feat, "value": val,
                        "share": val / total, "rank": rank})
    pd.DataFrame(imp).to_csv(run_dir / "feature_importance.csv", index=False)

    save_json(done, {
        "dataset": dataset, "condition": condition, "seed": seed, "fold": fold,
        "model_seed": mseed, "features": features, "xgb_params": XGB_PARAMS,
        "n_train": int(len(train_idx)), "n_train_attack": int(y_train.sum()),
        "n_calibration_normal": int(len(val_idx)), "n_test": int(len(test_idx)),
        "threshold_p99_calibration_normal": thr_p99,
        "epoch_median_scaled_train_normal": epoch_median_scaled,
        "train_time_sec": train_time, "fold_file": fold_path,
    })
    log(f"[DONE] {dataset:<12} {condition:<18} seed={seed} fold={fold:02d} "
        f"FNR(P99,orig)={rows[[r['lambda'] for r in rows].index(1.0)]['fnr']:.4f} t={train_time:.1f}s")
    del clf, booster, X, X_train, X_val, X_test
    gc.collect()
# =============================================================================
# END: PART A - ONE RUN
# =============================================================================


# =============================================================================
# START: PART A - DRIVER (one dataset in memory at a time)
# =============================================================================
def run_part_a():
    inputs = []
    for d_idx, ds in enumerate(DATASETS):
        log("=" * 90)
        log(f"DATASET {ds}")
        sample = load_sample(ds)
        cfg = DATASETS[ds]
        y_all = (sample[cfg["label"]].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()
        corpus_idx, corpus_y = load_corpus(ds)
        if not np.array_equal(y_all[corpus_idx], corpus_y):
            raise RuntimeError(f"{ds}: corpus labels do not match sample labels")
        ton_map = ton_stratum_mapping() if ds == "ton_iot" else None
        inputs.append({"dataset": ds,
                       "sample_sha256": sha256_file(SOURCE_V2 / "materialized_samples" / f"{ds}_audit_sample.parquet"),
                       "corpus_sha256": sha256_file(FROZEN_V4 / "metadata" / "confirmatory_corpus" / f"{ds}_corpus_indices.npz")})

        for seed in SEEDS:
            for fold in FOLDS:
                tr_n, val_n, test_idx, fpath = load_fold(ds, seed, fold)
                # Supervised training set = corpus minus test minus calibration normals.
                excl = np.concatenate([test_idx, val_n])
                trainval_attack = np.setdiff1d(corpus_idx[corpus_y == 1], test_idx, assume_unique=False)
                train_idx = np.concatenate([tr_n, trainval_attack]).astype(np.int64)
                # Integrity checks: role disjointness and full corpus coverage.
                if np.intersect1d(train_idx, excl).size or np.intersect1d(val_n, test_idx).size:
                    raise RuntimeError(f"{ds} s{seed} f{fold}: role overlap")
                if not np.array_equal(np.sort(np.concatenate([train_idx, val_n, test_idx])), np.sort(corpus_idx)):
                    raise RuntimeError(f"{ds} s{seed} f{fold}: roles do not cover the frozen corpus")
                if y_all[tr_n].any() or y_all[val_n].any():
                    raise RuntimeError(f"{ds} s{seed} f{fold}: attack inside normal role")
                groups_test = attack_groups(ds, sample, test_idx, y_all[test_idx], ton_map)
                for cond, feats in CONDITIONS.items():
                    run_one(ds, d_idx, cond, feats, seed, fold, sample, y_all,
                            train_idx, val_n, test_idx, groups_test, fpath)
        del sample, y_all, corpus_idx, corpus_y
        gc.collect()
    return inputs
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
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = math.sqrt(((1.0 / n) + NB_TEST_TRAIN_RATIO) * sd ** 2) if sd > 0 else 0.0
    if se == 0:
        return dict(n=n, mean_delta=mean, sd_delta=sd, nb_t=np.nan,
                    nb_p_raw=1.0 if mean == 0 else 0.0, nb_ci95_low=mean, nb_ci95_high=mean)
    t = mean / se
    p = float(2.0 * stats.t.sf(abs(t), df=n - 1))
    crit = float(stats.t.ppf(1 - ALPHA / 2, df=n - 1))
    return dict(n=n, mean_delta=mean, sd_delta=sd, nb_t=t, nb_p_raw=p,
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
    imp = collect("feature_importance.csv")
    grp = collect("group_detection.csv")
    met.to_csv(DIRS["tables"] / "xgb_run_metrics_long.csv", index=False)
    imp.to_csv(DIRS["tables"] / "xgb_feature_importance_long.csv", index=False)
    grp.to_csv(DIRS["tables"] / "xgb_group_detection_long.csv", index=False)

    # Descriptive summary.
    summ = (met.groupby(["dataset", "condition", "threshold_rule", "lambda"])
               [["fnr", "fpr", "f1", "auc", "pr_auc"]].agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).rstrip("_") for c in summ.columns]
    summ.to_csv(DIRS["tables"] / "xgb_summary_mean_sd.csv", index=False)

    # Paired contrasts over seed-fold units.
    key = ["seed", "fold"]
    tests = []
    for rule in ("P99_calibration_normal", "fixed_0.5"):
        for ds in DATASETS:
            m = met[(met.dataset == ds) & (met.threshold_rule == rule)]
            s34 = m[m.condition == "S34_with_epoch"]
            l1 = s34[s34["lambda"] == 1.0].set_index(key)
            l0 = s34[s34["lambda"] == 0.0].set_index(key)
            s33 = m[m.condition == "S33_without_epoch"].set_index(key)
            contrasts = {
                "C1_lambda0_minus_lambda1_S34": (l0, l1),        # primary: fixed-weight intervention
                "C2_S33_minus_S34_lambda1": (s33, l1),           # secondary: retraining without epoch
            }
            for cname, (a, b) in contrasts.items():
                common = a.index.intersection(b.index)
                for metric in ("fnr", "fpr", "auc"):
                    diff = (a.loc[common, metric] - b.loc[common, metric]).to_numpy()
                    nb = nadeau_bengio(diff)
                    tests.append({"threshold_rule": rule, "dataset": ds, "contrast": cname,
                                  "metric": metric, **nb, "wilcoxon_p": wilcoxon_p(diff)})
    tests = pd.DataFrame(tests)
    # Holm within each family: (threshold_rule, contrast, metric) across the 3 datasets.
    tests["holm_p"] = np.nan
    for _, idx in tests.groupby(["threshold_rule", "contrast", "metric"]).groups.items():
        tests.loc[idx, "holm_p"] = holm_adjust(tests.loc[idx, "nb_p_raw"].to_numpy())
    tests.to_csv(DIRS["tables"] / "xgb_paired_tests.csv", index=False)

    # Epoch importance summary (S34 only).
    ie = imp[(imp.condition == "S34_with_epoch") & (imp.feature == EPOCH)]
    ie_s = (ie.groupby(["dataset", "importance_type"])
              .agg(share_mean=("share", "mean"), share_sd=("share", "std"),
                   rank_median=("rank", "median"), rank_min=("rank", "min"),
                   rank_max=("rank", "max"), n_runs=("share", "size")).reset_index())
    ie_s.to_csv(DIRS["tables"] / "xgb_epoch_importance_summary.csv", index=False)

    # Group-level FNR (S34, P99 threshold), original vs lambda=0.
    g = grp[grp.condition == "S34_with_epoch"]
    if not g.empty:
        gs = g.groupby(["dataset", "attack_group", "score"])["fnr"].mean().unstack("score").reset_index()
        gs.to_csv(DIRS["tables"] / "xgb_group_fnr_S34_original_vs_lambda0.csv", index=False)
    return met, tests, ie_s
# =============================================================================
# END: PART A - AGGREGATION AND STATISTICS
# =============================================================================


# =============================================================================
# START: PART A - FIGURES (read from saved tables only)
# =============================================================================
def figures_part_a():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    summ = pd.read_csv(DIRS["tables"] / "xgb_summary_mean_sd.csv")

    # Figure 1: lambda response (S34, primary threshold).
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        s = summ[(summ.dataset == ds) & (summ.condition == "S34_with_epoch")
                 & (summ.threshold_rule == "P99_calibration_normal")].sort_values("lambda")
        x = s["lambda"].to_numpy()
        ax.errorbar(np.where(x == 0, 5e-4, x), 100 * s["fnr_mean"], yerr=100 * s["fnr_std"],
                    marker="o", ms=3, lw=1, capsize=2, color="#2166ac")
        s33 = summ[(summ.dataset == ds) & (summ.condition == "S33_without_epoch")
                   & (summ.threshold_rule == "P99_calibration_normal")]
        if len(s33):
            ax.axhline(100 * s33["fnr_mean"].iloc[0], ls="--", lw=0.8, color="#b2182b",
                       label="Retrained without epoch")
        ax.set_xscale("log")
        ax.set_title(DATASET_DISPLAY[ds])
        ax.set_xlabel("λ (λ=0 plotted at 5e-4)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Attack FNR (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "xgb_lambda_response.png", dpi=600)
    plt.close(fig)

    # Figure 2: epoch share of total gain (S34).
    ie = pd.read_csv(DIRS["tables"] / "xgb_epoch_importance_summary.csv")
    ie = ie[ie.importance_type == "total_gain"].set_index("dataset").reindex(list(DATASETS))
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.bar([DATASET_DISPLAY[d] for d in ie.index], 100 * ie["share_mean"],
           yerr=100 * ie["share_sd"], capsize=3, color="#4393c3")
    ax.set_ylabel("frame.time_epoch share of total gain (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "xgb_epoch_gain_share.png", dpi=600)
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

    cap = (df.groupby(["dataset", "cls", "raw_capture_id"])
             .agg(n=("frame.time_epoch", "size"), epoch_min=("frame.time_epoch", "min"),
                  epoch_max=("frame.time_epoch", "max"),
                  certified_fraction=("raw_capture_block_certified", "mean")).reset_index())
    cap["duration_h"] = (cap.epoch_max - cap.epoch_min) / 3600
    cap["start_utc"] = pd.to_datetime(cap.epoch_min, unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    cap["end_utc"] = pd.to_datetime(cap.epoch_max, unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    cap.to_csv(DIRS["tables"] / "capture_time_structure.csv", index=False)

    rows = []
    for ds, g in df.groupby("dataset"):
        n = g[g.cls == "normal"]["frame.time_epoch"]
        a = g[g.cls == "attack"]["frame.time_epoch"]
        rows.append({
            "dataset": ds,
            "normal_start_utc": pd.to_datetime(n.min(), unit="s", utc=True).strftime("%Y-%m-%d %H:%M"),
            "normal_end_utc": pd.to_datetime(n.max(), unit="s", utc=True).strftime("%Y-%m-%d %H:%M"),
            "normal_span_h": (n.max() - n.min()) / 3600,
            "attack_start_utc": pd.to_datetime(a.min(), unit="s", utc=True).strftime("%Y-%m-%d %H:%M"),
            "attack_end_utc": pd.to_datetime(a.max(), unit="s", utc=True).strftime("%Y-%m-%d %H:%M"),
            "attack_span_h": (a.max() - a.min()) / 3600,
            "attack_rows_inside_normal_calendar_range": float(((a >= n.min()) & (a <= n.max())).mean()),
            "normal_rows_inside_attack_calendar_range": float(((n >= a.min()) & (n <= a.max())).mean()),
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
    log(f"VALIDATION_MODE={VALIDATION_MODE} seeds={SEEDS} folds={FOLDS} n_jobs={N_JOBS}")
    log(f"Output: {ROOT}")
    inputs = run_part_a()
    met, tests, ie_s = aggregate_part_a()
    figures_part_a()
    win = run_part_b()

    save_json(DIRS["metadata"] / "run_manifest.json", {
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "evidence_class": "REVISION_SUPERVISED_COMPARATOR_POSTHOC",
        "validation_mode": VALIDATION_MODE,
        "seeds": SEEDS, "folds": FOLDS, "lambdas": LAMBDAS,
        "conditions": CONDITIONS, "xgb_params": XGB_PARAMS,
        "roles": {"train": "train_normal_idx + trainval attacks (frozen v4 folds)",
                  "calibration": "val_normal_idx (P99 threshold only)",
                  "test": "test_idx (frozen v4 folds)",
                  "scaler": "MinMax fitted on train_normal only"},
        "intervention": "t(lambda) = m + lambda*(t - m); m = median scaled epoch of train_normal; weights and threshold fixed",
        "statistics": "Nadeau-Bengio corrected paired t (test/train=1/9), Holm across 3 datasets per (threshold, contrast, metric); Wilcoxon secondary",
        "inputs": inputs,
        "split_manifest_part_b": str(SPLIT_MANIFEST_ALL),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgb.__version__},
        "runtime_sec": time.time() - t0,
    })

    # Compact console report (only what is needed).
    pd.set_option("display.width", 200)
    prim = tests[(tests.threshold_rule == "P99_calibration_normal") & (tests.metric == "fnr")]
    print("\n=== PAIRED TESTS (FNR, P99 calibration-normal threshold) ===")
    print(prim[["dataset", "contrast", "n", "mean_delta", "nb_ci95_low", "nb_ci95_high",
                "holm_p", "wilcoxon_p"]].round(4).to_string(index=False))
    s = met[(met.threshold_rule == "P99_calibration_normal") & (met["lambda"].isin([0.0, 1.0]))]
    print("\n=== MEAN FNR / FPR / AUC (P99) ===")
    print(s.groupby(["dataset", "condition", "lambda"])[["fnr", "fpr", "auc"]].mean().round(4).to_string())
    print("\n=== frame.time_epoch IMPORTANCE (S34) ===")
    print(ie_s.round(4).to_string(index=False))
    if win is not None:
        print("\n=== CLASS TIME WINDOWS (Part B) ===")
        print(win.round(3).to_string(index=False))
    log(f"FINISHED in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
# =============================================================================
# END: MAIN
# =============================================================================
