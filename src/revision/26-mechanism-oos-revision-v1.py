#!/usr/bin/env python3
"""
26 - Out-of-support (OOS) mechanism tests for the revision (R2 M1/M6, R7).

Inference only on the 360 saved weight-fixed lambda-intervention models
(ae3_3datasets_30fold_v1 + snn_3datasets_30fold_v1). Splits, scalers, weights and
the P99 validation-normal non-epoch threshold are reused unchanged; only the scaled
frame.time_epoch input is manipulated. Pre-specified predictions:
controlled_temporal_intervention/mechanism_oos_revision_v1/metadata/PREREGISTERED_PREDICTIONS.md

Usage: python 26-mechanism-oos-revision-v1.py --dataset ciciot2023 [--limit N]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
LEGACY_ROOT = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
FINAL_SCRIPT = LEGACY_ROOT / "04-confirmatory-allminmax-3seed-10fold.py"
CTI = LEGACY_ROOT / "controlled_temporal_intervention"
AE3_RUNS = CTI / "ae3_3datasets_30fold_v1" / "runs"
SNN_RUNS = CTI / "snn_3datasets_30fold_v1" / "runs"
OUT = CTI / "mechanism_oos_revision_v1"
PRED_FILE = OUT / "metadata" / "PREREGISTERED_PREDICTIONS.md"
PRED_SHA = "68dee8a6dbea70a17044f0c2c57b6c3f9b04768f29d008170fa4935d3d415d38"

spec = importlib.util.spec_from_file_location("final_exp", FINAL_SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
audit = exp.audit

MODELS = ["Dense_AE", "LSTM_AE", "CNN_AE", "SNN_AE"]
SEEDS = [0, 1, 2]
FOLDS = list(range(1, 11))
CONDITION = "with_epoch_34F"
LAMBDAS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
CONST_D = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
FUT_D = [0.1, 0.5, 1.0, 2.0]
DIST_BINS = [(-np.inf, 0.0), (0.0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, np.inf)]  # (lo, hi]
DIST_LABELS = ["0", "(0,0.1]", "(0.1,1]", "(1,10]", ">10"]
FIRST_LAYER = {"Dense_AE": "encoder", "LSTM_AE": "enc_lstm1", "CNN_AE": "conv1", "SNN_AE": "batch_norm"}

features = list(exp.FEATURES_34)
epoch_idx = features.index("frame.time_epoch")
keep_idx = np.array([i for i in range(len(features)) if i != epoch_idx], dtype=int)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_dir_for(ds, model, seed, fold):
    if model == "SNN_AE":
        return SNN_RUNS / ds / f"seed_{seed:02d}" / f"fold_{fold:02d}"
    return AE3_RUNS / ds / model / f"seed_{seed:02d}" / f"fold_{fold:02d}"


def metrics(y, score, thr):
    att, nor = y == 1, y == 0
    fnr = float(np.mean(score[att] <= thr))
    fpr = float(np.mean(score[nor] > thr))
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        auc = float("nan")
    return fnr, fpr, auc


def non_epoch_score(model, model_name, X):
    rec = audit.predict_flat(model, X, model_name)
    se = np.square(X - rec)
    return np.mean(se[:, keep_idx], axis=1)


def oos_distance(t):
    return np.maximum.reduce([np.zeros_like(t), -t, t - 1.0])


def process_run(ds, sample, model_name, seed, fold, cfg):
    rd = run_dir_for(ds, model_name, seed, fold)
    thr_info = json.loads((rd / "thresholds.json").read_text())
    thr = float(thr_info["non_epoch_threshold"])
    neutral = float(thr_info["neutral_scaled_epoch_value"])

    with np.load(LEGACY_ROOT / "splits" / ds / f"seed_{seed:02d}" / f"fold_{fold:02d}.npz") as z:
        test_idx = z["test_idx"]
    scaler = joblib.load(LEGACY_ROOT / "scalers" / ds / CONDITION / f"seed_{seed:02d}" / f"fold_{fold:02d}.pkl")
    Xall = sample[features].to_numpy(dtype=np.float32)
    X = scaler.transform(Xall[test_idx]).astype(np.float32)
    y = (sample.iloc[test_idx][cfg["label"]].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()
    t0 = X[:, epoch_idx].astype(np.float64).copy()

    tf.keras.backend.clear_session()
    model = audit.build_model(model_name, len(features), ds)
    model.load_weights(rd / "model.weights.h5")

    key = dict(dataset=ds, model=model_name, seed=seed, fold=fold)
    rows, act_rows, dist_rows = [], [], []

    def arm(name, param, t_new):
        Xp = X.copy()
        Xp[:, epoch_idx] = t_new.astype(np.float32)
        s = non_epoch_score(model, model_name, Xp)
        fnr, fpr, auc = metrics(y, s, thr)
        rows.append({**key, "arm": name, "param": param, "fnr": fnr, "fpr": fpr, "auc": auc})
        return s

    # lambda grid (also used for sanity gate and DIST)
    for lam in LAMBDAS:
        t_lam = neutral + lam * (t0 - neutral)
        s = arm("LAMBDA", lam, t_lam)
        d = oos_distance(t_lam)[y == 1]
        det = (s[y == 1] > thr)
        for (lo, hi), lab in zip(DIST_BINS, DIST_LABELS):
            m = (d > lo) & (d <= hi) if np.isfinite(lo) else (d <= hi)
            dist_rows.append({**key, "lambda": lam, "dist_bin": lab, "n": int(m.sum()), "detected": int(det[m].sum())})

    arm("CLIP", np.nan, np.clip(t0, 0.0, 1.0))
    for side in ["above", "below"]:
        for dd in CONST_D:
            val = 1.0 + dd if side == "above" else -dd
            arm(f"CONST_{side}", dd, np.full_like(t0, val))
    rng = np.random.RandomState(260000 + 1000 * seed + fold)
    arm("PERM", np.nan, t0[rng.permutation(len(t0))])
    for dd in FUT_D:
        t_f = t0.copy()
        t_f[y == 0] = 1.0 + dd
        arm("FUT", dd, t_f)

    # activations of first hidden layer
    sub = tf.keras.Model(model.input, model.get_layer(FIRST_LAYER[model_name]).output)
    for lam_name, t_new in [("L1", t0), ("L0", np.full_like(t0, neutral))]:
        Xp = X.copy()
        Xp[:, epoch_idx] = t_new.astype(np.float32)
        a = sub.predict(audit.input_for_model(Xp, model_name), batch_size=1024, verbose=0)
        a = a.reshape(len(Xp), -1)
        nrm = np.linalg.norm(a, axis=1)
        for cls, mask in [("attack", y == 1), ("normal", y == 0)]:
            act_rows.append({**key, "state": lam_name, "class": cls, "layer": FIRST_LAYER[model_name],
                             "l2_mean": float(np.mean(nrm[mask])), "l2_median": float(np.median(nrm[mask]))})

    # support descriptors
    da = oos_distance(t0)
    sup = {**key, "attack_oos_frac": float(np.mean(da[y == 1] > 0)), "attack_dist_median": float(np.median(da[y == 1])),
           "normal_oos_frac": float(np.mean(da[y == 0] > 0)), "normal_dist_median": float(np.median(da[y == 0])),
           "threshold": thr, "neutral": neutral}
    del model, sub
    return rows, act_rows, dist_rows, sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["edge_iiotset", "ciciot2023", "ton_iot"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if sha256_file(PRED_FILE) != PRED_SHA:
        raise RuntimeError("Pre-registered prediction file changed; refusing to run")

    ds = args.dataset
    cfg = exp.DATASETS[ds]
    needed = list(dict.fromkeys(features + [cfg["label"], "label_type", "source_row_index"]))
    sample = pd.read_parquet(cfg["sample"], columns=needed)
    for f in features:
        sample[f] = pd.to_numeric(sample[f], errors="coerce")
    sample[features] = sample[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    out = OUT / "runs" / (ds + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    R, A, D, S = [], [], [], []
    n = 0
    t_start = time.time()
    for model_name in MODELS:
        for seed in SEEDS:
            for fold in FOLDS:
                r, a, d, s = process_run(ds, sample, model_name, seed, fold, cfg)
                R += r; A += a; D += d; S.append(s)
                l1 = next(x for x in r if x["arm"] == "LAMBDA" and x["param"] == 1.0)
                cl = next(x for x in r if x["arm"] == "CLIP")
                print(f"[{time.time()-t_start:7.1f}s] {ds} {model_name} s{seed} f{fold:02d} "
                      f"L1 FNR={l1['fnr']:.4f} CLIP FNR={cl['fnr']:.4f}", flush=True)
                n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break

    pd.DataFrame(R).to_csv(out / "arm_metrics.csv", index=False)
    pd.DataFrame(A).to_csv(out / "activations.csv", index=False)
    pd.DataFrame(D).to_csv(out / "distance_bins.csv", index=False)
    pd.DataFrame(S).to_csv(out / "support.csv", index=False)
    (out / "DONE.json").write_text(json.dumps({"runs": n, "seconds": time.time() - t_start,
                                               "script_sha256": sha256_file(Path(__file__)),
                                               "predictions_sha256": PRED_SHA}, indent=2))
    print("DONE", n)


if __name__ == "__main__":
    main()
