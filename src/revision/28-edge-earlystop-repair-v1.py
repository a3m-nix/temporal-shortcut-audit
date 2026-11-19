#!/usr/bin/env python3
"""
28 - Edge-IIoTset Experiment II re-run with repaired early stopping .

Reuses the frozen v1.2 pipeline (script 20) and the placebo feature construction (script 21).
Only the early-stopping data changes. Pre-specified predictions:
$TEMPORAL_AUDIT_ROOT/temporal_audit_edge_earlystop_repair_v1/metadata/PREREGISTERED_PREDICTIONS.md

Usage: python 28-edge-earlystop-repair-v1.py --variant FITVAL --models Dense_AE LSTM_AE [--conditions ...] [--seeds 0 1 2]
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
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
V4 = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
SRC = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2"
OUT = BASE / "temporal_audit_edge_earlystop_repair_v1"
PRED_FILE = OUT / "metadata" / "PREREGISTERED_PREDICTIONS.md"
PRED_SHA = "b5e696e6f67e0f8f57c3c774d5e5c92e44cff7d0c24b988309625e0035645a41"
DS = "edge_iiotset"
DS_IDX = 0


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


base = load_mod("jk_base_v12", V4 / "20-end-to-end-audit-informed-v1.2.py")
plc = load_mod("jk_placebo_v1", V4 / "21-acquisition-counter-placebo-v1.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["FITVAL", "FIXED30", "ORIG"])
    ap.add_argument("--models", nargs="+", default=list(base.MODELS))
    ap.add_argument("--conditions", nargs="+", default=["A_q_t", "B_q", "E_q_acquisition_counter", "N_q_in_support_noise"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    if sha256_file(PRED_FILE) != PRED_SHA:
        raise RuntimeError("Prediction file changed; refusing to run")

    legacy = base.load_legacy_driver()
    _, specs = base.resolve_condition_specs(legacy)
    q = list(json.loads((SRC / "metadata" / "feature_contract.json").read_text())["q_features"])
    specs.update(plc.new_specs(q))
    df = plc.augmented_prepared(base, {"prepared": SRC / "prepared_data"}, DS)
    a_idx = base.SAFE_CONDITION_NAMES.index("A_q_t")

    for cond in args.conditions:
        spec = specs[cond]
        feats = list(spec["features"])
        fit = df[df["split"] == "fit_normal"]
        scaler = MinMaxScaler().fit(base.condition_raw_matrix(fit, feats, spec["preprocess"]))
        _, xfit = base.transform_partition(df, "fit_normal", spec, scaler)
        _, xearly = base.transform_partition(df, "earlystop_normal", spec, scaler)
        _, xcal = base.transform_partition(df, "calibration_normal", spec, scaler)
        test_part, xtest = base.transform_partition(df, "test", spec, scaler)
        ytest = test_part["y"].to_numpy(dtype=np.int8)
        qi = base.q_indices(feats, q)
        cidx = a_idx if cond.startswith(("E_", "N_")) else base.SAFE_CONDITION_NAMES.index(cond)

        for model_name in args.models:
            mi = base.MODELS.index(model_name)
            for seed in args.seeds:
                rd = OUT / "runs" / args.variant / cond / model_name / f"seed_{seed:02d}"
                if (rd / "metrics.json").exists():
                    print("[RESUME]", rd, flush=True)
                    continue
                rd.mkdir(parents=True, exist_ok=True)
                mseed = base.model_seed(DS_IDX, cidx, mi, seed)
                model = base.build_model(legacy, model_name, len(feats), DS, mseed, compile_model=True)
                inp = lambda x: legacy.audit.input_for_model(x, model_name)
                tgt = lambda x: legacy.audit.target_for_model(x, model_name)
                t0 = time.time()
                if args.variant == "ORIG":
                    xtr, xval = xfit, xearly
                    cbs = [EarlyStopping(monitor="val_loss", patience=base.PATIENCE, restore_best_weights=True, verbose=0)]
                elif args.variant == "FITVAL":
                    rng = np.random.RandomState(280000 + seed)
                    perm = rng.permutation(len(xfit))
                    nval = int(round(0.15 * len(xfit)))
                    xval, xtr = xfit[perm[:nval]], xfit[perm[nval:]]
                    cbs = [EarlyStopping(monitor="val_loss", patience=base.PATIENCE, restore_best_weights=True, verbose=0)]
                else:  # FIXED30
                    xtr, xval, cbs = xfit, xearly, []
                hist = model.fit(inp(xtr), tgt(xtr), validation_data=(inp(xval), tgt(xval)),
                                 epochs=base.MAX_EPOCHS, batch_size=base.BATCH_SIZE, callbacks=cbs, verbose=0)
                h = pd.DataFrame(hist.history)
                h.insert(0, "epoch", np.arange(1, len(h) + 1))
                h.to_csv(rd / "training_history.csv", index=False)
                cal_pred = base.predict_flat(legacy, model, xcal, model_name)
                test_pred = base.predict_flat(legacy, model, xtest, model_name)
                cal_q, _, _ = base.reconstruction_scores(xcal, cal_pred, qi)
                test_q, _, _ = base.reconstruction_scores(xtest, test_pred, qi)
                thr = base.percentile_threshold(cal_q)
                m = base.metric_dict(ytest, test_q, thr)
                model.save_weights(rd / "model.weights.h5")
                pd.DataFrame({"source_row_index": test_part["source_row_index"].to_numpy(np.int64),
                              "analysis_unit_id": test_part["analysis_unit_id"].astype(str).to_numpy(),
                              "label_type": test_part["label_type"].astype(str).to_numpy(),
                              "y_true": ytest, "q_score": test_q, "q_prediction": (test_q > thr).astype(np.int8)}
                             ).to_csv(rd / "test_predictions.csv.gz", index=False, compression="gzip")
                best_epoch = int(np.argmin(h["val_loss"].to_numpy()) + 1)
                metrics = {"dataset": DS, "variant": args.variant, "condition": cond, "model": model_name, "seed": seed,
                           "model_seed": int(mseed), "features": feats, "epochs_used": int(len(h)),
                           "best_val_epoch": best_epoch, "train_sec": time.time() - t0, "q_threshold": thr,
                           "q_metrics": m, "script_sha256": sha256_file(Path(__file__)), "predictions_sha256": PRED_SHA}
                (rd / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
                print(f"[{args.variant}] {cond} {model_name} s{seed} epochs={len(h)} best={best_epoch} "
                      f"FNR={m['fnr']:.4f} FPR={m['fpr']:.4f} AUC={m.get('roc_auc', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
