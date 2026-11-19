#!/usr/bin/env python3
"""
31 - Export frozen Exp II A/B models (seed 0) and chronological replay streams for the RPi5
deployment replay . Verifies that each exported SavedModel reproduces the
stored q-score metrics on the server before packaging.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
V4 = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
SRC = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2"
REPAIR = BASE / "temporal_audit_edge_earlystop_repair_v1"
OUT = BASE / "rpi5_temporal_deployment_replay_v1"
BUNDLE = OUT / "bundle"
MODELS = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
CONDS = ["A_q_t", "B_q"]
DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("m28", V4 / "28-edge-earlystop-repair-v1.py")
m28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m28)
base = m28.base
legacy = base.load_legacy_driver()
_, specs = base.resolve_condition_specs(legacy)
q = list(json.loads((SRC / "metadata" / "feature_contract.json").read_text())["q_features"])


class Serve(tf.Module):
    def __init__(self, model, name, d):
        super().__init__()
        self.model, self.name_, self.d = model, name, d

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def recon(self, x):
        if self.name_ == "LSTM_AE":
            y = self.model(tf.reshape(x, [-1, 1, self.d]), training=False)
        elif self.name_ == "CNN_AE":
            y = self.model(tf.reshape(x, [-1, self.d, 1]), training=False)
        else:
            y = self.model(x, training=False)
        return tf.reshape(y, [-1, self.d])


def main():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ds in DATASETS:
        df = base.load_prepared({"prepared": SRC / "prepared_data"}, ds)
        fit = df[df["split"] == "fit_normal"]
        test = df[df["split"] == "test"].copy()
        order = np.argsort(test["frame.time_epoch"].to_numpy(np.float64), kind="mergesort")
        test = test.iloc[order].reset_index(drop=True)
        featsA = list(specs["A_q_t"]["features"])
        Xraw = base.condition_raw_matrix(test, featsA, "precision_safe")
        np.savez_compressed(BUNDLE / f"{ds}_replay.npz", X_raw_A=Xraw, y=test["y"].to_numpy(np.int8),
                            label_type=test["label_type"].astype(str).to_numpy(),
                            features_A=np.asarray(featsA), fit_epoch_min=float(fit["frame.time_epoch"].min()),
                            fit_epoch_max=float(fit["frame.time_epoch"].max()))
        for cond in CONDS:
            sp = specs[cond]
            feats = list(sp["features"])
            if ds == "edge_iiotset":
                scaler = MinMaxScaler().fit(base.condition_raw_matrix(fit, feats, sp["preprocess"]))
            else:
                scaler = joblib.load(SRC / "metadata" / "scalers" / ds / cond / "scaler.pkl")
            Xc = base.condition_raw_matrix(test, feats, sp["preprocess"])
            Xs = scaler.transform(Xc).astype(np.float32)
            y = test["y"].to_numpy(np.int8)
            qi = base.q_indices(feats, q)
            for mdl in MODELS:
                if ds == "edge_iiotset":
                    rd = REPAIR / "runs" / "FITVAL" / cond / mdl / "seed_00"
                    stored = json.loads((rd / "metrics.json").read_text())
                    thr, sfnr, sfpr = stored["q_threshold"], stored["q_metrics"]["fnr"], stored["q_metrics"]["fpr"]
                else:
                    rd = SRC / "runs" / ds / cond / mdl / "seed_00"
                    stored = json.loads((rd / "metrics.json").read_text())
                    thr = stored["q_threshold_p99_calibration_normal"]
                    sfnr, sfpr = stored["q_score_metrics"]["fnr"], stored["q_score_metrics"]["fpr"]
                model = base.build_model(legacy, mdl, len(feats), ds, 0, compile_model=False)
                model.load_weights(rd / "model.weights.h5")
                serve = Serve(model, mdl, len(feats))
                rec = serve.recon(tf.constant(Xs)).numpy()
                qs = np.mean(np.square(Xs.astype(np.float64) - rec.astype(np.float64))[:, qi], axis=1)
                fnr, fpr = float(np.mean(qs[y == 1] <= thr)), float(np.mean(qs[y == 0] > thr))
                ok = abs(fnr - sfnr) < 2e-3 and abs(fpr - sfpr) < 2e-3
                key = f"{ds}__{cond}__{mdl}"
                tf.saved_model.save(serve, str(BUNDLE / "models" / key), signatures={"recon": serve.recon})
                meta = {"key": key, "dataset": ds, "cond": cond, "model": mdl, "features": feats,
                        "q_idx": qi.tolist(), "epoch_idx": feats.index("frame.time_epoch") if "frame.time_epoch" in feats else None,
                        "scaler_min": scaler.min_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                        "threshold": float(thr), "server_fnr": fnr, "server_fpr": fpr,
                        "stored_fnr": sfnr, "stored_fpr": sfpr, "reproduces_stored": bool(ok),
                        "source_run": str(rd)}
                (BUNDLE / "models" / key / "meta.json").write_text(json.dumps(meta, indent=2))
                manifest.append(meta)
                print(f"{key:40s} FNR {100*fnr:6.2f} (stored {100*sfnr:6.2f})  FPR {100*fpr:5.2f} (stored {100*sfpr:5.2f})  ok={ok}", flush=True)
                tf.keras.backend.clear_session()
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(m["reproduces_stored"] for m in manifest):
        raise SystemExit("EXPORT VERIFICATION FAILED")
    print("EXPORT OK", len(manifest))


if __name__ == "__main__":
    main()
