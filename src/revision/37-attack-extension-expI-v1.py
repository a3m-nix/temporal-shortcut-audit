#!/usr/bin/env python3
"""
37 - Attack-family extension under the Experiment I protocol (revision, R5.2).

  build                                  -> draw the ToN-IoT DoS and Edge-IIoTset non-flood samples (fixed seeds)
  run --dataset ton_dos|edge_nonflood --models ...   -> 3 seeds x 10 folds per model, lambda grid + clipping

Pre-specified predictions: temporal_audit_attack_extension_v1/metadata/PREREGISTERED_PREDICTIONS.md
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
V4 = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
OUT = BASE / "temporal_audit_attack_extension_v1"
PRED_SHA = "4b8e417c3e8b44092cfae56dd644f06211f891ff76ab2006947facd7579cbf7e"
LAMBDAS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
DS = {
    "ton_dos": {"base": "ton_iot", "idx": 10, "normal": "Normal"},
    "edge_nonflood": {"base": "edge_iiotset", "idx": 11, "normal": "Normal"},
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_exp():
    spec = importlib.util.spec_from_file_location("final_exp", V4 / "04-confirmatory-allminmax-3seed-10fold.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    return exp


def build():
    import pyarrow.parquet as pq
    exp = load_exp()
    feats = list(exp.FEATURES_34)
    cols = list(dict.fromkeys(feats + ["label", "label_type", "source_row_index"]))
    # --- ToN-IoT DoS
    base = pd.read_parquet(exp.DATASETS["ton_iot"]["sample"], columns=cols)
    normal = base[base["label"].astype(str) == "Normal"].copy()
    parent = BASE / "dataset-from-pcap/ton_iot_bidir.parquet"
    pf = pq.ParquetFile(parent)
    parts, offset = [], 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=feats + ["label", "label_type"]).to_pandas()
        m = t["label_type"].astype(str) == "DoS"
        if m.any():
            sel = t[m].copy()
            sel["source_row_index"] = np.arange(offset, offset + len(t))[m.to_numpy()]
            parts.append(sel)
        offset += len(t)
    dos = pd.concat(parts, ignore_index=True)
    e = dos["frame.time_epoch"].astype(float).to_numpy()
    # Addendum 01: equal-count (quantile) bins, because DoS occurs in four bursts and equal-width bins are empty
    edges = np.quantile(e, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(e, edges[1:-1], right=False), 0, 9)
    rng = np.random.RandomState(42)
    pick = []
    for k in range(10):
        idx = np.flatnonzero(b == k)
        if len(idx) < 1000:
            raise RuntimeError(f"DoS bin {k} has only {len(idx)} rows")
        pick.append(rng.choice(idx, 1000, replace=False))
    pick = np.concatenate(pick)
    att = dos.iloc[pick].copy()
    att["label_type"] = [f"DoS_bin_{int(x):02d}" for x in b[pick]]
    ton = pd.concat([normal, att[cols]], ignore_index=True)
    ton.to_parquet(OUT / "ton_dos_sample.parquet", index=False)
    # --- Edge-IIoTset non-flood
    base = pd.read_parquet(exp.DATASETS["edge_iiotset"]["sample"], columns=cols)
    normal = base[base["label"].astype(str) == "Normal"].copy()
    nf = pd.read_parquet(OUT / "edge_nonflood_packets.parquet")
    rng = np.random.RandomState(42)
    att = []
    for fam in sorted(nf["label_type"].unique()):
        g = nf[nf["label_type"] == fam]
        att.append(g.iloc[rng.choice(len(g), 1000, replace=False)])
    att = pd.concat(att, ignore_index=True)
    att["source_row_index"] = 10_000_000_000 + np.arange(len(att))  # synthetic ids, outside parent range
    edge = pd.concat([normal, att[cols]], ignore_index=True)
    edge.to_parquet(OUT / "edge_nonflood_sample.parquet", index=False)
    meta = {"ton_dos": {"normal": int(len(normal)), "attack": 10000, "dos_parent_rows": int(len(dos)),
                        "bin_edges": edges.tolist(), "sha256": sha(OUT / "ton_dos_sample.parquet")},
            "edge_nonflood": {"normal": int((edge["label"] == "Normal").sum()), "attack": int(len(att)),
                              "families": sorted(nf["label_type"].unique().tolist()), "sha256": sha(OUT / "edge_nonflood_sample.parquet")}}
    (OUT / "metadata" / "sample_build.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1)[:1500])


def run(ds_name, models):
    import tensorflow as tf
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    if sha(OUT / "metadata" / "PREREGISTERED_PREDICTIONS.md") != PRED_SHA:
        raise SystemExit("prediction file changed")
    exp = load_exp()
    audit = exp.audit
    cfg = DS[ds_name]
    feats = list(exp.FEATURES_34)
    ei = feats.index("frame.time_epoch")
    keep = np.array([i for i in range(len(feats)) if i != ei])
    sample = pd.read_parquet(OUT / f"{ds_name}_sample.parquet")
    for f in feats:
        sample[f] = pd.to_numeric(sample[f], errors="coerce")
    sample[feats] = sample[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_all = (sample["label"].astype(str) != cfg["normal"]).astype(np.int8).to_numpy()
    attack_idx, normal_idx = np.where(y_all == 1)[0], np.where(y_all == 0)[0]
    rng = np.random.RandomState(exp.CORPUS_SEED)
    corpus = np.concatenate([rng.choice(normal_idx, size=len(attack_idx), replace=False), attack_idx]).astype(np.int64)
    rng.shuffle(corpus)
    strata = np.where(y_all[corpus] == 0, "NORMAL", sample["label_type"].astype(str).to_numpy()[corpus])
    Xall = sample[feats].to_numpy(dtype=np.float32)
    fam_all = sample["label_type"].astype(str).to_numpy()
    for model_name in models:
        for seed in exp.SEEDS:
            skf = StratifiedKFold(n_splits=exp.N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tv, te) in enumerate(skf.split(np.zeros((len(corpus), 1)), strata), start=1):
                rd = OUT / "runs" / ds_name / model_name / f"seed_{seed:02d}" / f"fold_{fold:02d}"
                if (rd / "metrics.json").exists():
                    continue
                rd.mkdir(parents=True, exist_ok=True)
                tv_idx, te_idx = corpus[tv], corpus[te]
                pool = tv_idx[y_all[tv_idx] == 0].copy()
                np.random.RandomState(900000 + seed * 1000 + fold).shuffle(pool)
                cut = int(round((1.0 - exp.VAL_RATIO_WITHIN_TRAIN_NORMAL) * len(pool)))
                tr_idx, va_idx = pool[:cut], pool[cut:]
                scaler = audit.make_scaler(cfg["base"], len(tr_idx))
                Xtr = scaler.fit_transform(Xall[tr_idx]).astype(np.float32)
                Xva = scaler.transform(Xall[va_idx]).astype(np.float32)
                Xte = scaler.transform(Xall[te_idx]).astype(np.float32)
                yte = y_all[te_idx]
                fam = fam_all[te_idx]
                mseed = exp.model_seed(cfg["idx"], seed, fold, model_name)
                exp.set_seed(mseed)
                tf.keras.backend.clear_session()
                model = audit.build_model(model_name, len(feats), cfg["base"])
                model.compile(optimizer=Adam(learning_rate=exp.LEARNING_RATE,
                                             clipnorm=1.0 if model_name in ("Dense_AE", "SNN_AE") else None), loss="mse")
                t0 = time.time()
                h = model.fit(audit.input_for_model(Xtr, model_name), audit.target_for_model(Xtr, model_name),
                              validation_data=(audit.input_for_model(Xva, model_name), audit.target_for_model(Xva, model_name)),
                              epochs=exp.EPOCHS, batch_size=exp.BATCH_SIZE,
                              callbacks=[EarlyStopping(monitor="val_loss", patience=exp.PATIENCE, restore_best_weights=True, verbose=0)],
                              verbose=0)
                va_pred = audit.predict_flat(model, Xva, model_name)
                thr = float(np.percentile(np.mean(np.square(Xva - va_pred)[:, keep], axis=1), exp.OPERATING_PERCENTILE))
                neutral = float(np.median(Xtr[:, ei]))
                t_orig = Xte[:, ei].copy()
                rows, famrows = [], []
                arms = [("LAMBDA", lam, neutral + lam * (t_orig - neutral)) for lam in LAMBDAS] + [("CLIP", np.nan, np.clip(t_orig, 0, 1))]
                for arm, par, tnew in arms:
                    Xp = Xte.copy()
                    Xp[:, ei] = tnew.astype(np.float32)
                    s = np.mean(np.square(Xp - audit.predict_flat(model, Xp, model_name))[:, keep], axis=1)
                    m = audit.metric_dict(yte, s, thr)
                    rows.append({"arm": arm, "param": par, "fnr": m["fnr"], "fpr": m["fpr"], "auc": m.get("auc", np.nan)})
                    if arm == "CLIP" or par in (0.0, 1.0):
                        for f in np.unique(fam[yte == 1]):
                            mm = (yte == 1) & (fam == f)
                            famrows.append({"arm": arm, "param": par, "family": f, "n": int(mm.sum()), "fnr": float(np.mean(s[mm] <= thr))})
                d = np.maximum.reduce([np.zeros_like(t_orig), -t_orig, t_orig - 1.0])
                meta = {"dataset": ds_name, "model": model_name, "seed": seed, "fold": fold, "model_seed": int(mseed),
                        "epochs_used": len(h.history["loss"]), "threshold": thr, "neutral": neutral, "train_sec": time.time() - t0,
                        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
                        "attack_oos_frac": float(np.mean(d[yte == 1] > 0)), "attack_dist_median": float(np.median(d[yte == 1])),
                        "normal_oos_frac": float(np.mean(d[yte == 0] > 0)), "predictions_sha256": PRED_SHA}
                pd.DataFrame(rows).to_csv(rd / "arms.csv", index=False)
                pd.DataFrame(famrows).to_csv(rd / "families.csv", index=False)
                (rd / "metrics.json").write_text(json.dumps(meta, indent=2))
                l1 = [r for r in rows if r["arm"] == "LAMBDA" and r["param"] == 1.0][0]
                l0 = [r for r in rows if r["arm"] == "LAMBDA" and r["param"] == 0.0][0]
                cl = rows[-1]
                print(f"{ds_name} {model_name} s{seed} f{fold:02d} ep={meta['epochs_used']} FNR l1={l1['fnr']:.4f} l0={l0['fnr']:.4f} clip={cl['fnr']:.4f} oos={meta['attack_oos_frac']:.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "run":
        a = sys.argv
        run(a[a.index("--dataset") + 1], a[a.index("--models") + 1:])
