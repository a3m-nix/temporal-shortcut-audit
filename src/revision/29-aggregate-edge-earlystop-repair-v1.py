#!/usr/bin/env python3
"""29 - Aggregate the Edge early-stopping repair (script 28) and score the pre-specified predictions."""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "/temporal_audit_edge_earlystop_repair_v1")
TAB = OUT / "tables"
TAB.mkdir(exist_ok=True)
NON_SNN = ["Dense_AE", "LSTM_AE", "CNN_AE"]
MODELS = ["SNN_AE"] + NON_SNN
SHORT = {"A_q_t": "A", "B_q": "B", "E_q_acquisition_counter": "E", "N_q_in_support_noise": "N"}
# Original manuscript Table 6, Edge-IIoTset rows (FNR %)
OLD = {"SNN_AE": (62.05, 16.79, 16.82, 31.92), "Dense_AE": (61.18, 61.92, 61.86, 61.96),
       "LSTM_AE": (61.57, 62.02, 58.49, 46.78), "CNN_AE": (61.88, 61.52, 60.15, 60.90)}

rows = []
for p in (OUT / "runs").glob("*/*/*/seed_*/metrics.json"):
    m = json.loads(p.read_text())
    if m["variant"] == "ORIG":
        continue
    q = m["q_metrics"]
    rows.append({"variant": m["variant"], "cond": SHORT[m["condition"]], "model": m["model"], "seed": m["seed"],
                 "epochs": m["epochs_used"], "best_epoch": m["best_val_epoch"], "fnr": 100 * q["fnr"],
                 "fpr": 100 * q["fpr"], "auc": q.get("roc_auc", np.nan), "pr_auc": q.get("pr_auc", np.nan)})
runs = pd.DataFrame(rows)
assert len(runs) == 96, len(runs)
runs.to_csv(TAB / "edge_repair_run_metrics.csv", index=False)

summ = runs.groupby(["variant", "model", "cond"]).agg(fnr=("fnr", "mean"), fnr_sd=("fnr", "std"), fpr=("fpr", "mean"),
                                                     auc=("auc", "mean"), epochs=("epochs", "median"),
                                                     best_epoch=("best_epoch", "median")).reset_index()
summ.to_csv(TAB / "edge_repair_summary.csv", index=False)

t6 = []
for v in ["FITVAL", "FIXED30"]:
    for mdl in MODELS:
        s = summ[(summ.variant == v) & (summ.model == mdl)].set_index("cond")
        r = {"variant": v, "model": mdl}
        for c in "ABEN":
            r[c] = s.loc[c, "fnr"]
            r[f"{c}_fpr"] = s.loc[c, "fpr"]
            r[f"{c}_auc"] = s.loc[c, "auc"]
        r["B-A"], r["B-E"], r["N-B"] = r["B"] - r["A"], r["B"] - r["E"], r["N"] - r["B"]
        o = OLD[mdl]
        r.update({"old_A": o[0], "old_B": o[1], "old_E": o[2], "old_N": o[3], "old_B-A": o[1] - o[0]})
        t6.append(r)
t6 = pd.DataFrame(t6)
t6.to_csv(TAB / "edge_table6_repaired_vs_original.csv", index=False)

# per-seed sign consistency of B-A
piv = runs.pivot_table(index=["variant", "model", "seed"], columns="cond", values="fnr").reset_index()
piv["B-A"] = piv["B"] - piv["A"]
piv.to_csv(TAB / "edge_repair_seed_contrasts.csv", index=False)

V = []
fv = summ[summ.variant == "FITVAL"]
ep = fv[fv.model.isin(NON_SNN)].groupby("model").epochs.median()
aucB = fv[(fv.cond == "B") & fv.model.isin(NON_SNN)].set_index("model").auc
V.append(("R1", ep.median() > 10 and (aucB > 0.90).sum() >= 2, f"median epochs non-SNN={ep.to_dict()}; B AUC={aucB.round(3).to_dict()}"))
fnrB = fv[(fv.cond == "B") & fv.model.isin(NON_SNN)].set_index("model").fnr
V.append(("R2", (fnrB < 30).sum() >= 2, f"B FNR={fnrB.round(2).to_dict()}"))
tf = t6[t6.variant == "FITVAL"].set_index("model")
V.append(("R3", (tf.loc[NON_SNN, "B-A"] > 0).sum() >= 2, f"B-A={tf.loc[NON_SNN, 'B-A'].round(2).to_dict()}"))
V.append(("R4", (tf.loc[NON_SNN, "N-B"].abs() < 5).sum() >= 2, f"N-B={tf.loc[NON_SNN, 'N-B'].round(2).to_dict()}"))
V.append(("R5", abs(tf.loc["SNN_AE", "B-A"]) < 45.26, f"SNN |B-A|={abs(tf.loc['SNN_AE', 'B-A']):.2f} vs 45.26"))
tx = t6[t6.variant == "FIXED30"].set_index("model")
agree = int((np.sign(tx.loc[MODELS, "B-A"]) == np.sign(tf.loc[MODELS, "B-A"])).sum())
V.append(("R6", agree >= 3, f"sign agreement FIXED30 vs FITVAL: {agree}/4"))
V = pd.DataFrame(V, columns=["prediction", "met", "detail"])
V.to_csv(TAB / "PREDICTION_VERDICTS.csv", index=False)

pd.set_option("display.width", 250)
print(summ.round(3).to_string())
print(t6[["variant", "model", "A", "B", "E", "N", "B-A", "B-E", "N-B", "old_A", "old_B", "old_E", "old_N", "A_fpr", "B_fpr", "A_auc", "B_auc"]].round(2).to_string())
print(piv.round(2).to_string())
print(V.to_string())
