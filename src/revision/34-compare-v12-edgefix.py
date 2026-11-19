#!/usr/bin/env python3
"""34 - Compare v1.2 vs v1.2_edgefix: (1) CIC/ToN identical, (2) Edge-IIoTset old -> new for manuscript numbers."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
OLD, NEW = B / "temporal_audit_temporal_upgrade_e2e_v1_2", B / "temporal_audit_temporal_upgrade_e2e_v1_2_edgefix"
P_OLD = B / "temporal_audit_posthoc_strengthening_v1/06_thresholdfree_multibudget_translation"
P_NEW = B / "temporal_audit_posthoc_strengthening_v1/06_thresholdfree_multibudget_translation_edgefix"
OUT = NEW / "edgefix_comparison"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

# (1) identity of non-Edge rows
id_rows = []
for root_old, root_new, sub in [(OLD, NEW, "tables"), (P_OLD, P_NEW, "tables")]:
    for f in sorted((root_old / sub).glob("*.csv")):
        g = root_new / sub / f.name
        if not g.exists():
            id_rows.append({"table": f"{root_old.name}/{f.name}", "status": "MISSING_NEW"})
            continue
        a, b = pd.read_csv(f), pd.read_csv(g)
        if "dataset" not in a.columns:
            id_rows.append({"table": f"{root_old.name}/{f.name}", "status": "no dataset column"})
            continue
        a, b = a[a.dataset != "edge_iiotset"], b[b.dataset != "edge_iiotset"]
        drop = [c for c in a.columns if any(k in c.lower() for k in ["time", "sec", "rss", "path", "utc", "sha"])]
        a, b = a.drop(columns=drop, errors="ignore"), b.drop(columns=drop, errors="ignore")
        keys = [c for c in a.columns if a[c].dtype == object]
        a = a.sort_values(keys).reset_index(drop=True) if keys else a
        b = b.sort_values(keys).reset_index(drop=True) if keys else b
        if a.shape != b.shape:
            id_rows.append({"table": f"{root_old.name}/{f.name}", "status": f"SHAPE {a.shape} vs {b.shape}"})
            continue
        num = a.select_dtypes("number").columns
        diff = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if len(num) else 0.0
        same_obj = all((a[c].astype(str) == b[c].astype(str)).all() for c in keys)
        id_rows.append({"table": f"{root_old.name}/{f.name}", "status": "IDENTICAL" if diff < 1e-12 and same_obj else "DIFF",
                        "max_abs_numeric_diff": diff, "rows": len(a)})
ident = pd.DataFrame(id_rows)
ident.to_csv(OUT / "non_edge_identity_check.csv", index=False)
print(ident.to_string())


def edge(path):
    d = pd.read_csv(path)
    return d[d.dataset == "edge_iiotset"]


# (2a) neural summary
cols = ["condition", "model", "q_fnr_mean", "q_fpr_mean", "q_roc_auc_mean", "epochs_used_mean"]
no, nn = edge(OLD / "tables/neural_metrics_summary.csv")[cols], edge(NEW / "tables/neural_metrics_summary.csv")[cols]
ns = no.merge(nn, on=["condition", "model"], suffixes=("_old", "_new"))
ns.to_csv(OUT / "edge_neural_summary_old_vs_new.csv", index=False)
print((ns.set_index(["condition", "model"]) * 1).round(4).to_string())

# (2b) OR ensemble
oo, on = edge(OLD / "tables/security_or_ensemble.csv"), edge(NEW / "tables/security_or_ensemble.csv")
ens = oo.groupby("condition")[["recall", "fpr"]].mean().join(on.groupby("condition")[["recall", "fpr"]].mean(), lsuffix="_old", rsuffix="_new")
ens.to_csv(OUT / "edge_or_ensemble_old_vs_new.csv")
print((100 * ens).round(2).to_string())

# (2c) shared transitions A -> B (detected by all four -> missed by all four)
def trans(path):
    d = edge(path)
    d = d[(d.contrast == "B_minus_A_timestamp_removal_penalty") & (d.attack_group == "ALL")]
    return pd.Series({"evaluations": d.attack_packet_seed_n.sum(), "detected_all_to_missed_all": d.left_detected_all_to_right_missed_all_n.sum(),
                      "missed_all_to_detected_all": d.left_missed_all_to_right_detected_all_n.sum()})
tr = pd.DataFrame({"old": trans(OLD / "tables/security_shared_cross_model_transitions.csv"),
                   "new": trans(NEW / "tables/security_shared_cross_model_transitions.csv")})
tr.to_csv(OUT / "edge_shared_transitions_old_vs_new.csv")
print(tr.to_string())

# (2d) analysis-unit contrasts
c = ["contrast", "model", "analysis_unit_n", "mean_delta_fnr_right_minus_left", "analysis_unit_bootstrap_ci95_low", "analysis_unit_bootstrap_ci95_high", "signflip_p_holm"]
co, cn = edge(OLD / "tables/analysis_unit_contrast_statistics.csv")[c], edge(NEW / "tables/analysis_unit_contrast_statistics.csv")[c]
cc = co.merge(cn, on=["contrast", "model"], suffixes=("_old", "_new"))
cc.to_csv(OUT / "edge_analysis_unit_contrasts_old_vs_new.csv", index=False)
print(cc.round(4).to_string())

# (2e) interventions (A only, both tracks)
i = ["condition", "model", "scheme", "track", "mean_original_fnr", "mean_shifted_fnr", "mean_delta_fnr_shifted_minus_original", "signflip_p_holm"]
io, inew = edge(OLD / "tables/intervention_analysis_unit_statistics.csv")[i], edge(NEW / "tables/intervention_analysis_unit_statistics.csv")[i]
ii = io.merge(inew, on=["condition", "model", "scheme", "track"], suffixes=("_old", "_new"))
ii = ii[ii.condition == "A_q_t"]
ii.to_csv(OUT / "edge_interventions_old_vs_new.csv", index=False)
print(ii.round(4).to_string())

# (2f) multibudget (script 24)
for f in ["manuscript_multibudget_summary.csv", "manuscript_translation_fpr_summary.csv", "p999_calibration_tail_support.csv"]:
    a, b = edge(P_OLD / "tables" / f), edge(P_NEW / "tables" / f)
    a.to_csv(OUT / f"edge_{f.replace('.csv', '')}_old.csv", index=False)
    b.to_csv(OUT / f"edge_{f.replace('.csv', '')}_new.csv", index=False)
    print("==", f, "OLD"); print(a.round(3).to_string()); print("== NEW"); print(b.round(3).to_string())
