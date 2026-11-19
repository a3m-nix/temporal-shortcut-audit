#!/usr/bin/env python3
"""32 - Aggregate the RPi5 deployment replay (rpi_runner.py output) and score predictions D1-D5."""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "/rpi5_temporal_deployment_replay_v1")
RAW = OUT / "device_results" / "full"
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
H = ["as_recorded", "+1h", "+1d", "+7d", "+30d", "+365d", "NOW"]
DS = ["edge_iiotset", "ciciot2023", "ton_iot"]
LAB = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023", "ton_iot": "ToN-IoT"}
MODELS = ["CNN_AE", "Dense_AE", "LSTM_AE", "SNN_AE"]

acc = pd.DataFrame(json.loads((RAW / "accuracy_horizons.json").read_text()))
pw = pd.DataFrame(json.loads((RAW / "power_windows.json").read_text()))
acc["fpr_pct"], acc["fnr_pct"] = 100 * acc.fpr, 100 * acc.fnr
wide = acc.pivot_table(index=["dataset", "cond", "model"], columns="horizon", values="fpr_pct")[H].reset_index()
wide.to_csv(TAB / "fpr_by_horizon.csv", index=False)
acc.pivot_table(index=["dataset", "cond", "model"], columns="horizon", values="fnr_pct")[H].reset_index().to_csv(TAB / "fnr_by_horizon.csv", index=False)
pw.to_csv(TAB / "edge_cost_windows.csv", index=False)

V = []
b = acc[acc.cond == "B_q"].groupby("key")
V.append(("D1", bool(all((g.fpr.nunique() == 1) and (g.fnr.nunique() == 1) for _, g in b)), "B FPR/FNR identical across horizons"))
ok2, det2 = True, []
for (ds, mdl), g in acc[acc.cond == "A_q_t"].groupby(["dataset", "model"]):
    f = g.set_index("horizon").loc[["+1d", "+7d", "+30d", "+365d"], "fpr_pct"].to_numpy()
    mono = bool(np.all(np.diff(f) >= -1.0))
    ok2 &= mono
    if not mono:
        det2.append(f"{ds}/{mdl}:{np.round(f, 2).tolist()}")
V.append(("D2", ok2, "non-monotone: " + ("; ".join(det2) or "none")))
ok3, det3 = True, []
for ds in DS:
    a = wide[(wide.dataset == ds) & (wide.cond == "A_q_t")]
    n365, nnow = int((a["+365d"] >= 90).sum()), int((a["NOW"] >= 90).sum())
    bb = wide[(wide.dataset == ds) & (wide.cond == "B_q")]
    ok3 &= n365 >= 3 and nnow >= 3
    det3.append(f"{ds}: A>=90% at +365d {n365}/4, NOW {nnow}/4; B FPR {bb['NOW'].round(2).tolist()}")
V.append(("D3", ok3, "; ".join(det3)))
ok4, det4 = True, []
for ds in ["ciciot2023", "ton_iot"]:
    a = wide[(wide.dataset == ds) & (wide.cond == "A_q_t")].set_index("model")["+1d"]
    bb = wide[(wide.dataset == ds) & (wide.cond == "B_q")].set_index("model")["+1d"]
    n = int((a > bb.reindex(a.index)).sum())
    ok4 &= n >= 3
    det4.append(f"{ds}: A>B at +1d {n}/4 (A={a.round(2).to_dict()})")
V.append(("D4", ok4, "; ".join(det4)))
cost = pw.pivot_table(index=["dataset", "model"], columns="cond", values=["latency_ms_per_row", "active_mj_per_row", "active_w", "delta_w"])
rl = (cost[("latency_ms_per_row", "A_q_t")] / cost[("latency_ms_per_row", "B_q")] - 1).abs()
re = (cost[("active_mj_per_row", "A_q_t")] / cost[("active_mj_per_row", "B_q")] - 1).abs()
V.append(("D5", bool((rl < 0.10).all() and (re < 0.10).all()),
          f"max |A/B-1| latency={100*rl.max():.1f}%, energy={100*re.max():.1f}%"))
cost.columns = [f"{a}__{c}" for a, c in cost.columns]
cost.reset_index().to_csv(TAB / "edge_cost_A_vs_B.csv", index=False)
V = pd.DataFrame(V, columns=["prediction", "met", "detail"])
V.to_csv(TAB / "PREDICTION_VERDICTS.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
x = np.arange(len(H))
for ax, ds in zip(axes, DS):
    for mdl, c in zip(MODELS, ["C0", "C1", "C2", "C3"]):
        a = wide[(wide.dataset == ds) & (wide.cond == "A_q_t") & (wide.model == mdl)][H].to_numpy().ravel()
        bb = wide[(wide.dataset == ds) & (wide.cond == "B_q") & (wide.model == mdl)][H].to_numpy().ravel()
        ax.plot(x, a, "-o", color=c, ms=3, label=f"{mdl} with time (A)")
        ax.plot(x, bb, ":", color=c, label=f"{mdl} without time (B)")
    ax.set_xticks(x, H, rotation=35, fontsize=8)
    ax.set_title(LAB[ds])
    ax.grid(alpha=.3)
axes[0].set_ylabel("FPR on normal traffic (%)")
fig.supxlabel("deployment horizon after the training capture (RPi5 replay, fixed threshold)", fontsize=9, y=0.12)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(FIG / "fig_rpi5_alert_flood_by_horizon.png", dpi=300)

pd.set_option("display.width", 250)
print(wide.round(2).to_string())
print(pw[["key", "throughput_rows_s", "latency_ms_per_row", "idle_w", "active_w", "delta_w", "active_mj_per_row", "idle_n", "active_n", "cpu_temp_c"]].round(4).to_string())
print(V.to_string())
