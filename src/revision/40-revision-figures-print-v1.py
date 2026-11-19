#!/usr/bin/env python3
"""40 - Print-size versions (7.2 in wide, 8-9 pt) of the revision figures: Fig. 5 (class-independent OOS control),
Fig. 6 (detection by OOS distance), Fig. 7 (RPi5 alert flood). Data: outputs of scripts 26/27 and 32."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
M = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold/controlled_temporal_intervention/mechanism_oos_revision_v1"
RPI = BASE / "rpi5_temporal_deployment_replay_v1"
OUT = BASE / "temporal_audit_revision_figures_v1"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
DS = [("edge_iiotset", "Edge-IIoTset"), ("ciciot2023", "CICIoT2023"), ("ton_iot", "ToN-IoT")]
MODELS = [("CNN_AE", "CNN-AE", "#1f77b4"), ("Dense_AE", "Dense-AE", "#ff7f0e"), ("LSTM_AE", "LSTM-AE", "#2ca02c"), ("SNN_AE", "SNN-AE", "#d62728")]


def style(ax):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, handles, labels, ncol, y=0.0):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, y),
               handlelength=2.2, columnspacing=1.2)


# Fig. 5
arms = pd.concat([pd.read_csv(M / "runs" / ds / "arm_metrics.csv") for ds, _ in DS])
ca = arms[arms.arm == "CONST_above"].groupby(["dataset", "model", "param"])[["fnr", "fpr"]].mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
for ax, (ds, lab) in zip(axes, DS):
    for m, ml, c in MODELS:
        s = ca[(ca.dataset == ds) & (ca.model == m)].sort_values("param")
        ax.plot(s.param, 100 * (1 - s.fnr), "-o", color=c, ms=2.5, lw=1.3)
        ax.plot(s.param, 100 * s.fpr, "--s", color=c, ms=2.5, lw=1.3)
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    ax.set_xticks([0, 0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(["0", "0.01", "0.1", "1", "10", "100"])
    ax.set_title(lab, fontsize=9.5); style(ax)
axes[0].set_ylabel("Rows flagged (%)")
fig.supxlabel("Out-of-support distance d given to every test row (t = 1 + d; symmetric-log axis)", fontsize=8.5, y=0.17)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=1.6) for _, _, c in MODELS] + [Line2D([], [], color="k", ls="-", marker="o", ms=3), Line2D([], [], color="k", ls="--", marker="s", ms=3)]
l = [ml for _, ml, _ in MODELS] + ["attacks detected", "normal rows flagged"]
legend(fig, h, l, ncol=6)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT / "fig5_class_independent_oos_print.png", dpi=300); plt.close(fig)

# Fig. 6
dist = pd.read_csv(M / "tables/mechanism_distance_binned_detection.csv")
order = ["0", "(0,0.1]", "(0.1,1]", "(1,10]", ">10"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, (ds, lab) in zip(axes, DS):
    for m, ml, c in MODELS:
        s = dist[(dist.dataset == ds) & (dist.model == m) & (dist.n > 0)]
        s = s.set_index("dist_bin").reindex([b for b in order if b in set(s.dist_bin)])
        ax.plot(range(len(s)), 100 * s.detection_rate, "-o", color=c, ms=3, lw=1.3, label=ml)
        ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=0, fontsize=7.5)
    ax.set_title(lab, fontsize=9.5); style(ax)
axes[0].set_ylabel("Attacks detected (%)")
fig.supxlabel("Out-of-support distance of the attack timestamp (pooled over the λ grid)", fontsize=8.5, y=0.14)
h, l = axes[1].get_legend_handles_labels()
legend(fig, h, l, ncol=4)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(OUT / "fig6_detection_by_oos_distance_print.png", dpi=300); plt.close(fig)

# Fig. 7
fpr = pd.read_csv(RPI / "tables/fpr_by_horizon.csv")
H = ["as_recorded", "+1h", "+1d", "+7d", "+30d", "+365d", "NOW"]
HL = ["as recorded", "+1 h", "+1 d", "+7 d", "+30 d", "+365 d", "device clock"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, (ds, lab) in zip(axes, DS):
    for m, ml, c in MODELS:
        a = fpr[(fpr.dataset == ds) & (fpr.cond == "A_q_t") & (fpr.model == m)][H].to_numpy().ravel()
        b = fpr[(fpr.dataset == ds) & (fpr.cond == "B_q") & (fpr.model == m)][H].to_numpy().ravel()
        ax.plot(range(7), a, "-o", color=c, ms=2.5, lw=1.3)
        ax.plot(range(7), b, ":", color=c, lw=1.4)
    ax.set_xticks(range(7)); ax.set_xticklabels(HL, fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_title(lab, fontsize=9.5); style(ax)
axes[0].set_ylabel("FPR on normal traffic (%)")
fig.supxlabel("Deployment horizon after the training capture (fixed threshold)", fontsize=8.5, y=0.13)
h = [Line2D([], [], color=c, lw=1.6) for _, _, c in MODELS] + [Line2D([], [], color="k", ls="-", marker="o", ms=3), Line2D([], [], color="k", ls=":")]
l = [ml for _, ml, _ in MODELS] + ["with timestamp (A)", "without timestamp (B)"]
legend(fig, h, l, ncol=6)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.tight_layout(rect=(0, 0.15, 1, 1))
fig.savefig(OUT / "fig7_rpi5_alert_flood_print.png", dpi=300); plt.close(fig)
print("saved")
