#!/usr/bin/env python3
"""39 - Redrawn Fig. 1 (evidence workflow, print-size fonts, revision blocks) and Fig. 3 (intervention-response
curves, symmetric-log lambda axis, one shared legend) for the revision. Data for Fig. 3: script 23 summary table."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
OUT = BASE / "temporal_audit_revision_figures_v1"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# ------------------------------------------------------------------ Fig. 3
d = pd.read_csv(BASE / "temporal_audit_posthoc_strengthening_v1/02_dose_response_support_score/tables/dose_response_summary.csv")
DS = [("edge_iiotset", "Edge-IIoTset"), ("ciciot2023", "CICIoT2023"), ("ton_iot", "ToN-IoT")]
MODELS = [("CNN_AE", "CNN-AE", "#1f77b4"), ("Dense_AE", "Dense-AE", "#ff7f0e"), ("LSTM_AE", "LSTM-AE", "#2ca02c"), ("SNN_AE", "SNN-AE", "#d62728")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (ds, lab) in zip(axes, DS):
    for m, ml, c in MODELS:
        s = d[(d.dataset == ds) & (d.model == m)].sort_values("lambda")
        ax.plot(s["lambda"], 100 * s["non_epoch_fnr_mean"], "-o", ms=3, lw=1.4, color=c, label=ml)
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.8)
    ax.set_xlim(-0.0002, 1.15)
    ax.set_xticks([0, 0.001, 0.01, 0.1, 1])
    ax.set_xticklabels(["0", "0.001", "0.01", "0.1", "1"])
    ax.set_title(lab, fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Non-time FNR (%)")
fig.supxlabel("λ (symmetric-log axis; linear between 0 and 0.001)", fontsize=9, y=0.13)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(OUT / "fig3_intervention_response_symlog.png", dpi=300)
fig.savefig(OUT / "fig3_intervention_response_symlog.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Fig. 1
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")
INK, GREY, BLUE, GREEN = "#2f3e4e", "#f2f2f2", "#dbe8f5", "#e3f1e3"

def box(x, y, w, h, text, face="white", ls="-", lw=1.4, bold=False, fs=8.2, edge=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=face, ec=edge, ls=ls, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", color=INK)

stages = [("1", "Provenance\naudit", ["Source-row\nlinkage", "Timestamp\ncontract", "Parent-table\norder"], ["-", "-", "-"]),
          ("2", "Features\nand scoring", ["q, t, b\nfeatures", "Four AEs", "Non-time\nscore"], ["-", "-", "-"]),
          ("3", "Experiments,\ninterventions", ["Exp I:\nλ = 0 vs 1", "Exp II:\nA / B / C / D", "Fixed-weight\ncontrols"], ["-", "-", "-"]),
          ("4", "Mechanism\nchecks", ["Support distance,\nλ response,\nsurrogate / noise", "Clipping,\nclass-independent\nOOS control", "Early-stopping\nrepair, further\nattack families"], ["--", ":", ":"]),
          ("5", "Security\nevidence", ["FNR / FPR,\nROC / PR-AUC", "Attack transitions,\nsupervised\nmodels", "Deployment\nreplay\n(Raspberry Pi 5)"], ["-", "--", ":"])]
x0, w, gap = 12.5, 15.0, 2.6
for i, (num, title, subs, styles) in enumerate(stages):
    x = x0 + i * (w + gap)
    ls = "--" if num == "4" else "-"
    box(x, 37, w, 10, title, ls=ls, bold=True, fs=8.2)
    ax.add_patch(plt.Circle((x + 1.6, 47), 1.5, color=INK if num != "4" else "white", ec=INK, lw=1.2, zorder=5))
    ax.text(x + 1.6, 47, num, ha="center", va="center", color="white" if num != "4" else INK, fontsize=8, weight="bold", zorder=6)
    ax.plot([x + w / 2, x + w / 2], [37, 34], color=INK, lw=1.2, ls=ls)
    for j, (s, st) in enumerate(zip(subs, styles)):
        y = 26.5 - j * 9.3
        face = BLUE if s.startswith("Exp I:") else (GREEN if st == ":" else GREY)
        box(x, y, w, 7, s, face=face, ls=st, lw=1.8 if s.startswith("Exp I:") else 1.0,
            edge="#2a5d8f" if s.startswith("Exp I:") else INK, bold=s.startswith("Exp I:"), fs=6.8)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + w + gap - 0.3, 42), xytext=(x + w + 0.3, 42), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
ax.text(5.6, 42, "Public\nIoT/IIoT\nbench-\nmarks", ha="center", va="center", fontsize=8.2, weight="bold", color=INK)
ax.annotate("", xy=(x0 - 0.3, 42), xytext=(10.0, 42), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
xe = x0 + 5 * (w + gap) - gap
ax.annotate("", xy=(xe + 2.3, 42), xytext=(xe + 0.3, 42), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
ax.text(105.6, 42, "Audit-\ninformed\nclaim", ha="center", va="center", fontsize=8.2, weight="bold", color=INK)
ax.plot([x0, x0, xe, xe], [50.5, 52, 52, 50.5], color="#6b7785", lw=1.2)
ax.text((x0 + xe) / 2, 54, "Temporal-shortcut audit", ha="center", va="center", fontsize=9.5, style="italic", color=INK)
ly = -2.6
for k, (lab, st, face) in enumerate([("protocol-defined", "-", GREY), ("post-hoc", "--", GREY), ("pre-specified (revision)", ":", GREEN)]):
    lx = 4 + k * 24
    ax.add_patch(FancyBboxPatch((lx, ly), 4.5, 3, boxstyle="round,pad=0.15,rounding_size=0.6", fc=face, ec=INK, ls=st, lw=1.2, clip_on=False))
    ax.text(lx + 5.8, ly + 1.5, lab, va="center", fontsize=8, color=INK)
ax.add_patch(FancyBboxPatch((78, ly), 4.5, 3, boxstyle="round,pad=0.15,rounding_size=0.6", fc=BLUE, ec="#2a5d8f", lw=1.6, clip_on=False))
ax.text(83.8, ly + 1.5, "primary evidence (360 runs)", va="center", fontsize=8, color=INK)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.08)
fig.savefig(OUT / "fig1_evidence_workflow_revised.png", dpi=300)
fig.savefig(OUT / "fig1_evidence_workflow_revised.pdf")
print("saved", OUT)
