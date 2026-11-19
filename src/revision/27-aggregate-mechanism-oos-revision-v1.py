#!/usr/bin/env python3
"""27 - Aggregate the OOS mechanism tests (script 26) and score the pre-specified predictions."""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

OUT = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "/0-temporal-feature-audit-v4-allminmax-3seed10fold"
           "/controlled_temporal_intervention/mechanism_oos_revision_v1")
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
DATASETS = ["edge_iiotset", "ciciot2023", "ton_iot"]
DS_LABEL = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023", "ton_iot": "ToN-IoT"}
MODELS = ["CNN_AE", "Dense_AE", "LSTM_AE", "SNN_AE"]
NON_SNN = ["CNN_AE", "Dense_AE", "LSTM_AE"]
RATIO = 1.0 / 9.0
KEY = ["dataset", "model", "seed", "fold"]


def load(name):
    return pd.concat([pd.read_csv(OUT / "runs" / ds / name) for ds in DATASETS], ignore_index=True)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def nb(diff):
    diff = np.asarray(diff, float); n = len(diff); mu = diff.mean(); sd = diff.std(ddof=1)
    se = np.sqrt((1 / n + RATIO) * sd ** 2)
    if se == 0:
        return mu, mu, mu, (0.0 if mu != 0 else 1.0)
    tstat = mu / se; q = student_t.ppf(0.975, n - 1)
    return mu, mu - q * se, mu + q * se, float(2 * student_t.sf(abs(tstat), n - 1))


arms = load("arm_metrics.csv")
acts = load("activations.csv")
dist = load("distance_bins.csv")
sup = load("support.csv")
assert len(sup) == 360, len(sup)
arms["param"] = arms["param"].astype(float)


def get(arm, param=None):
    d = arms[arms.arm == arm]
    if param is not None:
        d = d[np.isclose(d.param, param)]
    return d.set_index(KEY)[["fnr", "fpr", "auc"]]


L1, L0, CLIP, PERM = get("LAMBDA", 1.0), get("LAMBDA", 0.0), get("CLIP"), get("PERM")

# ---------- summary table ----------
rows = []
for (ds, m), _ in sup.groupby(["dataset", "model"]):
    r = {"dataset": ds, "model": m}
    for nm, tb in [("L1", L1), ("L0", L0), ("CLIP", CLIP), ("PERM", PERM)]:
        x = tb.xs((ds, m), level=["dataset", "model"])
        for k in ["fnr", "fpr", "auc"]:
            r[f"{nm}_{k}"] = x[k].mean()
    for d in [1.0, 2.0, 10.0]:
        x = get("CONST_above", d).xs((ds, m), level=["dataset", "model"])
        r[f"CONST{d:g}_fnr"], r[f"CONST{d:g}_fpr"] = x.fnr.mean(), x.fpr.mean()
    for d in [0.5, 1.0, 2.0]:
        x = get("FUT", d).xs((ds, m), level=["dataset", "model"])
        r[f"FUT{d:g}_fpr"] = x.fpr.mean()
    s = sup[(sup.dataset == ds) & (sup.model == m)]
    r["attack_oos_frac"], r["attack_dist_median"] = s.attack_oos_frac.mean(), s.attack_dist_median.median()
    rows.append(r)
summary = pd.DataFrame(rows)
summary.to_csv(TAB / "mechanism_summary.csv", index=False)

# ---------- NB tests: CLIP vs L1, CLIP vs L0 ----------
tests = []
for contrast, A, B in [("CLIP_minus_L1", CLIP, L1), ("CLIP_minus_L0", CLIP, L0), ("PERM_minus_L1_fpr", PERM, L1)]:
    metric = "fpr" if contrast.endswith("_fpr") else "fnr"
    for ds in DATASETS:
        for m in MODELS:
            a = A.xs((ds, m), level=["dataset", "model"])[metric]
            b = B.xs((ds, m), level=["dataset", "model"])[metric].reindex(a.index)
            mu, lo, hi, p = nb((a - b).to_numpy())
            tests.append({"contrast": contrast, "metric": metric, "dataset": ds, "model": m,
                          "mean_diff_pp": 100 * mu, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi, "p_raw": p})
tests = pd.DataFrame(tests)
tests["p_holm12"] = tests.groupby("contrast")["p_raw"].transform(holm)
tests.to_csv(TAB / "mechanism_nb_tests.csv", index=False)

# ---------- activations ----------
a = acts[acts["class"] == "attack"].pivot_table(index=KEY, columns="state", values=["l2_mean", "l2_median"])
a["ratio_mean"] = a[("l2_mean", "L1")] / a[("l2_mean", "L0")]
a["ratio_median"] = a[("l2_median", "L1")] / a[("l2_median", "L0")]
act_sum = a[["ratio_mean", "ratio_median"]].droplevel(1, axis=1).groupby(["dataset", "model"]).median().reset_index()
nrm = acts[(acts["class"] == "normal") & (acts.state == "L1")].groupby(["dataset", "model"])["l2_median"].median()
act_sum = act_sum.merge(nrm.rename("normal_L1_l2_median").reset_index(), on=["dataset", "model"])
att = acts[(acts["class"] == "attack")].pivot_table(index=["dataset", "model"], columns="state", values="l2_median", aggfunc="median")
act_sum = act_sum.merge(att.rename(columns={"L1": "attack_L1_l2_median", "L0": "attack_L0_l2_median"}).reset_index(), on=["dataset", "model"])
act_sum.to_csv(TAB / "mechanism_activation_summary.csv", index=False)

# ---------- distance bins ----------
dsum = dist.groupby(["dataset", "model", "dist_bin"])[["n", "detected"]].sum().reset_index()
dsum["detection_rate"] = dsum.detected / dsum.n.replace(0, np.nan)
order = ["0", "(0,0.1]", "(0.1,1]", "(1,10]", ">10"]
dsum["bin_order"] = dsum.dist_bin.map({b: i for i, b in enumerate(order)})
dsum = dsum.sort_values(["dataset", "model", "bin_order"])
dsum.to_csv(TAB / "mechanism_distance_binned_detection.csv", index=False)

# ---------- prediction verdicts ----------
V = []
def verdict(pid, scope, met, detail):
    V.append({"prediction": pid, "scope": scope, "met": bool(met), "detail": detail})

for ds in ["ciciot2023", "ton_iot"]:
    for m in MODELS:
        s = summary[(summary.dataset == ds) & (summary.model == m)].iloc[0]
        t = tests[(tests.contrast == "CLIP_minus_L1") & (tests.dataset == ds) & (tests.model == m)].iloc[0]
        gap = 100 * abs(s.CLIP_fnr - s.L0_fnr)
        verdict("P1", f"{ds}/{m}", gap < 5 and t.ci_low_pp > 0,
                f"|CLIP-L0|={gap:.2f}pp; CLIP-L1={t.mean_diff_pp:.2f}pp CI[{t.ci_low_pp:.2f},{t.ci_high_pp:.2f}]")
for m in MODELS:
    s = summary[(summary.dataset == "edge_iiotset") & (summary.model == m)].iloc[0]
    g = 100 * abs(s.CLIP_fnr - s.L1_fnr)
    verdict("P1", f"edge_iiotset/{m}", g < 1, f"|CLIP-L1|={g:.2f}pp")

ca = arms[arms.arm == "CONST_above"].groupby(["dataset", "model", "param"])["fpr"].mean().reset_index()
for ds in DATASETS:
    mono_all = True
    for m in MODELS:
        f = ca[(ca.dataset == ds) & (ca.model == m)].sort_values("param").fpr.to_numpy()
        mono_all &= bool(np.all(np.diff(f) >= -0.01))
    n50 = sum(ca[(ca.dataset == ds) & (ca.model == m) & np.isclose(ca.param, 10)].fpr.iloc[0] >= 0.5 for m in NON_SNN)
    verdict("P2", ds, mono_all and n50 >= 2, f"FPR non-decreasing in d (tol 1pp) for all models={mono_all}; non-SNN with FPR>=50% at d=10: {n50}/3")

for ds in ["ciciot2023", "ton_iot"]:
    for m in NON_SNN:
        s = summary[(summary.dataset == ds) & (summary.model == m)].iloc[0]
        verdict("P3", f"{ds}/{m}", s.PERM_fpr > 0.25, f"PERM FPR={100*s.PERM_fpr:.2f}% vs L1 {100*s.L1_fpr:.2f}%")

cnt = 0
for _, s in summary.iterrows():
    cnt += int(s.FUT1_fpr > 2 * s.L1_fpr)
verdict("P4", "12 cells", cnt >= 7, f"FUT(d=1) FPR > 2x L1 FPR in {cnt}/12 cells")

for _, s in act_sum.iterrows():
    if s.model == "SNN_AE":
        continue
    if s.dataset == "edge_iiotset":
        verdict("P5", f"{s.dataset}/{s.model}", 0.8 <= s.ratio_mean <= 1.25,
                f"pre-specified mean-L2 ratio={s.ratio_mean:.2f} (median-L2 ratio={s.ratio_median:.2f}, descriptive)")
    else:
        verdict("P5", f"{s.dataset}/{s.model}", s.ratio_mean > 3,
                f"pre-specified mean-L2 ratio={s.ratio_mean:.2f} (median-L2 ratio={s.ratio_median:.2f}, descriptive)")

for ds in ["ciciot2023", "ton_iot"]:
    for m in NON_SNN:
        d = dsum[(dsum.dataset == ds) & (dsum.model == m) & (dsum.n > 0)].sort_values("bin_order")
        r = d.detection_rate.to_numpy()
        verdict("P6", f"{ds}/{m}", bool(np.all(np.diff(r) >= -0.01)),
                "; ".join(f"{b}:{x:.3f}(n={n})" for b, x, n in zip(d.dist_bin, r, d.n)))
V = pd.DataFrame(V)
V.to_csv(TAB / "PREDICTION_VERDICTS.csv", index=False)

# ---------- figures ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, ds in zip(axes, DATASETS):
    for m, c in zip(MODELS, ["C0", "C1", "C2", "C3"]):
        x = arms[(arms.arm == "CONST_above") & (arms.dataset == ds) & (arms.model == m)].groupby("param")[["fnr", "fpr"]].mean()
        xd = x.index.to_numpy() + 0.003
        ax.plot(xd, 100 * (1 - x.fnr), "-o", color=c, ms=3, label=f"{m} attack detected")
        ax.plot(xd, 100 * x.fpr, "--s", color=c, ms=3, label=f"{m} normal flagged")
    ax.set_xscale("log"); ax.set_title(DS_LABEL[ds])
    ax.grid(alpha=.3)
axes[0].set_ylabel("% of rows flagged"); fig.supxlabel("OOS distance d assigned to every test row, t = 1 + d (log scale; d = 0 plotted at 0.003)", fontsize=9, y=0.10)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(FIG / "fig_class_independent_oos_response.png", dpi=300)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, ds in zip(axes, DATASETS):
    for m in MODELS:
        d = dsum[(dsum.dataset == ds) & (dsum.model == m) & (dsum.n > 0)]
        ax.plot(d.dist_bin, 100 * d.detection_rate, "-o", ms=3, label=m)
    ax.set_title(DS_LABEL[ds]); ax.grid(alpha=.3)
axes[0].set_ylabel("attack detection rate (%)"); axes[0].legend(fontsize=7); fig.supxlabel("attack OOS distance at the applied λ (pooled over the λ grid)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig_detection_by_oos_distance.png", dpi=300)

print(summary.assign(**{c: 100 * summary[c] for c in summary.columns if c.endswith(("_fnr", "_fpr", "_frac"))}).round(2).to_string())
print(tests.round(3).to_string())
print(act_sum.round(2).to_string())
print(V.to_string())
json.dump({"n_runs": int(len(sup)), "predictions_met": int(V.met.sum()), "predictions_total": int(len(V))},
          open(OUT / "AGGREGATE_DONE.json", "w"), indent=2)
