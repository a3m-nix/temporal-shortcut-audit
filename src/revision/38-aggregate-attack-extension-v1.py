#!/usr/bin/env python3
"""38 - Aggregate the attack-family extension (script 37) and score predictions X1-X4."""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

OUT = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "/temporal_audit_attack_extension_v1")
TAB = OUT / "tables"
TAB.mkdir(exist_ok=True)
MODELS = ["CNN_AE", "Dense_AE", "LSTM_AE", "SNN_AE"]
DSETS = ["ton_dos", "edge_nonflood"]


def nb(d, ratio=1 / 9):
    d = np.asarray(d, float); n = len(d); mu = d.mean(); sd = d.std(ddof=1)
    se = np.sqrt((1 / n + ratio) * sd ** 2)
    if se == 0:
        return mu, mu, mu, (1.0 if mu == 0 else 0.0)
    q = student_t.ppf(0.975, n - 1)
    return mu, mu - q * se, mu + q * se, float(2 * student_t.sf(abs(mu / se), n - 1))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); a = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); a[i] = min(1, run)
    return a


arms, fams, metas = [], [], []
for mf in (OUT / "runs").glob("*/*/seed_*/fold_*/metrics.json"):
    m = json.loads(mf.read_text())
    key = {k: m[k] for k in ("dataset", "model", "seed", "fold")}
    a = pd.read_csv(mf.parent / "arms.csv").assign(**key); arms.append(a)
    f = pd.read_csv(mf.parent / "families.csv").assign(**key); fams.append(f)
    metas.append(m)
arms, fams, metas = pd.concat(arms), pd.concat(fams), pd.DataFrame(metas)
assert len(metas) == 240, len(metas)
K = ["dataset", "model", "seed", "fold"]
get = lambda arm, par=None: (arms[(arms.arm == arm) & (np.isclose(arms.param, par) if par is not None else True)]).set_index(K)
L1, L0, CL = get("LAMBDA", 1.0), get("LAMBDA", 0.0), get("CLIP")

rows = []
for ds in DSETS:
    for mdl in MODELS:
        sel = lambda t: t.xs((ds, mdl), level=["dataset", "model"]).sort_index()
        l1, l0, cl = sel(L1), sel(L0), sel(CL)
        r = {"dataset": ds, "model": mdl, "fnr_l1": 100 * l1.fnr.mean(), "fnr_l0": 100 * l0.fnr.mean(), "fnr_clip": 100 * cl.fnr.mean(),
             "fpr_l1": 100 * l1.fpr.mean(), "fpr_l0": 100 * l0.fpr.mean(), "auc_l1": l1.auc.mean(), "auc_l0": l0.auc.mean()}
        for nm, a, b in [("l0_minus_l1", l0, l1), ("clip_minus_l0", cl, l0), ("clip_minus_l1", cl, l1)]:
            mu, lo, hi, p = nb((a.fnr - b.fnr).to_numpy())
            r.update({f"{nm}_pp": 100 * mu, f"{nm}_ci_lo": 100 * lo, f"{nm}_ci_hi": 100 * hi, f"{nm}_p": p})
        mm = metas[(metas.dataset == ds) & (metas.model == mdl)]
        r.update({"attack_oos_frac": mm.attack_oos_frac.mean(), "attack_dist_median": mm.attack_dist_median.median(),
                  "normal_oos_frac": mm.normal_oos_frac.mean(), "epochs_median": mm.epochs_used.median()})
        rows.append(r)
S = pd.DataFrame(rows)
for nm in ["l0_minus_l1", "clip_minus_l0", "clip_minus_l1"]:
    S[f"{nm}_p_holm8"] = holm(S[f"{nm}_p"])
S.to_csv(TAB / "attack_extension_summary.csv", index=False)

fam = fams[fams.arm.isin(["LAMBDA", "CLIP"])].copy()
fam["arm2"] = np.where(fam.arm == "CLIP", "clip", np.where(np.isclose(fam.param, 1.0), "l1", "l0"))
F = fam.groupby(["dataset", "family", "model", "arm2"]).fnr.mean().unstack("arm2") * 100
F["l0_minus_l1"] = F["l0"] - F["l1"]
F.reset_index().to_csv(TAB / "attack_extension_by_family.csv", index=False)

V = []
t = S[S.dataset == "ton_dos"]
n1 = int(((t.l0_minus_l1_pp > 20) & (t.l0_minus_l1_ci_lo > 0) & (t.l0_minus_l1_p_holm8 < 0.05)).sum())
V.append(("X1", n1 >= 3, f"ToN DoS: models with dFNR>20pp, CI>0, Holm<0.05: {n1}/4; dFNR={t.l0_minus_l1_pp.round(2).tolist()}"))
n2 = int((abs(t.fnr_clip - t.fnr_l0) < 5).sum())
V.append(("X2", n2 >= 3, f"ToN DoS: |clip-l0|<5pp: {n2}/4; diffs={(t.fnr_clip - t.fnr_l0).round(2).tolist()}"))
e = S[S.dataset == "edge_nonflood"]
n3 = int((((e.l0_minus_l1_ci_lo <= 0) & (e.l0_minus_l1_ci_hi >= 0)) | (abs(e.l0_minus_l1_pp) < 5)).sum())
V.append(("X3", n3 >= 3, f"Edge non-flood: CI incl 0 or |dFNR|<5pp: {n3}/4; dFNR={e.l0_minus_l1_pp.round(2).tolist()}"))
n4 = int((abs(e.fnr_clip - e.fnr_l1) < 1).sum())
V.append(("X4", n4 == 4, f"Edge non-flood: |clip-l1|<1pp: {n4}/4; diffs={(e.fnr_clip - e.fnr_l1).round(3).tolist()}"))
V = pd.DataFrame(V, columns=["prediction", "met", "detail"])
V.to_csv(TAB / "PREDICTION_VERDICTS.csv", index=False)
pd.set_option("display.width", 250)
print(S[["dataset", "model", "fnr_l1", "fnr_l0", "fnr_clip", "fpr_l1", "fpr_l0", "l0_minus_l1_pp", "l0_minus_l1_ci_lo", "l0_minus_l1_ci_hi",
         "l0_minus_l1_p_holm8", "attack_oos_frac", "attack_dist_median", "epochs_median"]].round(3).to_string())
print(F.round(2).to_string())
print(V.to_string())
