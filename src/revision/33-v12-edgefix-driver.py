#!/usr/bin/env python3
"""
33 - Driver for temporal_audit_temporal_upgrade_e2e_v1_2_edgefix .

Re-runs Experiment II for Edge-IIoTset with the ORIGINAL v1.2 code (script 20). The single change:
during Edge-IIoTset training, early stopping uses a fixed random 15% hold-out of fit_normal
(RandomState(280000 + seed)), and the model trains on the remaining 85% (= FITVAL in script 28).

  python 33-v12-edgefix-driver.py train-edge --conditions A_q_t B_q --models Dense_AE
  python 33-v12-edgefix-driver.py finish       # train (resume/aggregate) -> intervene -> analyze
"""
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
V4 = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
ROOT = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2_edgefix"
V12_HASH = "6921766ff0f11d825b5f0dc36fc915ad6873a627daa312b53fe11bb2ca41e45c"

mode = sys.argv[1]
extra = sys.argv[2:]
spec = importlib.util.spec_from_file_location("jk_base_v12", V4 / "20-end-to-end-audit-informed-v1.2.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

sys.argv = ["20", "--root", str(ROOT)]
args = base.parse_args()
root = args.root.resolve()
dirs = base.ensure_dirs(root)
protocol = base.static_protocol(args)
protocol["implementation_sha256"] = base.sha256_file(Path(base.__file__).resolve())
protocol["capture_mapping_parameters"] = {
    "capture_prefix_frames": int(args.capture_prefix_frames),
    "parent_scan_batch": int(args.parent_scan_batch),
    "min_offset_support": base.MIN_CAPTURE_OFFSET_SUPPORT,
    "min_offset_share": base.MIN_CAPTURE_OFFSET_SHARE,
    "min_offset_dominance": base.MIN_CAPTURE_OFFSET_DOMINANCE,
    "block_length_tolerance_frames": base.BLOCK_LENGTH_TOLERANCE_FRAMES,
    "fallback_parent_order_block_rows": int(args.fallback_block_rows),
}
protocol["analysis_parameters"] = {"bootstrap_iterations": int(args.bootstrap_iterations), "alpha": base.ALPHA}
protocol_hash = base.stable_json_hash(protocol)
if protocol_hash != V12_HASH:
    msg = f"protocol hash {protocol_hash} != v1.2 {V12_HASH}"
    if os.environ.get("TEMPORAL_ALLOW_PROTOCOL_HASH_CHANGE") != "1":
        raise SystemExit(msg + "; refusing to mix evidence (see REVISION_PUBLIC_SANITIZATION.txt)")
    print("WARNING:", msg, "(public checkout; allowed by TEMPORAL_ALLOW_PROTOCOL_HASH_CHANGE=1)", flush=True)

# ---- the single change: Edge-IIoTset early stopping on a fit-normal hold-out ----
_orig_tp = base.transform_partition
PATCH = {"active": False, "seed": None}


def patched_transform_partition(prepared, split, spec_, scaler):
    if PATCH["active"] and split in ("fit_normal", "earlystop_normal"):
        part, x = _orig_tp(prepared, "fit_normal", spec_, scaler)
        perm = np.random.RandomState(280000 + int(PATCH["seed"])).permutation(len(x))
        nval = int(round(0.15 * len(x)))
        idx = perm[:nval] if split == "earlystop_normal" else perm[nval:]
        return part.iloc[idx].reset_index(drop=True), x[idx]
    return _orig_tp(prepared, split, spec_, scaler)


base.transform_partition = patched_transform_partition
base.EDGEFIX_NOTE = "Edge-IIoTset early stopping on fit-normal 15% hold-out (script 33)"

if mode == "train-edge":
    conds = extra[extra.index("--conditions") + 1: extra.index("--models")]
    models = extra[extra.index("--models") + 1:]
    legacy = base.load_legacy_driver()
    q_features, specs = base.resolve_condition_specs(legacy)
    conditions = list(base.SAFE_CONDITION_NAMES) + [base.PRECISION_CONDITION_NAME]
    ds = "edge_iiotset"
    di = base.DATASET_ORDER.index(ds)
    prepared = base.load_prepared(dirs, ds)
    import joblib
    for cond in conds:
        ci = conditions.index(cond)
        scaler = joblib.load(base.scaler_dir(dirs, ds, cond) / "scaler.pkl")
        for m in models:
            mi = list(args.models).index(m)
            for seed in args.seeds:
                PATCH.update(active=True, seed=int(seed))
                met = base._run_one_neural(args, root, dirs, protocol_hash, legacy, ds, di, cond, ci, specs[cond],
                                           m, mi, int(seed), prepared, scaler, q_features)
                PATCH.update(active=False, seed=None)
                q = met["q_score_metrics"]
                print(f"[EDGEFIX] {cond} {m} s{seed} epochs={met['epochs_used']} FNR={q['fnr']:.4f} FPR={q['fpr']:.4f}", flush=True)
elif mode == "finish":
    legacy = base.load_legacy_driver()
    q_features, specs = base.resolve_condition_specs(legacy)
    base.train_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)  # all runs resume -> aggregate + mark
    base.intervention_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
    base.analyze_stage(args, root, dirs, protocol_hash, list(q_features))
    print("FINISH OK", flush=True)
else:
    raise SystemExit(mode)
