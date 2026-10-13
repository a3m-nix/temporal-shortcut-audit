#!/usr/bin/env python3
"""
20-end-to-end-audit-informed.py

One-file, resume-safe empirical pipeline for the TEMPORAL_AUDIT temporal-shortcut upgrade.

Scientific role
---------------
This script operationalizes the remaining *experimental* tasks after the frozen
legacy v3 evidence and the completed provenance/integrity audits. It does not
rewrite the article, make novelty claims, or replace historical evidence.

The pipeline is deliberately fail-closed on *claims*, not on experiment
completion. It will not call a split "capture-disjoint" unless every evaluated
materialized row has a certified raw-capture block mapping. If raw-capture
coverage is incomplete, the pipeline records that limitation and switches to a
pre-specified fixed parent-order block protocol. Such blocks are never called
captures or sessions. This prevents incomplete provenance from being silently
upgraded into a stronger independence claim while still allowing the remaining
experiment to be completed.

Stages
------
  preflight      : verify frozen code/data/audit inputs, archive dependencies
  capture_map    : reconstruct raw-capture -> parent-row blocks from multi-field
                   packet-prefix anchors; validate block lengths and sample mapping
  split          : create fit-normal / early-stop-normal / calibration-normal /
                   test manifest using whole pure-normal captures where feasible;
                   otherwise use explicitly named fixed parent-order block fallback
  behavior       : build causal, translation-invariant flow-local temporal context
                   from ordered parent traffic, reset at analysis units and within-flow resets
  pilot          : technical QA only, development-normal data only, no attack-test
  baselines      : timestamp-range and PCA reconstruction baselines
  train          : six main/bridge conditions x four legacy reconstruction
                   architectures x three model seeds, plus epoch precision sensitivity
  intervene      : fixed-weight epoch neutralization plus predefined raw-time shifts,
                   with fixed-threshold and normal-calibration-only recalibration tracks
  analyze        : security impact, acquisition-group effects, statistics, residuals,
                   ensemble diagnostics, claim-evidence/RQ matrices, figures
  package        : checksums, artifact inventory, lightweight review ZIP
  all            : run every stage in order

Primary new representation design
---------------------------------
Let q = legacy non-temporal packet features after removing BOTH
        frame.time_epoch and the legacy table-predecessor inter_arrival_time.
Let r = the legacy inter_arrival_time (retained only as a bridge/sensitivity cue).
Let t = precision-safe absolute frame.time_epoch.
Let b = newly reconstructed causal flow-local timing/traffic context.

  L34 = q + r + t  : bridge to historical 34F representation
  L33 = q + r      : bridge to historical 33F representation
  A   = q + t      : absolute time without legacy global-predecessor IAT
  B   = q          : packet-only reference
  C   = q + b + t  : corrected behavioral context plus absolute time
  D   = q + b      : corrected behavioral context without absolute time

All main neural conditions use the same *q-only* anomaly score for the primary
operational comparison. Thus the scoring denominator is common across A-D and
bridge conditions; context/timestamp inputs can influence reconstruction but do
not directly enter the primary score. Full-input reconstruction score is retained
as a secondary diagnostic.

Important interpretation boundaries
-----------------------------------
- Existing final v3 results remain historical final evidence; this script creates
  a new evidence block with explicit provenance labels.
- Model seeds are not independent environments.
- LSTM uses one packet/time-step and CNN convolves over the feature axis because
  the validated legacy builders are reused; this is cross-architecture packet
  reconstruction, not cross-packet sequence modelling.
- Raw-PCAP mapping here is a header/feature block mapping, not payload identity.
- Capture-file disjointness is not called session independence.
- The materialized audit samples are not full deployment streams.
- Literature novelty review and manuscript rewriting are intentionally outside
  this executable pipeline and remain human/chat review gates.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Set framework environment BEFORE importing TensorFlow through the legacy driver.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy import stats
from scipy.stats import wilcoxon
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

# TensorFlow is imported only after environment variables are set.
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# =============================================================================
# CONSTANTS / PATH CONTRACT
# =============================================================================

SCRIPT_VERSION = "1.2.0"
EVIDENCE_CLASS = "NEW_PROSPECTIVE_AFTER_AUDIT_VALIDATION"
BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
LEGACY_ROOT = BASE / "0-temporal-feature-audit-v4-allminmax-3seed10fold"
LEGACY_FINAL_SCRIPT = LEGACY_ROOT / "04-confirmatory-allminmax-3seed-10fold.py"
LEGACY_BASE_SCRIPT = BASE / "0-temporal-feature-audit" / "03-model-epoch-audit.py"
SOURCE_V2 = BASE / "0-temporal-feature-audit-v2-3datasets"
DEFAULT_ROOT = BASE / "temporal_audit_temporal_upgrade_e2e_v1_2"
V11_ROOT = BASE / "temporal_audit_temporal_upgrade_e2e_v1"
PROVENANCE_ROOT = BASE / "temporal_audit_provenance_audits"
FLOW_AUDIT_ROOT = BASE / "temporal_audit_flow_boundary_audits_v2"

EXPECTED_LEGACY_HASHES = {
    "04-confirmatory-allminmax-3seed-10fold.py": "71aa55026845bbe40d112c7dc0b36b3a4bc00d8f76860389245c25080d6fb21f",
    "03-model-epoch-audit.py": "cac1a5cca1a91190967248e592a113a88b3fa71bc8dc5851d841e900005a25a2",
}

DATASET_ORDER = ["edge_iiotset", "ciciot2023", "ton_iot"]
DATASETS = {
    "edge_iiotset": {
        "sample": SOURCE_V2 / "materialized_samples" / "edge_iiotset_audit_sample.parquet",
        "label": "label",
        "normal": "Normal",
    },
    "ciciot2023": {
        "sample": SOURCE_V2 / "materialized_samples" / "ciciot2023_audit_sample.parquet",
        "label": "label",
        "normal": "Benign",
    },
    "ton_iot": {
        "sample": SOURCE_V2 / "materialized_samples" / "ton_iot_audit_sample.parquet",
        "label": "label",
        "normal": "Normal",
    },
}

MODELS = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
MODEL_SEEDS = [0, 1, 2]
LEARNING_RATE = 5e-4
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
OPERATING_PERCENTILE = 99.0
BOOTSTRAP_ITERATIONS = 20000
ALPHA = 0.05

# Capture mapping.
TSHARK = Path("/usr/bin/tshark")
CAPINFOS = Path("/usr/bin/capinfos")
CAPTURE_PREFIX_FRAMES = 512
PARENT_SCAN_BATCH = 250_000
MIN_CAPTURE_OFFSET_SUPPORT = 16
MIN_CAPTURE_OFFSET_SHARE = 0.03
MIN_CAPTURE_OFFSET_DOMINANCE = 1.50
BLOCK_LENGTH_TOLERANCE_FRAMES = 4

# Split / behavior.
EMBARGO_ROWS = 5_000
PARENT_ORDER_BLOCK_ROWS = 100_000
NORMAL_FRACTIONS = {"fit_normal": 0.60, "earlystop_normal": 0.15, "calibration_normal": 0.10, "test_normal": 0.15}
BEHAVIOR_WINDOW = 8
RATE_DENOM_FLOOR_SECONDS = 1e-6
MIN_FIT_NORMAL = 500
MIN_EARLY_NORMAL = 100
MIN_CAL_NORMAL = 100
MIN_TEST_NORMAL = 100
MIN_TEST_ATTACK = 100

# Fixed intervention rules, selected without attack-test outcomes.
SHIFT_SCHEMES = {
    # Direct endpoint: erase absolute-time position at inference while preserving q/r/b.
    "neutralize_to_fit_median": {"kind": "neutralize_fit_median"},
    # Translation tests preserve within-capture relative chronology in the raw epoch channel.
    "global_plus_one_fit_span": {"kind": "global", "multiplier": 1.0},
    "analysis_group_hash_signed_half_fit_span": {"kind": "hash_signed", "multiplier": 0.5},
}

# Main safe conditions. q/b/t are resolved after importing the legacy feature contract.
SAFE_CONDITION_NAMES = ["L34_q_r_t", "L33_q_r", "A_q_t", "B_q", "C_q_b_t", "D_q_b"]
PRECISION_CONDITION_NAME = "A_q_t_epoch_float32_precast_sensitivity"

# For every pair (left, right), the planned primary effect is
# delta_FNR = FNR(right) - FNR(left). Positive values therefore mean that
# removing the component represented by the left condition increases missed attacks.
PRIMARY_CONTRASTS = {
    "B_minus_A_timestamp_removal_penalty": ("A_q_t", "B_q"),
    "D_minus_C_timestamp_removal_penalty_with_behavior": ("C_q_b_t", "D_q_b"),
    "B_minus_D_behavior_added_value": ("D_q_b", "B_q"),
}
BRIDGE_CONTRASTS = {
    "L33_minus_L34_legacy_bridge": ("L34_q_r_t", "L33_q_r"),
    "B_minus_L33_legacy_iat_bridge": ("L33_q_r", "B_q"),
    "A_float32_minus_A_safe_precision_penalty": ("A_q_t", PRECISION_CONDITION_NAME),
}

# New behavior features are causal within a capture/split unit and reset at unit start.
BEHAVIOR_FEATURES = [
    "b_prev_iat_log1p",
    "b_mean_iat8_log1p",
    "b_std_iat8_log1p",
    "b_packet_rate8_log1p",
    "b_byte_rate8_log1p",
    "b_mean_len8_log1p",
    "b_cv_iat8_log1p",
]

DISCOVERY_FIELDS = [
    "frame.number",
    "frame.time_epoch",
    "frame.len",
    "ip.proto",
    "ip.id",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "icmp.type",
    "icmp.code",
]
PARENT_DISCOVERY_FIELDS = [x for x in DISCOVERY_FIELDS if x != "frame.number"]

STAGE_ORDER = [
    "preflight",
    "capture_map",
    "split",
    "behavior",
    "pilot",
    "baselines",
    "train",
    "intervene",
    "analyze",
    "package",
]


# =============================================================================
# GENERIC UTILITIES
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def json_safe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    return x


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(raw)


def as_bool(value: Any) -> bool:
    """Parse booleans from CSV/JSON values without bool("False") surprises."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return False
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "nan", "none", "null"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def log(root: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    p = root / "logs" / "pipeline.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(args: list[str], timeout: int | None = None, check: bool = False) -> subprocess.CompletedProcess:
    cp = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if check and cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {args}\nSTDERR:\n{cp.stderr[-4000:]}")
    return cp


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "metadata": root / "metadata",
        "capture": root / "capture_mapping",
        "splits": root / "splits",
        "behavior": root / "behavior",
        "prepared": root / "prepared_data",
        "pilot": root / "pilot_technical_only",
        "baselines": root / "baselines",
        "runs": root / "runs",
        "interventions": root / "interventions",
        "tables": root / "tables",
        "figures": root / "figures",
        "review": root / "review_package",
        "logs": root / "logs",
        "source_archive": root / "metadata" / "source_archive",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def archive_partial_dir(path: Path) -> None:
    if not path.exists():
        return
    if (path / "RUN_COMPLETE.json").exists():
        return
    dst = path.with_name(path.name + f"__INCOMPLETE_{utc_tag()}")
    path.rename(dst)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def model_seed(dataset_idx: int, condition_idx: int, model_idx: int, seed: int) -> int:
    return 2_000_000 + dataset_idx * 100_000 + condition_idx * 10_000 + model_idx * 1_000 + seed * 17 + 101


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return float("nan")


def metric_dict(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    pred = (score > threshold).astype(np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {
        "n": int(len(y)),
        "normal_n": int(np.sum(y == 0)),
        "attack_n": int(np.sum(y == 1)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": float(recall),
        "fnr": float(fnr),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": safe_auc(y, score),
        "pr_auc": safe_ap(y, score),
    }


def percentile_threshold(scores: np.ndarray, percentile: float = OPERATING_PERCENTILE) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise RuntimeError("Calibration score is empty or non-finite")
    # Conservative empirical quantile. 'higher' avoids interpolating below an observed score.
    return float(np.percentile(scores, percentile, method="higher"))


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    m = len(p)
    out = np.full(m, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out
    pv = p[valid]
    order_local = np.argsort(pv)
    running = 0.0
    adj_local = np.empty(len(pv), dtype=float)
    for rank, loc in enumerate(order_local):
        raw = (len(pv) - rank) * pv[loc]
        running = max(running, raw)
        adj_local[loc] = min(1.0, running)
    out[valid] = adj_local
    return out


def bootstrap_mean_ci(values: np.ndarray, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = 73001) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    # Small acquisition-group n, so vectorized bootstrap is inexpensive.
    idx = rng.integers(0, len(x), size=(iterations, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_safe(diff: np.ndarray) -> float:
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan")
    if np.allclose(x, 0):
        return 1.0
    try:
        return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except Exception:
        return float("nan")


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB.
    return float(r) / 1024.0


def stage_marker(dirs: dict[str, Path], stage: str) -> Path:
    return dirs["metadata"] / "stages" / f"{stage}.json"


def stage_is_complete(dirs: dict[str, Path], stage: str, protocol_hash: str) -> bool:
    p = stage_marker(dirs, stage)
    if not p.exists():
        return False
    try:
        obj = load_json(p)
        return obj.get("status") == "COMPLETED" and obj.get("protocol_hash") == protocol_hash
    except Exception:
        return False


def mark_stage(dirs: dict[str, Path], stage: str, protocol_hash: str, extra: dict[str, Any] | None = None) -> None:
    p = stage_marker(dirs, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "status": "COMPLETED", "completed_utc": utc_now(), "protocol_hash": protocol_hash}
    if extra:
        payload.update(extra)
    save_json(p, payload)


def latest_completed_run(root: Path, required_file: str = "RUN_STATUS.json") -> Path:
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = sorted([p for p in root.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        status = p / required_file
        if not status.exists():
            continue
        try:
            obj = load_json(status)
            if obj.get("status") == "COMPLETED":
                return p
        except Exception:
            pass
    raise RuntimeError(f"No completed run found under {root}")


def find_table(root: Path, name: str) -> Path:
    p = root / name
    if p.exists():
        return p
    hits = list(root.rglob(name))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    raise RuntimeError(f"Multiple {name} under {root}; use exact run directory")


# =============================================================================
# PARQUET RANGE / SPARSE READERS
# =============================================================================

def parquet_num_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_schema_names(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_parquet_range(path: Path, start: int, end: int, columns: list[str]) -> pd.DataFrame:
    """Read inclusive parent row range [start, end] without loading the whole parquet."""
    if end < start:
        return pd.DataFrame(columns=["__parent_pos"] + columns)
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in available]
    pieces: list[pd.DataFrame] = []
    rg_start = 0
    for rg in range(pf.num_row_groups):
        n = int(pf.metadata.row_group(rg).num_rows)
        rg_end = rg_start + n - 1
        if rg_end < start:
            rg_start += n
            continue
        if rg_start > end:
            break
        lo = max(start, rg_start)
        hi = min(end, rg_end)
        tbl = pf.read_row_group(rg, columns=cols)
        local_lo = lo - rg_start
        local_n = hi - lo + 1
        sub = tbl.slice(local_lo, local_n).to_pandas()
        sub.insert(0, "__parent_pos", np.arange(lo, hi + 1, dtype=np.int64))
        pieces.append(sub)
        rg_start += n
    if not pieces:
        return pd.DataFrame(columns=["__parent_pos"] + cols)
    return pd.concat(pieces, ignore_index=True)


def sparse_parent_take(path: Path, positions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    positions = np.asarray(sorted(set(int(x) for x in positions)), dtype=np.int64)
    if len(positions) == 0:
        return pd.DataFrame()
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in available]
    pieces = []
    rg_start = 0
    for rg in range(pf.num_row_groups):
        n = int(pf.metadata.row_group(rg).num_rows)
        rg_end_excl = rg_start + n
        wanted = positions[(positions >= rg_start) & (positions < rg_end_excl)]
        if len(wanted):
            tbl = pf.read_row_group(rg, columns=cols)
            local = wanted - rg_start
            sub = tbl.take(pa.array(local)).to_pandas()
            sub.insert(0, "__parent_pos", wanted)
            pieces.append(sub)
        rg_start = rg_end_excl
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# =============================================================================
# ARGUMENTS / PROTOCOL
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end TEMPORAL_AUDIT temporal-shortcut upgrade experiment")
    ap.add_argument("--stage", choices=["all"] + STAGE_ORDER, default="all")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--prior-provenance", type=Path, default=None,
                    help="Exact completed Script-17 run. Default: newest completed run.")
    ap.add_argument("--prior-flow-audit", type=Path, default=None,
                    help="Exact completed Script-18-v2 run. Default: newest completed run.")
    ap.add_argument("--capture-prefix-frames", type=int, default=CAPTURE_PREFIX_FRAMES)
    ap.add_argument("--parent-scan-batch", type=int, default=PARENT_SCAN_BATCH)
    ap.add_argument("--embargo-rows", type=int, default=EMBARGO_ROWS)
    ap.add_argument("--fallback-block-rows", type=int, default=PARENT_ORDER_BLOCK_ROWS,
                    help="Fixed parent-row block size when complete raw-capture mapping is unavailable.")
    ap.add_argument("--reuse-capture-root", type=Path, default=V11_ROOT,
                    help="Optional prior partial v1.1 root whose per-dataset capture evidence may be reused after parameter checks.")
    ap.add_argument("--models", nargs="*", choices=MODELS, default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    ap.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    ap.add_argument("--skip-precision-sensitivity", action="store_true")
    ap.add_argument("--skip-interventions", action="store_true")
    ap.add_argument("--allow-code-hash-change", action="store_true",
                    help="Not recommended. Record and continue if frozen legacy dependency hash differs.")
    ap.add_argument("--hash-large-inputs", action="store_true",
                    help="Hash full parent/PCAP files; can be slow on very large sources.")
    return ap.parse_args()


def static_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "legacy_evidence_policy": {
            "v3_controlled_intervention": "historical final experimental evidence; not overwritten",
            "legacy_720_pair_experiment": "historical primary/frozen provenance label; retained",
            "development_probes_08_09": "exploratory; excluded from new final evidence",
            "new_pipeline": "separate prospective-after-audit evidence block",
        },
        "datasets": DATASET_ORDER,
        "capture_evidence_reuse_root": None if args.reuse_capture_root is None else str(args.reuse_capture_root),
        "models": list(args.models),
        "model_seeds": list(args.seeds),
        "split": {
            "preferred": "capture-file-disjoint pure-normal development partitions plus attack/mixed captures held out",
            "fallback": "fixed parent-order blocks classified from parent labels; only pure-normal blocks may enter fit/early-stop/calibration; attack/mixed blocks are test; explicitly not capture- or session-disjoint",
            "partitions": ["fit_normal", "earlystop_normal", "calibration_normal", "test"],
            "normal_fractions_if_blocked": NORMAL_FRACTIONS,
            "fallback_parent_order_block_rows": int(args.fallback_block_rows),
            "embargo_rows": int(args.embargo_rows),
            "old_sample_exposure": "all materialized sample rows were previously available in historical experiments; new test means held out from this new fit/tuning/calibration only",
        },
        "representation": {
            "q": "legacy FINAL_33 excluding inter_arrival_time; no frame.time_epoch",
            "r": "legacy inter_arrival_time (global-table predecessor delta per audits; bridge only)",
            "t": "frame.time_epoch scaled in float64 before float32 model input",
            "b": BEHAVIOR_FEATURES,
            "conditions": SAFE_CONDITION_NAMES,
            "precision_sensitivity": None if args.skip_precision_sensitivity else PRECISION_CONDITION_NAME,
            "primary_score": "mean squared reconstruction error over common q components only",
            "secondary_score": "mean squared reconstruction error over all condition input/output components",
        },
        "behavior": {
            "group": "analysis_unit + flow_id; analysis_unit is certified capture when complete, otherwise fixed parent-order block",
            "state_reset": "at every analysis unit and on within-flow packet_index non-increase or timestamp backstep",
            "window_interarrival_observations": BEHAVIOR_WINDOW,
            "chunk_state_prior_packets": BEHAVIOR_WINDOW,
            "causal": True,
            "translation_invariant": True,
            "no_time_since_capture_start": True,
            "capture_id_is_model_feature": False,
        },
        "training": {
            "normal_only": True,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "early_stopping_data": "earlystop_normal only",
            "threshold_data": "calibration_normal only",
            "operating_percentile": OPERATING_PERCENTILE,
            "scaler_fit": "fit_normal only",
            "scaler": "MinMaxScaler",
        },
        "intervention": {
            "enabled": not args.skip_interventions,
            "schemes": SHIFT_SCHEMES,
            "direct_neutralization": "set only raw frame.time_epoch to fit-normal median; q/r/b remain fixed",
            "translation_controls": "global or analysis-group-hash-signed raw epoch offsets; preserve q/r/b",
            "track1": "fixed weights + fixed scaler + fixed original calibration threshold",
            "track2": "fixed weights + fixed scaler + threshold recalibrated on shifted normal-calibration only",
            "score": "q-only common-component score",
        },
        "baselines": [
            "always-normal sanity baseline",
            "timestamp distance to fit-normal interval",
            "PCA reconstruction on each safe condition with q-only score",
        ],
        "statistics": {
            "primary_group": "analysis unit: certified raw capture only when complete mapping exists; otherwise fixed parent-order block",
            "model_seed_handling": "average analysis-unit FNR across model seeds before acquisition-group inference",
            "effect_ci": "paired analysis-unit bootstrap mean difference",
            "primary_p_value": "paired analysis-unit sign-flip randomization test",
            "secondary_test": "Wilcoxon signed-rank only when >=5 acquisition groups",
            "multiplicity": "Holm separately within each planned contrast family across dataset-model comparisons",
            "fpr": "measured held-out FPR; no assumption that P99 guarantees 1%",
        },
        "claim_boundary": [
            "no deployment generalization",
            "no universal timestamp claim",
            "no session-independence claim from capture files alone",
            "no causal internal-mechanism proof",
            "materialized sample evaluation is not full-stream deployment evaluation",
        ],
        "non_executable_gates": {
            "T02_literature_novelty": "requires primary-source literature review outside this script",
            "T14_manuscript_rewrite": "requires explicit user confirmation after evidence review",
            "T16_promotor_submission": "human decision/submission task",
        },
    }

# =============================================================================
# LEGACY MODEL CONTRACT / PREFLIGHT
# =============================================================================

def load_legacy_driver():
    if not LEGACY_FINAL_SCRIPT.exists():
        raise FileNotFoundError(LEGACY_FINAL_SCRIPT)
    spec = importlib.util.spec_from_file_location("temporal_audit_legacy_final", LEGACY_FINAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def resolve_condition_specs(legacy) -> tuple[list[str], dict[str, dict[str, Any]]]:
    final33 = list(legacy.FEATURES_33)
    final34 = list(legacy.FEATURES_34)
    if final33 != list(legacy.audit.FINAL_33):
        raise RuntimeError("Legacy driver FEATURES_33 no longer matches base audit FINAL_33")
    if final34 != list(legacy.audit.FEATURES_34):
        raise RuntimeError("Legacy driver FEATURES_34 no longer matches base audit FEATURES_34")
    if "ton_iot" not in legacy.audit.DATASETS:
        raise RuntimeError("Legacy driver did not configure the base audit module for ton_iot")
    if "frame.time_epoch" in final33:
        raise RuntimeError("Legacy FEATURES_33 unexpectedly contains frame.time_epoch")
    if "inter_arrival_time" not in final33:
        raise RuntimeError("Legacy FEATURES_33 does not contain inter_arrival_time; cannot construct q/r bridge")
    q = [f for f in final33 if f != "inter_arrival_time"]
    if len(q) != len(final33) - 1:
        raise RuntimeError("q feature derivation failed")
    specs = {
        # Bridge conditions preserve the historical feature order exactly.  This matters
        # especially for the feature-axis CNN; changing order would change adjacency.
        "L34_q_r_t": {"features": final34, "preprocess": "precision_safe", "role": "legacy_bridge_with_epoch"},
        "L33_q_r": {"features": final33, "preprocess": "precision_safe", "role": "legacy_bridge_without_epoch"},
        "A_q_t": {"features": q + ["frame.time_epoch"], "preprocess": "precision_safe", "role": "primary_A"},
        "B_q": {"features": q, "preprocess": "precision_safe", "role": "primary_B"},
        "C_q_b_t": {"features": q + BEHAVIOR_FEATURES + ["frame.time_epoch"], "preprocess": "precision_safe", "role": "primary_C"},
        "D_q_b": {"features": q + BEHAVIOR_FEATURES, "preprocess": "precision_safe", "role": "primary_D"},
        PRECISION_CONDITION_NAME: {"features": q + ["frame.time_epoch"], "preprocess": "epoch_float32_precast_only", "role": "precision_sensitivity"},
    }
    return q, specs


def preflight(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol: dict[str, Any], protocol_hash: str):
    if stage_is_complete(dirs, "preflight", protocol_hash):
        log(root, "[RESUME] preflight already complete")
        return
    log(root, "START preflight")

    for p in [LEGACY_FINAL_SCRIPT, LEGACY_BASE_SCRIPT, SOURCE_V2]:
        if not p.exists():
            raise FileNotFoundError(p)

    actual_hashes = {
        LEGACY_FINAL_SCRIPT.name: sha256_file(LEGACY_FINAL_SCRIPT),
        LEGACY_BASE_SCRIPT.name: sha256_file(LEGACY_BASE_SCRIPT),
    }
    mismatches = []
    for name, expected in EXPECTED_LEGACY_HASHES.items():
        actual = actual_hashes.get(name)
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    if mismatches and not args.allow_code_hash_change:
        raise RuntimeError(f"Frozen legacy dependency hash mismatch; refusing changed implementation: {mismatches}")

    prior_prov = args.prior_provenance.resolve() if args.prior_provenance else latest_completed_run(PROVENANCE_ROOT)
    prior_flow = args.prior_flow_audit.resolve() if args.prior_flow_audit else latest_completed_run(FLOW_AUDIT_ROOT)
    for p in [prior_prov, prior_flow]:
        if not p.exists():
            raise FileNotFoundError(p)

    prov_status = load_json(find_table(prior_prov, "RUN_STATUS.json"))
    flow_status = load_json(find_table(prior_flow, "RUN_STATUS.json"))
    if prov_status.get("status") != "COMPLETED":
        raise RuntimeError("Prior provenance audit is not COMPLETED")
    if flow_status.get("status") != "COMPLETED":
        raise RuntimeError("Prior flow audit is not COMPLETED")

    parent_decisions = pd.read_csv(find_table(prior_prov, "14_parent_decisions.csv"))
    flow_feas = pd.read_csv(find_table(prior_flow, "30_feasibility.csv"))
    decision_rows = []
    for ds in DATASET_ORDER:
        r = parent_decisions[parent_decisions["dataset"].astype(str) == ds]
        if len(r) != 1:
            raise RuntimeError(f"{ds}: expected exactly one Script17 parent decision")
        rr = r.iloc[0]
        if str(rr["status"]) != "CONTENT_VALIDATED_PARENT_LINK":
            raise RuntimeError(f"{ds}: parent not content-validated by Script17")
        if not as_bool(rr["complete_mapping"]):
            raise RuntimeError(f"{ds}: Script17 did not certify complete sample-to-parent mapping")
        parent = Path(str(rr["parent_path"]))
        if not parent.exists():
            raise FileNotFoundError(parent)
        cfg = DATASETS[ds]
        sample = Path(cfg["sample"])
        if not sample.exists():
            raise FileNotFoundError(sample)
        decision_rows.append({
            "dataset": ds,
            "sample_path": str(sample),
            "sample_sha256": sha256_file(sample),
            "parent_path": str(parent),
            "parent_rows": parquet_num_rows(parent),
            "parent_sha256": sha256_file(parent) if args.hash_large_inputs else "NOT_HASHED_BY_DEFAULT",
            "script17_status": str(rr["status"]),
            "script17_complete_mapping": as_bool(rr["complete_mapping"]),
            "script18_capture_ready": as_bool(flow_feas.loc[flow_feas["dataset"] == ds, "capture_disjoint_ready"].iloc[0]) if ds in set(flow_feas["dataset"]) else False,
            "script18_session_ready": as_bool(flow_feas.loc[flow_feas["dataset"] == ds, "session_disjoint_ready"].iloc[0]) if ds in set(flow_feas["dataset"]) else False,
        })

    pd.DataFrame(decision_rows).to_csv(dirs["metadata"] / "input_inventory.csv", index=False)
    shutil.copy2(LEGACY_FINAL_SCRIPT, dirs["source_archive"] / LEGACY_FINAL_SCRIPT.name)
    shutil.copy2(LEGACY_BASE_SCRIPT, dirs["source_archive"] / LEGACY_BASE_SCRIPT.name)
    shutil.copy2(Path(__file__).resolve(), dirs["source_archive"] / Path(__file__).name)

    env = {
        "created_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pa.__version__,
        "legacy_hashes": actual_hashes,
        "legacy_hash_mismatches_allowed": bool(mismatches and args.allow_code_hash_change),
        "legacy_hash_mismatches": mismatches,
        "prior_provenance": str(prior_prov),
        "prior_flow_audit": str(prior_flow),
        "tshark_exists": TSHARK.exists(),
        "capinfos_exists": CAPINFOS.exists(),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER"),
    }
    save_json(dirs["metadata"] / "environment.json", env)
    save_json(dirs["metadata"] / "resolved_inputs.json", {"prior_provenance": str(prior_prov), "prior_flow_audit": str(prior_flow)})
    save_json(dirs["metadata"] / "protocol_freeze.json", {"protocol_hash": protocol_hash, "protocol": protocol})
    save_json(dirs["metadata"] / "non_executable_gates.json", protocol["non_executable_gates"])
    mark_stage(dirs, "preflight", protocol_hash, {"prior_provenance": str(prior_prov), "prior_flow_audit": str(prior_flow)})
    log(root, "END preflight [COMPLETED]")


# =============================================================================
# CAPTURE-FILE BLOCK RECONSTRUCTION
# =============================================================================

def classify_capture_dataset(path: str) -> str | None:
    low = path.lower()
    if "edge_iiotset" in low:
        return "edge_iiotset"
    if "ciciot2023" in low:
        return "ciciot2023"
    if "/dataset-raw/pcap/" in low or "ton_iot" in low or "ton-iot" in low:
        return "ton_iot"
    return None


def parse_numeric_token(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip().strip('"')
    if not s:
        return None
    # tshark can return comma-separated repeated values; use first occurrence.
    if "," in s:
        s = s.split(",", 1)[0].strip()
    try:
        if s.lower().startswith("0x"):
            return float(int(s, 16))
        return float(s)
    except Exception:
        return None


def tshark_prefix(path: Path, nframes: int, cache_dir: Path) -> pd.DataFrame:
    key = sha256_text(f"{path}|{path.stat().st_size}|{path.stat().st_mtime_ns}|{nframes}|{'|'.join(DISCOVERY_FIELDS)}")[:20]
    cache = cache_dir / f"{key}.csv.gz"
    meta = cache_dir / f"{key}.json"
    if cache.exists() and meta.exists():
        return pd.read_csv(cache, compression="gzip")
    if not TSHARK.exists():
        raise FileNotFoundError(TSHARK)
    args = [str(TSHARK), "-n", "-r", str(path), "-c", str(nframes), "-T", "fields",
            "-E", "header=y", "-E", "separator=/t", "-E", "quote=n", "-E", "occurrence=f"]
    for f in DISCOVERY_FIELDS:
        args += ["-e", f]
    cp = run_cmd(args, timeout=600)
    if cp.returncode != 0:
        save_json(meta, {"path": str(path), "status": "ERROR", "returncode": cp.returncode, "stderr": cp.stderr[-4000:]})
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(cp.stdout), sep="\t", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=DISCOVERY_FIELDS)
    # Normalize columns even if tshark omits a field header unexpectedly.
    for c in DISCOVERY_FIELDS:
        if c not in df.columns:
            df[c] = ""
    df = df[DISCOVERY_FIELDS].copy()
    df.to_csv(cache, index=False, compression="gzip")
    save_json(meta, {"path": str(path), "status": "READ", "returncode": cp.returncode, "rows": len(df), "stderr": cp.stderr[-1000:]})
    return df


def capinfos_packet_count(path: Path) -> int | None:
    if not CAPINFOS.exists():
        return None
    # -M requests machine-readable numeric output on supported Wireshark builds,
    # avoiding human-format suffixes such as k/M for large captures.
    cp = run_cmd([str(CAPINFOS), "-M", "-c", str(path)], timeout=300)
    if cp.returncode != 0:
        cp = run_cmd([str(CAPINFOS), "-c", str(path)], timeout=300)
    if cp.returncode != 0:
        return None
    for line in cp.stdout.splitlines():
        if "Number of packets" not in line:
            continue
        tail = line.split(":", 1)[-1].strip().replace(",", "")
        # First prefer an exact integer.
        m = re.fullmatch(r"([0-9]+)", tail)
        if m:
            return int(m.group(1))
        # Defensive fallback for human-readable SI suffixes.
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmMgG])?", tail)
        if m:
            mult = {None: 1.0, "k": 1e3, "m": 1e6, "g": 1e9}[m.group(2).lower() if m.group(2) else None]
            value = float(m.group(1)) * mult
            # A rounded human-readable count is not safe enough to define a raw
            # capture boundary. Return only when it is effectively integral.
            if abs(value - round(value)) <= 1e-6:
                return int(round(value))
    return None


def discovery_record_from_row(capture_path: str, row: pd.Series) -> dict[str, Any] | None:
    frame = parse_numeric_token(row.get("frame.number"))
    epoch = parse_numeric_token(row.get("frame.time_epoch"))
    length = parse_numeric_token(row.get("frame.len"))
    if frame is None or epoch is None or length is None:
        return None
    rec = {
        "capture_path": capture_path,
        "frame_number": int(frame),
        "epoch_us": int(round(epoch * 1_000_000.0)),
        "frame_len": int(length),
    }
    for f in PARENT_DISCOVERY_FIELDS:
        if f in {"frame.time_epoch", "frame.len"}:
            continue
        rec[f] = parse_numeric_token(row.get(f))
    return rec


def parent_candidate_matches(rec: dict[str, Any], batch: pd.DataFrame, idx: int) -> bool:
    try:
        plen = parse_numeric_token(batch.iloc[idx]["frame.len"])
        if plen is None or int(plen) != int(rec["frame_len"]):
            return False
    except Exception:
        return False
    # Compare available protocol fields. Missing tshark fields do not veto.
    compared = 0
    for f in ["ip.proto", "ip.id", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "icmp.type", "icmp.code"]:
        rv = rec.get(f)
        if rv is None:
            continue
        pv = parse_numeric_token(batch.iloc[idx].get(f))
        if pv is None:
            # Parent extractors often encode absent fields as NaN/0. If tshark has a real field,
            # absence in parent is a mismatch.
            return False
        compared += 1
        if not math.isclose(float(rv), float(pv), rel_tol=0.0, abs_tol=1e-9):
            return False
    return True if compared >= 1 else True


def derive_capture_offsets(parent: Path, captures: list[Path], nframes: int, batch_size: int, cache_dir: Path, root: Path, ds: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    cap_meta_rows = []
    for i, cap in enumerate(captures, start=1):
        if not cap.exists() or cap.stat().st_size <= 0:
            cap_meta_rows.append({"dataset": ds, "capture_path": str(cap), "status": "MISSING_OR_EMPTY"})
            continue
        log(root, f"{ds}: tshark prefix {i}/{len(captures)} {cap.name}")
        df = tshark_prefix(cap, nframes, cache_dir)
        for _, r in df.iterrows():
            rec = discovery_record_from_row(str(cap), r)
            if rec is not None:
                records.append(rec)
        cap_meta_rows.append({
            "dataset": ds,
            "capture_path": str(cap),
            "status": "PREFIX_READ" if len(df) else "PREFIX_EMPTY",
            "prefix_rows": len(df),
            "packet_count": capinfos_packet_count(cap),
            "bytes": cap.stat().st_size,
            "mtime_ns": cap.stat().st_mtime_ns,
        })
    if not records:
        raise RuntimeError(f"{ds}: no usable capture prefix records")
    pd.DataFrame(cap_meta_rows).to_csv(cache_dir.parent / f"{ds}_capture_source_metadata.csv", index=False)

    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_epoch[int(rec["epoch_us"])].append(rec)
    target_epochs = np.asarray(sorted(by_epoch.keys()), dtype=np.int64)
    offset_counts: dict[str, Counter] = defaultdict(Counter)
    matched_frames: dict[str, int] = defaultdict(int)
    match_examples: list[dict[str, Any]] = []

    pf = pq.ParquetFile(parent)
    parent_cols = [c for c in PARENT_DISCOVERY_FIELDS if c in pf.schema_arrow.names]
    global_pos = 0
    for bno, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=parent_cols, use_threads=True), start=1):
        pdf = batch.to_pandas()
        epoch = pd.to_numeric(pdf["frame.time_epoch"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(epoch)
        eus = np.zeros(len(pdf), dtype=np.int64)
        eus[valid] = np.rint(epoch[valid] * 1_000_000.0).astype(np.int64)
        # np.isin sorts internally and is substantially faster than Python-looping all rows.
        mask = valid & np.isin(eus, target_epochs, assume_unique=False)
        idxs = np.flatnonzero(mask)
        for idx in idxs:
            e = int(eus[idx])
            for rec in by_epoch.get(e, []):
                if parent_candidate_matches(rec, pdf, int(idx)):
                    parent_pos = global_pos + int(idx)
                    offset = int(parent_pos - int(rec["frame_number"]) + 1)
                    offset_counts[rec["capture_path"]][offset] += 1
                    matched_frames[rec["capture_path"]] += 1
                    if len(match_examples) < 10000:
                        match_examples.append({
                            "dataset": ds,
                            "capture_path": rec["capture_path"],
                            "frame_number": rec["frame_number"],
                            "parent_position": parent_pos,
                            "offset": offset,
                            "epoch_us": e,
                            "frame_len": rec["frame_len"],
                        })
        global_pos += len(pdf)
        if bno == 1 or bno % 20 == 0 or global_pos == pf.metadata.num_rows:
            log(root, f"{ds}: capture discovery parent rows={global_pos:,}/{pf.metadata.num_rows:,}")
        del pdf
    pd.DataFrame(match_examples).to_csv(cache_dir.parent / f"{ds}_capture_match_examples.csv.gz", index=False, compression="gzip")

    meta = pd.read_csv(cache_dir.parent / f"{ds}_capture_source_metadata.csv")
    meta_map = {str(r.capture_path): r for r in meta.itertuples(index=False)}
    rows = []
    for cap in captures:
        p = str(cap)
        ctr = offset_counts.get(p, Counter())
        total_prefix = int(getattr(meta_map.get(p), "prefix_rows", 0) or 0)
        if not ctr:
            rows.append({
                "dataset": ds, "capture_path": p, "mapping_status": "NO_PARENT_OFFSET",
                "best_offset": np.nan, "best_support": 0, "second_support": 0,
                "best_share_of_prefix": 0.0, "dominance": np.nan,
                "prefix_rows": total_prefix,
                "packet_count": getattr(meta_map.get(p), "packet_count", np.nan),
            })
            continue
        best = ctr.most_common(2)
        best_off, best_n = best[0]
        second_n = best[1][1] if len(best) > 1 else 0
        share = best_n / max(total_prefix, 1)
        dominance = best_n / max(second_n, 1)
        ok = (
            best_n >= MIN_CAPTURE_OFFSET_SUPPORT
            and share >= MIN_CAPTURE_OFFSET_SHARE
            and (dominance >= MIN_CAPTURE_OFFSET_DOMINANCE or (best_n - second_n) >= 10)
            and 0 <= int(best_off) < parquet_num_rows(parent)
        )
        rows.append({
            "dataset": ds,
            "capture_path": p,
            "mapping_status": "OFFSET_VERIFIED_PREFIX" if ok else "OFFSET_AMBIGUOUS",
            "best_offset": int(best_off),
            "best_support": int(best_n),
            "second_support": int(second_n),
            "best_share_of_prefix": float(share),
            "dominance": float(dominance),
            "prefix_rows": total_prefix,
            "matched_prefix_records": int(matched_frames.get(p, 0)),
            "distinct_offset_candidates": int(len(ctr)),
            "packet_count": getattr(meta_map.get(p), "packet_count", np.nan),
        })
    return pd.DataFrame(rows)


def summarize_block_labels(parent: Path, blocks: pd.DataFrame, label_col: str, normal_label: str, root: Path, ds: str) -> pd.DataFrame:
    out = []
    for i, r in enumerate(blocks.itertuples(index=False), start=1):
        start = int(r.start_row)
        end = int(r.end_row)
        if not bool(r.capture_block_certified) or end < start:
            out.append({
                "dataset": ds, "capture_id": r.capture_id, "capture_path": r.capture_path,
                "start_row": start, "end_row": end, "block_rows": max(0, end - start + 1),
                "normal_n_parent": 0, "attack_n_parent": 0, "block_class": "UNCERTIFIED",
                "label_counts_json": "{}",
            })
            continue
        counts_counter: Counter = Counter()
        row_n = 0
        for _, piece in iter_parquet_range_batches(parent, start, end, [label_col], batch_size=250_000):
            vals = piece[label_col].astype(str)
            counts_counter.update(vals.value_counts().to_dict())
            row_n += len(piece)
        if row_n != end - start + 1:
            raise RuntimeError(f"{ds}: streamed label rows {row_n} != expected {end-start+1} for {r.capture_id}")
        normal_n = int(counts_counter.get(str(normal_label), 0))
        attack_n = int(row_n - normal_n)
        if attack_n == 0:
            block_class = "PURE_NORMAL"
        elif normal_n == 0:
            block_class = "PURE_ATTACK"
        else:
            block_class = "MIXED"
        counts = {str(k): int(v) for k, v in counts_counter.items()}
        out.append({
            "dataset": ds,
            "capture_id": r.capture_id,
            "capture_path": r.capture_path,
            "start_row": start,
            "end_row": end,
            "block_rows": end - start + 1,
            "normal_n_parent": normal_n,
            "attack_n_parent": attack_n,
            "block_class": block_class,
            "label_counts_json": json.dumps(counts, sort_keys=True),
        })
        if i == 1 or i % 5 == 0 or i == len(blocks):
            log(root, f"{ds}: label summary blocks={i}/{len(blocks)}")
    return pd.DataFrame(out)


def _capture_reuse_is_compatible(reuse_root: Path, args: argparse.Namespace) -> bool:
    freeze = reuse_root / "metadata" / "protocol_freeze.json"
    if not freeze.exists():
        return False
    try:
        obj = load_json(freeze)
        p = obj.get("protocol", obj).get("capture_mapping_parameters", obj.get("capture_mapping_parameters", {}))
        expected = {
            "capture_prefix_frames": int(args.capture_prefix_frames),
            "parent_scan_batch": int(args.parent_scan_batch),
            "min_offset_support": MIN_CAPTURE_OFFSET_SUPPORT,
            "min_offset_share": MIN_CAPTURE_OFFSET_SHARE,
            "min_offset_dominance": MIN_CAPTURE_OFFSET_DOMINANCE,
            "block_length_tolerance_frames": BLOCK_LENGTH_TOLERANCE_FRAMES,
        }
        return all(str(p.get(k)) == str(v) for k, v in expected.items())
    except Exception:
        return False


def _try_reuse_capture_dataset(args: argparse.Namespace, ds: str, dsdir: Path, root: Path):
    reuse_root = args.reuse_capture_root
    if reuse_root is None:
        return None
    reuse_root = reuse_root.expanduser().resolve()
    if reuse_root == root.resolve() or not reuse_root.exists() or not _capture_reuse_is_compatible(reuse_root, args):
        return None
    old = reuse_root / "capture_mapping" / ds
    required = [old / "10_capture_blocks.csv", old / "11_sample_capture_map.csv.gz", old / "RUN_COMPLETE.json"]
    if not all(p.exists() for p in required):
        return None
    try:
        blocks = pd.read_csv(old / "10_capture_blocks.csv")
        mapped = pd.read_csv(old / "11_sample_capture_map.csv.gz")
        status = load_json(old / "RUN_COMPLETE.json")
        cfg = DATASETS[ds]
        sample_probe = pd.read_parquet(cfg["sample"], columns=["source_row_index"])
        if len(mapped) != len(sample_probe):
            return None
        if not np.array_equal(
            mapped["source_row_index"].to_numpy(dtype=np.int64),
            sample_probe["source_row_index"].to_numpy(dtype=np.int64),
        ):
            return None
        shutil.copytree(old, dsdir, dirs_exist_ok=True)
        status = dict(status)
        status["evidence_reused_from"] = str(old)
        status["reuse_validated_by"] = "mapping parameter equality + exact sample source_row_index identity"
        save_json(dsdir / "RUN_COMPLETE.json", status)
        log(root, f"{ds}: reused capture evidence from {old}")
        return blocks, mapped, status
    except Exception as exc:
        log(root, f"{ds}: capture-evidence reuse rejected: {type(exc).__name__}: {exc}")
        return None


def capture_map_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str):
    if stage_is_complete(dirs, "capture_map", protocol_hash):
        log(root, "[RESUME] capture_map already complete")
        return
    log(root, "START capture_map")
    resolved = load_json(dirs["metadata"] / "resolved_inputs.json")
    prior_prov = Path(resolved["prior_provenance"])
    parent_decisions = pd.read_csv(find_table(prior_prov, "14_parent_decisions.csv"))
    pcap_scans = pd.read_csv(find_table(prior_prov, "20_pcap_scans.csv"))

    all_blocks = []
    all_sample_maps = []
    dataset_status = []
    for ds in DATASET_ORDER:
        cfg = DATASETS[ds]
        rr = parent_decisions[parent_decisions["dataset"] == ds].iloc[0]
        parent = Path(str(rr["parent_path"]))
        nrows = parquet_num_rows(parent)
        dsdir = dirs["capture"] / ds
        dsdir.mkdir(parents=True, exist_ok=True)

        reused = _try_reuse_capture_dataset(args, ds, dsdir, root)
        if reused is not None:
            blocks, mapped, status = reused
        else:
            captures = []
            for p in pcap_scans["path"].astype(str).unique():
                if classify_capture_dataset(p) == ds and Path(p).exists() and Path(p).is_file() and Path(p).stat().st_size > 0:
                    captures.append(Path(p))
            captures = sorted(set(captures), key=lambda p: str(p))
            if not captures:
                raise RuntimeError(f"{ds}: no existing raw capture candidates from Script17")

            prefix_cache = dsdir / "prefix_cache"
            prefix_cache.mkdir(parents=True, exist_ok=True)
            offset_df = derive_capture_offsets(parent, captures, args.capture_prefix_frames, args.parent_scan_batch, prefix_cache, root, ds)
            offset_df.to_csv(dsdir / "01_capture_offset_candidates.csv", index=False)

            verified = offset_df[offset_df["mapping_status"] == "OFFSET_VERIFIED_PREFIX"].copy()
            if verified.empty:
                # No raw-capture offset is still a valid provenance result. The experiment
                # will use the pre-specified parent-order fallback, never a capture claim.
                log(root, f"{ds}: no verified raw-capture offset; parent-order fallback will be required")
                resolved_df = pd.DataFrame(columns=offset_df.columns)
            else:
                verified["best_offset"] = verified["best_offset"].astype(np.int64)
                conflict_rows = []
                keep_rows = []
                for start_row, g in verified.groupby("best_offset", sort=True):
                    if len(g) == 1:
                        keep_rows.append(g.iloc[0])
                        continue
                    gs = g.sort_values("best_support", ascending=False)
                    top = gs.iloc[0]
                    second = gs.iloc[1]
                    if float(top["best_support"]) >= 2.0 * max(float(second["best_support"]), 1.0):
                        keep_rows.append(top)
                        for _, loser in gs.iloc[1:].iterrows():
                            conflict_rows.append({"dataset": ds, "start_row": int(start_row), "kept": top["capture_path"], "rejected": loser["capture_path"], "reason": "duplicate_start_weaker_support"})
                    else:
                        for _, x in gs.iterrows():
                            conflict_rows.append({"dataset": ds, "start_row": int(start_row), "kept": "", "rejected": x["capture_path"], "reason": "duplicate_start_unresolved"})
                if conflict_rows:
                    pd.DataFrame(conflict_rows).to_csv(dsdir / "02_start_conflicts.csv", index=False)
                resolved_df = pd.DataFrame(keep_rows).sort_values("best_offset").reset_index(drop=True) if keep_rows else pd.DataFrame(columns=offset_df.columns)

            block_rows = []
            starts = resolved_df["best_offset"].astype(int).tolist() if len(resolved_df) else []
            for i, r in resolved_df.iterrows():
                start_row = int(r["best_offset"])
                next_start = starts[i + 1] if i + 1 < len(starts) else None
                pcount = r.get("packet_count")
                pcount_i = None if pd.isna(pcount) else int(pcount)
                expected_end = None if pcount_i is None else start_row + pcount_i - 1
                within_parent = bool(expected_end is not None and start_row >= 0 and expected_end < nrows)
                no_overlap_next = bool(expected_end is not None and (next_start is None or expected_end < next_start))
                cert = bool(pcount_i is not None and pcount_i > 0 and within_parent and no_overlap_next)
                if expected_end is None:
                    end_row = start_row - 1
                    block_len = 0
                else:
                    end_row = min(expected_end, nrows - 1)
                    block_len = max(0, end_row - start_row + 1)
                gap_after = (int(next_start) - (int(expected_end) + 1)) if (next_start is not None and expected_end is not None) else np.nan
                cap_id = f"{ds}::cap_{i:03d}::{Path(str(r['capture_path'])).name}"
                block_rows.append({
                    "dataset": ds,
                    "capture_id": cap_id,
                    "capture_path": str(r["capture_path"]),
                    "start_row": start_row,
                    "end_row": end_row,
                    "block_rows": block_len,
                    "pcap_packet_count": pcount_i,
                    "next_verified_start_row": next_start,
                    "parent_gap_after_expected_capture_rows": gap_after,
                    "mapping_support": int(r["best_support"]),
                    "mapping_second_support": int(r["second_support"]),
                    "mapping_share": float(r["best_share_of_prefix"]),
                    "mapping_dominance": float(r["dominance"]),
                    "raw_count_expected_end_within_parent": within_parent,
                    "raw_count_block_does_not_overlap_next_verified_start": no_overlap_next,
                    "block_length_consistent_with_pcap": bool(pcount_i is not None and block_len == pcount_i),
                    "capture_block_certified": cert,
                    "certification_scope": "multi_field_prefix_offset_plus_raw_packet_count_boundary; gaps remain unmapped; not payload identity or session independence",
                })
            blocks = pd.DataFrame(block_rows)
            if len(blocks):
                label_summary = summarize_block_labels(parent, blocks, cfg["label"], cfg["normal"], root, ds)
                blocks = blocks.merge(label_summary[["capture_id", "normal_n_parent", "attack_n_parent", "block_class", "label_counts_json"]], on="capture_id", how="left")
            else:
                blocks = pd.DataFrame(columns=[
                    "dataset", "capture_id", "capture_path", "start_row", "end_row", "block_rows",
                    "capture_block_certified", "normal_n_parent", "attack_n_parent", "block_class", "label_counts_json"
                ])
            blocks.to_csv(dsdir / "10_capture_blocks.csv", index=False)

            sample = pd.read_parquet(cfg["sample"], columns=["source_row_index", cfg["label"], "label_type", "frame.time_epoch"])
            src = sample["source_row_index"].to_numpy(dtype=np.int64)
            cap_ids = np.full(len(sample), "UNMAPPED", dtype=object)
            cap_paths = np.full(len(sample), "", dtype=object)
            cert = np.zeros(len(sample), dtype=bool)
            if len(blocks):
                starts_np = blocks["start_row"].to_numpy(dtype=np.int64)
                ends_np = blocks["end_row"].to_numpy(dtype=np.int64)
                ix = np.searchsorted(starts_np, src, side="right") - 1
                valid = ix >= 0
                valid &= np.where(valid, src <= ends_np[np.clip(ix, 0, len(ends_np) - 1)], False)
                for j in np.flatnonzero(valid):
                    br = blocks.iloc[int(ix[j])]
                    cap_ids[j] = br["capture_id"]
                    cap_paths[j] = br["capture_path"]
                    cert[j] = as_bool(br["capture_block_certified"])
            mapped = pd.DataFrame({
                "dataset": ds,
                "sample_local_index": np.arange(len(sample), dtype=np.int64),
                "source_row_index": src,
                "capture_id": cap_ids,
                "capture_path": cap_paths,
                "capture_block_certified": cert,
                "label": sample[cfg["label"]].astype(str).to_numpy(),
                "label_type": sample["label_type"].astype(str).to_numpy(),
                "frame.time_epoch": pd.to_numeric(sample["frame.time_epoch"], errors="coerce").to_numpy(dtype=float),
            })
            mapped.to_csv(dsdir / "11_sample_capture_map.csv.gz", index=False, compression="gzip")
            ready = bool(np.all(mapped["capture_block_certified"]) and np.all(mapped["capture_id"] != "UNMAPPED"))
            covered_rows = int(blocks.loc[blocks["capture_block_certified"].map(as_bool), "block_rows"].sum()) if len(blocks) else 0
            status = {
                "dataset": ds,
                "raw_capture_candidates_existing": len(captures),
                "verified_capture_starts": len(blocks),
                "certified_capture_blocks": int(blocks["capture_block_certified"].map(as_bool).sum()) if len(blocks) else 0,
                "certified_parent_rows_covered": covered_rows,
                "parent_rows_total": int(nrows),
                "parent_rows_not_claimed_by_certified_capture_blocks": int(max(nrows - covered_rows, 0)),
                "sample_rows": len(mapped),
                "sample_rows_mapped": int(np.sum(mapped["capture_id"] != "UNMAPPED")),
                "sample_rows_certified": int(mapped["capture_block_certified"].map(as_bool).sum()),
                "sample_capture_mapping_ready": ready,
                "pure_normal_capture_blocks": int(np.sum(blocks["block_class"] == "PURE_NORMAL")) if len(blocks) else 0,
                "pure_attack_capture_blocks": int(np.sum(blocks["block_class"] == "PURE_ATTACK")) if len(blocks) else 0,
                "mixed_capture_blocks": int(np.sum(blocks["block_class"] == "MIXED")) if len(blocks) else 0,
                "claim": "capture-file block mapping only; session independence is NOT certified",
            }
            save_json(dsdir / "RUN_COMPLETE.json", status)

        # Recompute readiness from loaded evidence so reused v1.1 JSON cannot override v1.2 semantics.
        if len(blocks):
            blocks["capture_block_certified"] = blocks["capture_block_certified"].map(as_bool)
        mapped["capture_block_certified"] = mapped["capture_block_certified"].map(as_bool)
        ready = bool(np.all(mapped["capture_block_certified"].to_numpy(dtype=bool)) and np.all(mapped["capture_id"].astype(str) != "UNMAPPED"))
        status = dict(status)
        status["sample_capture_mapping_ready"] = ready
        status["v12_policy"] = (
            "USE_CERTIFIED_CAPTURE_GROUPS" if ready else
            "FALLBACK_TO_FIXED_PARENT_ORDER_BLOCKS; DO_NOT CLAIM CAPTURE_OR_SESSION_DISJOINTNESS"
        )
        save_json(dsdir / "RUN_COMPLETE.json", status)
        dataset_status.append(status)
        all_blocks.append(blocks)
        all_sample_maps.append(mapped)

        # Persist after every dataset so an external kill cannot erase completed provenance work.
        pd.concat(all_blocks, ignore_index=True).to_csv(dirs["capture"] / "capture_blocks_all.csv", index=False)
        pd.concat(all_sample_maps, ignore_index=True).to_csv(dirs["capture"] / "sample_capture_map_all.csv.gz", index=False, compression="gzip")
        pd.DataFrame(dataset_status).to_csv(dirs["capture"] / "capture_mapping_status.csv", index=False)
        log(root, f"{ds}: capture_ready={ready} certified_sample_rows={int(mapped['capture_block_certified'].sum()):,}/{len(mapped):,}")

    capture_ready = [x["dataset"] for x in dataset_status if as_bool(x.get("sample_capture_mapping_ready", False))]
    fallback = [x["dataset"] for x in dataset_status if x["dataset"] not in capture_ready]
    mark_stage(dirs, "capture_map", protocol_hash, {
        "capture_ready_datasets": capture_ready,
        "parent_order_fallback_required_datasets": fallback,
        "claim_policy": "Incomplete raw-capture coverage triggers parent-order grouping, never an invented capture/session ID.",
    })
    log(root, f"END capture_map [COMPLETED] capture_ready={capture_ready} fallback={fallback}")


# =============================================================================
# ACQUISITION-AWARE / PARENT-ORDER SPLIT CONSTRUCTION
# =============================================================================

def _sample_counts_by_group(sample_map: pd.DataFrame, group_col: str, normal_label: str) -> pd.DataFrame:
    tmp = sample_map.copy()
    tmp["is_normal"] = tmp["label"].astype(str) == str(normal_label)
    out = tmp.groupby(group_col, as_index=False).agg(
        sample_rows=("source_row_index", "size"),
        sample_normal_n=("is_normal", "sum"),
    )
    out["sample_attack_n"] = out["sample_rows"] - out["sample_normal_n"]
    return out


def _allocate_normal_groups(normal: pd.DataFrame, id_col: str) -> tuple[dict[str, str], str | None]:
    normal = normal.sort_values("start_row").reset_index(drop=True)
    normal = normal[normal["sample_normal_n"] > 0].reset_index(drop=True)
    if len(normal) < 4:
        return {}, f"only_{len(normal)}_pure_normal_groups_with_sample_rows"
    total = float(normal["sample_normal_n"].sum())
    if total <= 0:
        return {}, "zero_sample_normal_rows"
    csum = normal["sample_normal_n"].cumsum().to_numpy(dtype=float)
    n = len(normal)
    fit_last = int(np.argmin(np.abs(csum / total - 0.60)))
    fit_last = min(max(fit_last, 0), n - 4)
    rem1 = np.arange(fit_last + 1, n - 2)
    if len(rem1) == 0:
        return {}, "cannot_allocate_earlystop_group"
    early_last = int(rem1[np.argmin(np.abs(csum[rem1] / total - 0.75))])
    rem2 = np.arange(early_last + 1, n - 1)
    if len(rem2) == 0:
        return {}, "cannot_allocate_calibration_group"
    cal_last = int(rem2[np.argmin(np.abs(csum[rem2] / total - 0.85))])
    assign = {}
    for i, r in normal.iterrows():
        if i <= fit_last:
            role = "fit_normal"
        elif i <= early_last:
            role = "earlystop_normal"
        elif i <= cal_last:
            role = "calibration_normal"
        else:
            role = "test"
        assign[str(r[id_col])] = role
    return assign, None


def _whole_capture_partition(blocks: pd.DataFrame, sample_map: pd.DataFrame, normal_label: str) -> tuple[pd.DataFrame, str] | tuple[None, str]:
    counts = _sample_counts_by_group(sample_map, "capture_id", normal_label)
    b = blocks.merge(counts, on="capture_id", how="left").fillna({"sample_rows": 0, "sample_normal_n": 0, "sample_attack_n": 0})
    normal = b[(b["block_class"] == "PURE_NORMAL") & (b["capture_block_certified"].map(as_bool))].copy()
    assign, reason = _allocate_normal_groups(normal, "capture_id")
    if reason is not None:
        return None, reason
    units = []
    for r in blocks.sort_values("start_row").itertuples(index=False):
        if not as_bool(r.capture_block_certified):
            continue
        split = assign.get(str(r.capture_id), "test") if str(r.block_class) == "PURE_NORMAL" else "test"
        units.append({
            "dataset": r.dataset,
            "behavior_unit_id": f"{r.capture_id}::whole",
            "analysis_unit_id": str(r.capture_id),
            "analysis_unit_kind": "CERTIFIED_RAW_CAPTURE_FILE",
            "raw_capture_id": str(r.capture_id),
            "raw_capture_path": str(r.capture_path),
            "capture_id": str(r.capture_id),  # internal backward-compatible alias
            "capture_path": str(r.capture_path),
            "unit_start_row": int(r.start_row),
            "unit_end_row": int(r.end_row),
            "unit_rows": int(r.end_row) - int(r.start_row) + 1,
            "split": split,
            "block_class": r.block_class,
            "unit_kind": "WHOLE_CERTIFIED_CAPTURE",
            "embargo_before_rows": 0,
            "embargo_after_rows": 0,
        })
    return pd.DataFrame(units), "CERTIFIED_CAPTURE_FILE_DISJOINT_NORMAL_PARTITIONS"


def _blocked_within_capture_partition(blocks: pd.DataFrame, embargo_rows: int) -> tuple[pd.DataFrame, str]:
    units = []
    half_left = int(embargo_rows // 2)
    half_right = int(embargo_rows - half_left)
    roles = ["fit_normal", "earlystop_normal", "calibration_normal", "test"]
    fractions = [0.60, 0.15, 0.10, 0.15]
    for r in blocks.sort_values("start_row").itertuples(index=False):
        if not as_bool(r.capture_block_certified):
            continue
        start_row, end_row = int(r.start_row), int(r.end_row)
        if str(r.block_class) != "PURE_NORMAL":
            aid = f"{r.capture_id}::test_full"
            units.append({
                "dataset": r.dataset, "behavior_unit_id": aid, "analysis_unit_id": aid,
                "analysis_unit_kind": "CERTIFIED_CAPTURE_SUBBLOCK", "raw_capture_id": str(r.capture_id),
                "raw_capture_path": str(r.capture_path), "capture_id": aid, "capture_path": str(r.capture_path),
                "unit_start_row": start_row, "unit_end_row": end_row, "unit_rows": end_row - start_row + 1,
                "split": "test", "block_class": r.block_class, "unit_kind": "WHOLE_CAPTURE_TEST",
                "embargo_before_rows": 0, "embargo_after_rows": 0,
            })
            continue
        length = end_row - start_row + 1
        cuts = [start_row]
        cumulative = 0.0
        for frac in fractions[:-1]:
            cumulative += frac
            cuts.append(start_row + int(round(cumulative * length)))
        cuts.append(end_row + 1)
        cuts = np.maximum.accumulate(np.asarray(cuts, dtype=np.int64)).tolist()
        for j, role in enumerate(roles):
            lo, hi = int(cuts[j]), int(cuts[j + 1]) - 1
            before = 0 if j == 0 else half_right
            after = 0 if j == len(roles) - 1 else half_left
            lo2, hi2 = lo + before, hi - after
            if hi2 < lo2:
                continue
            aid = f"{r.capture_id}::{role}"
            units.append({
                "dataset": r.dataset, "behavior_unit_id": aid, "analysis_unit_id": aid,
                "analysis_unit_kind": "CERTIFIED_CAPTURE_SUBBLOCK", "raw_capture_id": str(r.capture_id),
                "raw_capture_path": str(r.capture_path), "capture_id": aid, "capture_path": str(r.capture_path),
                "unit_start_row": lo2, "unit_end_row": hi2, "unit_rows": hi2 - lo2 + 1,
                "split": role, "block_class": r.block_class, "unit_kind": "BLOCKED_WITHIN_CERTIFIED_PURE_NORMAL_CAPTURE",
                "embargo_before_rows": before, "embargo_after_rows": after,
            })
    return pd.DataFrame(units), "BLOCKED_NORMAL_WITHIN_CERTIFIED_CAPTURES"


def _summarize_parent_order_blocks(parent: Path, label_col: str, normal_label: str, block_rows: int, root: Path, ds: str) -> pd.DataFrame:
    if block_rows < 10_000:
        raise ValueError("fallback block rows must be >= 10,000")
    pf = pq.ParquetFile(parent)
    if label_col not in pf.schema_arrow.names:
        raise RuntimeError(f"{ds}: parent lacks label column {label_col}")
    nrows = int(pf.metadata.num_rows)
    nblocks = int(math.ceil(nrows / block_rows))
    normal_counts = np.zeros(nblocks, dtype=np.int64)
    attack_counts = np.zeros(nblocks, dtype=np.int64)
    global_pos = 0
    batch_no = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=[label_col], use_threads=True):
        batch_no += 1
        vals = batch.to_pandas()[label_col].astype(str).to_numpy(dtype=object)
        local = 0
        while local < len(vals):
            pos = global_pos + local
            bid = int(pos // block_rows)
            boundary = min(len(vals), int((bid + 1) * block_rows - global_pos))
            seg = vals[local:boundary]
            nnormal = int(np.sum(seg == str(normal_label)))
            normal_counts[bid] += nnormal
            attack_counts[bid] += int(len(seg) - nnormal)
            local = boundary
        global_pos += len(vals)
        if batch_no == 1 or batch_no % 20 == 0 or global_pos == nrows:
            log(root, f"{ds}: parent-order label scan rows={global_pos:,}/{nrows:,}")
    if global_pos != nrows:
        raise RuntimeError(f"{ds}: parent-order scan rows {global_pos} != parent rows {nrows}")
    rows = []
    for bid in range(nblocks):
        start_row = bid * block_rows
        end_row = min(nrows - 1, (bid + 1) * block_rows - 1)
        nn = int(normal_counts[bid])
        aa = int(attack_counts[bid])
        if aa == 0:
            cls = "PURE_NORMAL"
        elif nn == 0:
            cls = "PURE_ATTACK"
        else:
            cls = "MIXED"
        aid = f"{ds}::parent_order_block_{bid:05d}"
        rows.append({
            "dataset": ds, "analysis_unit_id": aid, "analysis_unit_kind": "FIXED_PARENT_ORDER_BLOCK",
            "start_row": start_row, "end_row": end_row, "block_rows": end_row - start_row + 1,
            "normal_n_parent": nn, "attack_n_parent": aa, "block_class": cls,
            "label_counts_json": json.dumps({str(normal_label): nn, "NON_NORMAL_TOTAL": aa}, sort_keys=True),
            "capture_block_certified": False, "capture_id": "UNMAPPED", "capture_path": "",
            "grouping_claim": "fixed parent-row-order block; not a capture, session, flow, or independent environment",
        })
    return pd.DataFrame(rows)


def _parent_order_partition(blocks: pd.DataFrame, sample_map: pd.DataFrame, normal_label: str, embargo_rows: int) -> tuple[pd.DataFrame, str]:
    sm = sample_map.copy()
    starts = blocks["start_row"].to_numpy(dtype=np.int64)
    src = sm["source_row_index"].to_numpy(dtype=np.int64)
    ix = np.searchsorted(starts, src, side="right") - 1
    if np.any(ix < 0):
        raise RuntimeError("materialized row precedes first parent-order block")
    sm["analysis_unit_id"] = blocks.iloc[ix]["analysis_unit_id"].astype(str).to_numpy()
    counts = _sample_counts_by_group(sm, "analysis_unit_id", normal_label)
    b = blocks.merge(counts, on="analysis_unit_id", how="left").fillna({"sample_rows": 0, "sample_normal_n": 0, "sample_attack_n": 0})
    # A parent block may enter development only if its FULL parent-label content is normal.
    normal = b[b["block_class"] == "PURE_NORMAL"].copy()
    assign, reason = _allocate_normal_groups(normal, "analysis_unit_id")
    if reason is not None:
        raise RuntimeError(f"parent-order fallback infeasible: {reason}; block_rows={int(blocks.iloc[0]['block_rows']) if len(blocks) else 'NA'}")
    units = []
    for r in b.sort_values("start_row").itertuples(index=False):
        role = assign.get(str(r.analysis_unit_id), "test") if str(r.block_class) == "PURE_NORMAL" else "test"
        units.append({
            "dataset": r.dataset,
            "behavior_unit_id": str(r.analysis_unit_id),
            "analysis_unit_id": str(r.analysis_unit_id),
            "analysis_unit_kind": "FIXED_PARENT_ORDER_BLOCK",
            "raw_capture_id": "",
            "raw_capture_path": "",
            "capture_id": str(r.analysis_unit_id),  # internal alias; never interpreted as raw capture
            "capture_path": "",
            "unit_start_row": int(r.start_row),
            "unit_end_row": int(r.end_row),
            "unit_rows": int(r.end_row) - int(r.start_row) + 1,
            "split": role,
            "block_class": r.block_class,
            "unit_kind": "FIXED_PARENT_ORDER_BLOCK",
            "embargo_before_rows": 0,
            "embargo_after_rows": 0,
        })
    units = pd.DataFrame(units).sort_values("unit_start_row").reset_index(drop=True)

    # Embargo only where adjacent fixed blocks change protocol roles. The whole
    # behavior state is reset at every block regardless, so this is an extra
    # autocorrelation buffer rather than a leakage fix.
    if embargo_rows > 0 and len(units) > 1:
        before = int(embargo_rows // 2)
        after = int(embargo_rows - before)
        for i in range(len(units) - 1):
            if str(units.loc[i, "split"]) == str(units.loc[i + 1, "split"]):
                continue
            new_hi = int(units.loc[i, "unit_end_row"]) - before
            new_lo = int(units.loc[i + 1, "unit_start_row"]) + after
            if new_hi >= int(units.loc[i, "unit_start_row"]):
                units.loc[i, "unit_end_row"] = new_hi
                units.loc[i, "embargo_after_rows"] = before
            if new_lo <= int(units.loc[i + 1, "unit_end_row"]):
                units.loc[i + 1, "unit_start_row"] = new_lo
                units.loc[i + 1, "embargo_before_rows"] = after
        units["unit_rows"] = units["unit_end_row"].astype(int) - units["unit_start_row"].astype(int) + 1
        units = units[units["unit_rows"] > 0].reset_index(drop=True)
    return units, "FIXED_PARENT_ORDER_BLOCK_DISJOINT_NORMAL_PARTITIONS"


def _map_samples_to_units(ds: str, cfg: dict[str, Any], sample_map: pd.DataFrame, units: pd.DataFrame, split_mode: str) -> pd.DataFrame:
    units = units.sort_values(["unit_start_row", "unit_end_row"]).reset_index(drop=True)
    starts = units["unit_start_row"].to_numpy(dtype=np.int64)
    ends = units["unit_end_row"].to_numpy(dtype=np.int64)
    src = sample_map["source_row_index"].to_numpy(dtype=np.int64)
    ix = np.searchsorted(starts, src, side="right") - 1
    valid = ix >= 0
    clipped = np.clip(ix, 0, max(len(ends) - 1, 0))
    if len(ends):
        valid &= src <= ends[clipped]
    else:
        valid[:] = False

    split = np.full(len(sample_map), "DROP_EMBARGO", dtype=object)
    unit_id = np.full(len(sample_map), "", dtype=object)
    unit_kind = np.full(len(sample_map), "", dtype=object)
    behavior_id = np.full(len(sample_map), "", dtype=object)
    for j in np.flatnonzero(valid):
        u = units.iloc[int(ix[j])]
        split[j] = str(u["split"])
        unit_id[j] = str(u["analysis_unit_id"])
        unit_kind[j] = str(u["analysis_unit_kind"])
        behavior_id[j] = str(u["behavior_unit_id"])

    out = sample_map.copy()
    out = out.rename(columns={
        "capture_id": "raw_capture_id",
        "capture_path": "raw_capture_path",
        "capture_block_certified": "raw_capture_block_certified",
    })
    out["split"] = split
    out["analysis_unit_id"] = unit_id
    out["analysis_unit_kind"] = unit_kind
    out["behavior_unit_id"] = behavior_id
    out["behavior_unit_kind"] = unit_kind
    out["split_mode"] = split_mode
    out["y"] = (out["label"].astype(str) != str(cfg["normal"])).astype(np.int8)
    out["new_protocol_exposure"] = np.where(out["split"] == "DROP_EMBARGO", "EXCLUDED_EMBARGO", "USED_NEW_PROTOCOL")
    out["historical_exposure"] = "PREVIOUSLY_AVAILABLE_IN_LEGACY_EXPERIMENTS"
    # Backward-compatible internal alias used by legacy-shaped helper code. Never
    # interpret this field as a raw capture unless analysis_unit_kind says so.
    out["capture_id"] = out["analysis_unit_id"]
    out["capture_path"] = np.where(out["analysis_unit_kind"] == "CERTIFIED_RAW_CAPTURE_FILE", out["raw_capture_path"], "")
    out["capture_id_alias_semantics"] = "ANALYSIS_UNIT_ID_ALIAS"
    return out


def _validate_split(ds: str, cfg: dict[str, Any], manifest: pd.DataFrame, units: pd.DataFrame, split_mode: str) -> dict[str, Any]:
    used = manifest[manifest["split"] != "DROP_EMBARGO"].copy()
    dev = used[used["split"].isin(["fit_normal", "earlystop_normal", "calibration_normal"])]
    if len(dev) and int(dev["y"].sum()) != 0:
        raise RuntimeError(f"{ds}: attack row leaked into normal-only development partition")
    counts = used.groupby(["split", "y"]).size().rename("n").reset_index()
    def n(role: str, y: int) -> int:
        r = counts[(counts["split"] == role) & (counts["y"] == y)]
        return 0 if r.empty else int(r["n"].iloc[0])
    checks = {
        "fit_normal_n": n("fit_normal", 0),
        "earlystop_normal_n": n("earlystop_normal", 0),
        "calibration_normal_n": n("calibration_normal", 0),
        "test_normal_n": n("test", 0),
        "test_attack_n": n("test", 1),
        "embargo_dropped_n": int(np.sum(manifest["split"] == "DROP_EMBARGO")),
    }
    minimums = {
        "fit_normal_n": MIN_FIT_NORMAL, "earlystop_normal_n": MIN_EARLY_NORMAL,
        "calibration_normal_n": MIN_CAL_NORMAL, "test_normal_n": MIN_TEST_NORMAL,
        "test_attack_n": MIN_TEST_ATTACK,
    }
    failed = {k: (checks[k], v) for k, v in minimums.items() if checks[k] < v}
    if failed:
        raise RuntimeError(f"{ds}: insufficient materialized rows for new split: {failed}")
    if manifest["source_row_index"].duplicated().any():
        raise RuntimeError(f"{ds}: duplicate source_row_index in split manifest")
    if (used["analysis_unit_id"].astype(str).str.len() == 0).any():
        raise RuntimeError(f"{ds}: used row without analysis unit")

    role_units = {}
    for role in ["fit_normal", "earlystop_normal", "calibration_normal", "test"]:
        role_units[role] = sorted(used.loc[used["split"] == role, "analysis_unit_id"].astype(str).unique().tolist())
    # Analysis units must never cross roles regardless of their physical meaning.
    roles = list(role_units)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            overlap = set(role_units[a]) & set(role_units[b])
            if overlap:
                raise RuntimeError(f"{ds}: analysis-unit overlap {a}/{b}: {sorted(overlap)[:5]}")

    capture_mode = split_mode == "CERTIFIED_CAPTURE_FILE_DISJOINT_NORMAL_PARTITIONS"
    if capture_mode:
        rawsets = {
            role: set(used.loc[used["split"] == role, "raw_capture_id"].astype(str)) - {"", "UNMAPPED"}
            for role in roles
        }
        for i, a in enumerate(roles):
            for b in roles[i + 1:]:
                overlap = rawsets[a] & rawsets[b]
                if overlap:
                    raise RuntimeError(f"{ds}: raw-capture overlap in certified capture mode {a}/{b}: {sorted(overlap)[:5]}")

    return {
        "dataset": ds,
        "split_mode": split_mode,
        **checks,
        "used_rows": int(len(used)),
        "total_materialized_rows": int(len(manifest)),
        "analysis_unit_counts_by_role": {k: len(set(v)) for k, v in role_units.items()},
        "capture_file_disjoint_normal_partitions": capture_mode,
        "session_independence_certified": False,
        "analysis_unit_claim": (
            "verified raw capture file" if capture_mode else
            "fixed parent-row-order block; not capture/session/environment independence"
        ),
    }


def split_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str):
    if stage_is_complete(dirs, "split", protocol_hash):
        log(root, "[RESUME] split already complete")
        return
    if not stage_is_complete(dirs, "capture_map", protocol_hash):
        raise RuntimeError("capture_map must complete before split")
    log(root, "START split")
    inventory = pd.read_csv(dirs["metadata"] / "input_inventory.csv")
    summaries, unit_frames, manifest_frames = [], [], []
    semantics_rows = []
    for ds in DATASET_ORDER:
        cfg = DATASETS[ds]
        parent = Path(inventory.loc[inventory["dataset"] == ds, "parent_path"].iloc[0])
        ds_cap = dirs["capture"] / ds
        blocks = pd.read_csv(ds_cap / "10_capture_blocks.csv")
        sample_map = pd.read_csv(ds_cap / "11_sample_capture_map.csv.gz")
        if len(blocks):
            blocks["capture_block_certified"] = blocks["capture_block_certified"].map(as_bool)
        sample_map["capture_block_certified"] = sample_map["capture_block_certified"].map(as_bool)
        capture_ready = bool(
            len(blocks) > 0
            and np.all(sample_map["capture_block_certified"].to_numpy(dtype=bool))
            and np.all(sample_map["capture_id"].astype(str) != "UNMAPPED")
        )
        preferred_attempt = "complete_raw_capture_mapping" if capture_ready else "raw_capture_mapping_incomplete"

        if capture_ready:
            cert_blocks = blocks[blocks["capture_block_certified"]].copy().sort_values("start_row").reset_index(drop=True)
            units, reason = _whole_capture_partition(cert_blocks, sample_map, cfg["normal"])
            if units is None:
                preferred_attempt += f"; whole_capture_partition_infeasible:{reason}"
                units, split_mode = _blocked_within_capture_partition(cert_blocks, args.embargo_rows)
            else:
                split_mode = reason
            manifest = _map_samples_to_units(ds, cfg, sample_map, units, split_mode)
            try:
                summary = _validate_split(ds, cfg, manifest, units, split_mode)
            except RuntimeError as first_error:
                if split_mode == "CERTIFIED_CAPTURE_FILE_DISJOINT_NORMAL_PARTITIONS":
                    log(root, f"{ds}: whole-capture split count failure; fallback within certified captures. reason={first_error}")
                    units, split_mode = _blocked_within_capture_partition(cert_blocks, args.embargo_rows)
                    manifest = _map_samples_to_units(ds, cfg, sample_map, units, split_mode)
                    summary = _validate_split(ds, cfg, manifest, units, split_mode)
                    summary["preferred_capture_mode_fallback_reason"] = str(first_error)
                else:
                    raise
        else:
            log(root, f"{ds}: raw capture mapping incomplete; constructing fixed parent-order blocks ({args.fallback_block_rows:,} rows)")
            order_blocks = _summarize_parent_order_blocks(parent, cfg["label"], cfg["normal"], int(args.fallback_block_rows), root, ds)
            dsdir = dirs["splits"] / ds
            dsdir.mkdir(parents=True, exist_ok=True)
            order_blocks.to_csv(dsdir / "parent_order_blocks.csv", index=False)
            units, split_mode = _parent_order_partition(order_blocks, sample_map, cfg["normal"], args.embargo_rows)
            manifest = _map_samples_to_units(ds, cfg, sample_map, units, split_mode)
            summary = _validate_split(ds, cfg, manifest, units, split_mode)
            summary["raw_capture_mapping_incomplete"] = True
            summary["fallback_block_rows"] = int(args.fallback_block_rows)

        summary["preferred_capture_partition_attempt"] = preferred_attempt
        dsdir = dirs["splits"] / ds
        dsdir.mkdir(parents=True, exist_ok=True)
        units.to_csv(dsdir / "behavior_units.csv", index=False)
        manifest.to_csv(dsdir / "split_manifest.csv.gz", index=False, compression="gzip")
        save_json(dsdir / "split_summary.json", summary)
        exposure = manifest.groupby(["split", "historical_exposure", "new_protocol_exposure", "y"], dropna=False).size().rename("n").reset_index()
        exposure.to_csv(dsdir / "exposure_ledger.csv", index=False)
        summaries.append(summary)
        unit_frames.append(units)
        manifest_frames.append(manifest)
        semantics_rows.append({
            "dataset": ds,
            "split_mode": split_mode,
            "analysis_unit_kind": "CERTIFIED_RAW_CAPTURE_FILE" if split_mode == "CERTIFIED_CAPTURE_FILE_DISJOINT_NORMAL_PARTITIONS" else (
                "CERTIFIED_CAPTURE_SUBBLOCK" if split_mode == "BLOCKED_NORMAL_WITHIN_CERTIFIED_CAPTURES" else "FIXED_PARENT_ORDER_BLOCK"
            ),
            "capture_independence_claim_allowed": split_mode == "CERTIFIED_CAPTURE_FILE_DISJOINT_NORMAL_PARTITIONS",
            "session_independence_claim_allowed": False,
            "note": "capture_id in downstream legacy-shaped artifacts is an internal alias of analysis_unit_id; use analysis_unit_kind for interpretation",
        })
        log(root, f"{ds}: split={split_mode} fit={summary['fit_normal_n']:,} early={summary['earlystop_normal_n']:,} cal={summary['calibration_normal_n']:,} testN={summary['test_normal_n']:,} testA={summary['test_attack_n']:,}")

    pd.DataFrame(summaries).to_csv(dirs["splits"] / "split_summary.csv", index=False)
    pd.DataFrame(semantics_rows).to_csv(dirs["splits"] / "analysis_unit_semantics.csv", index=False)
    pd.concat(unit_frames, ignore_index=True).to_csv(dirs["splits"] / "behavior_units_all.csv", index=False)
    pd.concat(manifest_frames, ignore_index=True).to_csv(dirs["splits"] / "split_manifest_all.csv.gz", index=False, compression="gzip")
    mark_stage(dirs, "split", protocol_hash, {"dataset_summaries": summaries, "analysis_unit_semantics": semantics_rows})
    log(root, "END split [COMPLETED]")


# =============================================================================
# CAUSAL FLOW-LOCAL BEHAVIOR RECONSTRUCTION
# =============================================================================

def iter_parquet_range_batches(path: Path, start: int, end: int, columns: list[str], batch_size: int = 250_000):
    """Yield (global_start, DataFrame) chunks intersecting inclusive [start,end]."""
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in available]
    rg_start = 0
    for rg in range(pf.num_row_groups):
        n = int(pf.metadata.row_group(rg).num_rows)
        rg_end = rg_start + n - 1
        if rg_end < start:
            rg_start += n
            continue
        if rg_start > end:
            break
        lo = max(start, rg_start)
        hi = min(end, rg_end)
        tbl = pf.read_row_group(rg, columns=cols)
        local_lo = lo - rg_start
        local_n = hi - lo + 1
        sub = tbl.slice(local_lo, local_n)
        for off in range(0, local_n, batch_size):
            piece = sub.slice(off, min(batch_size, local_n - off)).to_pandas()
            yield lo + off, piece
        rg_start += n


def _rolling_behavior_frame(frame: pd.DataFrame, window: int = BEHAVIOR_WINDOW) -> pd.DataFrame:
    """Causal flow-local behavior with explicit within-flow episode resets.

    The reset rule is label-free and uses only already-observed metadata:
    - packet_index non-increase within the same flow_id, or
    - timestamp backstep within the same flow_id.
    This prevents a reused flow_id from silently carrying rolling state across an
    obvious new packet-index episode.  Missing packet_index falls back to the
    timestamp-only reset rule.
    """
    if frame.empty:
        return frame.copy()
    x = frame.sort_values("__parent_pos").copy()
    x["flow_id"] = x["flow_id"].astype(str)
    x["__t"] = pd.to_numeric(x["frame.time_epoch"], errors="coerce")
    x["__len"] = pd.to_numeric(x["frame.len"], errors="coerce")
    if x[["__t", "__len"]].isna().any().any():
        raise RuntimeError("Non-numeric frame.time_epoch/frame.len while building behavior features")
    flow_g = x.groupby("flow_id", sort=False, group_keys=False)
    raw_flow_iat = flow_g["__t"].diff()
    reset = (raw_flow_iat < 0).fillna(False)
    if "packet_index" in x.columns:
        pi = pd.to_numeric(x["packet_index"], errors="coerce")
        dpi = pi.groupby(x["flow_id"], sort=False).diff()
        reset = reset | ((dpi <= 0) & dpi.notna())
        x["__pi"] = pi
    x["__negative_flow_iat"] = (raw_flow_iat < 0).fillna(False)
    x["__segment_reset"] = reset.astype(bool)
    segno = reset.groupby(x["flow_id"], sort=False).cumsum().astype(np.int64)
    x["__segment_key"] = x["flow_id"].astype(str) + "::seg" + segno.astype(str)
    g = x.groupby("__segment_key", sort=False, group_keys=False)
    raw_iat = g["__t"].diff()
    iat = raw_iat.clip(lower=0).fillna(0.0)
    x["__iat"] = iat

    def grow(series: pd.Series, fn: str, **kwargs) -> pd.Series:
        rr = series.groupby(x["__segment_key"], sort=False).rolling(window, min_periods=1)
        z = getattr(rr, fn)(**kwargs)
        z.index = z.index.droplevel(0)
        return z.reindex(x.index)

    mean_iat = grow(x["__iat"], "mean")
    std_iat = grow(x["__iat"], "std", ddof=0).fillna(0.0)
    sum_iat = grow(x["__iat"], "sum")
    sum_len = grow(x["__len"], "sum")
    mean_len = grow(x["__len"], "mean")
    x["__valid_interval"] = (g.cumcount() > 0).astype(float)
    interval_count = grow(x["__valid_interval"], "sum").to_numpy(dtype=float)
    duration = sum_iat.to_numpy(dtype=float)
    valid_rate = (interval_count > 0) & (duration > 0)
    denom = np.maximum(duration, RATE_DENOM_FLOOR_SECONDS)
    packet_rate = np.zeros(len(x), dtype=float)
    byte_rate = np.zeros(len(x), dtype=float)
    packet_rate[valid_rate] = interval_count[valid_rate] / denom[valid_rate]
    byte_rate[valid_rate] = sum_len.to_numpy(dtype=float)[valid_rate] / denom[valid_rate]
    cv = np.divide(
        std_iat.to_numpy(dtype=float), mean_iat.to_numpy(dtype=float),
        out=np.zeros(len(x), dtype=float), where=mean_iat.to_numpy(dtype=float) > 0,
    )
    x["b_prev_iat_log1p"] = np.log1p(x["__iat"].to_numpy(dtype=float))
    x["b_mean_iat8_log1p"] = np.log1p(mean_iat.to_numpy(dtype=float))
    x["b_std_iat8_log1p"] = np.log1p(std_iat.to_numpy(dtype=float))
    x["b_packet_rate8_log1p"] = np.log1p(packet_rate)
    x["b_byte_rate8_log1p"] = np.log1p(byte_rate)
    x["b_mean_len8_log1p"] = np.log1p(np.maximum(mean_len.to_numpy(dtype=float), 0.0))
    x["b_cv_iat8_log1p"] = np.log1p(np.maximum(cv, 0.0))
    return x


def _translation_invariance_unit_test() -> dict[str, Any]:
    fixture = pd.DataFrame({
        "__parent_pos": np.arange(10, dtype=np.int64),
        "flow_id": ["a", "a", "b", "a", "b", "b", "a", "a", "b", "a"],
        "packet_index": [1, 2, 1, 3, 2, 3, 4, 5, 4, 6],
        "frame.time_epoch": [100.0, 100.1, 101.0, 100.25, 101.2, 101.25, 100.40, 100.41, 101.7, 100.9],
        "frame.len": [60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    })
    a = _rolling_behavior_frame(fixture)
    shifted = fixture.copy()
    shifted["frame.time_epoch"] = shifted["frame.time_epoch"] + 123456.789
    b = _rolling_behavior_frame(shifted)
    diffs = np.abs(a[BEHAVIOR_FEATURES].to_numpy(dtype=float) - b[BEHAVIOR_FEATURES].to_numpy(dtype=float))
    max_diff = float(np.max(diffs)) if diffs.size else 0.0
    return {"max_abs_difference": max_diff, "pass": bool(max_diff <= 1e-8), "tolerance": 1e-8}


def _chunk_continuity_unit_test() -> dict[str, Any]:
    rows = []
    counters = {"a": 0, "b": 0, "c": 0}
    for i in range(60):
        f = ["a", "b", "c"][i % 3]
        counters[f] += 1
        t = 1000.0 + i * 0.013 + {"a": 0.0, "b": 0.0002, "c": 0.0004}[f]
        rows.append((i, f, counters[f], t, 60 + (i % 100)))
    fixture = pd.DataFrame(rows, columns=["__parent_pos", "flow_id", "packet_index", "frame.time_epoch", "frame.len"])
    whole = _rolling_behavior_frame(fixture)[["__parent_pos"] + BEHAVIOR_FEATURES].sort_values("__parent_pos")
    state: dict[str, pd.DataFrame] = {}
    parts = []
    for lo, hi in [(0, 17), (17, 35), (35, 60)]:
        cur = fixture.iloc[lo:hi].copy()
        present = cur["flow_id"].astype(str).unique().tolist()
        prepend = [state[f] for f in present if f in state and not state[f].empty]
        work = pd.concat(prepend + [cur], ignore_index=True) if prepend else cur.reset_index(drop=True)
        feat = _rolling_behavior_frame(work)
        parts.append(feat[feat["__parent_pos"].isin(cur["__parent_pos"])][["__parent_pos"] + BEHAVIOR_FEATURES])
        for f, gf in work.groupby("flow_id", sort=False):
            state[str(f)] = gf[["__parent_pos", "flow_id", "packet_index", "frame.time_epoch", "frame.len"]].tail(BEHAVIOR_WINDOW).copy()
    streamed = pd.concat(parts, ignore_index=True).sort_values("__parent_pos")
    a = whole[BEHAVIOR_FEATURES].to_numpy(dtype=float)
    b = streamed[BEHAVIOR_FEATURES].to_numpy(dtype=float)
    max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    tolerance = 1e-8
    return {"max_abs_difference": max_diff, "pass": bool(max_diff <= tolerance), "tolerance": tolerance}


def _packet_index_reset_unit_test() -> dict[str, Any]:
    fixture = pd.DataFrame({
        "__parent_pos": np.arange(6, dtype=np.int64),
        "flow_id": ["x"] * 6,
        "packet_index": [1, 2, 3, 1, 2, 3],
        "frame.time_epoch": [10.0, 10.1, 10.2, 20.0, 20.1, 20.2],
        "frame.len": [60, 60, 60, 60, 60, 60],
    })
    feat = _rolling_behavior_frame(fixture)
    # At packet_index reset, previous-IAT must reset to zero rather than encode the 9.8s cross-episode gap.
    observed = float(feat.loc[feat["__parent_pos"] == 3, "b_prev_iat_log1p"].iloc[0])
    resets = int(feat["__segment_reset"].sum())
    return {"reset_row_prev_iat_log1p": observed, "segment_reset_n": resets, "pass": bool(abs(observed) <= 1e-12 and resets >= 1)}


def _extract_unit_behavior(parent: Path, unit: pd.Series, target_rows: pd.DataFrame, root: Path, ds: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_row, end_row = int(unit["unit_start_row"]), int(unit["unit_end_row"])
    if target_rows.empty:
        return pd.DataFrame(), {"negative_flow_iat_n": 0, "segment_reset_n": 0, "target_flow_n": 0, "scanned_unit_rows": end_row - start_row + 1}
    target_pos = set(target_rows["source_row_index"].astype(np.int64).tolist())
    target_flows = set(target_rows["flow_id"].astype(str).tolist())
    if not target_flows:
        raise RuntimeError(f"{ds}: behavior unit {unit['behavior_unit_id']} has no target flows")
    state: dict[str, pd.DataFrame] = {}
    out_parts = []
    neg_n = 0
    reset_n = 0
    scanned = 0
    for gstart, chunk in iter_parquet_range_batches(parent, start_row, end_row, ["flow_id", "packet_index", "frame.time_epoch", "frame.len"], batch_size=250_000):
        scanned += len(chunk)
        chunk.insert(0, "__parent_pos", np.arange(gstart, gstart + len(chunk), dtype=np.int64))
        if "flow_id" not in chunk.columns:
            raise RuntimeError(f"{ds}: parent lacks flow_id")
        chunk["flow_id"] = chunk["flow_id"].astype("string")
        if chunk["flow_id"].isna().any():
            raise RuntimeError(f"{ds}: missing flow_id in behavior source interval")
        cur = chunk[chunk["flow_id"].astype(str).isin(target_flows)].copy()
        if cur.empty:
            continue
        present = cur["flow_id"].astype(str).unique().tolist()
        prepend = [state[f] for f in present if f in state and not state[f].empty]
        work = pd.concat(prepend + [cur], ignore_index=True) if prepend else cur.reset_index(drop=True)
        feat = _rolling_behavior_frame(work)
        current_mask = feat["__parent_pos"].isin(cur["__parent_pos"].to_numpy(dtype=np.int64))
        neg_n += int(feat.loc[current_mask, "__negative_flow_iat"].sum())
        reset_n += int(feat.loc[current_mask, "__segment_reset"].sum())
        emit = feat[feat["__parent_pos"].isin(target_pos)].copy()
        if not emit.empty:
            out_parts.append(emit[["__parent_pos", "flow_id"] + BEHAVIOR_FEATURES])
        keep_cols = ["__parent_pos", "flow_id", "frame.time_epoch", "frame.len"]
        if "packet_index" in work.columns:
            keep_cols.insert(2, "packet_index")
        for f, gf in work.groupby("flow_id", sort=False):
            state[str(f)] = gf[keep_cols].tail(BEHAVIOR_WINDOW).copy()
        del chunk, cur, work, feat
    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["__parent_pos", "flow_id"] + BEHAVIOR_FEATURES)
    out = out.drop_duplicates("__parent_pos", keep="last")
    missing = target_pos - set(out["__parent_pos"].astype(np.int64).tolist())
    if missing:
        raise RuntimeError(f"{ds}: {len(missing)} target rows missing causal behavior features in {unit['behavior_unit_id']}")
    return out, {
        "negative_flow_iat_n": int(neg_n),
        "segment_reset_n": int(reset_n),
        "target_flow_n": int(len(target_flows)),
        "target_row_n": int(len(target_pos)),
        "scanned_unit_rows": int(scanned),
    }


def behavior_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, legacy, q_features: list[str]):
    if stage_is_complete(dirs, "behavior", protocol_hash):
        log(root, "[RESUME] behavior already complete")
        return
    if not stage_is_complete(dirs, "split", protocol_hash):
        raise RuntimeError("split must complete before behavior")
    log(root, "START behavior")
    translation_test = _translation_invariance_unit_test()
    continuity_test = _chunk_continuity_unit_test()
    reset_test = _packet_index_reset_unit_test()
    save_json(dirs["behavior"] / "translation_invariance_unit_test.json", translation_test)
    save_json(dirs["behavior"] / "streaming_chunk_continuity_unit_test.json", continuity_test)
    save_json(dirs["behavior"] / "packet_index_reset_unit_test.json", reset_test)
    if not translation_test["pass"]:
        raise RuntimeError(f"Behavior feature translation-invariance unit test failed: {translation_test}")
    if not continuity_test["pass"]:
        raise RuntimeError(f"Behavior feature streaming continuity unit test failed: {continuity_test}")
    if not reset_test["pass"]:
        raise RuntimeError(f"Behavior packet-index reset unit test failed: {reset_test}")

    input_inventory = pd.read_csv(dirs["metadata"] / "input_inventory.csv")
    all_qc = []
    feature_contract = {
        "q_features": q_features,
        "r_feature": "inter_arrival_time",
        "t_feature": "frame.time_epoch",
        "b_features": BEHAVIOR_FEATURES,
        "legacy_33": list(legacy.FEATURES_33),
        "legacy_34": list(legacy.FEATURES_34),
    }
    save_json(dirs["metadata"] / "feature_contract.json", feature_contract)

    for ds in DATASET_ORDER:
        cfg = DATASETS[ds]
        parent = Path(input_inventory.loc[input_inventory["dataset"] == ds, "parent_path"].iloc[0])
        manifest = pd.read_csv(dirs["splits"] / ds / "split_manifest.csv.gz")
        units = pd.read_csv(dirs["splits"] / ds / "behavior_units.csv")

        # Retrieve flow_id exactly at each materialized source row; flow_id is metadata only.
        pos = manifest["source_row_index"].to_numpy(dtype=np.int64)
        flow_at_sample = sparse_parent_take(parent, pos, ["flow_id"])
        if len(flow_at_sample) != len(manifest):
            raise RuntimeError(f"{ds}: failed to retrieve flow_id for all materialized rows")
        fmap = dict(zip(flow_at_sample["__parent_pos"].astype(np.int64), flow_at_sample["flow_id"].astype(str)))
        manifest["flow_id"] = manifest["source_row_index"].map(fmap)
        if manifest["flow_id"].isna().any():
            raise RuntimeError(f"{ds}: missing sample flow_id after sparse parent take")

        used_targets = manifest[manifest["split"] != "DROP_EMBARGO"][[
            "source_row_index", "behavior_unit_id", "flow_id"
        ]].copy()
        behavior_parts = []
        unit_qc = []
        for i, unit in units.iterrows():
            targets = used_targets[used_targets["behavior_unit_id"].astype(str) == str(unit["behavior_unit_id"])]
            if targets.empty:
                continue
            log(root, f"{ds}: behavior unit {i+1}/{len(units)} target_rows={len(targets):,} {unit['behavior_unit_id']}")
            bdf, qc = _extract_unit_behavior(parent, unit, targets, root, ds)
            bdf["behavior_unit_id"] = str(unit["behavior_unit_id"])
            behavior_parts.append(bdf)
            qc.update({"dataset": ds, "behavior_unit_id": unit["behavior_unit_id"], "split": unit["split"]})
            unit_qc.append(qc)

        behavior = pd.concat(behavior_parts, ignore_index=True) if behavior_parts else pd.DataFrame()
        if len(behavior) != len(used_targets):
            raise RuntimeError(f"{ds}: behavior rows {len(behavior)} != used target rows {len(used_targets)}")
        if behavior["__parent_pos"].duplicated().any():
            raise RuntimeError(f"{ds}: duplicate behavior row output")
        behavior.to_parquet(dirs["behavior"] / f"{ds}_causal_behavior.parquet", index=False)
        pd.DataFrame(unit_qc).to_csv(dirs["behavior"] / f"{ds}_behavior_unit_qc.csv", index=False)

        needed_sample_cols = list(dict.fromkeys(list(legacy.FEATURES_34) + [cfg["label"], "label_type", "source_row_index"]))
        sample = pd.read_parquet(cfg["sample"], columns=needed_sample_cols).copy()
        sample.insert(0, "sample_local_index", np.arange(len(sample), dtype=np.int64))
        meta_cols = manifest[[
            "sample_local_index", "source_row_index", "analysis_unit_id", "analysis_unit_kind",
            "raw_capture_id", "raw_capture_path", "raw_capture_block_certified", "capture_id", "capture_path",
            "split", "behavior_unit_id", "behavior_unit_kind", "split_mode", "y",
            "historical_exposure", "new_protocol_exposure", "flow_id"
        ]].copy()
        prepared = sample.merge(meta_cols, on=["sample_local_index", "source_row_index"], how="left", validate="one_to_one")
        bmerge = behavior.rename(columns={"__parent_pos": "source_row_index"})[["source_row_index"] + BEHAVIOR_FEATURES]
        prepared = prepared.merge(bmerge, on="source_row_index", how="left", validate="one_to_one")

        sanitation_rows = []
        numeric_features = list(dict.fromkeys(q_features + ["inter_arrival_time", "frame.time_epoch"] + BEHAVIOR_FEATURES))
        for feat in numeric_features:
            raw = pd.to_numeric(prepared[feat], errors="coerce")
            invalid = int((~np.isfinite(raw.to_numpy(dtype=float))).sum())
            missing = int(raw.isna().sum())
            is_b = feat in BEHAVIOR_FEATURES
            is_t = feat == "frame.time_epoch"
            # b is only required for rows participating in the new protocol; embargo rows may legitimately be NaN.
            required_mask = prepared["split"].astype(str) != "DROP_EMBARGO"
            required_invalid = int((~np.isfinite(raw[required_mask].to_numpy(dtype=float))).sum())
            if is_t and invalid:
                raise RuntimeError(f"{ds}: frame.time_epoch has {invalid} invalid values; refusing timestamp imputation")
            if is_b and required_invalid:
                raise RuntimeError(f"{ds}: behavior feature {feat} missing/nonfinite on {required_invalid} used rows")
            if not is_b and not is_t:
                # Preserve historical numeric sanitation semantics for non-timestamp packet features, but audit it explicitly.
                raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            prepared[feat] = raw
            vals = prepared.loc[required_mask, feat]
            sanitation_rows.append({
                "dataset": ds, "feature": feat, "invalid_all_rows_before": invalid, "missing_all_rows_before": missing,
                "invalid_used_rows_before": required_invalid,
                "filled_with_zero": (not is_b and not is_t),
                "used_min": float(vals.min()) if len(vals) and np.isfinite(vals).any() else np.nan,
                "used_max": float(vals.max()) if len(vals) and np.isfinite(vals).any() else np.nan,
                "used_unique": int(vals.nunique(dropna=True)),
            })

        # Strong postconditions on all used feature rows.
        used = prepared[prepared["split"] != "DROP_EMBARGO"]
        for feat in numeric_features:
            if not np.all(np.isfinite(pd.to_numeric(used[feat], errors="coerce").to_numpy(dtype=float))):
                raise RuntimeError(f"{ds}: non-finite used value remains in {feat}")
        prepared_path = dirs["prepared"] / f"{ds}_prepared.parquet"
        prepared.to_parquet(prepared_path, index=False)
        pd.DataFrame(sanitation_rows).to_csv(dirs["prepared"] / f"{ds}_sanitation.csv", index=False)
        prepared_summary = {
            "dataset": ds,
            "prepared_path": str(prepared_path),
            "prepared_sha256": sha256_file(prepared_path),
            "rows": len(prepared),
            "used_rows": int(np.sum(prepared["split"] != "DROP_EMBARGO")),
            "q_features": len(q_features),
            "behavior_features": len(BEHAVIOR_FEATURES),
            "negative_flow_iat_total": int(pd.DataFrame(unit_qc)["negative_flow_iat_n"].sum()) if unit_qc else 0,
            "flow_segment_reset_total": int(pd.DataFrame(unit_qc)["segment_reset_n"].sum()) if unit_qc else 0,
            "translation_invariance_test": translation_test,
            "streaming_chunk_continuity_test": continuity_test,
            "packet_index_reset_test": reset_test,
        }
        save_json(dirs["prepared"] / f"{ds}_prepared_summary.json", prepared_summary)
        all_qc.extend(unit_qc)
        log(root, f"{ds}: prepared new-protocol data rows={len(prepared):,} used={prepared_summary['used_rows']:,}")

    pd.DataFrame(all_qc).to_csv(dirs["behavior"] / "behavior_unit_qc_all.csv", index=False)
    mark_stage(dirs, "behavior", protocol_hash, {"translation_invariance": translation_test, "streaming_chunk_continuity": continuity_test, "packet_index_reset": reset_test})
    log(root, "END behavior [COMPLETED]")


# =============================================================================
# PREPROCESSING / MODEL HELPERS
# =============================================================================

def load_prepared(dirs: dict[str, Path], ds: str) -> pd.DataFrame:
    p = dirs["prepared"] / f"{ds}_prepared.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def condition_raw_matrix(df: pd.DataFrame, features: list[str], preprocess: str) -> np.ndarray:
    x = df[features].to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(x)):
        bad = np.argwhere(~np.isfinite(x))
        raise RuntimeError(f"Non-finite raw matrix values; first bad positions={bad[:10].tolist()}")
    if preprocess == "epoch_float32_precast_only":
        if "frame.time_epoch" not in features:
            raise RuntimeError("epoch_float32_precast_only condition lacks frame.time_epoch")
        j = features.index("frame.time_epoch")
        x[:, j] = x[:, j].astype(np.float32).astype(np.float64)
    elif preprocess != "precision_safe":
        raise ValueError(preprocess)
    return x


def scaler_dir(dirs: dict[str, Path], ds: str, condition: str) -> Path:
    p = dirs["metadata"] / "scalers" / ds / condition
    p.mkdir(parents=True, exist_ok=True)
    return p


def fit_or_load_scaler(
    dirs: dict[str, Path], ds: str, condition: str, spec: dict[str, Any], prepared: pd.DataFrame,
    protocol_hash: str,
) -> MinMaxScaler:
    sdir = scaler_dir(dirs, ds, condition)
    sp = sdir / "scaler.pkl"
    mp = sdir / "scaler_manifest.json"
    fit = prepared[prepared["split"] == "fit_normal"]
    prepared_hash = sha256_file(dirs["prepared"] / f"{ds}_prepared.parquet")
    expected_meta = {
        "protocol_hash": protocol_hash,
        "dataset": ds,
        "condition": condition,
        "features": list(spec["features"]),
        "preprocess": spec["preprocess"],
        "fit_rows": int(len(fit)),
        "prepared_sha256": prepared_hash,
    }
    if sp.exists() and mp.exists():
        meta = load_json(mp)
        if all(meta.get(k) == v for k, v in expected_meta.items()):
            scaler = joblib.load(sp)
            return scaler
        # Never silently reuse stale preprocessing.
        stale = sdir.with_name(sdir.name + f"__STALE_{utc_tag()}")
        sdir.rename(stale)
        sdir.mkdir(parents=True, exist_ok=True)
        sp = sdir / "scaler.pkl"
        mp = sdir / "scaler_manifest.json"

    xfit = condition_raw_matrix(fit, list(spec["features"]), spec["preprocess"])
    scaler = MinMaxScaler()
    scaler.fit(xfit)
    joblib.dump(scaler, sp)
    meta = dict(expected_meta)
    meta.update({
        "scaler_sha256": sha256_file(sp),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    })
    save_json(mp, meta)
    return scaler


def transform_partition(
    prepared: pd.DataFrame, split: str, spec: dict[str, Any], scaler: MinMaxScaler
) -> tuple[pd.DataFrame, np.ndarray]:
    part = prepared[prepared["split"] == split].copy().reset_index(drop=True)
    raw = condition_raw_matrix(part, list(spec["features"]), spec["preprocess"])
    scaled64 = scaler.transform(raw)
    if not np.all(np.isfinite(scaled64)):
        raise RuntimeError(f"Non-finite transformed values in split={split}")
    return part, scaled64.astype(np.float32)


def q_indices(features: list[str], q_features: list[str]) -> np.ndarray:
    missing = [q for q in q_features if q not in features]
    if missing:
        raise RuntimeError(f"Condition missing q features: {missing}")
    return np.asarray([features.index(q) for q in q_features], dtype=np.int64)


def build_model(legacy, model_name: str, input_dim: int, ds: str, seed: int, compile_model: bool = True):
    set_seed(seed)
    tf.keras.backend.clear_session()
    model = legacy.audit.build_model(model_name, input_dim, ds)
    if compile_model:
        optimizer = Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0 if model_name in {"SNN_AE", "Dense_AE"} else None,
        )
        model.compile(optimizer=optimizer, loss="mse")
    return model


def predict_flat(legacy, model, x: np.ndarray, model_name: str) -> np.ndarray:
    y = legacy.audit.predict_flat(model, x, model_name)
    y = np.asarray(y, dtype=np.float32)
    if y.shape != x.shape:
        raise RuntimeError(f"Prediction shape {y.shape} != input shape {x.shape} for {model_name}")
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"Non-finite model prediction for {model_name}")
    return y


def reconstruction_scores(x: np.ndarray, pred: np.ndarray, qi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sq = np.square(x.astype(np.float64) - pred.astype(np.float64))
    qres = sq[:, qi]
    qscore = np.mean(qres, axis=1)
    fullscore = np.mean(sq, axis=1)
    return qscore.astype(np.float64), fullscore.astype(np.float64), qres.astype(np.float32)


def shared_scaler_consistency(dirs: dict[str, Path], ds: str, specs: dict[str, dict[str, Any]], q_features: list[str]) -> pd.DataFrame:
    rows = []
    base_cond = "B_q"
    base = joblib.load(scaler_dir(dirs, ds, base_cond) / "scaler.pkl")
    bfeatures = list(specs[base_cond]["features"])
    for cond in [c for c in SAFE_CONDITION_NAMES if c in specs]:
        sc = joblib.load(scaler_dir(dirs, ds, cond) / "scaler.pkl")
        feats = list(specs[cond]["features"])
        for q in q_features:
            ib = bfeatures.index(q)
            ic = feats.index(q)
            rows.append({
                "dataset": ds,
                "condition": cond,
                "feature": q,
                "abs_data_min_diff_vs_B": abs(float(sc.data_min_[ic]) - float(base.data_min_[ib])),
                "abs_data_max_diff_vs_B": abs(float(sc.data_max_[ic]) - float(base.data_max_[ib])),
                "abs_scale_diff_vs_B": abs(float(sc.scale_[ic]) - float(base.scale_[ib])),
                "abs_min_param_diff_vs_B": abs(float(sc.min_[ic]) - float(base.min_[ib])),
            })
    return pd.DataFrame(rows)


def run_artifacts_valid(run_dir: Path, protocol_hash: str) -> bool:
    marker = run_dir / "RUN_COMPLETE.json"
    if not marker.exists():
        return False
    try:
        obj = load_json(marker)
        if obj.get("status") != "COMPLETED" or obj.get("protocol_hash") != protocol_hash:
            return False
        for rel, expected in obj.get("artifact_sha256", {}).items():
            p = run_dir / rel
            if not p.exists() or sha256_file(p) != expected:
                return False
        return True
    except Exception:
        return False


def archive_invalid_run(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    dst = run_dir.with_name(run_dir.name + f"__ARCHIVED_INVALID_{utc_tag()}")
    run_dir.rename(dst)


# =============================================================================
# TECHNICAL PILOT (DEVELOPMENT NORMAL ONLY)
# =============================================================================

def pilot_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, legacy, q_features: list[str], specs: dict[str, dict[str, Any]]):
    if stage_is_complete(dirs, "pilot", protocol_hash):
        log(root, "[RESUME] pilot already complete")
        return
    if not stage_is_complete(dirs, "behavior", protocol_hash):
        raise RuntimeError("behavior must complete before pilot")
    log(root, "START pilot")
    ds = "edge_iiotset"
    condition = "C_q_b_t"
    spec = specs[condition]
    prepared = load_prepared(dirs, ds)
    fit = prepared[prepared["split"] == "fit_normal"].head(512).copy()
    early = prepared[prepared["split"] == "earlystop_normal"].head(256).copy()
    if len(fit) < 64 or len(early) < 32:
        raise RuntimeError("Technical pilot lacks normal-only development rows")
    scaler = MinMaxScaler().fit(condition_raw_matrix(fit, spec["features"], spec["preprocess"]))
    xfit = scaler.transform(condition_raw_matrix(fit, spec["features"], spec["preprocess"])).astype(np.float32)
    xearly = scaler.transform(condition_raw_matrix(early, spec["features"], spec["preprocess"])).astype(np.float32)
    qi = q_indices(spec["features"], q_features)
    rows = []
    for mi, model_name in enumerate(args.models):
        seed = model_seed(0, SAFE_CONDITION_NAMES.index(condition), mi, 991)
        mdir = dirs["pilot"] / model_name
        mdir.mkdir(parents=True, exist_ok=True)
        model = build_model(legacy, model_name, xfit.shape[1], ds, seed, compile_model=True)
        hist = model.fit(
            legacy.audit.input_for_model(xfit, model_name),
            legacy.audit.target_for_model(xfit, model_name),
            validation_data=(legacy.audit.input_for_model(xearly, model_name), legacy.audit.target_for_model(xearly, model_name)),
            epochs=2,
            batch_size=min(BATCH_SIZE, len(xfit)),
            verbose=0,
        )
        pred1 = predict_flat(legacy, model, xearly, model_name)
        q1, _, _ = reconstruction_scores(xearly, pred1, qi)
        weights = mdir / "pilot.weights.h5"
        model.save_weights(weights)
        del model
        tf.keras.backend.clear_session()
        model2 = build_model(legacy, model_name, xfit.shape[1], ds, seed + 1, compile_model=False)
        model2.load_weights(weights)
        pred2 = predict_flat(legacy, model2, xearly, model_name)
        q2, _, _ = reconstruction_scores(xearly, pred2, qi)
        max_pred_diff = float(np.max(np.abs(pred1.astype(float) - pred2.astype(float))))
        max_score_diff = float(np.max(np.abs(q1 - q2)))
        passed = max_pred_diff <= 1e-6 and max_score_diff <= 1e-8
        rows.append({
            "model": model_name,
            "development_normal_only": True,
            "epochs": len(hist.history.get("loss", [])),
            "input_dim": xfit.shape[1],
            "q_dim": len(qi),
            "reload_max_prediction_abs_diff": max_pred_diff,
            "reload_max_q_score_abs_diff": max_score_diff,
            "passed": passed,
        })
        if not passed:
            raise RuntimeError(f"Pilot save/reload failed for {model_name}: pred={max_pred_diff} q={max_score_diff}")
        del model2, pred1, pred2, q1, q2
        tf.keras.backend.clear_session()
        gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(dirs["pilot"] / "pilot_results.csv", index=False)
    save_json(dirs["pilot"] / "RUN_COMPLETE.json", {
        "status": "COMPLETED", "protocol_hash": protocol_hash,
        "scope": "technical QA only; Edge development-normal rows; no attack-test labels used",
        "models": rows,
    })
    mark_stage(dirs, "pilot", protocol_hash, {"models_passed": len(rows)})
    log(root, "END pilot [COMPLETED]")


# =============================================================================
# BASELINES
# =============================================================================

def baseline_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, q_features: list[str], specs: dict[str, dict[str, Any]]):
    if stage_is_complete(dirs, "baselines", protocol_hash):
        log(root, "[RESUME] baselines already complete")
        return
    if not stage_is_complete(dirs, "pilot", protocol_hash):
        raise RuntimeError("pilot must complete before baselines")
    log(root, "START baselines")
    metric_rows = []
    for ds in DATASET_ORDER:
        prepared = load_prepared(dirs, ds)
        fit = prepared[prepared["split"] == "fit_normal"].copy()
        cal = prepared[prepared["split"] == "calibration_normal"].copy()
        test = prepared[prepared["split"] == "test"].copy().reset_index(drop=True)
        ytest = test["y"].to_numpy(dtype=np.int8)

        # Sanity baseline: a detector that never alarms is temporally invariant but useless for attacks.
        sanity_score = np.zeros(len(test), dtype=float)
        sanity_threshold = 1.0
        m = metric_dict(ytest, sanity_score, sanity_threshold)
        metric_rows.append({"dataset": ds, "baseline": "always_normal_sanity", "condition": "NA", "threshold": sanity_threshold, **m})

        # Timestamp-only interval distance, normalized by fit-normal span.
        fit_t = fit["frame.time_epoch"].to_numpy(dtype=np.float64)
        cal_t = cal["frame.time_epoch"].to_numpy(dtype=np.float64)
        test_t = test["frame.time_epoch"].to_numpy(dtype=np.float64)
        tmin, tmax = float(np.min(fit_t)), float(np.max(fit_t))
        span = tmax - tmin
        if not np.isfinite(span) or span <= 0:
            raise RuntimeError(f"{ds}: zero/nonfinite fit-normal epoch span")
        def tdist(x):
            return np.maximum(np.maximum(tmin - x, 0.0), np.maximum(x - tmax, 0.0)) / span
        cal_score = tdist(cal_t)
        test_score = tdist(test_t)
        th = percentile_threshold(cal_score)
        m = metric_dict(ytest, test_score, th)
        metric_rows.append({"dataset": ds, "baseline": "timestamp_fit_interval_distance", "condition": "timestamp_only", "threshold": th, "fit_epoch_min": tmin, "fit_epoch_max": tmax, **m})
        tpred = (test_score > th).astype(np.int8)
        pd.DataFrame({
            "source_row_index": test["source_row_index"].astype(np.int64),
            "analysis_unit_id": test["analysis_unit_id"].astype(str),
            "analysis_unit_kind": test["analysis_unit_kind"].astype(str),
            "capture_id": test["capture_id"].astype(str),
            "label_type": test["label_type"].astype(str),
            "y_true": ytest,
            "score": test_score,
            "prediction": tpred,
        }).to_csv(dirs["baselines"] / f"{ds}_timestamp_only_predictions.csv.gz", index=False, compression="gzip")

        # PCA linear reconstruction baseline under each safe representation. Same q-only score contract as neural models.
        for condition in SAFE_CONDITION_NAMES:
            spec = specs[condition]
            scaler = fit_or_load_scaler(dirs, ds, condition, spec, prepared, protocol_hash)
            fit_part, xfit = transform_partition(prepared, "fit_normal", spec, scaler)
            cal_part, xcal = transform_partition(prepared, "calibration_normal", spec, scaler)
            test_part, xtest = transform_partition(prepared, "test", spec, scaler)
            qi = q_indices(spec["features"], q_features)
            pca = PCA(n_components=0.95, svd_solver="full")
            pca.fit(xfit.astype(np.float64))
            cal_rec = pca.inverse_transform(pca.transform(xcal.astype(np.float64)))
            test_rec = pca.inverse_transform(pca.transform(xtest.astype(np.float64)))
            cal_q = np.mean(np.square(xcal.astype(np.float64)[:, qi] - cal_rec[:, qi]), axis=1)
            test_q = np.mean(np.square(xtest.astype(np.float64)[:, qi] - test_rec[:, qi]), axis=1)
            thq = percentile_threshold(cal_q)
            m = metric_dict(ytest, test_q, thq)
            metric_rows.append({
                "dataset": ds, "baseline": "PCA_reconstruction_q_score", "condition": condition,
                "threshold": thq, "pca_components": int(pca.n_components_),
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)), **m,
            })
            bdir = dirs["baselines"] / "pca" / ds / condition
            bdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pca, bdir / "pca.pkl")
            pred = (test_q > thq).astype(np.int8)
            pd.DataFrame({
                "source_row_index": test_part["source_row_index"].astype(np.int64),
                "analysis_unit_id": test_part["analysis_unit_id"].astype(str),
                "analysis_unit_kind": test_part["analysis_unit_kind"].astype(str),
                "capture_id": test_part["capture_id"].astype(str),
                "label_type": test_part["label_type"].astype(str),
                "y_true": ytest,
                "q_score": test_q,
                "prediction": pred,
            }).to_csv(bdir / "test_predictions.csv.gz", index=False, compression="gzip")
            np.savez_compressed(bdir / "calibration_scores.npz", q_score=cal_q.astype(np.float64))
            save_json(bdir / "metrics.json", metric_rows[-1])
            del xfit, xcal, xtest, cal_rec, test_rec, pca
            gc.collect()
        log(root, f"{ds}: baselines complete")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(dirs["tables"] / "baseline_metrics.csv", index=False)
    mark_stage(dirs, "baselines", protocol_hash, {"baseline_rows": len(metrics)})
    log(root, "END baselines [COMPLETED]")


# =============================================================================
# MAIN NEURAL TRAINING
# =============================================================================

def _run_one_neural(
    args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, legacy,
    ds: str, dataset_idx: int, condition: str, condition_idx: int, spec: dict[str, Any],
    model_name: str, model_idx: int, seed: int, prepared: pd.DataFrame,
    scaler: MinMaxScaler, q_features: list[str],
) -> dict[str, Any]:
    run_dir = dirs["runs"] / ds / condition / model_name / f"seed_{seed:02d}"
    if run_artifacts_valid(run_dir, protocol_hash):
        log(root, f"[RESUME] {ds} {condition} {model_name} seed={seed}")
        return load_json(run_dir / "metrics.json")
    if run_dir.exists():
        archive_invalid_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    features = list(spec["features"])
    fit_part, xfit = transform_partition(prepared, "fit_normal", spec, scaler)
    early_part, xearly = transform_partition(prepared, "earlystop_normal", spec, scaler)
    cal_part, xcal = transform_partition(prepared, "calibration_normal", spec, scaler)
    test_part, xtest = transform_partition(prepared, "test", spec, scaler)
    if not np.all(cal_part["y"].to_numpy(dtype=np.int8) == 0):
        raise RuntimeError(f"{ds} {condition}: calibration partition is not normal-only")
    qi = q_indices(features, q_features)
    mseed = model_seed(dataset_idx, condition_idx, model_idx, seed)
    model = build_model(legacy, model_name, len(features), ds, mseed, compile_model=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0)
    t0 = time.time()
    history = model.fit(
        legacy.audit.input_for_model(xfit, model_name),
        legacy.audit.target_for_model(xfit, model_name),
        validation_data=(
            legacy.audit.input_for_model(xearly, model_name),
            legacy.audit.target_for_model(xearly, model_name),
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )
    train_sec = time.time() - t0
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(run_dir / "training_history.csv", index=False)

    cal_pred = predict_flat(legacy, model, xcal, model_name)
    test_pred = predict_flat(legacy, model, xtest, model_name)
    cal_q, cal_full, cal_qres = reconstruction_scores(xcal, cal_pred, qi)
    test_q, test_full, test_qres = reconstruction_scores(xtest, test_pred, qi)
    qth = percentile_threshold(cal_q)
    fullth = percentile_threshold(cal_full)
    ytest = test_part["y"].to_numpy(dtype=np.int8)
    mq = metric_dict(ytest, test_q, qth)
    mf = metric_dict(ytest, test_full, fullth)
    qpred = (test_q > qth).astype(np.int8)
    fpred = (test_full > fullth).astype(np.int8)

    weights = run_dir / "model.weights.h5"
    model.save_weights(weights)
    scaler_path = scaler_dir(dirs, ds, condition) / "scaler.pkl"
    predictions_path = run_dir / "test_predictions.csv.gz"
    pred_df = pd.DataFrame({
        "sample_local_index": test_part["sample_local_index"].to_numpy(dtype=np.int64),
        "source_row_index": test_part["source_row_index"].to_numpy(dtype=np.int64),
        "analysis_unit_id": test_part["analysis_unit_id"].astype(str).to_numpy(),
        "analysis_unit_kind": test_part["analysis_unit_kind"].astype(str).to_numpy(),
        "capture_id": test_part["capture_id"].astype(str).to_numpy(),
        "capture_path": test_part["capture_path"].astype(str).to_numpy(),
        "raw_capture_id": test_part["raw_capture_id"].astype(str).to_numpy(),
        "raw_capture_path": test_part["raw_capture_path"].astype(str).to_numpy(),
        "flow_id": test_part["flow_id"].astype(str).to_numpy(),
        "label": test_part[DATASETS[ds]["label"]].astype(str).to_numpy(),
        "label_type": test_part["label_type"].astype(str).to_numpy(),
        "y_true": ytest,
        "q_score": test_q,
        "q_prediction": qpred,
        "full_score": test_full,
        "full_prediction": fpred,
    })
    pred_df.to_csv(predictions_path, index=False, compression="gzip")
    cal_path = run_dir / "calibration_scores.npz"
    np.savez_compressed(
        cal_path,
        source_row_index=cal_part["source_row_index"].to_numpy(dtype=np.int64),
        q_score=cal_q.astype(np.float64),
        full_score=cal_full.astype(np.float64),
    )
    residual_path = run_dir / "test_q_squared_residuals.npz"
    np.savez_compressed(
        residual_path,
        source_row_index=test_part["source_row_index"].to_numpy(dtype=np.int64),
        q_feature_names=np.asarray(q_features, dtype=str),
        q_squared_residuals=test_qres.astype(np.float32),
    )

    metrics = {
        "dataset": ds,
        "condition": condition,
        "condition_role": spec["role"],
        "model": model_name,
        "seed": int(seed),
        "model_seed": int(mseed),
        "features": features,
        "q_features": q_features,
        "input_dim": len(features),
        "q_dim": len(q_features),
        "params": int(model.count_params()),
        "epochs_used": int(len(history.history.get("loss", []))),
        "train_time_sec": float(train_sec),
        "rss_max_mb": rss_mb(),
        "q_threshold_p99_calibration_normal": qth,
        "full_threshold_p99_calibration_normal": fullth,
        "q_score_metrics": mq,
        "full_score_metrics": mf,
        "fit_normal_n": int(len(fit_part)),
        "earlystop_normal_n": int(len(early_part)),
        "calibration_normal_n": int(len(cal_part)),
        "test_n": int(len(test_part)),
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "preprocess": spec["preprocess"],
    }
    save_json(run_dir / "metrics.json", metrics)
    artifact_hashes = {
        "model.weights.h5": sha256_file(weights),
        "training_history.csv": sha256_file(run_dir / "training_history.csv"),
        "test_predictions.csv.gz": sha256_file(predictions_path),
        "calibration_scores.npz": sha256_file(cal_path),
        "test_q_squared_residuals.npz": sha256_file(residual_path),
        "metrics.json": sha256_file(run_dir / "metrics.json"),
    }
    save_json(run_dir / "RUN_COMPLETE.json", {
        "status": "COMPLETED",
        "completed_utc": utc_now(),
        "protocol_hash": protocol_hash,
        "dataset": ds,
        "condition": condition,
        "model": model_name,
        "seed": int(seed),
        "artifact_sha256": artifact_hashes,
    })
    log(root, f"[DONE] {ds} {condition} {model_name} seed={seed} FNR={mq['fnr']:.4f} FPR={mq['fpr']:.4f} AUC={mq['roc_auc']:.4f}")
    del model, history, cal_pred, test_pred, cal_qres, test_qres, xfit, xearly, xcal, xtest
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics


def train_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, legacy, q_features: list[str], specs: dict[str, dict[str, Any]]):
    if stage_is_complete(dirs, "train", protocol_hash):
        log(root, "[RESUME] train already complete")
        return
    if not stage_is_complete(dirs, "baselines", protocol_hash):
        raise RuntimeError("baselines must complete before train")
    log(root, "START train")
    conditions = list(SAFE_CONDITION_NAMES)
    if not args.skip_precision_sensitivity:
        conditions.append(PRECISION_CONDITION_NAME)
    expected = len(DATASET_ORDER) * len(conditions) * len(args.models) * len(args.seeds)
    rows = []
    scaler_qc = []
    for di, ds in enumerate(DATASET_ORDER):
        prepared = load_prepared(dirs, ds)
        # Fit all condition scalers first so common-q preprocessing can be checked before any model result is accepted.
        for condition in conditions:
            fit_or_load_scaler(dirs, ds, condition, specs[condition], prepared, protocol_hash)
        qc = shared_scaler_consistency(dirs, ds, specs, q_features)
        scaler_qc.append(qc)
        safe_q_max = float(qc[["abs_data_min_diff_vs_B", "abs_data_max_diff_vs_B", "abs_scale_diff_vs_B", "abs_min_param_diff_vs_B"]].to_numpy().max())
        if safe_q_max > 1e-12:
            raise RuntimeError(f"{ds}: q-feature MinMax parameters differ across safe conditions; max={safe_q_max}")
        for ci, condition in enumerate(conditions):
            spec = specs[condition]
            scaler = joblib.load(scaler_dir(dirs, ds, condition) / "scaler.pkl")
            for mi, model_name in enumerate(args.models):
                for seed in args.seeds:
                    metrics = _run_one_neural(
                        args, root, dirs, protocol_hash, legacy,
                        ds, di, condition, ci, spec, model_name, mi, int(seed), prepared, scaler, q_features,
                    )
                    flat = {
                        "dataset": ds, "condition": condition, "model": model_name, "seed": int(seed),
                        "input_dim": metrics["input_dim"], "q_dim": metrics["q_dim"], "params": metrics["params"],
                        "epochs_used": metrics["epochs_used"], "train_time_sec": metrics["train_time_sec"],
                        "rss_max_mb": metrics["rss_max_mb"],
                        "q_threshold": metrics["q_threshold_p99_calibration_normal"],
                        "full_threshold": metrics["full_threshold_p99_calibration_normal"],
                    }
                    flat.update({f"q_{k}": v for k, v in metrics["q_score_metrics"].items()})
                    flat.update({f"full_{k}": v for k, v in metrics["full_score_metrics"].items()})
                    rows.append(flat)
        del prepared
        gc.collect()
    qcdf = pd.concat(scaler_qc, ignore_index=True)
    qcdf.to_csv(dirs["tables"] / "q_scaler_consistency.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "neural_run_metrics.csv", index=False)
    if len(out) != expected:
        raise RuntimeError(f"Training aggregation has {len(out)} rows, expected {expected}")
    mark_stage(dirs, "train", protocol_hash, {"completed_runs": len(out), "expected_runs": expected})
    log(root, f"END train [COMPLETED] runs={len(out)}")


# =============================================================================
# FIXED-WEIGHT TIMESTAMP-SHIFT INTERVENTIONS
# =============================================================================

def analysis_group_hash_sign(group_id: str) -> float:
    h = hashlib.sha256(str(group_id).encode("utf-8")).digest()
    return 1.0 if (h[0] & 1) else -1.0


def shifted_scaled_matrix(
    part: pd.DataFrame, spec: dict[str, Any], scaler: MinMaxScaler,
    fit_epoch_span: float, fit_epoch_median: float, scheme_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if "frame.time_epoch" not in spec["features"]:
        raise RuntimeError("Timestamp shift requested for condition without frame.time_epoch")
    raw = condition_raw_matrix(part, list(spec["features"]), spec["preprocess"])
    j = list(spec["features"]).index("frame.time_epoch")
    scheme = SHIFT_SCHEMES[scheme_name]
    if scheme["kind"] == "neutralize_fit_median":
        original = raw[:, j].copy()
        raw[:, j] = float(fit_epoch_median)
        offsets = raw[:, j] - original
    elif scheme["kind"] == "global":
        offsets = np.full(len(part), float(scheme["multiplier"]) * fit_epoch_span, dtype=np.float64)
        raw[:, j] += offsets
    elif scheme["kind"] == "hash_signed":
        signs = np.asarray([analysis_group_hash_sign(x) for x in part["analysis_unit_id"].astype(str)], dtype=np.float64)
        offsets = signs * float(scheme["multiplier"]) * fit_epoch_span
        raw[:, j] += offsets
    else:
        raise ValueError(scheme)
    scaled = scaler.transform(raw).astype(np.float32)
    if not np.all(np.isfinite(scaled)):
        raise RuntimeError(f"Non-finite shifted transformed matrix for {scheme_name}")
    return scaled, offsets


def intervention_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, legacy, q_features: list[str], specs: dict[str, dict[str, Any]]):
    if stage_is_complete(dirs, "intervene", protocol_hash):
        log(root, "[RESUME] intervene already complete")
        return
    if not stage_is_complete(dirs, "train", protocol_hash):
        raise RuntimeError("train must complete before intervene")
    if args.skip_interventions:
        mark_stage(dirs, "intervene", protocol_hash, {"skipped_by_user": True})
        save_json(dirs["interventions"] / "RUN_COMPLETE.json", {"status": "SKIPPED_BY_USER", "protocol_hash": protocol_hash})
        log(root, "END intervene [SKIPPED_BY_USER]")
        return
    log(root, "START intervene")
    t_conditions = ["L34_q_r_t", "A_q_t", "C_q_b_t"]
    rows = []
    for di, ds in enumerate(DATASET_ORDER):
        prepared = load_prepared(dirs, ds)
        fit = prepared[prepared["split"] == "fit_normal"].copy()
        fit_t = fit["frame.time_epoch"].to_numpy(dtype=np.float64)
        fit_span = float(np.max(fit_t) - np.min(fit_t))
        fit_median = float(np.median(fit_t))
        if not np.isfinite(fit_span) or fit_span <= 0 or not np.isfinite(fit_median):
            raise RuntimeError(f"{ds}: invalid fit-normal epoch span/median for interventions")
        for condition in t_conditions:
            spec = specs[condition]
            scaler = joblib.load(scaler_dir(dirs, ds, condition) / "scaler.pkl")
            cal_part, xcal_orig = transform_partition(prepared, "calibration_normal", spec, scaler)
            test_part, xtest_orig = transform_partition(prepared, "test", spec, scaler)
            qi = q_indices(spec["features"], q_features)
            for mi, model_name in enumerate(args.models):
                for seed in args.seeds:
                    main_dir = dirs["runs"] / ds / condition / model_name / f"seed_{int(seed):02d}"
                    if not run_artifacts_valid(main_dir, protocol_hash):
                        raise RuntimeError(f"Main run invalid before intervention: {main_dir}")
                    main_metrics = load_json(main_dir / "metrics.json")
                    original_threshold = float(main_metrics["q_threshold_p99_calibration_normal"])
                    saved_pred = pd.read_csv(main_dir / "test_predictions.csv.gz")
                    model = build_model(legacy, model_name, len(spec["features"]), ds, seed=990000 + di * 1000 + mi * 100 + int(seed), compile_model=False)
                    model.load_weights(main_dir / "model.weights.h5")
                    orig_test_pred = predict_flat(legacy, model, xtest_orig, model_name)
                    orig_test_q, _, _ = reconstruction_scores(xtest_orig, orig_test_pred, qi)
                    if not np.array_equal(saved_pred["source_row_index"].to_numpy(dtype=np.int64), test_part["source_row_index"].to_numpy(dtype=np.int64)):
                        raise RuntimeError(f"{ds} {condition} {model_name} seed={seed}: test identity mismatch")
                    saved_q = saved_pred["q_score"].to_numpy(dtype=float)
                    identity_max = float(np.max(np.abs(orig_test_q - saved_q)))
                    identity_ok = bool(np.allclose(orig_test_q, saved_q, rtol=1e-5, atol=1e-7))
                    if not identity_ok:
                        raise RuntimeError(f"{ds} {condition} {model_name} seed={seed}: weight reload q-score mismatch max={identity_max}")
                    original_qpred = saved_pred["q_prediction"].to_numpy(dtype=np.int8)
                    ytest = saved_pred["y_true"].to_numpy(dtype=np.int8)

                    for scheme_name in SHIFT_SCHEMES:
                        idir = dirs["interventions"] / ds / condition / model_name / f"seed_{int(seed):02d}" / scheme_name
                        marker = idir / "RUN_COMPLETE.json"
                        if marker.exists():
                            try:
                                obj = load_json(marker)
                                if obj.get("protocol_hash") == protocol_hash and obj.get("status") == "COMPLETED":
                                    rows.append(load_json(idir / "metrics.json"))
                                    continue
                            except Exception:
                                pass
                            archive_invalid_run(idir)
                        idir.mkdir(parents=True, exist_ok=True)
                        xcal_shift, cal_offsets = shifted_scaled_matrix(cal_part, spec, scaler, fit_span, fit_median, scheme_name)
                        xtest_shift, test_offsets = shifted_scaled_matrix(test_part, spec, scaler, fit_span, fit_median, scheme_name)
                        cal_pred = predict_flat(legacy, model, xcal_shift, model_name)
                        test_pred = predict_flat(legacy, model, xtest_shift, model_name)
                        cal_q, _, _ = reconstruction_scores(xcal_shift, cal_pred, qi)
                        test_q, _, _ = reconstruction_scores(xtest_shift, test_pred, qi)
                        recal_threshold = percentile_threshold(cal_q)
                        fixed_metrics = metric_dict(ytest, test_q, original_threshold)
                        recal_metrics = metric_dict(ytest, test_q, recal_threshold)
                        fixed_pred = (test_q > original_threshold).astype(np.int8)
                        recal_pred = (test_q > recal_threshold).astype(np.int8)
                        fixed_d2m = int(np.sum((ytest == 1) & (original_qpred == 1) & (fixed_pred == 0)))
                        fixed_m2d = int(np.sum((ytest == 1) & (original_qpred == 0) & (fixed_pred == 1)))
                        recal_d2m = int(np.sum((ytest == 1) & (original_qpred == 1) & (recal_pred == 0)))
                        recal_m2d = int(np.sum((ytest == 1) & (original_qpred == 0) & (recal_pred == 1)))

                        packet_path = idir / "test_predictions_shifted.csv.gz"
                        pd.DataFrame({
                            "source_row_index": test_part["source_row_index"].to_numpy(dtype=np.int64),
                            "analysis_unit_id": test_part["analysis_unit_id"].astype(str).to_numpy(),
                            "analysis_unit_kind": test_part["analysis_unit_kind"].astype(str).to_numpy(),
                            "capture_id": test_part["capture_id"].astype(str).to_numpy(),
                            "label_type": test_part["label_type"].astype(str).to_numpy(),
                            "y_true": ytest,
                            "raw_epoch_offset_seconds": test_offsets,
                            "original_q_score": saved_q,
                            "original_q_prediction": original_qpred,
                            "shifted_q_score": test_q,
                            "shifted_fixed_prediction": fixed_pred,
                            "shifted_recalibrated_prediction": recal_pred,
                        }).to_csv(packet_path, index=False, compression="gzip")
                        cal_path = idir / "calibration_shifted_scores.npz"
                        np.savez_compressed(
                            cal_path,
                            source_row_index=cal_part["source_row_index"].to_numpy(dtype=np.int64),
                            analysis_unit_id=cal_part["analysis_unit_id"].astype(str).to_numpy(dtype=str),
                            capture_id=cal_part["capture_id"].astype(str).to_numpy(dtype=str),
                            raw_epoch_offset_seconds=cal_offsets.astype(np.float64),
                            shifted_q_score=cal_q.astype(np.float64),
                        )
                        result = {
                            "dataset": ds,
                            "condition": condition,
                            "model": model_name,
                            "seed": int(seed),
                            "scheme": scheme_name,
                            "fit_epoch_span_seconds": fit_span,
                            "fit_epoch_median_seconds": fit_median,
                            "original_threshold": original_threshold,
                            "shifted_recalibrated_threshold": recal_threshold,
                            "weight_reload_identity_max_abs_q_score_diff": identity_max,
                            "fixed_threshold_metrics": fixed_metrics,
                            "recalibrated_threshold_metrics": recal_metrics,
                            "fixed_detected_to_missed_attack_packet_seed_evaluations": fixed_d2m,
                            "fixed_missed_to_detected_attack_packet_seed_evaluations": fixed_m2d,
                            "recal_detected_to_missed_attack_packet_seed_evaluations": recal_d2m,
                            "recal_missed_to_detected_attack_packet_seed_evaluations": recal_m2d,
                            "scope": "controlled frame.time_epoch intervention; fixed weights/scaler; q/r/b and non-time packet features unchanged. Neutralization is not a constant translation; the two shift schemes are.",
                        }
                        save_json(idir / "metrics.json", result)
                        save_json(marker, {
                            "status": "COMPLETED", "protocol_hash": protocol_hash, "completed_utc": utc_now(),
                            "artifact_sha256": {
                                "metrics.json": sha256_file(idir / "metrics.json"),
                                "test_predictions_shifted.csv.gz": sha256_file(packet_path),
                                "calibration_shifted_scores.npz": sha256_file(cal_path),
                            },
                        })
                        rows.append(result)
                        del xcal_shift, xtest_shift, cal_pred, test_pred
                    del model, orig_test_pred
                    tf.keras.backend.clear_session()
                    gc.collect()
            del xcal_orig, xtest_orig
        del prepared
    # Flatten metrics for convenient auditing while preserving complete JSON per run.
    flat_rows = []
    for r in rows:
        base = {k: r[k] for k in ["dataset", "condition", "model", "seed", "scheme", "fit_epoch_span_seconds", "fit_epoch_median_seconds", "original_threshold", "shifted_recalibrated_threshold"]}
        for track_key, prefix in [("fixed_threshold_metrics", "fixed"), ("recalibrated_threshold_metrics", "recal")]:
            for k, v in r[track_key].items():
                base[f"{prefix}_{k}"] = v
        flat_rows.append(base)
    pd.DataFrame(flat_rows).to_csv(dirs["tables"] / "intervention_run_metrics.csv", index=False)
    save_json(dirs["interventions"] / "RUN_COMPLETE.json", {"status": "COMPLETED", "protocol_hash": protocol_hash, "runs": len(rows)})
    mark_stage(dirs, "intervene", protocol_hash, {"runs": len(rows)})
    log(root, f"END intervene [COMPLETED] runs={len(rows)}")


# =============================================================================
# ACQUISITION-GROUP STATISTICS / SECURITY ANALYSIS
# =============================================================================

def signflip_pvalue(diff: np.ndarray, seed: int = 87031, monte_carlo: int = 200000) -> tuple[float, str, int]:
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), "none", 0
    if np.allclose(x, 0):
        return 1.0, "degenerate_all_zero", n
    obs = abs(float(np.mean(x)))
    if n <= 20:
        total = 1 << n
        count = 0
        # Exact two-sided sign-flip distribution under exchangeable capture-group signs.
        for mask in range(total):
            signs = np.ones(n, dtype=float)
            for j in range(n):
                if (mask >> j) & 1:
                    signs[j] = -1.0
            if abs(float(np.mean(x * signs))) >= obs - 1e-15:
                count += 1
        return float(count / total), "exact_sign_flip", n
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    chunk = 10000
    while done < monte_carlo:
        k = min(chunk, monte_carlo - done)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(k, n))
        means = np.abs((signs * x[None, :]).mean(axis=1))
        ge += int(np.sum(means >= obs - 1e-15))
        done += k
    return float((ge + 1) / (monte_carlo + 1)), "monte_carlo_sign_flip", n


def _contrast_family(contrast: str) -> str:
    return "primary" if contrast in PRIMARY_CONTRASTS else "bridge"


def _load_prediction(dirs: dict[str, Path], ds: str, condition: str, model: str, seed: int) -> pd.DataFrame:
    p = dirs["runs"] / ds / condition / model / f"seed_{int(seed):02d}" / "test_predictions.csv.gz"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _analysis_unit_fnr(pred: pd.DataFrame) -> pd.DataFrame:
    a = pred[pred["y_true"].astype(int) == 1].copy()
    if a.empty:
        return pd.DataFrame(columns=["analysis_unit_id", "attack_n", "fnr"])
    return a.groupby("analysis_unit_id", as_index=False).agg(
        attack_n=("y_true", "size"),
        missed=("q_prediction", lambda s: int(np.sum(np.asarray(s, dtype=int) == 0))),
    ).assign(fnr=lambda d: d["missed"] / d["attack_n"])


def planned_contrast_analysis(args: argparse.Namespace, dirs: dict[str, Path], q_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrasts = dict(PRIMARY_CONTRASTS)
    if not args.skip_precision_sensitivity:
        contrasts.update(BRIDGE_CONTRASTS)
    else:
        contrasts.update({k: v for k, v in BRIDGE_CONTRASTS.items() if PRECISION_CONDITION_NAME not in v})
    unit_seed_rows = []
    transition_rows = []
    residual_rows = []

    for contrast, (left, right) in contrasts.items():
        for ds in DATASET_ORDER:
            for model in args.models:
                for seed in args.seeds:
                    lp = _load_prediction(dirs, ds, left, model, int(seed))
                    rp = _load_prediction(dirs, ds, right, model, int(seed))
                    key = ["source_row_index", "analysis_unit_id", "y_true"]
                    m = lp[key + ["q_prediction", "q_score"]].merge(
                        rp[key + ["q_prediction", "q_score"]], on=key, how="inner", suffixes=("_left", "_right"), validate="one_to_one"
                    )
                    if len(m) != len(lp) or len(m) != len(rp):
                        raise RuntimeError(f"{contrast} {ds} {model} seed={seed}: paired test rows mismatch")
                    attack = m[m["y_true"] == 1].copy()
                    for cap, g in attack.groupby("analysis_unit_id"):
                        fnr_left = float(np.mean(g["q_prediction_left"].to_numpy(dtype=int) == 0))
                        fnr_right = float(np.mean(g["q_prediction_right"].to_numpy(dtype=int) == 0))
                        unit_seed_rows.append({
                            "contrast": contrast, "contrast_family": _contrast_family(contrast),
                            "dataset": ds, "model": model, "seed": int(seed), "analysis_unit_id": cap,
                            "attack_n": len(g), "fnr_left": fnr_left, "fnr_right": fnr_right,
                            "delta_fnr_right_minus_left": fnr_right - fnr_left,
                            "left_condition": left, "right_condition": right,
                        })
                    d2m = int(np.sum((attack["q_prediction_left"] == 1) & (attack["q_prediction_right"] == 0)))
                    m2d = int(np.sum((attack["q_prediction_left"] == 0) & (attack["q_prediction_right"] == 1)))
                    transition_rows.append({
                        "contrast": contrast, "dataset": ds, "model": model, "seed": int(seed),
                        "left_condition": left, "right_condition": right,
                        "attack_packet_seed_evaluations": len(attack),
                        "left_detected_to_right_missed": d2m,
                        "left_missed_to_right_detected": m2d,
                        "net_more_missed_right": d2m - m2d,
                    })

                    # Descriptive q-residual decomposition only on left-detected -> right-missed attack transitions.
                    switched_src = attack.loc[(attack["q_prediction_left"] == 1) & (attack["q_prediction_right"] == 0), "source_row_index"].to_numpy(dtype=np.int64)
                    if len(switched_src):
                        lz = np.load(dirs["runs"] / ds / left / model / f"seed_{int(seed):02d}" / "test_q_squared_residuals.npz", allow_pickle=False)
                        rz = np.load(dirs["runs"] / ds / right / model / f"seed_{int(seed):02d}" / "test_q_squared_residuals.npz", allow_pickle=False)
                        lsrc = lz["source_row_index"].astype(np.int64)
                        rsrc = rz["source_row_index"].astype(np.int64)
                        lnames = lz["q_feature_names"].astype(str).tolist()
                        rnames = rz["q_feature_names"].astype(str).tolist()
                        if lnames != q_features or rnames != q_features:
                            raise RuntimeError("q residual feature order mismatch")
                        if not np.array_equal(lsrc, rsrc):
                            raise RuntimeError("q residual source-row order mismatch across paired conditions")
                        pos = {int(v): i for i, v in enumerate(lsrc)}
                        idx = np.asarray([pos[int(v)] for v in switched_src], dtype=np.int64)
                        lr = lz["q_squared_residuals"][idx].astype(np.float64)
                        rr = rz["q_squared_residuals"][idx].astype(np.float64)
                        for j, feat in enumerate(q_features):
                            residual_rows.append({
                                "contrast": contrast, "dataset": ds, "model": model, "seed": int(seed),
                                "feature": feat, "switched_attack_packet_seed_n": int(len(idx)),
                                "left_residual_mean": float(np.mean(lr[:, j])),
                                "right_residual_mean": float(np.mean(rr[:, j])),
                                "delta_right_minus_left_mean": float(np.mean(rr[:, j] - lr[:, j])),
                                "interpretation": "descriptive reconstruction-error decomposition; not causal feature importance",
                            })
    unit_seed = pd.DataFrame(unit_seed_rows)
    transitions = pd.DataFrame(transition_rows)
    residuals = pd.DataFrame(residual_rows)
    return unit_seed, transitions, residuals


def shared_cross_model_transition_analysis(args: argparse.Namespace, dirs: dict[str, Path]) -> pd.DataFrame:
    """Direct failure shared by every selected architecture for the same packet and seed.

    This is a descriptive security endpoint, not an independent statistical sample:
    one row can be evaluated under several model seeds, and model outputs for a
    packet are correlated.  Counts are therefore explicitly packet-seed counts.
    """
    contrasts = dict(PRIMARY_CONTRASTS)
    if not args.skip_precision_sensitivity:
        contrasts.update(BRIDGE_CONTRASTS)
    else:
        contrasts.update({k: v for k, v in BRIDGE_CONTRASTS.items() if PRECISION_CONDITION_NAME not in v})
    rows = []
    for contrast, (left, right) in contrasts.items():
        for ds in DATASET_ORDER:
            for seed in args.seeds:
                merged = None
                key = ["source_row_index", "analysis_unit_id", "label_type", "y_true"]
                for model in args.models:
                    lp = _load_prediction(dirs, ds, left, model, int(seed))[key + ["q_prediction"]].rename(
                        columns={"q_prediction": f"left_{model}"}
                    )
                    rp = _load_prediction(dirs, ds, right, model, int(seed))[key + ["q_prediction"]].rename(
                        columns={"q_prediction": f"right_{model}"}
                    )
                    pair = lp.merge(rp, on=key, how="inner", validate="one_to_one")
                    merged = pair if merged is None else merged.merge(pair, on=key, how="inner", validate="one_to_one")
                if merged is None or merged.empty:
                    continue
                attack = merged[merged["y_true"].astype(int) == 1].copy()
                lcols = [f"left_{m}" for m in args.models]
                rcols = [f"right_{m}" for m in args.models]
                lm = attack[lcols].to_numpy(dtype=np.int8)
                rm = attack[rcols].to_numpy(dtype=np.int8)
                attack["left_detected_all"] = np.all(lm == 1, axis=1)
                attack["left_missed_all"] = np.all(lm == 0, axis=1)
                attack["right_detected_all"] = np.all(rm == 1, axis=1)
                attack["right_missed_all"] = np.all(rm == 0, axis=1)

                groups = [("ALL", attack)] + [(str(g), z) for g, z in attack.groupby("label_type", dropna=False)]
                for group, g in groups:
                    rows.append({
                        "contrast": contrast,
                        "dataset": ds,
                        "seed": int(seed),
                        "attack_group": group,
                        "models": "|".join(args.models),
                        "attack_packet_seed_n": int(len(g)),
                        "left_detected_all_n": int(g["left_detected_all"].sum()),
                        "right_missed_all_n": int(g["right_missed_all"].sum()),
                        "left_detected_all_to_right_missed_all_n": int((g["left_detected_all"] & g["right_missed_all"]).sum()),
                        "left_missed_all_to_right_detected_all_n": int((g["left_missed_all"] & g["right_detected_all"]).sum()),
                        "unit_note": "descriptive packet-seed evaluations; not independent observations",
                    })
    return pd.DataFrame(rows)


def summarize_analysis_unit_contrasts(unit_seed: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unit_seed.empty:
        return pd.DataFrame(), pd.DataFrame()
    per_analysis_unit = unit_seed.groupby(
        ["contrast", "contrast_family", "dataset", "model", "analysis_unit_id", "left_condition", "right_condition"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        attack_n_mean=("attack_n", "mean"),
        fnr_left=("fnr_left", "mean"),
        fnr_right=("fnr_right", "mean"),
        delta_fnr_right_minus_left=("delta_fnr_right_minus_left", "mean"),
    )
    stat_rows = []
    for keys, g in per_analysis_unit.groupby(["contrast", "contrast_family", "dataset", "model", "left_condition", "right_condition"]):
        contrast, family, ds, model, left, right = keys
        diff = g["delta_fnr_right_minus_left"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(diff, iterations=args.bootstrap_iterations, seed=91000 + len(stat_rows))
        p, method, n = signflip_pvalue(diff, seed=92000 + len(stat_rows))
        stat_rows.append({
            "contrast": contrast, "contrast_family": family, "dataset": ds, "model": model,
            "left_condition": left, "right_condition": right,
            "analysis_unit_n": int(n),
            "mean_fnr_left": float(g["fnr_left"].mean()),
            "mean_fnr_right": float(g["fnr_right"].mean()),
            "mean_delta_fnr_right_minus_left": float(np.mean(diff)),
            "median_delta_fnr_right_minus_left": float(np.median(diff)),
            "analysis_unit_bootstrap_ci95_low": lo,
            "analysis_unit_bootstrap_ci95_high": hi,
            "signflip_p_raw": p,
            "signflip_method": method,
            "wilcoxon_p_secondary": wilcoxon_safe(diff),
            "unit_note": "analysis unit defined by split protocol; model seeds averaged within analysis unit before inference",
        })
    stats_df = pd.DataFrame(stat_rows)
    # Each planned contrast is its own multiplicity family across dataset-model comparisons.
    out = []
    for contrast, g in stats_df.groupby("contrast", sort=False):
        z = g.copy()
        z["signflip_p_holm"] = holm_adjust(z["signflip_p_raw"].to_numpy(dtype=float))
        z["signflip_holm_lt_0_05"] = z["signflip_p_holm"] < 0.05
        out.append(z)
    stats_df = pd.concat(out, ignore_index=True)
    return per_analysis_unit, stats_df


def attack_group_and_ensemble_analysis(args: argparse.Namespace, dirs: dict[str, Path], conditions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows = []
    ensemble_rows = []
    shared_rows = []
    for ds in DATASET_ORDER:
        for condition in conditions:
            for seed in args.seeds:
                model_frames = []
                for model in args.models:
                    p = _load_prediction(dirs, ds, condition, model, int(seed))
                    p = p[["source_row_index", "analysis_unit_id", "label_type", "y_true", "q_prediction", "q_score"]].copy()
                    for (cap, grp), g in p[p["y_true"] == 1].groupby(["analysis_unit_id", "label_type"]):
                        group_rows.append({
                            "dataset": ds, "condition": condition, "model": model, "seed": int(seed),
                            "analysis_unit_id": cap, "attack_group": grp, "n": len(g),
                            "recall": float(np.mean(g["q_prediction"].to_numpy(dtype=int) == 1)),
                            "fnr": float(np.mean(g["q_prediction"].to_numpy(dtype=int) == 0)),
                            "q_score_mean": float(g["q_score"].mean()),
                        })
                    p = p.rename(columns={"q_prediction": f"pred_{model}", "q_score": f"score_{model}"})
                    if model_frames:
                        base = model_frames[0]
                        model_frames[0] = base.merge(
                            p[["source_row_index", "analysis_unit_id", "label_type", "y_true", f"pred_{model}", f"score_{model}"]],
                            on=["source_row_index", "analysis_unit_id", "label_type", "y_true"], how="inner", validate="one_to_one"
                        )
                    else:
                        model_frames.append(p)
                merged = model_frames[0]
                pred_cols = [f"pred_{m}" for m in args.models]
                if len(merged) == 0:
                    continue
                pred_mat = merged[pred_cols].to_numpy(dtype=np.int8)
                or_pred = (np.max(pred_mat, axis=1) > 0).astype(np.int8)
                y = merged["y_true"].to_numpy(dtype=np.int8)
                tp = int(np.sum((y == 1) & (or_pred == 1)))
                fn = int(np.sum((y == 1) & (or_pred == 0)))
                fp = int(np.sum((y == 0) & (or_pred == 1)))
                tn = int(np.sum((y == 0) & (or_pred == 0)))
                ensemble_rows.append({
                    "dataset": ds, "condition": condition, "seed": int(seed), "models_in_or": "|".join(args.models),
                    "test_n": len(y), "recall": tp / max(tp + fn, 1), "fnr": fn / max(tp + fn, 1),
                    "fpr": fp / max(fp + tn, 1), "precision": tp / max(tp + fp, 1),
                })
                attack = merged[y == 1].copy()
                amat = attack[pred_cols].to_numpy(dtype=np.int8)
                shared_rows.append({
                    "dataset": ds, "condition": condition, "seed": int(seed), "models": "|".join(args.models),
                    "attack_packet_n": len(attack),
                    "missed_by_all_models": int(np.sum(np.all(amat == 0, axis=1))),
                    "detected_by_all_models": int(np.sum(np.all(amat == 1, axis=1))),
                    "detected_by_any_model": int(np.sum(np.any(amat == 1, axis=1))),
                    "denominator_note": "unique materialized attack packets within one seed; seeds reported separately",
                })
    return pd.DataFrame(group_rows), pd.DataFrame(ensemble_rows), pd.DataFrame(shared_rows)


def intervention_analysis_unit_analysis(args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.skip_interventions:
        return pd.DataFrame(), pd.DataFrame()
    seed_rows = []
    for ds in DATASET_ORDER:
        for condition in ["L34_q_r_t", "A_q_t", "C_q_b_t"]:
            for model in args.models:
                for seed in args.seeds:
                    original = _load_prediction(dirs, ds, condition, model, int(seed))
                    orig = original[["source_row_index", "analysis_unit_id", "y_true", "q_prediction"]].copy()
                    for scheme in SHIFT_SCHEMES:
                        p = dirs["interventions"] / ds / condition / model / f"seed_{int(seed):02d}" / scheme / "test_predictions_shifted.csv.gz"
                        shifted = pd.read_csv(p)
                        m = orig.merge(
                            shifted[["source_row_index", "analysis_unit_id", "y_true", "shifted_fixed_prediction", "shifted_recalibrated_prediction"]],
                            on=["source_row_index", "analysis_unit_id", "y_true"], how="inner", validate="one_to_one"
                        )
                        attack = m[m["y_true"] == 1]
                        for track, predcol in [("fixed_threshold", "shifted_fixed_prediction"), ("recalibrated_threshold", "shifted_recalibrated_prediction")]:
                            for cap, g in attack.groupby("analysis_unit_id"):
                                orig_fnr = float(np.mean(g["q_prediction"].to_numpy(dtype=int) == 0))
                                shift_fnr = float(np.mean(g[predcol].to_numpy(dtype=int) == 0))
                                seed_rows.append({
                                    "dataset": ds, "condition": condition, "model": model, "seed": int(seed),
                                    "scheme": scheme, "track": track, "analysis_unit_id": cap, "attack_n": len(g),
                                    "original_fnr": orig_fnr, "shifted_fnr": shift_fnr,
                                    "delta_fnr_shifted_minus_original": shift_fnr - orig_fnr,
                                })
    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, pd.DataFrame()
    cap = seed_df.groupby(["dataset", "condition", "model", "scheme", "track", "analysis_unit_id"], as_index=False).agg(
        seeds=("seed", "nunique"), original_fnr=("original_fnr", "mean"), shifted_fnr=("shifted_fnr", "mean"),
        delta_fnr_shifted_minus_original=("delta_fnr_shifted_minus_original", "mean"),
    )
    rows = []
    for keys, g in cap.groupby(["dataset", "condition", "model", "scheme", "track"]):
        ds, condition, model, scheme, track = keys
        diff = g["delta_fnr_shifted_minus_original"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(diff, iterations=args.bootstrap_iterations, seed=111000 + len(rows))
        p, method, n = signflip_pvalue(diff, seed=112000 + len(rows))
        rows.append({
            "dataset": ds, "condition": condition, "model": model, "scheme": scheme, "track": track,
            "analysis_unit_n": n,
            "mean_original_fnr": float(g["original_fnr"].mean()),
            "mean_shifted_fnr": float(g["shifted_fnr"].mean()),
            "mean_delta_fnr_shifted_minus_original": float(np.mean(diff)),
            "analysis_unit_bootstrap_ci95_low": lo, "analysis_unit_bootstrap_ci95_high": hi,
            "signflip_p_raw": p, "signflip_method": method, "wilcoxon_p_secondary": wilcoxon_safe(diff),
        })
    stats_df = pd.DataFrame(rows)
    out = []
    # Separate family for each representation x shift scheme x calibration track across 12 dataset-model tests.
    for keys, g in stats_df.groupby(["condition", "scheme", "track"], sort=False):
        z = g.copy()
        z["signflip_p_holm"] = holm_adjust(z["signflip_p_raw"].to_numpy(dtype=float))
        z["signflip_holm_lt_0_05"] = z["signflip_p_holm"] < 0.05
        out.append(z)
    return cap, pd.concat(out, ignore_index=True)


# =============================================================================
# FINAL ANALYSIS TABLES / FIGURES (OBSERVATIONAL; NO ARTICLE DRAFTING)
# =============================================================================

def summarize_neural_metrics(dirs: dict[str, Path]) -> pd.DataFrame:
    runs = pd.read_csv(dirs["tables"] / "neural_run_metrics.csv")
    metric_cols = ["q_fnr", "q_recall", "q_fpr", "q_precision", "q_f1", "q_balanced_accuracy", "q_roc_auc", "q_pr_auc"]
    rows = []
    for keys, g in runs.groupby(["dataset", "condition", "model"]):
        row = dict(zip(["dataset", "condition", "model"], keys))
        row["seed_n"] = int(g["seed"].nunique())
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd_across_model_seeds"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
            row[f"{c}_min"] = float(g[c].min())
            row[f"{c}_max"] = float(g[c].max())
        row["train_time_sec_mean"] = float(g["train_time_sec"].mean())
        row["train_time_sec_sd"] = float(g["train_time_sec"].std(ddof=1)) if len(g) > 1 else 0.0
        row["rss_max_mb_max_observed"] = float(g["rss_max_mb"].max()) if "rss_max_mb" in g else float("nan")
        row["params"] = int(g["params"].iloc[0]) if "params" in g else -1
        row["input_dim"] = int(g["input_dim"].iloc[0]) if "input_dim" in g else -1
        row["epochs_used_mean"] = float(g["epochs_used"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "neural_metrics_summary.csv", index=False)
    return out


def plot_main_condition_fnr(summary: pd.DataFrame, dirs: dict[str, Path]) -> None:
    safe = summary[summary["condition"].isin(SAFE_CONDITION_NAMES)].copy()
    safe.to_csv(dirs["figures"] / "main_condition_fnr_source.csv", index=False)
    labels = [f"{d}\n{m}" for d in DATASET_ORDER for m in MODELS if m in set(safe["model"])]
    if not labels:
        return
    x = np.arange(len(labels), dtype=float)
    conditions = SAFE_CONDITION_NAMES
    width = 0.12
    fig, ax = plt.subplots(figsize=(16, 7))
    for ci, condition in enumerate(conditions):
        vals = []
        errs = []
        for d in DATASET_ORDER:
            for m in MODELS:
                if m not in set(safe["model"]):
                    continue
                r = safe[(safe["dataset"] == d) & (safe["model"] == m) & (safe["condition"] == condition)]
                vals.append(float(r["q_fnr_mean"].iloc[0]) if len(r) else np.nan)
                errs.append(float(r["q_fnr_sd_across_model_seeds"].iloc[0]) if len(r) else np.nan)
        ax.bar(x + (ci - (len(conditions)-1)/2) * width, vals, width, yerr=errs, capsize=2, label=condition)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("False-negative rate (q-only score)")
    ax.set_title("New certified-capture / parent-order evaluation: mean across model seeds")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "main_condition_fnr.png", dpi=250, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "main_condition_fnr.svg", bbox_inches="tight")
    plt.close(fig)


def plot_contrast_forest(stats_df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    if stats_df.empty:
        return
    for contrast, g in stats_df.groupby("contrast", sort=False):
        z = g.copy().sort_values(["dataset", "model"]).reset_index(drop=True)
        z["label"] = z["dataset"] + " | " + z["model"]
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(contrast))[:100]
        z.to_csv(dirs["figures"] / f"contrast_{stem}_source.csv", index=False)
        y = np.arange(len(z))
        x = z["mean_delta_fnr_right_minus_left"].to_numpy(dtype=float)
        lo = z["analysis_unit_bootstrap_ci95_low"].to_numpy(dtype=float)
        hi = z["analysis_unit_bootstrap_ci95_high"].to_numpy(dtype=float)
        lower = np.maximum(x - lo, 0)
        upper = np.maximum(hi - x, 0)
        fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(z))))
        ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(z["label"])
        ax.set_xlabel("Delta FNR = right condition - left condition")
        ax.set_title(f"Analysis-unit effect: {contrast}")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"contrast_{stem}.png", dpi=250, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"contrast_{stem}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_intervention_forest(stats_df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    if stats_df.empty:
        return
    # A_q_t is the cleanest direct timestamp-input intervention; plot both calibration tracks and schemes.
    src = stats_df[stats_df["condition"] == "A_q_t"].copy()
    src.to_csv(dirs["figures"] / "intervention_A_q_t_source.csv", index=False)
    for (scheme, track), g in src.groupby(["scheme", "track"]):
        z = g.sort_values(["dataset", "model"]).reset_index(drop=True)
        y = np.arange(len(z))
        x = z["mean_delta_fnr_shifted_minus_original"].to_numpy(dtype=float)
        lo = z["analysis_unit_bootstrap_ci95_low"].to_numpy(dtype=float)
        hi = z["analysis_unit_bootstrap_ci95_high"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(z))))
        ax.errorbar(x, y, xerr=np.vstack([np.maximum(x-lo, 0), np.maximum(hi-x, 0)]), fmt="o", capsize=3)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels((z["dataset"] + " | " + z["model"]).tolist())
        ax.set_xlabel("Delta FNR = shifted - original")
        ax.set_title(f"A_q_t timestamp translation: {scheme} | {track}")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{scheme}_{track}")
        fig.savefig(dirs["figures"] / f"intervention_A_q_t_{stem}.png", dpi=250, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"intervention_A_q_t_{stem}.svg", bbox_inches="tight")
        plt.close(fig)


def write_evidence_matrices(args: argparse.Namespace, dirs: dict[str, Path], contrast_stats: pd.DataFrame, intervention_stats: pd.DataFrame) -> None:
    evidence = [
        {"task": "T03_capture_provenance", "status": "EXECUTED", "evidence": "capture_mapping/capture_mapping_status.csv + splits/analysis_unit_semantics.csv", "boundary": "raw-capture mapping retained when certified; incomplete coverage triggers fixed parent-order blocks, never an invented session/capture claim"},
        {"task": "T04_precision_feature_contract", "status": "EXECUTED", "evidence": "metadata/feature_contract.json; tables/q_scaler_consistency.csv; precision-sensitivity condition", "boundary": "precision impact quantified under new protocol; historical v3 retained"},
        {"task": "T05_acquisition_split", "status": "EXECUTED", "evidence": "splits/split_summary.csv; splits/*/split_manifest.csv.gz", "boundary": "capture-disjoint only where whole capture files support all normal roles; otherwise explicitly blocked"},
        {"task": "T07_behavior_context", "status": "EXECUTED", "evidence": "behavior/*_causal_behavior.parquet; behavior/translation_invariance_unit_test.json", "boundary": "past-only flow-local features; no time-since-capture-start"},
        {"task": "T09_technical_pilot", "status": "EXECUTED", "evidence": "pilot_technical_only/pilot_results.csv", "boundary": "development-normal technical QA; not scientific outcome"},
        {"task": "T10_main_conditions_baselines", "status": "EXECUTED", "evidence": "tables/neural_run_metrics.csv; tables/baseline_metrics.csv", "boundary": "materialized samples; four packet-reconstruction architectures"},
        {"task": "T11_timestamp_shift", "status": "SKIPPED_BY_USER" if args.skip_interventions else "EXECUTED", "evidence": "tables/intervention_run_metrics.csv", "boundary": "synthetic timestamp translation; not deployment validation"},
        {"task": "T12_security_residual", "status": "EXECUTED", "evidence": "tables/security_* including shared cross-model transitions; tables/residual_*", "boundary": "packet/analysis-unit diagnostics; residual changes are descriptive reconstruction-error decomposition, not causal feature importance"},
        {"task": "T13_statistics_claims", "status": "EXECUTED", "evidence": "tables/analysis_unit_contrast_statistics.csv; tables/intervention_analysis_unit_statistics.csv", "boundary": "analysis-unit inference; raw capture files are used only when completely certified, otherwise fixed parent-order blocks are explicitly non-session/non-capture units"},
        {"task": "T15_reproducibility_package", "status": "PENDING_PACKAGE_STAGE", "evidence": "review_package + artifact inventory created next", "boundary": "heavy run artifacts remain in full output root"},
        {"task": "T02_literature_novelty", "status": "NON_EXECUTABLE_REMAINS", "evidence": "human/chat primary-source literature review", "boundary": "script cannot establish novelty"},
        {"task": "T14_manuscript_rewrite_RQ_closure", "status": "NON_EXECUTABLE_REMAINS", "evidence": "requires supervisor interpretation and explicit user confirmation", "boundary": "no article sections drafted automatically"},
        {"task": "T16_submission_promotor", "status": "NON_EXECUTABLE_REMAINS", "evidence": "human decision", "boundary": "not automatable by experiment script"},
    ]
    pd.DataFrame(evidence).to_csv(dirs["tables"] / "task_evidence_matrix.csv", index=False)

    rq = [
        {"candidate_question": "E1 acquisition-time structure and timestamp representation", "evidence": "capture mapping; timestamp-only baseline; precision sensitivity", "status": "MEASURED_AWAITING_SUPERVISOR_INTERPRETATION"},
        {"candidate_question": "E2 timestamp influence on common non-timestamp reconstruction", "evidence": "A vs B and C vs D q-only contrasts; q residual transitions", "status": "MEASURED_AWAITING_SUPERVISOR_INTERPRETATION"},
        {"candidate_question": "E3 detection and FPR consequences under acquisition/time shift", "evidence": "held-out new split; timestamp translations; fixed and recalibrated threshold tracks", "status": "MEASURED_AWAITING_SUPERVISOR_INTERPRETATION" if not args.skip_interventions else "PARTIAL_INTERVENTION_SKIPPED"},
        {"candidate_question": "E4 behavioral temporal context without absolute chronology", "evidence": "D vs B and C vs D; PCA baseline; FNR/FPR/security analyses", "status": "MEASURED_AWAITING_SUPERVISOR_INTERPRETATION"},
    ]
    pd.DataFrame(rq).to_csv(dirs["tables"] / "candidate_research_question_evidence.csv", index=False)

    # Machine-readable quantitative index; deliberately no novelty or universal claim.
    index = {
        "contrast_statistics_rows": int(len(contrast_stats)),
        "intervention_statistics_rows": int(len(intervention_stats)),
        "interpretation_state": "AWAITING_SUPERVISOR_REVIEW_BEFORE_MANUSCRIPT_DRAFTING",
        "historical_evidence_retained": True,
        "new_evidence_class": EVIDENCE_CLASS,
    }
    save_json(dirs["metadata"] / "analysis_interpretation_gate.json", index)


def analyze_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, q_features: list[str]):
    if stage_is_complete(dirs, "analyze", protocol_hash):
        log(root, "[RESUME] analyze already complete")
        return
    if not stage_is_complete(dirs, "intervene", protocol_hash):
        raise RuntimeError("intervene stage (completed or explicitly skipped) must precede analyze")
    log(root, "START analyze")
    summary = summarize_neural_metrics(dirs)
    unit_seed, transitions, residuals = planned_contrast_analysis(args, dirs, q_features)
    per_unit, contrast_stats = summarize_analysis_unit_contrasts(unit_seed, args)
    unit_seed.to_csv(dirs["tables"] / "analysis_unit_contrast_seed_level.csv", index=False)
    per_unit.to_csv(dirs["tables"] / "analysis_unit_contrast_per_unit.csv", index=False)
    contrast_stats.to_csv(dirs["tables"] / "analysis_unit_contrast_statistics.csv", index=False)
    transitions.to_csv(dirs["tables"] / "security_packet_transitions.csv", index=False)
    residuals.to_csv(dirs["tables"] / "residual_switched_packet_decomposition.csv", index=False)
    shared_transitions = shared_cross_model_transition_analysis(args, dirs)
    shared_transitions.to_csv(dirs["tables"] / "security_shared_cross_model_transitions.csv", index=False)

    conditions = list(SAFE_CONDITION_NAMES)
    if not args.skip_precision_sensitivity:
        conditions.append(PRECISION_CONDITION_NAME)
    groups, ensemble, shared = attack_group_and_ensemble_analysis(args, dirs, conditions)
    groups.to_csv(dirs["tables"] / "security_attack_group_analysis_unit_recall.csv", index=False)
    ensemble.to_csv(dirs["tables"] / "security_or_ensemble.csv", index=False)
    shared.to_csv(dirs["tables"] / "security_common_missed_across_models.csv", index=False)

    int_unit, int_stats = intervention_analysis_unit_analysis(args, dirs)
    int_unit.to_csv(dirs["tables"] / "intervention_analysis_unit_per_unit.csv", index=False)
    int_stats.to_csv(dirs["tables"] / "intervention_analysis_unit_statistics.csv", index=False)

    plot_main_condition_fnr(summary, dirs)
    plot_contrast_forest(contrast_stats, dirs)
    plot_intervention_forest(int_stats, dirs)
    write_evidence_matrices(args, dirs, contrast_stats, int_stats)

    analysis_summary = {
        "completed_utc": utc_now(),
        "neural_summary_rows": int(len(summary)),
        "planned_contrast_analysis_unit_seed_rows": int(len(unit_seed)),
        "planned_contrast_statistics_rows": int(len(contrast_stats)),
        "packet_transition_rows": int(len(transitions)),
        "shared_cross_model_transition_rows": int(len(shared_transitions)),
        "residual_decomposition_rows": int(len(residuals)),
        "attack_group_rows": int(len(groups)),
        "ensemble_rows": int(len(ensemble)),
        "common_missed_rows": int(len(shared)),
        "intervention_statistics_rows": int(len(int_stats)),
        "statistical_boundary": "analysis units follow each dataset split mode: certified capture files only when complete mapping exists, otherwise fixed parent-order blocks; session independence is never asserted",
        "drafting_boundary": "These tables are evidence, not manuscript prose. Supervisor review is required before RQ/title/contribution rewrite.",
    }
    save_json(dirs["metadata"] / "analysis_summary.json", analysis_summary)
    mark_stage(dirs, "analyze", protocol_hash, analysis_summary)
    log(root, "END analyze [COMPLETED]")


# =============================================================================
# REPRODUCIBILITY INVENTORY / REVIEW PACKAGE
# =============================================================================

def artifact_category(rel: Path) -> str:
    return rel.parts[0] if rel.parts else "root"


def build_artifact_inventory(root: Path, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    skip_names = {"artifact_inventory.csv", "CHECKSUMS.csv"}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.name in skip_names:
            continue
        if rel.parts and rel.parts[0] == "review_package":
            continue
        rows.append({
            "relative_path": str(rel),
            "category": artifact_category(rel),
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        })
    inv = pd.DataFrame(rows)
    inv.to_csv(dirs["review"] / "artifact_inventory.csv", index=False)
    inv.to_csv(root / "CHECKSUMS.csv", index=False)
    return inv


def write_results_index(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str, inventory: pd.DataFrame | None = None) -> None:
    split = pd.read_csv(dirs["splits"] / "split_summary.csv")
    train = pd.read_csv(dirs["tables"] / "neural_run_metrics.csv")
    baseline = pd.read_csv(dirs["tables"] / "baseline_metrics.csv")
    text = [
        "# TEMPORAL_AUDIT Temporal Upgrade - Evidence Index",
        "",
        f"Protocol hash: `{protocol_hash}`",
        f"Evidence class: `{EVIDENCE_CLASS}`",
        "",
        "This directory is a new post-audit experimental evidence block. It does not overwrite the historical v3 controlled-intervention evidence or the legacy 720-run experiment.",
        "",
        "## Executed experimental blocks",
        "",
        f"- Capture/source block reconstruction: `{dirs['capture']}`",
        f"- New acquisition-aware split manifests: `{dirs['splits']}`",
        f"- Causal flow-local behavior features: `{dirs['behavior']}`",
        f"- Prepared materialized data: `{dirs['prepared']}`",
        f"- Technical development-normal pilot: `{dirs['pilot']}`",
        f"- Baselines: `{dirs['baselines']}`",
        f"- Neural model runs and saved weights: `{dirs['runs']}`",
        f"- Timestamp-shift interventions: `{dirs['interventions']}`",
        f"- Final analysis tables: `{dirs['tables']}`",
        f"- Figures and figure-source tables: `{dirs['figures']}`",
        "",
        "## Split modes actually selected",
        "",
        "```text",
        split[["dataset", "split_mode", "fit_normal_n", "earlystop_normal_n", "calibration_normal_n", "test_normal_n", "test_attack_n"]].to_string(index=False),
        "```",
        "",
        "## Run counts",
        "",
        f"- Neural model runs: {len(train)}",
        f"- Baseline metric rows: {len(baseline)}",
        f"- Model seeds: {sorted(train['seed'].unique().tolist())}",
        "",
        "## Evidence boundaries",
        "",
        "- Capture-file mapping is not called session independence.",
        "- The new held-out test is held out from this new fitting/tuning/calibration protocol; materialized rows existed in historical experiments.",
        "- LSTM/CNN/SNN remain heterogeneous packet-reconstruction architectures, not cross-packet sequence-model comparisons.",
        "- Synthetic timestamp shifts are mechanism/robustness tests, not deployment validation.",
        "- Residual changes are descriptive reconstruction-error decomposition, not causal feature importance.",
        "- Literature novelty review, article drafting, RQ finalization, and submission remain non-executable supervisor/human tasks.",
        "",
        "## Main review files",
        "",
        "- tables/task_evidence_matrix.csv",
        "- tables/candidate_research_question_evidence.csv",
        "- tables/baseline_metrics.csv",
        "- tables/neural_metrics_summary.csv",
        "- tables/analysis_unit_contrast_statistics.csv",
        "- tables/intervention_analysis_unit_statistics.csv",
        "- tables/security_packet_transitions.csv",
        "- tables/security_attack_group_analysis_unit_recall.csv",
        "- tables/security_common_missed_across_models.csv",
        "- tables/security_or_ensemble.csv",
        "- tables/residual_switched_packet_decomposition.csv",
        "- metadata/protocol_freeze.json",
        "- metadata/analysis_summary.json",
        "- CHECKSUMS.csv",
    ]
    if inventory is not None:
        text += ["", f"Full artifact inventory entries: {len(inventory)}"]
    (root / "RESULTS_INDEX.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def create_review_zip(root: Path, dirs: dict[str, Path]) -> Path:
    zip_path = dirs["review"] / "TEMPORAL_AUDIT_E2E_REVIEW.zip"
    if zip_path.exists():
        zip_path.rename(zip_path.with_name(zip_path.stem + f"__OLD_{utc_tag()}.zip"))
    allowed_suffixes = {".csv", ".json", ".md", ".txt", ".png", ".svg", ".py", ".gz"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # Always include the compact high-value roots; exclude heavy packet/residual/prepared/model artifacts.
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p == zip_path:
                continue
            rel = p.relative_to(root)
            if rel.parts and rel.parts[0] in {"runs", "prepared_data"}:
                continue
            if rel.parts and rel.parts[0] == "baselines" and "test_predictions" in p.name:
                continue
            if rel.parts and rel.parts[0] == "interventions" and "test_predictions_shifted" in p.name:
                continue
            if rel.parts and rel.parts[0] == "capture_mapping" and "prefix_cache" in rel.parts:
                continue
            if p.suffix.lower() not in allowed_suffixes and not p.name.endswith(".csv.gz"):
                continue
            # Review ZIP stays lightweight; full inventory points to anything omitted.
            if p.stat().st_size > 25 * 1024 * 1024:
                continue
            zf.write(p, arcname=str(rel))
        inv = dirs["review"] / "artifact_inventory.csv"
        if inv.exists():
            zf.write(inv, arcname="review_package/artifact_inventory.csv")
    return zip_path


def package_stage(args: argparse.Namespace, root: Path, dirs: dict[str, Path], protocol_hash: str):
    if stage_is_complete(dirs, "package", protocol_hash):
        log(root, "[RESUME] package already complete")
        return
    if not stage_is_complete(dirs, "analyze", protocol_hash):
        raise RuntimeError("analyze must complete before package")
    log(root, "START package")
    # Create a pre-inventory results index first so it is itself hashed.
    write_results_index(args, root, dirs, protocol_hash, inventory=None)
    inventory = build_artifact_inventory(root, dirs)
    write_results_index(args, root, dirs, protocol_hash, inventory=inventory)
    # Re-hash the rewritten RESULTS_INDEX and refresh its row in CHECKSUMS for exactness.
    inventory = build_artifact_inventory(root, dirs)
    review_zip = create_review_zip(root, dirs)
    review_hash = sha256_file(review_zip)
    save_json(dirs["review"] / "review_package_manifest.json", {
        "review_zip": str(review_zip),
        "review_zip_sha256": review_hash,
        "full_output_root": str(root),
        "full_artifact_inventory": str(dirs["review"] / "artifact_inventory.csv"),
        "heavy_artifacts_are_not_deleted": True,
    })
    (root / "UPLOAD_THIS_ZIP.txt").write_text(str(review_zip) + "\n", encoding="utf-8")
    final = {
        "status": "COMPLETED",
        "completed_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "protocol_hash": protocol_hash,
        "evidence_class": EVIDENCE_CLASS,
        "review_zip": str(review_zip),
        "review_zip_sha256": review_hash,
        "artifact_inventory_rows": int(len(inventory)),
        "historical_evidence_overwritten": False,
        "article_drafted": False,
        "remaining_non_executable_tasks": ["literature/novelty final review", "supervisor interpretation", "RQ/manuscript rewrite after explicit confirmation", "submission decision"],
    }
    save_json(root / "RUN_COMPLETE.json", final)
    mark_stage(dirs, "package", protocol_hash, {"review_zip": str(review_zip), "review_zip_sha256": review_hash})
    log(root, f"END package [COMPLETED] REVIEW_ZIP={review_zip}")


# =============================================================================
# MAIN DISPATCH / FAIL-CLOSED PROTOCOL FREEZE
# =============================================================================

def _require_previous_stages(dirs: dict[str, Path], stage: str, protocol_hash: str) -> None:
    idx = STAGE_ORDER.index(stage)
    for prior in STAGE_ORDER[:idx]:
        if prior == "preflight":
            pass
        if not stage_is_complete(dirs, prior, protocol_hash):
            raise RuntimeError(f"Stage {stage} requires completed prior stage {prior} under the same protocol hash")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    dirs = ensure_dirs(root)
    protocol = static_protocol(args)
    protocol["implementation_sha256"] = sha256_file(Path(__file__).resolve())
    protocol["capture_mapping_parameters"] = {
        "capture_prefix_frames": int(args.capture_prefix_frames),
        "parent_scan_batch": int(args.parent_scan_batch),
        "min_offset_support": MIN_CAPTURE_OFFSET_SUPPORT,
        "min_offset_share": MIN_CAPTURE_OFFSET_SHARE,
        "min_offset_dominance": MIN_CAPTURE_OFFSET_DOMINANCE,
        "block_length_tolerance_frames": BLOCK_LENGTH_TOLERANCE_FRAMES,
        "fallback_parent_order_block_rows": int(args.fallback_block_rows),
    }
    protocol["analysis_parameters"] = {"bootstrap_iterations": int(args.bootstrap_iterations), "alpha": ALPHA}
    protocol_hash = stable_json_hash(protocol)

    freeze = dirs["metadata"] / "protocol_freeze.json"
    if freeze.exists():
        old = load_json(freeze)
        old_hash = old.get("protocol_hash")
        if old_hash != protocol_hash:
            raise RuntimeError(
                f"Output root already contains a different frozen protocol ({old_hash}); current={protocol_hash}. "
                "Use a new --root instead of mixing evidence."
            )

    log(root, f"PIPELINE START stage={args.stage} protocol_hash={protocol_hash} root={root}")

    if args.stage == "all":
        preflight(args, root, dirs, protocol, protocol_hash)
        capture_map_stage(args, root, dirs, protocol_hash)
        split_stage(args, root, dirs, protocol_hash)
        legacy = load_legacy_driver()
        q_features, specs = resolve_condition_specs(legacy)
        behavior_stage(args, root, dirs, protocol_hash, legacy, q_features)
        pilot_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
        baseline_stage(args, root, dirs, protocol_hash, q_features, specs)
        train_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
        intervention_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
        analyze_stage(args, root, dirs, protocol_hash, q_features)
        package_stage(args, root, dirs, protocol_hash)
    else:
        stage = args.stage
        if stage == "preflight":
            preflight(args, root, dirs, protocol, protocol_hash)
        else:
            _require_previous_stages(dirs, stage, protocol_hash)
            legacy = None
            q_features = None
            specs = None
            if stage in {"behavior", "pilot", "baselines", "train", "intervene"}:
                legacy = load_legacy_driver()
                q_features, specs = resolve_condition_specs(legacy)
            if stage == "capture_map":
                capture_map_stage(args, root, dirs, protocol_hash)
            elif stage == "split":
                split_stage(args, root, dirs, protocol_hash)
            elif stage == "behavior":
                assert legacy is not None and q_features is not None
                behavior_stage(args, root, dirs, protocol_hash, legacy, q_features)
            elif stage == "pilot":
                assert legacy is not None and q_features is not None and specs is not None
                pilot_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
            elif stage == "baselines":
                assert q_features is not None and specs is not None
                baseline_stage(args, root, dirs, protocol_hash, q_features, specs)
            elif stage == "train":
                assert legacy is not None and q_features is not None and specs is not None
                train_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
            elif stage == "intervene":
                assert legacy is not None and q_features is not None and specs is not None
                intervention_stage(args, root, dirs, protocol_hash, legacy, q_features, specs)
            elif stage == "analyze":
                # q features can be loaded from the archived feature contract; avoid importing legacy just for analysis.
                fc = load_json(dirs["metadata"] / "feature_contract.json")
                analyze_stage(args, root, dirs, protocol_hash, list(fc["q_features"]))
            elif stage == "package":
                package_stage(args, root, dirs, protocol_hash)
            else:
                raise ValueError(stage)

    log(root, f"PIPELINE END stage={args.stage}")
    if (root / "UPLOAD_THIS_ZIP.txt").exists():
        print("UPLOAD THIS ZIP:", (root / "UPLOAD_THIS_ZIP.txt").read_text(encoding="utf-8").strip(), flush=True)


if __name__ == "__main__":
    main()
