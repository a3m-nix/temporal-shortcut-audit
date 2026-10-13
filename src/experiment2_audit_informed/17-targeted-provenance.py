#!/usr/bin/env python3
"""T03b: read-only, bounded provenance investigation for the TEMPORAL_AUDIT project.

No training, no model imports, no pickle loading, no source-script execution,
no creation of experimental splits, and no live packet capture.

A sample-to-parent content match is NOT a sample-to-PCAP identity proof.
PCAP time/length matches remain candidates, including when locally unique.
All scan limits, incomplete stages and alternative index hypotheses are exported.
Dependencies on the server: numpy, pandas, pyarrow. TShark is optional.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

VERSION = "1.0.0"
BASE_DEFAULT = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
LEGACY_NAME = "0-temporal-feature-audit-v4-allminmax-3seed10fold"
PRIOR_RUN = "run_20260914T011645_984156Z_2bc493"
DATASETS = ("edge_iiotset", "ciciot2023", "ton_iot")
PARENT_NAMES = {
    "edge_iiotset": "edge-iiotset_bidir.parquet",
    "ciciot2023": "ciciot2023_bidir.parquet",
    "ton_iot": "ton_iot_bidir.parquet",
}
FEATURES = [
    "frame.len", "frame.time_epoch", "ip.version", "ip.hdr_len",
    "ip.dsfield.dscp", "ip.len", "ip.id", "ip.frag_offset", "ip.flags.df",
    "ip.flags.mf", "ip.ttl", "ip.proto", "tcp.srcport", "tcp.dstport",
    "tcp.seq", "tcp.ack", "tcp.hdr_len", "tcp.len", "tcp.window_size",
    "tcp.flags.fin", "tcp.flags.syn", "tcp.flags.reset", "tcp.flags.push",
    "tcp.flags.ack", "tcp.flags.urg", "tcp.flags.ns", "tcp.flags.cwr",
    "udp.srcport", "udp.dstport", "udp.length", "icmp.type", "icmp.code",
    "data.len", "inter_arrival_time",
]
EPOCH = "frame.time_epoch"
IAT = "inter_arrival_time"
LABELS = ("label", "label_type")
CAPTURE_COLS = (
    "capture_id", "capture_file", "pcap_file", "pcap_path", "source_file",
    "source_path", "filename", "file_name", "capture", "pcap", "file_id",
)
SESSION_COLS = ("session_id", "episode_id", "session", "episode", "scenario_id")
FRAME_COLS = ("frame.number", "packet_index", "packet_number", "frame_number")
KEY_COLS = ("source_row_index", "original_index", "row_id", "__index_level_0__", "index")
TEXT_EXTS = {".py", ".ipynb", ".json", ".yaml", ".yml", ".txt", ".md", ".sh", ".csv", ".tsv"}
PRIOR_TABLES = ("02_file_inventory.csv", "30_code_hints.csv", "31_source_path_candidates.csv", "32_raw_file_candidates.csv", "34_feasibility.csv")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [safe_json(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    return x


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(safe_json(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stat_signature(path: Path) -> tuple[int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def strict_int(value: Any) -> int | None:
    if value is None or value is pd.NA or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        k = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer() or abs(value) > 2**53:
            return None
        k = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        k = int(value)
    else:
        return None
    return k if -(2**63) <= k < 2**63 else None


def numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def exact_numeric(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a == b) | (np.isnan(a) & np.isnan(b))


def exact_labels(a: pd.Series, b: pd.Series) -> np.ndarray:
    # Missing labels never count as positive verification.
    return (a.notna() & b.notna() & a.astype("string").eq(b.astype("string"))).fillna(False).to_numpy(dtype=bool)


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def ds_hint(path: str) -> str:
    p = path.lower()
    hits = []
    if any(x in p for x in ("edge_iiot", "edge-iiot", "edgeiiot")):
        hits.append("edge_iiotset")
    if "ciciot" in p:
        hits.append("ciciot2023")
    if any(x in p for x in ("ton_iot", "ton-iot", "toniot")):
        hits.append("ton_iot")
    return hits[0] if len(hits) == 1 else "UNASSIGNED"


def normalized_text(path: Path, limit: int) -> tuple[str, bool]:
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    truncated = len(raw) > limit
    text = raw[:limit].decode("utf-8-sig", errors="replace")
    if path.suffix.lower() == ".ipynb" and not truncated:
        doc = json.loads(text)
        chunks = []
        for i, cell in enumerate(doc.get("cells", [])):
            if cell.get("cell_type") == "code":
                src = cell.get("source", [])
                chunks.append(f"# NOTEBOOK CELL {i}\n" + ("".join(src) if isinstance(src, list) else str(src)))
        text = "\n\n".join(chunks)
    return text, truncated


class Report:
    def __init__(self, output_root: Path):
        output_root.mkdir(parents=True, exist_ok=True)
        self.path = output_root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:6])
        self.path.mkdir(exist_ok=False)
        self.cache = self.path / "local_header_cache"
        self.cache.mkdir()
        self.rows: dict[str, list[dict]] = defaultdict(list)
        self.definitions: dict[str, str] = {
            "00_findings.csv": "Warnings/errors and interpretation boundaries; REVIEW is not a training failure",
            "01_stage_status.csv": "Stages completed/failed and elapsed time",
            "02_inventory.csv": "Source existence, size, and full-hash status; large parent files may be un-hashed",
            "03_prior_inputs.csv": "Checksums of prior audit tables actually read",
            "04_search_coverage.csv": "Roots, depths, file/selection limits and completeness of bounded searches",
            "05_source_files.csv": "Static source-file hashes, truncation and evidence coverage",
            "06_code_evidence.csv": "Quoted source blocks for index/IAT/extractor review; not executed",
            "07_path_references.csv": "Literal source path references only",
            "08_capture_candidates.csv": "Raw/derived-name hints, sizes, and scan selection; filenames do not identify sessions",
            "09_sample_metadata.csv": "Materialized sample shape and hash continuity against the previous audit",
            "10_parent_metadata.csv": "Parent schema, row counts, stored pandas index and group columns",
            "11_parent_read_coverage.csv": "Explicit key scans and sparse row reads, including caps/deadlines",
            "12_parent_field_comparison.csv": "Exact numeric/label comparisons and separately labeled sanitization diagnostic",
            "13_parent_method_summary.csv": "Content validation under each tested index hypothesis",
            "14_parent_decisions.csv": "Accepted, ambiguous or unresolved sample-to-parent content links; never PCAP proof",
            "16_iat_predecessor_summary.csv": "Stored IAT versus adjacent global parent rows, not inferred flow history",
            "17_negative_iat_context.csv": "Negative IAT observations at content-validated parent positions",
            "18_reported_parent_groups.csv": "Literal parent group values associated with selected samples",
            "18_reported_parent_group_coverage.csv": "Group value export caps and missingness",
            "19_unresolved_mapping_examples.csv": "Bounded examples for failed or ambiguous index hypotheses",
            "20_pcap_scans.csv": "Offline header read command, timeout/error, prefix/full-EOF scope",
            "21_header_match_candidates.csv": "Timestamp-and-length candidate pairs; not unique packet identity proofs",
            "22_header_candidate_summary.csv": "Candidate counts in scanned records only, not full-capture membership",
            "22_header_export_coverage.csv": "Pair export limit; counts and exported example rows are distinct",
            "30_feasibility.csv": "Unresolved evidence for source identity, complete history and independent sessions",
        }
        self.findings: list[dict] = []
        self.stages: list[dict] = []
        self.bigfiles: list[str] = []
        self.logger = logging.getLogger("temporal_audit.provenance." + self.path.name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.path / "audit.log", encoding="utf-8")):
            h.setFormatter(fmt)
            self.logger.addHandler(h)

    def add(self, table: str, **row: Any) -> None:
        self.rows[table].append(safe_json(row))

    def finding(self, severity: str, code: str, scope: str, detail: str) -> None:
        self.findings.append(dict(severity=severity, code=code, scope=scope, detail=detail))
        self.logger.info("%s %s [%s] %s", severity, code, scope, detail)

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        self.logger.info("START %s", name)
        status, error = "COMPLETED", ""
        try:
            yield
        except Exception as exc:
            status, error = "ERROR", f"{type(exc).__name__}: {exc}"
            self.finding("ERROR", "STAGE_ERROR", name, error)
            self.logger.error(traceback.format_exc())
        finally:
            self.stages.append(dict(stage=name, status=status, seconds=round(time.monotonic()-t0, 3), error=error))
            self.logger.info("END %s [%s]", name, status)

    def frame(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.path / name, index=False, compression="gzip" if name.endswith(".gz") else None)
        self.bigfiles.append(name)

    def finish(self, status: str = "COMPLETED") -> Path:
        self.rows["00_findings.csv"] = self.findings
        self.rows["01_stage_status.csv"] = self.stages
        for name, rows in self.rows.items():
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["status", "note"])
            df.to_csv(self.path / name, index=False, encoding="utf-8-sig")
        if "00_findings.csv" not in self.rows:
            raise RuntimeError("Missing findings table")
        dict_rows = [dict(file=n, description=self.definitions.get(n, "Evidence table; inspect scope/status columns")) for n in sorted(set(self.rows) | set(self.bigfiles))]
        pd.DataFrame(dict_rows).to_csv(self.path / "TABLE_DICTIONARY.csv", index=False)
        errors = sum(x["status"] == "ERROR" for x in self.stages)
        final_status = "COMPLETED_WITH_ERRORS" if errors and status == "COMPLETED" else status
        write_json(self.path / "RUN_STATUS.json", dict(
            script_version=VERSION, status=final_status, completed_utc=utc(),
            stage_errors=errors, finding_counts=dict(Counter(x["severity"] for x in self.findings)),
            no_training=True, no_input_mutation=True, no_live_capture=True,
            session_independence="NOT_CERTIFIED", training_readiness="REQUIRES_REVIEW",
            interpretation="Completion is not proof of provenance, completeness of traffic, or scientific validity.",
        ))
        (self.path / "READ_ME_FIRST.md").write_text(REPORT_README, encoding="utf-8")
        included = [p for p in self.path.iterdir() if p.is_file() and p.name not in ("SHA256SUMS.txt", "audit.log")]
        lines = [f"{sha256(p)}  {p.name}" for p in sorted(included)]
        (self.path / "SHA256SUMS.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        for h in self.logger.handlers:
            h.flush()
        out = self.path.parent / ("TEMPORAL_AUDIT_PROVENANCE_RESULTS_" + self.path.name + ".zip")
        with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.path.iterdir()):
                if p.is_file():
                    z.write(p, p.name)
        self.logger.info("REPORT DIRECTORY: %s", self.path)
        self.logger.info("UPLOAD THIS ZIP: %s", out)
        self.logger.info("STATUS: %s; training was not run.", final_status)
        return out


REPORT_README = """# T03b provenance report

Read 00_findings.csv, 01_stage_status.csv, 13_parent_method_summary.csv,
14_parent_decisions.csv, 20_pcap_scans.csv and 30_feasibility.csv first.

Evidence levels:
- Paths/code snippets/column names are clues, not packet or session proofs.
- A source-row positional hypothesis must pass content comparison.
- Physical pandas index/source-ID hypotheses are checked separately when present.
- RAW_COMPLETE_CONTENT_MATCH means exact equality on all 34 model fields and both
  label fields under the tested index hypothesis. It is not raw-PCAP identity.
- SANITIZED_COMPLETE_CONTENT_MATCH requires exact epoch, frame.len and labels;
  only the explicitly reported numeric nonfinite-to-zero diagnostic differs.
  It is not proof of which preprocessing code was used historically.
- Matching only a subset of features is PARTIAL, not a confirmed full mapping.
- Different hypotheses with different matching positions remain AMBIGUOUS.
- Parent row p-1 is a global-table predecessor, NOT automatically a same-flow or
  same-capture predecessor. IAT agreement is descriptive, not a certified rule.
- PCAP matches use timestamp tolerance and frame length only. They are candidates,
  never proofs of unique packet identity, even when one candidate is found.
- PCAP prefix absence does not mean a packet is absent from the full file.
- Capture filenames, label groups, time intervals and row groups are NOT certified
  independent sessions. All session definitions still need review.

Reads use bounded batches. Metadata/deadlines/limits and partial coverage are
reported. Parent bytes are not fully hashed above the configured hash-size cap;
this is explicitly distinguished from a full SHA-256. Sources are never rewritten.

The ZIP contains CSV/JSON/log reports and compressed selected row mappings.
Raw header TSV files remain in local_header_cache on the server and are NOT
included. No raw packet payload, PCAP, parent Parquet, model or private key is copied.
No training, no source-code execution, no pickle loading, and no live capture.
"""


def inventory(path: Path, report: Report, role: str, hash_limit: int, dataset: str = "") -> dict:
    row = dict(path=str(path), role=role, dataset=dataset, exists=False)
    try:
        s = path.stat()
        row.update(exists=True, is_file=path.is_file(), bytes=s.st_size, modified_ns=s.st_mtime_ns)
        if path.is_file() and s.st_size <= hash_limit:
            row.update(sha256=sha256(path), hash_scope="FULL_FILE")
        elif path.is_file():
            row["hash_scope"] = "NOT_HASHED_SIZE_LIMIT"
        else:
            row["hash_scope"] = "DIRECTORY"
    except OSError as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    report.add("02_inventory.csv", **row)
    return row


def read_prior(root: Path, report: Report) -> dict[str, pd.DataFrame]:
    data = {}
    hashes = {}
    checksum_path = root / "SHA256SUMS.txt"
    if checksum_path.is_file():
        for line in checksum_path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                hashes[parts[1].lstrip(" *")] = parts[0]
    for name in PRIOR_TABLES:
        p = root / name
        if not p.is_file():
            report.add("03_prior_inputs.csv", path=str(p), status="MISSING")
            continue
        observed = sha256(p)
        expected = hashes.get(name)
        if expected and observed != expected:
            report.add("03_prior_inputs.csv", path=str(p), status="CHECKSUM_MISMATCH", expected=expected, observed=observed)
            report.finding("ERROR", "PRIOR_CHECKSUM_MISMATCH", name, "Table not used as a source hint")
            continue
        if p.stat().st_size > 32 * 1024**2:
            report.add("03_prior_inputs.csv", path=str(p), status="SIZE_LIMIT_NOT_READ")
            continue
        data[name] = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
        report.add("03_prior_inputs.csv", path=str(p), status="READ_MATCHED_HASH" if expected else "READ_HASH_UNVERIFIED", sha256=observed, rows=len(data[name]))
    return data


def verify_previous_sample(path: Path, observed: str, prior: dict) -> str:
    df = prior.get("02_file_inventory.csv", pd.DataFrame())
    if not {"path", "sha256"}.issubset(df.columns):
        return "NO_PRIOR_HASH_AVAILABLE"
    values = set(df.loc[df["path"] == str(path), "sha256"].astype(str))
    expected = {h for h in values if re.fullmatch(r"[0-9a-fA-F]{64}", h)}
    if not expected:
        return "NO_PRIOR_HASH_AVAILABLE"
    if len(expected) != 1 or observed not in expected:
        raise ValueError("Materialized sample hash differs from prior audit; refusing a legacy lineage claim")
    return "MATCHES_PRIOR_AUDIT_FULL_SHA256"


class ParquetReader:
    """Arrow adapter. No training-driver import or pandas index reconstruction."""
    def __init__(self, path: Path):
        import pyarrow.parquet as pq
        self.path = path
        self.pf = pq.ParquetFile(path)
        self.columns = list(self.pf.schema_arrow.names)
        self.n = self.pf.metadata.num_rows
        self.group_sizes = [self.pf.metadata.row_group(i).num_rows for i in range(self.pf.num_row_groups)]
        self.types = {f.name: str(f.type) for f in self.pf.schema_arrow}
        md = self.pf.schema_arrow.metadata or {}
        self.pandas = json.loads(md.get(b"pandas", b"{}").decode("utf-8"))

    def batches(self, columns: list[str], batch_size: int, groups: list[int] | None = None):
        which = list(range(len(self.group_sizes))) if groups is None else groups
        offsets = np.r_[0, np.cumsum(self.group_sizes)]
        for rg in which:
            pos = int(offsets[rg])
            for batch in self.pf.iter_batches(batch_size=batch_size, row_groups=[rg], columns=columns, use_threads=False, use_pandas_metadata=False):
                df = batch.to_pandas(ignore_metadata=True)
                yield pos, df
                pos += len(df)

    def close(self):
        close = getattr(self.pf, "close", None)
        if close:
            close()


def read_positions(reader, positions: Iterable[int], columns: list[str], args, deadline: float) -> tuple[pd.DataFrame, dict]:
    wanted = np.array(sorted(set(int(p) for p in positions if 0 <= int(p) < reader.n)), dtype=np.int64)
    meta = dict(requested_positions=len(wanted), returned_positions=0, rows_decoded=0, complete=True, stop_reason="")
    if not len(wanted):
        return pd.DataFrame(columns=["_parent_position"] + columns).set_index("_parent_position"), meta
    starts = np.r_[0, np.cumsum(reader.group_sizes)]
    groups = np.unique(np.searchsorted(starts[1:], wanted, side="right")).astype(int).tolist()
    parts = []
    for start, df in reader.batches(columns, args.batch_size, groups):
        if time.monotonic() > deadline or meta["rows_decoded"] + len(df) > args.max_parent_rows_decoded:
            meta.update(complete=False, stop_reason="DEADLINE_OR_ROWS_LIMIT")
            break
        meta["rows_decoded"] += len(df)
        lo, hi = np.searchsorted(wanted, [start, start+len(df)])
        if hi > lo:
            take = wanted[lo:hi]
            sub = df.iloc[take-start].copy()
            sub.insert(0, "_parent_position", take)
            parts.append(sub)
    result = pd.concat(parts, ignore_index=True).set_index("_parent_position") if parts else pd.DataFrame(columns=["_parent_position"]+columns).set_index("_parent_position")
    meta["returned_positions"] = len(result)
    meta["complete"] = bool(meta["complete"] and len(result) == len(wanted))
    return result, meta


def lookup_key(reader, column: str, wanted_ids: list[int | None], args, deadline: float) -> tuple[dict[int, int], Counter, dict]:
    wanted = set(k for k in wanted_ids if k is not None)
    first, counts = {}, Counter()
    rows = 0
    complete = True
    invalid = 0
    for start, df in reader.batches([column], args.batch_size):
        if time.monotonic() > deadline or rows + len(df) > args.max_key_scan_rows:
            complete = False
            break
        vals = df[column]
        # Vectorized preselection for numeric columns; strict validation follows.
        if pd.api.types.is_numeric_dtype(vals.dtype):
            locs = np.flatnonzero(vals.isin(wanted).to_numpy())
            pairs = [(int(i), strict_int(vals.iloc[int(i)])) for i in locs]
            invalid += int(vals.isna().sum())
        else:
            pairs = [(i, strict_int(v)) for i, v in enumerate(vals.tolist())]
            invalid += sum(v is None for _, v in pairs)
        for i, v in pairs:
            if v is not None and v in wanted:
                counts[v] += 1
                first.setdefault(v, int(start+i))
        rows += len(df)
    complete = complete and rows == reader.n
    return first, counts, dict(column=column, scanned_rows=rows, total_rows=reader.n, complete=complete, requested_ids=len(wanted), found_ids=len(counts), duplicated_requested_ids=sum(v>1 for v in counts.values()), invalid_or_missing_key_values=invalid)


def compare_content(sample: pd.DataFrame, parent: pd.DataFrame, positions: list[int | None], unique_keys: np.ndarray, full_key_scan: bool) -> tuple[pd.DataFrame, list[dict], dict]:
    n = len(sample)
    pos_array = np.array([-1 if p is None else int(p) for p in positions], dtype=np.int64)
    selected = parent.reindex(pos_array).reset_index(drop=True)
    has = np.array([p is not None and int(p) in parent.index for p in positions], dtype=bool)
    numeric_cols = [c for c in FEATURES if c in sample and c in parent]
    label_cols = [c for c in LABELS if c in sample and c in parent]
    raw_ok = has.copy()
    sanitized_ok = has.copy()
    feature_errors = np.zeros(n, dtype=np.int16)
    stats = []
    critical_ok = has.copy()
    for c in numeric_cols:
        a, b = numeric(sample[c]), numeric(selected[c])
        eq = exact_numeric(a, b)
        sa, sb = np.where(np.isfinite(a), a, 0.0), np.where(np.isfinite(b), b, 0.0)
        se = sa == sb
        raw_ok &= eq
        sanitized_ok &= se
        feature_errors += (~eq & has)
        if c in (EPOCH, "frame.len"):
            critical_ok &= eq & np.isfinite(a) & np.isfinite(b)
        finite = has & np.isfinite(a) & np.isfinite(b)
        error = np.abs(a[finite]-b[finite])
        stats.append(dict(field=c, n_compared=int(has.sum()), raw_equal_n=int((eq&has).sum()), after_numeric_nonfinite_to_zero_equal_n=int((se&has).sum()), max_abs_error=float(error.max()) if len(error) else None, comparison="numeric exact; no relative tolerance"))
    labs_ok = has.copy()
    for c in label_cols:
        eq = exact_labels(sample[c].reset_index(drop=True), selected[c])
        labs_ok &= eq
        stats.append(dict(field=c, n_compared=int(has.sum()), raw_equal_n=int((eq&has).sum()), comparison="nonmissing exact string; no class remapping"))
    complete_columns = set(FEATURES).issubset(numeric_cols) and set(LABELS).issubset(label_cols)
    eligible = has & unique_keys & labs_ok & critical_ok
    if not full_key_scan:
        eligible[:] = False
    raw = eligible & raw_ok
    sanitized = eligible & sanitized_ok
    status = np.full(n, "CONTENT_MISMATCH", dtype=object)
    status[~has] = "PARENT_ROW_NOT_READ"
    status[has & ~unique_keys] = "KEY_NOT_UNIQUE"
    if not full_key_scan:
        status[has] = "KEY_SCAN_INCOMPLETE"
    if complete_columns:
        status[sanitized] = "SANITIZED_COMPLETE_CONTENT_MATCH"
        status[raw] = "RAW_COMPLETE_CONTENT_MATCH"
    else:
        status[sanitized] = "PARTIAL_FIELDS_MATCH_ONLY"
    details = pd.DataFrame(dict(sample_local_index=np.arange(n), parent_row_position=pos_array, parent_row_read=has, unique_key=unique_keys, raw_common_numeric_mismatches_n=feature_errors, labels_equal=labs_ok, critical_fields_equal=critical_ok, status=status))
    summary = dict(sample_rows=n, parent_rows_read=int(has.sum()), common_numeric_fields=len(numeric_cols), missing_model_fields="|".join(c for c in FEATURES if c not in numeric_cols), common_label_fields="|".join(label_cols), all_required_columns_present=complete_columns, key_scan_complete=full_key_scan, raw_complete_match_n=int(np.sum(status=="RAW_COMPLETE_CONTENT_MATCH")), sanitized_complete_match_n=int(np.sum(status=="SANITIZED_COMPLETE_CONTENT_MATCH")), partial_fields_match_n=int(np.sum(status=="PARTIAL_FIELDS_MATCH_ONLY")), full_materialized_mapping_confirmed=bool(n and complete_columns and np.all(raw | sanitized)))
    return details, stats, summary


def range_positions(ids: list[int | None], metadata: dict) -> list[int | None] | None:
    indexes = metadata.get("index_columns", [])
    ranges = [x for x in indexes if isinstance(x, dict) and x.get("kind") == "range"]
    if len(ranges) != 1:
        return None
    info = ranges[0]
    start, step = strict_int(info.get("start", 0)), strict_int(info.get("step", 1))
    if start is None or step in (None, 0):
        return None
    return [(k-start)//step if k is not None and (k-start)%step == 0 else None for k in ids]


def predecessor_evidence(dataset: str, sample: pd.DataFrame, parent: pd.DataFrame, positions: list[int | None], detail: pd.DataFrame, method: str, report: Report) -> None:
    verified = detail.status.isin(["RAW_COMPLETE_CONTENT_MATCH", "SANITIZED_COMPLETE_CONTENT_MATCH"])
    ids = np.flatnonzero(verified.to_numpy())
    rows = []
    for i in ids:
        p = positions[int(i)]
        if p is None or p <= 0 or p-1 not in parent.index or p not in parent.index:
            continue
        cur, prev = parent.loc[p], parent.loc[p-1]
        t, tprev = pd.to_numeric(cur.get(EPOCH), errors="coerce"), pd.to_numeric(prev.get(EPOCH), errors="coerce")
        iat = pd.to_numeric(cur.get(IAT), errors="coerce")
        if not np.isfinite([t, tprev, iat]).all():
            continue
        delta = float(t-tprev)
        tol = max(1e-9, 2*float(max(abs(np.spacing(float(t))), abs(np.spacing(float(tprev))))))
        tokens = {c: str(cur[c]) for c in CAPTURE_COLS+SESSION_COLS if c in cur.index and pd.notna(cur[c])}
        boundary = "UNKNOWN_NO_CAPTURE_METADATA"
        if tokens:
            changes = [c for c in tokens if pd.isna(prev[c]) or str(prev[c]) != tokens[c]]
            boundary = "REPORTED_GROUP_TOKEN_CHANGED" if changes else "SAME_REPORTED_GROUP_TOKEN_NOT_SESSION_PROOF"
        rows.append(dict(sample_local_index=int(i), source_row_index=sample.iloc[int(i)]["source_row_index"], parent_row_position=int(p), stored_iat_seconds=float(iat), parent_global_predecessor_delta_seconds=delta, abs_difference_seconds=abs(float(iat)-delta), tolerance_seconds=tol, agrees_with_global_predecessor=bool(abs(float(iat)-delta)<=tol), predecessor_group_status=boundary))
    data = pd.DataFrame(rows)
    if len(data):
        for boundary, g in data.groupby("predecessor_group_status"):
            report.add("16_iat_predecessor_summary.csv", dataset=dataset, method=method, group_status=boundary, n=len(g), matches=int(g.agrees_with_global_predecessor.sum()), stored_negative_n=int((g.stored_iat_seconds<0).sum()), actual_global_delta_negative_n=int((g.parent_global_predecessor_delta_seconds<0).sum()), max_abs_difference_seconds=float(g.abs_difference_seconds.max()), note="Global-table predecessor is not automatically a same-capture/flow predecessor")
        problem = data[data.stored_iat_seconds<0].copy()
        if len(problem):
            problem.insert(0, "dataset", dataset)
            problem.insert(1, "method", method)
            for r in problem.to_dict("records"):
                report.add("17_negative_iat_context.csv", **r)
    else:
        report.add("16_iat_predecessor_summary.csv", dataset=dataset, method=method, n=0, status="NO_VERIFIED_PREDECESSOR_PAIRS")


def inspect_parent(dataset: str, sample: pd.DataFrame, path: Path, report: Report, args, reader_factory=ParquetReader) -> dict:
    result = dict(dataset=dataset, parent_path=str(path), status="NOT_READ", complete_mapping=False)
    before = stat_signature(path)
    reader = reader_factory(path)
    deadline = time.monotonic()+args.parent_seconds
    try:
        report.add("10_parent_metadata.csv", dataset=dataset, path=str(path), rows=reader.n, row_groups=len(reader.group_sizes), schema_json=json.dumps(reader.types), pandas_index_json=json.dumps(reader.pandas.get("index_columns", [])), contains_explicit_capture_columns="|".join(c for c in CAPTURE_COLS if c in reader.columns), contains_explicit_session_columns="|".join(c for c in SESSION_COLS if c in reader.columns), note="Parquet rows and group-token names do not establish complete packet history or independent sessions")
        ids = [strict_int(v) for v in sample["source_row_index"].tolist()]
        if any(x is None for x in ids) or len(set(ids)) != len(ids):
            report.finding("ERROR", "SAMPLE_IDS_INVALID_OR_DUPLICATED", dataset, "Parent content matching stopped; sample IDs must be exact integers and unique within this materialized file")
            result["status"] = "INVALID_SAMPLE_IDS"
            return result
        methods = {"source_id_as_zero_based_position": (ids, np.ones(len(ids), dtype=bool), True)}
        ranged = range_positions(ids, reader.pandas)
        if ranged is not None and ranged != ids:
            methods["source_id_as_pandas_range_index"] = (ranged, np.ones(len(ids), dtype=bool), True)
        declared = [x for x in reader.pandas.get("index_columns", []) if isinstance(x, str) and x in reader.columns]
        keys = list(dict.fromkeys([c for c in KEY_COLS if c in reader.columns]+declared))
        for key in keys[:args.max_key_columns]:
            report.logger.info("PARENT key scan %s column=%s rows=%s", dataset, key, reader.n)
            first, counts, cov = lookup_key(reader, key, ids, args, deadline)
            report.add("11_parent_read_coverage.csv", dataset=dataset, parent_path=str(path), operation="explicit_index_scan", **cov)
            pos = [first.get(k) for k in ids]
            methods["source_id_equals_column:"+key] = (pos, np.array([counts.get(k,0)==1 for k in ids]), bool(cov["complete"]))
        if len(keys)>args.max_key_columns:
            report.finding("REVIEW", "KEY_COLUMNS_LIMIT", dataset, f"Inspected {args.max_key_columns}/{len(keys)} candidate index columns")
        targets = set()
        for positions, _, _ in methods.values():
            for p in positions:
                if p is not None and 0 <= p < reader.n:
                    targets.add(p)
                    if p > 0:
                        targets.add(p-1)
        cols = list(dict.fromkeys([c for c in FEATURES+list(LABELS)+list(CAPTURE_COLS+SESSION_COLS+FRAME_COLS) if c in reader.columns]))
        report.logger.info("PARENT selected-row read %s requested_positions=%s model_columns=%s", dataset, len(targets), len(cols))
        parent, cov = read_positions(reader, targets, cols, args, deadline)
        report.add("11_parent_read_coverage.csv", dataset=dataset, parent_path=str(path), operation="selected_positions_and_predecessors", **cov)
        accepted = []
        raw_details = {}
        for method, (positions, unique, complete) in methods.items():
            detail, stats, summary = compare_content(sample, parent, positions, unique, complete)
            detail.insert(1, "source_row_index", sample["source_row_index"].to_numpy())
            detail.insert(0, "dataset", dataset)
            detail["index_hypothesis"] = method
            detail["parent_path"] = str(path)
            # Report literal parent metadata tokens, not inferred sessions.
            matched_parent = parent.reindex([-1 if p is None else p for p in positions]).reset_index(drop=True)
            for c in CAPTURE_COLS+SESSION_COLS+FRAME_COLS:
                if c in matched_parent:
                    detail["reported_parent__"+c] = matched_parent[c]
            raw_details[method] = detail
            for st in stats:
                report.add("12_parent_field_comparison.csv", dataset=dataset, parent_path=str(path), method=method, **st)
            report.add("13_parent_method_summary.csv", dataset=dataset, parent_path=str(path), method=method, **summary)
            if summary["full_materialized_mapping_confirmed"]:
                accepted.append(method)
        mappings = {tuple(methods[m][0]) for m in accepted}
        changed = before != stat_signature(path)
        if changed:
            report.finding("ERROR", "PARENT_CHANGED_DURING_READ", dataset, str(path))
            result["status"] = "INPUT_CHANGED"
        elif len(mappings) > 1:
            result["status"] = "AMBIGUOUS_FULL_CONTENT_MATCH_MULTIPLE_POSITIONS"
            report.finding("REVIEW", "AMBIGUOUS_PARENT_INDEX", dataset, "Multiple index hypotheses match content at different parent positions; none chosen")
        elif accepted:
            best = next((m for m in accepted if m.startswith("source_id_equals_column:")), accepted[0])
            result.update(status="CONTENT_VALIDATED_PARENT_LINK", complete_mapping=True, chosen_index_hypothesis=best, equivalent_hypotheses="|".join(accepted))
            chosen = raw_details[best]
            report.frame(f"15_parent_row_map_{dataset}.csv.gz", chosen)
            predecessor_evidence(dataset, sample, parent, methods[best][0], chosen, best, report)
            for col in CAPTURE_COLS+SESSION_COLS:
                if col in parent:
                    values = parent.reindex(methods[best][0])[col]
                    counts = values.dropna().astype(str).value_counts()
                    for token, n in counts.head(args.max_group_tokens).items():
                        report.add("18_reported_parent_groups.csv", dataset=dataset, column=col, token=token, sample_rows=int(n), interpretation="Literal metadata value only; not certified independent capture/session")
                    report.add("18_reported_parent_group_coverage.csv", dataset=dataset, column=col, unique_tokens=int(len(counts)), exported_tokens=min(args.max_group_tokens,len(counts)), missing_n=int(values.isna().sum()))
        else:
            result["status"] = "NO_COMPLETE_PARENT_LINK"
            report.finding("REVIEW", "PARENT_LINK_UNRESOLVED", dataset, "Inspect per-field errors, index hypotheses, missing columns and scan coverage; no nearest-row repair applied")
        if not result["complete_mapping"]:
            for method, detail in raw_details.items():
                # Bounded diagnostics preserve failures and correct cases without a false canonical mapping.
                for record in detail.head(20).to_dict("records"):
                    report.add("19_unresolved_mapping_examples.csv", **record)
        report.logger.info("PARENT decision %s %s", dataset, result["status"])
        result.update(parent_capture_metadata_present=any(c in parent for c in CAPTURE_COLS), parent_session_metadata_present=any(c in parent for c in SESSION_COLS), capture_mapping_verified=False, session_independence_verified=False)
        return result
    finally:
        reader.close()


def bounded_files(root: Path, max_depth: int, max_entries: int, report: Report, purpose: str) -> list[Path]:
    files, seen = [], 0
    if not root.is_dir():
        report.add("04_search_coverage.csv", root=str(root), purpose=purpose, status="ROOT_MISSING", entries_seen=0)
        return files
    stack = [(root,0)]
    limited = False
    while stack and not limited:
        folder, depth = stack.pop()
        try:
            with os.scandir(folder) as it:
                for item in it:
                    seen += 1
                    if seen>max_entries:
                        limited = True
                        break
                    if item.is_file(follow_symlinks=False):
                        files.append(Path(item.path))
                    elif item.is_dir(follow_symlinks=False) and depth<max_depth:
                        stack.append((Path(item.path),depth+1))
        except OSError as exc:
            report.add("04_search_coverage.csv", root=str(folder), purpose=purpose, status="DIRECTORY_ERROR", error=str(exc))
    report.add("04_search_coverage.csv", root=str(root), purpose=purpose, status="ENTRY_LIMIT" if limited else "BOUNDED_WALK_COMPLETE", entries_seen=min(seen,max_entries), max_depth=max_depth, max_entries=max_entries, returned_files=len(files), note="Subdirectories beyond max_depth and symlinked directories were not traversed")
    return files


def source_candidates(base: Path, legacy: Path, prior: dict, parents: dict, args, report: Report) -> list[Path]:
    scores = {}
    def put(path, score):
        p=Path(path)
        if p.is_file() and p.suffix.lower() in TEXT_EXTS and not p.name.startswith(("16-temporal_audit", "17-temporal_audit")):
            scores[p] = max(scores.get(p,0),score)
    named = [base/"0-temporal-feature-audit-v2-3datasets"/"01-run-three-dataset-audit.py", base/"0-snn-ae-unified-v8-3-1-fair-benchmark-canonical.py", legacy/"metadata"/"ciciot2023_raw_pcap_provenance.txt", legacy/"ton_iot_raw_pcap_time_audit.txt"]
    for p in named:
        put(p,200)
    df=prior.get("31_source_path_candidates.csv",pd.DataFrame())
    if len(df) and {"path","referenced_by"}.issubset(df.columns):
        for r in df.itertuples(index=False):
            text=str(r.path).lower()
            if "bidir.parquet" in text or any(x in text for x in ("pcap","episode_manifest","capture_manifest")):
                put(r.referenced_by, 150 if "bidir.parquet" in text else 65)
            if any(x in Path(str(r.path)).name.lower() for x in ("episode_manifest","capture_manifest","source_manifest")):
                put(r.path,95)
    # Intentionally narrow roots; do not recursively walk the entire project.
    roots=[(base,0),(base/"dataset-from-pcap",2),(base/"0-temporal-feature-audit-v2-3datasets",1),(base/"0-temporal-feature-audit",1)]
    for root,depth in roots:
        for p in bounded_files(root,depth,args.max_search_entries,report,"extractor_clues"):
            if p.suffix.lower() in (".py",".ipynb",".sh"):
                score=90 if re.search(r"extract|pcap|parquet|bidir|convert|sample|dataset-audit",p.name,re.I) else 20
                put(p,score)
    return [p for p,_ in sorted(scores.items(),key=lambda kv:(-kv[1],str(kv[0])))[:args.max_source_files]]


def inspect_sources(paths: list[Path], args, report: Report) -> list[Path]:
    refs=[]
    # Excerpts are evidence for human review; neither Python nor notebooks are executed.
    pattern=re.compile(r"source_row_index|inter_arrival_time|\.diff\(|groupby\(|reset_index\(|sort_values\(|pd\.concat|read_parquet|to_parquet|bidir|tshark|capinfos|FileCapture|rdpcap|pcap.Reader|packet_index|frame.number|capture_id|session_id|episode_id|pcap_path|capture_file|source_file",re.I)
    assignment=re.compile(r"inter_arrival_time.*(?:=|diff\(|shift\()|(?:diff\(|shift\().*inter_arrival_time",re.I)
    for path in paths:
        try:
            before=stat_signature(path)
            text,trunc=normalized_text(path,args.max_text_bytes)
            lines=text.splitlines()
            hits=[i for i,line in enumerate(lines) if pattern.search(line)]
            specific=[i for i in hits if assignment.search(lines[i]) or "source_row_index" in lines[i]]
            order=list(dict.fromkeys(specific+hits))[:args.max_snippets_per_file]
            report.add("05_source_files.csv",path=str(path),sha256=sha256(path) if path.stat().st_size<=args.hash_limit else "",bytes=path.stat().st_size,text_truncated=trunc,matching_lines=len(hits),exported_blocks=len(order),notebook_lines_are_code_view=path.suffix.lower()==".ipynb")
            for i in order:
                lo,hi=max(0,i-args.context_lines),min(len(lines),i+args.context_lines+1)
                excerpt="\n".join(f"{k+1}: {lines[k]}" for k in range(lo,hi))
                report.add("06_code_evidence.csv",path=str(path),start_line=lo+1,end_line=hi,hit_line=i+1,excerpt=excerpt,evidence_level="STATIC_TEXT_HINT_NOT_EXECUTED")
            for literal in re.findall(r"(?:/home/[^\s\"'`<>\)\]\},;]+)", text):
                candidate=Path(literal.rstrip(":."))
                if any(x in str(candidate) for x in ("$","{","}")):
                    continue
                kind="file" if candidate.is_file() else "directory" if candidate.is_dir() else "unresolved"
                if candidate.suffix.lower() in TEXT_EXTS|{".parquet",".pcap",".pcapng"} or kind=="directory":
                    refs.append(candidate)
                    report.add("07_path_references.csv",referenced_by=str(path),path=str(candidate),kind=kind,evidence_level="PATH_REFERENCE_ONLY",packet_mapping_verified=False)
            if before!=stat_signature(path):
                report.finding("REVIEW","SOURCE_TEXT_CHANGED",str(path),"Static excerpts may not represent one immutable file version")
        except Exception as exc:
            report.finding("REVIEW","SOURCE_READ_ERROR",str(path),f"{type(exc).__name__}: {exc}")
    return list(dict.fromkeys(refs))


def capture_candidates(base: Path, prior: dict, refs: list[Path], args, report: Report) -> list[Path]:
    paths=set()
    for name in ("32_raw_file_candidates.csv","31_source_path_candidates.csv"):
        df=prior.get(name,pd.DataFrame())
        if "path" in df:
            paths.update(Path(x) for x in df.path if str(x).lower().endswith((".pcap",".pcapng")))
    paths.update(p for p in refs if p.suffix.lower() in (".pcap",".pcapng"))
    roots=[base/"dataset-csv"/"edge_iiotset",base/"dataset-raw"/"pcap",(Path(os.environ["TEMPORAL_DATA_ROOT"]).expanduser().resolve() / "ciciot2023pcap")]
    roots.extend(Path(x) for x in args.raw_root)
    for root in list(dict.fromkeys(roots)):
        for p in bounded_files(root,args.raw_depth,args.max_search_entries,report,"targeted_capture_candidates"):
            if p.suffix.lower() in (".pcap",".pcapng"):
                paths.add(p)
    scored=[]
    for p in sorted(paths):
        text=str(p).lower()
        derived=any(x in text for x in ("/replay", "paced-replay", "/replay_", "_replay", "/loopback", "/test_fixture"))
        exists=p.is_file()
        size=p.stat().st_size if exists else None
        report.add("08_capture_candidates.csv",path=str(p),dataset_hint=ds_hint(str(p)),exists=exists,bytes=size,derived_name_hint=derived,selected_for_header_scan=False,session_independence_verified=False)
        if not exists or size==0 or derived:
            if exists and size==0:
                report.finding("REVIEW","EMPTY_CAPTURE_NOT_USED",str(p),"Zero-byte file is not a readable source capture")
            continue
        score=(0 if "ciciot2023pcap" in text else 1 if "dataset-csv/edge_iiotset" in text else 2 if "dataset-raw/pcap" in text else 3)
        scored.append((score,str(p),p))
    selected=[p for _,_,p in sorted(scored)[:args.max_pcaps]]
    chosen=set(map(str,selected))
    for r in report.rows["08_capture_candidates.csv"]:
        r["selected_for_header_scan"]=r["path"] in chosen
    report.add("04_search_coverage.csv",purpose="capture_selection",available_nonempty_nonderived=len(scored),selected=len(selected),max_pcaps=args.max_pcaps,status="LIMITED" if len(scored)>args.max_pcaps else "ALL_CANDIDATES_SELECTED",note="Filename exclusion is a heuristic; selected files are not certified original captures")
    return selected


def run_tshark(path: Path, cache: Path, args, tshark: str) -> tuple[Path, dict]:
    token=hashlib.sha256(str(path).encode()).hexdigest()[:16]
    output=cache/(token+".tsv")
    stderr=cache/(token+".stderr.txt")
    command=[tshark,"-n","-r",str(path),"-c",str(args.pcap_prefix_packets),"-T","fields","-E","header=y","-E","separator=/t","-E","quote=d","-e","frame.number","-e","frame.time_epoch","-e","frame.len"]
    t0=time.monotonic()
    code=None
    status="READ_RETURNED"
    with output.open("wb") as out, stderr.open("wb") as err:
        try:
            r=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=args.pcap_seconds,check=False,shell=False)
            code=r.returncode
            if code:
                status="TSHARK_ERROR"
        except subprocess.TimeoutExpired:
            status="TIME_LIMIT_PARTIAL_OUTPUT"
        except OSError as exc:
            status="START_ERROR"
            err.write(str(exc).encode())
    return output,dict(status=status,returncode=code,seconds=round(time.monotonic()-t0,3),command=json.dumps(command),stderr_excerpt=stderr.read_text(encoding="utf-8",errors="replace")[:1500])


def header_candidates(headers: pd.DataFrame, samples: dict[str,pd.DataFrame], tolerance: float, max_pairs: int) -> tuple[list[dict],Counter,bool]:
    rows=[]
    counts=Counter()
    truncated=False
    htime=numeric(headers[EPOCH])
    hlen=numeric(headers["frame.len"])
    for dataset,s in samples.items():
        times=numeric(s[EPOCH]);lens=numeric(s["frame.len"])
        order=np.argsort(times,kind="stable")
        st=times[order]
        left=np.searchsorted(st,htime-tolerance,side="left")
        right=np.searchsorted(st,htime+tolerance,side="right")
        for j in np.flatnonzero((right>left)&np.isfinite(htime)&np.isfinite(hlen)):
            ids=order[left[j]:right[j]]
            for i in ids:
                if not np.isfinite(times[i]) or not np.isfinite(lens[i]) or lens[i]!=hlen[j] or abs(times[i]-htime[j])>tolerance:
                    continue
                counts[(dataset,int(i))]+=1
                if len(rows)>=max_pairs:
                    truncated=True
                    continue
                rows.append(dict(dataset=dataset,sample_local_index=int(i),source_row_index=s.iloc[int(i)]["source_row_index"],pcap_frame_number=strict_int(headers.iloc[int(j)]["frame.number"]),sample_epoch_seconds=float(times[i]),header_epoch_seconds=float(htime[j]),abs_time_difference_seconds=float(abs(times[i]-htime[j])),frame_len=int(hlen[j]),identity_verified=False,evidence_level="TIMESTAMP_LENGTH_CANDIDATE_ONLY"))
    return rows,counts,truncated


def scan_captures(paths: list[Path], samples: dict, args, report: Report) -> None:
    if args.skip_pcap:
        report.add("20_pcap_scans.csv",status="SKIPPED_BY_OPTION",note="No PCAP membership inference")
        return
    tshark=shutil.which("tshark")
    if not tshark:
        report.finding("REVIEW","TSHARK_NOT_AVAILABLE","pcap_headers","Parent and code audits continue; raw packet header scan skipped")
        report.add("20_pcap_scans.csv",status="SKIPPED_TSHARK_NOT_AVAILABLE")
        return
    try:
        ver=subprocess.run([tshark,"--version"],capture_output=True,text=True,timeout=10,check=False)
        report.add("20_tool_version.csv",tool=tshark,returncode=ver.returncode,version=ver.stdout[:2000])
    except Exception as exc:
        report.add("20_tool_version.csv",tool=tshark,error=str(exc))
    remaining=args.max_header_pairs
    all_counts=Counter()
    exported=0
    for k,path in enumerate(paths,1):
        report.logger.info("PCAP prefix %d/%d %s",k,len(paths),path)
        before=stat_signature(path)
        tsv,meta=run_tshark(path,report.cache,args,tshark)
        count=invalid=0
        min_t=max_t=None
        file_pairs=[]
        trunc=False
        file_count=Counter()
        try:
            for df in pd.read_csv(tsv,sep="\t",dtype=str,chunksize=args.batch_size,keep_default_na=False):
                if not {"frame.number",EPOCH,"frame.len"}.issubset(df.columns):
                    raise ValueError("Unexpected TShark header columns")
                numbers=[strict_int(v) for v in df["frame.number"]]
                times=numeric(df[EPOCH]);lengths=numeric(df["frame.len"])
                ok=np.array([v is not None and v>0 for v in numbers])&np.isfinite(times)&np.isfinite(lengths)&(lengths>=0)&(np.floor(lengths)==lengths)
                count+=len(df);invalid+=int((~ok).sum())
                clean=df.loc[ok].reset_index(drop=True)
                if len(clean):
                    goodtimes=numeric(clean[EPOCH]);mn=float(goodtimes.min());mx=float(goodtimes.max())
                    min_t=mn if min_t is None else min(min_t,mn)
                    max_t=mx if max_t is None else max(max_t,mx)
                    pairs,hits,cut=header_candidates(clean,samples,args.pcap_time_tolerance,remaining)
                    file_count.update(hits);file_pairs.extend(pairs);trunc|=cut
                    remaining-=len(pairs)
            changed=before!=stat_signature(path)
            full=bool(meta["returncode"]==0 and count<args.pcap_prefix_packets and invalid==0 and not changed)
            scope="COMPLETE_HEADER_SCAN_EOF" if full else "BOUNDED_PREFIX_OR_PARTIAL_SCAN"
            report.add("20_pcap_scans.csv",path=str(path),**meta,header_rows=count,invalid_header_rows=invalid,prefix_limit=args.pcap_prefix_packets,scan_scope=scope,min_epoch_in_scanned_records=min_t,max_epoch_in_scanned_records=max_t,input_changed=changed,header_candidate_pairs=sum(file_count.values()),exported_pairs=len(file_pairs),pair_export_limited=trunc,full_payload_identity_verified=False)
            if not changed:
                all_counts.update(file_count)
            for r in file_pairs:
                report.add("21_header_match_candidates.csv",capture_path=str(path),scan_scope=scope,input_changed=changed,**r)
            exported+=len(file_pairs)
        except Exception as exc:
            report.add("20_pcap_scans.csv",path=str(path),**{**meta,"status":"HEADER_PARSE_ERROR"},error=str(exc),header_rows_parsed=count,identity_verified=False)
    for dataset,s in samples.items():
        hits=np.array([all_counts[(dataset,i)] for i in range(len(s))])
        report.add("22_header_candidate_summary.csv",dataset=dataset,sample_rows=len(s),with_candidate_in_scanned_records=int((hits>0).sum()),with_one_candidate_in_scanned_records=int((hits==1).sum()),with_multiple_candidates_in_scanned_records=int((hits>1).sum()),without_candidate_in_scanned_records=int((hits==0).sum()),tolerance_seconds=args.pcap_time_tolerance,note="Counts across successfully parsed unchanged files only; uniqueness within scanned headers is not a packet identity proof; absence is not absence from whole captures")
    report.add("22_header_export_coverage.csv",exported_pairs=exported,max_exported_pairs=args.max_header_pairs,identity_verified=False)


def parse_kv(values: list[str]) -> dict[str,Path]:
    result={}
    for text in values:
        if "=" not in text:
            raise ValueError("Use DATASET=/absolute/path for --parent")
        k,v=text.split("=",1)
        if k not in DATASETS or not Path(v).is_absolute():
            raise ValueError("Unknown dataset or nonabsolute parent path")
        result[k]=Path(v)
    return result


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base",type=Path,default=BASE_DEFAULT)
    p.add_argument("--prior-audit",type=Path,default=None)
    p.add_argument("--legacy-root",type=Path,default=None)
    p.add_argument("--samples-dir",type=Path,default=None)
    p.add_argument("--output-root",type=Path,default=None)
    p.add_argument("--parent",action="append",default=[],metavar="DATASET=/path/to/file.parquet")
    p.add_argument("--raw-root",action="append",default=[],metavar="/path/to/capture/directory")
    p.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    p.add_argument("--batch-size",type=int,default=65536)
    p.add_argument("--max-sample-rows",type=int,default=250000)
    p.add_argument("--max-parent-rows-decoded",type=int,default=100000000)
    p.add_argument("--max-key-scan-rows",type=int,default=100000000)
    p.add_argument("--max-key-columns",type=int,default=3)
    p.add_argument("--parent-seconds",type=float,default=900)
    p.add_argument("--hash-limit",type=int,default=32*1024**2,help="Full-file hash cap in bytes; larger files explicitly not hashed")
    p.add_argument("--max-source-files",type=int,default=80)
    p.add_argument("--max-text-bytes",type=int,default=2*1024**2)
    p.add_argument("--max-snippets-per-file",type=int,default=60)
    p.add_argument("--context-lines",type=int,default=8)
    p.add_argument("--max-search-entries",type=int,default=5000)
    p.add_argument("--max-group-tokens",type=int,default=1000)
    p.add_argument("--raw-depth",type=int,default=4)
    p.add_argument("--max-pcaps",type=int,default=100)
    p.add_argument("--pcap-prefix-packets",type=int,default=50000)
    p.add_argument("--pcap-seconds",type=float,default=60)
    p.add_argument("--pcap-time-tolerance",type=float,default=1e-6)
    p.add_argument("--max-header-pairs",type=int,default=200000)
    p.add_argument("--skip-pcap",action="store_true")
    return p


def validate_args(args) -> None:
    for field in ("batch_size","max_sample_rows","max_parent_rows_decoded","max_key_scan_rows","max_key_columns","parent_seconds","hash_limit","max_source_files","max_text_bytes","max_snippets_per_file","context_lines","max_search_entries","max_group_tokens","max_pcaps","pcap_prefix_packets","pcap_seconds","max_header_pairs"):
        if getattr(args,field)<=0:
            raise ValueError(field+" must be positive")
    if args.raw_depth<0 or args.pcap_time_tolerance<0 or not math.isfinite(args.pcap_time_tolerance):
        raise ValueError("Invalid depth/time tolerance")


def main(argv=None) -> int:
    args=parser().parse_args(argv)
    validate_args(args)
    base=args.base.resolve()
    legacy=(args.legacy_root or base/LEGACY_NAME).resolve()
    sample_dir=(args.samples_dir or base/"0-temporal-feature-audit-v2-3datasets"/"materialized_samples").resolve()
    prior_root=(args.prior_audit or base/"temporal_audit_upgrade_audits"/PRIOR_RUN).resolve()
    out_root=(args.output_root or base/"temporal_audit_provenance_audits").resolve()
    # Refuse placing output inside an input artifact directory.
    for protected in (sample_dir,prior_root,legacy/"runs",legacy/"splits",legacy/"scalers",base/"dataset-from-pcap"):
        if within(out_root,protected):
            raise ValueError(f"Output root must not be inside input directory: {protected}")
    report=Report(out_root)
    package_versions={}
    for pkg in ("numpy","pandas","pyarrow"):
        try:package_versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:package_versions[pkg]="MISSING"
    write_json(report.path/"run_config.json",dict(script_version=VERSION,created_utc=utc(),args=vars(args),resolved_base=base,resolved_legacy=legacy,resolved_samples=sample_dir,resolved_prior=prior_root,python=sys.version,executable=sys.executable,platform=platform.platform(),packages=package_versions,script_sha256=sha256(Path(__file__)),no_training=True,no_pickle_loads=True,no_source_execution=True))
    samples={};decisions=[];prior={};parents={d:base/"dataset-from-pcap"/PARENT_NAMES[d] for d in args.datasets}
    parents.update(parse_kv(args.parent))
    status="COMPLETED"
    try:
        with report.stage("read_previous_audit_hints"):
            prior=read_prior(prior_root,report)
        with report.stage("source_code_and_provenance_clues"):
            srcs=source_candidates(base,legacy,prior,parents,args,report)
            refs=inspect_sources(srcs,args,report)
        for d in args.datasets:
            with report.stage("sample_to_parent:"+d):
                path=sample_dir/(d+"_audit_sample.parquet")
                inv=inventory(path,report,"materialized_sample",args.hash_limit,d)
                if not inv["exists"]:
                    raise FileNotFoundError(path)
                sample_before=stat_signature(path)
                sample_hash=inv.get("sha256") or sha256(path)
                prior_hash_status=verify_previous_sample(path,sample_hash,prior)
                sr=ParquetReader(path)
                try:
                    if sr.n>args.max_sample_rows:
                        raise ValueError(f"Sample has {sr.n} rows; limit={args.max_sample_rows}; no subsampling performed")
                    cols=[c for c in FEATURES+list(LABELS)+["source_row_index"] if c in sr.columns]
                    missing=set(FEATURES+list(LABELS)+["source_row_index"])-set(cols)
                    if missing:
                        raise ValueError("Materialized sample missing required columns: "+str(sorted(missing)))
                    frames=[df for _,df in sr.batches(cols,args.batch_size)]
                    sample=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=cols)
                    if not len(sample):
                        raise ValueError("Empty materialized sample")
                    if sample_before != stat_signature(path):
                        raise ValueError("Materialized sample changed during read")
                    samples[d]=sample
                    report.add("09_sample_metadata.csv",dataset=d,path=str(path),sample_sha256=sample_hash,prior_hash_status=prior_hash_status,rows=len(sample),dtype_epoch=str(sample[EPOCH].dtype),source_ids_unique=int(sample.source_row_index.nunique()),negative_iat_n=int((numeric(sample[IAT])<0).sum()),label_counts_json=json.dumps(sample.label.value_counts().to_dict()),scope="WHOLE_MATERIALIZED_SAMPLE_NOT_BALANCED_CORPUS")
                finally:sr.close()
                parent=parents[d]
                info=inventory(parent,report,"candidate_parent_from_previous_audit",args.hash_limit,d)
                if not info["exists"]:
                    report.finding("REVIEW","PARENT_MISSING",d,str(parent))
                    decisions.append(dict(dataset=d,parent_path=str(parent),status="MISSING",complete_mapping=False))
                    continue
                dec=inspect_parent(d,sample,parent,report,args)
                decisions.append(dec)
                report.add("14_parent_decisions.csv",**dec)
        with report.stage("targeted_capture_inventory"):
            cap_paths=capture_candidates(base,prior,locals().get("refs",[]),args,report)
        with report.stage("bounded_capture_header_checks"):
            scan_captures(locals().get("cap_paths",[]),samples,args,report)
        with report.stage("feasibility_report"):
            by_ds={r["dataset"]:r for r in decisions}
            for d in args.datasets:
                dec=by_ds.get(d,{})
                report.add("30_feasibility.csv",dataset=d,capability="sample_to_parent_content_link",status=dec.get("status","NOT_COMPLETED"),note="Matching parent row content does not establish PCAP or session identity")
                report.add("30_feasibility.csv",dataset=d,capability="raw_packet_identity",status="NOT_CERTIFIED",note="Use code/metadata chain and header candidates for review; timestamp+length alone is insufficient")
                report.add("30_feasibility.csv",dataset=d,capability="complete_causal_history",status="NEEDS_SOURCE_REVIEW",note="Parent completeness, filtering, order and IAT unit must be established before behavioral window extraction")
                report.add("30_feasibility.csv",dataset=d,capability="independent_session_split",status="CANDIDATE_GROUP_METADATA" if dec.get("parent_capture_metadata_present") or dec.get("parent_session_metadata_present") else "GROUP_DEFINITION_NOT_ESTABLISHED",note="Files, row groups, label strata or time intervals are not automatically independent sessions")
    except KeyboardInterrupt:
        status="INTERRUPTED_REPORT_SAVED"
        report.finding("REVIEW","INTERRUPTED","run","Partial reports retained; no input artifacts deleted")
    except Exception as exc:
        status="FAILED_REPORT_SAVED"
        report.finding("ERROR","UNHANDLED", "run",f"{type(exc).__name__}: {exc}")
        report.logger.error(traceback.format_exc())
    finally:
        report.finish(status)
    return 0 if status=="COMPLETED" and not any(s["status"]=="ERROR" for s in report.stages) else 2


if __name__=="__main__":
    raise SystemExit(main())
