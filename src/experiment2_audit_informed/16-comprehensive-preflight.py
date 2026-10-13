#!/usr/bin/env python3
"""Read-only TEMPORAL_AUDIT data, precision, frozen-split and existing-result audit.

No experiment-driver imports, no TensorFlow, no training, no new data splits.
Only a NEW timestamped report directory and its upload ZIP are written.
Pickled scalers are NOT loaded unless --trust-local-scalers is supplied.
Source: user-supplied 04/10/11 scripts, final-v3 handoff and upgrade task plan.
See README_AUDIT.md for scope, denominators, resource bounds and limitations.
Python >=3.10. Dependencies: numpy, pandas, pyarrow, scikit-learn, scipy.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.metadata as ilm
import itertools
import json
import logging
import os
import platform
import re
import stat
import sys
import time
import traceback
import uuid
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import t as student_t

VERSION = '1.0.0'
BASE_DEFAULT = os.environ["TEMPORAL_AUDIT_ROOT"]
NORMAL = {'edge_iiotset': 'Normal', 'ciciot2023': 'Benign', 'ton_iot': 'Normal'}
MODELS = ['SNN_AE', 'Dense_AE', 'LSTM_AE', 'CNN_AE']
LAMBDAS = [0., .001, .0025, .005, .01, .025, .05, .10, .25, .50, 1.]
FEATURES = [
    'frame.len', 'frame.time_epoch', 'ip.version', 'ip.hdr_len', 'ip.dsfield.dscp',
    'ip.len', 'ip.id', 'ip.frag_offset', 'ip.flags.df', 'ip.flags.mf', 'ip.ttl',
    'ip.proto', 'tcp.srcport', 'tcp.dstport', 'tcp.seq', 'tcp.ack', 'tcp.hdr_len',
    'tcp.len', 'tcp.window_size', 'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset',
    'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.urg', 'tcp.flags.ns', 'tcp.flags.cwr',
    'udp.srcport', 'udp.dstport', 'udp.length', 'icmp.type', 'icmp.code', 'data.len',
    'inter_arrival_time']
EPOCH = 'frame.time_epoch'
GROUP_COLS = ['capture_id', 'capture_file', 'pcap_file', 'pcap_path', 'source_file',
              'session_id', 'episode_id', 'flow_id', 'frame.number']
TABLES = {
 '00_findings': 'severity, evidence and next action; not an acceptance probability',
 '01_stage_status': 'execution status; COMPLETED does not mean scientific validation passed',
 '02_file_inventory': 'presence, size, provenance class, optional full SHA256',
 '03_dataset_summary': 'whole materialized files, not necessarily final balanced corpus',
 '04_schema': 'stored column types and roles',
 '05_feature_quality': 'numeric finite/missing/zero/negative and cast diagnostics',
 '06_epoch_precision': 'stored versus cast values by scope/class/subtype; seconds',
 '07_temporal_support': 'descriptive min-max interval, not density support',
 '08_value_examples': 'bounded timestamp/negative-IAT examples; no packet payload',
 '09_order_diagnostics': 'file/source-index order is not established capture order',
 '10_provenance_columns': 'candidate group metadata; no automatic session inference',
 '11_corpus_validation': 'frozen manifests and source hashes',
 '12_split_validation': 'indices, labels, disjointness, corpus membership',
 '13_cv_coverage': 'each corpus row must be held out once per seed',
 '14_signature_overlap': 'feature-hash matches, not proof of duplicated packets or leakage',
 '15_group_overlap': 'observed metadata-group overlap across frozen partitions',
 '16_scaler_validation': 'saved-scaler checks are opt-in; otherwise reconstruction only',
 '17_fold_epoch_scaling': 'float32-first versus centered float64 normalization',
 '18_fold_feature_ranges': 'per-feature training range and held-out extrapolation',
 '19_origin_invariance': 'global origin change, not test-only session intervention',
 '20_timestamp_only': 'diagnostic distance baseline fitted only on training/calibration normal',
 '21_run_availability': 'primary and controlled final paths; historical failures excluded',
 '22_endpoint_qc': 'validation of stored endpoints, IDs, labels, thresholds',
 '23_endpoint_metrics': 'recomputed full/non-epoch endpoint metrics from saved scores',
 '24_endpoint_transitions': 'counts and denominators for both classes',
 '25_shared_failures': 'direct four-model joins only; missing model means no four-model result',
 '26_pairwise_failures': 'Jaccard of attack detected-to-missed sets; descriptive',
 '27_or_coverage': 'OR recall and FPR; binary decision diagnostic, not trained ensemble',
 '28_residual_components': 'signed lambda1-lambda0 squared-error components; descriptive',
 '29_endpoint_statistics': 'legacy NB ratio=1/9; Holm-12 only when all planned pairs complete',
 '30_code_hints': 'static source excerpts; NOT executed and not proof of data provenance',
 '31_source_path_candidates': 'referenced paths and existence; not verified packet mapping',
 '32_raw_file_candidates': 'bounded search inventory; file count is not session count',
 '33_search_coverage': 'explicit search scope and limits',
 '34_feasibility': 'what remains unknown before the upgraded experiment',
}


def plain(x: Any) -> Any:
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return [plain(v) for v in x.tolist()]
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if x is pd.NA: return None
    return x


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def is_regular(path: Path) -> bool:
    try: return stat.S_ISREG(path.stat().st_mode)
    except OSError: return False


def read_json(path: Path, max_bytes: int = 8 * 1024 * 1024) -> dict:
    if not is_regular(path): raise FileNotFoundError(str(path))
    if path.stat().st_size > max_bytes: raise ValueError('JSON exceeds safety limit')
    with path.open(encoding='utf-8') as f: obj = json.load(f)
    if not isinstance(obj, dict): raise ValueError('Expected a JSON object')
    return obj


def checked_npz(path: Path, max_mb: float = 512):
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > max_mb * 1024**2:
            raise ValueError('NPZ uncompressed size exceeds configured limit')
    return np.load(path, allow_pickle=False)


def indices(x: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1 or a.dtype.kind not in 'iu':
        raise ValueError(f'{name}: expected a one-dimensional integer array')
    if not len(a): raise ValueError(f'{name}: empty index array')
    if np.any(a < 0) or np.any(a >= n): raise ValueError(f'{name}: out-of-bounds index')
    a = a.astype(np.int64, copy=False)
    if len(np.unique(a)) != len(a): raise ValueError(f'{name}: duplicate indices')
    return a


def ids64(x: Any) -> np.ndarray:
    s = pd.Series(x)
    if s.isna().any(): raise ValueError('Missing source_row_index')
    if pd.api.types.is_integer_dtype(s.dtype): return s.to_numpy(dtype=np.int64)
    # Do not route CSV integer IDs through a floating-point conversion.
    vals = [int(str(v)) for v in s]
    return np.asarray(vals, dtype=np.int64)


def describe(values: Any) -> dict:
    a = np.asarray(values, dtype=np.float64)
    v = a[np.isfinite(a)]
    r = {'n': len(a), 'finite_n': len(v), 'nonfinite_n': len(a)-len(v)}
    if len(v):
        q = np.percentile(v, [0, 1, 50, 99, 100])
        r.update(dict(zip(['min', 'p01', 'median', 'p99', 'max'], map(float, q))))
        r.update(mean=float(np.mean(v)), zero_n=int(np.sum(v == 0)),
                 negative_n=int(np.sum(v < 0)), unique_n=len(np.unique(v)))
    return r


def cast_diagnostics(values: Any) -> dict:
    a = np.asarray(values, dtype=np.float64)
    x = a[np.isfinite(a)]
    r = describe(a)
    if not len(x): return r
    with np.errstate(over='ignore', invalid='ignore'): y = x.astype(np.float32)
    ok = np.isfinite(y)
    r['float32_overflow_n'] = int(np.sum(~ok))
    if not ok.all(): return r
    back = y.astype(np.float64)
    e = np.abs(x-back)
    r.update(unique_stored=len(np.unique(x)), unique_float32=len(np.unique(y)),
             distinct_values_merged=len(np.unique(x))-len(np.unique(y)),
             changed_n=int(np.count_nonzero(e)), changed_pct=float(100*np.mean(e != 0)),
             abs_error_p50=float(np.median(e)), abs_error_p99=float(np.percentile(e, 99)),
             abs_error_max=float(e.max()),
             float32_spacing_at_median=float(abs(np.spacing(np.float32(np.median(x))))))
    return r


def prefix(d: dict, p: str) -> dict: return {p+k: v for k, v in d.items()}


def interval_score(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(lo+hi) or hi <= lo: raise ValueError('No nonzero finite training epoch range')
    return np.maximum.reduce([np.zeros(len(t)), lo-t, t-hi]) / (hi-lo)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    if len(y) != len(score) or not np.isfinite(score).all() or not np.isfinite(threshold):
        raise ValueError('Nonfinite score/threshold or length mismatch')
    if not np.isin(y, [0, 1]).all(): raise ValueError('Nonbinary y')
    p = score > threshold
    a, b = y == 1, y == 0
    tp, fn = int(np.sum(p & a)), int(np.sum(~p & a))
    fp, tn = int(np.sum(p & b)), int(np.sum(~p & b))
    div = lambda n, d: float(n/d) if d else np.nan
    return dict(n=len(y), n_attack=int(a.sum()), n_normal=int(b.sum()), tp=tp, fn=fn, fp=fp, tn=tn,
                recall=div(tp, tp+fn), fnr=div(fn, tp+fn), fpr=div(fp, fp+tn),
                precision=div(tp, tp+fp), f1=div(2*tp, 2*tp+fp+fn),
                accuracy=div(tp+tn, len(y)),
                auc=float(roc_auc_score(y, score)) if a.any() and b.any() else np.nan,
                threshold=float(threshold))


def centered_epoch(t: np.ndarray, train: np.ndarray) -> np.ndarray:
    lo, hi = float(train.min()), float(train.max())
    if hi <= lo: raise ValueError('Zero training epoch range; do not silently replace denominator')
    return ((t.astype(np.float64)-lo)/(hi-lo)).astype(np.float32)


class Audit:
    def __init__(self, args):
        self.args = args
        self.base = Path(args.base).expanduser().resolve()
        self.source = Path(args.samples_dir).expanduser().resolve() if args.samples_dir else self.base/'0-temporal-feature-audit-v2-3datasets'/'materialized_samples'
        self.legacy = Path(args.legacy_root).expanduser().resolve() if args.legacy_root else self.base/'0-temporal-feature-audit-v4-allminmax-3seed10fold'
        parent = Path(args.output_root).expanduser().resolve() if args.output_root else self.base/'temporal_audit_upgrade_audits'
        forbidden = [self.source, self.legacy, self.base/'0-temporal-feature-audit-v2-3datasets', self.base/'0-temporal-feature-audit']
        if any(parent == p or p in parent.parents for p in forbidden):
            raise ValueError('Output must be outside historical input/experiment trees')
        self.out = parent/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:6])
        self.out.mkdir(parents=True, exist_ok=False)
        self.rows = {k: [] for k in TABLES}
        self.cache = {}
        self.inventory_seen = set()
        self.feature_names = list(FEATURES)
        self.features_verified_by_manifest = False
        self.feature_source = 'fallback_uploaded_schema_not_manifest_verified'
        self.epoch_idx = self.feature_names.index(EPOCH)
        self.data = {}
        self.splitdata = {}
        self.corpora = {}
        self.started = time.monotonic()
        self.cancelled = False
        self.log = logging.getLogger('temporal_audit-audit-'+self.out.name)
        self.log.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(self.out/'audit.log', encoding='utf-8')]:
            h.setFormatter(fmt); self.log.addHandler(h)

    def add(self, table: str, **row): self.rows[table].append(plain(row))

    def issue(self, severity: str, area: str, dataset: str, evidence: str, action: str):
        self.add('00_findings', severity=severity, area=area, dataset=dataset, evidence=evidence, next_action=action)
        if severity == 'ERROR': self.log.error('%s %s: %s', area, dataset, evidence)

    def stage(self, name, fn, *args):
        start = time.monotonic()
        self.log.info('START %s', name)
        try:
            fn(*args)
            status, error = 'COMPLETED', ''
        except Exception as e:
            status, error = 'ERROR', f'{type(e).__name__}: {e}'
            self.issue('ERROR', name, '', error, 'Review report/log; no automatic source repair')
            self.log.error(traceback.format_exc())
        self.add('01_stage_status', stage=name, status=status, seconds=time.monotonic()-start, error=error)
        self.flush()
        self.log.info('END %s [%s]', name, status)

    def file(self, path: Path, kind: str, dataset='', force_hash=False):
        key = str(path)
        if key in self.cache: return self.cache[key]
        r = dict(path=key, kind=kind, dataset=dataset, exists=False, bytes=None, sha256='', hash_status='NOT_PRESENT')
        try:
            if is_regular(path):
                s = path.stat(); r.update(exists=True, bytes=s.st_size, modified_ns=s.st_mtime_ns)
                if force_hash or s.st_size <= self.args.hash_max_mb*1024**2:
                    r.update(sha256=digest(path), hash_status='FULL_SHA256')
                else: r['hash_status'] = 'SKIPPED_SIZE_LIMIT'
        except Exception as e: r.update(error=f'{type(e).__name__}: {e}', hash_status='ERROR')
        self.cache[key] = r
        self.add('02_file_inventory', **r)
        return r

    def flush(self):
        for name, records in self.rows.items():
            cols = list(dict.fromkeys(k for r in records for k in r)) or ['status', 'note']
            dest = self.out/(name+'.csv'); tmp = dest.with_suffix('.csv.tmp')
            with tmp.open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction='raise')
                w.writeheader()
                for r in records:
                    w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})
            os.replace(tmp, dest)

    def config(self):
        p = self.legacy/'metadata'/'CONFIRMATORY_EXPERIMENT_MANIFEST.json'
        self.file(p, 'primary_manifest')
        self.feature_source = 'exact_feature_list_from_uploaded_schema_and_scripts'
        if is_regular(p):
            cfg = read_json(p)
            fs = cfg.get('feature_conditions', {}).get('34F')
            if isinstance(fs, list) and len(fs) == 34 and set(fs) == set(FEATURES) and len(set(fs)) == 34:
                self.feature_names = fs
                self.epoch_idx = fs.index(EPOCH)
                self.feature_source = str(p)
            else:
                self.issue('ERROR', 'feature_contract', '', 'Manifest 34F order/set cannot be verified', 'Confirm feature contract before reusing stored scalers')
        self.features_verified_by_manifest = self.feature_source == str(p)
        env = {k: (ilm.version(k) if has_distribution(k) else 'NOT_INSTALLED') for k in ['numpy','pandas','pyarrow','scikit-learn','scipy','joblib','tensorflow']}
        cfg = dict(script_version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                   executable=sys.executable, platform=platform.platform(), packages=env, args=vars(self.args),
                   feature_order=self.feature_names, feature_order_source=self.feature_source,
                   no_training=True, no_input_mutation=True, source_hash=self.file(Path(__file__), 'audit_script', force_hash=True)['sha256'])
        (self.out/'run_config.json').write_text(json.dumps(plain(cfg), indent=2, ensure_ascii=False), encoding='utf-8')
        if not self.args.trust_local_scalers:
            self.issue('INFO','saved_scalers','','Pickle deserialization disabled', 'Current-sklearn reconstruction will be distinguished from actual saved-scaler checks')

    def read_dataset(self, dataset):
        try: import pyarrow.parquet as pq
        except ImportError as e: raise RuntimeError('pyarrow is required to read your materialized Parquet files') from e
        p = self.source/(dataset+'_audit_sample.parquet')
        info = self.file(p, 'materialized_sample', dataset, force_hash=True)
        if not info['exists']: raise FileNotFoundError(str(p))
        with p.open('rb') as f:
            pf = pq.ParquetFile(f)
            schema = pf.schema_arrow
            n = pf.metadata.num_rows
            for field in schema:
                self.add('04_schema', dataset=dataset, column=field.name, dtype=str(field.type),
                         role='model_feature' if field.name in self.feature_names else 'label_or_metadata')
            if n > self.args.max_materialized_rows: raise ValueError(f'{n} rows exceeds --max-materialized-rows; no partial audit of frozen indices')
            required = self.feature_names+['label','label_type','source_row_index']
            missing = sorted(set(required)-set(schema.names))
            if missing: raise ValueError(f'Missing required columns: {missing}')
            cols = required+[c for c in GROUP_COLS if c in schema.names and c not in required]
            df = pf.read(columns=cols).to_pandas()
        self.analyze_frame(dataset, df, info)

    def analyze_frame(self, dataset, df, info):
        raw = df[self.feature_names].apply(pd.to_numeric, errors='coerce')
        x = raw.to_numpy(dtype=np.float64)
        src = ids64(df['source_row_index'])
        labels = df['label'].astype('string')
        if labels.isna().any() or not labels.isin([NORMAL[dataset], 'Attack']).all():
            raise ValueError('Unexpected or missing labels; not silently treating unknown values as attack')
        y = (labels != NORMAL[dataset]).to_numpy(dtype=np.int8)
        groups = np.where(y == 0, 'NORMAL', df['label_type'].fillna('MISSING_SUBTYPE').astype(str))
        sanitized = np.where(np.isfinite(x), x, 0.)  # Diagnostic replay only; source never changed.
        with np.errstate(over='ignore', invalid='ignore'): x32 = sanitized.astype(np.float32)
        ids_unique = len(np.unique(src)) == len(src)
        self.data[dataset] = dict(df=df, x=x, x32=x32, y=y, src=src, groups=groups, source_info=info)
        self.add('03_dataset_summary', dataset=dataset, scope='whole_materialized_sample', rows=len(df),
                 normal_n=int(np.sum(y==0)), attack_n=int(np.sum(y==1)),
                 source_id_unique=ids_unique, source_hash=info.get('sha256',''),
                 feature_nonfinite_cells=int(np.sum(~np.isfinite(x))),
                 stored_epoch_invalid_n=int(np.sum(~np.isfinite(x[:, self.epoch_idx]))),
                 normalization_fitted=False)
        if not ids_unique: self.issue('ERROR','source_id',dataset,'Nonunique IDs in materialized file','Resolve composite identity before joins')
        for j, name in enumerate(self.feature_names):
            for label, mask in [('ALL',np.ones(len(df),bool)),('NORMAL',y==0),('ATTACK',y==1)]:
                self.add('05_feature_quality', dataset=dataset, feature=name, class_name=label,
                         pandas_dtype=str(df[name].dtype), **cast_diagnostics(x[mask,j]))
        self.describe_scopes(dataset, np.arange(len(df)), 'whole_materialized_sample')
        t = x[:, self.epoch_idx]; iat = x[:, self.feature_names.index('inter_arrival_time')]
        for ordername, order in [('file_rows_not_capture_order', np.arange(len(df))), ('source_index_order_not_verified_capture_order', np.argsort(src, kind='stable'))]:
            dd = np.diff(t[order]); valid = np.isfinite(dd)
            self.add('09_order_diagnostics', dataset=dataset, order=ordername, pairs=len(dd),
                     nonfinite_pairs=int(np.sum(~valid)), backwards_n=int(np.sum(dd[valid]<0)),
                     equal_n=int(np.sum(dd[valid]==0)), source_index_step_one_n=int(np.sum(np.diff(src[order])==1)),
                     note='Do not recompute IAT or traffic rates from the sampled row sequence')
        for label, mask in [('ALL',np.ones(len(df),bool)),('NORMAL',y==0),('ATTACK',y==1)]:
            self.add('09_order_diagnostics', dataset=dataset, order='stored_inter_arrival', class_name=label, **describe(iat[mask]))
        for i in np.flatnonzero(np.isfinite(iat) & (iat<0))[:self.args.max_examples]:
            self.add('08_value_examples', dataset=dataset, kind='negative_stored_iat', source_row_index=int(src[i]),
                     sample_local_index=int(i), label=str(labels.iloc[i]), subtype=str(groups[i]),
                     stored_epoch=format(t[i],'.17g'), stored_iat=format(iat[i],'.17g'))
        if np.any(iat<0): self.issue('REVIEW','inter_arrival',dataset,f'{int(np.sum(iat<0))} negative stored IAT values', 'Trace extractor and capture ordering; no clipping/sorting/repair performed')
        with np.errstate(over='ignore', invalid='ignore'): back = t.astype(np.float32).astype(np.float64)
        for i in np.flatnonzero(np.isfinite(t) & np.isfinite(back) & (t!=back))[:self.args.max_examples]:
            self.add('08_value_examples', dataset=dataset, kind='epoch_cast_changed', source_row_index=int(src[i]), sample_local_index=int(i),
                     stored_epoch=format(t[i],'.17g'), float32_epoch=format(back[i],'.17g'), abs_error_seconds=float(abs(t[i]-back[i])))
        if np.any(np.isfinite(t) & (t!=back)):
            self.issue('REVIEW','epoch_precision',dataset,'Stored timestamps change on float32 cast','Inspect frozen-fold normalized differences before attributing model changes to quantization')
        for c in GROUP_COLS:
            present = c in df
            self.add('10_provenance_columns', dataset=dataset, column=c, present=present,
                     unique_values=int(df[c].nunique()) if present else None,
                     missing_n=int(df[c].isna().sum()) if present else None,
                     interpretation='Candidate only; independent sessions and source mapping not established')
        self.add('34_feasibility',dataset=dataset, capability='causal_behavior_features', status='NOT_ESTABLISHED',
                 reason='Materialized sampled rows alone do not establish complete chronological packet history')
        self.add('34_feasibility',dataset=dataset, capability='session_disjoint_validation', status='NEEDS_PROVENANCE_REVIEW',
                 reason='Metadata candidates are not a certified session/episode definition')

    def describe_scopes(self, dataset, idx, scope):
        d=self.data[dataset]; t=d['x'][idx,self.epoch_idx]; y=d['y'][idx]; groups=d['groups'][idx]
        masks=[('ALL',np.ones(len(idx),bool)),('NORMAL',y==0),('ATTACK',y==1)]
        masks += [('subtype:'+str(g),(y==1)&(groups==g)) for g in sorted(set(groups[y==1]))]
        for label, mask in masks:
            self.add('06_epoch_precision',dataset=dataset, scope=scope, group=label, units='seconds', **cast_diagnostics(t[mask]))
        normal=t[(y==0)&np.isfinite(t)]
        if len(normal) and normal.max()>normal.min():
            lo,hi=float(normal.min()),float(normal.max())
            for label,mask in masks:
                z=t[mask & np.isfinite(t)]
                self.add('07_temporal_support',dataset=dataset, scope=scope, group=label,
                         interval_reference='all_normal_in_this_scope_descriptive_not_holdout_baseline',
                         normal_min=lo,normal_max=hi,normal_width_seconds=hi-lo,n=len(z),
                         below_n=int(np.sum(z<lo)),above_n=int(np.sum(z>hi)),
                         outside_ratio=float(np.mean((z<lo)|(z>hi))) if len(z) else None)

    def corpus(self,dataset):
        d=self.data[dataset]; n=len(d['y'])
        root=self.legacy/'metadata'/'confirmatory_corpus'
        p=root/(dataset+'_corpus_indices.npz'); mf=root/(dataset+'_corpus_manifest.csv.gz'); sm=root/(dataset+'_summary.json')
        for q in [p,mf,sm]: self.file(q,'frozen_corpus',dataset)
        if not is_regular(p):
            self.issue('ERROR','corpus',dataset,'Frozen corpus NPZ missing','Supply historical artifact; it is not regenerated')
            return
        with checked_npz(p,self.args.max_npz_mb) as z:
            ix=indices(z['sample_local_indices'],n,'corpus')
            if not np.array_equal(z['y'],d['y'][ix]): raise ValueError('Corpus labels mismatch source')
        status='VALIDATED_INDICES_AND_LABELS'; manifest_ok=None; hash_ok=None
        if is_regular(mf):
            f=pd.read_csv(mf,dtype={'source_row_index':'Int64'})
            manifest_ok=(np.array_equal(f['sample_local_index'].to_numpy(),ix) and np.array_equal(ids64(f['source_row_index']),d['src'][ix]) and np.array_equal(f['y'].to_numpy(),d['y'][ix]))
            if not manifest_ok: raise ValueError('Corpus manifest order/ID/label mismatch')
        if is_regular(sm):
            js=read_json(sm); old=js.get('sample_file_sha256')
            if old:
                hash_ok=old==d['source_info']['sha256']
                if not hash_ok: raise ValueError('Materialized-file hash differs from frozen corpus summary')
        self.corpora[dataset]=ix
        if dataset=='ton_iot' and is_regular(mf):
            d['groups'][ix]=f['stratum'].astype(str).to_numpy()
        self.add('11_corpus_validation',dataset=dataset,status=status,rows=len(ix),normal_n=int(np.sum(d['y'][ix]==0)),
                 attack_n=int(np.sum(d['y'][ix]==1)),manifest_verified=manifest_ok,source_hash_verified=hash_ok)
        self.describe_scopes(dataset,ix,'frozen_confirmatory_corpus')

    def frozen_splits(self,dataset):
        d=self.data[dataset]; n=len(d['y']); corpus=self.corpora.get(dataset)
        hashes={}
        for kind,cols in [('34F',self.feature_names),('33F',[f for f in self.feature_names if f!=EPOCH])]:
            hashes[kind]=pd.util.hash_pandas_object(d['df'][cols],index=False).to_numpy()
        for seed in self.args.seeds:
            coverage=np.zeros(n,dtype=np.int32); valid_folds=0
            for fold in self.args.folds:
                key=dict(dataset=dataset,seed=seed,fold=fold)
                p=self.legacy/'splits'/dataset/f'seed_{seed:02d}'/f'fold_{fold:02d}.npz'
                fi=self.file(p,'frozen_split',dataset)
                try:
                    with checked_npz(p,self.args.max_npz_mb) as z:
                        tr=indices(z['train_normal_idx'],n,'train'); va=indices(z['val_normal_idx'],n,'val'); te=indices(z['test_idx'],n,'test')
                    if any(len(np.intersect1d(a,b)) for a,b in [(tr,va),(tr,te),(va,te)]): raise ValueError('Partition index overlap')
                    if np.any(d['y'][tr]!=0) or np.any(d['y'][va]!=0): raise ValueError('Non-normal in training/validation')
                    if len(np.unique(d['src']))!=n: raise ValueError('Ambiguous source IDs')
                    corpus_ok=None
                    if corpus is not None:
                        inside=np.isin(np.concatenate([tr,va,te]),corpus).all()
                        expected=np.setdiff1d(corpus[d['y'][corpus]==0],te)
                        corpus_ok=bool(inside and np.array_equal(np.sort(np.concatenate([tr,va])),expected))
                        if not corpus_ok: raise ValueError('Partitions do not match frozen corpus normal pool')
                    self.splitdata[(dataset,seed,fold)]=(tr,va,te,fi['sha256'])
                    coverage[te]+=1; valid_folds+=1
                    self.add('12_split_validation',**key,status='VALIDATED',train_n=len(tr),val_n=len(va),test_n=len(te),
                             test_normal_n=int(np.sum(d['y'][te]==0)),test_attack_n=int(np.sum(d['y'][te]==1)),
                             corpus_membership_verified=corpus_ok,split_sha256=fi['sha256'])
                    for name,h in hashes.items():
                        for an,a,bn,b in [('train',tr,'test',te),('val',va,'test',te),('train',tr,'val',va)]:
                            self.add('14_signature_overlap',**key,feature_set=name,partition_a=an,partition_b=bn,
                                     shared_hashes_n=len(np.intersect1d(h[a],h[b])),b_rows_matching_a_n=int(np.isin(h[b],h[a]).sum()),
                                     note='Potential identical representations only, not evidence of duplicated original packets')
                    for col in GROUP_COLS:
                        if col not in d['df'] or col=='frame.number': continue
                        v=d['df'][col]
                        for an,a,bn,b in [('train',tr,'test',te),('val',va,'test',te)]:
                            ga=set(v.iloc[a].dropna().astype(str)); gb=set(v.iloc[b].dropna().astype(str))
                            self.add('15_group_overlap',**key,column=col,partition_a=an,partition_b=bn,
                                     groups_a=len(ga),groups_b=len(gb),shared_groups=len(ga&gb),
                                     group_semantics='unverified_candidate')
                    try: self.scaling(dataset,seed,fold,tr,va,te)
                    except Exception as e:
                        self.issue('ERROR','fold_scaling',dataset,f'seed={seed} fold={fold}: {type(e).__name__}: {e}','Inspect fold; do not silently exclude from scientific analysis')
                except Exception as e:
                    self.add('12_split_validation',**key,status='ERROR',error=f'{type(e).__name__}: {e}')
                    self.issue('ERROR','frozen_split',dataset,f'seed={seed} fold={fold}: {e}','Locate/correct artifact provenance; no new split generated')
            ix=corpus if corpus is not None else np.flatnonzero(coverage)
            self.add('13_cv_coverage',dataset=dataset,seed=seed,valid_folds=valid_folds,requested_folds=len(self.args.folds),
                     denominator='frozen_corpus' if corpus is not None else 'union_of_available_test_rows',
                     rows=len(ix),never_tested_n=int(np.sum(coverage[ix]==0)),tested_once_n=int(np.sum(coverage[ix]==1)),
                     tested_multiple_n=int(np.sum(coverage[ix]>1)),
                     exactly_once_verified=bool(corpus is not None and valid_folds==len(self.args.folds) and np.all(coverage[ix]==1)))

    def scaling(self,dataset,seed,fold,tr,va,te):
        d=self.data[dataset]; key=dict(dataset=dataset,seed=seed,fold=fold)
        x32=d['x32']; t=d['x'][:,self.epoch_idx]
        if not np.isfinite(x32[np.concatenate([tr,va,te])]).all(): raise ValueError('Float32 overflow after legacy sanitization')
        if not np.isfinite(t[np.concatenate([tr,va,te])]).all(): raise ValueError('Invalid source timestamp; not silently repaired for precision comparison')
        model=MinMaxScaler().fit(x32[tr]); scaler=model; source='reconstructed_current_sklearn'
        path=self.legacy/'scalers'/dataset/'with_epoch_34F'/f'seed_{seed:02d}'/f'fold_{fold:02d}.pkl'
        self.file(path,'saved_scaler',dataset)
        self.file(self.legacy/'scalers'/dataset/'without_epoch_33F'/f'seed_{seed:02d}'/f'fold_{fold:02d}.pkl','saved_scaler_33F_presence_only',dataset)
        r=dict(**key,condition='with_epoch_34F',path=str(path),status='NOT_LOADED_DEFAULT',source='reconstructed_current_sklearn',feature_order_source=self.feature_source)
        if self.args.trust_local_scalers and is_regular(path):
            if not self.features_verified_by_manifest:
                r['status']='NOT_LOADED_UNVERIFIED_FEATURE_ORDER'
            else:
                try:
                    import joblib
                    with warnings.catch_warnings(record=True) as ww:
                        warnings.simplefilter('always'); saved=joblib.load(path)
                    if not isinstance(saved,MinMaxScaler) or saved.n_features_in_!=len(self.feature_names): raise ValueError('Unexpected scaler type/feature count')
                    if tuple(saved.feature_range)!=(0,1) or saved.clip: raise ValueError('Unexpected range or clipping')
                    exact=bool(np.array_equal(saved.data_min_,model.data_min_) and np.array_equal(saved.data_max_,model.data_max_))
                    r.update(status='LOADED_MATCH' if exact else 'LOADED_TRAIN_RANGE_MISMATCH',
                             data_min_max_exact=exact,scale_max_abs_diff=float(np.max(np.abs(saved.scale_-model.scale_))),
                             warnings=' | '.join(str(w.message) for w in ww))
                    if not exact: raise ValueError('Saved scaler not fitted on these frozen training values')
                    scaler=saved; source='verified_saved_scaler'; r['source']=source
                except Exception as e:
                    r.update(status='ERROR',error=f'{type(e).__name__}: {e}')
                    self.issue('ERROR','saved_scaler',dataset,f'seed={seed} fold={fold}: {e}','No claim of saved-scaler replay; current fitted scaler used only as a diagnostic')
        self.add('16_scaler_validation',**r)
        p33=self.legacy/'scalers'/dataset/'without_epoch_33F'/f'seed_{seed:02d}'/f'fold_{fold:02d}.pkl'
        r33=dict(**key,condition='without_epoch_33F',path=str(p33),status='NOT_LOADED_DEFAULT')
        if not is_regular(p33): r33['status']='MISSING'
        elif self.args.trust_local_scalers and self.features_verified_by_manifest:
            try:
                import joblib
                with warnings.catch_warnings(record=True) as ww:
                    warnings.simplefilter('always'); saved33=joblib.load(p33)
                keep=[j for j in range(len(self.feature_names)) if j!=self.epoch_idx]
                if not isinstance(saved33,MinMaxScaler) or saved33.n_features_in_!=33:
                    raise ValueError('Unexpected 33F scaler type/count')
                original_manifest=read_json(self.legacy/'metadata'/'CONFIRMATORY_EXPERIMENT_MANIFEST.json')
                declared33=original_manifest.get('feature_conditions',{}).get('33F')
                if declared33 != [self.feature_names[j] for j in keep]:
                    raise ValueError('Cannot verify 33F feature order')
                if tuple(saved33.feature_range)!=(0,1) or saved33.clip:
                    raise ValueError('Unexpected 33F scaling range/clipping')
                expected=MinMaxScaler().fit(x32[tr][:,keep])
                match=bool(np.array_equal(saved33.data_min_,expected.data_min_) and np.array_equal(saved33.data_max_,expected.data_max_))
                a33=saved33.transform(x32[te][:,keep]).astype(np.float32)
                a34=scaler.transform(x32[te]).astype(np.float32)[:,keep]
                r33.update(status='LOADED_MATCH' if match else 'TRAIN_RANGE_MISMATCH',data_min_max_exact=match,
                           non_epoch_34F_33F_transform_max_abs_diff=float(np.max(np.abs(a33.astype(np.float64)-a34.astype(np.float64)))),
                           warnings=' | '.join(str(w.message) for w in ww))
                if not match: raise ValueError('33F scaler does not match frozen training data')
            except Exception as e:
                r33.update(status='ERROR',error=f'{type(e).__name__}: {e}')
                self.issue('ERROR','saved_scaler_33F',dataset,f'seed={seed} fold={fold}: {e}','Verify non-epoch preprocessing pairing')
        self.add('16_scaler_validation',**r33)
        sel=np.concatenate([tr,va,te])
        tr_t=t[tr]; lo,hi=float(tr_t.min()),float(tr_t.max())
        if hi<=lo:
            self.add('17_fold_epoch_scaling',**key,status='ZERO_SOURCE_TRAIN_RANGE',train_width_seconds=hi-lo)
            return
        quant_train=x32[tr,self.epoch_idx].astype(np.float64); qlo,qhi=quant_train.min(),quant_train.max()
        if qhi<=qlo: self.issue('REVIEW','quantized_train_range',dataset,f'seed={seed} fold={fold}: float32 range collapsed','Evaluate precision-safe path before experiments')
        ranges=[('train_normal',tr),('validation_normal',va),('test_normal',te[d['y'][te]==0]),('test_attack',te[d['y'][te]==1])]
        for part,ix in ranges:
            if not len(ix): continue
            legacy_scaled=scaler.transform(x32[ix]).astype(np.float32)
            a=legacy_scaled[:,self.epoch_idx].astype(np.float64); b=centered_epoch(t[ix],tr_t).astype(np.float64)
            err=np.abs(a-b)
            quant_center=((x32[ix,self.epoch_idx].astype(np.float64)-qlo)/(qhi-qlo)).astype(np.float32).astype(np.float64) if qhi>qlo else np.full(len(ix),np.nan)
            self.add('17_fold_epoch_scaling',**key,partition=part,status='COMPUTED',legacy_scaler_source=source,
                     train_min64=lo,train_max64=hi,train_width64_seconds=hi-lo,
                     train_width32_seconds=float(qhi-qlo),train_unique64=len(np.unique(tr_t)),train_unique32=len(np.unique(quant_train)),
                     **prefix(describe(a),'legacy_'),**prefix(describe(b),'precision_safe_'),
                     scaled_abs_error_p50=float(np.median(err)),scaled_abs_error_p99=float(np.percentile(err,99)),
                     scaled_abs_error_max=float(err.max()),
                     legacy_vs_quantized_centered_max=float(np.nanmax(np.abs(a-quant_center))) if np.isfinite(quant_center).any() else None,
                     outside_legacy_ratio=float(np.mean((a<0)|(a>1))),outside_safe_ratio=float(np.mean((b<0)|(b>1))),
                     support_membership_change_n=int(np.sum(((a<0)|(a>1))!=((b<0)|(b>1)))))
            if part.startswith('test'):
                for j,feat in enumerate(self.feature_names):
                    v=legacy_scaled[:,j]
                    self.add('18_fold_feature_ranges',**key,partition=part,feature=feat,train_min=float(model.data_min_[j]),
                             train_max=float(model.data_max_[j]),train_range=float(model.data_range_[j]),
                             scaled_min=float(v.min()),scaled_median=float(np.median(v)),scaled_max=float(v.max()),
                             below_zero_ratio=float(np.mean(v<0)),above_one_ratio=float(np.mean(v>1)),source=source)
        # Training and calibration reference are fixed; this is not the descriptive all-normal audit.
        for path_name,all_t in [('stored_float64',t),('quantized_float32_promoted',d['x32'][:,self.epoch_idx].astype(np.float64))]:
            ll,hh=float(all_t[tr].min()),float(all_t[tr].max())
            try:
                sv=interval_score(all_t[va],ll,hh); st=interval_score(all_t[te],ll,hh)
                th=float(np.percentile(sv,99))
                self.add('20_timestamp_only',**key,representation=path_name,calibration='legacy_validation_normal_not_independent_of_legacy_early_stopping',
                         **metrics(d['y'][te],st,th))
            except ValueError as e: self.add('20_timestamp_only',**key,representation=path_name,status='NOT_COMPUTED',reason=str(e))
        # Global shifts refit timestamp-only scalers on shifted train. No neural model is run.
        st=t[sel]; base32=MinMaxScaler().fit(tr_t.astype(np.float32).reshape(-1,1))
        base_scaled=base32.transform(st.astype(np.float32).reshape(-1,1)).ravel()
        safe_base=centered_epoch(st,tr_t).astype(np.float64)
        for shift in [1.,64.,86400.,-lo]:
            ts=tr_t+shift; es=st+shift
            mm=MinMaxScaler().fit(ts.astype(np.float32).reshape(-1,1))
            z=mm.transform(es.astype(np.float32).reshape(-1,1)).ravel()
            zs=centered_epoch(es,ts).astype(np.float64)
            self.add('19_origin_invariance',**key,global_shift_seconds=shift,n=len(st),
                     legacy_max_abs_change=float(np.max(np.abs(z.astype(np.float64)-base_scaled.astype(np.float64)))),
                     safe_max_abs_change=float(np.max(np.abs(zs-safe_base))),
                     legacy_changed_n=int(np.sum(z!=base_scaled)),safe_changed_n=int(np.sum(zs!=safe_base)),
                     note='Numerical global-origin test; refitted scalers; no claims about detection performance')

    def inventory_runs(self):
        for dataset in self.args.datasets:
            for model in MODELS:
                for seed in self.args.seeds:
                    for fold in self.args.folds:
                        for condition in ['with_epoch_34F','without_epoch_33F']:
                            root=self.legacy/'runs'/dataset/condition/model/f'seed_{seed:02d}'/f'fold_{fold:02d}'
                            present={}
                            for name in ['metrics.json','test_predictions.csv.gz','training_history.csv','model.keras']:
                                present[name]=self.file(root/name,'primary_final_artifact',dataset)['exists']
                            self.add('21_run_availability',block='primary_34F_33F',dataset=dataset,model=model,seed=seed,fold=fold,
                                     condition=condition,root=str(root),**present,interpretation='presence only, not retraining/replay')
                        root=self.controlled_root(dataset,model,seed,fold)
                        names=['RUN_COMPLETE.json','run_manifest.json','thresholds.json','lambda_summary.csv',
                               'sample_level_intervention.csv.gz','reconstruction_se_by_lambda.npz','model.keras','model.weights.h5']
                        present={name:self.file(root/name,'controlled_final_candidate',dataset)['exists'] for name in names}
                        self.add('21_run_availability',block='controlled_final_only',dataset=dataset,model=model,seed=seed,fold=fold,
                                 condition='with_epoch_34F',root=str(root),**present,interpretation='Only designated final directories are eligible; archived failures are never aggregated')

    def controlled_root(self,dataset,model,seed,fold):
        p=self.legacy/'controlled_temporal_intervention'
        if model=='SNN_AE': return p/'snn_3datasets_30fold_v1'/'runs'/dataset/f'seed_{seed:02d}'/f'fold_{fold:02d}'
        return p/'ae3_3datasets_30fold_v1'/'runs'/dataset/model/f'seed_{seed:02d}'/f'fold_{fold:02d}'

    def endpoints(self,dataset):
        if self.args.skip_endpoints:
            self.issue('NOT_ASSESSED','endpoints',dataset,'Disabled by --skip-endpoints','Run without flag to analyze stored final decisions')
            return
        for seed in self.args.seeds:
            for fold in self.args.folds:
                self.log.info('READ stored endpoints %s seed=%s fold=%s',dataset,seed,fold)
                bymodel={}
                for model in MODELS:
                    root=self.controlled_root(dataset,model,seed,fold)
                    if not is_regular(root/'sample_level_intervention.csv.gz'):
                        self.add('22_endpoint_qc',dataset=dataset,model=model,seed=seed,fold=fold,status='MISSING_FINAL_PREDICTIONS')
                        continue
                    try:
                        ep=self.read_endpoint(dataset,model,seed,fold,root)
                        bymodel[model]=ep
                        if not self.args.skip_residuals and is_regular(root/'reconstruction_se_by_lambda.npz'):
                            try: self.residuals(dataset,model,seed,fold,root,ep)
                            except Exception as e: self.issue('ERROR','residuals',dataset,f'{model} seed={seed} fold={fold}: {e}','Residual analysis excluded; endpoint metrics remain separately assessed')
                    except Exception as e:
                        self.add('22_endpoint_qc',dataset=dataset,model=model,seed=seed,fold=fold,status='ERROR',error=f'{type(e).__name__}: {e}')
                        self.issue('ERROR','endpoints',dataset,f'{model} seed={seed} fold={fold}: {e}','Resolve before using paired/four-model analysis')
                if len(bymodel)==4:
                    try: self.shared(dataset,seed,fold,bymodel)
                    except Exception as e:
                        self.add('25_shared_failures',dataset=dataset,seed=seed,fold=fold,status='ERROR_ALIGNMENT',error=str(e))
                        self.issue('ERROR','four_model_join',dataset,f'seed={seed} fold={fold}: {e}','No four-model conclusion for this unit')
                else:
                    self.add('25_shared_failures',dataset=dataset,seed=seed,fold=fold,status='NOT_COMPUTED_MISSING_MODELS',
                             available_models=';'.join(bymodel),required_models=4)
                del bymodel
                gc.collect()

    def read_endpoint(self,dataset,model,seed,fold,root):
        d=self.data[dataset]; sp=self.splitdata.get((dataset,seed,fold))
        if sp is None: raise ValueError('No validated frozen split')
        tr,va,te,split_hash=sp; key=dict(dataset=dataset,model=model,seed=seed,fold=fold)
        done=read_json(root/'RUN_COMPLETE.json'); manifest=read_json(root/'run_manifest.json'); th=read_json(root/'thresholds.json')
        if done.get('status')!='PASS' or manifest.get('status')!='PASS': raise ValueError('Final completion/manifest does not state PASS')
        for field,expected in [('dataset',dataset),('model',model),('seed',seed),('fold',fold)]:
            if manifest.get(field)!=expected: raise ValueError(f'Manifest {field} mismatch')
        hash_verified=True
        for field,expected in [('sample_sha256',d['source_info']['sha256']),('split_file_sha256',split_hash)]:
            found=manifest.get(field)
            if found and found!=expected: raise ValueError(f'Manifest {field} mismatch')
            if not found: hash_verified=False
        cols=['source_row_index','y_true','attack_group','lambda','full_score','non_epoch_score','full_prediction','non_epoch_prediction']
        chunks=[]; count=Counter(); total=0
        with pd.read_csv(root/'sample_level_intervention.csv.gz',usecols=cols,dtype={'source_row_index':'Int64','attack_group':'string'},chunksize=self.args.csv_chunksize) as reader:
            for ch in reader:
                total+=len(ch)
                if total>self.args.max_prediction_rows: raise ValueError('Prediction CSV exceeds configured row limit')
                lamb=pd.to_numeric(ch['lambda'],errors='coerce').to_numpy(float)
                if not np.isfinite(lamb).all() or not np.isin(lamb,LAMBDAS).all(): raise ValueError('Unexpected lambda values')
                count.update(lamb.tolist())
                chunks.append(ch.loc[np.isin(lamb,[0.,1.])].copy())
        if any(count[v]!=len(te) for v in LAMBDAS): raise ValueError(f'Incomplete lambda grid counts: {dict(count)}')
        frame=pd.concat(chunks,ignore_index=True); endpoints={}; pending_metrics=[]
        expected_ids=d['src'][te]
        for lam in [0.,1.]:
            f=frame[frame['lambda']==lam].copy(); f['source_row_index']=ids64(f['source_row_index'])
            if f['source_row_index'].duplicated().any(): raise ValueError('Duplicate source ID at endpoint')
            if set(f['source_row_index'])!=set(expected_ids): raise ValueError('Test identity set mismatch')
            f=f.set_index('source_row_index').loc[expected_ids].reset_index()
            if not np.array_equal(f['y_true'].to_numpy(),d['y'][te]): raise ValueError('Test labels mismatch')
            if f['attack_group'].isna().any(): raise ValueError('Missing attack group')
            groups_verified=(dataset!='ton_iot' or dataset in self.corpora)
            if groups_verified and not np.array_equal(f['attack_group'].astype(str).to_numpy(),d['groups'][te].astype(str)):
                raise ValueError('Attack groups disagree with source/corpus manifest')
            for track in ['full','non_epoch']:
                s=f[track+'_score'].to_numpy(float); tau=float(th[track+'_threshold'])
                manifest_tau=manifest.get('full_threshold' if track=='full' else 'non_epoch_threshold')
                if manifest_tau is not None and float(manifest_tau)!=tau: raise ValueError('Threshold file disagrees with manifest')
                pred=f[track+'_prediction'].to_numpy()
                if not np.isin(pred,[0,1]).all() or not np.array_equal((s>tau).astype(np.int8),pred): raise ValueError('Stored score/threshold/prediction mismatch')
                row=metrics(d['y'][te],s,tau)
                pending_metrics.append(dict(**key,track=track,lambda_value=lam,**row))
            endpoints[lam]=f
        f1,f0=endpoints[1.],endpoints[0.]
        if not np.array_equal(f1['attack_group'].astype(str),f0['attack_group'].astype(str)): raise ValueError('Group changed across endpoints')
        for row in pending_metrics: self.add('23_endpoint_metrics',**row)
        self.add('22_endpoint_qc',**key,status='VALIDATED_STORED_ENDPOINTS',rows_per_endpoint=len(te),
                 all_lambda_row_counts_valid=True,manifest_input_hashes_verified=hash_verified,attack_groups_source_verified=groups_verified,
                 middle_lambda_unique_ids_verified=False,neural_inference_replayed=False,
                 note='Original source IDs/labels and fixed-threshold decisions checked; not checkpoint replay')
        for track in ['full','non_epoch']:
            p1=f1[track+'_prediction'].to_numpy(int);p0=f0[track+'_prediction'].to_numpy(int);y=d['y'][te]
            for cl,mask in [('NORMAL',y==0),('ATTACK',y==1)]:
                b=p1[mask];c=p0[mask];n=len(b)
                self.add('24_endpoint_transitions',**key,track=track,class_name=cl,n=n,
                         baseline0_current0=int(np.sum((b==0)&(c==0))),baseline0_current1=int(np.sum((b==0)&(c==1))),
                         baseline1_current0=int(np.sum((b==1)&(c==0))),baseline1_current1=int(np.sum((b==1)&(c==1))),
                         any_change_ratio=float(np.mean(b!=c)) if n else None)
        return endpoints

    def shared(self,dataset,seed,fold,by):
        f=by[MODELS[0]][1.]; y=f['y_true'].to_numpy(int);groups=f['attack_group'].astype(str).to_numpy();ids=ids64(f['source_row_index'])
        for model in MODELS:
            for lam in [0.,1.]:
                g=by[model][lam]
                if not np.array_equal(ids,ids64(g['source_row_index'])) or not np.array_equal(y,g['y_true'].to_numpy()) or not np.array_equal(groups,g['attack_group'].astype(str).to_numpy()):
                    raise ValueError('Cross-model source/label/group alignment mismatch')
        p1=np.column_stack([by[m][1.]['non_epoch_prediction'].to_numpy(int) for m in MODELS])
        p0=np.column_stack([by[m][0.]['non_epoch_prediction'].to_numpy(int) for m in MODELS])
        flips=(p1==1)&(p0==0); k=flips.sum(axis=1)
        masks=[('ALL_ATTACK',y==1)]+[(str(g),(y==1)&(groups==g)) for g in sorted(set(groups[y==1]))]
        for name,mask in masks:
            if not mask.any(): continue
            self.add('25_shared_failures',dataset=dataset,seed=seed,fold=fold,status='DIRECT_FOUR_MODEL_JOIN',group=name,n=int(mask.sum()),
                     **{f'lost_detection_in_{j}_models_n':int(np.sum(mask&(k==j))) for j in range(5)},
                     all_four_missed_lambda0_n=int(np.sum(mask&np.all(p0==0,axis=1))),
                     all_four_detected1_to_all_four_missed0_n=int(np.sum(mask&np.all(p1==1,axis=1)&np.all(p0==0,axis=1))),
                     denominator='held-out packet-seed evaluations; folds partition packets within a seed')
            for a,b in itertools.combinations(range(4),2):
                union=mask&(flips[:,a]|flips[:,b]);inter=mask&flips[:,a]&flips[:,b]
                self.add('26_pairwise_failures',dataset=dataset,seed=seed,fold=fold,group=name,model_a=MODELS[a],model_b=MODELS[b],
                         union_n=int(union.sum()),intersection_n=int(inter.sum()),jaccard=float(inter.sum()/union.sum()) if union.any() else None)
        for lam,p in [(0.,p0),(1.,p1)]:
            s=np.any(p==1,axis=1).astype(float)
            self.add('27_or_coverage',dataset=dataset,seed=seed,fold=fold,lambda_value=lam,
                     interpretation='OR of four fixed binary decisions, no ensemble retraining',**metrics(y,s,.5))

    def residuals(self,dataset,model,seed,fold,root,ep):
        p=root/'reconstruction_se_by_lambda.npz'
        block=self.legacy/'controlled_temporal_intervention'/('snn_3datasets_30fold_v1' if model=='SNN_AE' else 'ae3_3datasets_30fold_v1')
        protocol_paths=[block/'metadata'/'protocol_freeze_v3.json',block/'metadata'/'protocol_freeze.json']
        protocol_path=next((q for q in protocol_paths if is_regular(q)),None)
        if protocol_path is None or read_json(protocol_path).get('feature_columns')!=self.feature_names:
            raise ValueError('Cannot verify residual feature order from controlled protocol')
        with checked_npz(p,self.args.max_npz_mb) as z:
            lam=np.asarray(z['lambdas'],float); ids=ids64(z['source_row_index']); y=np.asarray(z['y_true'])
            if len(np.unique(lam))!=len(lam) or not np.isin([0.,1.],lam).all(): raise ValueError('Residual endpoints missing/duplicate')
            se=np.asarray(z['squared_reconstruction_error'])
        if se.shape!=(len(lam),len(ids),len(self.feature_names)): raise ValueError('Residual tensor shape mismatch')
        if not np.isfinite(se).all() or np.any(se<0): raise ValueError('Invalid squared errors')
        if len(np.unique(ids))!=len(ids): raise ValueError('Residual duplicate IDs')
        fs={l:ep[l].set_index('source_row_index').loc[ids] for l in [0.,1.]}
        if not np.array_equal(y,fs[1.]['y_true'].to_numpy()): raise ValueError('Residual labels mismatch')
        a=se[np.flatnonzero(lam==1.)[0]].astype(np.float64);b=se[np.flatnonzero(lam==0.)[0]].astype(np.float64)
        keep=[j for j in range(len(self.feature_names)) if j!=self.epoch_idx]
        for l,v in [(1.,a),(0.,b)]:
            if not np.allclose(v[:,keep].mean(axis=1),fs[l]['non_epoch_score'].to_numpy(float),rtol=2e-5,atol=1e-8): raise ValueError('Residual non-epoch score mismatch')
        p1=fs[1.]['non_epoch_prediction'].to_numpy(int);p0=fs[0.]['non_epoch_prediction'].to_numpy(int)
        strata=[('normal',y==0),('all_attack',y==1),('attack_detected_to_missed',(y==1)&(p1==1)&(p0==0)),
                ('attack_still_detected',(y==1)&(p1==1)&(p0==1))]
        for name,mask in strata:
            if not mask.any(): continue
            for j in keep:
                diff=a[mask,j]-b[mask,j]
                self.add('28_residual_components',dataset=dataset,model=model,seed=seed,fold=fold,subset=name,n=int(mask.sum()),feature=self.feature_names[j],
                         error_lambda1_mean=float(a[mask,j].mean()),error_lambda0_mean=float(b[mask,j].mean()),
                         signed_error_drop_mean=float(diff.mean()),signed_error_drop_median=float(np.median(diff)),
                         contribution_to_mean_non_epoch_score_drop=float(diff.mean()/len(keep)),
                         interpretation='Controlled residual difference, not causal feature importance')
        del se,a,b

    def statistics(self):
        rows=self.rows['23_endpoint_metrics']
        expected=[(d,m) for d in self.args.datasets for m in MODELS]
        out=[]
        for dataset,model in expected:
            r=[v for v in rows if v['dataset']==dataset and v['model']==model and v['track']=='non_epoch']
            f=pd.DataFrame(r)
            if f.empty:
                out.append(dict(dataset=dataset,model=model,status='MISSING_ENDPOINTS'));continue
            pv=f.pivot(index=['seed','fold'],columns='lambda_value',values='fnr')
            if 0. not in pv or 1. not in pv: out.append(dict(dataset=dataset,model=model,status='INCOMPLETE_ENDPOINTS'));continue
            delta=(pv[0.]-pv[1.]).dropna().to_numpy(float); n=len(delta)
            if n!=len(self.args.seeds)*len(self.args.folds) or n<2:
                out.append(dict(dataset=dataset,model=model,status='INCOMPLETE_PLANNED_PAIRS',n_pairs=n));continue
            mean=float(delta.mean());sd=float(delta.std(ddof=1));se=float(np.sqrt((1/n+1/9)*sd**2));df=n-1
            p=float(2*student_t.sf(abs(mean/se),df)) if se else (1. if mean==0 else 0.)
            crit=float(student_t.ppf(.975,df))
            out.append(dict(dataset=dataset,model=model,status='COMPUTED_LEGACY_NB_CONTRACT',n_pairs=n,
                            lambda0_fnr_mean=float(pv[0.].mean()),lambda1_fnr_mean=float(pv[1.].mean()),delta_fnr=mean,
                            nb_se=se,nb_ci95_low=mean-crit*se,nb_ci95_high=mean+crit*se,nb_p_raw=p,nb_ratio=1/9,
                            note='NB ratio inherited from legacy protocol; not independent capture-level inference'))
        eligible=len(out)==12 and set(self.args.datasets)==set(NORMAL) and self.args.seeds==[0,1,2] and self.args.folds==list(range(1,11)) and all(r['status']=='COMPUTED_LEGACY_NB_CONTRACT' for r in out)
        if eligible:
            vals=np.array([r['nb_p_raw'] for r in out]);order=np.argsort(vals);run=0.
            for rank,i in enumerate(order):
                run=max(run,(len(vals)-rank)*vals[i]);out[i]['nb_p_holm12']=min(1.,run)
        for r in out:
            r['holm12_status']='COMPUTED_ALL_12_COMPLETE' if eligible else 'NOT_COMPUTED_INCOMPLETE_OR_NONCANONICAL_SCOPE'
            self.add('29_endpoint_statistics',**r)

    def discovery(self):
        roots=[self.base,self.source.parent,self.base/'0-temporal-feature-audit',self.legacy]+[Path(p).expanduser().resolve() for p in self.args.raw_root]
        seen=set(); hints=set(); candidate_paths=set(); max_count=self.args.scan_max_files
        excluded={'runs','scalers','splits','.git','__pycache__','.ipynb_checkpoints','node_modules','temporal_audit_upgrade_audits','failed_runs','archive','archived_failures'}
        count=0
        for root in dict.fromkeys(roots):
            if not root.is_dir():
                self.add('33_search_coverage',root=str(root),status='DIRECTORY_MISSING',files_considered=0);continue
            scanned=0;truncated=False;err=[]
            def onerror(e): err.append(str(e))
            for cur,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
                cp=Path(cur);depth=len(cp.relative_to(root).parts)
                dirs[:]=sorted(d for d in dirs if d not in excluded and not (cp/d).is_symlink() and self.out.parent not in (cp/d).resolve().parents and (cp/d).resolve()!=self.out.parent)
                if depth>=self.args.scan_depth: dirs[:]=[]
                for name in sorted(files):
                    p=cp/name
                    if str(p) in seen or p.is_symlink() or not is_regular(p):continue
                    seen.add(str(p));count+=1;scanned+=1
                    if count>max_count: truncated=True;break
                    low=name.lower()
                    if low.endswith(('.pcap','.pcapng','.cap','.pcap.gz','.pcapng.gz')):
                        self.add('32_raw_file_candidates',path=str(p),bytes=p.stat().st_size,
                                 relation='search_candidate_only',session_id_verified=False,full_stream_read=False)
                    if p.suffix.lower() in {'.py','.json','.txt','.md','.yaml','.yml'} and p.stat().st_size<=self.args.max_text_mb*1024**2:
                        try:
                            text=p.read_text(encoding='utf-8',errors='replace')
                            local=0
                            for no,line in enumerate(text.splitlines(),1):
                                if any(term in line.lower() for term in ['inter_arrival_time','capinfos','read_pcap','rdpcap','pcap_path','source_row_index','capture_id','session_id','pcap_provenance']):
                                    if local<self.args.max_code_hints_per_file:
                                        self.add('30_code_hints',path=str(p),line_number=no,excerpt=line[:400],evidence='static_hint_only');local+=1
                            if p.suffix.lower()=='.py':
                                try:
                                    tree=ast.parse(text)
                                    strings=[node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str)]
                                except SyntaxError:strings=[]
                            else:strings=re.findall(r"(?:/home/|/mnt/|/data/)[^\s\"'<>`\[\]{}]+",text)
                            for s in strings:
                                if s.startswith('/') and len(s)<1500 and '\n' not in s and any(w in s.lower() for w in ['pcap','dataset','parquet','materialized','capture','ton','cic','iiot']):
                                    candidate_paths.add((str(p),s.rstrip(',;:')))
                            if local or 'provenance' in low:self.file(p,'provenance_or_extractor_hint')
                        except Exception as e: self.issue('REVIEW','source_scan','',f'{p}: {e}','Inspect file manually if relevant')
                if truncated:break
            self.add('33_search_coverage',root=str(root),status='CAPPED' if truncated else 'BOUNDED_SEARCH_COMPLETE',files_considered=scanned,
                     max_depth=self.args.scan_depth,global_max_files=max_count,walk_errors=' | '.join(err),
                     note='Excluded run/scaler/archive trees; absence means not found in this scope, not absent on server')
            if truncated:break
        for origin,s in sorted(candidate_paths):
            p=Path(s)
            try:exists=p.exists();kind='directory' if p.is_dir() else 'file' if p.is_file() else 'unresolved'
            except OSError:exists=False;kind='unresolved'
            self.add('31_source_path_candidates',referenced_by=origin,path=s,exists=exists,kind=kind,
                     packet_mapping_verified=False,note='Path reference only; not automatically executed or recursively scanned')
        for dataset in self.args.datasets:
            self.add('34_feasibility',dataset=dataset,capability='raw_capture_packet_mapping',status='MANUAL_REVIEW_REQUIRED',
                     reason='Review static path/extractor hints; sample row IDs are not automatically equated with PCAP frame numbers')

    def finish(self):
        self.add('34_feasibility',dataset='ALL',capability='start_training_A_D',status='NOT_AUTHORIZED_BY_AUDIT_ALONE',
                 reason='Need reviewed data/provenance, novelty matrix, feature/window contract and locked new protocol; this program does not train')
        self.flush()
        pd.DataFrame([{'file':k+'.csv','description':v,'rows':len(self.rows[k])} for k,v in TABLES.items()]).to_csv(self.out/'TABLE_DICTIONARY.csv',index=False,encoding='utf-8-sig')
        counts=Counter(r['severity'] for r in self.rows['00_findings'])
        stage_errors=sum(r['status']=='ERROR' for r in self.rows['01_stage_status'])
        result=dict(status='INTERRUPTED_PARTIAL' if self.cancelled else 'COMPLETED_WITH_FINDINGS' if counts['ERROR'] or stage_errors else 'COMPLETED',
                    elapsed_seconds=time.monotonic()-self.started,findings_by_severity=dict(counts),
                    stage_errors=stage_errors,version=VERSION,rows_by_table={k:len(v) for k,v in self.rows.items()},
                    scientific_pass=False,inputs_modified=False,training_executed=False)
        (self.out/'RUN_STATUS.json').write_text(json.dumps(plain(result),indent=2),encoding='utf-8')
        text=f'''# TEMPORAL_AUDIT consolidated audit results\n\nExecution status: {result['status']}\n\nRead 00_findings.csv, 01_stage_status.csv and TABLE_DICTIONARY.csv first.\nThis is a diagnostic report, NOT a certification of data quality or novelty.\nAll raw-source files, original splits, models and previous results were left unchanged.\nWhole-materialized and frozen-corpus scopes are separate.\nNo negative IAT values were clipped. No sequence/sessions were invented.\nNo full-stream behavior extraction, new splits, neural training or checkpoint inference was performed.\n\nPer-fold scaling: current-sklearn reconstruction is explicitly distinguished from a loaded verified scaler.\nThe precision-safe timestamp diagnostic centers in float64 before float32 conversion.\nIts results are input changes, NOT evidence of a change in FNR.\n\nFeature hash matches are candidate representation matches, NOT proof of duplicated packets.\nFour-model intersections use only verified final endpoint identities; incomplete groups are marked.\nThe NB/Holm table only rechecks the old endpoint contract, not the upgraded RQs.\nRaw capture counts are not independent session counts. Search bounds are in 33_search_coverage.csv.\n\nThe upload ZIP contains reports, configuration and bounded code/value hints only.\nIt does not include datasets, checkpoints, payloads or the original prediction files.\nPaths, labels and small timestamp/ID examples may be sensitive; review before sharing externally.\n'''
        (self.out/'READ_ME_FIRST.md').write_text(text,encoding='utf-8')
        checks=[]
        for p in sorted(self.out.iterdir()):
            if p.is_file() and p.name not in {'SHA256SUMS.txt','audit.log'}:checks.append(f'{digest(p)}  {p.name}')
        (self.out/'SHA256SUMS.txt').write_text('\n'.join(checks)+'\n',encoding='ascii')
        zp=self.out.parent/('TEMPORAL_AUDIT_AUDIT_RESULTS_'+self.out.name+'.zip')
        for h in self.log.handlers: h.flush()
        with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,arcname=p.name)
        self.log.info('REPORT DIRECTORY: %s',self.out)
        self.log.info('UPLOAD THIS ZIP: %s',zp)
        self.log.info('STATUS: %s. REVIEW/ERROR findings do not mean a training job ran.',result['status'])
        return result,zp


def has_distribution(name):
    try:ilm.version(name);return True
    except ilm.PackageNotFoundError:return False


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base',default=BASE_DEFAULT)
    p.add_argument('--samples-dir',default='')
    p.add_argument('--legacy-root',default='')
    p.add_argument('--output-root',default='')
    p.add_argument('--datasets',nargs='+',choices=list(NORMAL),default=list(NORMAL))
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--folds',nargs='+',type=int,default=list(range(1,11)))
    p.add_argument('--trust-local-scalers',action='store_true',help='Explicitly permit joblib/pickle loading of YOUR trusted local MinMaxScaler files; never use for untrusted files')
    p.add_argument('--skip-endpoints',action='store_true',help='Only inventories final prediction files; skips decision analysis')
    p.add_argument('--skip-residuals',action='store_true',help='Skip reading residual NPZ tensors to reduce I/O')
    p.add_argument('--raw-root',action='append',default=[],help='Additional explicitly known data root to inventory; repeatable')
    p.add_argument('--scan-depth',type=int,default=6)
    p.add_argument('--scan-max-files',type=int,default=20000)
    p.add_argument('--max-text-mb',type=float,default=2.)
    p.add_argument('--max-code-hints-per-file',type=int,default=12)
    p.add_argument('--hash-max-mb',type=float,default=16.)
    p.add_argument('--max-npz-mb',type=float,default=512.)
    p.add_argument('--max-materialized-rows',type=int,default=1000000)
    p.add_argument('--max-prediction-rows',type=int,default=2000000)
    p.add_argument('--csv-chunksize',type=int,default=50000)
    p.add_argument('--max-examples',type=int,default=30)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    args.datasets=list(dict.fromkeys(args.datasets));args.seeds=sorted(set(args.seeds));args.folds=sorted(set(args.folds))
    if not args.seeds or min(args.seeds)<0 or min(args.folds)<1:raise SystemExit('Invalid seed/fold values')
    for k in ['scan_depth','scan_max_files','max_text_mb','max_code_hints_per_file','hash_max_mb','max_npz_mb','max_materialized_rows','max_prediction_rows','csv_chunksize','max_examples']:
        if getattr(args,k)<=0:raise SystemExit(f'--{k.replace("_","-")} must be positive')
    ctx=Audit(args)
    try:
        ctx.stage('configuration',ctx.config)
        for dataset in args.datasets:
            ctx.stage('dataset:'+dataset,ctx.read_dataset,dataset)
            if dataset in ctx.data:
                ctx.stage('corpus:'+dataset,ctx.corpus,dataset)
                ctx.stage('frozen_splits:'+dataset,ctx.frozen_splits,dataset)
        ctx.stage('artifact_inventory',ctx.inventory_runs)
        ctx.stage('provenance_search',ctx.discovery)
        for dataset in args.datasets:
            if dataset in ctx.data:ctx.stage('stored_endpoints:'+dataset,ctx.endpoints,dataset)
        ctx.stage('legacy_endpoint_statistics',ctx.statistics)
    except KeyboardInterrupt:
        ctx.cancelled=True
        ctx.issue('ERROR','execution','','Interrupted by user','Partial CSVs will be packaged; no completion claim')
    result,_=ctx.finish()
    return 130 if ctx.cancelled else 2 if result['stage_errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
