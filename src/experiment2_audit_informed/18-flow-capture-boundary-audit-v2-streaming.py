#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
SOURCE_V2 = BASE / '0-temporal-feature-audit-v2-3datasets'

DATASETS = {
    'edge_iiotset': {
        'sample': SOURCE_V2 / 'materialized_samples' / 'edge_iiotset_audit_sample.parquet',
    },
    'ciciot2023': {
        'sample': SOURCE_V2 / 'materialized_samples' / 'ciciot2023_audit_sample.parquet',
    },
    'ton_iot': {
        'sample': SOURCE_V2 / 'materialized_samples' / 'ton_iot_audit_sample.parquet',
    },
}

SESSION_HINTS = ('capture','pcap','session','trace','episode','scenario','source_file','capture_file','pcap_file')
BASE_COLUMNS = ['frame.time_epoch','frame.len','inter_arrival_time','flow_id','packet_index','label','label_type']


def utc_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def sha256_file(path: Path, block=1024*1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-provenance', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, default=BASE / 'temporal_audit_flow_boundary_audits_v2')
    ap.add_argument('--batch-size', type=int, default=100000)
    ap.add_argument('--probe-n', type=int, default=96)
    ap.add_argument('--event-sample-max', type=int, default=2000)
    ap.add_argument('--negative-iat-context-max', type=int, default=5000)
    ap.add_argument('--datasets', nargs='*', choices=list(DATASETS), default=list(DATASETS))
    return ap.parse_args()


def find_one(root: Path, name: str):
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f'{name} not found under {root}')
    raise RuntimeError(f'Multiple {name} files under {root}; pass exact Script 17 run directory')


def load_parent_decisions(prior: Path):
    p = find_one(prior, '14_parent_decisions.csv')
    df = pd.read_csv(p)
    required = {'dataset','parent_path','status','complete_mapping'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'{p}: missing columns {sorted(missing)}')
    return df, p


def validated_parent(decisions, dataset):
    r = decisions.loc[decisions['dataset'].astype(str) == dataset]
    if len(r) != 1:
        raise RuntimeError(f'{dataset}: expected one parent decision, got {len(r)}')
    r = r.iloc[0]
    if str(r['status']) != 'CONTENT_VALIDATED_PARENT_LINK':
        raise RuntimeError(f"{dataset}: Script 17 parent status is {r['status']}")
    complete = str(r['complete_mapping']).strip().lower() in {'true','1','yes'}
    if not complete:
        raise RuntimeError(f'{dataset}: Script 17 did not report complete_mapping')
    p = Path(str(r['parent_path']))
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def schema_names(path: Path):
    return pq.ParquetFile(path).schema_arrow.names


def sparse_take(path: Path, positions, columns):
    pf = pq.ParquetFile(path)
    avail = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in avail]
    positions = np.asarray(sorted(set(map(int, positions))), dtype=np.int64)
    if len(positions) == 0:
        return pd.DataFrame()
    parts = []
    start = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        end = start + n
        wanted = positions[(positions >= start) & (positions < end)]
        if len(wanted):
            tbl = pf.read_row_group(rg, columns=cols)
            local = wanted - start
            d = tbl.take(pa.array(local)).to_pandas()
            d.insert(0, '__parent_pos', wanted)
            parts.append(d)
        start = end
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verify_parent_probe(dataset, cfg, parent, probe_n):
    sample_path = cfg['sample']
    snames = set(schema_names(sample_path))
    cols = [c for c in ['source_row_index','frame.time_epoch','frame.len','label','label_type'] if c in snames]
    if 'source_row_index' not in cols:
        raise RuntimeError(f'{dataset}: sample lacks source_row_index')
    sample = pd.read_parquet(sample_path, columns=cols)
    if len(sample) > probe_n:
        idx = np.linspace(0, len(sample)-1, probe_n, dtype=int)
        sample = sample.iloc[idx].copy()
    pos = sample['source_row_index'].to_numpy(dtype=np.int64)
    fields = [c for c in cols if c != 'source_row_index']
    par = sparse_take(parent, pos, fields)
    m = sample.merge(par, left_on='source_row_index', right_on='__parent_pos', suffixes=('_sample','_parent'))
    out = {'dataset':dataset,'probe_n':int(len(sample)),'matched_rows':int(len(m))}
    if len(m) != len(sample):
        out['status'] = 'FAIL'
        return out
    all_ok = True
    for c in fields:
        a, b = m[f'{c}_sample'], m[f'{c}_parent']
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-12, equal_nan=True)
        else:
            ok = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        out[f'{c}_match_n'] = int(np.sum(ok))
        all_ok = all_ok and bool(np.all(ok))
    out['status'] = 'PASS' if all_ok else 'FAIL'
    return out


def direct_metadata_cols(names):
    out = []
    for c in names:
        low = c.lower()
        if any(tok in low for tok in SESSION_HINTS):
            if c not in {'frame.time_epoch','inter_arrival_time','flow_id','packet_index'}:
                out.append(c)
    return out


def open_flow_db(path: Path):
    con = sqlite3.connect(path)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    con.execute('PRAGMA temp_store=FILE')
    con.execute('''
        CREATE TABLE IF NOT EXISTS flow_stats(
            flow_id TEXT PRIMARY KEY,
            row_count INTEGER NOT NULL,
            first_row INTEGER NOT NULL,
            last_row INTEGER NOT NULL,
            first_epoch REAL,
            last_epoch REAL,
            label_repr TEXT,
            multiple_labels INTEGER NOT NULL,
            table_runs INTEGER NOT NULL,
            packet_index_min REAL,
            packet_index_max REAL
        )
    ''')
    con.commit()
    return con


UPSERT_SQL = '''
INSERT INTO flow_stats(
    flow_id,row_count,first_row,last_row,first_epoch,last_epoch,
    label_repr,multiple_labels,table_runs,packet_index_min,packet_index_max
) VALUES(?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(flow_id) DO UPDATE SET
    row_count = flow_stats.row_count + excluded.row_count,
    first_row = MIN(flow_stats.first_row, excluded.first_row),
    last_row = MAX(flow_stats.last_row, excluded.last_row),
    first_epoch = CASE
        WHEN flow_stats.first_epoch IS NULL THEN excluded.first_epoch
        WHEN excluded.first_epoch IS NULL THEN flow_stats.first_epoch
        ELSE MIN(flow_stats.first_epoch, excluded.first_epoch)
    END,
    last_epoch = CASE
        WHEN flow_stats.last_epoch IS NULL THEN excluded.last_epoch
        WHEN excluded.last_epoch IS NULL THEN flow_stats.last_epoch
        ELSE MAX(flow_stats.last_epoch, excluded.last_epoch)
    END,
    multiple_labels = CASE
        WHEN flow_stats.multiple_labels = 1 OR excluded.multiple_labels = 1 THEN 1
        WHEN flow_stats.label_repr IS NOT NULL AND excluded.label_repr IS NOT NULL
             AND flow_stats.label_repr <> excluded.label_repr THEN 1
        ELSE 0
    END,
    label_repr = COALESCE(flow_stats.label_repr, excluded.label_repr),
    table_runs = flow_stats.table_runs + excluded.table_runs,
    packet_index_min = CASE
        WHEN flow_stats.packet_index_min IS NULL THEN excluded.packet_index_min
        WHEN excluded.packet_index_min IS NULL THEN flow_stats.packet_index_min
        ELSE MIN(flow_stats.packet_index_min, excluded.packet_index_min)
    END,
    packet_index_max = CASE
        WHEN flow_stats.packet_index_max IS NULL THEN excluded.packet_index_max
        WHEN excluded.packet_index_max IS NULL THEN flow_stats.packet_index_max
        ELSE MAX(flow_stats.packet_index_max, excluded.packet_index_max)
    END
'''


def scalar_or_none(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, np.generic):
        return x.item()
    return x


def aggregate_batch_to_db(con, df, global_start, prev_flow):
    if 'flow_id' not in df.columns:
        return prev_flow, 0
    flow = df['flow_id'].astype('string')
    missing_n = int(flow.isna().sum())
    key = flow.fillna('<NA>')
    rowpos = np.arange(global_start, global_start + len(df), dtype=np.int64)
    work = pd.DataFrame({
        '__flow': key,
        '__row': rowpos,
        '__epoch': pd.to_numeric(df['frame.time_epoch'], errors='coerce'),
        '__label': df['label'].astype('string') if 'label' in df.columns else pd.Series([pd.NA]*len(df), dtype='string'),
        '__pi': pd.to_numeric(df['packet_index'], errors='coerce') if 'packet_index' in df.columns else np.nan,
    })
    kval = key.to_numpy(dtype=object)
    starts = np.ones(len(work), dtype=bool)
    if len(work) > 1:
        starts[1:] = kval[1:] != kval[:-1]
    if len(work) and prev_flow is not None and kval[0] == prev_flow:
        starts[0] = False
    run_counts = pd.Series(kval[starts]).value_counts(dropna=False)
    gb = work.groupby('__flow', sort=False, dropna=False)
    base = gb.agg(
        row_count=('__row','size'), first_row=('__row','min'), last_row=('__row','max'),
        first_epoch=('__epoch','min'), last_epoch=('__epoch','max'),
        packet_index_min=('__pi','min'), packet_index_max=('__pi','max')
    )
    label_first = gb['__label'].first()
    label_nuniq = gb['__label'].nunique(dropna=True)
    records = []
    for f, r in base.iterrows():
        if f == '<NA>':
            continue
        records.append((
            str(f), int(r['row_count']), int(r['first_row']), int(r['last_row']),
            scalar_or_none(r['first_epoch']), scalar_or_none(r['last_epoch']),
            scalar_or_none(label_first.get(f)), int(label_nuniq.get(f,0)>1),
            int(run_counts.get(f,0)), scalar_or_none(r['packet_index_min']), scalar_or_none(r['packet_index_max'])
        ))
    con.executemany(UPSERT_SQL, records)
    con.commit()
    return (str(kval[-1]) if len(kval) else prev_flow), missing_n


def sqlite_summary(con):
    cur = con.cursor()
    return {
        'unique_flow_id': int(cur.execute('SELECT COUNT(*) FROM flow_stats').fetchone()[0]),
        'flows_with_multiple_labels_n': int(cur.execute('SELECT COUNT(*) FROM flow_stats WHERE multiple_labels=1').fetchone()[0]),
        'flows_in_multiple_table_runs_n': int(cur.execute('SELECT COUNT(*) FROM flow_stats WHERE table_runs>1').fetchone()[0]),
        'flow_row_count_mean': float(cur.execute('SELECT COALESCE(AVG(row_count),0) FROM flow_stats').fetchone()[0]),
        'flow_row_count_max': int(cur.execute('SELECT COALESCE(MAX(row_count),0) FROM flow_stats').fetchone()[0]),
    }


def event_append(events, event_type, positions, max_n):
    current = sum(1 for e in events if e['event_type'] == event_type)
    room = max_n - current
    if room <= 0:
        return
    for p in list(positions)[:room]:
        events.append({'event_type':event_type,'parent_row_position':int(p)})


def build_context(parent, centers, radius=2):
    if not centers:
        return pd.DataFrame()
    names = set(schema_names(parent))
    cols = [c for c in BASE_COLUMNS if c in names]
    wanted = set()
    for c in centers:
        for p in range(max(0,int(c)-radius), int(c)+radius+1):
            wanted.add(p)
    rows = sparse_take(parent, sorted(wanted), cols)
    if rows.empty:
        return rows
    out = []
    for center in centers:
        sub = rows[(rows['__parent_pos'] >= int(center)-radius) & (rows['__parent_pos'] <= int(center)+radius)].copy()
        sub.insert(0,'center_position',int(center))
        sub['relative_position'] = sub['__parent_pos'] - int(center)
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def audit_dataset(dataset, cfg, parent, out, args, log):
    pf = pq.ParquetFile(parent)
    names = pf.schema_arrow.names
    nrows = int(pf.metadata.num_rows)
    meta_cols = direct_metadata_cols(names)
    cols = [c for c in dict.fromkeys(BASE_COLUMNS + meta_cols) if c in names]
    if 'frame.time_epoch' not in cols:
        raise RuntimeError(f'{dataset}: parent lacks frame.time_epoch')

    probe = verify_parent_probe(dataset, cfg, parent, args.probe_n)
    if probe['status'] != 'PASS':
        raise RuntimeError(f'{dataset}: sparse parent verification failed: {probe}')

    dsdir = out / dataset
    dsdir.mkdir(parents=True, exist_ok=True)
    save_json(dsdir/'01_parent_probe.json', probe)
    con = open_flow_db(dsdir/'flow_stats.sqlite')

    global_pos = 0
    prev_epoch = prev_pi = prev_flow = None
    prev_meta = {}
    missing_flow_total = 0
    packet_reset_n = epoch_backstep_n = negative_iat_n = 0
    negative_iat_min = negative_iat_max = None
    metadata_change_counts = {c:0 for c in meta_cols}
    events = []
    negative_centers = []
    batch_no = 0

    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols, use_threads=True):
        batch_no += 1
        df = batch.to_pandas()
        n = len(df)
        if not n:
            continue
        t = pd.to_numeric(df['frame.time_epoch'], errors='coerce').to_numpy(dtype=float)
        if prev_epoch is not None and np.isfinite(prev_epoch) and np.isfinite(t[0]) and t[0] < prev_epoch:
            epoch_backstep_n += 1
            event_append(events,'epoch_backstep',[global_pos],args.event_sample_max)
        if n > 1:
            dt = np.diff(t)
            loc = np.where(np.isfinite(dt) & (dt < 0))[0] + 1
            epoch_backstep_n += int(len(loc))
            event_append(events,'epoch_backstep',global_pos+loc,args.event_sample_max)
        prev_epoch = t[-1]

        if 'packet_index' in df.columns:
            pi = pd.to_numeric(df['packet_index'], errors='coerce').to_numpy(dtype=float)
            if prev_pi is not None and np.isfinite(prev_pi) and np.isfinite(pi[0]) and ((pi[0] < prev_pi) or (pi[0] <= 1 and prev_pi > 1)):
                packet_reset_n += 1
                event_append(events,'packet_index_reset',[global_pos],args.event_sample_max)
            if n > 1:
                valid = np.isfinite(pi[1:]) & np.isfinite(pi[:-1])
                loc = np.where(valid & ((pi[1:] < pi[:-1]) | ((pi[1:] <= 1) & (pi[:-1] > 1))))[0] + 1
                packet_reset_n += int(len(loc))
                event_append(events,'packet_index_reset',global_pos+loc,args.event_sample_max)
            prev_pi = pi[-1]

        if 'inter_arrival_time' in df.columns:
            iat = pd.to_numeric(df['inter_arrival_time'], errors='coerce').to_numpy(dtype=float)
            neg = np.where(np.isfinite(iat) & (iat < 0))[0]
            if len(neg):
                vals = iat[neg]
                negative_iat_n += int(len(neg))
                mn, mx = float(np.min(vals)), float(np.max(vals))
                negative_iat_min = mn if negative_iat_min is None else min(negative_iat_min,mn)
                negative_iat_max = mx if negative_iat_max is None else max(negative_iat_max,mx)
                room = max(0,args.negative_iat_context_max-len(negative_centers))
                negative_centers.extend([int(global_pos+x) for x in neg[:room]])

        for c in meta_cols:
            s = df[c].astype('string').fillna('<NA>').to_numpy(dtype=object)
            positions = []
            if prev_meta.get(c) is not None and s[0] != prev_meta[c]:
                positions.append(global_pos)
            if n > 1:
                loc = np.where(s[1:] != s[:-1])[0] + 1
                positions.extend((global_pos+loc).tolist())
            metadata_change_counts[c] += len(positions)
            event_append(events,f'metadata_change:{c}',positions,args.event_sample_max)
            prev_meta[c] = s[-1]

        prev_flow, miss = aggregate_batch_to_db(con,df,global_pos,prev_flow)
        missing_flow_total += int(miss)
        global_pos += n
        if batch_no == 1 or batch_no % 10 == 0 or global_pos == nrows:
            log(f'{dataset}: rows={global_pos:,}/{nrows:,} batches={batch_no}')
        del df

    if global_pos != nrows:
        raise RuntimeError(f'{dataset}: streamed {global_pos} rows, expected {nrows}')

    fs = sqlite_summary(con) if 'flow_id' in names else {
        'unique_flow_id':0,'flows_with_multiple_labels_n':0,'flows_in_multiple_table_runs_n':0,
        'flow_row_count_mean':0.0,'flow_row_count_max':0
    }
    pd.read_sql_query('SELECT * FROM flow_stats ORDER BY row_count DESC LIMIT 200',con).to_csv(dsdir/'10_largest_flows.csv',index=False)
    pd.read_sql_query('SELECT * FROM flow_stats WHERE table_runs>1 OR multiple_labels=1 ORDER BY multiple_labels DESC, table_runs DESC, row_count DESC LIMIT 500',con).to_csv(dsdir/'11_recurring_or_mixed_flows.csv',index=False)

    seq = {
        'dataset':dataset,'parent_path':str(parent),'parent_rows':nrows,
        'columns_scanned':'|'.join(cols),'direct_metadata_columns':'|'.join(meta_cols),
        'packet_index_reset_n':packet_reset_n,'epoch_backstep_n':epoch_backstep_n,
        'negative_iat_n':negative_iat_n,'negative_iat_min_seconds':negative_iat_min,
        'negative_iat_max_seconds':negative_iat_max,'missing_flow_id_n':missing_flow_total,**fs
    }
    for c,nchange in metadata_change_counts.items():
        seq[f'metadata_change_count__{c}'] = int(nchange)
    save_csv(dsdir/'12_sequence_flow_summary.csv',[seq])
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev.insert(0,'dataset',dataset)
    ev.to_csv(dsdir/'13_boundary_event_samples.csv',index=False)
    ctx = build_context(parent,negative_centers,radius=2)
    if not ctx.empty:
        ctx.insert(0,'dataset',dataset)
    ctx.to_csv(dsdir/'14_negative_iat_context.csv',index=False)

    has_flow = 'flow_id' in names
    flow_ready = has_flow and missing_flow_total == 0 and fs['flows_with_multiple_labels_n'] == 0
    feas = {
        'dataset':dataset,
        'flow_grouping_available':bool(has_flow),
        'flow_candidate_ready':bool(flow_ready),
        'flow_candidate_reason':'flow_id complete and no mixed-label flow observed' if flow_ready else 'flow_id absent/missing or mixed-label flows observed; review required',
        'capture_disjoint_ready':False,
        'capture_disjoint_reason':'No independently validated complete row-level capture mapping is certified by this audit.',
        'session_disjoint_ready':False,
        'session_disjoint_reason':'No independently validated session identifier is certified; capture and session are not treated as synonyms.',
        'blocked_time_ready':True,
        'blocked_time_reason':'frame.time_epoch is available for chronological blocked-time candidates.',
        'direct_metadata_columns':'|'.join(meta_cols),
        'recommended_candidate':'FLOW_DISJOINT_CANDIDATE' if flow_ready else 'BLOCKED_TIME_CANDIDATE'
    }
    save_csv(dsdir/'30_feasibility.csv',[feas])
    save_json(dsdir/'RUN_COMPLETE.json',{
        'dataset':dataset,'status':'COMPLETED','rows_streamed':nrows,'batch_size':args.batch_size,
        'parent_path':str(parent),'parent_mapping_source':'Script17_CONTENT_VALIDATED_PARENT_LINK','no_training':True
    })
    con.close()
    return seq,feas,probe


def main():
    args = parse_args()
    prior = args.prior_provenance.resolve()
    if not prior.exists():
        raise FileNotFoundError(prior)
    decisions, decisions_path = load_parent_decisions(prior)
    run_id = f'run_{utc_id()}_{os.getpid()}'
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    log_path = out/'00_run_log.txt'

    def log(msg):
        line = f'{datetime.now(timezone.utc).isoformat()} {msg}'
        print(line,flush=True)
        with log_path.open('a',encoding='utf-8') as f:
            f.write(line+'\n')

    stages=[]; summaries=[]; feasibilities=[]; probes=[]
    try:
        for dataset in args.datasets:
            parent = validated_parent(decisions,dataset)
            log(f'START {dataset} parent={parent}')
            seq,feas,probe = audit_dataset(dataset,DATASETS[dataset],parent,out,args,log)
            summaries.append(seq); feasibilities.append(feas); probes.append(probe)
            stages.append({'stage':dataset,'status':'COMPLETED','detail':f"{seq['parent_rows']:,} rows streamed"})
            save_csv(out/'01_stage_status.csv',stages)
            save_csv(out/'10_parent_probe_summary.csv',probes)
            save_csv(out/'20_sequence_flow_summary.csv',summaries)
            save_csv(out/'30_feasibility.csv',feasibilities)
            log(f'END {dataset} [COMPLETED]')
        save_json(out/'RUN_STATUS.json',{
            'script_version':'2.0.0-streaming','status':'COMPLETED','prior_provenance':str(prior),
            'parent_decisions_file':str(decisions_path),'stage_errors':0,'scientific_pass':False,
            'no_training':True,'interpretation':'Completion means the streaming audit finished; it does not certify capture/session independence.'
        })
    except Exception as exc:
        with (out/'ERROR.txt').open('w',encoding='utf-8') as f:
            f.write(traceback.format_exc())
        save_json(out/'RUN_STATUS.json',{'script_version':'2.0.0-streaming','status':'ERROR','error':f'{type(exc).__name__}: {exc}','scientific_pass':False,'no_training':True})
        log(f'ERROR {type(exc).__name__}: {exc}')
        raise
    finally:
        checks=[]
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name != 'CHECKSUMS.csv':
                if p.suffix == '.sqlite' or p.name.endswith('.sqlite-wal') or p.name.endswith('.sqlite-shm'):
                    checks.append({'relative_path':str(p.relative_to(out)),'sha256':'','size_bytes':p.stat().st_size,'hash_status':'SKIPPED_SQLITE'})
                else:
                    try:
                        checks.append({'relative_path':str(p.relative_to(out)),'sha256':sha256_file(p),'size_bytes':p.stat().st_size,'hash_status':'SHA256'})
                    except Exception as e:
                        checks.append({'relative_path':str(p.relative_to(out)),'sha256':'','size_bytes':p.stat().st_size,'hash_status':f'ERROR:{type(e).__name__}'})
        save_csv(out/'CHECKSUMS.csv',checks)
        zpath = args.output_root / f'TEMPORAL_AUDIT_FLOW_BOUNDARY_STREAMING_RESULTS_{run_id}.zip'
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob('*')):
                if not p.is_file():
                    continue
                if p.suffix == '.sqlite' or p.name.endswith('.sqlite-wal') or p.name.endswith('.sqlite-shm'):
                    continue
                zf.write(p,arcname=str(Path(run_id)/p.relative_to(out)))
        log(f'REPORT DIRECTORY: {out}')
        log(f'UPLOAD THIS ZIP: {zpath}')

if __name__ == '__main__':
    main()
