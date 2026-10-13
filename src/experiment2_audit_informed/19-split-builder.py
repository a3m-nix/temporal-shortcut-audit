#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"]).expanduser().resolve()
SOURCE_V2 = BASE / "0-temporal-feature-audit-v2-3datasets"

DATASETS = {
    "edge_iiotset": {
        "sample": SOURCE_V2 / "materialized_samples" / "edge_iiotset_audit_sample.parquet",
        "label_col": "label",
        "normal_label": "Normal",
    },
    "ciciot2023": {
        "sample": SOURCE_V2 / "materialized_samples" / "ciciot2023_audit_sample.parquet",
        "label_col": "label",
        "normal_label": "Benign",
    },
    "ton_iot": {
        "sample": SOURCE_V2 / "materialized_samples" / "ton_iot_audit_sample.parquet",
        "label_col": "label",
        "normal_label": "Normal",
    },
}

def now_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

def sha256_file(path, block=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def save_json(path,obj):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--flow-audit",type=Path,required=True)
    ap.add_argument("--mode",choices=["auto","flow","blocked-time","capture","session"],default="auto")
    ap.add_argument("--output-root",type=Path,default=BASE/"temporal_audit_upgrade_splits")
    ap.add_argument("--seed",type=int,default=20260914)
    ap.add_argument("--test-fraction",type=float,default=0.20)
    ap.add_argument("--earlystop-fraction-of-development-normal",type=float,default=0.15)
    ap.add_argument("--calibration-fraction-of-development-normal",type=float,default=0.15)
    ap.add_argument("--n-time-blocks",type=int,default=5)
    ap.add_argument("--blocked-test-block",type=int,default=-1)
    return ap.parse_args()

def find_csv(root,name):
    p=root/name
    if p.exists():
        return p
    hits=list(root.rglob(name))
    if len(hits)==1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    raise RuntimeError(f"Multiple {name} files found under {root}")

def load_audit(root):
    return (
        pd.read_csv(find_csv(root,"30_feasibility.csv")),
        pd.read_csv(find_csv(root,"11_parent_decisions.csv")),
    )

def parent_for(parents,ds):
    r=parents.loc[parents["dataset"]==ds]
    if len(r)!=1:
        raise RuntimeError(f"{ds}: parent decision count={len(r)}")
    p=Path(r.iloc[0]["selected_parent"])
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def read_parent(path,cfg):
    names=set(pq.ParquetFile(path).schema_arrow.names)
    wanted=[
        "frame.time_epoch","flow_id","packet_index",
        cfg["label_col"],"label_type",
        "ip.src","ip.dst","ip.proto",
        "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport",
    ]
    cols=[c for c in wanted if c in names]
    if "frame.time_epoch" not in cols or cfg["label_col"] not in cols:
        raise RuntimeError(f"{path}: missing timestamp/label")
    df=pd.read_parquet(path,columns=cols)
    df.insert(0,"parent_row_position",np.arange(len(df),dtype=np.int64))
    df["y"]=(df[cfg["label_col"]].astype(str)!=cfg["normal_label"]).astype(np.int8)
    return df

def group_summary(df,col):
    return df.groupby(col,dropna=False).agg(
        n=("parent_row_position","size"),
        normal_n=("y",lambda s:int(np.sum(np.asarray(s)==0))),
        attack_n=("y",lambda s:int(np.sum(np.asarray(s)==1))),
        first_epoch=("frame.time_epoch","min"),
        last_epoch=("frame.time_epoch","max"),
    ).reset_index()

def make_flow_split(ds,df,seed,test_fraction,early_frac,calib_frac):
    if "flow_id" not in df:
        raise RuntimeError(f"{ds}: flow_id unavailable")
    if df["flow_id"].isna().any():
        raise RuntimeError(f"{ds}: missing flow_id; fail-closed")

    gs=group_summary(df,"flow_id")
    shuffled=gs.sample(frac=1.0,random_state=seed).reset_index(drop=True)
    target=max(1,int(round(len(df)*test_fraction)))
    test_flows=[]; n=0
    for _,r in shuffled.iterrows():
        test_flows.append(r["flow_id"])
        n+=int(r["n"])
        if n>=target and len(test_flows)<len(gs):
            break

    test=df[df["flow_id"].isin(test_flows)].copy()
    dev=df[~df["flow_id"].isin(test_flows)].copy()

    for yy in [0,1]:
        if not np.any(test["y"].to_numpy()==yy):
            cand=group_summary(dev,"flow_id")
            col="normal_n" if yy==0 else "attack_n"
            cand=cand[cand[col]>0].sort_values("n")
            if cand.empty:
                raise RuntimeError(f"{ds}: cannot get class {yy} into flow-disjoint test")
            f=cand.iloc[0]["flow_id"]
            add=dev["flow_id"]==f
            test=pd.concat([test,dev[add]],axis=0)
            dev=dev[~add]

    dev_normal=dev[dev["y"]==0].copy()
    ng=group_summary(dev_normal,"flow_id")
    flow_ids=list(np.random.RandomState(seed+101).permutation(ng["flow_id"].tolist()))
    total=int(ng["n"].sum())
    target_early=max(1,int(round(total*early_frac)))
    target_calib=max(1,int(round(total*calib_frac)))

    def allocate(candidates,target,used):
        chosen=[]; count=0
        for f in candidates:
            if f in used:
                continue
            nf=int(ng.loc[ng["flow_id"]==f,"n"].iloc[0])
            chosen.append(f); count+=nf
            if count>=target:
                break
        return chosen

    early_flows=allocate(flow_ids,target_early,set())
    calib_flows=allocate(flow_ids,target_calib,set(early_flows))
    used=set(early_flows)|set(calib_flows)
    fit_flows=[f for f in flow_ids if f not in used]
    if not fit_flows or not early_flows or not calib_flows:
        raise RuntimeError(f"{ds}: unable to allocate normal flows")

    fit=dev_normal[dev_normal["flow_id"].isin(fit_flows)].copy()
    early=dev_normal[dev_normal["flow_id"].isin(early_flows)].copy()
    calib=dev_normal[dev_normal["flow_id"].isin(calib_flows)].copy()
    return fit,early,calib,test,{
        "split_type":"FLOW_DISJOINT_CANDIDATE",
        "group_col":"flow_id",
        "test_group_n":int(test["flow_id"].nunique()),
        "fit_group_n":int(fit["flow_id"].nunique()),
        "earlystop_group_n":int(early["flow_id"].nunique()),
        "calibration_group_n":int(calib["flow_id"].nunique()),
    }

def assign_time_blocks(df,n_blocks):
    if n_blocks<3:
        raise ValueError("n_time_blocks must be >= 3")
    ordered=df.sort_values(["frame.time_epoch","parent_row_position"]).copy()
    rank=np.arange(len(ordered))
    ordered["time_block"]=np.minimum((rank*n_blocks)//len(ordered),n_blocks-1).astype(int)
    return ordered

def make_blocked_split(ds,df,n_blocks,test_block,early_frac,calib_frac):
    ordered=assign_time_blocks(df,n_blocks)
    if test_block<0:
        test_block=n_blocks-1
    if not 0<=test_block<n_blocks:
        raise ValueError("blocked_test_block out of range")
    test=ordered[ordered["time_block"]==test_block].copy()
    dev=ordered[ordered["time_block"]!=test_block].copy()
    if not np.any(test["y"]==0) or not np.any(test["y"]==1):
        raise RuntimeError(f"{ds}: selected time block lacks normal or attack traffic")
    devn=dev[dev["y"]==0].sort_values(["frame.time_epoch","parent_row_position"]).copy()
    n=len(devn)
    ne=max(1,int(round(n*early_frac)))
    nc=max(1,int(round(n*calib_frac)))
    nf=n-ne-nc
    if nf<=0:
        raise RuntimeError(f"{ds}: insufficient development normal rows")
    fit=devn.iloc[:nf].copy()
    early=devn.iloc[nf:nf+ne].copy()
    calib=devn.iloc[nf+ne:].copy()
    return fit,early,calib,test,{
        "split_type":"BLOCKED_TIME_CANDIDATE",
        "group_col":"time_block",
        "test_block":int(test_block),
        "n_time_blocks":int(n_blocks),
    }

def overlap(a,b,col):
    if col not in a or col not in b:
        return np.nan
    return len(set(a[col].dropna().astype(str)) & set(b[col].dropna().astype(str)))

def role_summary(ds,role,df,stype):
    return {
        "dataset":ds,"split_type":stype,"role":role,
        "n":int(len(df)),
        "normal_n":int(np.sum(df["y"].to_numpy()==0)),
        "attack_n":int(np.sum(df["y"].to_numpy()==1)),
        "epoch_min":float(df["frame.time_epoch"].min()) if len(df) else np.nan,
        "epoch_max":float(df["frame.time_epoch"].max()) if len(df) else np.nan,
        "flow_n":int(df["flow_id"].nunique()) if "flow_id" in df else np.nan,
    }

def manifest(ds,role,df,stype):
    cols=[c for c in [
        "parent_row_position","frame.time_epoch","flow_id","packet_index",
        "label","label_type","y","time_block",
        "ip.src","ip.dst","ip.proto",
        "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"
    ] if c in df]
    x=df[cols].copy()
    x.insert(0,"dataset",ds)
    x.insert(1,"role",role)
    x.insert(2,"split_type",stype)
    return x

def main():
    args=parse_args()
    audit_root=args.flow_audit.resolve()
    if not audit_root.exists():
        raise FileNotFoundError(audit_root)
    feas,parents=load_audit(audit_root)

    run_id=f"run_{now_id()}_{os.getpid()}"
    out=args.output_root/run_id
    out.mkdir(parents=True,exist_ok=False)
    logf=out/"00_run_log.txt"
    def log(msg):
        line=f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line,flush=True)
        with open(logf,"a",encoding="utf-8") as f:
            f.write(line+"\n")

    summaries=[]; overlaps=[]; protocols=[]; manifests=[]; findings=[]
    try:
        for i,(ds,cfg) in enumerate(DATASETS.items()):
            fr=feas.loc[feas["dataset"]==ds]
            if len(fr)!=1:
                raise RuntimeError(f"{ds}: feasibility count={len(fr)}")
            fr=fr.iloc[0]
            mode=args.mode
            if mode=="auto":
                mode="flow" if bool(fr["flow_grouping_available"]) else "blocked-time"
            if mode=="capture" and not bool(fr["capture_disjoint_ready"]):
                raise RuntimeError(f"{ds}: capture-disjoint not validated by Script 18")
            if mode=="session" and not bool(fr["session_disjoint_ready"]):
                raise RuntimeError(f"{ds}: session-disjoint not validated by Script 18")
            if mode=="flow" and not bool(fr["flow_grouping_available"]):
                raise RuntimeError(f"{ds}: flow grouping unavailable")
            if mode in ("capture","session"):
                raise RuntimeError(f"{ds}: row-level {mode} ID not yet implemented because evidence is not validated")

            parent=parent_for(parents,ds)
            log(f"READ {ds} {parent}")
            df=read_parent(parent,cfg)
            seed=args.seed+i*1000

            if mode=="flow":
                fit,early,calib,test,meta=make_flow_split(
                    ds,df,seed,args.test_fraction,
                    args.earlystop_fraction_of_development_normal,
                    args.calibration_fraction_of_development_normal
                )
            else:
                fit,early,calib,test,meta=make_blocked_split(
                    ds,df,args.n_time_blocks,args.blocked_test_block,
                    args.earlystop_fraction_of_development_normal,
                    args.calibration_fraction_of_development_normal
                )

            stype=meta["split_type"]
            roles={"fit_normal":fit,"earlystop_normal":early,"calibration_normal":calib,"test":test}
            for name in ["fit_normal","earlystop_normal","calibration_normal"]:
                if np.any(roles[name]["y"].to_numpy()!=0):
                    raise RuntimeError(f"{ds}: {name} contains attack rows")
            if not np.any(test["y"]==0) or not np.any(test["y"]==1):
                raise RuntimeError(f"{ds}: test lacks one class")

            names=list(roles)
            for a in range(len(names)):
                for b in range(a+1,len(names)):
                    A=roles[names[a]]; B=roles[names[b]]
                    ro=len(set(A["parent_row_position"]) & set(B["parent_row_position"]))
                    fo=overlap(A,B,"flow_id")
                    overlaps.append({
                        "dataset":ds,"split_type":stype,
                        "role_a":names[a],"role_b":names[b],
                        "row_overlap_n":ro,"flow_overlap_n":fo
                    })
                    if ro!=0:
                        raise RuntimeError(f"{ds}: row leakage {names[a]} vs {names[b]}")
                    if stype=="FLOW_DISJOINT_CANDIDATE" and fo!=0:
                        raise RuntimeError(f"{ds}: flow leakage {names[a]} vs {names[b]}")

            for role,frame in roles.items():
                summaries.append(role_summary(ds,role,frame,stype))
                manifests.append(manifest(ds,role,frame,stype))

            protocols.append({
                "dataset":ds,"selected_mode":mode,"seed":seed,
                "parent_path":str(parent),"parent_rows":len(df),**meta
            })
            findings.append({
                "severity":"REVIEW","dataset":ds,
                "finding":"candidate_split_not_final_protocol",
                "detail":(
                    "Flow-disjoint does not imply capture/session independence."
                    if stype=="FLOW_DISJOINT_CANDIDATE"
                    else "Blocked-time does not imply capture/session independence."
                )
            })
            log(f"DONE {ds} {stype}")

        mf=pd.concat(manifests,ignore_index=True)
        mf.to_csv(out/"10_split_manifest.csv.gz",index=False,compression="gzip")
        pd.DataFrame(summaries).to_csv(out/"11_split_summary.csv",index=False)
        pd.DataFrame(overlaps).to_csv(out/"12_overlap_checks.csv",index=False)
        pd.DataFrame(protocols).to_csv(out/"13_protocol_candidates.csv",index=False)
        pd.DataFrame(findings).to_csv(out/"14_findings.csv",index=False)

        for ds in DATASETS:
            d=mf[mf["dataset"]==ds]
            arrays={}
            for role in ["fit_normal","earlystop_normal","calibration_normal","test"]:
                arrays[role+"_parent_row_position"]=d.loc[d["role"]==role,"parent_row_position"].to_numpy(dtype=np.int64)
            np.savez_compressed(out/f"{ds}_split_indices.npz",**arrays)

        save_json(out/"RUN_STATUS.json",{
            "status":"COMPLETED","scientific_pass":False,
            "source_flow_audit":str(audit_root),
            "note":"Split candidates built under fail-closed naming rules; review before training."
        })
    except Exception as exc:
        with open(out/"ERROR.txt","w",encoding="utf-8") as f:
            f.write(traceback.format_exc())
        save_json(out/"RUN_STATUS.json",{"status":"ERROR","error":f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        checks=[]
        for fp in sorted(out.rglob("*")):
            if fp.is_file() and fp.name!="CHECKSUMS.csv":
                checks.append({"relative_path":str(fp.relative_to(out)),"sha256":sha256_file(fp),"size_bytes":fp.stat().st_size})
        pd.DataFrame(checks).to_csv(out/"CHECKSUMS.csv",index=False)
        zpath=args.output_root/f"TEMPORAL_AUDIT_SPLIT_RESULTS_{run_id}.zip"
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(out.rglob("*")):
                if fp.is_file():
                    zf.write(fp,arcname=str(Path(run_id)/fp.relative_to(out)))
        log(f"REPORT DIRECTORY: {out}")
        log(f"UPLOAD THIS ZIP: {zpath}")

if __name__=="__main__":
    main()
