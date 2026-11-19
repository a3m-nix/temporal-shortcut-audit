#!/usr/bin/env python3
"""
36 - Extract Edge-IIoTset attack PCAPs to the packet-level 34-feature schema of the parent table
(revision, R5.2 extension). Same tshark field list as dataset-raw/edge_iiotset_extract_pcap_to_csv.sh;
inter_arrival_time = frame.time_epoch.diff() in capture order, first packet 0 (verified to match the parent table).

Usage: python 36-extract-edge-nonflood-v1.py [--validate-only]
"""
import os
import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
PCAP_DIR = BASE / "dataset-csv/edge_iiotset/attack_traffic"
PARENT = BASE / "dataset-from-pcap/edge-iiotset_bidir.parquet"
OUT = BASE / "temporal_audit_attack_extension_v1"
FIELDS = ["frame.len", "frame.time_epoch", "ip.version", "ip.hdr_len", "ip.dsfield.dscp", "ip.len", "ip.id",
          "ip.frag_offset", "ip.flags.df", "ip.flags.mf", "ip.ttl", "ip.proto", "ip.src", "ip.dst",
          "tcp.srcport", "tcp.dstport", "tcp.seq", "tcp.ack", "tcp.hdr_len", "tcp.len", "tcp.window_size",
          "tcp.flags.fin", "tcp.flags.syn", "tcp.flags.reset", "tcp.flags.push", "tcp.flags.ack", "tcp.flags.urg",
          "tcp.flags.ns", "tcp.flags.cwr", "udp.srcport", "udp.dstport", "udp.length", "icmp.type", "icmp.code", "data.len"]
FEATS = [f for f in FIELDS if f not in ("ip.src", "ip.dst")] + ["inter_arrival_time"]
NONFLOOD = {"backdoor_attack.pcap": "backdoor", "mitm_arp_spoofing_dns_attack.pcap": "mitm",
            "os_fingerprinting_attack.pcap": "os_fingerprinting", "password_attacks.pcap": "password",
            "port_scanning_attack.pcap": "port_scanning", "ransomware_attack.pcap": "ransomware",
            "sql_injection_attack.pcap": "sql_injection", "uploading_attack.pcap": "uploading",
            "vulnerability_scanner_attack.pcap": "vulnerability_scanner", "xss_attacks.pcap": "xss"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_num(s):
    s = s.astype(str).str.strip().str.split(",").str[0]
    hexm = s.str.startswith("0x")
    s2 = s.replace({"True": "1", "False": "0", "true": "1", "false": "0", "": None, "nan": None})
    num = pd.to_numeric(s2, errors="coerce")
    if hexm.any():
        num[hexm] = s[hexm].apply(lambda v: int(v, 16))
    return num


def extract(pcap):
    cmd = ["tshark", "-r", str(pcap), "-n", "-N", "m", "-T", "fields"]
    for f in FIELDS:
        cmd += ["-e", f]
    cmd += ["-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"]
    raw = OUT / "raw_csv" / (pcap.stem + ".csv")
    raw.parent.mkdir(parents=True, exist_ok=True)
    with open(raw, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
    df = pd.read_csv(raw, dtype=str, keep_default_na=False)
    out = pd.DataFrame({f: to_num(df[f]) for f in FIELDS if f not in ("ip.src", "ip.dst")})
    out["frame.time_epoch"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce").astype(np.float64)
    out["inter_arrival_time"] = out["frame.time_epoch"].diff().fillna(0.0)
    return out, r.returncode, r.stderr[-500:]


def validate():
    pcap = PCAP_DIR / "ddos_http_flood_attacks.pcap"
    mine, rc, err = extract(pcap)
    par = pd.read_parquet(PARENT, columns=FEATS + ["label_type"], filters=[("label_type", "==", "ddos-http-flood-attacks")])
    par = par.drop(columns="label_type").reset_index(drop=True)
    res = {"pcap_rows": len(mine), "parent_rows": len(par), "tshark_rc": rc}
    if len(mine) == len(par):
        a, b = mine[FEATS].fillna(0).to_numpy(np.float64), par[FEATS].fillna(0).to_numpy(np.float64)
        eq = np.isclose(a, b, rtol=0, atol=1e-6)
        res["all_equal"] = bool(eq.all())
        res["per_feature_match"] = {f: float(eq[:, i].mean()) for i, f in enumerate(FEATS)}
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUT / "metadata" / "edge_extraction_validation.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v = validate()
    print("VALIDATION", {k: v[k] for k in v if k != "per_feature_match"})
    if v.get("per_feature_match"):
        bad = {k: round(x, 4) for k, x in v["per_feature_match"].items() if x < 1}
        print("features not 100% equal:", bad or "none")
    if args.validate_only or not v.get("all_equal"):
        return

    def job(item):
        name, fam = item
        df, rc, err = extract(PCAP_DIR / name)
        df["label"] = "Attack"
        df["label_type"] = fam
        df["source_file"] = name
        df["packet_order"] = np.arange(len(df))
        return fam, df, rc, err, sha(PCAP_DIR / name)

    parts, inv = [], []
    with ThreadPoolExecutor(6) as ex:
        for fam, df, rc, err, h in ex.map(job, NONFLOOD.items()):
            parts.append(df)
            inv.append({"family": fam, "rows": len(df), "tshark_rc": rc, "pcap_sha256": h,
                        "epoch_min": float(df["frame.time_epoch"].min()), "epoch_max": float(df["frame.time_epoch"].max()),
                        "stderr_tail": err.strip()[-200:]})
            print(fam, len(df), "rc", rc, flush=True)
    allp = pd.concat(parts, ignore_index=True)
    allp.to_parquet(OUT / "edge_nonflood_packets.parquet", index=False)
    pd.DataFrame(inv).to_csv(OUT / "metadata" / "edge_nonflood_inventory.csv", index=False)
    print("WROTE", len(allp))


if __name__ == "__main__":
    main()
