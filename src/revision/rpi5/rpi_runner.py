#!/usr/bin/env python3
"""
RPi5 deployment replay runner . Single process.

Phase 1 (no power meter): chronological replay of every exported model under the deployment
  horizons -> FPR/FNR at the frozen threshold.
Phase 2 (power/latency): the FNIRSI logger is started ONCE (FNIRSI_NO_RESET=1) for the whole
  phase, validated within 15 s (abort without retry), then for every model: 20 s idle window +
  >=20 s active batch-256 inference window. Stopped once with SIGINT. Windows are aligned to
  logger samples by Unix timestamp.

Usage: python rpi_runner.py --out DIR [--dry-run] [--python-for-logger /usr/bin/python3]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--dry-run", action="store_true", help="no FNIRSI, 2 configs, short windows")
ap.add_argument("--logger", default=os.environ.get("FNIRSI_LOGGER", "fnirsi_logger.py"))
ap.add_argument("--python-for-logger", default=sys.executable)
ap.add_argument("--idle-s", type=float, default=20.0)
ap.add_argument("--active-s", type=float, default=20.0)
ap.add_argument("--core", type=int, default=3)
args = ap.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    os.sched_setaffinity(0, {args.core})
except Exception as e:
    print("affinity not set:", e)
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

HERE = Path(__file__).resolve().parent
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
H_SECONDS = {"as_recorded": None, "+1h": 3600.0, "+1d": 86400.0, "+7d": 7 * 86400.0,
             "+30d": 30 * 86400.0, "+365d": 365 * 86400.0, "NOW": "now"}
BATCH = 256


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "runner.log", "a") as f:
        f.write(msg + "\n")


def cpu_temp():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return float("nan")


manifest = json.loads((HERE / "manifest.json").read_text())
if args.dry_run:
    manifest = [m for m in manifest if m["dataset"] == "ton_iot" and m["model"] in ("SNN_AE", "CNN_AE")]
replay = {}
for ds in sorted({m["dataset"] for m in manifest}):
    z = np.load(HERE / f"{ds}_replay.npz", allow_pickle=False)
    replay[ds] = {k: z[k] for k in z.files if k != "label_type"}
models = {}
for m in manifest:
    models[m["key"]] = tf.saved_model.load(str(HERE / "models" / m["key"])).signatures["recon"]
log("loaded", len(models), "models; tf", tf.__version__, "; temp", cpu_temp())


def prepare(m, offset):
    r = replay[m["dataset"]]
    featsA = list(r["features_A"])
    raw = r["X_raw_A"].copy()
    ja = featsA.index("frame.time_epoch")
    raw[:, ja] = raw[:, ja] + offset
    cols = [featsA.index(f) for f in m["features"]]
    X = raw[:, cols] * np.asarray(m["scaler_scale"]) + np.asarray(m["scaler_min"])
    return X.astype(np.float32)


def scores(fn, X, qi):
    out = np.empty(len(X))
    for i in range(0, len(X), BATCH):
        xb = X[i:i + BATCH]
        rec = fn(x=tf.constant(xb))
        rec = list(rec.values())[0].numpy()
        out[i:i + BATCH] = np.mean(np.square(xb.astype(np.float64) - rec.astype(np.float64))[:, qi], axis=1)
    return out


# ---------------- Phase 1: accuracy under deployment horizons ----------------
now = time.time()
acc_rows = []
for m in manifest:
    r = replay[m["dataset"]]
    y = r["y"]
    t_first = float(np.min(r["X_raw_A"][:, list(r["features_A"]).index("frame.time_epoch")]))
    fit_end = float(r["fit_epoch_max"])
    for h, hs in H_SECONDS.items():
        if hs is None:
            off = 0.0
        elif hs == "now":
            off = now - t_first
        else:
            off = (fit_end + hs) - t_first
        X = prepare(m, off)
        s = scores(models[m["key"]], X, np.asarray(m["q_idx"]))
        thr = m["threshold"]
        row = {"key": m["key"], "dataset": m["dataset"], "cond": m["cond"], "model": m["model"], "horizon": h,
               "offset_s": off, "fpr": float(np.mean(s[y == 0] > thr)), "fnr": float(np.mean(s[y == 1] <= thr)),
               "n_normal": int(np.sum(y == 0)), "n_attack": int(np.sum(y == 1))}
        acc_rows.append(row)
        log(f"ACC {m['key']:38s} {h:12s} FPR {100*row['fpr']:6.2f}  FNR {100*row['fnr']:6.2f}")
    (OUT / "accuracy_horizons.json").write_text(json.dumps(acc_rows, indent=2))

# ---------------- Phase 2: power + latency, ONE FNIRSI session ----------------
windows = []
logger_proc = None
fn_out = OUT / "fnirsi_raw.txt"
fn_err = OUT / "fnirsi_stderr.txt"
idle_s, active_s = (2.0, 2.0) if args.dry_run else (args.idle_s, args.active_s)


def stop_logger():
    # Built-in fnirsi_logger.py stop: file 'fnirsi_stop' in its cwd (S7p-03 lesson); SIGINT only as fallback.
    global logger_proc
    if logger_proc is not None and logger_proc.poll() is None:
        (OUT / "fnirsi_stop").touch()
        try:
            logger_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            log("logger still alive 20 s after fnirsi_stop; sending SIGINT (drain)")
            logger_proc.send_signal(signal.SIGINT)
            try:
                logger_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log("WARNING logger did not exit; leaving it (no kill, to avoid HID hang)")
    log("FNIRSI session closed; rc =", None if logger_proc is None else logger_proc.poll())


def data_lines():
    try:
        return sum(1 for ln in open(fn_out) if ln.strip() and not ln.startswith("#") and len(ln.split()) >= 4)
    except FileNotFoundError:
        return 0


try:
    if not args.dry_run:
        env = dict(os.environ, FNIRSI_NO_RESET="1")
        if (OUT / "fnirsi_stop").exists():
            (OUT / "fnirsi_stop").unlink()

        def _child():
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.sched_setaffinity(0, {0})
        logger_proc = subprocess.Popen([args.python_for_logger, "-u", args.logger], stdout=open(fn_out, "w"),
                                       stderr=open(fn_err, "w"), env=env, cwd=str(OUT), preexec_fn=_child)
        log("FNIRSI logger started ONCE, pid", logger_proc.pid)
        t0 = time.time()
        while time.time() - t0 < 15 and data_lines() < 10:
            if logger_proc.poll() is not None:
                break
            time.sleep(0.5)
        n = data_lines()
        if n < 10:
            log(f"ABORT: only {n} FNIRSI data lines within 15 s; no retry")
            raise SystemExit(4)
        log("FNIRSI streaming OK:", n, "lines in", round(time.time() - t0, 1), "s")
    for m in manifest:
        if logger_proc is not None and logger_proc.poll() is not None:
            log("ABORT: FNIRSI logger died before", m["key"], "rc", logger_proc.poll())
            raise SystemExit(5)
        X = prepare(m, 0.0)
        qi = np.asarray(m["q_idx"])
        fn = models[m["key"]]
        _ = scores(fn, X[:BATCH * 4], qi)  # warm-up
        ti0 = time.time()
        time.sleep(idle_s)
        ti1 = time.time()
        lat, rows = [], 0
        ta0 = time.time()
        while True:
            for i in range(0, len(X), BATCH):
                xb = X[i:i + BATCH]
                b0 = time.perf_counter()
                rec = list(fn(x=tf.constant(xb)).values())[0].numpy()
                _ = np.mean(np.square(xb - rec)[:, qi], axis=1)
                lat.append((time.perf_counter() - b0, len(xb)))
                rows += len(xb)
                if time.time() - ta0 >= active_s and rows >= len(X):
                    break
            if time.time() - ta0 >= active_s and rows >= len(X):
                break
        ta1 = time.time()
        lt = np.asarray([a for a, _ in lat]); ln = np.asarray([b for _, b in lat])
        w = {"key": m["key"], "dataset": m["dataset"], "cond": m["cond"], "model": m["model"],
             "idle_t0": ti0, "idle_t1": ti1, "active_t0": ta0, "active_t1": ta1, "rows": int(rows),
             "batches": int(len(lt)), "latency_ms_per_row": float(1000 * lt.sum() / ln.sum()),
             "latency_ms_per_batch_p50": float(1000 * np.median(lt)), "latency_ms_per_batch_p95": float(1000 * np.percentile(lt, 95)),
             "throughput_rows_s": float(rows / (ta1 - ta0)), "cpu_temp_c": cpu_temp()}
        windows.append(w)
        (OUT / "power_windows.json").write_text(json.dumps(windows, indent=2))
        log(f"PWR {m['key']:38s} {w['throughput_rows_s']:9.0f} rows/s  {w['latency_ms_per_row']*1000:7.2f} us/row  temp {w['cpu_temp_c']:.1f}")
finally:
    stop_logger()

# ---------------- align power ----------------
if not args.dry_run:
    ts, pw = [], []
    for ln in open(fn_out):
        p = ln.split()
        if ln.startswith("#") or len(p) < 4:
            continue
        try:
            ts.append(float(p[0])); pw.append(float(p[2]) * float(p[3]))
        except ValueError:
            pass
    ts, pw = np.asarray(ts), np.asarray(pw)
    for w in windows:
        mi = (ts >= w["idle_t0"] + 2) & (ts <= w["idle_t1"])
        ma = (ts >= w["active_t0"] + 1) & (ts <= w["active_t1"])
        w.update({"idle_w": float(pw[mi].mean()) if mi.any() else None, "idle_n": int(mi.sum()),
                  "active_w": float(pw[ma].mean()) if ma.any() else None, "active_n": int(ma.sum())})
        if w["active_w"] is not None:
            w["active_mj_per_row"] = 1000 * w["active_w"] / w["throughput_rows_s"]
            w["delta_w"] = w["active_w"] - (w["idle_w"] or np.nan)
            w["delta_mj_per_row"] = 1000 * w["delta_w"] / w["throughput_rows_s"]
    (OUT / "power_windows.json").write_text(json.dumps(windows, indent=2))
    log("power aligned:", len(ts), "FNIRSI samples total")
(OUT / "DONE.json").write_text(json.dumps({"finished": time.time(), "dry_run": args.dry_run, "tf": tf.__version__}))
log("DONE")
