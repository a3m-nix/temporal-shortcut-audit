from __future__ import annotations

import gc
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Cropping1D,
    Dense,
    Flatten,
    Input,
    Layer,
    LSTM,
    MaxPooling1D,
    RepeatVector,
    Reshape,
    TimeDistributed,
    UpSampling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(
    os.environ["TEMPORAL_AUDIT_ROOT"] + "/"
    "0-temporal-feature-audit"
)
ROOT.mkdir(parents=True, exist_ok=True)

DIRS = {
    "samples": ROOT / "materialized_samples",
    "splits": ROOT / "splits",
    "runs": ROOT / "runs",
    "tables": ROOT / "tables",
    "figures": ROOT / "figures",
    "metadata": ROOT / "metadata",
    "models": ROOT / "models",
    "scalers": ROOT / "scalers",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "edge_iiotset": {
        "path": (
            os.environ["TEMPORAL_AUDIT_ROOT"] + "/"
            "dataset-from-pcap/edge-iiotset_bidir.parquet"
        ),
        "label": "label",
        "normal": "Normal",
        "scaler": "quantile",
        # final LIF configuration from dissertation experiment
        "snn": {
            "threshold": 0.5480,
            "decay": 0.7011,
            "surrogate_scale": 13.4029,
        },
    },
    "ciciot2023": {
        "path": (
            os.environ["TEMPORAL_AUDIT_ROOT"] + "/"
            "dataset-from-pcap/ciciot2023_bidir.parquet"
        ),
        "label": "label",
        "normal": "Benign",
        "scaler": "minmax",
        "snn": {
            "threshold": 0.3584,
            "decay": 0.8315,
            "surrogate_scale": 6.9200,
        },
    },
}

FINAL_33 = [
    "frame.len",
    "ip.version",
    "ip.hdr_len",
    "ip.dsfield.dscp",
    "ip.len",
    "ip.id",
    "ip.frag_offset",
    "ip.flags.df",
    "ip.flags.mf",
    "ip.ttl",
    "ip.proto",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.seq",
    "tcp.ack",
    "tcp.hdr_len",
    "tcp.len",
    "tcp.window_size",
    "tcp.flags.fin",
    "tcp.flags.syn",
    "tcp.flags.reset",
    "tcp.flags.push",
    "tcp.flags.ack",
    "tcp.flags.urg",
    "tcp.flags.ns",
    "tcp.flags.cwr",
    "udp.srcport",
    "udp.dstport",
    "udp.length",
    "icmp.type",
    "icmp.code",
    "data.len",
    "inter_arrival_time",
]

# Historical representation: same 33 features + absolute timestamp.
FEATURES_34 = [
    "frame.len",
    "frame.time_epoch",
    *FINAL_33[1:],
]

assert len(FINAL_33) == 33
assert len(FEATURES_34) == 34
assert "frame.time_epoch" not in FINAL_33
assert "frame.time_epoch" in FEATURES_34

CONDITIONS = {
    "with_epoch_34F": FEATURES_34,
    "without_epoch_33F": FINAL_33,
}

MODELS = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]

ATTACK_PER_SUBTYPE = 2000

# Normal-only partitions relative to total sampled attack A:
# train = 0.6 A, validation = 0.2 A, test normal = 1.0 A
# Therefore the final test is balanced normal vs attack.
TRAIN_NORMAL_RATIO = 0.60
VAL_NORMAL_RATIO = 0.20
TEST_NORMAL_RATIO = 1.00

SEED = 42

# This is an audit protocol, not a replacement for the official benchmark.
# Same optimization settings are deliberately retained between 34F and 33F.
LEARNING_RATE = 5e-4
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
LATENT_DIM = 32
TIME_STEPS = 3
OPERATING_PERCENTILE = 99
SPIKE_SAMPLE_MAX = 10000

SAVE_MODELS = True


# =============================================================================
# UTILITIES
# =============================================================================

def json_safe(x):
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
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_scaler(dataset_name: str, n_train: int):
    stype = DATASETS[dataset_name]["scaler"]

    if stype == "quantile":
        return QuantileTransformer(
            n_quantiles=min(1000, n_train),
            output_distribution="normal",
            random_state=SEED,
            subsample=200000,
        )

    if stype == "minmax":
        return MinMaxScaler()

    raise ValueError(stype)


def metric_dict(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    pred = (scores > threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = np.nan

    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "auc": float(auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =============================================================================
# MEMORY-BOUNDED SAMPLE MATERIALIZATION
# =============================================================================

def sample_path(dataset_name):
    return DIRS["samples"] / f"{dataset_name}_audit_sample.parquet"


def split_path(dataset_name):
    return DIRS["splits"] / f"{dataset_name}_fixed_split.npz"


def scan_and_sample_source_rows(dataset_name):
    cfg = DATASETS[dataset_name]
    source = Path(cfg["path"])

    pf = pq.ParquetFile(source)

    normal_parts = []
    attack_parts = defaultdict(list)

    offset = 0

    print(f"[{dataset_name}] Pass 1/2: scanning label indices...")

    for rg in range(pf.num_row_groups):
        columns = [cfg["label"], "label_type"]
        table = pf.read_row_group(
            rg,
            columns=columns,
            use_threads=True,
        )
        df = table.to_pandas()

        labels = df[cfg["label"]].astype(str).to_numpy()
        types = df["label_type"].astype(str).to_numpy()

        local = np.arange(len(df), dtype=np.int64)
        normal_mask = labels == cfg["normal"]

        if np.any(normal_mask):
            normal_parts.append(local[normal_mask] + offset)

        attack_local = local[~normal_mask]
        attack_types = types[~normal_mask]

        for subtype in np.unique(attack_types):
            mask = attack_types == subtype
            attack_parts[str(subtype)].append(
                attack_local[mask] + offset
            )

        offset += len(df)

        del table, df, labels, types, local
        gc.collect()

    normal_all = np.concatenate(normal_parts).astype(np.int64)

    rng = np.random.RandomState(SEED)

    attack_selected = {}
    attack_rows = []

    for subtype in sorted(attack_parts):
        available = np.concatenate(
            attack_parts[subtype]
        ).astype(np.int64)

        take = min(ATTACK_PER_SUBTYPE, len(available))
        idx = rng.choice(
            len(available),
            size=take,
            replace=False,
        )
        selected = available[idx]

        attack_selected[subtype] = selected
        attack_rows.append(selected)

    attack_all = np.concatenate(attack_rows)
    attack_total = len(attack_all)

    n_train_normal = int(round(
        TRAIN_NORMAL_RATIO * attack_total
    ))
    n_val_normal = int(round(
        VAL_NORMAL_RATIO * attack_total
    ))
    n_test_normal = int(round(
        TEST_NORMAL_RATIO * attack_total
    ))

    normal_needed = (
        n_train_normal
        + n_val_normal
        + n_test_normal
    )

    if len(normal_all) < normal_needed:
        raise RuntimeError(
            f"{dataset_name}: normal traffic insufficient. "
            f"Need {normal_needed}, available {len(normal_all)}."
        )

    normal_idx = rng.choice(
        len(normal_all),
        size=normal_needed,
        replace=False,
    )
    normal_selected = normal_all[normal_idx]

    summary_rows = []
    for subtype, rows in attack_selected.items():
        summary_rows.append({
            "dataset": dataset_name,
            "subtype": subtype,
            "sampled_attack": len(rows),
        })

    pd.DataFrame(summary_rows).to_csv(
        DIRS["metadata"]
        / f"{dataset_name}_attack_sampling.csv",
        index=False,
    )

    return (
        normal_selected,
        attack_selected,
        {
            "source_rows": int(offset),
            "attack_total": int(attack_total),
            "normal_selected": int(normal_needed),
            "train_normal": int(n_train_normal),
            "val_normal": int(n_val_normal),
            "test_normal": int(n_test_normal),
            "subtype_count": len(attack_selected),
        },
    )


def materialize_sample(dataset_name):
    out = sample_path(dataset_name)

    if out.exists() and split_path(dataset_name).exists():
        print(f"[{dataset_name}] Materialized sample exists; resume.")
        return pd.read_parquet(out)

    cfg = DATASETS[dataset_name]

    normal_rows, attack_selected, summary = (
        scan_and_sample_source_rows(dataset_name)
    )

    attack_lookup = {}
    for subtype, rows in attack_selected.items():
        for row in rows:
            attack_lookup[int(row)] = subtype

    selected_rows = np.concatenate([
        normal_rows,
        np.asarray(list(attack_lookup.keys()), dtype=np.int64),
    ])
    selected_rows = np.unique(selected_rows)

    selected_sorted = np.sort(selected_rows)

    pf = pq.ParquetFile(cfg["path"])

    read_columns = list(dict.fromkeys(
        FEATURES_34
        + [cfg["label"], "label_type"]
    ))

    pieces = []
    offset = 0

    print(
        f"[{dataset_name}] Pass 2/2: materializing "
        f"{len(selected_sorted):,} selected rows..."
    )

    for rg in range(pf.num_row_groups):
        n_rows = pf.metadata.row_group(rg).num_rows

        start = offset
        stop = offset + n_rows

        left = np.searchsorted(
            selected_sorted, start, side="left"
        )
        right = np.searchsorted(
            selected_sorted, stop, side="left"
        )

        if right > left:
            global_rows = selected_sorted[left:right]
            local_rows = global_rows - start

            table = pf.read_row_group(
                rg,
                columns=read_columns,
                use_threads=True,
            )
            df = table.to_pandas()

            part = df.iloc[local_rows].copy()
            part["source_row_index"] = global_rows
            pieces.append(part)

            del table, df, part
            gc.collect()

        offset = stop

    sample = pd.concat(
        pieces,
        axis=0,
        ignore_index=True,
    )

    # Numeric sanitization.
    for col in FEATURES_34:
        sample[col] = pd.to_numeric(
            sample[col], errors="coerce"
        )

    sample[FEATURES_34] = (
        sample[FEATURES_34]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    sample.to_parquet(out, index=False)

    # -------------------------------------------------------------------------
    # Fixed split: ALL sampled attacks go to test.
    # Test normal count equals total attack count.
    # -------------------------------------------------------------------------
    y = (
        sample[cfg["label"]].astype(str)
        != cfg["normal"]
    ).astype(np.int8).to_numpy()

    normal_local = np.where(y == 0)[0]
    attack_local = np.where(y == 1)[0]

    rng = np.random.RandomState(SEED)
    rng.shuffle(normal_local)
    rng.shuffle(attack_local)

    A = len(attack_local)

    n_train = int(round(TRAIN_NORMAL_RATIO * A))
    n_val = int(round(VAL_NORMAL_RATIO * A))
    n_test_normal = int(round(TEST_NORMAL_RATIO * A))

    train_idx = normal_local[:n_train]
    val_idx = normal_local[n_train:n_train+n_val]
    test_normal_idx = normal_local[
        n_train+n_val:
        n_train+n_val+n_test_normal
    ]

    if len(test_normal_idx) != A:
        raise RuntimeError(
            f"{dataset_name}: test normal is not balanced "
            f"with attacks ({len(test_normal_idx)} vs {A})."
        )

    test_idx = np.concatenate([
        test_normal_idx,
        attack_local,
    ])
    rng.shuffle(test_idx)

    np.savez_compressed(
        split_path(dataset_name),
        train_normal_idx=train_idx.astype(np.int64),
        val_normal_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        test_normal_idx=test_normal_idx.astype(np.int64),
        test_attack_idx=attack_local.astype(np.int64),
    )

    save_json(
        DIRS["metadata"]
        / f"{dataset_name}_sample_summary.json",
        {
            **summary,
            "materialized_rows": len(sample),
            "train_normal": len(train_idx),
            "val_normal": len(val_idx),
            "test_normal": len(test_normal_idx),
            "test_attack": len(attack_local),
            "test_total": len(test_idx),
            "test_attack_ratio": (
                len(attack_local) / len(test_idx)
            ),
        },
    )

    return sample


# =============================================================================
# SNN
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="Audit")
class SpikingLayer(Layer):
    def __init__(
        self,
        units,
        threshold=0.5,
        decay=0.7,
        surrogate_scale=10.0,
        time_steps=3,
        return_sequences=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.surrogate_scale = float(surrogate_scale)
        self.time_steps = int(time_steps)
        self.return_sequences = bool(return_sequences)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            name="weights",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    @tf.custom_gradient
    def spike_function(self, membrane, threshold):
        spikes = tf.cast(
            tf.greater(membrane, threshold),
            tf.float32,
        )

        def grad(dy):
            diff = membrane - threshold
            surrogate = (
                self.surrogate_scale
                / tf.square(
                    self.surrogate_scale * tf.abs(diff) + 1.0
                )
            )
            return dy * surrogate, None

        return spikes, grad

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        membrane = tf.zeros(
            (batch_size, self.units),
            dtype=inputs.dtype,
        )

        spikes = []
        is_sequence = len(inputs.shape) == 3

        for t in range(self.time_steps):
            current_input = (
                inputs[:, t, :]
                if is_sequence
                else inputs
            )

            current = (
                tf.matmul(current_input, self.w)
                + self.b
            )

            current = current / tf.sqrt(
                tf.cast(
                    tf.shape(current_input)[-1],
                    tf.float32,
                )
            )

            membrane = (
                self.decay * membrane
                + current
            )

            spike = self.spike_function(
                membrane,
                self.threshold,
            )

            membrane = (
                membrane
                - spike * self.threshold
            )

            spikes.append(spike)

        stacked = tf.stack(spikes, axis=1)

        if self.return_sequences:
            return stacked

        return tf.reduce_mean(stacked, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "threshold": self.threshold,
            "decay": self.decay,
            "surrogate_scale": self.surrogate_scale,
            "time_steps": self.time_steps,
            "return_sequences": self.return_sequences,
        })
        return config


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_snn_ae(input_dim, snn_cfg):
    inp = Input(shape=(input_dim,), name="input")

    x = BatchNormalization(
        name="batch_norm"
    )(inp)

    x = SpikingLayer(
        LATENT_DIM,
        threshold=snn_cfg["threshold"],
        decay=snn_cfg["decay"],
        surrogate_scale=snn_cfg["surrogate_scale"],
        time_steps=TIME_STEPS,
        return_sequences=True,
        name="encoder",
    )(x)

    x = SpikingLayer(
        LATENT_DIM,
        threshold=snn_cfg["threshold"],
        decay=snn_cfg["decay"],
        surrogate_scale=snn_cfg["surrogate_scale"],
        time_steps=TIME_STEPS,
        return_sequences=False,
        name="decoder",
    )(x)

    out = Dense(
        input_dim,
        activation="linear",
        name="output",
    )(x)

    return Model(inp, out, name="SNN_AE")


def build_dense_ae(input_dim):
    inp = Input(shape=(input_dim,), name="input")

    x = BatchNormalization(
        name="batch_norm"
    )(inp)

    x = Dense(
        LATENT_DIM,
        activation="relu",
        name="encoder",
    )(x)

    x = Dense(
        LATENT_DIM,
        activation="relu",
        name="decoder",
    )(x)

    out = Dense(
        input_dim,
        activation="linear",
        name="output",
    )(x)

    return Model(inp, out, name="Dense_AE")


def build_lstm_ae(input_dim):
    inp = Input(
        shape=(1, input_dim),
        name="input",
    )

    x = LSTM(
        64,
        activation="relu",
        return_sequences=True,
        name="enc_lstm1",
    )(inp)

    x = LSTM(
        LATENT_DIM,
        activation="relu",
        return_sequences=False,
        name="enc_lstm2",
    )(x)

    x = RepeatVector(
        1,
        name="repeat",
    )(x)

    x = LSTM(
        LATENT_DIM,
        activation="relu",
        return_sequences=True,
        name="dec_lstm1",
    )(x)

    x = LSTM(
        64,
        activation="relu",
        return_sequences=True,
        name="dec_lstm2",
    )(x)

    out = TimeDistributed(
        Dense(input_dim),
        name="output",
    )(x)

    return Model(inp, out, name="LSTM_AE")


def build_cnn_ae(input_dim):
    pooled1 = (input_dim + 1) // 2
    pooled2 = (pooled1 + 1) // 2
    upsampled = pooled2 * 4
    crop_right = upsampled - input_dim

    inp = Input(
        shape=(input_dim, 1),
        name="input",
    )

    x = Conv1D(
        32, 3,
        activation="relu",
        padding="same",
        name="conv1",
    )(inp)

    x = MaxPooling1D(
        2,
        padding="same",
        name="pool1",
    )(x)

    x = Conv1D(
        16, 3,
        activation="relu",
        padding="same",
        name="conv2",
    )(x)

    x = MaxPooling1D(
        2,
        padding="same",
        name="pool2",
    )(x)

    x = Flatten(name="flatten")(x)

    x = Dense(
        LATENT_DIM,
        activation="relu",
        name="latent",
    )(x)

    x = Dense(
        16 * pooled2,
        activation="relu",
        name="dense_expand",
    )(x)

    x = Reshape(
        (pooled2, 16),
        name="reshape",
    )(x)

    x = Conv1D(
        16, 3,
        activation="relu",
        padding="same",
        name="deconv1",
    )(x)

    x = UpSampling1D(
        2,
        name="up1",
    )(x)

    x = Conv1D(
        32, 3,
        activation="relu",
        padding="same",
        name="deconv2",
    )(x)

    x = UpSampling1D(
        2,
        name="up2",
    )(x)

    x = Conv1D(
        1, 3,
        activation="linear",
        padding="same",
        name="deconv_out",
    )(x)

    if crop_right > 0:
        x = Cropping1D(
            cropping=(0, crop_right),
            name="crop",
        )(x)

    out = Flatten(name="output")(x)

    return Model(inp, out, name="CNN_AE")


def build_model(model_name, input_dim, dataset_name):
    if model_name == "SNN_AE":
        return build_snn_ae(
            input_dim,
            DATASETS[dataset_name]["snn"],
        )

    if model_name == "Dense_AE":
        return build_dense_ae(input_dim)

    if model_name == "LSTM_AE":
        return build_lstm_ae(input_dim)

    if model_name == "CNN_AE":
        return build_cnn_ae(input_dim)

    raise ValueError(model_name)


# =============================================================================
# MODEL IO SHAPES
# =============================================================================

def input_for_model(X, model_name):
    if model_name == "LSTM_AE":
        return X.reshape(-1, 1, X.shape[1])

    if model_name == "CNN_AE":
        return X.reshape(-1, X.shape[1], 1)

    return X


def target_for_model(X, model_name):
    if model_name == "LSTM_AE":
        return X.reshape(-1, 1, X.shape[1])

    # CNN output is flattened.
    return X


def predict_flat(model, X, model_name):
    pred = model.predict(
        input_for_model(X, model_name),
        batch_size=1024,
        verbose=0,
    )

    if model_name == "LSTM_AE":
        return pred.reshape(len(X), -1)

    return pred.reshape(len(X), -1)


# =============================================================================
# SPIKE DIAGNOSTICS
# =============================================================================

def spike_rate_for_subset(model, X):
    if len(X) == 0:
        return np.nan

    encoder = model.get_layer("encoder")

    enc_model = Model(
        model.input,
        encoder.output,
    )

    sample = X[:min(SPIKE_SAMPLE_MAX, len(X))]

    spikes = enc_model.predict(
        sample,
        batch_size=1024,
        verbose=0,
    )

    rate = float(np.mean(spikes))

    del enc_model, spikes
    gc.collect()

    return rate


def snn_spike_stats(model, X_test, y_test):
    normal = X_test[y_test == 0]
    attack = X_test[y_test == 1]

    rn = spike_rate_for_subset(model, normal)
    ra = spike_rate_for_subset(model, attack)

    return {
        "spike_rate_normal": rn,
        "spike_rate_attack": ra,
        "spike_rate_delta_attack_minus_normal": ra - rn,
        "sparsity_normal": 1.0 - rn,
        "sparsity_attack": 1.0 - ra,
    }


# =============================================================================
# RUN ONE MODEL
# =============================================================================

def run_one(
    dataset_name,
    condition_name,
    feature_cols,
    model_name,
    sample,
    split,
):
    run_dir = (
        DIRS["runs"]
        / dataset_name
        / condition_name
        / model_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.json"

    if metrics_file.exists():
        print(
            f"  [RESUME] {dataset_name} "
            f"{condition_name} {model_name}"
        )
        return

    cfg = DATASETS[dataset_name]

    train_idx = split["train_normal_idx"]
    val_idx = split["val_normal_idx"]
    test_idx = split["test_idx"]

    Xall = sample[feature_cols].to_numpy(
        dtype=np.float32
    )

    yall = (
        sample[cfg["label"]].astype(str)
        != cfg["normal"]
    ).astype(np.int8).to_numpy()

    X_train_raw = Xall[train_idx]
    X_val_raw = Xall[val_idx]
    X_test_raw = Xall[test_idx]
    y_test = yall[test_idx]

    subtype_test = (
        sample.iloc[test_idx]["label_type"]
        .astype(str)
        .to_numpy()
    )

    source_rows_test = (
        sample.iloc[test_idx]["source_row_index"]
        .to_numpy(dtype=np.int64)
    )

    scaler = make_scaler(
        dataset_name,
        len(X_train_raw),
    )

    X_train = scaler.fit_transform(
        X_train_raw
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val_raw
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test_raw
    ).astype(np.float32)

    scaler_dir = (
        DIRS["scalers"]
        / dataset_name
        / condition_name
    )
    scaler_dir.mkdir(parents=True, exist_ok=True)

    scaler_file = (
        scaler_dir
        / "scaler.pkl"
    )

    # identical scaler for all model types under this condition
    if not scaler_file.exists():
        joblib.dump(scaler, scaler_file)

    model_seed = SEED + {
        "SNN_AE": 101,
        "Dense_AE": 202,
        "LSTM_AE": 303,
        "CNN_AE": 404,
    }[model_name]

    set_seed(model_seed)

    model = build_model(
        model_name,
        len(feature_cols),
        dataset_name,
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0 if model_name in {
            "SNN_AE", "Dense_AE"
        } else None,
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
    )

    es = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    print(
        f"\n  RUN {dataset_name} | "
        f"{condition_name} | {model_name}"
    )
    print(
        f"      train={len(X_train):,} "
        f"val={len(X_val):,} "
        f"test={len(X_test):,}"
    )

    t0 = time.time()

    history = model.fit(
        input_for_model(X_train, model_name),
        target_for_model(X_train, model_name),
        validation_data=(
            input_for_model(X_val, model_name),
            target_for_model(X_val, model_name),
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )

    train_time = time.time() - t0

    pd.DataFrame(history.history).to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    val_pred = predict_flat(
        model,
        X_val,
        model_name,
    )

    test_pred = predict_flat(
        model,
        X_test,
        model_name,
    )

    val_se = np.square(
        X_val - val_pred
    )
    test_se = np.square(
        X_test - test_pred
    )

    val_scores = np.mean(
        val_se,
        axis=1,
    )

    test_scores = np.mean(
        test_se,
        axis=1,
    )

    threshold = float(np.percentile(
        val_scores,
        OPERATING_PERCENTILE,
    ))

    main_metrics = metric_dict(
        y_test,
        test_scores,
        threshold,
    )

    test_prediction = (
        test_scores > threshold
    ).astype(np.int8)

    # -------------------------------------------------------------------------
    # Reconstruction summary.
    # -------------------------------------------------------------------------
    recon_summary_rows = []

    for j, feature in enumerate(feature_cols):
        nvals = test_se[y_test == 0, j]
        avals = test_se[y_test == 1, j]

        recon_summary_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "model": model_name,
            "feature": feature,
            "normal_mse_mean": float(np.mean(nvals)),
            "normal_mse_median": float(np.median(nvals)),
            "attack_mse_mean": float(np.mean(avals)),
            "attack_mse_median": float(np.median(avals)),
        })

    pd.DataFrame(recon_summary_rows).to_csv(
        run_dir / "feature_reconstruction_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Per-subtype attack detection.
    # -------------------------------------------------------------------------
    subtype_rows = []

    attack_mask = y_test == 1

    for subtype in sorted(
        np.unique(subtype_test[attack_mask])
    ):
        mask = (
            attack_mask
            & (subtype_test == subtype)
        )

        sub_scores = test_scores[mask]
        sub_pred = test_prediction[mask]

        subtype_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "model": model_name,
            "subtype": subtype,
            "n": int(np.sum(mask)),
            "detected": int(np.sum(sub_pred == 1)),
            "missed": int(np.sum(sub_pred == 0)),
            "recall": float(np.mean(sub_pred)),
            "fnr": float(1.0 - np.mean(sub_pred)),
            "score_mean": float(np.mean(sub_scores)),
            "score_median": float(np.median(sub_scores)),
        })

    pd.DataFrame(subtype_rows).to_csv(
        run_dir / "subtype_detection.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Diagnostics specific to 34F model.
    # These require NO retraining.
    # -------------------------------------------------------------------------
    diagnostics = {}

    prediction_df = pd.DataFrame({
        "source_row_index": source_rows_test,
        "y_true": y_test,
        "subtype": subtype_test,
        "score": test_scores,
        "prediction": test_prediction,
    })

    if "frame.time_epoch" in feature_cols:
        epoch_idx = feature_cols.index(
            "frame.time_epoch"
        )

        # A. Same trained 34F model, but remove the epoch component
        #    from reconstruction score.
        keep = [
            i for i in range(len(feature_cols))
            if i != epoch_idx
        ]

        val_score_no_epoch = np.mean(
            val_se[:, keep],
            axis=1,
        )

        test_score_no_epoch = np.mean(
            test_se[:, keep],
            axis=1,
        )

        threshold_no_epoch = float(
            np.percentile(
                val_score_no_epoch,
                OPERATING_PERCENTILE,
            )
        )

        metrics_score_no_epoch = metric_dict(
            y_test,
            test_score_no_epoch,
            threshold_no_epoch,
        )

        # B. Neutralize epoch input at inference using
        #    median epoch of training-normal after scaling.
        neutral_value = float(
            np.median(X_train[:, epoch_idx])
        )

        X_test_neutral = X_test.copy()
        X_test_neutral[:, epoch_idx] = neutral_value

        neutral_pred = predict_flat(
            model,
            X_test_neutral,
            model_name,
        )

        neutral_score = np.mean(
            np.square(
                X_test_neutral - neutral_pred
            ),
            axis=1,
        )

        metrics_neutral = metric_dict(
            y_test,
            neutral_score,
            threshold,
        )

        prediction_df[
            "score_excluding_epoch_component"
        ] = test_score_no_epoch

        prediction_df[
            "score_epoch_neutralized"
        ] = neutral_score

        prediction_df[
            "prediction_epoch_neutralized"
        ] = (
            neutral_score > threshold
        ).astype(np.int8)

        # Contribution of epoch reconstruction error.
        attack_se = test_se[y_test == 1]
        normal_se = test_se[y_test == 0]

        epoch_attack_contribution = float(
            np.sum(attack_se[:, epoch_idx])
            / max(np.sum(attack_se), 1e-12)
        )

        epoch_normal_contribution = float(
            np.sum(normal_se[:, epoch_idx])
            / max(np.sum(normal_se), 1e-12)
        )

        diagnostics.update({
            "epoch_scaled_train_normal_median":
                neutral_value,
            "score_excluding_epoch_component":
                metrics_score_no_epoch,
            "epoch_neutralized_input":
                metrics_neutral,
            "epoch_reconstruction_contribution_attack":
                epoch_attack_contribution,
            "epoch_reconstruction_contribution_normal":
                epoch_normal_contribution,
        })

        # SNN internal representation after epoch neutralization.
        if model_name == "SNN_AE":
            original_spikes = snn_spike_stats(
                model,
                X_test,
                y_test,
            )

            neutral_spikes = snn_spike_stats(
                model,
                X_test_neutral,
                y_test,
            )

            diagnostics[
                "snn_spike_original"
            ] = original_spikes

            diagnostics[
                "snn_spike_epoch_neutralized"
            ] = neutral_spikes

        del X_test_neutral, neutral_pred

    elif model_name == "SNN_AE":
        diagnostics[
            "snn_spike_original"
        ] = snn_spike_stats(
            model,
            X_test,
            y_test,
        )

    prediction_df.to_csv(
        run_dir / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    model_path = None

    if SAVE_MODELS:
        model_dir = (
            DIRS["models"]
            / dataset_name
            / condition_name
        )
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            model_dir
            / f"{model_name}.keras"
        )
        model.save(model_path)

    result = {
        "dataset": dataset_name,
        "condition": condition_name,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "model": model_name,
        "model_seed": model_seed,
        "scaler": cfg["scaler"],
        "learning_rate": LEARNING_RATE,
        "epochs_max": EPOCHS,
        "epochs_used": len(
            history.history.get("loss", [])
        ),
        "train_time_sec": train_time,
        "params": int(model.count_params()),
        "operating_percentile": OPERATING_PERCENTILE,
        "main_metrics": main_metrics,
        "diagnostics": diagnostics,
        "model_path": model_path,
        "scaler_path": scaler_file,
    }

    save_json(
        metrics_file,
        result,
    )

    print(
        f"      F1={main_metrics['f1']:.4f} "
        f"R={main_metrics['recall']:.4f} "
        f"FNR={main_metrics['fnr']:.4f} "
        f"AUC={main_metrics['auc']:.4f}"
    )

    del (
        model,
        history,
        scaler,
        Xall,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        X_train,
        X_val,
        X_test,
        val_pred,
        test_pred,
        val_se,
        test_se,
        prediction_df,
    )

    tf.keras.backend.clear_session()
    gc.collect()


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_outputs():
    metric_rows = []
    subtype_frames = []
    perturb_rows = []
    spike_rows = []
    epoch_contrib_rows = []

    for dataset_name in DATASETS:
        for condition_name in CONDITIONS:
            for model_name in MODELS:
                run_dir = (
                    DIRS["runs"]
                    / dataset_name
                    / condition_name
                    / model_name
                )

                metric_file = run_dir / "metrics.json"

                if not metric_file.exists():
                    continue

                with metric_file.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    payload = json.load(f)

                m = payload["main_metrics"]

                metric_rows.append({
                    "dataset": dataset_name,
                    "condition": condition_name,
                    "model": model_name,
                    "feature_count": payload["feature_count"],
                    "params": payload["params"],
                    "epochs_used": payload["epochs_used"],
                    "train_time_sec": payload["train_time_sec"],
                    **m,
                })

                subtype_file = (
                    run_dir
                    / "subtype_detection.csv"
                )

                if subtype_file.exists():
                    subtype_frames.append(
                        pd.read_csv(subtype_file)
                    )

                diag = payload.get(
                    "diagnostics", {}
                )

                if diag.get(
                    "score_excluding_epoch_component"
                ):
                    mm = diag[
                        "score_excluding_epoch_component"
                    ]
                    perturb_rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "diagnostic":
                            "34F_score_excluding_epoch_component",
                        **mm,
                    })

                if diag.get(
                    "epoch_neutralized_input"
                ):
                    mm = diag[
                        "epoch_neutralized_input"
                    ]
                    perturb_rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "diagnostic":
                            "34F_epoch_neutralized_input",
                        **mm,
                    })

                if (
                    "epoch_reconstruction_contribution_attack"
                    in diag
                ):
                    epoch_contrib_rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "attack_epoch_error_fraction":
                            diag[
                                "epoch_reconstruction_contribution_attack"
                            ],
                        "normal_epoch_error_fraction":
                            diag[
                                "epoch_reconstruction_contribution_normal"
                            ],
                    })

                for key in [
                    "snn_spike_original",
                    "snn_spike_epoch_neutralized",
                ]:
                    if key in diag:
                        spike_rows.append({
                            "dataset": dataset_name,
                            "condition": condition_name,
                            "model": model_name,
                            "spike_condition": key,
                            **diag[key],
                        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        DIRS["tables"]
        / "model_condition_metrics.csv",
        index=False,
    )

    if subtype_frames:
        subtype = pd.concat(
            subtype_frames,
            ignore_index=True,
        )
    else:
        subtype = pd.DataFrame()

    subtype.to_csv(
        DIRS["tables"]
        / "subtype_detection_metrics.csv",
        index=False,
    )

    pd.DataFrame(perturb_rows).to_csv(
        DIRS["tables"]
        / "inference_perturbation_metrics.csv",
        index=False,
    )

    pd.DataFrame(spike_rows).to_csv(
        DIRS["tables"]
        / "snn_spike_statistics.csv",
        index=False,
    )

    pd.DataFrame(epoch_contrib_rows).to_csv(
        DIRS["tables"]
        / "epoch_reconstruction_contribution.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 34F vs 33F model effect
    # Delta convention = 33F - 34F.
    # Negative F1/Recall means performance fell after removing epoch.
    # -------------------------------------------------------------------------
    effects = []

    for (dataset, model), g in metrics.groupby(
        ["dataset", "model"]
    ):
        a = g[
            g["condition"] == "with_epoch_34F"
        ]
        b = g[
            g["condition"] == "without_epoch_33F"
        ]

        if len(a) != 1 or len(b) != 1:
            continue

        a = a.iloc[0]
        b = b.iloc[0]

        effects.append({
            "dataset": dataset,
            "model": model,

            "f1_34F": a["f1"],
            "f1_33F": b["f1"],
            "delta_f1_33F_minus_34F":
                b["f1"] - a["f1"],

            "recall_34F": a["recall"],
            "recall_33F": b["recall"],
            "delta_recall_33F_minus_34F":
                b["recall"] - a["recall"],

            "fnr_34F": a["fnr"],
            "fnr_33F": b["fnr"],
            "delta_fnr_33F_minus_34F":
                b["fnr"] - a["fnr"],

            "fpr_34F": a["fpr"],
            "fpr_33F": b["fpr"],
            "delta_fpr_33F_minus_34F":
                b["fpr"] - a["fpr"],

            "auc_34F": a["auc"],
            "auc_33F": b["auc"],
            "delta_auc_33F_minus_34F":
                b["auc"] - a["auc"],
        })

    effect_df = pd.DataFrame(effects)

    effect_df.to_csv(
        DIRS["tables"]
        / "model_epoch_effect.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Per-subtype delta.
    # -------------------------------------------------------------------------
    subtype_effect_rows = []

    if not subtype.empty:
        for (
            dataset,
            model,
            subtype_name,
        ), g in subtype.groupby(
            ["dataset", "model", "subtype"]
        ):
            a = g[
                g["condition"] == "with_epoch_34F"
            ]
            b = g[
                g["condition"] == "without_epoch_33F"
            ]

            if len(a) != 1 or len(b) != 1:
                continue

            a = a.iloc[0]
            b = b.iloc[0]

            subtype_effect_rows.append({
                "dataset": dataset,
                "model": model,
                "subtype": subtype_name,
                "recall_34F": a["recall"],
                "recall_33F": b["recall"],
                "delta_recall_33F_minus_34F":
                    b["recall"] - a["recall"],
                "fnr_34F": a["fnr"],
                "fnr_33F": b["fnr"],
                "delta_fnr_33F_minus_34F":
                    b["fnr"] - a["fnr"],
            })

    pd.DataFrame(
        subtype_effect_rows
    ).to_csv(
        DIRS["tables"]
        / "subtype_epoch_effect.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figure source + PNG.
    # -------------------------------------------------------------------------
    if not effect_df.empty:
        effect_df.to_csv(
            DIRS["figures"]
            / "figure_delta_metrics_source.csv",
            index=False,
        )

        for metric, label in [
            (
                "delta_f1_33F_minus_34F",
                "ΔF1 (33F - 34F)",
            ),
            (
                "delta_recall_33F_minus_34F",
                "ΔRecall (33F - 34F)",
            ),
        ]:
            pivot = effect_df.pivot(
                index="model",
                columns="dataset",
                values=metric,
            )

            ax = pivot.plot(
                kind="bar",
                figsize=(9, 5),
            )

            ax.axhline(
                0,
                linewidth=1,
            )
            ax.set_ylabel(label)
            ax.set_xlabel("Model")
            ax.set_title(
                f"Effect of Removing frame.time_epoch: {label}"
            )
            plt.xticks(rotation=0)
            plt.tight_layout()

            filename = (
                "delta_f1.png"
                if "f1" in metric
                else "delta_recall.png"
            )

            plt.savefig(
                DIRS["figures"] / filename,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

    # -------------------------------------------------------------------------
    # Markdown summary.
    # -------------------------------------------------------------------------
    lines = [
        "# Temporal Feature Model Audit",
        "",
        "Audit purpose: quantify the effect of "
        "`frame.time_epoch` on SNN-AE and reconstruction-based baselines.",
        "",
        "This audit is diagnostic and does not replace the official benchmark.",
        "",
        "## Main 34F vs 33F effect",
        "",
    ]

    if not effect_df.empty:
        lines.append(
            effect_df.to_markdown(index=False)
        )

    lines.extend([
        "",
        "## Output files",
        "",
        "- `tables/model_condition_metrics.csv`",
        "- `tables/model_epoch_effect.csv`",
        "- `tables/subtype_detection_metrics.csv`",
        "- `tables/subtype_epoch_effect.csv`",
        "- `tables/inference_perturbation_metrics.csv`",
        "- `tables/epoch_reconstruction_contribution.csv`",
        "- `tables/snn_spike_statistics.csv`",
        "- `figures/delta_f1.png`",
        "- `figures/delta_recall.png`",
        "- per-run training history, raw predictions, and feature reconstruction summaries",
        "",
    ])

    (ROOT / "MODEL_AUDIT_SUMMARY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("TEMPORAL FEATURE MODEL AUDIT")
    print("=" * 100)
    print(
        "Purpose: controlled diagnostic of "
        "frame.time_epoch across SNN-AE and baseline AEs."
    )
    print(
        "IMPORTANT: this is NOT a replacement "
        "for the official benchmark."
    )
    print()

    save_json(
        DIRS["metadata"] / "audit_config.json",
        {
            "datasets": DATASETS,
            "features_34": FEATURES_34,
            "features_33": FINAL_33,
            "attack_per_subtype": ATTACK_PER_SUBTYPE,
            "train_normal_ratio": TRAIN_NORMAL_RATIO,
            "val_normal_ratio": VAL_NORMAL_RATIO,
            "test_normal_ratio": TEST_NORMAL_RATIO,
            "seed": SEED,
            "models": MODELS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "time_steps": TIME_STEPS,
            "operating_percentile": OPERATING_PERCENTILE,
        },
    )

    save_json(
        DIRS["metadata"] / "environment.json",
        {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
    )

    for dataset_name in DATASETS:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        sample = materialize_sample(
            dataset_name
        )

        with np.load(
            split_path(dataset_name),
            allow_pickle=False,
        ) as z:
            split = {
                k: z[k]
                for k in z.files
            }

        cfg = DATASETS[dataset_name]

        # Audit split validation.
        labels = (
            sample[cfg["label"]].astype(str)
            != cfg["normal"]
        ).astype(np.int8).to_numpy()

        if np.any(
            labels[split["train_normal_idx"]] != 0
        ):
            raise RuntimeError(
                "Attack leaked into train-normal."
            )

        if np.any(
            labels[split["val_normal_idx"]] != 0
        ):
            raise RuntimeError(
                "Attack leaked into validation-normal."
            )

        for condition_name, features in CONDITIONS.items():
            print(
                f"\n[{dataset_name}] CONDITION: "
                f"{condition_name}"
            )

            # fit scaler once per dataset/condition
            # run_one saves the first copy; transformation itself
            # remains deterministic for every model.
            for model_name in MODELS:
                run_one(
                    dataset_name,
                    condition_name,
                    features,
                    model_name,
                    sample,
                    split,
                )

        del sample, split, labels
        tf.keras.backend.clear_session()
        gc.collect()

    aggregate_outputs()

    save_json(
        ROOT / "RUN_COMPLETE.json",
        {
            "status": "PASS",
            "expected_training_runs":
                len(DATASETS)
                * len(CONDITIONS)
                * len(MODELS),
            "completed": True,
            "note": (
                "Diagnostic temporal-feature audit only; "
                "official benchmark remains unchanged."
            ),
        },
    )

    print("\n" + "=" * 100)
    print("TEMPORAL FEATURE MODEL AUDIT COMPLETE")
    print("=" * 100)
    print(f"Output: {ROOT}")
    print()
    print("Main result:")
    print(
        DIRS["tables"]
        / "model_epoch_effect.csv"
    )


if __name__ == "__main__":
    main()
