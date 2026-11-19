# Attack-family extension: ToN-IoT DoS and Edge-IIoTset non-flood families — pre-specified predictions

Written 2026-11-19, BEFORE the samples were drawn and before any model was trained. The SHA-256 is in PREREGISTERED_PREDICTIONS.sha256.

## Data (facts established before writing)
- ToN-IoT DoS: 7,088,124 packet rows in the parent table (ton_iot_bidir.parquet). They span 2019-04-24 18:13 to 2019-04-25 01:22,
  which is 100% after the normal window (2019-04-02 15:52 to 2019-04-03 10:25).
- Edge-IIoTset non-flood: 10 attack PCAPs from the official release, extracted with the parent table's tshark field list.
  Validation: re-extracting the DDoS-HTTP PCAP reproduced all 34 features of its 229,142 parent rows exactly.
  The vulnerability-scanner PCAP ends with a corrupt record, so its first 265,827 packets are used.
  All 10 families fall between 2021-11-24 and 2021-12-28, inside the Edge normal window (2021-11-21 to 2022-01-05).

## Samples (fixed before training)
- ToN-IoT DoS: 10 equal-width time bins over the DoS range, 1,000 rows drawn at random per bin (NumPy seed 42) = 10,000 attack rows.
- Edge non-flood: 1,000 packets drawn at random per family (seed 42) = 10,000 attack rows.
- Normal rows: the normal rows of the existing fixed audit samples (ToN 18,000; Edge 14,400). They are placed first in the same
  order, so the balanced-corpus draw (seed 4242, 10,000 normal rows) follows the Experiment I rule.

## Protocol (Experiment I, unchanged)
34 features (q + IAT + t); 3 seeds × 10 stratified folds (strata: ToN time bin / Edge family); train on train-normal (80%);
MinMax fitted on train-normal; early stopping and P99 threshold on val-normal (20%); score = MSE over the 33 non-time components.
Four architectures. λ grid as in Experiment I, plus clipping of t to [0,1]. Primary contrast: FNR at λ = 0 minus λ = 1.
Nadeau–Bengio corrected t-test (ratio 1/9), Holm over the 8 dataset × model tests.

## Predictions
X1 ToN-IoT DoS (attacks out of support): ΔFNR(λ0 − λ1) > +20 pp with a corrected CI above 0 and Holm p < 0.05 for at least 3 of 4 models.
X2 ToN-IoT DoS: |FNR_clip − FNR_λ0| < 5 pp for at least 3 of 4 models.
X3 Edge non-flood (attacks in support under random folds): the corrected CI of ΔFNR(λ0 − λ1) includes 0, or |ΔFNR| < 5 pp, for at least 3 of 4 models.
X4 Edge non-flood: |FNR_clip − FNR_λ1| < 1 pp for all 4 models (clipping is approximately a no-op).
Per-family ΔFNR and the λ = 1 detection level are reported descriptively, with no prediction.
Each prediction is reported as met / not met. No prediction is changed after the run.
