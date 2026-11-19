# Edge-IIoTset Experiment II re-run with repaired early stopping — pre-specified predictions

Written: 2026-11-19, BEFORE any re-run model was trained. The SHA-256 is in PREREGISTERED_PREDICTIONS.sha256.

## Diagnosis (computed before writing this file)
Original Exp II Edge runs stopped at epoch 6–10 (best epoch ≈1). earlystop_normal is a later
parent-order block: 18.2% of its rows have q features outside fit-normal support (data.len up to
107, tcp.ack 75.7, tcp.seq 31.1). Its val_loss (≈10) rose from epoch 1, so the restored weights
were essentially untrained (non-SNN ROC-AUC 0.47–0.68). calibration_normal is in-support for q
(0.1% of rows). Exp II Edge timestamp support: test-normal 0% OOS; test-attack 100% OOS (median
1.53 below support); calibration-normal 37% OOS (median 0); earlystop-normal 77% OOS.

## Unchanged from Exp II
Prepared data (v1.2), split manifest, conditions A=q+t, B=q, E=q+counter, N=q+in-support noise
(the same feature construction as script 21), MinMax scaler fitted on all fit_normal rows,
architectures, Adam 5e-4, batch 256, max 30 epochs, model seeds (A/B: own condition index;
E/N: A's index, as in script 21), q-only score, P99 calibration-normal threshold, test set.

## Changed (the repair)
- FITVAL (primary): early stopping (patience 5, restore best) on a fixed random 15% hold-out of
  fit_normal (RandomState(280000 + seed)); the model is trained on the remaining 85%.
  earlystop_normal is not used for training decisions.
- FIXED30 (sensitivity): the full fit_normal, exactly 30 epochs, no early stopping.
- ORIG (pilot only): the original protocol re-run for 2 configurations, to verify that the
  re-implementation reproduces the stored Exp II metrics.

## Predictions (non-SNN = Dense, LSTM, CNN)
R1 Training repaired: under FITVAL, the median epochs used is > 10 and B_q ROC-AUC > 0.90 for at least 2 of 3 non-SNN models.
R2 Under FITVAL, B_q FNR < 30% for at least 2 of 3 non-SNN models (originally 61.5–62.0%).
R3 Mechanism (directional): Edge attacks are OOS (1.53), so B−A FNR > 0 for at least 2 of 3 non-SNN models.
   Stated risk: the A threshold is calibrated on calibration-normal, of which 37% is t-OOS,
   which may inflate the A threshold and mask this effect.
R4 Under FITVAL, |N−B| FNR < 5 pp for at least 2 of 3 non-SNN models.
R5 SNN: under FITVAL, |B−A| for SNN is smaller than the original 45.26 pp.
R6 FIXED30 agrees with FITVAL on the sign of B−A for at least 3 of 4 models.
Each prediction is reported as met / not met. No prediction is changed after the run.
