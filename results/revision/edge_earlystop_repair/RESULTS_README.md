# Edge-IIoTset Exp II re-run with repaired early stopping 

- Scripts: `0-temporal-feature-audit-v4.../28-edge-earlystop-repair-v1.py` (training) and `29-aggregate-edge-earlystop-repair-v1.py` (aggregation)
- Pre-specified predictions: `metadata/PREREGISTERED_PREDICTIONS.md`, SHA-256 `b5e696e6…5a41`, hashed before training
- Reproduction gate (ORIG protocol): Dense-AE B seed 0 FNR 61.88% / AUC 0.564 and SNN-AE B seed 0 FNR 4.04% match the stored v1.2 runs exactly
- 96 runs = {FITVAL, FIXED30} × {A, B, E, N} × 4 AEs × 3 seeds

## Main result: the PCA gap was an early-stopping artefact
| Model | Original A / B / E / N (FNR %) | FITVAL A / B / E / N | FIXED30 A / B / E / N |
|---|---|---|---|
| SNN-AE | 62.05 / 16.79 / 16.82 / 31.92 | 53.90 / 4.68 / 14.18 / 8.25 | 54.19 / 1.87 / 11.98 / 3.16 |
| Dense-AE | 61.18 / 61.92 / 61.86 / 61.96 | 26.63 / 0.08 / 0.12 / 0.08 | 27.78 / 0.07 / 0.18 / 0.08 |
| LSTM-AE | 61.57 / 62.02 / 58.49 / 46.78 | 37.75 / 14.56 / 6.22 / 14.80 | 38.73 / 2.92 / 3.25 / 2.22 |
| CNN-AE | 61.88 / 61.52 / 60.15 / 60.90 | 13.37 / 0.31 / 0.42 / 0.88 | 8.43 / 0.17 / 0.09 / 0.11 |

PCA (B, q-only) = 0.075%. Repaired Dense-AE B = 0.075%. B ROC-AUC is now 0.987–0.995 (originally 0.58–0.61).
The E (counter) arm raises FPR to 21–47%, because Edge test-normal counter values are OOS (median 0.54).

## Verdicts: 4 of 6 met (`tables/PREDICTION_VERDICTS.csv`)
- R1 training repaired: met. R2 B FNR < 30%: met. R4 N ≈ B: met. R6 FIXED30 agrees with FITVAL: met (4/4).
- R3 (B−A > 0): **not met**. B−A = −13 to −27 pp for the non-SNN models.
- R5 (SNN gap shrinks): **not met** (49.2 vs 45.3 pp).
- Both failures match the risk stated before the run (37% of calibration-normal rows have OOS t).
  Post-hoc diagnostic (`posthoc/`, labelled post-hoc): in A, calibration rows with OOS t have
  q-scores 6–30× higher than in-support rows, so the P99 threshold is inflated 2–25×. With the
  threshold taken from in-support calibration rows only, A FNR = 0.0–2.2% (vs B 0.08–14.6%)
  and FPR = 2.3–2.6%.

## Interpretation
The OOS timestamp mechanism acts on both sides of the decision. Attacks with OOS t receive higher
scores, and calibration normals with OOS t raise the threshold. Its net sign depends on the
acquisition layout. On Edge the threshold inflation dominates, which explains the original
"SNN exception" and the negative Edge B−A.
