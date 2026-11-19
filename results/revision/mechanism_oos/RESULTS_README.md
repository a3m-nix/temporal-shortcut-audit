# OOS mechanism tests — revision (R2 M1/M6, R7)

- Scripts: `26-mechanism-oos-revision-v1.py` (inference) and `27-aggregate-mechanism-oos-revision-v1.py` (aggregation)
- Pre-specified predictions: `metadata/PREREGISTERED_PREDICTIONS.md`, SHA-256 `68dee8a6…5d38`.
  The file was written and hashed before the run, and script 26 refuses to run if the hash changes.
- Units: 360 saved weight-fixed λ models (3 datasets × 4 AEs × 3 seeds × 10 folds). Inference only.
  Splits, scalers, weights and the P99 non-epoch threshold are unchanged.
- Sanity gate: the recomputed λ-grid FNR/FPR is identical to the stored `lambda_summary.csv` (max |Δ| = 0).
- Pilot (2 CICIoT2023 Dense-AE runs, used only for the sanity gate): `pilot/`

## Verdicts: 32 of 37 pre-specified predictions met (`tables/PREDICTION_VERDICTS.csv`)

| Prediction | Met | Key numbers |
|---|---|---|
| P1 clipping to [0,1] ≈ neutralisation (λ=0) | 12/12 | CIC/ToN: CLIP−L1 = +29.3 to +39.7 pp (all CIs > 0); \|CLIP−L0\| ≤ 2.1 pp, Holm n.s.; Edge: CLIP = L1 exactly (attacks in support) |
| P2 class-independent OOS raises FPR, Edge included | 3/3 | d=10 assigned to all rows: FPR 96–100% on all datasets, FNR ≈ 0 |
| P3 permuting timestamps raises FPR | 6/6 | FPR 1% → 50.3–50.5% (= share of attack timestamps in the balanced corpus); FNR ≈ λ0/2 |
| P4 future normal traffic produces false alarms | 12/12 cells | normal at t = 2 (one span past training): FPR 2.8–42.9%; at t = 3: 21.7–97.3% |
| P5 first-layer activation inflation (mean L2) | 6/9 | the mean is dominated by heavy-tailed rows; the median-L2 ratio (descriptive, added after the pilot) is 34–61× on CIC and 3.9–17× on ToN, and ≈1.0 on Edge |
| P6 detection increases with OOS distance | 4/6 | the (0,0.1] bin dips by 1.2–2.3 pp for CIC Dense/LSTM; otherwise 0.70–0.73 at d=0 → 0.92–0.96 at (1,10] → 1.00 at >10 |

## Figures
- `figures/fig_class_independent_oos_response.png`: attack-detected and normal-flagged rates vs the OOS value assigned to every row
- `figures/fig_detection_by_oos_distance.png`: attack detection by OOS distance, pooled over λ

## Tables
`mechanism_summary.csv`, `mechanism_nb_tests.csv` (Nadeau–Bengio, ratio 1/9, Holm-12),
`mechanism_activation_summary.csv`, `mechanism_distance_binned_detection.csv`, `PREDICTION_VERDICTS.csv`
