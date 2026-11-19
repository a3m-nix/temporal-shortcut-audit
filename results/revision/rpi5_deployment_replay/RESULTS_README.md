# RPi5 deployment replay 

**Scope for the paper: phase 1 (accuracy by deployment horizon) only.**
The FNIRSI power/energy data (phase 2, prediction D5) were collected in the same run, but by the
author's decision they are NOT used or reported for this paper.

- Pre-specified predictions: `metadata/PREREGISTERED_PREDICTIONS.md`, SHA-256 `f58fa6c8…635b`, hashed before export
- Export: `31-export-rpi5-deployment-replay-v1.py`. 24 SavedModels (Exp II seed 0; A=q+t and B=q; 4 AEs × 3 datasets).
  Every export reproduces the stored server FNR/FPR exactly. On the device, as-recorded FNR equals the server values.
- Device run: RPi5 (TF 2.20, 1 thread, pinned core). Other workloads on the device were paused during the run (2026-11-19, 17 min).
- Aggregation: `32-aggregate-rpi5-deployment-replay-v1.py`

## Result: an absolute-time detector floods alerts as soon as it is deployed
FPR on normal traffic (%), fixed threshold, the same packets arriving H after the training capture:

| Dataset | A (with t): as-recorded → +1d → +30d → NOW | B (without t), all H |
|---|---|---|
| CICIoT2023 | CNN/Dense/LSTM 0.0 → 100 → 100 → 100; SNN 0.0 → 7.4 → 7.4 → 7.4 | 0.00 |
| ToN-IoT | 1.5–2.9 → 14–100 → 100 → 100 (all four) | 0.84–1.14 |
| Edge-IIoTset | 0.2–1.5 → 1.1–79.5 → 31–100 → 52–100 | 1.45–1.67 |

Verdicts on the phase-1 predictions: D1 met (B identical across H); D2 met (A FPR non-decreasing from +1d);
D3 met (A FPR ≥ 90% at +365d and NOW for ≥ 3/4 models in every dataset); D4 met (A > B at +1d, 4/4 on CIC and ToN).

Figure: `figures/fig_rpi5_alert_flood_by_horizon.png`. Tables: `tables/fpr_by_horizon.csv`, `tables/fnr_by_horizon.csv`.
