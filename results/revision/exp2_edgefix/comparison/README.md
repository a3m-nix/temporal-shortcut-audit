# Exp II Edge-IIoTset numbers: original v1.2 vs edgefix (script 33/34)

Identity check (non_edge_identity_check.csv): CICIoT2023 and ToN-IoT models were not retrained (their runs were copied
from v1.2); the intervention and analysis stages were re-run for all datasets. In 26 of the 28 v1.2 and script-24 tables
every CICIoT2023 and ToN-IoT value equals v1.2 (max |diff| = 0; runtime and path columns excluded). The two exceptions are
the Holm-adjusted p-values in two statistics tables, because the Holm family of 12 dataset × model tests includes
Edge-IIoTset. No significance decision for CICIoT2023 or ToN-IoT changed.

Correction (2026-11-19, final audit): the first edgefix run reused the Edge-IIoTset intervention outputs copied from
v1.2 (per-scheme RUN_COMPLETE.json resume), so the shifted predictions came from the old models. They were moved to
`_superseded/stale_edge_interventions_fix_2026-11-19/` and recomputed with the retrained models; script 24 and this
comparison were then re-run. The neutralisation row below is the corrected value.

| Manuscript quantity (Edge-IIoTset) | Original v1.2 | Edgefix |
|---|---|---|
| Neural ROC-AUC (all conditions, non-SNN) | 0.56–0.68 | 0.98–0.999 |
| B=q FNR at P99 | 61.5–62.0% (SNN 16.8%) | 0.08–14.6% (SNN 4.7%) |
| OR ensemble A → B recall (FPR) | 39.38% → 83.50% (2.86% → 3.55%) | 91.40% → 99.93% (1.54% → 1.86%) |
| A → B detected-by-all → missed-by-all | 0 of 24,000 | 0 of 24,000 (467 reverse) |
| P95 B−A (CNN/Dense/LSTM; SNN) | −1.3 to +3.7; −52.7 | −3.3 to −22.5; −32.7 |
| P99 B−A (CNN/Dense/LSTM; SNN) | −0.4 to +0.7; −45.3 | −12.9 to −26.6; −49.5 |
| P99.9 | ≈62% FNR for A and B (1.4 tail observations) | unchanged (degenerate tail) |
| Weight-fixed neutralisation of A (packet-level ΔFNR) | ≈0 | fixed threshold: CNN +1.7, Dense +8.3, LSTM +1.4, SNN +0.6 pp; recalibrated threshold: CNN −8.8, Dense −26.6, LSTM −26.3, SNN −46.7 pp (FPR 1.54–1.58%) |
| Weight-fixed neutralisation of A (analysis-unit ΔFNR, sign-flip Holm p) | ≈0 | fixed: +0.4 (p=0.50), +2.6 (p<0.001), +0.7 (p=0.20), −1.2 (p=0.20); recalibrated: −2.2 (p=0.25), −16.1, −20.4, −41.6 (p<0.001) |
| B−D behaviour added value (analysis-unit ΔFNR) | ≈0 (SNN −41.8) | −20.9 to −44.0 (SNN −50.4) |
| float32 precast − precision-safe (A) | ≈0 (SNN −11.8) | CNN −1.9, Dense −3.9, LSTM +4.1, SNN −5.0 |
