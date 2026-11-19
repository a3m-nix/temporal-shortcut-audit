# Exp II Edge-IIoTset numbers: original v1.2 vs edgefix (script 33/34)

Identity check (non_edge_identity_check.csv): every CICIoT2023 and ToN-IoT row in all 28 v1.2 and script-24 tables is
bit-identical. The only exception is signflip_p_holm in two statistics tables, because Holm is taken over 12
dataset × model tests that include Edge. No significance decision changed.

| Manuscript quantity (Edge-IIoTset) | Original v1.2 | Edgefix |
|---|---|---|
| Neural ROC-AUC (all conditions, non-SNN) | 0.56–0.68 | 0.98–0.999 |
| B=q FNR at P99 | 61.5–62.0% (SNN 16.8%) | 0.08–14.6% (SNN 4.7%) |
| OR ensemble A → B recall (FPR) | 39.38% → 83.50% (2.86% → 3.55%) | 91.40% → 99.93% (1.54% → 1.86%) |
| A → B detected-by-all → missed-by-all | 0 of 24,000 | 0 of 24,000 (467 reverse) |
| P95 B−A (CNN/Dense/LSTM; SNN) | −1.3 to +3.7; −52.7 | −3.3 to −22.5; −32.7 |
| P99 B−A (CNN/Dense/LSTM; SNN) | −0.4 to +0.7; −45.3 | −12.9 to −26.6; −49.5 |
| P99.9 | ≈62% FNR for A and B (1.4 tail observations) | unchanged (degenerate tail) |
| Weight-fixed neutralisation of A, fixed threshold (ΔFNR) | ≈0 | CNN +49.4, Dense +36.7, LSTM +29.1, SNN +8.3 pp (Holm p < 0.001) |
| B−D behaviour added value (analysis-unit ΔFNR) | ≈0 (SNN −41.8) | −20.9 to −44.0 (SNN −50.4) |
| float32 precast − precision-safe (A) | ≈0 (SNN −11.8) | CNN −1.9, Dense −3.9, LSTM +4.1, SNN −5.0 |
