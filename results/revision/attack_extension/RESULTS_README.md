# Attack-family extension: ToN-IoT DoS and Edge-IIoTset non-flood (revision, R5.2)

- Scripts: `36-extract-edge-nonflood-v1.py` (PCAP extraction; the DDoS-HTTP re-extraction matches all 34 parent features
  of 229,142 rows exactly), `37-attack-extension-expI-v1.py` (samples + Experiment I protocol), `38-aggregate-attack-extension-v1.py`.
- Pre-specified predictions: `metadata/PREREGISTERED_PREDICTIONS.md` (SHA-256 4b8e417c…f7e). Addendum 01 (b4d7fd2d…ea83),
  written before training: ToN DoS uses equal-count time bins, because DoS occurs in four bursts and equal-width bins were empty.
- 240 trainings = 2 samples × 4 AEs × 3 seeds × 10 folds. Each sample has 10,000 attack rows and a balanced 10,000-row normal draw.

## Verdicts: 3 of 4 met
| | Result |
|---|---|
| X1 ToN DoS ΔFNR > +20 pp | **not met**. ΔFNR = +4.6 / +4.3 / +4.5 pp (CNN / Dense / SNN; corrected CI > 0, Holm-8 p < 0.001) and +18.4 pp for LSTM (CI −1.0 to +37.9). Direction as predicted; size smaller, because packet features alone already detect DoS (λ0 FNR about 4%). |
| X2 ToN DoS clip ≈ λ0 | met (|diff| ≤ 3.5 pp, Holm n.s.) |
| X3 Edge non-flood null | met (ΔFNR −1.0 to +1.8 pp; all CIs include 0) |
| X4 Edge non-flood clip = λ1 | met (identical in all 4 models; attacks in support) |

The time contribution on ToN DoS is concentrated in the first DoS burst: per-bin ΔFNR is 12–37 pp in bins 00–01
and about 1 pp in later bins (LSTM-AE about 17 pp). Edge non-flood families are hard for packet-reconstruction AEs
(λ1 FNR 32–65%), with no time effect in any family (per-family |ΔFNR| ≤ 5.4 pp; the largest is MITM with CNN-AE, about 100 attack rows per test fold).

Tables: `tables/attack_extension_summary.csv`, `tables/attack_extension_by_family.csv`, `tables/PREDICTION_VERDICTS.csv`.
