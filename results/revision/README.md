# Revision analyses (2026-11)

These blocks were added after peer review. The mechanism, repair, attack-extension and deployment blocks carry pre-specified predictions: each block's
`metadata/PREREGISTERED_PREDICTIONS.md` was written and SHA-256-hashed before the block was run. The verdict for
every prediction, met or not, is in that block's `tables/PREDICTION_VERDICTS.csv`.

| Folder | Script(s) in `src/revision/` | What it tests | Predictions met |
|---|---|---|---|
| `mechanism_oos/` | 26, 27 | Out-of-support (OOS) mechanism on the 360 weight-fixed Exp I models: clipping, class-independent OOS values, permutation, future-normal shift, activations, distance bins | 32 / 37 |
| `edge_earlystop_repair/` | 28, 29, 30 | Edge-IIoTset Exp II re-run with early stopping on a fit-normal hold-out (FITVAL) and a fixed-30-epoch variant | 4 / 6 |
| `exp2_edgefix/` | 33, 34, 35 | Exp II (script 20) with the FITVAL repair: Edge-IIoTset models retrained, intervention and analysis stages re-run for all datasets. CIC/ToN models were not retrained; their rows in the result tables equal the original run (`comparison/non_edge_identity_check.csv`; Holm-adjusted p differs in two statistics tables because the Holm family includes Edge-IIoTset) | recomputation, no predictions |
| `rpi5_deployment_replay/` | 31, 32, `rpi5/rpi_runner.py` | Replay of held-out normal and attack rows on a Raspberry Pi 5 with timestamps shifted to later deployment horizons (A=q+t vs B=q); not live traffic | D1–D4 (phase 1): 4 / 4; D5 (phase 2, cost) is not part of this release |
| `supervised_comparators/` | 25 | XGBoost, RF, LR and MLP under the λ intervention (post-hoc) | post-hoc |
| `attack_extension/` | 36, 37, 38 | ToN-IoT DoS and ten Edge-IIoTset non-flood families under the Experiment I protocol (Edge non-flood packets re-extracted from the official PCAPs; extraction validated bit-for-bit against the parent table) | 3 / 4 |
| `figures_print/` | 39, 40 | Print-size figures used in the revised manuscript (Figs. 1, 3, 5, 6, 7) | — |

Notes
- The RPi5 runner also contains a power-measurement phase. Its outputs are not part of this release and are not used.
- Model weights, SavedModels and per-row predictions are not distributed (see `.gitignore`). They can be regenerated
  with the scripts and the manifests in `manifests/`.
- Path and label sanitization follows `reproducibility/protocol/REVISION_PUBLIC_SANITIZATION.txt`.
