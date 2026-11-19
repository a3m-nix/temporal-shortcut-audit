# Revision analyses (2026-11)

These blocks were added after peer review. Blocks A, B and D carry pre-specified predictions: each block's
`metadata/PREREGISTERED_PREDICTIONS.md` was written and SHA-256-hashed before the block was run. The verdict for
every prediction, met or not, is in that block's `tables/PREDICTION_VERDICTS.csv`.

| Folder | Script(s) in `src/revision/` | What it tests | Predictions met |
|---|---|---|---|
| `mechanism_oos/` | 26, 27 | Out-of-support (OOS) mechanism on the 360 weight-fixed Exp I models: clipping, class-independent OOS values, permutation, future-normal shift, activations, distance bins | 32 / 37 |
| `edge_earlystop_repair/` | 28, 29, 30 | Edge-IIoTset Exp II re-run with early stopping on a fit-normal hold-out (FITVAL) and a fixed-30-epoch variant | 4 / 6 |
| `exp2_edgefix/` | 33, 34, 35 | Full Exp II pipeline (script 20) re-run for Edge-IIoTset with the FITVAL repair. CIC/ToN results are bit-identical to the original run | recomputation, no predictions |
| `rpi5_deployment_replay/` | 31, 32, `rpi5/rpi_runner.py` | Chronological replay of A=q+t vs B=q on a Raspberry Pi 5 at later deployment horizons | 4 / 4 (phase 1) |
| `supervised_comparators/` | 25 | XGBoost, RF, LR and MLP under the λ intervention (post-hoc) | post-hoc |

Notes
- The RPi5 runner also contains a power-measurement phase. Its outputs are not part of this release and are not used.
- Model weights, SavedModels and per-row predictions are not distributed (see `.gitignore`). They can be regenerated
  with the scripts and the manifests in `manifests/`.
- Path and label sanitization follows `reproducibility/protocol/REVISION_PUBLIC_SANITIZATION.txt`.
