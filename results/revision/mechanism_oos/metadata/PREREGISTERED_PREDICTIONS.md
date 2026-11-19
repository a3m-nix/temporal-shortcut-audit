# Out-of-support (OOS) mechanism tests — pre-specified predictions

Written: 2026-11-19, BEFORE any of the analyses below were run.
Freeze: the SHA-256 of this file is recorded in PREREGISTERED_PREDICTIONS.sha256.

## Scope
Inference only, on the 360 saved weight-fixed λ-intervention models:
- ae3_3datasets_30fold_v1: Dense, LSTM and CNN; 3 datasets × 3 seeds × 10 folds
- snn_3datasets_30fold_v1: SNN; 3 datasets × 3 seeds × 10 folds

Everything is held fixed: the same splits, scalers, weights and P99 validation-normal
non-epoch threshold. The score is the non-epoch MSE over 33 components (the primary
endpoint score). Only the scaled frame.time_epoch input is manipulated.

## Mechanism under test
Normal-only autoencoders flag attacks because the attack timestamp lies outside the
fit-normal support [0,1]. The out-of-support input inflates reconstruction error on the
non-time features. Under this mechanism the effect is novelty-driven and class-agnostic:
any row that receives an OOS timestamp is flagged, whether it is normal or attack.

## Arms
- L1    : original timestamp (λ=1), the reference
- L0    : training-normal median (λ=0)
- CLIP  : t clipped to [0,1]. This keeps the in-support ordering and removes the OOS magnitude.
- CONST(d), side ∈ {above, below}: EVERY test row (normal and attack) gets t = 1+d (above)
  or t = −d (below), with d ∈ {0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 100}.
  This is the class-independent OOS placebo (R2 M1a, R7).
- PERM  : the test-row timestamps permuted across all test rows. Same marginal values,
  class link broken. Fixed permutation seed per run.
- FUT(d): test-normal rows only are moved to t = 1+d, d ∈ {0.1, 0.5, 1, 2}, while attack rows
  keep their original t. This is the deployment / future-traffic test (R2 M6).
- ACT   : mean L2 norm of the first hidden layer (Dense: encoder; LSTM: enc_lstm1; CNN: conv1;
  SNN: batch_norm output) for attack rows at L1 vs L0 and for normal rows at L1 (R2 M1c).
- DIST  : over the λ grid, per-attack detection binned by its OOS distance at that λ.

## Predictions (CIC = CICIoT2023, ToN = ToN-IoT, Edge = Edge-IIoTset; non-SNN = Dense, LSTM, CNN)
P1 CLIP ≈ L0: for CIC and ToN, all four models, |mean FNR_CLIP − mean FNR_L0| < 5 pp,
   and the FNR_CLIP − FNR_L1 Nadeau–Bengio CI lies above 0.
   For Edge, CLIP is approximately a no-op: |FNR_CLIP − FNR_L1| < 1 pp.
P2 Class-agnostic OOS: under CONST(d) above, the normal flag rate (FPR) increases with d
   for every dataset, Edge included. At d = 10, FPR ≥ 50% for at least 2 of the 3 non-SNN models
   per dataset.
P3 PERM: for CIC and ToN, FPR rises from ≈1% to > 25% (non-SNN models).
P4 Deployment cost: FUT(d=1) FPR > 2 × the L1 FPR for a majority of the 12 dataset-model cells.
P5 Activation: for CIC and ToN non-SNN models, attack first-layer norm at L1 / L0 > 3;
   for Edge, the ratio is within [0.8, 1.25].
P6 Distance, not λ: the attack detection rate increases monotonically across distance bins
   [0], (0,0.1], (0.1,1], (1,10], >10, pooled over λ, for CIC and ToN non-SNN models.
SNN: directionally the same, but weaker or saturating effects are expected because the spike
   non-linearity bounds the encoder output. SNN is reported separately; it does not decide P1–P6.

Every prediction is reported as met / not met. No prediction is changed after the run.
