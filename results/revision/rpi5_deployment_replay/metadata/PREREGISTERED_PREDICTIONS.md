# RPi5 deployment replay — absolute-time detectors on an edge device

Written: 2026-11-19, BEFORE any model was exported or run on the device. The SHA-256 is in PREREGISTERED_PREDICTIONS.sha256.

## Question
When the same packets arrive later than the training capture (as they always do in deployment),
what happens to a reconstruction IDS that uses absolute time, compared with an identical one that does not?
What does the shortcut cost on edge hardware?

## Models (frozen; no retraining)
Exp II audit-informed models, seed 0, all four AEs: A = q+t and B = q.
- CICIoT2023 and ToN-IoT: the v1.2 runs.
- Edge-IIoTset: the repaired FITVAL runs (script 28).
Exported unchanged: weights, fit-normal MinMax scaler and P99 calibration-normal q-threshold.

## Replay stream
The Exp II test partition (test-normal + attack) in chronological order (by frame.time_epoch).
Deployment horizon H: every timestamp is replaced by t + offset_H, which keeps relative timing,
where offset_H moves the first test packet to (fit_end + H).
H ∈ {as-recorded (offset 0), +1 h, +1 d, +7 d, +30 d, +365 d, NOW}. NOW is the device wall clock at run time.
The B input does not contain t, so B is evaluated once and is identical across H by construction.
Inference runs on the RPi5 CPU in micro-batches of 256 rows, in stream order, with the fixed threshold.
Metrics per (dataset, model, cond, H): FPR on test-normal and FNR on attacks.

## Power and latency (FNIRSI FNB-58)
ONE logger process (FNIRSI_NO_RESET=1) for the whole session. No preflight and no reconnects.
Stopped with SIGINT at the end. Per (dataset, model, cond): 20 s idle window, then a ≥ 20 s
active window of repeated batch-256 inference on the replay stream, aligned by Unix timestamp.
The session aborts without retry if no data line arrives within 15 s of start.

## Predictions
D1 (sanity) B metrics are identical across H.
D2 A FPR is non-decreasing in H from +1 d onward, for every dataset and model (tolerance 1 pp).
D3 At H = +365 d and at NOW, A FPR ≥ 90% for at least 3 of 4 models in every dataset (an "alert flood"),
   while B FPR stays at its as-recorded value (≈ 1–2%).
D4 At H = +1 d, A FPR > B FPR for CICIoT2023 and ToN-IoT, for at least 3 of 4 models each
   (one day ≫ fit span of 3.4 h and 10.2 h). For Edge-IIoTset (fit span 16.4 d) no prediction is made at +1 d.
D5 Edge cost: the A and B per-row latency and active energy per row differ by < 10% for the same architecture.
   The shortcut buys no efficiency.
Each prediction is reported as met / not met. No prediction is changed after the run.
