# Temporal Shortcut Audit for Reconstruction-Based IoT Intrusion Detection

Reproducibility package for the study:

**When Absolute Capture Time Acts as Novelty: Out-of-Support Timestamp Shortcuts in Reconstruction-Based IoT Intrusion Detection**

This repository contains source code, protocol metadata, manifests, and analysis outputs used to audit whether absolute packet capture time acts as an acquisition-derived shortcut in reconstruction-based IoT/IIoT intrusion detection.

## Evidence Structure

This repository provides the reproducibility materials for the primary temporal-shortcut audit, the audit-informed validation, and four additional evidence blocks introduced during peer-review revision. These blocks include out-of-support mechanism tests, early-stopping repair, evaluation of additional attack families, and chronological deployment replay on a Raspberry Pi 5:

1. **Experiment I — Frozen primary evidence**  
   Controlled temporal intervention using frozen repeated-cross-validation models.  
   The primary comparison is λ=0 versus λ=1 across 3 datasets, 4 architectures, 3 seeds, and 10 folds, giving 360 seed-fold runs.

2. **Experiment II — Audit-informed validation**  
   Separated fitting, early stopping, calibration, and testing with q-only scoring, A/B/C/D representations, PCA reference, and fixed-weight timestamp controls.

3. **Post-hoc mechanistic strengthening**  
   Temporal-support analysis, intervention-response analysis, acquisition-counter/noise placebo, threshold sensitivity, and translation-based checks.

## Datasets

The study evaluates:

- Edge-IIoTset
- CICIoT2023
- ToN-IoT

Original datasets are not redistributed in this repository.

## Repository Structure

```text
src/
manifests/
results/
reproducibility/

## Revision analyses (2026-11)

Post-review analyses are in `src/revision/` (scripts 25–40) and `results/revision/`. They include pre-specified
out-of-support mechanism tests, an Edge-IIoTset early-stopping repair (Edge-IIoTset Experiment II models retrained;
intervention and analysis stages re-run for all datasets), a timestamp-shifted replay on a Raspberry Pi 5, further
attack families, and the supervised-classifier comparison. See `results/revision/README.md`.
