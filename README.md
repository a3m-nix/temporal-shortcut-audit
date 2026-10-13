# Temporal Shortcut Audit for Reconstruction-Based IoT Intrusion Detection

Reproducibility package for the study:

**When Absolute Capture Time Behaves Like a Label: Temporal Shortcuts in Reconstruction-Based IoT Intrusion Detection**

This repository contains source code, protocol metadata, manifests, and analysis outputs used to audit whether absolute packet capture time acts as an acquisition-derived shortcut in reconstruction-based IoT/IIoT intrusion detection.

## Evidence Structure

The repository preserves three evidence layers:

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
