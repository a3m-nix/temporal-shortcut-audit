# v1.2 Edge-IIoTset early-stopping fix 

This root is a copy of temporal_audit_temporal_upgrade_e2e_v1_2. Everything is identical (protocol hash, prepared data, splits,
scalers, baselines, CIC/ToN runs) except:
- All Edge-IIoTset neural runs (7 conditions × 4 models × 3 seeds) were retrained by the ORIGINAL v1.2 code
  (`_run_one_neural`), with ONE change: early stopping uses a fixed random 15% hold-out of fit_normal
  (RandomState(280000 + seed)), and the model trains on the remaining 85%. This is the FITVAL variant of script 28.
- The train, intervene and analyze stages were then re-run with the original v1.2 code (driver: script 33).
- The original Edge runs and stage markers are kept in `_superseded/`.
CIC/ToN results are checked against v1.2 by script 34 (`edgefix_comparison/non_edge_identity_check.csv`).

Correction (2026-11-19): the Edge-IIoTset intervention outputs copied from v1.2 were first reused by the resumable
intervene stage. They were moved to `_superseded/stale_edge_interventions_fix_2026-11-19/` and recomputed with the
retrained models; intervene, analyze, script 24 and script 34 were re-run (logs in `logs/edgefix/`).
