# v1.2 Edge-IIoTset early-stopping fix 

This root is a copy of temporal_audit_temporal_upgrade_e2e_v1_2. Everything is identical (protocol hash, prepared data, splits,
scalers, baselines, CIC/ToN runs) except:
- All Edge-IIoTset neural runs (7 conditions × 4 models × 3 seeds) were retrained by the ORIGINAL v1.2 code
  (`_run_one_neural`), with ONE change: early stopping uses a fixed random 15% hold-out of fit_normal
  (RandomState(280000 + seed)), and the model trains on the remaining 85%. This is the FITVAL variant of script 28.
- The train, intervene and analyze stages were then re-run with the original v1.2 code (driver: script 33).
- The original Edge runs and stage markers are kept in `_superseded/`.
CIC/ToN results must be bit-identical to v1.2; this is checked by script 33.
