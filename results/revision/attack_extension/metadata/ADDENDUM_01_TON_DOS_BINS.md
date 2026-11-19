# Addendum 01 — ToN-IoT DoS sampling bins (written 2026-11-19, before any model was trained)

The pre-specified rule (10 equal-width time bins × 1,000 rows) cannot be applied. ToN-IoT DoS occurs in four bursts
(2,588,124 / 1,500,000 / 1,500,000 / 1,500,000 packets) separated by gaps longer than 5 minutes. As a result, 4 of the
10 equal-width bins are empty (counts: 2151455, 1632365, 304304, 0, 0, 338791, 1161209, 0, 0, 1500000).

Change: 10 equal-count (quantile) time bins, with 1,000 rows drawn at random per bin (NumPy seed 42) = 10,000 attack
rows. All four bursts are represented. Everything else, including all predictions X1–X4, is unchanged.
This addendum was written after the sample-building step failed and before any training or evaluation.
