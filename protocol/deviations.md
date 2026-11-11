# Deviations from protocol v1.0

None recorded at lock time. Add one entry per deviation: date, what changed, why, and whether results had been seen.

## D-1 (2026-11-11, before any LOSO result was inspected)
Site 14 of RICH has maximum follow-up 2.92 years, shorter than the 3-year horizon, so no 3-year metric can be computed when it is the test site. It stays in every training set and is not a test fold, leaving 12 test sites instead of the 13 stated in section 8 of the locked protocol. Effect on claims: none (C2 and C3 rest on the cross-cohort analyses; LOSO is descriptive).

## D-2 (2026-11-11, before any R1, LOSO, shift or cross-cohort result was inspected)
The kit's `scripts/run_experiments.py` dropped every row with a missing predictor in all analyses (complete-case), which for tier T4 in RICH keeps 49,695 of 211,833 participants and contradicts section 5 of the locked protocol (training-fold median fill plus missingness indicators; complete-case is sensitivity analysis S2 only). The code was corrected to use all rows, and S2 was added as its own unit (`sensS2`, RICH T4 complete cases, 1 repeat of 5-fold). This is a correction of the code to the protocol, not a change to the protocol. Effect on claims: none; the analysis now matches what was pre-specified.
