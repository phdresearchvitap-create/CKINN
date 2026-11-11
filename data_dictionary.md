# Data dictionary and harmonization map

Generated outputs of `harmonize.py`: `rich_harmonized.csv`, `nagala_harmonized.csv`. Missing stays missing; nothing is imputed.

## Harmonized variables
| Variable | RICH source column | NAGALA source column | Unit / coding | Missing (RICH / NAGALA) |
| --- | --- | --- | --- | --- |
| cohort, id | `id` | `Patient Number` | | 0% / 0% |
| site | `site` (15 sites; 0 and 16 tiny) | single centre = 1 | | 0% / 0% |
| age | `Age (y)` | `Baseline Age, yrs` | years | 0% / 0% |
| sex_male | `Gender(1, male; 2, female)` | `Sex, 1:Men, 2:Women` | 1 = male, 0 = female | 0% / 0% |
| bmi | `BMI(kg/m2)` | `Baseline BMI, kg/m2` | kg/m2 | 0% / 0% |
| sbp, dbp | `SBP(mmHg)`, `DBP(mmHg)` | `Baseline Systolic/Diastolic blood pressure, mmHg` | mmHg | 0% / 0% |
| fpg | `FPG (mmol/L)` | `Baseline Fasting plasma glucose, mmol/L` | mmol/L | 0% / 0% |
| tc | `Cholesterol(mmol/L)` | `Baseline Total Cholesterol, mmol/L` | mmol/L | 2.3% / 0% |
| tg | `Triglyceride(mmol/L)` | `Baseline Triglycerides, mmol/L` | mmol/L | 2.3% / 0% |
| hdl | `HDL-c(mmol/L)` | `Baseline HDL-cholesterol, mmol/L` | mmol/L | 44.6% / 0% |
| alt, ast | `ALT(U/L)`, `AST(U/L)` | `Baseline ALT, IU/L`, `Baseline AST, IU/L` | U/L | 0.8% and 58.2% / 0% |
| smoke | `smoking status(1 current, 2 ever, 3 never)` | `Baseline Smoking status, 1 never, 2 past, 3 current` | 0 never, 1 former, 2 current (codes are reversed between cohorts) | 71.6% / 0% |
| famhx | `family histroy of diabetes` | not collected | 0/1 | 0% / n.a. |
| hba1c, exercise, waist, fatty_liver, ggt | not collected | `Baseline HbA1c, %`, `Habit of exercise 0/1`, `Waist circumference(WC), cm`, `Fatty liver 0/1`, `GGT` | % / 0-1 / cm / 0-1 / IU/L | n.a. / 0% |
| time_y | `year of followup` | `Follow up duration, days` / 365.25 | years | 0% / 0% |
| event | `censor of diabetes at followup` = 1 | `Incident DM 0/1` | 1 = incident diabetes | 0% / 0% |
| event_dx_only | `Diabetes diagnosed during followup` (blank = 0) | n.a. | 1 = diagnosed | RICH only |

## Known issues
1. NAGALA sex is 0/1 in the data although the codebook says 1/2. Male = 1 is inferred (incidence 3.4% vs 1.2%); confirm against the source record (decision D2).
2. Smoking codes are reversed between cohorts; `harmonize.py` maps both to 0/1/2.
3. RICH smoking is missing for 4% to 99% depending on site, and HDL and AST missingness relates to the outcome (event rate 2.3% when HDL recorded vs 1.6% when missing). Missingness is informative.
4. The RICH outcome is partly defined by FPG at the final visit, and baseline FPG is the strongest predictor. Always show tiers without FPG.
5. RICH follow-up is at least 2.0 years and events appear only after 2.0 years, so a 2-year binary label is empty; 3-year labels leave 50.1% of RICH censored before the horizon.
6. Both cohorts exclude baseline diabetes, so they cannot support cross-sectional detection of undiagnosed diabetes.
7. NAGALA provides both mg/dL and mmol/L columns; use mmol/L only (the conversions agree exactly).
