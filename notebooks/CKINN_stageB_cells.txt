# %%
# CELL 7 - structure of the raw tables: identifiers, duplicates, missingness, category codes     [status L; protocol sections 2 to 4]
# Why: establishes one row per participant and exposes coding problems before anything is harmonized. This cell also
# gives every raw column a short working name (by its Excel column letter); no value is changed.
# session guard (same three lines in every cell): restores the earlier set-up cells by itself if the kernel was restarted
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 7)
require_gate("A")
A = Audit(7, "Structure, identifiers, missingness, category codes")

# short name -> (Excel column letter, a word that must appear in the published header of that column)
NAG_COLS = {"id": ("A", "Patient Number"), "sex": ("B", "Sex"), "age": ("C", "Age"), "ethanol": ("D", "Ethanol"),
            "fatty_liver": ("E", "Fatty liver"), "bmi": ("F", "BMI, kg/m2"), "waist": ("G", "Waist"), "wc_high": ("H", "WC"),
            "bmi25": ("I", "BMI"), "phenotype": ("J", "Obesity phenotype"), "days": ("K", "Follow up duration"),
            "alt": ("L", "ALT"), "ast": ("M", "AST"), "weight": ("N", "Body Weight"), "exercise": ("O", "exercise"),
            "ggt": ("P", "GGT"), "hdl_mg": ("Q", "HDL-cholesterol, mg/dL"), "hdl": ("R", "HDL-cholesterol, mmol/L"),
            "tc_mg": ("S", "Total Cholesterol, mg/dL"), "tc": ("T", "Total Cholesterol, mmol/L"),
            "tg_mg": ("U", "Triglycerides, mg/dL"), "tg": ("V", "Triglycerides, mmol/L"), "hba1c": ("W", "HbA1c, %"),
            "hba1c_mmol": ("X", "HbA1c, mmol/mol"), "alcohol_cat": ("Y", "Alcohol consumption"), "smoke": ("Z", "Smoking status"),
            "fpg_mg": ("AA", "glucose, mg/dl"), "fpg": ("AB", "glucose, mmol/L"), "sbp": ("AC", "Systolic"),
            "dbp": ("AD", "Diastolic"), "event": ("AE", "Incident DM")}
RIC_COLS = {"id": ("A", "id"), "age": ("B", "Age"), "sex": ("C", "Gender"), "site": ("D", "site"), "height": ("E", "height"),
            "weight": ("F", "weight"), "bmi": ("G", "BMI"), "sbp": ("H", "SBP"), "dbp": ("I", "DBP"), "fpg": ("J", "FPG (mmol/L)"),
            "tc": ("K", "Cholesterol"), "tg": ("L", "Triglyceride"), "hdl": ("M", "HDL-c"), "ldl": ("N", "LDL"),
            "alt": ("O", "ALT"), "ast": ("P", "AST"), "bun": ("Q", "BUN"), "ccr": ("R", "CCR"),
            "fpg_final": ("S", "FPG of final visit"), "dm_dx": ("T", "Diabetes diagnosed during followup"),
            "event": ("U", "censor of diabetes at followup"), "time_y": ("V", "year of followup"),
            "smoke": ("W", "smoking status"), "drink": ("X", "drinking status"), "famhx": ("Y", "of diabetes")}


def excel_index(letters):
    k = 0
    for ch in letters:
        k = k * 26 + (ord(ch) - 64)
    return k - 1


def short_frame(cohort, cols):
    """The raw table with short column names. Returns (frame, list of columns whose header lacks the expected word)."""
    raw, out, bad = RAW[cohort], {}, []
    for short, (letter, word) in cols.items():
        j = excel_index(letter)
        if word.lower() not in str(raw.columns[j]).lower():
            bad.append(f"{letter}:{short}")
        out[short] = raw.iloc[:, j].to_numpy()
    return pd.DataFrame(out), bad


NAG, _bad_n = short_frame("NAGALA", NAG_COLS)
RIC, _bad_r = short_frame("RICH", RIC_COLS)
NAG["time_y"] = NAG["days"] / 365.25           # follow-up in years (protocol section 3); the only derived column here
A.check("NAGALA: all 31 columns named, each header contains its expected word", len(NAG_COLS) == 31 and not _bad_n, _bad_n)
A.check("RICH: all 25 columns named, each header contains its expected word", len(RIC_COLS) == 25 and not _bad_r, _bad_r)

# 1. identifiers and duplicates
for name, d in (("NAGALA", NAG), ("RICH", RIC)):
    A.check(f"{name}: identifier is a positive whole number, never missing, never repeated",
            d["id"].notna().all() and (d["id"] > 0).all() and (d["id"] % 1 == 0).all() and d["id"].is_unique, f"{d['id'].nunique():,} unique")
    _dup = int(d.drop(columns="id").duplicated().sum())
    A.check(f"{name}: no two participants have identical values in every other column", _dup == 0, _dup)

# 2. missingness per column
MISSING_EXPECTED = {"NAGALA": {"hdl_mg": 11},
                    "RICH": {"height": 2, "sbp": 23, "dbp": 24, "tc": 4854, "tg": 4887, "hdl": 94562, "ldl": 93421, "alt": 1782,
                             "ast": 123290, "bun": 21551, "ccr": 11175, "fpg_final": 19, "dm_dx": 210529, "smoke": 151603, "drink": 151603}}
_miss_rows = []
for name, d in (("NAGALA", NAG), ("RICH", RIC)):
    m = d.drop(columns=[c for c in ("time_y",) if c in d and name == "NAGALA"]).isna().sum()
    for c, k in m.items():
        _miss_rows.append({"cohort": name, "column": c, "missing": int(k), "percent": round(100 * k / len(d), 2)})
    _got = {c: int(k) for c, k in m.items() if k > 0}
    A.check(f"{name}: missing counts per column equal the audited file", _got == MISSING_EXPECTED[name],
            {c: (_got.get(c), MISSING_EXPECTED[name].get(c)) for c in set(_got) | set(MISSING_EXPECTED[name]) if _got.get(c) != MISSING_EXPECTED[name].get(c)})
MISSING = pd.DataFrame(_miss_rows)
MISSING.to_csv(DIR["audit"] / "missingness_raw.csv", index=False)
print("\n  Columns with missing values:")
print(MISSING[MISSING["missing"] > 0].to_string(index=False))
A.check("RICH: HDL 44.64% and AST 58.20% missing", (round(100 * RIC["hdl"].isna().mean(), 2), round(100 * RIC["ast"].isna().mean(), 2)) == (44.64, 58.20))
A.check("RICH: smoking and drinking are missing for the same 71.57% of participants",
        (RIC["smoke"].isna() == RIC["drink"].isna()).all() and round(100 * RIC["smoke"].isna().mean(), 2) == 71.57)


# 3. category codes
def codes(s):
    return {(int(k) if float(k).is_integer() else float(k)): int(v) for k, v in s.value_counts(dropna=True).sort_index().items()}


CODES = {"NAGALA": {c: codes(NAG[c]) for c in ("sex", "fatty_liver", "wc_high", "bmi25", "phenotype", "exercise", "alcohol_cat", "smoke", "event")},
         "RICH": {c: codes(RIC[c]) for c in ("sex", "site", "smoke", "drink", "famhx", "dm_dx", "event")}}
print("\n  Category codes (value: number of participants):")
for name in CODES:
    for c, tab in CODES[name].items():
        print(f"   {name:6s} {c:12s} {tab}")
A.check("NAGALA: sex is coded 0/1 (the header says 1/2), 7,034 and 8,430", CODES["NAGALA"]["sex"] == {0: 7034, 1: 8430})
A.check("RICH: sex is coded 1/2, 116,123 and 95,710", CODES["RICH"]["sex"] == {1: 116123, 2: 95710})
A.check("NAGALA: 373 events; RICH: 4,174 events; both coded 0/1",
        CODES["NAGALA"]["event"] == {0: 15091, 1: 373} and CODES["RICH"]["event"] == {0: 207659, 1: 4174})
A.check("RICH: 15 site codes; sites 0 and 16 are tiny (61 and 247)",
        len(CODES["RICH"]["site"]) == 15 and CODES["RICH"]["site"][0] == 61 and CODES["RICH"]["site"][16] == 247, sorted(CODES["RICH"]["site"]))
A.check("RICH: smoking and drinking coded 1 to 3; family history 0/1 with 4,344 positive; diagnosis column holds only the value 1",
        set(CODES["RICH"]["smoke"]) == {1, 2, 3} and set(CODES["RICH"]["drink"]) == {1, 2, 3}
        and CODES["RICH"]["famhx"] == {0: 207489, 1: 4344} and CODES["RICH"]["dm_dx"] == {1: 1304})
A.check("NAGALA: smoking coded 1 to 3, alcohol category 1 to 4, binary flags 0/1",
        set(CODES["NAGALA"]["smoke"]) == {1, 2, 3} and set(CODES["NAGALA"]["alcohol_cat"]) == {1, 2, 3, 4}
        and all(set(CODES["NAGALA"][c]) == {0, 1} for c in ("fatty_liver", "wc_high", "bmi25", "exercise")))
_ph_out = {k: v for k, v in CODES["NAGALA"]["phenotype"].items() if not 1 <= k <= 8}
A.check("NAGALA: 'Obesity phenotype 1-8' holds 7 codes outside 1 to 8 (6,943 participants); the column is not used",
        sorted(_ph_out) == [0, 10, 11, 100, 101, 110, 111] and sum(_ph_out.values()) == 6943, _ph_out)
_ = A.close(outputs={"n": {"NAGALA": len(NAG), "RICH": len(RIC)}, "missing": MISSING_EXPECTED, "codes": {k: {c: {str(a): b for a, b in t.items()} for c, t in v.items()} for k, v in CODES.items()}})
keep_for_session(7)

# %%
# CELL 8 - zero-coded and implausible values, by site                                           [status D; deviation D-3]
# Why: a triglyceride, HDL, ALT or AST of exactly zero is not a possible result; it is a code for "not measured". Left
# in, it would be read as the lowest value in the cohort. This cell only counts and reports. Nothing is deleted or
# changed here; the D-3 rule (zero becomes missing) is applied at harmonization in cell 16.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 8)
A = Audit(8, "Zero-coded and implausible values", status="D")

MEASURED = {"NAGALA": ["age", "bmi", "waist", "weight", "alt", "ast", "ggt", "hdl_mg", "hdl", "tc_mg", "tc", "tg_mg", "tg",
                       "hba1c", "hba1c_mmol", "fpg_mg", "fpg", "sbp", "dbp", "days"],
            "RICH": ["age", "height", "weight", "bmi", "sbp", "dbp", "fpg", "tc", "tg", "hdl", "ldl", "alt", "ast", "bun", "ccr",
                     "fpg_final", "time_y"]}
ZERO_EXPECTED = {"NAGALA": {"hdl": 11}, "RICH": {"tg": 860, "hdl": 102, "ldl": 1, "alt": 2, "ast": 1}}
D3_VARIABLES = ["tg", "hdl", "alt", "ast"]     # deviation D-3 names exactly these four

ZEROS = {}
for name, d in (("NAGALA", NAG), ("RICH", RIC)):
    ZEROS[name] = {c: int((d[c] == 0).sum()) for c in MEASURED[name] if (d[c] == 0).any()}
    A.check(f"{name}: exact zeros per measured variable equal the audited file", ZEROS[name] == ZERO_EXPECTED[name], ZEROS[name])
    A.check(f"{name}: no measured variable has a negative value", not any((d[c] < 0).any() for c in MEASURED[name]))
A.check("NAGALA: the 11 zero HDL values (mmol/L) are exactly the 11 rows whose HDL in mg/dL is blank",
        ((NAG["hdl"] == 0) == NAG["hdl_mg"].isna()).all())

print("\n  RICH: zero-coded values by site (a zero concentrated in a few sites is a code, not a measurement)")
ZERO_BY_SITE = RIC.groupby("site").agg(n=("id", "size"), tg_zero=("tg", lambda s: int((s == 0).sum())),
                                       hdl_zero=("hdl", lambda s: int((s == 0).sum())), alt_zero=("alt", lambda s: int((s == 0).sum())),
                                       ast_zero=("ast", lambda s: int((s == 0).sum()))).reset_index()
print(ZERO_BY_SITE.to_string(index=False))
ZERO_BY_SITE.to_csv(DIR["audit"] / "rich_zero_by_site.csv", index=False)
A.check("RICH: 637 of the 860 zero triglycerides are at site 2", int(ZERO_BY_SITE.loc[ZERO_BY_SITE["site"] == 2, "tg_zero"].iloc[0]) == 637)

D3_CELLS = {name: {c: ZEROS[name].get(c, 0) for c in D3_VARIABLES} for name in ZEROS}
_rows_r = int((RIC[D3_VARIABLES] == 0).any(axis=1).sum())
_rows_n = int((NAG[D3_VARIABLES] == 0).any(axis=1).sum())
print(f"\n  D-3 will set to missing: RICH {sum(D3_CELLS['RICH'].values())} values in {_rows_r:,} participants "
      f"({100 * _rows_r / len(RIC):.2f}%); NAGALA {sum(D3_CELLS['NAGALA'].values())} values in {_rows_n} participants. No row is removed.")
print(f"  Events among them: RICH {int(RIC.loc[(RIC[D3_VARIABLES] == 0).any(axis=1), 'event'].sum())}, "
      f"NAGALA {int(NAG.loc[(NAG[D3_VARIABLES] == 0).any(axis=1), 'event'].sum())}.")
print("  Outside D-3: one zero LDL in RICH. LDL is in no locked tier; it is only reported here.")
A.check("D-3 covers 965 values in RICH and 11 in NAGALA", (sum(D3_CELLS["RICH"].values()), sum(D3_CELLS["NAGALA"].values())) == (965, 11))

# values outside wide physiological limits: flagged and counted only, never altered (limits are for flagging, not for exclusion)
LIMITS = {"age": (18, 100), "height": (120, 210), "weight": (30, 200), "bmi": (12, 70), "waist": (40, 160), "sbp": (60, 260),
          "dbp": (30, 160), "fpg": (2.5, 7.0), "fpg_final": (2.5, 35), "tc": (1.0, 20), "tg": (0.1, 35), "hdl": (0.2, 5.0),
          "ldl": (0.2, 12), "alt": (1, 2000), "ast": (1, 2000), "ggt": (1, 2000), "bun": (0.5, 40), "ccr": (15, 1500), "hba1c": (3.0, 6.5)}
_flag = []
for name, d in (("NAGALA", NAG), ("RICH", RIC)):
    for c, (lo, hi) in LIMITS.items():
        if c in d:
            s = d[c].dropna()
            _flag.append({"cohort": name, "variable": c, "flag_below": lo, "flag_above": hi, "n_below": int((s < lo).sum()),
                          "n_above": int((s > hi).sum()), "minimum": float(s.min()), "maximum": float(s.max())})
FLAGS = pd.DataFrame(_flag)
FLAGS.to_csv(DIR["audit"] / "implausible_values.csv", index=False)
print("\n  Values outside wide physiological limits (flagged only; zeros appear here as 'below'):")
print(FLAGS[(FLAGS["n_below"] > 0) | (FLAGS["n_above"] > 0)].to_string(index=False))
_bmi_gap = (RIC["bmi"] - RIC["weight"] / (RIC["height"] / 100) ** 2).abs()
print(f"\n  RICH: recorded BMI differs from weight/height^2 by more than 1 kg/m2 in {int((_bmi_gap > 1).sum())} participants "
      f"(largest difference {_bmi_gap.max():.1f}). Reported only.")
A.check("no baseline fasting glucose reaches the diabetes threshold of 7.0 mmol/L in either cohort",
        RIC["fpg"].max() < 7.0 and NAG["fpg"].max() < 7.0, (float(RIC["fpg"].max()), round(float(NAG["fpg"].max()), 3)))
_ = A.close(outputs={"zeros": ZEROS, "d3_cells": D3_CELLS, "d3_rows": {"RICH": _rows_r, "NAGALA": _rows_n},
                     "flags": "ckinn_results/audit/implausible_values.csv"})

# %%
# CELL 9 - the outcome: rebuild each event flag from its components                               [status L; protocol section 3]
# Why: the label must be understood before any model sees it. RICH's event flag is a composite of a recorded diagnosis
# and a fasting glucose of 7.0 mmol/L or more at the final visit; this cell proves that, and counts each part.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 9)
A = Audit(9, "Outcome reconstruction")

_dx = RIC["dm_dx"].fillna(0).eq(1)             # diagnosed with diabetes during follow-up
_hi = RIC["fpg_final"] >= 7.0                  # final-visit fasting glucose at or above 7.0 mmol/L (False where it is blank)
_ev = RIC["event"].eq(1)
A.check("RICH: event flag = diagnosis OR final-visit glucose >= 7.0 mmol/L, for every participant", (_ev == (_dx | _hi)).all(),
        f"{int((_ev != (_dx | _hi)).sum())} disagreements")
PARTS = {"diagnosis only": int((_dx & ~_hi).sum()), "glucose only": int((~_dx & _hi).sum()), "both": int((_dx & _hi).sum())}
A.check("RICH: 4,174 events = 770 diagnosis only + 2,870 glucose only + 534 both",
        PARTS == {"diagnosis only": 770, "glucose only": 2870, "both": 534} and int(_ev.sum()) == 4174, PARTS)
A.check("RICH: 1,304 diagnoses and 3,404 final-visit glucose values >= 7.0; none of them in a non-event",
        int(_dx.sum()) == 1304 and int(_hi.sum()) == 3404 and not ((_dx | _hi) & ~_ev).any())
A.check("RICH: the 19 participants with no final-visit glucose are all diagnosed events",
        int(RIC["fpg_final"].isna().sum()) == 19 and _dx[RIC["fpg_final"].isna()].all())
A.check("NAGALA: 373 events (as defined by the source study; no components are given in the file)", int(NAG["event"].sum()) == 373)

RIC_PART = np.select([_dx & ~_hi, ~_dx & _hi, _dx & _hi], ["diagnosis only", "glucose only", "both"], default="no event")
print("\n  RICH: follow-up time (years) by how the event was defined")
_t = RIC.assign(part=RIC_PART).groupby("part")["time_y"].agg(n="size", minimum="min", median="median", maximum="max").round(2)
print(_t.loc[["diagnosis only", "glucose only", "both", "no event"]].to_string())
print(f"\n  Share of RICH events defined by glucose at the final visit (alone or with a diagnosis): {100 * _hi.sum() / _ev.sum():.1f}%.")
print("  Consequence: baseline fasting glucose is close to an earlier measurement of the outcome itself. This is why the")
print("  headline tier (T2) contains no glucose, and why sensitivity analysis S1 uses the diagnosis-only outcome (1,304 events),")
print("  in which the 2,870 glucose-only events are treated as censored at their final visit.")
print(f"\n  Crude event proportions: RICH {100 * _ev.mean():.2f}%, NAGALA {100 * NAG['event'].mean():.2f}% (not comparable: follow-up differs).")
A.check("crude event proportions are 1.97% (RICH) and 2.41% (NAGALA)",
        (round(100 * _ev.mean(), 2), round(100 * NAG["event"].mean(), 2)) == (1.97, 2.41))
_ = A.close(outputs={"rich_parts": PARTS, "rich_events": int(_ev.sum()), "nagala_events": int(NAG["event"].sum()),
                     "rich_diagnosis_only_outcome_events": int(_dx.sum())})

# %%
# CELL 10 - follow-up, censoring and support at the prediction horizon                            [status L, D; protocol sections 3, 6; D-1]
# Why: confirms that 3 years is a supportable horizon, shows why a yes/no label at 3 years would be wrong (half of RICH
# is censored before then), and reproduces deviation D-1 (site 14 cannot be a test site).
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 10)
A = Audit(10, "Follow-up, censoring and horizon support")


def km_survival(time, event, at):
    """Kaplan-Meier probability of being event-free at each time in `at` (descriptive use in stage B)."""
    t, e = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    uniq, first = np.unique(t, return_index=True)
    deaths, at_risk = np.add.reduceat(e, first), len(t) - first
    surv = np.cumprod(1.0 - deaths / at_risk)
    pos = np.searchsorted(uniq, np.atleast_1d(np.asarray(at, dtype=float)), side="right") - 1
    return np.where(pos >= 0, surv[np.clip(pos, 0, None)], 1.0)


def median_follow_up(time, event):
    """Reverse Kaplan-Meier median: censoring is treated as the event, so early events do not shorten it."""
    t = np.unique(np.asarray(time, dtype=float))
    s = km_survival(time, 1 - np.asarray(event), t)
    return float(t[np.argmax(s <= 0.5)]) if (s <= 0.5).any() else float("nan")


# toy check of the estimator on four hand-made observations (not study data)
_toy = km_survival([1, 2, 3, 4], [1, 0, 1, 1], [0.5, 1, 2, 3, 4])
A.check("Kaplan-Meier function reproduces a hand calculation (1, 0.75, 0.75, 0.375, 0)", np.allclose(_toy, [1, 0.75, 0.75, 0.375, 0.0]), _toy.round(3).tolist())

_rows = []
for name, d in (("RICH", RIC), ("NAGALA", NAG)):
    for h in (2, 3, 4, 5):
        _rows.append({"cohort": name, "horizon_y": h, "followed_at_least": int((d["time_y"] >= h).sum()),
                      "percent_followed": round(100 * (d["time_y"] >= h).mean(), 1),
                      "events_by_horizon": int(((d["event"] == 1) & (d["time_y"] <= h)).sum()),
                      "censored_before": int(((d["event"] == 0) & (d["time_y"] < h)).sum()),
                      "percent_censored_before": round(100 * ((d["event"] == 0) & (d["time_y"] < h)).mean(), 1),
                      "km_risk_percent": round(100 * (1 - float(km_survival(d["time_y"], d["event"], h)[0])), 3)})
SUPPORT = pd.DataFrame(_rows)
SUPPORT.to_csv(DIR["audit"] / "horizon_support.csv", index=False)
print(SUPPORT.to_string(index=False))


def _s(c, h, col):
    return SUPPORT.loc[(SUPPORT["cohort"] == c) & (SUPPORT["horizon_y"] == h), col].iloc[0]


A.check("RICH at 3 years: 104,014 followed, 1,764 events, 106,055 (50.1%) censored earlier",
        (_s("RICH", 3, "followed_at_least"), _s("RICH", 3, "events_by_horizon"), _s("RICH", 3, "censored_before"), _s("RICH", 3, "percent_censored_before")) == (104014, 1764, 106055, 50.1))
A.check("NAGALA at 3 years: 10,998 followed, 101 events, 4,365 censored earlier",
        (_s("NAGALA", 3, "followed_at_least"), _s("NAGALA", 3, "events_by_horizon"), _s("NAGALA", 3, "censored_before")) == (10998, 101, 4365))
A.check("followed at least 4 and 5 years: RICH 45,133 and 8,912; NAGALA 9,659 and 8,345",
        [_s(c, h, "followed_at_least") for c in ("RICH", "NAGALA") for h in (4, 5)] == [45133, 8912, 9659, 8345])
A.check("events by 5 years: RICH 3,928; NAGALA 161", (_s("RICH", 5, "events_by_horizon"), _s("NAGALA", 5, "events_by_horizon")) == (3928, 161))
A.check("follow-up range: RICH 2.00 to 7.56 years; NAGALA 164 to 4,732 days",
        (round(RIC["time_y"].min(), 2), round(RIC["time_y"].max(), 2), int(NAG["days"].min()), int(NAG["days"].max())) == (2.0, 7.56, 164, 4732))
FOLLOW_UP = {name: {"median_reverse_km": round(median_follow_up(d["time_y"], d["event"]), 2), "median_observed": round(float(d["time_y"].median()), 2),
                    "person_years": round(float(d["time_y"].sum()), 0)} for name, d in (("RICH", RIC), ("NAGALA", NAG))}
print(f"\n  Follow-up in years: {FOLLOW_UP}")
print("  A yes/no label at 3 years would leave 50.1% of RICH and 28.2% of NAGALA unlabelled, or wrongly labelled as non-events.")
print("  All analyses therefore use time-to-event methods with the horizon fixed at 3 years (protocol section 6).")

SITES = RIC.groupby("site").agg(n=("id", "size"), events=("event", "sum"), max_follow_up=("time_y", "max"),
                                events_by_3y=("time_y", lambda s: int(((RIC.loc[s.index, "event"] == 1) & (s <= 3)).sum()))).reset_index()
SITES["max_follow_up"] = SITES["max_follow_up"].round(2)
SITES["role"] = np.where(SITES["site"].isin(SMALL_SITES), "training only (tiny site)",
                         np.where(SITES["site"].isin(NO_TEST_SITES), "training only (D-1)", "test site in leave-one-site-out"))
SITES.to_csv(DIR["audit"] / "rich_sites.csv", index=False)
print("\n  RICH sites:")
print(SITES.to_string(index=False))
A.check("only sites 14 and 16 have under 3 years of maximum follow-up (2.92 and 2.89)",
        SITES.loc[SITES["max_follow_up"] < 3, "site"].tolist() == [14, 16] and SITES.loc[SITES["max_follow_up"] < 3, "max_follow_up"].tolist() == [2.92, 2.89])
A.check("leave-one-site-out will have 12 test sites (D-1)", int((SITES["role"] == "test site in leave-one-site-out").sum()) == 12)
_ = A.close(outputs={"support": SUPPORT.to_dict("records"), "follow_up": FOLLOW_UP, "test_sites": SITES.loc[SITES["role"].str.startswith("test"), "site"].tolist()})
keep_for_session(10)

# %%
# CELL 11 - a design difference between the cohorts: early events                                [status L (descriptive)]
# Why: RICH contains only people followed for at least 2 years, and no RICH event is dated before 2.0 years. NAGALA has
# events from the first year. A 3-year risk therefore means different things in the two cohorts, and transport between
# them must compare like with like. This cell quantifies the difference; it fits no model.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 11)
A = Audit(11, "Design difference: early events and the two risk estimands")

_re, _ne = RIC["event"] == 1, NAG["event"] == 1
A.check("RICH: nobody has under 2 years of follow-up, and the earliest event is at 2.00 years",
        RIC["time_y"].min() > 2.0 and round(float(RIC.loc[_re, "time_y"].min()), 2) == 2.0, round(float(RIC["time_y"].min()), 4))
A.check("RICH: 0 events before 2 years; NAGALA: 61 events before 2 years",
        (int((_re & (RIC["time_y"] <= 2)).sum()), int((_ne & (NAG["time_y"] <= 2)).sum())) == (0, 61))
_at_risk_2 = NAG["time_y"] > 2
A.check("NAGALA: 12,513 still at risk at 2 years; 40 events between 2 and 3 years; 2,890 censored before 2 years",
        (int(_at_risk_2.sum()), int((_ne & _at_risk_2 & (NAG["time_y"] <= 3)).sum()), int((~_ne & (NAG["time_y"] <= 2)).sum())) == (12513, 40, 2890))

_sN = km_survival(NAG["time_y"], NAG["event"], [2, 3])
_sR = km_survival(RIC["time_y"], RIC["event"], [2, 3])
ESTIMANDS = {"NAGALA unconditional risk by 3 years (%)": round(100 * (1 - _sN[1]), 3),
             "NAGALA risk between 2 and 3 years given event-free at 2 years (%)": round(100 * (1 - _sN[1] / _sN[0]), 3),
             "RICH risk by 3 years (%)": round(100 * (1 - _sR[1]), 3),
             "RICH event-free probability at 2 years": round(float(_sR[0]), 6)}
print("\n  Kaplan-Meier estimates (descriptive):")
for k, v in ESTIMANDS.items():
    print(f"   {k:68s} {v}")
A.check("in RICH the event-free probability at 2 years is exactly 1 (by design, not by biology)", float(_sR[0]) == 1.0)
print("""
  What this means for the study:
   1. Unconditional 3-year risk, risk(3 | x) = 1 - exp(-H0(3) exp(lp)): what the locked protocol estimates in each cohort.
   2. In RICH this is, by design, the risk of an event between years 2 and 3 among people event-free at 2 years.
   3. A NAGALA-trained model predicts risk over the whole of years 0 to 3; a RICH-trained model cannot see years 0 to 2.
   4. The matching quantity is the conditional risk 1 - exp(-(H0(3) - H0(2)) exp(lp)) among people still at risk at 2 years.
  The locked cross-cohort analysis (cells 57 to 60) is run as specified. The two-year landmark comparison is an
  exploratory sensitivity analysis (amendment item A1.7, cell 61) and needs approval before it runs. With only 40
  NAGALA events between 2 and 3 years, it will be imprecise.""")
_ = A.close(outputs={"estimands": ESTIMANDS, "nagala_events_before_2y": 61, "nagala_at_risk_2y": int(_at_risk_2.sum()), "nagala_events_2_to_3y": 40})

# %%
# CELL 12 - sex coding and laboratory units                                                      [status L; protocol section 4]
# Why: sex is a predictor in every tier, and the NAGALA header says "1:Men, 2:Women" while the data hold 0 and 1. A
# reversed code would corrupt every cross-cohort result. Units are checked so that no conversion is needed.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 12)
A = Audit(12, "Sex coding and laboratory units")

# three independent lines of evidence that NAGALA code 1 = men
_h = np.sqrt(NAG["weight"] / NAG["bmi"]) * 100            # height in cm, recovered from weight and BMI
SEX_N = NAG.assign(height_cm=_h).groupby("sex").agg(n=("id", "size"), mean_height_cm=("height_cm", "mean"), mean_weight_kg=("weight", "mean"),
                                                    events=("event", "sum"), event_percent=("event", lambda s: 100 * s.mean())).round(2)
SEX_R = RIC.groupby("sex").agg(n=("id", "size"), mean_height_cm=("height", "mean"), mean_weight_kg=("weight", "mean"),
                               events=("event", "sum"), event_percent=("event", lambda s: 100 * s.mean())).round(2)
print("  NAGALA by sex code (height recovered from weight and BMI):")
print(SEX_N.to_string())
print("\n  RICH by sex code (1 = male, 2 = female, as its header states):")
print(SEX_R.to_string())
A.check("NAGALA: code 1 has 8,430 participants and code 0 has 7,034, the numbers of men and women in the source study",
        (int(SEX_N.loc[1, "n"]), int(SEX_N.loc[0, "n"])) == (8430, 7034))
A.check("NAGALA: code 1 is taller (170.8 cm against 158.3 cm), as men are in RICH (171.7 against 160.1)",
        (round(SEX_N.loc[1, "mean_height_cm"], 1), round(SEX_N.loc[0, "mean_height_cm"], 1), round(SEX_R.loc[1, "mean_height_cm"], 1), round(SEX_R.loc[2, "mean_height_cm"], 1)) == (170.8, 158.3, 171.7, 160.1))
A.check("NAGALA: code 1 has the higher event proportion (3.39% against 1.24%), as men have in RICH (2.58% against 1.23%)",
        (SEX_N.loc[1, "event_percent"], SEX_N.loc[0, "event_percent"], SEX_R.loc[1, "event_percent"], SEX_R.loc[2, "event_percent"]) == (3.39, 1.24, 2.58, 1.23))
print("\n  Decision: sex_male = 1 for NAGALA code 1 and for RICH code 1; 0 otherwise. Sensitivity analysis S3 repeats the")
print("  cross-cohort analysis without sex.")

# units: NAGALA gives each lipid and glucose twice; the mmol/L column is the mg/dL column times a fixed factor
PAIRS = {"fpg": ("fpg_mg", 0.05551), "tc": ("tc_mg", 0.02586), "tg": ("tg_mg", 0.01129), "hdl": ("hdl_mg", 0.02586)}
_units = []
for mm, (mg, factor) in PAIRS.items():
    ok = NAG[mg].notna()
    gap = float((NAG.loc[ok, mm] - NAG.loc[ok, mg] * factor).abs().max())
    _units.append({"variable": mm, "mmol_per_mg_dl": factor, "largest_gap": gap, "rows_compared": int(ok.sum())})
    A.check(f"NAGALA: {mm} in mmol/L equals {mg} times {factor} in every row", gap < 1e-9, f"largest gap {gap:.1e}")
_gap_a1c = float((NAG["hba1c_mmol"] - (10.93 * NAG["hba1c"] - 23.5)).abs().max())
A.check("NAGALA: HbA1c in mmol/mol equals 10.93 x HbA1c(%) - 23.5 in every row", _gap_a1c < 1e-9, f"largest gap {_gap_a1c:.1e}")
print("\n  Because the mmol/L columns are computed from the mg/dL columns, a blank HDL in mg/dL became 0 mmol/L (the 11 zeros of")
print("  cell 8). RICH is recorded in mmol/L. The harmonized data use the mmol/L columns of both cohorts: no unit is converted.")
_cmp = pd.DataFrame({"RICH median": RIC[["fpg", "tc", "tg", "hdl", "bmi", "sbp", "dbp", "alt", "ast", "age"]].median(),
                     "NAGALA median": NAG[["fpg", "tc", "tg", "hdl", "bmi", "sbp", "dbp", "alt", "ast", "age"]].median()}).round(2)
print("\n  Medians on the common scale (a unit error would show as a ratio far from 1):")
print(_cmp.assign(ratio=(_cmp["NAGALA median"] / _cmp["RICH median"]).round(2)).to_string())
A.check("medians of all ten shared variables agree within a factor of 2 between cohorts (no unit mismatch)",
        ((_cmp["NAGALA median"] / _cmp["RICH median"]).between(0.5, 2.0)).all())
_ = A.close(outputs={"nagala_sex": SEX_N.reset_index().to_dict("records"), "rich_sex": SEX_R.reset_index().to_dict("records"),
                     "unit_pairs": _units, "decision": "sex_male = 1 for NAGALA code 1 and RICH code 1; mmol/L columns used; no conversion"})

# %%
# CELL 13 - glycaemic eligibility and glycaemic states                                           [status L (descriptive)]
# Why: the cohorts admitted different people (NAGALA stops at a baseline glucose of 6.05 mmol/L, RICH at 6.99), which
# matters for transport. The same audit decides which proposal options (pp. 3 to 4) these data can support: current
# diabetes classification, prediabetes to diabetes, and healthy to prediabetes. This cell fits no model.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 13)
A = Audit(13, "Glycaemic eligibility and states")

# 1. eligibility ceilings and prevalent diabetes
A.check("baseline glucose ceiling: RICH 6.99 mmol/L, NAGALA 6.05 mmol/L (109 mg/dL); NAGALA HbA1c ceiling 6.4%",
        (float(RIC["fpg"].max()), round(float(NAG["fpg"].max()), 2), float(NAG["fpg_mg"].max()), float(NAG["hba1c"].max())) == (6.99, 6.05, 109.0, 6.4))
_prev = int((RIC["fpg"] >= 7.0).sum()) + int((NAG["fpg"] >= 7.0).sum()) + int((NAG["hba1c"] >= 6.5).sum())
A.check("prevalent diabetes at baseline by glucose or HbA1c: none in either cohort", _prev == 0, _prev)
print("  Proposal option A (classify current diabetes) is not identifiable from these data: there is no prevalent case.\n")

_r61 = RIC["fpg"] >= 6.1
A.check("RICH: 7,193 participants (3.4%) have baseline glucose >= 6.1 mmol/L, above NAGALA's ceiling; they carry 1,936 of 4,174 events",
        (int(_r61.sum()), round(100 * _r61.mean(), 1), int(RIC.loc[_r61, "event"].sum())) == (7193, 3.4, 1936))
A.check("RICH restricted to baseline glucose < 6.1 mmol/L: 204,640 participants, 2,238 events",
        (int((~_r61).sum()), int(RIC.loc[~_r61, "event"].sum())) == (204640, 2238))

# 2. baseline strata (5.6 mmol/L fasting glucose; HbA1c 5.7% in NAGALA)
_rn, _rp = RIC["fpg"] < 5.6, RIC["fpg"] >= 5.6
_np = (NAG["fpg"] >= 5.6) | (NAG["hba1c"] >= 5.7)
STRATA = pd.DataFrame([
    {"cohort": "RICH", "baseline stratum": "glucose < 5.6", "n": int(_rn.sum()), "events": int(RIC.loc[_rn, "event"].sum())},
    {"cohort": "RICH", "baseline stratum": "glucose 5.6 to 6.99 (prediabetes range)", "n": int(_rp.sum()), "events": int(RIC.loc[_rp, "event"].sum())},
    {"cohort": "NAGALA", "baseline stratum": "glucose < 5.6 and HbA1c < 5.7", "n": int((~_np).sum()), "events": int(NAG.loc[~_np, "event"].sum())},
    {"cohort": "NAGALA", "baseline stratum": "glucose >= 5.6 or HbA1c >= 5.7 (prediabetes range)", "n": int(_np.sum()), "events": int(NAG.loc[_np, "event"].sum())}])
STRATA["event_percent"] = (100 * STRATA["events"] / STRATA["n"]).round(2)
print(STRATA.to_string(index=False))
A.check("RICH strata: 185,586 (1,278 events) below 5.6; 26,247 (2,896 events) at 5.6 to 6.99",
        STRATA.loc[STRATA["cohort"] == "RICH", ["n", "events"]].values.tolist() == [[185586, 1278], [26247, 2896]])
A.check("NAGALA strata: 12,292 (102 events) normoglycaemic; 3,172 (271 events) in the prediabetes range",
        STRATA.loc[STRATA["cohort"] == "NAGALA", ["n", "events"]].values.tolist() == [[12292, 102], [3172, 271]])
A.check("NAGALA components: glucose >= 5.6 in 2,620 (230 events); HbA1c >= 5.7 in 987 (135 events)",
        (int((NAG["fpg"] >= 5.6).sum()), int(NAG.loc[NAG["fpg"] >= 5.6, "event"].sum()), int((NAG["hba1c"] >= 5.7).sum()), int(NAG.loc[NAG["hba1c"] >= 5.7, "event"].sum())) == (2620, 230, 987, 135))
print(f"\n  Note: NAGALA glucose of exactly 100 mg/dL converts to 5.55 mmol/L and falls below 5.6 ({int((NAG['fpg_mg'] == 100).sum())} participants).")

# 3. RICH only: glycaemic state at the final visit (one later time point per person; the time of change is not known)
_ev = RIC["event"] == 1
_f = RIC["fpg_final"]
RIC_STATE = np.select([_ev, _f >= 5.6, _f < 5.6], ["diabetes", "glucose 5.6 to 6.99", "glucose < 5.6"], default="unknown")
TRANSITIONS = pd.crosstab(np.where(_rn, "baseline < 5.6", "baseline 5.6 to 6.99"), RIC_STATE).reindex(columns=["glucose < 5.6", "glucose 5.6 to 6.99", "diabetes"])
TRANSITIONS.index.name, TRANSITIONS.columns.name = "RICH baseline", "state at final visit"
print("\n", TRANSITIONS.to_string(), sep="")
print("\n  Row percentages:")
print((100 * TRANSITIONS.div(TRANSITIONS.sum(axis=1), axis=0)).round(1).to_string())
A.check("RICH from below 5.6: 163,523 stay, 20,785 move to 5.6-6.99, 1,278 develop diabetes",
        TRANSITIONS.loc["baseline < 5.6"].tolist() == [163523, 20785, 1278])
A.check("RICH from 5.6-6.99: 11,909 return below 5.6, 11,442 stay, 2,896 develop diabetes",
        TRANSITIONS.loc["baseline 5.6 to 6.99"].tolist() == [11909, 11442, 2896])
A.check("every RICH participant has a known state at the final visit", not (RIC_STATE == "unknown").any())
A.check("NAGALA has no later glucose measurement (no column for it)", not any("final" in str(c).lower() for c in RAW["NAGALA"].columns))
print("""
  What these data can and cannot support:
   - Prediabetes to diabetes: yes, in both cohorts, as incident diabetes among people in the prediabetes range at baseline.
   - Healthy to prediabetes: in RICH only, as the state at the final visit (20,785 + 1,278 of 185,586). The time of the
     change is unknown, a single glucose value fluctuates (11,909 of 26,247 moved back below 5.6), and NAGALA offers no
     external check.
   - A full multistate model with dated transitions: no; it needs dated intermediate visits.
  The progression analyses are exploratory (amendment item A1.14, cells 79 and 80) and need approval before they run.""")
_ = A.close(outputs={"strata": STRATA.to_dict("records"), "rich_transitions": {str(k): {str(a): int(b) for a, b in v.items()} for k, v in TRANSITIONS.to_dict("index").items()},
                     "rich_fpg_ge_6_1": {"n": int(_r61.sum()), "events": int(RIC.loc[_r61, "event"].sum())}, "prevalent_diabetes": _prev})

# %%
# CELL 14 - GATE B: raw data and outcome audit                                                   [status L]
# Why: a stop point. It re-derives the key figures of stage B from the raw tables, independently of cells 7 to 13, and
# confirms those cells passed. Stage C (harmonization) starts only if this cell ends with ALL CHECKS PASSED.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 14)
require_gate("A")
A = Audit("gate_B", "Gate B: raw data and outcome audit")

_recs = {c: load_record(c) for c in range(7, 14)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else r["time"])
A.check("the raw tables in memory are still the ones audited at Gate A",
        all(content_sha(RAW[k]) == load_record(4)["outputs"]["content_sha"][k] for k in RAW))

# independent re-derivation straight from the raw tables, by column position
_n, _r = RAW["NAGALA"], RAW["RICH"]
_r_event, _r_time, _r_dx, _r_fin = _r.iloc[:, 20], _r.iloc[:, 21], _r.iloc[:, 19], _r.iloc[:, 18]
_n_event, _n_years = _n.iloc[:, 30], _n.iloc[:, 10] / 365.25
FACTS = {
    "n": {"RICH": int(len(_r)), "NAGALA": int(len(_n))},
    "events": {"RICH": int(_r_event.sum()), "NAGALA": int(_n_event.sum())},
    "rich_event_parts": {"diagnosis": int(_r_dx.notna().sum()), "final_glucose_ge_7": int((_r_fin >= 7.0).sum())},
    "events_by_3y": {"RICH": int(((_r_event == 1) & (_r_time <= 3)).sum()), "NAGALA": int(((_n_event == 1) & (_n_years <= 3)).sum())},
    "followed_3y": {"RICH": int((_r_time >= 3).sum()), "NAGALA": int((_n_years >= 3).sum())},
    "d3_zero_cells": {"RICH": int(sum((_r.iloc[:, j] == 0).sum() for j in (11, 12, 14, 15))), "NAGALA": int(sum((_n.iloc[:, j] == 0).sum() for j in (21, 17, 11, 12)))},
    "nagala_male_code_1": int((_n.iloc[:, 1] == 1).sum()),
    "baseline_glucose_max": {"RICH": float(_r.iloc[:, 9].max()), "NAGALA": round(float(_n.iloc[:, 27].max()), 2)},
}
EXPECTED = {
    "n": {"RICH": 211833, "NAGALA": 15464}, "events": {"RICH": 4174, "NAGALA": 373},
    "rich_event_parts": {"diagnosis": 1304, "final_glucose_ge_7": 3404},
    "events_by_3y": {"RICH": 1764, "NAGALA": 101}, "followed_3y": {"RICH": 104014, "NAGALA": 10998},
    "d3_zero_cells": {"RICH": 965, "NAGALA": 11}, "nagala_male_code_1": 8430,
    "baseline_glucose_max": {"RICH": 6.99, "NAGALA": 6.05},
}
for k in EXPECTED:
    A.check(f"re-derived from the raw tables: {k}", FACTS[k] == EXPECTED[k], FACTS[k])
write_json(DIR["audit"] / "stage_B_facts.json", FACTS)

print("""
  Decisions carried into stage C (harmonization):
   1. Outcome: RICH 'censor of diabetes at followup' = 1; NAGALA 'Incident DM' = 1. Time in years.
   2. Sex: male = 1 (NAGALA code 1; RICH code 1).
   3. Units: mmol/L columns of both cohorts; no conversion.
   4. D-3: exact zeros in triglycerides, HDL, ALT and AST become missing (RICH 965 values, NAGALA 11); no row removed.
   5. Never a predictor: identifier, site, follow-up time, event, RICH diagnosis flag, RICH final-visit glucose.
   6. Known design differences to carry into every transport result: RICH has no follow-up or events before 2 years;
      NAGALA admits baseline glucose only up to 6.05 mmol/L, RICH up to 6.99.""")
_ = A.close(outputs={"stage": "B", "facts": FACTS}, extra_path=DIR["gates"] / "gate_B.json")
print("\n  GATE B PASSED. Stage C (cells 15 to 21, harmonization) may start.")
