# %%
# CELL 15 - feature registry: every input named in the proposal, and where it is (or is not) in the data    [status L; proposal pp. 1-2, 5-6]
# Why: the proposal says variables must be "justified clinically and assessed for leakage, fairness, missingness and
# availability" and not put in blindly. This registry is the single list that cell 16 builds the harmonized data from.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 15)
require_gate("B")
A = Audit(15, "Feature registry")

# (proposal group, proposal input, harmonized variable, RICH source, NAGALA source, unit or coding, role, status, note)
# Sources are the short column names of cell 7; None means the cohort does not record it.
_REG = [
    ("demographic", "age", "age", "age", "age", "years", "T1", "L", "prior up, soft: risk levels off at older ages in NAGALA"),
    ("demographic", "sex", "sex_male", "sex", "sex", "1 male, 0 female", "T1", "L", "no prior; coding verified in cell 12"),
    ("demographic", "ethnicity or ancestry", None, None, None, "", "unavailable", "G", "not recorded; both cohorts are East Asian"),
    ("anthropometric", "BMI", "bmi", "bmi", "bmi", "kg/m2", "T1", "L", "prior up"),
    ("anthropometric", "waist circumference", "waist", None, "waist", "cm", "NAGALA extras", "L", "no prior (protocol section 7)"),
    ("anthropometric", "weight", None, "weight", "weight", "kg", "not used", "L", "contained in BMI"),
    ("anthropometric", "height", None, "height", None, "cm", "not used", "L", "contained in BMI; RICH only"),
    ("clinical", "blood pressure (systolic)", "sbp", "sbp", "sbp", "mmHg", "T1", "L", "prior up; stands for hypertension"),
    ("clinical", "blood pressure (diastolic)", "dbp", "dbp", "dbp", "mmHg", "T1", "L", "no prior"),
    ("clinical", "hypertension", "sbp", "sbp", "sbp", "mmHg", "T1", "L", "no diagnosis recorded; represented by SBP"),
    ("clinical", "family history", "famhx", "famhx", None, "1 yes, 0 no", "RICH extras", "L", "prior up; RICH only"),
    ("clinical", "previous prediabetes", "predm_base", "fpg", "fpg", "1 if in the prediabetes range at baseline", "stratifier", "A", "derived; never a predictor; amendment A1.14"),
    ("clinical", "cardiovascular disease", None, None, None, "", "unavailable", "G", "not recorded"),
    ("clinical", "relevant medications", None, None, None, "", "unavailable", "G", "not recorded"),
    ("laboratory", "fasting glucose", "fpg", "fpg", "fpg", "mmol/L", "T3", "L", "prior up; tiers T3 and T4 only"),
    ("laboratory", "HbA1c", "hba1c", None, "hba1c", "%", "NAGALA extras E3", "L", "prior up; NAGALA only; never across cohorts"),
    ("laboratory", "HDL", "hdl", "hdl", "hdl", "mmol/L", "T4", "L", "prior down; 44.6% missing in RICH"),
    ("laboratory", "LDL", "ldl", "ldl", None, "mmol/L", "E4", "A", "RICH only; amendment A1.5; one zero value, outside D-3"),
    ("laboratory", "triglycerides", "tg", "tg", "tg", "mmol/L", "T2", "L", "prior up"),
    ("laboratory", "total cholesterol", "tc", "tc", "tc", "mmol/L", "T2", "L", "no prior"),
    ("laboratory", "ALT (protocol tier T2)", "alt", "alt", "alt", "U/L", "T2", "L", "no prior; in the protocol, not in the proposal list"),
    ("laboratory", "AST (protocol tier T4)", "ast", "ast", "ast", "U/L", "T4", "L", "no prior; 58.2% missing in RICH"),
    ("laboratory", "GGT (protocol extras)", "ggt", None, "ggt", "IU/L", "NAGALA extras", "L", "no prior; NAGALA only"),
    ("laboratory", "fatty liver (protocol extras)", "fatty_liver", None, "fatty_liver", "1 yes, 0 no", "NAGALA extras", "L", "no prior; NAGALA only"),
    ("lifestyle", "physical activity", "exercise", None, "exercise", "1 regular exercise, 0 none", "NAGALA extras", "L", "prior down; NAGALA only"),
    ("lifestyle", "diet", None, None, None, "", "unavailable", "G", "not recorded"),
    ("lifestyle", "smoking", "smoke", "smoke", "smoke", "0 never, 1 former, 2 current", "RICH extras", "L", "no prior; 71.6% missing in RICH; raw codes are reversed between cohorts"),
    ("lifestyle", "alcohol", "drink", "drink", None, "0 never, 1 former, 2 current", "E4", "A", "RICH; 71.6% missing; amendment A1.5"),
    ("lifestyle", "alcohol", "ethanol", None, "ethanol", "g per week", "E4", "A", "NAGALA; amendment A1.5"),
    ("lifestyle", "sleep", None, None, None, "", "unavailable", "G", "not recorded"),
    ("history", "medication information", None, None, None, "", "unavailable", "G", "not recorded"),
    ("history", "previous glucose measurements", None, None, None, "", "unavailable", "G", "one baseline value per person"),
]
REGISTRY = pd.DataFrame(_REG, columns=["group", "proposal_input", "variable", "rich_source", "nagala_source", "unit_or_coding", "role", "status", "note"])
REGISTRY["prior"] = REGISTRY["variable"].map(lambda v: {+1: "up", -1: "down"}.get(PRIORS.get(v), ""))
REGISTRY["rich_missing_percent"] = [round(100 * RIC[s].isna().mean(), 2) if s else None for s in REGISTRY["rich_source"]]
REGISTRY["nagala_missing_percent"] = [round(100 * NAG[s].isna().mean(), 2) if s else None for s in REGISTRY["nagala_source"]]
REGISTRY.to_csv(DIR["tables"] / "feature_registry.csv", index=False)
print(REGISTRY[["proposal_input", "variable", "rich_source", "nagala_source", "role", "prior", "status", "rich_missing_percent", "nagala_missing_percent"]]
      .fillna("-").to_string(index=False))
print("\n  (Missing percentages are those of the raw files, before the D-3 rule.)")

PROPOSAL_INPUTS = ["age", "sex", "ethnicity or ancestry", "BMI", "waist circumference", "weight", "height", "blood pressure (systolic)",
                   "blood pressure (diastolic)", "hypertension", "family history", "previous prediabetes", "cardiovascular disease",
                   "relevant medications", "fasting glucose", "HbA1c", "HDL", "LDL", "triglycerides", "total cholesterol",
                   "physical activity", "diet", "smoking", "alcohol", "sleep", "medication information", "previous glucose measurements"]
A.check("every input named in the proposal (pp. 1-2, 5-6) has a registry row", set(PROPOSAL_INPUTS) <= set(REGISTRY["proposal_input"]),
        sorted(set(PROPOSAL_INPUTS) - set(REGISTRY["proposal_input"])))
_vars = set(REGISTRY["variable"].dropna())
A.check("every tier variable is in the registry and recorded in both cohorts",
        all(v in _vars and REGISTRY.loc[REGISTRY["variable"] == v, ["rich_source", "nagala_source"]].notna().all().all() for v in TIERS["T4"]))
A.check("every cohort-specific extra of the protocol is in the registry", all(v in _vars for vs in EXTRAS.values() for v in vs))
A.check("every prior of protocol section 7 points at a registry variable; no prior on anything else",
        set(PRIORS) <= _vars and set(REGISTRY.loc[REGISTRY["prior"] != "", "variable"]) == set(PRIORS))
_un = REGISTRY[REGISTRY["role"] == "unavailable"]
A.check("the 7 unavailable inputs have no source in either cohort and are gated (status G)",
        len(_un) == 7 and _un[["rich_source", "nagala_source"]].isna().all().all() and (_un["status"] == "G").all(), len(_un))
A.check("inputs that exist in one cohort only are never placed in a shared tier",
        not REGISTRY.loc[REGISTRY["role"].isin(["T1", "T2", "T3", "T4"]), ["rich_source", "nagala_source"]].isna().any().any())
_ = A.close(outputs={"rows": len(REGISTRY), "unavailable": _un["proposal_input"].tolist(), "file": "ckinn_results/tables/feature_registry.csv"})
keep_for_session(15)

# %%
# CELL 16 - harmonization: one schema for both cohorts                                           [status L, D; protocol sections 3-5; D-3]
# Why: every later cell reads these two tables. Values are copied, never imputed: missing stays missing. The only
# change to any value is deviation D-3 (zero-coded triglycerides, HDL, ALT, AST become missing). No row is removed.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 16)
A = Audit(16, "Harmonization", status="L, D")

D3_VARIABLES = ["tg", "hdl", "alt", "ast"]
SCHEMA = ["cohort", "id", "site", "age", "sex_male", "bmi", "sbp", "dbp", "tc", "tg", "alt", "fpg", "hdl", "ast",      # tiers T1 to T4
          "famhx", "smoke", "waist", "exercise", "fatty_liver", "ggt", "hba1c",                                       # protocol extras
          "ldl", "drink", "ethanol",                                                                                  # amendment A1.5 only
          "time_y", "event", "event_dx_only", "age_band", "predm_base", "state_final"]                                # outcome, time, strata


def _band(age):
    return np.select([age < AGE_BANDS[0][1], age < AGE_BANDS[1][1]], [0, 1], default=2).astype(int)


def harmonize():
    f = lambda s: s.astype(float)
    nan_r, nan_n = np.full(len(RIC), np.nan), np.full(len(NAG), np.nan)
    r = pd.DataFrame({
        "cohort": "RICH", "id": RIC["id"].astype(int), "site": RIC["site"].astype(int),
        "age": f(RIC["age"]), "sex_male": f(RIC["sex"] == 1), "bmi": f(RIC["bmi"]), "sbp": f(RIC["sbp"]), "dbp": f(RIC["dbp"]),
        "tc": f(RIC["tc"]), "tg": f(RIC["tg"]), "alt": f(RIC["alt"]), "fpg": f(RIC["fpg"]), "hdl": f(RIC["hdl"]), "ast": f(RIC["ast"]),
        "famhx": f(RIC["famhx"]), "smoke": RIC["smoke"].map({3: 0.0, 2: 1.0, 1: 2.0}),            # raw: 1 current, 2 ever, 3 never
        "waist": nan_r, "exercise": nan_r, "fatty_liver": nan_r, "ggt": nan_r, "hba1c": nan_r,
        "ldl": f(RIC["ldl"]), "drink": RIC["drink"].map({3: 0.0, 2: 1.0, 1: 2.0}), "ethanol": nan_r,
        "time_y": f(RIC["time_y"]), "event": RIC["event"].astype(int), "event_dx_only": f(RIC["dm_dx"].fillna(0).eq(1)),
        "age_band": _band(RIC["age"]), "predm_base": (RIC["fpg"] >= 5.6).astype(int),
        "state_final": np.select([RIC["event"] == 1, RIC["fpg_final"] >= 5.6], [2.0, 1.0], default=0.0)})
    n = pd.DataFrame({
        "cohort": "NAGALA", "id": NAG["id"].astype(int), "site": 1,                                # single centre
        "age": f(NAG["age"]), "sex_male": f(NAG["sex"] == 1), "bmi": f(NAG["bmi"]), "sbp": f(NAG["sbp"]), "dbp": f(NAG["dbp"]),
        "tc": f(NAG["tc"]), "tg": f(NAG["tg"]), "alt": f(NAG["alt"]), "fpg": f(NAG["fpg"]), "hdl": f(NAG["hdl"]), "ast": f(NAG["ast"]),
        "famhx": nan_n, "smoke": NAG["smoke"].map({1: 0.0, 2: 1.0, 3: 2.0}),                       # raw: 1 never, 2 past, 3 current
        "waist": f(NAG["waist"]), "exercise": f(NAG["exercise"]), "fatty_liver": f(NAG["fatty_liver"]), "ggt": f(NAG["ggt"]),
        "hba1c": f(NAG["hba1c"]), "ldl": nan_n, "drink": nan_n, "ethanol": f(NAG["ethanol"]),
        "time_y": f(NAG["days"]) / 365.25, "event": NAG["event"].astype(int), "event_dx_only": nan_n,
        "age_band": _band(NAG["age"]), "predm_base": ((NAG["fpg"] >= 5.6) | (NAG["hba1c"] >= 5.7)).astype(int),
        "state_final": nan_n})                                                                     # no later glucose in NAGALA
    d3 = {}
    for name, d in (("RICH", r), ("NAGALA", n)):
        d3[name] = {}
        for v in D3_VARIABLES:                                                                     # deviation D-3
            zero = d[v] == 0
            d3[name][v] = int(zero.sum())
            d.loc[zero, v] = np.nan
    return r[SCHEMA].reset_index(drop=True), n[SCHEMA].reset_index(drop=True), d3


_r, _n, D3_APPLIED = harmonize()
HARM = {"RICH": _r, "NAGALA": _n}


def harm(cohort):
    """A copy of the harmonized table of one cohort. Later cells never modify HARM itself."""
    return HARM[cohort].copy()


# checks
A.check("both tables have the same 30 columns in the same order", list(_r.columns) == SCHEMA and list(_n.columns) == SCHEMA and len(SCHEMA) == 30)
A.check("no row removed: RICH 211,833 and NAGALA 15,464; identifiers unchanged and unique",
        (len(_r), len(_n)) == (211833, 15464) and (_r["id"].to_numpy() == RIC["id"].to_numpy()).all() and (_n["id"].to_numpy() == NAG["id"].to_numpy()).all()
        and _r["id"].is_unique and _n["id"].is_unique)
A.check("events unchanged: RICH 4,174; NAGALA 373; diagnosis-only outcome in RICH 1,304",
        (int(_r["event"].sum()), int(_n["event"].sum()), int(_r["event_dx_only"].sum())) == (4174, 373, 1304))
A.check("time is in years: RICH 2.00 to 7.56; NAGALA 0.45 to 12.96",
        (round(_r["time_y"].min(), 2), round(_r["time_y"].max(), 2), round(_n["time_y"].min(), 2), round(_n["time_y"].max(), 2)) == (2.0, 7.56, 0.45, 12.96))
A.check("sex: 116,123 men in RICH and 8,430 in NAGALA", (int(_r["sex_male"].sum()), int(_n["sex_male"].sum())) == (116123, 8430))
A.check("smoking mapped to 0 never, 1 former, 2 current in both cohorts (raw codes are reversed)",
        _r["smoke"].value_counts().sort_index().astype(int).to_dict() == {0.0: 45596, 1.0: 2559, 2.0: 12075}
        and _n["smoke"].value_counts().sort_index().astype(int).to_dict() == {0.0: 9031, 1.0: 2952, 2.0: 3481})
A.check("D-3 applied: RICH 860 TG, 102 HDL, 2 ALT, 1 AST set to missing; NAGALA 11 HDL",
        D3_APPLIED == {"RICH": {"tg": 860, "hdl": 102, "alt": 2, "ast": 1}, "NAGALA": {"tg": 0, "hdl": 11, "alt": 0, "ast": 0}}, D3_APPLIED)
A.check("after D-3 no zero remains in triglycerides, HDL, ALT or AST", not any((d[D3_VARIABLES] == 0).any().any() for d in HARM.values()))
_same = {}
for name, d, raw in (("RICH", _r, RIC), ("NAGALA", _n, NAG)):
    for v in ["age", "bmi", "sbp", "dbp", "tc", "tg", "alt", "fpg", "hdl", "ast"]:
        keep = ~(raw[v] == 0) if v in D3_VARIABLES else np.ones(len(raw), bool)
        _same[f"{name}.{v}"] = bool(np.array_equal(d.loc[keep, v].to_numpy(), raw.loc[keep, v].to_numpy(dtype=float), equal_nan=True))
A.check("every other predictor value is copied bit for bit from the raw file (nothing imputed, converted or rounded)", all(_same.values()),
        [k for k, ok in _same.items() if not ok])
A.check("glucose and lipids are in mmol/L in both cohorts (NAGALA taken from its mmol/L columns)",
        np.array_equal(_n["fpg"].to_numpy(), NAG["fpg"].to_numpy()) and _n["fpg"].max() < 7 and _r["fpg"].max() < 7)
A.check("strata: prediabetes range at baseline 26,247 in RICH and 3,172 in NAGALA; RICH final state 175,432 / 32,227 / 4,174",
        (int(_r["predm_base"].sum()), int(_n["predm_base"].sum())) == (26247, 3172)
        and _r["state_final"].value_counts().sort_index().astype(int).tolist() == [175432, 32227, 4174])
A.check("age bands: under 40, 40 to 54, 55 and over = RICH 111,609 / 62,640 / 37,584; NAGALA 5,591 / 7,761 / 2,112",
        _r["age_band"].value_counts().sort_index().tolist() == [111609, 62640, 37584] and _n["age_band"].value_counts().sort_index().tolist() == [5591, 7761, 2112])
A.check("final-visit glucose and the diagnosis flag are not columns of the harmonized data", not {"fpg_final", "dm_dx"} & set(SCHEMA))

LEDGER = pd.DataFrame([{"cohort": name, "rows_in": len(raw), "rows_out": len(d), "events_in": int(raw["event"].sum()), "events_out": int(d["event"].sum()),
                        "d3_values_set_missing": sum(D3_APPLIED[name].values()), "participants_touched_by_d3": int((raw[D3_VARIABLES] == 0).any(axis=1).sum())}
                       for name, d, raw in (("RICH", _r, RIC), ("NAGALA", _n, NAG))])
print("\n  Participant flow:")
print(LEDGER.to_string(index=False))
_shared = ["age", "sex_male", "bmi", "sbp", "dbp", "tc", "tg", "alt", "fpg", "hdl", "ast"]
MISSING_H = pd.DataFrame({"RICH missing": _r[_shared].isna().sum(), "RICH %": (100 * _r[_shared].isna().mean()).round(2),
                          "NAGALA missing": _n[_shared].isna().sum(), "NAGALA %": (100 * _n[_shared].isna().mean()).round(2)})
print("\n  Missing values in the shared predictors after harmonization:")
print(MISSING_H.to_string())
A.check("missing after D-3 in RICH: TG 5,747; HDL 94,664; ALT 1,784; AST 123,291; SBP 23; DBP 24; TC 4,854",
        MISSING_H["RICH missing"].to_dict() == {"age": 0, "sex_male": 0, "bmi": 0, "sbp": 23, "dbp": 24, "tc": 4854, "tg": 5747, "alt": 1784, "fpg": 0, "hdl": 94664, "ast": 123291})
A.check("missing after D-3 in NAGALA: HDL 11; nothing else", MISSING_H["NAGALA missing"].to_dict() == {**{v: 0 for v in _shared}, "hdl": 11})

HARM_SHA = {}
for name, d in HARM.items():
    HARM_SHA[name] = content_sha(d)
    if not restoring():
        d.to_pickle(DIR["data"] / f"{name.lower()}_harmonized.pkl")
        d.to_csv(DIR["data"] / f"{name.lower()}_harmonized.csv", index=False)
        write_json(DIR["data"] / f"{name.lower()}_harmonized.json", {"cohort": name, "rows": len(d), "columns": SCHEMA, "content_sha": HARM_SHA[name],
                                                                    "source_sha": DATA_SHA[name], "d3": D3_APPLIED[name], "written": now()})
LEDGER.to_csv(DIR["tables"] / "participant_flow.csv", index=False)
print("\n  Saved: ckinn_results/data/rich_harmonized.(pkl, csv) and nagala_harmonized.(pkl, csv).")
_ = A.close(outputs={"harm_sha": HARM_SHA, "d3": D3_APPLIED, "ledger": LEDGER.to_dict("records"), "schema": SCHEMA})
keep_for_session(16)

# %%
# CELL 17 - predictor sets, the prior for each, and the leakage blocklist                        [status L; protocol sections 4, 7; A1.5 if approved]
# Why: fixes exactly which columns each model may see. A model cell can only obtain predictors through predictors(),
# which refuses any column on the blocklist, so outcome information cannot enter a model by accident.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 17)
A = Audit(17, "Predictor sets, priors and leakage blocklist")

# never a predictor: identifiers, site, time, every outcome, and every stratum derived from the outcome or from glucose
BLOCKLIST = {"cohort", "id", "site", "time_y", "event", "event_dx_only", "state_final", "predm_base", "age_band",
             "fpg_final", "dm_dx", "days", "phenotype", "wc_high", "bmi25"}
FEATURE_SETS = {
    "RICH": {**TIERS, "E2": TIERS["T2"] + EXTRAS["RICH"], "E3": TIERS["T3"] + EXTRAS["RICH"]},
    "NAGALA": {**TIERS, "E2": TIERS["T2"] + EXTRAS["NAGALA"], "E3": TIERS["T3"] + EXTRAS["NAGALA"] + EXTRAS["NAGALA_E3_extra"]}}
E4_APPROVED = bool(read_json(APPROVAL_FILE)["items"]["A1.5"]["approved"])
if E4_APPROVED:                                    # exploratory extended extras (amendment A1.5)
    FEATURE_SETS["RICH"]["E4"] = FEATURE_SETS["RICH"]["E3"] + ["ldl", "drink"]
    FEATURE_SETS["NAGALA"]["E4"] = FEATURE_SETS["NAGALA"]["E3"] + ["ethanol"]
CROSS_COHORT_SETS = ["T1", "T2", "T3", "T4"]       # only these may be used when training in one cohort and testing in the other


def predictors(cohort, name, drop=()):
    """The predictor columns of a set, e.g. predictors('RICH', 'T2'). `drop` removes columns (sensitivity S3 drops sex)."""
    cols = [c for c in FEATURE_SETS[cohort][name] if c not in drop]
    leak = sorted(set(cols) & BLOCKLIST)
    if leak:
        raise ValueError(f"leakage: {leak} may never be a predictor")
    absent = [c for c in cols if c not in HARM[cohort].columns or HARM[cohort][c].isna().all()]
    if absent:
        raise ValueError(f"{absent} not recorded in {cohort}")
    return cols


def priors_for(cols):
    """The clinical priors that apply to a list of predictors: {variable: +1 or -1}. No prior on an absent feature."""
    return {c: PRIORS[c] for c in cols if c in PRIORS}


_rows = []
for cohort in FEATURE_SETS:
    for name in FEATURE_SETS[cohort]:
        cols = predictors(cohort, name)
        d = HARM[cohort]
        cc = d[cols].notna().all(axis=1)
        _rows.append({"cohort": cohort, "set": name, "predictors": len(cols), "priors": ", ".join(f"{k} {'up' if v > 0 else 'down'}" for k, v in priors_for(cols).items()),
                      "any_missing": int((~cc).sum()), "any_missing_percent": round(100 * (~cc).mean(), 1),
                      "complete_cases": int(cc.sum()), "events_in_complete_cases": int(d.loc[cc, "event"].sum())})
SETS = pd.DataFrame(_rows)
SETS.to_csv(DIR["tables"] / "predictor_sets.csv", index=False)
print(SETS.to_string(index=False))
print(f"\n  E4 (amendment A1.5: LDL and alcohol): {'defined' if E4_APPROVED else 'not defined, because A1.5 is not approved'}.")
print("  Main analyses use every row (protocol section 5, deviation D-2); complete cases are used only in sensitivity analysis S2.")

A.check("tiers are exactly those of protocol section 4 in both cohorts", all(FEATURE_SETS[c][t] == TIERS[t] for c in FEATURE_SETS for t in TIERS))
A.check("no predictor set contains a blocklisted column", not any(set(FEATURE_SETS[c][s]) & BLOCKLIST for c in FEATURE_SETS for s in FEATURE_SETS[c]))
A.check("every predictor is numeric and recorded in its cohort",
        all(pd.api.types.is_float_dtype(HARM[c][v]) and HARM[c][v].notna().any() for c in FEATURE_SETS for s in FEATURE_SETS[c] for v in FEATURE_SETS[c][s]))
A.check("cross-cohort sets use only variables recorded in both cohorts",
        all(HARM["RICH"][v].notna().any() and HARM["NAGALA"][v].notna().any() for s in CROSS_COHORT_SETS for v in TIERS[s]))
A.check("HbA1c, family history, exercise and the other extras are in no cross-cohort set",
        not ({"hba1c", "famhx", "exercise", "waist", "fatty_liver", "ggt", "smoke", "ldl", "drink", "ethanol"} & set(TIERS["T4"])))
A.check("priors by tier: T1 age, BMI, SBP; T2 adds TG; T3 adds FPG; T4 adds HDL",
        [sorted(priors_for(TIERS[t])) for t in ("T1", "T2", "T3", "T4")] == [["age", "bmi", "sbp"], ["age", "bmi", "sbp", "tg"],
                                                                             ["age", "bmi", "fpg", "sbp", "tg"], ["age", "bmi", "fpg", "hdl", "sbp", "tg"]])
A.check("extras add the family-history prior in RICH and the exercise and HbA1c priors in NAGALA",
        priors_for(predictors("RICH", "E2")).get("famhx") == 1 and priors_for(predictors("NAGALA", "E2")).get("exercise") == -1
        and priors_for(predictors("NAGALA", "E3")).get("hba1c") == 1 and "hba1c" not in predictors("NAGALA", "E2"))
try:
    FEATURE_SETS["RICH"]["_test"] = ["age", "event"]
    predictors("RICH", "_test")
    _blocked = False
except ValueError:
    _blocked = True
finally:
    FEATURE_SETS["RICH"].pop("_test", None)
A.check("predictors() refuses a set that contains the outcome (tested with a deliberately wrong set)", _blocked)
A.check("rows with a missing predictor: RICH T1 24, T2 7,404, T3 7,404; NAGALA T1 to T3 none",
        SETS.set_index(["cohort", "set"])["any_missing"].loc[[("RICH", "T1"), ("RICH", "T2"), ("RICH", "T3"), ("NAGALA", "T1"), ("NAGALA", "T2"), ("NAGALA", "T3")]].tolist() == [24, 7404, 7404, 0, 0, 0],
        SETS.set_index(["cohort", "set"])["any_missing"].to_dict())
_ = A.close(outputs={"feature_sets": FEATURE_SETS, "blocklist": sorted(BLOCKLIST), "e4_defined": E4_APPROVED, "sets": SETS.to_dict("records")})
keep_for_session(17)

# %%
# CELL 18 - Table 1, differences between the cohorts, and missingness by site                    [status L; proposal p. 12]
# Why: any transport result has to be read against how the two populations differ. Standardized differences describe
# the covariate shift; the site table shows how strongly missingness in RICH depends on the site.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 18)
A = Audit(18, "Table 1, cohort differences, missingness by site")

CONT = ["age", "bmi", "sbp", "dbp", "fpg", "tc", "tg", "hdl", "alt", "ast", "hba1c", "waist", "ggt", "ldl", "ethanol"]
SKEWED = {"tg", "alt", "ast", "ggt", "ethanol"}                    # reported as median [quartiles]
BINARY = ["sex_male", "famhx", "exercise", "fatty_liver"]


def _cell(s, v):
    s = s.dropna()
    if len(s) == 0:
        return "not recorded"
    if v in BINARY:
        return f"{int(s.sum()):,} ({100 * s.mean():.1f}%)"
    if v in SKEWED:
        q = s.quantile([0.25, 0.5, 0.75])
        return f"{q[0.5]:.2f} [{q[0.25]:.2f}, {q[0.75]:.2f}]"
    return f"{s.mean():.2f} ({s.std():.2f})"


def _smd(a, b, v):
    a, b = a.dropna(), b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if v in BINARY:
        p, q = a.mean(), b.mean()
        return (p - q) / np.sqrt((p * (1 - p) + q * (1 - q)) / 2)
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


_R, _N = HARM["RICH"], HARM["NAGALA"]
_groups = {"RICH all": _R, "RICH event": _R[_R["event"] == 1], "RICH no event": _R[_R["event"] == 0],
           "NAGALA all": _N, "NAGALA event": _N[_N["event"] == 1], "NAGALA no event": _N[_N["event"] == 0]}
_t1 = [{"variable": "participants", **{g: f"{len(d):,}" for g, d in _groups.items()}, "SMD RICH - NAGALA": ""},
       {"variable": "follow-up, years: median [quartiles]", **{g: "{:.2f} [{:.2f}, {:.2f}]".format(*d["time_y"].quantile([0.5, 0.25, 0.75])) for g, d in _groups.items()}, "SMD RICH - NAGALA": ""}]
for v in ["sex_male"] + CONT[:10] + ["famhx", "exercise", "fatty_liver"] + CONT[10:]:
    _t1.append({"variable": v + (" : median [quartiles]" if v in SKEWED else " : n (%)" if v in BINARY else " : mean (SD)"),
                **{g: _cell(d[v], v) for g, d in _groups.items()}, "SMD RICH - NAGALA": "" if np.isnan(_smd(_R[v], _N[v], v)) else f"{_smd(_R[v], _N[v], v):+.2f}"})
for lab, code in (("smoke: never", 0), ("smoke: former", 1), ("smoke: current", 2)):
    _t1.append({"variable": lab + " : n (% of recorded)", **{g: f"{int((d['smoke'] == code).sum()):,} ({100 * (d['smoke'].dropna() == code).mean():.1f}%)" for g, d in _groups.items()}, "SMD RICH - NAGALA": ""})
TABLE1 = pd.DataFrame(_t1)
TABLE1.to_csv(DIR["tables"] / "table1.csv", index=False)
print(TABLE1.to_string(index=False))

SMD = pd.Series({v: _smd(_R[v], _N[v], v) for v in TIERS["T4"]}).round(3)
print("\n  Standardized mean differences (RICH minus NAGALA) for the shared predictors; beyond +/-0.1 is a noticeable shift:")
print(SMD.sort_values(key=abs, ascending=False).to_string())
A.check("Table 1 has all 11 shared predictors for both cohorts", all(SMD.notna()) and len(SMD) == 11)
A.check("men: 54.8% of RICH and 54.5% of NAGALA", (round(100 * _R["sex_male"].mean(), 1), round(100 * _N["sex_male"].mean(), 1)) == (54.8, 54.5))
A.check("mean age 42.1 (RICH) and 43.7 (NAGALA); mean BMI 23.2 and 22.1",
        (round(_R["age"].mean(), 1), round(_N["age"].mean(), 1), round(_R["bmi"].mean(), 1), round(_N["bmi"].mean(), 1)) == (42.1, 43.7, 23.2, 22.1))
A.check("in both cohorts, participants with an event are older and have higher BMI, SBP, triglycerides and glucose",
        all(d.loc[d["event"] == 1, v].mean() > d.loc[d["event"] == 0, v].mean() for d in (_R, _N) for v in ("age", "bmi", "sbp", "tg", "fpg")))

# missingness by tier and by site
print("\n  Participants with at least one missing predictor, by tier (rows are kept; see protocol section 5):")
print(SETS.loc[SETS["set"].isin(["T1", "T2", "T3", "T4"]), ["cohort", "set", "any_missing", "any_missing_percent", "complete_cases", "events_in_complete_cases"]].to_string(index=False))
SITE_MISS = (100 * _R.groupby("site")[["tc", "tg", "alt", "hdl", "ast", "smoke"]].agg(lambda s: s.isna().mean())).round(1)
SITE_MISS.insert(0, "n", _R.groupby("site").size())
SITE_MISS.insert(1, "event_percent", (100 * _R.groupby("site")["event"].mean()).round(2))
SITE_MISS.to_csv(DIR["tables"] / "rich_missing_by_site.csv")
print("\n  RICH: percentage missing by site:")
print(SITE_MISS.to_string())
_hdl_rate = (round(100 * _R.loc[_R["hdl"].notna(), "event"].mean(), 2), round(100 * _R.loc[_R["hdl"].isna(), "event"].mean(), 2))
print(f"\n  Missingness is informative: event proportion {_hdl_rate[0]}% where HDL is recorded and {_hdl_rate[1]}% where it is missing.")
A.check("missingness depends on the site: smoking is missing for 4.3% to 98.9% of a site, HDL for 15.4% to 62.0%",
        (float(SITE_MISS["smoke"].min()), float(SITE_MISS["smoke"].max()), float(SITE_MISS["hdl"].min()), float(SITE_MISS["hdl"].max())) == (4.3, 98.9, 15.4, 62.0),
        (float(SITE_MISS["smoke"].min()), float(SITE_MISS["smoke"].max()), float(SITE_MISS["hdl"].min()), float(SITE_MISS["hdl"].max())))
A.check("event proportion is higher where HDL is recorded (2.29%) than where it is missing (1.57%)", _hdl_rate == (2.29, 1.57), _hdl_rate)
_ = A.close(outputs={"smd": SMD.to_dict(), "hdl_event_percent_recorded_vs_missing": _hdl_rate, "files": ["table1.csv", "rich_missing_by_site.csv"]})

# %%
# CELL 19 - do the clinical priors hold descriptively? Risk by fifths of each constrained variable       [status L; proposal pp. 6-7; protocol sections 0, 7]
# Why: the proposal calls the priors relationships that hold "generally", not laws. This cell shows, without any model,
# how the observed 3-year risk moves across fifths of each constrained variable in each cohort. It is descriptive and
# was disclosed in protocol section 0; it does not choose or change any prior.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 19)
A = Audit(19, "Descriptive check of the clinical priors")
import matplotlib
import matplotlib.pyplot as plt

PRIOR_VARS = {"RICH": ["age", "bmi", "sbp", "tg", "fpg", "hdl", "famhx"], "NAGALA": ["age", "bmi", "sbp", "tg", "fpg", "hdl", "hba1c", "exercise"]}
_rows = []
for cohort, vs in PRIOR_VARS.items():
    d = HARM[cohort]
    for v in vs:
        s = d[v].dropna()
        if s.nunique() <= 2:
            grp = s.astype(int)
        else:
            grp = pd.qcut(s, 5, labels=False, duplicates="drop") + 1
        for g, idx in grp.groupby(grp).groups.items():
            sub = d.loc[idx]
            _rows.append({"cohort": cohort, "variable": v, "prior": "up" if PRIORS[v] > 0 else "down", "group": int(g), "n": len(sub),
                          "low": float(sub[v].min()), "high": float(sub[v].max()), "events": int(sub["event"].sum()),
                          "km_risk_3y_percent": round(100 * (1 - float(km_survival(sub["time_y"], sub["event"], H)[0])), 3)})
QUINT = pd.DataFrame(_rows)
QUINT.to_csv(DIR["tables"] / "risk_by_fifths.csv", index=False)

_summ = []
for (cohort, v), t in QUINT.groupby(["cohort", "variable"], sort=False):
    r = t["km_risk_3y_percent"].to_numpy()
    steps = np.sign(np.diff(r)) * PRIORS[v]
    _summ.append({"cohort": cohort, "variable": v, "prior": t["prior"].iloc[0], "risk_lowest_group": r[0], "risk_highest_group": r[-1],
                  "ends_agree_with_prior": bool((r[-1] - r[0]) * PRIORS[v] > 0), "steps_with_prior": int((steps > 0).sum()), "steps_against_prior": int((steps < 0).sum())})
PRIOR_CHECK = pd.DataFrame(_summ)
PRIOR_CHECK.to_csv(DIR["tables"] / "prior_descriptive_check.csv", index=False)
print("  Observed 3-year risk (Kaplan-Meier, %) in the lowest and highest group of each constrained variable:")
print(PRIOR_CHECK.to_string(index=False))
print("\n  Full table by fifths: ckinn_results/tables/risk_by_fifths.csv")

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, v in zip(axes.ravel(), ["age", "bmi", "sbp", "tg", "fpg", "hdl", "famhx", "hba1c"]):
    for cohort, colour in (("RICH", "tab:blue"), ("NAGALA", "tab:orange")):
        t = QUINT[(QUINT["cohort"] == cohort) & (QUINT["variable"] == v)]
        if len(t):
            ax.plot(t["group"], t["km_risk_3y_percent"], marker="o", color=colour, label=cohort)
            ax.set_xticks(t["group"].tolist())
    ax.set_title(f"{v} (prior: {'up' if PRIORS[v] > 0 else 'down'})")
    ax.set_xlabel("group, lowest to highest")
    ax.set_ylabel("3-year risk, %")
    ax.legend(fontsize=8)
fig.suptitle("Observed 3-year risk by fifths of each constrained variable (two groups for yes/no variables); descriptive, no model")
fig.tight_layout()
fig.savefig(DIR["figs"] / "priors_descriptive.png", dpi=130)
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 4.2))
_grid = np.linspace(0, 7.5, 301)
for cohort, colour in (("RICH", "tab:blue"), ("NAGALA", "tab:orange")):
    d = HARM[cohort]
    ax.step(_grid, 100 * (1 - km_survival(d["time_y"], d["event"], _grid)), where="post", color=colour, label=cohort)
ax.axvline(H, color="grey", linestyle=":")
ax.set_xlabel("years from baseline")
ax.set_ylabel("cumulative incidence of diabetes, %")
ax.set_title("Kaplan-Meier cumulative incidence by cohort")
ax.legend()
fig.tight_layout()
fig.savefig(DIR["figs"] / "km_by_cohort.png", dpi=130)
plt.show()
print("  Reading the second figure: RICH is flat until 2 years (no earlier follow-up), and beyond 5 years it rests on very few")
print("  participants (4.2% were followed that long), so its steep rise there is unstable. The 3-year horizon (dotted line) is supported.")

A.check("all 15 cohort-variable pairs were tabulated (7 in RICH, 8 in NAGALA)", len(PRIOR_CHECK) == 15, len(PRIOR_CHECK))
A.check("every group of every continuous variable contains events (no empty group)", (QUINT["events"] > 0).all(), int((QUINT["events"] == 0).sum()))
A.check("RICH: observed risk is zero before 2 years, by design", float(1 - km_survival(HARM["RICH"]["time_y"], HARM["RICH"]["event"], 2.0)[0]) == 0.0)
_agree = PRIOR_CHECK["ends_agree_with_prior"]
print(f"\n  Lowest-to-highest comparison agrees with the prior in {int(_agree.sum())} of {len(_agree)} cohort-variable pairs.")
print("  Pairs that do not agree, or that are not monotone across the fifths, are the reason the constraints are soft penalties.")
A.check("glucose and BMI agree with their priors in both cohorts (lowest against highest group)",
        PRIOR_CHECK.loc[PRIOR_CHECK["variable"].isin(["fpg", "bmi"]), "ends_agree_with_prior"].all())
_ = A.close(outputs={"prior_check": PRIOR_CHECK.to_dict("records"), "files": ["risk_by_fifths.csv", "prior_descriptive_check.csv", "priors_descriptive.png", "km_by_cohort.png"]})

# %%
# CELL 20 - how many events each subgroup has                                                    [status L; proposal p. 15; protocol section 9]
# Why: a subgroup comparison is only as good as the number of events behind it. This table decides, before any model
# is fitted, which subgroup results can be estimated and which must be reported as underpowered.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 20)
A = Audit(20, "Subgroup event counts")

MIN_EVENTS_3Y = 20                                  # below this a group is reported as not estimable (amendment A1.10); a flag only
_bands = [b[2] for b in AGE_BANDS]
_rows = []
for cohort, d in HARM.items():
    d = d.assign(sex=np.where(d["sex_male"] == 1, "men", "women"), band=d["age_band"].map(dict(enumerate(_bands))),
                 by3=((d["event"] == 1) & (d["time_y"] <= H)).astype(int))
    _groups = [("all", "all", d)]
    _groups += [("sex", sx, d[d["sex"] == sx]) for sx in ("men", "women")]
    _groups += [("age band", bd, d[d["band"] == bd]) for bd in _bands]
    _groups += [("sex x age band", f"{sx} / {bd}", d[(d["sex"] == sx) & (d["band"] == bd)]) for sx in ("men", "women") for bd in _bands]
    for label, name, g in _groups:
        _rows.append({"cohort": cohort, "grouping": label, "group": name, "n": len(g), "events": int(g["event"].sum()), "events_by_3y": int(g["by3"].sum())})
SUBGROUPS = pd.DataFrame(_rows)
SUBGROUPS["adequacy"] = np.where(SUBGROUPS["events_by_3y"] >= MIN_EVENTS_3Y, "estimable", "underpowered")
SUBGROUPS.to_csv(DIR["tables"] / "subgroup_events.csv", index=False)
print(SUBGROUPS.to_string(index=False))


def _e3(cohort, grouping, group):
    return int(SUBGROUPS.loc[(SUBGROUPS["cohort"] == cohort) & (SUBGROUPS["grouping"] == grouping) & (SUBGROUPS["group"] == group), "events_by_3y"].iloc[0])


A.check("NAGALA events by 3 years: 101 in all; 84 in men and 17 in women", (_e3("NAGALA", "all", "all"), _e3("NAGALA", "sex", "men"), _e3("NAGALA", "sex", "women")) == (101, 84, 17))
A.check("NAGALA events by 3 years by age band: 23, 51, 27", [_e3("NAGALA", "age band", b) for b in ("under 40", "40 to 54", "55 and over")] == [23, 51, 27])
A.check("NAGALA women by age band: 5, 7 and 5 events by 3 years (none estimable)",
        [_e3("NAGALA", "sex x age band", f"women / {b}") for b in ("under 40", "40 to 54", "55 and over")] == [5, 7, 5])
A.check("RICH events by age band: 621, 1,383 and 2,170",
        [int(SUBGROUPS.loc[(SUBGROUPS["cohort"] == "RICH") & (SUBGROUPS["grouping"] == "age band") & (SUBGROUPS["group"] == b), "events"].iloc[0]) for b in ("under 40", "40 to 54", "55 and over")] == [621, 1383, 2170])
A.check("every RICH subgroup by sex, by age band and by their combination is estimable",
        (SUBGROUPS.loc[SUBGROUPS["cohort"] == "RICH", "adequacy"] == "estimable").all())
_under = SUBGROUPS.loc[SUBGROUPS["adequacy"] == "underpowered", ["cohort", "grouping", "group", "events_by_3y"]]
print(f"\n  Underpowered groups (fewer than {MIN_EVENTS_3Y} events by 3 years):")
print(_under.to_string(index=False))
print("""
  Consequences for later stages:
   - Subgroup results in RICH (cross-validation, leave-one-site-out) rest on hundreds of events and are estimable.
   - In NAGALA, results for women (17 events by 3 years) and for every sex-by-age cell are underpowered. The locked
     subgroup analysis by sex and by age band is still run as specified, and reported with intervals and this caveat.
   - No fairness conclusion can be drawn from the sex-by-age cells of NAGALA.""")
A.check("the underpowered groups are all in NAGALA", (_under["cohort"] == "NAGALA").all() and len(_under) > 0, len(_under))
_ = A.close(outputs={"min_events_3y": MIN_EVENTS_3Y, "underpowered": _under.to_dict("records"), "file": "ckinn_results/tables/subgroup_events.csv"})

# %%
# CELL 21 - GATE C: harmonized predictors and estimands                                          [status L]
# Why: a stop point before any statistical or model code. It reloads the saved harmonized tables from disk, confirms
# they are the ones audited in cell 16, and re-checks the leakage rules. Stage D starts only if this cell passes.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 21)
require_gate("B")
A = Audit("gate_C", "Gate C: harmonized predictors and estimands")

_recs = {c: load_record(c) for c in range(15, 21)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else r["time"])
for name in ("RICH", "NAGALA"):
    _disk = pd.read_pickle(DIR["data"] / f"{name.lower()}_harmonized.pkl")
    _meta = read_json(DIR["data"] / f"{name.lower()}_harmonized.json")
    _want = _recs[16]["outputs"]["harm_sha"][name] if _recs[16] else None
    A.check(f"{name}: harmonized table on disk, its description file and the table in memory are all the one audited in cell 16",
            content_sha(_disk) == _want and _meta["content_sha"] == _want and content_sha(HARM[name]) == _want)
    A.check(f"{name}: built from the workbook verified at Gate A", _meta["source_sha"] == RAW_SPEC[name]["sha"])
    A.check(f"{name}: no zero in triglycerides, HDL, ALT or AST; no final-visit glucose column; no row lost",
            not (_disk[D3_VARIABLES] == 0).any().any() and "fpg_final" not in _disk.columns and len(_disk) == len(RAW[name]))
A.check("tiers still equal the frozen settings and no predictor set contains a blocklisted column",
        settings_fingerprint() == LOCKED_SETTINGS_SHA and all(FEATURE_SETS[c][t] == TIERS[t] for c in FEATURE_SETS for t in TIERS)
        and not any(set(FEATURE_SETS[c][s]) & BLOCKLIST for c in FEATURE_SETS for s in FEATURE_SETS[c]))
print(f"\n  Model result files present so far: {len(list(DIR['parts'].iterdir()))} parts, {len(list(DIR['preds'].iterdir()))} prediction files.")

print("""
  Fixed for all later stages:
   1. Estimand: absolute risk of incident diabetes within 3 years, risk(3 | x) = 1 - exp(-H0(3) exp(lp(x))).
   2. Outcome: event and time_y of the harmonized tables; sensitivity S1 uses event_dx_only in RICH.
   3. Predictors: only through predictors(cohort, set); tiers T1 to T4 across cohorts, extras within one cohort.
   4. Priors: only those returned by priors_for() for the predictors actually present.
   5. Missing values stay missing in the data; each model fills them inside its own training fold (protocol section 5).""")
_ = A.close(outputs={"stage": "C", "harm_sha": HARM_SHA, "feature_sets": FEATURE_SETS}, extra_path=DIR["gates"] / "gate_C.json")
print("\n  GATE C PASSED. Stage D (cells 22 to 29, statistical toolkit) may start.")
