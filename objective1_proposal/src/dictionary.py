import re
import pandas as pd
from data import R, J, NH, CORE, PLUS, UNITS


def dictionary(raw_profile):
    rows = []
    nh_units = dict(
        Age="years",
        BMI="kg/m²",
        SystolicBP="mmHg",
        DiastolicBP="mmHg",
        HbA1c="%",
        FastingGlucose="mg/dL",
        MentHlth="derived PHQ-9 coding; see provenance and observed levels",
        Income="derived category",
        Education="derived category",
    )
    for row in raw_profile.to_dict("records"):
        c = row["cohort"]
        f = row["field"]
        mp = R if c == "RICH" else J if c == "NAGALA" else NH
        alias = next((k for k, v in mp.items() if v == f), "")
        unit = UNITS.get(alias, "")
        role = "audit only"
        code = "Numeric measurement; unusual positive values retained."
        if alias in CORE:
            role = "shared primary" if c in ["RICH", "NAGALA"] else "screening primary"
        elif alias in PLUS:
            role = "expanded sensitivity"
        elif alias in ("event", "time", "id", "site", "a1c", "reported"):
            role = {
                "event": "outcome",
                "time": "follow-up",
                "id": "partition key",
                "site": "group only",
                "a1c": "baseline eligibility / screening label",
                "reported": "screening eligibility",
            }[alias]
        if c in ("RICH", "NAGALA"):
            if alias == "sex":
                code = "1 male, 2 female" if c == "RICH" else "1 male, 0 female; header conflicts"
            elif alias == "smoking":
                code = "1 current, 2 ever, 3 never" if c == "RICH" else "1 never, 2 past, 3 current"
            elif alias == "event":
                code = "1 incident detection, 0 censored"
            elif alias == "id":
                code = "Nonmissing unique source participant ID"
            elif alias == "site":
                code = "Observed grouping codes; center mapping unavailable"
            elif alias == "time":
                unit = "days" if c == "NAGALA" else "years"
                code = "Positive observed duration"
            elif alias == "a1c":
                code = "Baseline percent; eligible 0 < HbA1c < 6.5"
            elif alias in PLUS:
                code = (
                    "Finite and positive; ALT zero retained only in sensitivity"
                    if alias == "alt"
                    else "Finite and positive"
                )
            elif any(
                (
                    s in f.lower()
                    for s in (
                        "status",
                        "0/1",
                        "consumption, non:",
                        "phenotype",
                        "histroy",
                        "diagnosed during",
                    )
                )
            ):
                code = "Source header coding; audit only"
            if not unit:
                known = [
                    ("mmol/mol", "mmol/mol"),
                    ("mmol/l", "mmol/L"),
                    ("umol/l", "µmol/L"),
                    ("mg/dl", "mg/dL"),
                    ("iu/l", "U/L"),
                    ("(cm)", "cm"),
                    (", cm", "cm"),
                    ("(kg)", "kg"),
                    (", kg", "kg"),
                    ("g/wk", "g/week"),
                ]
                unit = next((u for s, u in known if s in f.lower()), "source ID / code")
        else:
            unit = nh_units.get(f, "derived category / binary code")
            code = "Producer transformation unverified; see observed levels and JSON source variable."
            if f == "Sex":
                code = "0 and 1; category labels unverified"
            if f == "Diabetes_self":
                code = "Extract 0/1; source DIQ010 uses 1/2/3/7/9; as-coded eligibility uses 0"
            if f == "HbA1c":
                code = "LBXGH percent >0 and ≤100; single screen positive ≥6.5%"
            if f in ("BMI", "SystolicBP", "Age"):
                code = "Finite and positive; eligibility age ≥20"
            if f in ("HbA1c", "FastingGlucose", "Diabetes_lab", "Diabetes_self"):
                role += "; prohibited predictor"
        rows.append(
            dict(
                cohort=c,
                raw_field=f,
                canonical=alias,
                unit=unit,
                role=role,
                validity_or_codes=code,
                observed_levels=row["levels"],
            )
        )
    return pd.DataFrame(rows)
