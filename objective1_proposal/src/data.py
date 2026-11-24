from pathlib import Path, PurePosixPath
from hashlib import sha256
import hashlib, json, os, pickle, re, shutil, stat, tempfile, zipfile
from collections import Counter
import numpy as np
import pandas as pd
import openpyxl

HASH = {
    "RICH": "ffd341a1a91ac65585b9fba70847a2681f5671a6035d9cfbf92e50795d74fbf6",
    "NAGALA": "671359aea2019c848f969395f88e57f888eaba8463ea0351bcda8f1c947f5523",
    "NHANES": "3af63d0d53d8f88c38a7f54d80e6e448014322f7c8c3fe115ba9b3e47c2c6acf",
    "proposal": "fc5f4d618ba1bb0d33d8d44f541846e75699268a24e1c851fbee275c00307b10",
}
CORE = ["age", "sex", "bmi", "sbp", "alt", "fpg", "tg", "hdl"]
NONLAB = CORE[:4]
PLUS = CORE + ["dbp", "smoking", "tc", "ast"]
UNITS = dict(
    age="years",
    sex="category",
    bmi="kg/m²",
    sbp="mmHg",
    dbp="mmHg",
    alt="U/L",
    ast="U/L",
    fpg="mmol/L",
    tg="mmol/L",
    hdl="mmol/L",
    tc="mmol/L",
    smoking="current 1, other 0",
    a1c="%",
    time="years",
    event="binary",
)
R = dict(
    id="id",
    site="site",
    age="Age (y)",
    sex="Gender(1, male; 2, female)",
    bmi="BMI(kg/m2)",
    sbp="SBP(mmHg)",
    dbp="DBP(mmHg)",
    alt="ALT(U/L)",
    ast="AST(U/L)",
    fpg="FPG (mmol/L)",
    tc="Cholesterol(mmol/L)",
    tg="Triglyceride(mmol/L)",
    hdl="HDL-c(mmol/L)",
    event="censor of diabetes at followup(1, Yes; 0, No)",
    time="year of followup",
    smoking="smoking status(1,current smoker;2, ever smoker;3,never smoker)",
)
J = dict(
    id="Patient Number",
    age="Baseline Age, yrs",
    sex="Sex, 1:Men, 2:Women",
    bmi="Baseline BMI, kg/m2",
    sbp="Baseline Systolic blood pressure, mmHg",
    dbp="Baseline Diastolic blood pressure, mmHg",
    alt="Baseline ALT, IU/L",
    ast="Baseline AST, IU/L",
    fpg="Baseline Fasting plasma glucose, mmol/L",
    tc="Baseline Total Cholesterol, mmol/L",
    tg="Baseline Triglycerides, mmol/L",
    hdl="Baseline HDL-cholesterol, mmol/L",
    event="Incident DM 0/1",
    time="Follow up duration, days",
    smoking="Baseline Smoking status, 1:never, 2:past, 3:current",
    a1c="Baseline HbA1c, %",
)
PERIODS = ["2017-2020", "2021-2023"]
NHC = "Diabetes_self HighBP HighChol BMI Smoker Stroke HeartDiseaseorAttack PhysActivity AnyHealthcare GenHlth MentHlth Sex Age AgeGroup Education Income SystolicBP DiastolicBP HbA1c FastingGlucose Diabetes_lab".split()
NH = dict(age="Age", sex="Sex", bmi="BMI", sbp="SystolicBP", a1c="HbA1c", reported="Diabetes_self")
CONVERT = {
    "hdl": ("Baseline HDL-cholesterol, mg/dL", 0.02586),
    "tc": ("Baseline Total Cholesterol, mg/dL", 0.02586),
    "tg": ("Baseline Triglycerides, mg/dL", 0.01129),
    "fpg": ("Baseline Fasting plasma glucose, mg/dl", 0.05551),
}


def digest(path):
    with open(path, "rb") as f:
        h = sha256()
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(obj):
    return sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    os.close(fd)
    try:
        writer(Path(name))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def js(path, obj):
    atomic(path, lambda p: p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf8"))


def csv(path, rows):
    frame = pd.DataFrame(rows).copy()
    parts = Path(path).parts
    for profile in ("SMOKE", "STANDARD", "FULL", "AUDIT"):
        if profile in parts and "results" in parts:
            frame["profile"] = profile
            frame["evidence_use"] = (
                "software check only" if profile in ("SMOKE", "AUDIT") else "scientific reanalysis"
            )
            break
    atomic(path, lambda p: frame.to_csv(p, index=False))


def dump(path, obj):
    payload = pickle.dumps(obj, protocol=5)
    atomic(path, lambda p: p.write_bytes(b"CKOBJ1 " + sha256(payload).hexdigest().encode() + b"\n" + payload))


def restore(path):
    packed = Path(path).read_bytes()
    head, sep, payload = packed.partition(b"\n")
    if not sep or head != b"CKOBJ1 " + sha256(payload).hexdigest().encode():
        raise ValueError(f"Integrity failure: {path}")
    return pickle.loads(payload)


def discover(raw):
    found = {}
    for path in sorted(Path(raw).iterdir()):
        if path.is_file() and path.suffix.lower() in [".xlsx", ".zip", ".pdf"]:
            actual = digest(path)
            for name, expected in HASH.items():
                if actual == expected:
                    if name in found:
                        raise ValueError(f"Two identical inputs for {name} in data/raw")
                    found[name] = path
    required = {"RICH", "NAGALA", "NHANES"}
    if not required <= set(found):
        raise ValueError(f"Missing verified bytes for {sorted(required - set(found))}. See README.")
    return found


def safe_unpack(archive, dest):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for item in z.infolist():
            n = PurePosixPath(item.filename.replace(chr(92), "/"))
            target = (dest / str(n)).resolve()
            if (
                not target.is_relative_to(dest)
                or ".." in n.parts
                or n.is_absolute()
                or (":" in str(n))
                or stat.S_ISLNK(item.external_attr >> 16)
                or (str(n).casefold() in seen)
            ):
                raise ValueError(f"Unsafe or duplicate ZIP member: {n}")
            seen.add(str(n).casefold())
        for item in z.infolist():
            if not item.is_dir():
                target = dest / item.filename.replace(chr(92), "/")
                with z.open(item) as src:

                    def copy(p):
                        with p.open("wb") as out:
                            shutil.copyfileobj(src, out)

                    atomic(target, copy)
        return [i.filename for i in z.infolist()]


def mapped(raw, mapping):
    if not raw.columns.is_unique or len(set(mapping.values())) != len(mapping):
        raise ValueError("Ambiguous columns")
    missing = set(mapping.values()) - set(raw)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return raw[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()}).copy()


def allowed(names, task):
    whitelist = PLUS if task in ("A", "R") else NONLAB
    if not names or len(names) != len(set(names)) or set(names) - set(whitelist):
        raise ValueError(f"Prohibited or duplicate predictor in {names}")
    return list(names)


def snapshot(df, cohort, idcol=None):
    return dict(
        cohort=cohort,
        rows=len(df),
        columns=len(df.columns),
        unique_id=df[idcol].nunique() if idcol else None,
        missing_id=df[idcol].isna().sum() if idcol else None,
        duplicate_id_rows=int(df[idcol].duplicated().sum()) if idcol else None,
        exact_duplicate_rows=int(df.duplicated().sum()),
        id_kind="source" if idcol else "no retained ID",
    )


def describe(df, cohort, units=None):
    out = []
    for name, col in df.items():
        num = pd.to_numeric(col, errors="coerce")
        levels = col.value_counts(dropna=True)
        qq = num.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).tolist()
        out.append(
            dict(
                cohort=cohort,
                field=name,
                dtype=str(col.dtype),
                unit=(units or {}).get(name, "see source header / codebook"),
                n=len(col),
                missing=int(col.isna().sum()),
                missing_pct=100 * col.isna().mean(),
                zeros=int(num.eq(0).sum()),
                distinct=col.nunique(),
                levels=json.dumps({str(k): int(v) for k, v in levels.items()}) if len(levels) <= 35 else "",
                **dict(zip(["min", "p01", "q1", "median", "q3", "p99", "max"], qq)),
            )
        )
    return out


def workbook(path, cohort):
    sheet = {"RICH": "RC", "NAGALA": "Sheet 1"}[cohort]
    cached = openpyxl.load_workbook(path, read_only=True, data_only=True)
    formulas = openpyxl.load_workbook(path, read_only=True, data_only=False)
    inventory = []
    counts = Counter()
    result = None
    for s in cached:
        it = s.values
        headers = list(next(it))
        if len(headers) != len(set(headers)) or None in headers:
            raise ValueError(f"Invalid headers in {s.title}")
        raw = pd.DataFrame(it, columns=headers)
        inventory.append(
            dict(cohort=cohort, sheet=s.title, rows=len(raw), columns=len(headers), headers=headers)
        )
        for row in formulas[s.title].iter_rows(min_row=2):
            for c in row:
                if c.data_type == "f":
                    name = headers[c.column - 1]
                    val = raw.iloc[c.row - 2, c.column - 1]
                    counts[s.title, name, "formulas"] += 1
                    counts[s.title, name, "cache_empty"] += int(pd.isna(val))
                    counts[s.title, name, "cache_zero"] += int(val == 0)
        if s.title == sheet:
            result = raw
    cached.close()
    formulas.close()
    if result is None:
        raise ValueError(f"Missing sheet {sheet}")
    return (
        result,
        inventory,
        [dict(cohort=cohort, sheet=s, field=f, check=c, count=n) for (s, f, c), n in counts.items()],
    )


def harmonize(raw, cohort, keep_alt_zero=False):
    j = cohort == "NAGALA"
    d = mapped(raw, J if j else R)
    checks = []
    if j:
        for k, (source, factor) in CONVERT.items():
            reference = raw[source] * factor
            bad = (d[k].isna() | d[k].eq(0)) & reference.gt(0)
            good = d[k].notna() & ~bad & reference.notna()
            err = (d.loc[good, k] - reference[good]).abs()
            if (err > 0.002).any():
                raise ValueError(f"{k} mmol/L cache disagrees with mass concentration; inspect factors")
            checks.append(
                dict(
                    cohort=cohort,
                    field=k,
                    recovered=int(bad.sum()),
                    verified=int(good.sum()),
                    max_abs_error=err.max(),
                    factor=factor,
                )
            )
            d.loc[bad, k] = reference[bad]
        d["time"] = d.time / 365.25
        d["site"] = "NAGALA"
    for k in PLUS:
        if k not in ("sex", "smoking"):
            valid = d[k].ge(0) if k == "alt" and keep_alt_zero else d[k].gt(0)
            d[k] = d[k].where(valid & np.isfinite(d[k]))
    sexmap = {0: 0, 1: 1} if j else {1: 1, 2: 0}
    if set(d.sex.dropna().unique()) - set(sexmap):
        raise ValueError("Undocumented sex codes")
    d["sex"] = d.sex.map(sexmap)
    d["smoking"] = d.smoking.map({1: 0, 2: 0, 3: 1} if j else {1: 1, 2: 0, 3: 0})
    if d.id.isna().any() or d.id.duplicated().any():
        raise ValueError("Missing/repeated source ID; resolve person-level records")
    d["id"] = cohort + "-" + d.id.astype(str)
    d["cohort"] = cohort
    return (d, checks)


def status(d):
    return np.where((d.event == 1) & (d.time <= 3), 1, np.where(d.time > 3, 0, -1))


def screen_candidates(raw, period, no_code):
    d = mapped(raw, NH)
    d["id"] = period + "-row-" + pd.Series(np.arange(len(d)), index=d.index).astype(str)
    d["cohort"] = period
    d["site"] = period
    d["event"] = d.a1c.ge(6.5).astype(int).where(d.a1c.gt(0) & d.a1c.le(100))
    for k in ["age", "bmi", "sbp"]:
        d[k] = d[k].where(d[k].gt(0))
    if not set(d.sex.dropna()).issubset({0, 1}):
        raise ValueError("Unexpected NHANES sex codes")
    rules = [
        ("age >=20", d.age >= 20),
        ("valid measured HbA1c percent", d.event.notna()),
        (f"extract Diabetes_self equals {no_code}", d.reported.eq(no_code)),
    ]
    return flow(d, rules, period)


def flow(d, rules, cohort):
    rows = []
    stay = pd.Series(True, index=d.index)
    for label, mask in [("raw", stay.copy())] + rules:
        stay &= mask.fillna(False)
        subset = d.loc[stay]
        rows.append(
            dict(
                cohort=cohort,
                step=label,
                n=len(subset),
                events=int(subset.event.eq(1).sum()),
                cases3=int((status(subset) == 1).sum()) if "time" in d else None,
            )
        )
    return (d.loc[stay].reset_index(drop=True), rows)


def eligible(d, broad=False):
    rules = [
        ("age >=20", d.age >= 20),
        ("positive finite follow-up", d.time.gt(0) & np.isfinite(d.time)),
        ("binary event", d.event.isin([0, 1])),
        ("observed FPG <7 and source baseline exclusions", d.fpg.lt(7)),
    ]
    if "a1c" in d:
        rules.append(("observed baseline HbA1c >0 and <6.5%", d.a1c.gt(0) & d.a1c.lt(6.5)))
    if not broad:
        rules.append(("harmonized FPG <6.1", d.fpg.lt(6.1)))
    return flow(d, rules, d.cohort.iloc[0] + ("_broad" if broad else ""))


def audit(paths, out, cache, no_code=0):
    out, cache = (Path(out), Path(cache))
    rawdesc, summary, inventory, formulas, conversions, flows, baseline, sites, aux = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    by = {}
    availability = []
    for c, mp, shape in [("RICH", R, (211833, 25)), ("NAGALA", J, (15464, 31))]:
        raw, inv, fmt = workbook(paths[c], c)
        summary.append({**snapshot(raw, c, mp["id"]), "shape_matches_reference": raw.shape == shape})
        inventory += inv
        formulas += fmt
        rawdesc += describe(raw, c)
        d, chk = harmonize(raw, c)
        conversions += chk
        if c == "RICH":
            f = raw["FPG of final visit(mmol/L)"].ge(7)
            a = raw["Diabetes diagnosed during followup（1,Yes）"].fillna(0).eq(1)
            aux = (
                pd.DataFrame({"event": d.event, "final_FPG_ge7": f, "aux_reported": a})
                .value_counts()
                .rename("n")
                .reset_index()
                .to_dict("records")
            )
        for label, dd in [(c, d)] + ([("RICH_alt0", harmonize(raw, c, True)[0])] if c == "RICH" else []):
            for broad in [False, True] if label == "RICH" else [False]:
                name = label + ("_broad" if broad else "")
                cohort, log = eligible(dd, broad)
                for x in log:
                    x["cohort"] = name
                flows += log
                dump(cache / (name + ".pkl"), cohort)
                if name != label or name != c:
                    continue
                baseline += describe(cohort[PLUS + ["time", "event"]], c, UNITS)
                for setname, features in [("nonlaboratory", NONLAB), ("primary", CORE), ("expanded", PLUS)]:
                    availability.append(
                        dict(
                            cohort=c,
                            set=setname,
                            features=len(features),
                            n=len(cohort),
                            all_observed=int(cohort[features].notna().all(axis=1).sum()),
                            at_least_one_missing=int(cohort[features].isna().any(axis=1).sum()),
                        )
                    )
                for stage, part in [("raw", dd), ("eligible", cohort)]:
                    for site, g in list(part.groupby("site", dropna=False)) + [("ALL", part)]:
                        st = status(g)
                        sites.append(
                            dict(
                                cohort=c,
                                stage=stage,
                                site=site,
                                n=len(g),
                                events=int(g.event.eq(1).sum()),
                                cases3=int((st == 1).sum()),
                                controls3=int((st == 0).sum()),
                                early_or_at3=int((st == -1).sum()),
                                risk_set_3=int(g.time.ge(3).sum()),
                                followup_median=g.time.median(),
                            )
                        )
        dump(cache / (c + "_valid_masks.pkl"), d[["id"]].join(d[PLUS].notna().add_suffix("_observed")))
    members = safe_unpack(paths["NHANES"], cache / "nhanes_members")
    base = cache / "nhanes_members"
    for period, rows in zip(PERIODS, [8702, 6185]):
        files = [n for n in members if period in n and n.endswith(".csv")]
        prov = [n for n in members if period in n and n.endswith(".json")]
        if len(files) != 1 or len(prov) != 1:
            raise ValueError("Missing/ambiguous period CSV or provenance")
        raw = pd.read_csv(base / files[0])
        if list(raw) != NHC:
            raise ValueError(f"Unexpected {period} headers")
        meta = json.loads((base / prov[0]).read_text(encoding="utf-8-sig"))
        if meta["columns"]["HbA1c"] != "LBXGH" or meta["columns"]["Diabetes_self"] != "DIQ010":
            raise ValueError("Check endpoint provenance")
        by[period] = meta
        summary.append({**snapshot(raw, period), "shape_matches_reference": raw.shape == (rows, 21)})
        rawdesc += describe(raw, period)
        d, lg = screen_candidates(raw, period, no_code)
        for x in lg:
            x["status"] = "candidate count pending source recode review"
        flows += lg
        dump(cache / (period + ".pkl"), d)
        baseline += describe(d[NONLAB + ["a1c", "event"]], period, UNITS)
    reports = [n for n in members if n.endswith(".md")]
    if len(reports) != 1:
        raise ValueError("Missing consistency report")
    js(
        out / "inventory.json",
        dict(
            hashes={n: digest(p) for n, p in paths.items()},
            workbooks=inventory,
            zip_members=members,
            provenance=by,
            consistency_report=(base / reports[0]).read_text(encoding="utf-8-sig"),
        ),
    )
    for name, rows in dict(
        structure=summary,
        raw_features=rawdesc,
        formula_cache=formulas,
        unit_checks=conversions,
        cohort_flow=flows,
        baseline=baseline,
        site_support=sites,
        feature_set_availability=availability,
        rich_outcome_check=aux,
    ).items():
        csv(out / (name + ".csv"), rows)
    from dictionary import dictionary

    csv(out / "field_dictionary.csv", dictionary(pd.DataFrame(rawdesc)))
    return pd.DataFrame(summary)
