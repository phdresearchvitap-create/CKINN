"""Build CKINN_Objective1_Colab_STANDARD_v3.ipynb from the submitted v2 notebook (kept unchanged in evidence/).

What changes and what does not:
- The embedded v2 software bundle (study/*.py, config.json) is byte-identical, so the run identity of a Colab session
  with the same runtime and options is unchanged (96ccd0057ac85891 for the submitted STANDARD run) and every completed
  fit on Drive is reused.
- Added: a LightGBM warning filter (cosmetic warning, predictions unaffected), a resume check that names any identity
  field that differs from the expected run, an EXT stage (post hoc extension analyses, ext/ckinn_ext.py, embedded with
  a SHA-256 check), and the v3 run recipe. RUN_STAGE accepts "EXT" in addition to the v2 stage groups.
Run: python build_notebook_v3.py  (writes the notebook and the cell-by-cell text file next to this script)
"""
import base64, copy, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "evidence" / "CKINN_Objective1_Colab_STANDARD_v2_submitted.ipynb"
OUT = HERE / "CKINN_Objective1_Colab_STANDARD_v3.ipynb"
TXT = HERE / "CKINN_Objective1_Colab_STANDARD_v3_cells.txt"
EXT = HERE / "ext" / "ckinn_ext.py"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def src(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


nb = json.loads(SRC.read_text())
cells = nb["cells"]
assert len(cells) == 80 and src(cells[3]).startswith("# 01 · Connect Drive and set paths")

INTRO = """
# Clinical knowledge-informed prediction of recorded diabetes · version 3

**What changed from version 2.** The analysis software embedded in cell 01 is byte-identical to version 2, so a run with the same Colab runtime and the same options keeps the same run identifier and reuses every completed fit on Drive. Version 3 adds (i) a filter for a cosmetic LightGBM warning, (ii) a resume check after cell 04, and (iii) an **EXT** stage between stages 35 and 36 with post hoc extension analyses whose rules were fixed before they were computed (`EXT_PRESPECIFICATION.md`). EXT results are exploratory and never change the primary comparison.

**Run recipe (STANDARD, resuming run 96ccd0057ac85891).** Keep `RUN_MODE = "STANDARD"` and the three `OPTION_OVERRIDES` exactly as they are; changing either creates a new run and refits everything. Then run the notebook top to bottom once per `RUN_STAGE`, in this order: `"EXPLAIN"`, `"SUBGROUPS"`, `"SCREEN"`, `"EXT"`, `"SENSITIVITY"` (the longest, about nine hours; it resumes after a disconnect), and finally `"REPORT"`. `"ALL"` also works but runs SENSITIVITY before EXT and REPORT. If the resume check reports a different run identifier, see its message before fitting anything.
"""
cells[0] = md(INTRO + "\n" + src(cells[0]))

c01 = src(cells[3])
anchor = 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n'
assert c01.count(anchor) == 1
c01 = c01.replace(anchor, anchor + (
    "import warnings\n"
    "# LightGBM 4.6 gives NumPy inputs placeholder names (Column_0, ...); the resulting sklearn warning is cosmetic.\n"
    "# Predictions are identical with and without names (checked in OBJ1_REVIEW.md, section 2).\n"
    'warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")\n'
    'EXPECTED_RUN = "96ccd0057ac85891"  # the STANDARD run in results-20261123; used only by the resume check after cell 04\n'))
cells[3] = code(c01)

c04 = src(cells[9])
old = 'if RUN_MODE not in config["profiles"] or (RUN_STAGE != "ALL" and RUN_STAGE not in GROUPS):'
assert c04.count(old) == 1
c04 = c04.replace(old, 'if RUN_MODE not in config["profiles"] or (RUN_STAGE not in ("ALL", "EXT") and RUN_STAGE not in GROUPS):')
old = "    selected = RUN_STAGE == \"ALL\" or number in GROUPS[RUN_STAGE]\n"
assert c04.count(old) == 1
c04 = c04.replace(old, "    selected = RUN_STAGE == \"ALL\" or (RUN_STAGE in GROUPS and number in GROUPS[RUN_STAGE])\n")
cells[9] = code(c04)

RESUME_MD = """
## 04b · Check that completed fits will be reused

The run identifier hashes the input files, every file in `study/`, the configuration (profile and options), the device and the Python and package versions. Completed fits are reused only when all of these match.

**What to look for.** `same run: True` for STANDARD. Otherwise the cell lists each field that differs from the frozen manifest of the expected run.

**If the stage stops.** If versions differ, install the listed versions, restart the runtime and rerun from 00; or accept a new run, which refits everything (about 16 hours on CPU for stages 21 to 29).
"""
RESUME = """
# 04b · Check that completed fits will be reused
print("this run:", st.run, "| expected:", EXPECTED_RUN, "| same run:", st.run == EXPECTED_RUN)
if RUN_MODE == "STANDARD" and st.run != EXPECTED_RUN:
    ref = ROOT / "checkpoints" / "STANDARD" / EXPECTED_RUN / "frozen" / "A" / "manifest.json"
    if ref.exists():
        old_identity = json.loads(ref.read_text())["identity"]
        for field in ["data", "code", "config", "device", "runtime", "versions"]:
            a, b = old_identity.get(field), st.identity.get(field)
            if a != b:
                if isinstance(a, dict) and isinstance(b, dict):
                    diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
                    print(f"differs: {field}: (expected, now) {diff}")
                else:
                    print(f"differs: {field}")
        print("Completed fits of the expected run will not be reused unless these match.")
    else:
        print("No frozen manifest of the expected run on this Drive; this is a new run.")
"""

ext_bytes = EXT.read_bytes()
ext_hash = hashlib.sha256(ext_bytes).hexdigest()
b64 = base64.b64encode(ext_bytes).decode()
chunks = "\n".join("    '" + b64[i:i + 100] + "'" for i in range(0, len(b64), 100))

EXT_MD = """
## EXT · Post hoc extension analyses (exploratory, rules fixed before computation)

These analyses were added after the STANDARD results were seen. Their rules are in `EXT_PRESPECIFICATION.md` (E1 to E11). They read the frozen models and predictions of this run and add new fits only for E9 (penalty-strength path), E10 (training-size curve) and E11 (RICH without sites 14 and 16). New fits are checkpointed like every other fit and resume after a disconnect. Outputs: `results/tables/<mode>/<run>/A/EXT/*.csv` (aggregate only), included in the stage 39 ZIP.

**What to look for.** E2 shows whether the clinical penalty is active at all (wrong-way gradient shares and the penalty relative to the data loss). E3 and E4 compare M1 and M3 with the simple baselines; E9 and E10 show whether any penalty strength or training size changes external error.

**If the stage stops.** Rerun with `RUN_STAGE = "EXT"`; finished parts are restored from checkpoints. Delete nothing.
"""
EXT_CODE = f"""
# EXT · Post hoc extension analyses
import base64 as _b64, hashlib as _hl, importlib
_ext = _b64.b64decode((
{chunks}
), validate=True)
if _hl.sha256(_ext).hexdigest() != "{ext_hash}":
    raise ValueError("Extension module integrity check failed. Reopen an original copy of this notebook.")
_extdir = Path("/content/ckinn_ext")  # outside study/, so the run identity is unchanged
_extdir.mkdir(parents=True, exist_ok=True)
(_extdir / "ckinn_ext.py").write_bytes(_ext)
sys.path.insert(0, str(_extdir))
import ckinn_ext
importlib.reload(ckinn_ext)
if RUN_STAGE in ("ALL", "EXT") and RUN_MODE != "AUDIT":
    print(ckinn_ext.run_all(st))
    for _f in sorted((st.table / "A" / "EXT").glob("*.csv")):
        print(_f.name)
    show(st.table / "A" / "EXT" / "E3_paired_brier_vs_baselines.csv")
    show(st.table / "A" / "EXT" / "E9_lambda_path_expanded.csv")
    show(st.table / "A" / "EXT" / "E10_training_size_summary.csv")
else:
    print("EXT skipped for RUN_STAGE", RUN_STAGE)
"""

new = cells[:10] + [md(RESUME_MD), code(RESUME)] + cells[10:72] + [md(EXT_MD), code(EXT_CODE)] + cells[72:]
nb["cells"] = new
nb["metadata"].setdefault("colab", {})["name"] = OUT.name
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

lines = ["CKINN Objective 1, Colab notebook version 3: every cell in order, ready to paste into Colab",
         f"Source: {OUT.name} (identical text). Create one Colab cell per block below, in order.",
         "Code cells: paste only the lines between the markers (not the marker lines).",
         "Markdown cells: optional (text only); paste into a text cell if you want the headings.",
         "Cell 01 carries the verified software as base64 text: paste it whole; one changed character stops the integrity check.",
         ""]
for i, c in enumerate(new):
    kind = "CODE" if c["cell_type"] == "code" else "MARKDOWN"
    lines += [f"########## CELL {i:02d} | {kind} | start ##########", *src(c).splitlines(), f"########## CELL {i:02d} | end ##########", ""]
TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT.name}: {len(new)} cells; {TXT.name}: {len(lines)} lines; ckinn_ext sha256 {ext_hash}")
