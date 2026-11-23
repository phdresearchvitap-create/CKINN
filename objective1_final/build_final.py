"""Build the reviewer notebook CKINN_Objective1_Final.ipynb and its cell-by-cell text file.

Sources:
  - the analysis package embedded (base64) in the submitted STANDARD_v2 notebook (objective1_v3/evidence/), which
    produced Colab run 96ccd0057ac85891;
  - the extension module objective1_v3/ext/ckinn_ext.py and its checks.
Steps:
  1. decode the package; apply the one documented functional change (the proposal PDF is no longer a required input);
  2. remove comments and docstrings, format with black, and prove that each cleaned module has the same abstract
     syntax tree as its source without docstrings (same program, different presentation);
  3. write the cleaned modules to src/ and build the notebook, in which every module is a readable %%writefile cell.
Run: python build_final.py   (needs black)
"""
import ast, base64, io, json, zipfile
from pathlib import Path
import black

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "objective1_v3"
SRC_NB = V3 / "evidence" / "CKINN_Objective1_Colab_STANDARD_v2_submitted.ipynb"
OUT_NB = HERE / "CKINN_Objective1_Final.ipynb"
OUT_TXT = HERE / "CKINN_Objective1_Final_cells.txt"
SRC = HERE / "src"
MODE = black.Mode(line_length=110)

PATCHES = {
    "verify.py": [("    root0=Path(__file__).resolve().parents[1]\n", "    root0=Path(__file__).resolve().parent\n")],
    "data.py": [(
        '    if set(found)!=set(HASH): raise ValueError(f"Missing verified bytes for {sorted(set(HASH)-set(found))}. See README.")\n',
        '    required={"RICH","NAGALA","NHANES"}\n'
        '    if not required<=set(found): raise ValueError(f"Missing verified bytes for {sorted(required-set(found))}. See README.")\n')],
}
ORDER = ["data.py", "dictionary.py", "stats.py", "models.py", "architecture.py", "runner.py", "analysis.py", "stages.py",
         "verify.py", "extension.py", "verify_extension.py"]


def bundle():
    nb = json.loads(SRC_NB.read_text())
    cell = "".join(nb["cells"][3]["source"])
    lines = cell.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("_bundled = base64.b64decode(("))
    end = next(i for i, l in enumerate(lines) if l.startswith("), validate=True)"))
    raw = base64.b64decode("".join(l.strip().strip("'") for l in lines[start + 1:end]), validate=True)
    z = zipfile.ZipFile(io.BytesIO(raw))
    return {n.split("/", 1)[1]: z.read(n).decode("utf-8") for n in z.namelist() if n.startswith("study/") and n.endswith(".py")}, json.loads(z.read("config.json"))


def drop_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def clean(text):
    out = black.format_str(ast.unparse(drop_docstrings(ast.parse(text))), mode=MODE)
    assert ast.dump(ast.parse(out)) == ast.dump(drop_docstrings(ast.parse(text))), "cleaning changed the program"
    return out


def modules():
    mods, cfg = bundle()
    for name, reps in PATCHES.items():
        for a, b in reps:
            assert mods[name].count(a) == 1, (name, a[:60])
            mods[name] = mods[name].replace(a, b)
    ext = (V3 / "ext" / "ckinn_ext.py").read_text()
    mods["extension.py"] = ext
    test = (V3 / "ext" / "test_ext.py").read_text()
    test = test.replace("import ckinn_ext as X", "import extension as X")
    test = test.replace('PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")\nsys.path.insert(0, str(PKG / "study"))\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n',
                        'PKG = Path(__file__).resolve().parent\nsys.path.insert(0, str(PKG))\n')
    test = test.replace('cfg = json.loads((PKG / "config.json").read_text())', 'cfg = json.loads((PKG / "config.json").read_text())')
    test = test.replace('''if __name__ == "__main__":
    res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtChecks))
    if not res.wasSuccessful():
        raise SystemExit(1)
    print(integration())''', '''def run_checks():
    res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtChecks))
    if not res.wasSuccessful():
        raise AssertionError("Extension checks failed")
    return {"passed": res.testsRun}


if __name__ == "__main__":
    print(run_checks())
    print(integration())''')
    assert "import extension as X" in test and "def run_checks" in test
    mods["verify_extension.py"] = test
    cleaned = {n: clean(mods[n]) for n in ORDER}
    return cleaned, cfg


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cleaned, cfg = modules()
SRC.mkdir(exist_ok=True)
for n, t in cleaned.items():
    (SRC / n).write_text(t, encoding="utf-8")
(SRC / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

import sys
sys.path.insert(0, str(HERE))
import notebook_text as T
cells = [md(T.INTRO), md(T.SETTINGS_MD), code(T.SETTINGS_CODE)]
for n in ORDER:
    cells += [md(T.MODULES[n]), code(f"%%writefile ckinn_obj1/{n}\n" + cleaned[n])]
cells += [md(T.CONFIG_MD), code("%%writefile ckinn_obj1/config.json\n" + json.dumps(cfg, indent=2))]
for m, c in T.STAGES:
    cells += [md(m), code(c)]
nb = {"cells": cells, "metadata": {"colab": {"name": OUT_NB.name, "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        body = "".join(c["source"])
        lines = [l for l in body.splitlines() if not l.startswith("%%writefile")]
        assert not any(l.lstrip().startswith("#") for l in lines), body[:80]
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

lines = ["CKINN Objective 1: reviewer notebook, every cell in order",
         f"Source: {OUT_NB.name} (identical text). Create one Colab cell per block, in order.",
         "Code cells: paste the lines between the markers, without the marker lines. Cells that start with %%writefile write a module of the analysis package; keep that first line.",
         "Text cells: paste into a Colab text cell.", ""]
for i, c in enumerate(nb["cells"]):
    kind = "CODE" if c["cell_type"] == "code" else "TEXT"
    lines += [f"########## CELL {i:02d} | {kind} | start ##########", *"".join(c["source"]).splitlines(), f"########## CELL {i:02d} | end ##########", ""]
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{OUT_NB.name}: {len(cells)} cells; {OUT_TXT.name}: {len(lines)} lines; modules: {', '.join(ORDER)}")
