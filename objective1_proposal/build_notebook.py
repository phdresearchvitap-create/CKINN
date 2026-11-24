import ast
import json
import sys
from pathlib import Path
import black

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT_NB = HERE / "CKINN_Objective1_Proposal.ipynb"
OUT_TXT = HERE / "CKINN_Objective1_Proposal_cells.txt"
MODE = black.Mode(line_length=110)
ORDER = ["data.py", "dictionary.py", "stats.py", "models.py", "architecture.py", "runner.py", "analysis.py", "stages.py", "verify.py",
         "proposal.py", "xai.py", "figures.py", "verify_proposal.py"]


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


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


sys.path.insert(0, str(HERE))
import notebook_text as T

cleaned = {n: clean((SRC / n).read_text()) for n in ORDER}
for n, t in cleaned.items():
    (SRC / n).write_text(t, encoding="utf-8")
cfg = json.loads((SRC / "config.json").read_text())
cells = [md(T.INTRO), md(T.SETTINGS_MD), code(T.SETTINGS_CODE)]
for n in ORDER:
    cells += [md(T.MODULES[n]), code(f"%%writefile ckinn_obj1/{n}\n" + cleaned[n])]
cells += [md(T.CONFIG_MD), code("%%writefile ckinn_obj1/config.json\n" + json.dumps(cfg, indent=2))]
for m, c in T.STAGES:
    cells += [md(m), code(c)]
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:03d}"
    if c["cell_type"] == "code":
        body = [l for l in "".join(c["source"]).splitlines() if not l.startswith("%%writefile")]
        assert not any(l.lstrip().startswith("#") for l in body), "".join(c["source"])[:80]
nb = {"cells": cells, "metadata": {"colab": {"name": OUT_NB.name, "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
lines = ["CKINN Objective 1, protocol v4 (aligned with the research proposal): every notebook cell in order",
         f"Source: {OUT_NB.name} (identical text). Create one Colab cell per block, in order.",
         "Code cells: paste the lines between the markers, without the marker lines. Cells that start with %%writefile write a module of the analysis package; keep that first line.",
         "Text cells: paste into a Colab text cell.", ""]
for i, c in enumerate(cells):
    kind = "CODE" if c["cell_type"] == "code" else "TEXT"
    lines += [f"########## CELL {i:02d} | {kind} | start ##########", *"".join(c["source"]).splitlines(), f"########## CELL {i:02d} | end ##########", ""]
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{OUT_NB.name}: {len(cells)} cells; {OUT_TXT.name}: {len(lines)} lines")
