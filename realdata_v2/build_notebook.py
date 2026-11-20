"""Build CKINN_vs_MLP_real_data_final.ipynb from notebook_cells.py and the library files in ckv2/ and tests/.

Cell markers: '# %%' (code), '# %% [markdown]' (markdown, lines start with '# '), '# %% [writefile <path>]' (a %%writefile cell
that writes the repository file <path> into ckv2_lib/<path>, so the notebook is self-contained).
Usage: python build_notebook.py [--freeze HASH]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "notebook_cells.py"
NB = HERE / "CKINN_vs_MLP_real_data_final.ipynb"


def cells_from_source(text: str):
    blocks = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for i in range(1, len(blocks), 2):
        tag, body = blocks[i].strip(), blocks[i + 1].strip("\n")
        if tag == "[markdown]":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        elif tag.startswith("[writefile"):
            rel = tag[len("[writefile"):].strip(" ]")
            content = (HERE / rel).read_text()
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": f"%%writefile ckv2_lib/{rel}\n{content}"})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
    return cells


def main():
    text = SRC.read_text()
    if "--freeze" in sys.argv:
        h = sys.argv[sys.argv.index("--freeze") + 1]
        text = text.replace('EXPECTED_FULL_HASH = "PENDING-FILLED-BY-FREEZE-STEP"', f'EXPECTED_FULL_HASH = "{h}"')
        SRC.write_text(text)
    cells = cells_from_source(text)
    for c in cells:
        src = c["source"]
        c["source"] = [ln for ln in src.splitlines(keepends=True)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {NB.name}: {len(cells)} cells")


if __name__ == "__main__":
    main()
