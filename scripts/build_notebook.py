"""Build notebooks/CKINN_colab.ipynb and notebooks/CKINN_colab_cells.txt from the cell sources in notebooks/cells/stage*.py.

The sources use the percent format: a line starting with '# %%' opens a code cell and '# %% [markdown]' a markdown
cell. The notebook is written without outputs; it is executed by the investigator, cell by cell.
Usage: python scripts/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
for src in sorted((ROOT / 'notebooks' / 'cells').glob('stage*.py')):
    kind, buf = None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == 'markdown':
            lines = [ln[2:] if ln.startswith('# ') else ln[1:] if ln.startswith('#') else ln for ln in lines]
        text = '\n'.join(lines)
        cell = {'cell_type': kind, 'id': f'cell-{len(cells) + 1:02d}', 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in src.read_text(encoding='utf-8').splitlines():
        if line.startswith('# %%'):
            flush()
            kind, buf = ('markdown' if '[markdown]' in line else 'code'), []
        else:
            buf.append(line)
    flush()

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks' / 'CKINN_colab.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
txt = ROOT / 'notebooks' / 'CKINN_colab_cells.txt'
txt.write_text(''.join(src.read_text(encoding='utf-8').rstrip() + '\n\n' for src in sorted((ROOT / 'notebooks' / 'cells').glob('stage*.py'))), encoding='utf-8')
print(f'wrote {out.relative_to(ROOT)} and {txt.relative_to(ROOT)}: {len(cells)} cells '
      f"({sum(c['cell_type'] == 'code' for c in cells)} code, {sum(c['cell_type'] == 'markdown' for c in cells)} markdown)")
