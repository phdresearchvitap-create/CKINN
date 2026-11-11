# Setting up Claude Code for the CKINN thesis objective

## Prerequisites
1. Claude Code installed (npm i -g @anthropic-ai/claude-code, Node 18+) and logged in with your Claude account.
2. Python 3.10+ with: numpy pandas scipy scikit-learn scikit-survival lifelines xgboost torch matplotlib openpyxl jupyter pytest. Install with `pip install -r requirements.txt`.
3. A TeX distribution with latexmk (TeX Live or MiKTeX).
4. git. The raw files RCHealthCareData-20180820.xlsx and DatasetIntJObesity.xlsx in data/raw/ (Dryad: doi 10.5061/dryad.ft8750v and 10.5061/dryad.8q0p192).
5. 8 GB RAM, 2+ cores. No GPU needed (a full model fit is about 6 seconds).
6. Optional connectors: PubMed (works), Scite (needs paid plan), Consensus.

## What goes where
- CLAUDE.md: always-loaded rules (real data only, protocol lock, no hand-typed numbers).
- .claude/settings.json: permissions; the locked protocol cannot be edited.
- .claude/skills/ckinn-protocol: rules loaded when touching analysis or results text.
- .claude/agents: stats-reviewer, reference-checker (fresh context, read-only).
- .claude/commands: /run-stage, /audit.
- .claude/rules/writing.md: style and limitations.
- Memory (Claude Code auto memory, view with /memory): let it keep supervisor, journal target, and your corrections. Do not store data or credentials.

## Models and effort
Fable 5.2 (high or max): protocol decisions, interpretation, claims, discussion, statistical audit.
Sonnet 5.5 (medium): code, notebooks, running stages, LaTeX fixes. Switch with /model.

## Session 1: start from cell 1
1. `git clone` or unzip, `cd` into the folder, run `claude`.
2. Paste: "Read CLAUDE.md and protocol/protocol_v1.0_LOCKED.md. Open notebooks/01_data_harmonization_profile.ipynb and run it cell by cell from cell 1. After each cell, report only anything that disagrees with the protocol numbers. Do not run notebook 04 section 7."
3. Then notebooks 02, 03, 04 in order. Cross-cohort (04 section 7) only after you have read protocol/deviations.md.
4. `make cv sens loso shift cross` runs everything resumable; `make report`; `make paper`.

## Notebook map (56 cells: 29 code, 27 markdown)
01 harmonise and profile: 14 cells (7 code). 02 baselines on RICH: 11 (6). 03 CKINN on RICH: 14 (7). 04 transport: 17 (9; last code cell is lock-guarded).

## Things only you can supply
Target journal, supervisor's choice of headline tier (assumed T2), data-use or ethics wording, author details.
