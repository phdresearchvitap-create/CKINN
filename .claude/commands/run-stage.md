---
description: Run one resumable stage (cv, sens, loso, shift, cross) and report progress
argument-hint: <stage>
---
Run `CKINN_THREADS=1 python scripts/run_experiments.py $ARGUMENTS` in the background, log to logs/, poll until finished, then run `make report` and summarise only what changed in results/ and paper/tables/. Never delete results/parts; the runner skips finished units.
