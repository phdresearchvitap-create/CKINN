# CKINN: clinical-knowledge-informed survival models for incident T2D (RICH + NAGALA)

Real data only. Never create synthetic, simulated or augmented data. Unit tests use toy arrays only.
Locked analysis plan: protocol/protocol_v1.0_LOCKED.md (hash in protocol/LOCK.sha256, git tag protocol-v1.0).
Changes to the plan go in protocol/deviations.md with a reason; never edit the locked file.
Every number in paper/ comes from results/ via scripts/make_report.py (paper/numbers.tex, paper/tables/*.tex). Never type a result by hand.
Cite only references verified through Scite or PubMed; keep them in paper/references.bib.
Style: British (Oxford) spelling, LaTeX with BibTeX, PyTorch for networks. Report unsupported claims as unsupported.
Run: make test; make cv sens loso shift cross (resumable); make report; make paper. Set CKINN_THREADS=1.
Raw data: data/raw (checksums in CHECKSUMS.sha256). Do not commit raw or processed data.
