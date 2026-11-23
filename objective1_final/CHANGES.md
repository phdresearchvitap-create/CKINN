# Changes from the code version of the first full development run

Reference version: the package embedded in `CKINN_Objective1_Colab_STANDARD_v2.ipynb`, which produced the STANDARD development run `96ccd0057ac85891` (stages 5 to 28).

1. **Presentation only.** Comments and docstrings were removed and the code was formatted with black (line length 110). `build_final.py` proves that every cleaned module has the same abstract syntax tree as its source without docstrings. The program is unchanged.
2. **Input requirement and layout.** The proposal PDF is no longer a required input. It entered only the input identity, never an analysis. The three data files remain required and are verified by SHA-256. `config.json` now sits next to the modules, so the integration check looks for it there.
3. **Extension package.** `extension.py` and `verify_extension.py` add the exploratory EXT analyses (E2 to E12) and a generated table that maps the proposal's research questions and hypotheses to the estimates. Rules: `docs/EXT_PRESPECIFICATION.md`.
4. **Options.** `tree_shap` is enabled (random-forest probability SHAP for the explanation stage). The device is fixed to CPU for reproducibility.
5. **Notebook.** The package is written by readable `%%writefile` cells, not by an embedded archive. The architecture figure is drawn by `architecture.py` at run time. A cosmetic LightGBM feature-name warning is filtered; predictions are unaffected (checked). The notebook runs in Colab or locally (`CKINN_ROOT`, `CKINN_RUN_MODE`, `CKINN_RUN_STAGE`).

Because the code and options differ, the run identifier differs from `96ccd0057ac85891`, and a new run fits all models again.
