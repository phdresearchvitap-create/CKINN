"""Frozen v2 configuration, sign tables, feature lists and deterministic seeds (protocol_v2.md, sections 3 to 9)."""
import dataclasses
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

SEED = 20261120


@dataclass(frozen=True)
class Config:
    mode: str
    seed: int = SEED
    # development / evaluation split and tuning (v1 split kept; protocol 3, 5)
    dev_frac: float = 0.20
    dev_max_rows: int = 20000
    tune_folds: int = 3
    tune_mlp_configs: int = 32          # M1 and M2: default + 31 random from the 72-point v1 space
    tune_ck_base: int = 24              # CKINN starts from M2's best among its first 24 configurations
    lam_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    lam_reps: int = 2                   # lambda scored on 3-fold CV repeated twice (6 fits)
    tune_tree_configs: int = 16
    # evaluation design (protocol 6)
    n_splits: int = 5
    n_repeats: int = 5
    n_seeds: int = 5
    # networks (v1 settings, protocol 4)
    max_epochs: int = 40
    patience: int = 4
    batch: int = 1024
    val_frac: float = 0.15
    aug_prob: float = 0.5
    p_panel_drop: float = 0.3
    p_feat_drop: float = 0.1
    viol_tol: float = 1e-3
    winsor: Tuple[float, float] = (0.5, 99.5)
    # thresholds and tests (protocol 6 to 8)
    spec_target: float = 0.90
    sens_target: float = 0.80
    fpr_matched: float = 0.10
    ni_margin: float = 0.01
    alpha: float = 0.05
    n_boot: int = 2000
    n_boot_desc: int = 500
    n_boot_band: int = 200
    min_group_events: int = 20
    # missingness (protocol 8)
    miss_mechs: Tuple[str, ...] = ("mcar", "mar", "mnar")
    miss_rates: Tuple[float, ...] = (0.1, 0.3, 0.5)
    primary_miss: Tuple[str, float] = ("mcar", 0.3)
    # censoring-aware analyses
    horizon: float = 3.0
    nagala_horizon2: float = 5.0
    # competitors
    rf_trees: int = 200
    spline_knots: int = 5
    mi_m: int = 5
    mi_iter: int = 10
    # explanations
    n_explain: int = 1000
    ig_steps: int = 32
    n_pd_rows: int = 500
    sparse_quantile: float = 0.95
    # transport and site analyses
    recal_n: int = 1000
    weight_clip: Tuple[float, float] = (0.1, 10.0)
    site_min_events: int = 100
    # decision curves
    dca_b: Tuple[float, float, float] = (0.005, 0.10, 0.005)
    dca_a: Tuple[float, float, float] = (0.01, 0.20, 0.01)
    nb_points_b: Tuple[float, ...] = (0.02, 0.05)
    nb_points_a: Tuple[float, ...] = (0.05, 0.10)
    # smoke-mode subsampling (0 = all rows)
    max_rows: Dict[str, int] = field(default_factory=lambda: {"NAGALA": 0, "RICH": 0, "NHANES": 0})

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)


def make_config(mode: str) -> Config:
    if mode == "full":
        return Config(mode="full")
    if mode == "smoke":
        return Config(mode="smoke", tune_mlp_configs=3, tune_ck_base=2, lam_grid=(0.3, 3.0), lam_reps=1, tune_tree_configs=2,
                      n_splits=3, n_repeats=2, n_seeds=2, max_epochs=6, patience=2, n_boot=200, n_boot_desc=100, n_boot_band=50,
                      rf_trees=30, mi_m=2, mi_iter=3, n_explain=200, ig_steps=8, n_pd_rows=150, recal_n=300, site_min_events=20,
                      dev_max_rows=3000, miss_rates=(0.3,), min_group_events=10,
                      max_rows={"NAGALA": 4000, "RICH": 20000, "NHANES": 6000})
    raise ValueError(mode)


# ---- clinical sign table (v1, unchanged) and the sensitivity tables of protocol 8(e) ----
SIGNS_T2 = {"age_years": 1, "bmi_kgm2": 1, "waist_cm": 1, "sbp_mmHg": 1, "family_history_dm": 1, "tg_mmol": 1,
            "hdl_mmol": -1, "exercise_regular": -1}
SIGNS_T3 = dict(SIGNS_T2, fpg_mmol=1, hba1c_pct=1)
SIGNS_CORE = {"age_years": 1, "bmi_kgm2": 1, "waist_cm": 1, "sbp_mmHg": 1, "tg_mmol": 1}
SIGNS_EXT = dict(SIGNS_T2, dbp_mmHg=1, alt_u_l=1, ggt_u_l=1)

LOW_COST = ["age_years", "sex_female", "bmi_kgm2", "waist_cm", "sbp_mmHg", "dbp_mmHg", "smoker", "alcohol_level",
            "exercise_regular", "family_history_dm"]
PANEL = ["tg_mmol", "hdl_mmol", "ldl_mmol", "tc_mmol", "alt_u_l", "ast_u_l", "ggt_u_l"]
GLYC = ["fpg_mmol", "hba1c_pct"]
ALL_FEATURES = LOW_COST + PANEL + GLYC
BINARY = ("sex_female", "smoker", "exercise_regular", "family_history_dm")
LOG_COLS = ("tg_mmol", "alt_u_l", "ast_u_l", "ggt_u_l")
GROUP_OF = {**{c: "demo" for c in ("age_years", "sex_female", "smoker", "alcohol_level", "family_history_dm", "exercise_regular")},
            **{c: "clin" for c in ("bmi_kgm2", "waist_cm", "sbp_mmHg", "dbp_mmHg")},
            **{c: "panel" for c in PANEL}, **{c: "glyc" for c in GLYC}}
GROUPS = ("demo", "clin", "panel", "glyc")
# Post hoc features (protocol_v2.md section 10, PH): history questions in NHANES, the TG/HDL ratio, a cohort indicator for pooled
# training. They are outside ALL_FEATURES, so no pre-specified feature set or the configuration hash changes.
PH_FEATURES = ("prediabetes_told", "hypertension_told", "tg_hdl_ratio", "cohort_rich")
GROUP_OF.update({"prediabetes_told": "demo", "hypertension_told": "demo", "cohort_rich": "demo", "tg_hdl_ratio": "panel"})
BINARY = BINARY + ("prediabetes_told", "hypertension_told", "cohort_rich")
LOG_COLS = LOG_COLS + ("tg_hdl_ratio",)
RANGES = {"age_years": (18, 100), "bmi_kgm2": (12, 70), "waist_cm": (40, 190), "sbp_mmHg": (60, 260), "dbp_mmHg": (30, 160),
          "tg_mmol": (0.05, 60), "hdl_mmol": (0.1, 10), "ldl_mmol": (0.2, 15), "tc_mmol": (1.0, 20), "alt_u_l": (1, 2000),
          "ast_u_l": (1, 2000), "ggt_u_l": (1, 5000), "fpg_mmol": (2.0, 30), "hba1c_pct": (3.0, 18)}

NN_MODELS = ("M1", "M2", "CKINN")
MAIN_MODELS = ("LR", "cLR", "GAM", "RF", "XGB", "XGB_mono", "LGBM", "LGBM_mono", "M1", "M2", "CKINN")
RUN0_EXTRA = ("MI_LR", "LR_aug", "XGB_aug")
MONO_MODELS = ("cLR", "XGB_mono", "LGBM_mono", "CKINN")

MLP_SPACE = {"hidden": [(64, 32), (128, 64), (256, 128)], "dropout": [0.1, 0.2, 0.3], "lr": [1e-3, 3e-3],
             "weight_decay": [1e-4, 1e-3], "enc_dim": [16, 32]}
MLP_DEFAULT = {"hidden": (128, 64), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "enc_dim": 32}
TREE_SPACE = {"XGB": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [150, 300, 500]},
              "LGBM": {"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [150, 300, 500]}}
TREE_DEFAULT = {"XGB": {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
                "LGBM": {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300}}
C_GRID = [0.01, 0.1, 1.0, 10.0]
RF_GRID = [{"max_depth": d, "min_samples_leaf": l} for d in (6, 10, 14) for l in (10, 20, 50)]


def seed_for(*keys: Any) -> int:
    """Deterministic seed from a tuple of keys (identical to v1)."""
    return zlib.crc32("|".join(map(str, keys)).encode()) % (2 ** 31 - 1)


def config_hash(cfg: Config) -> str:
    frozen = {"config": json.loads(cfg.to_json()), "signs_T2": SIGNS_T2, "signs_T3": SIGNS_T3, "signs_core": SIGNS_CORE,
              "signs_ext": SIGNS_EXT, "main_models": MAIN_MODELS, "run0_extra": RUN0_EXTRA, "mlp_space": MLP_SPACE,
              "tree_space": TREE_SPACE, "c_grid": C_GRID, "rf_grid": RF_GRID,
              "hypotheses": ["H1 AUROC non-inferiority 0.01", "H2 Brier and |slope-1|", "H3 transport gap vs target CV and |CITL|",
                             "H4 across-seed attribution stability", "H5 TPR at FPR 0.10 gaps by sex and age band"],
              "families": ["A: CKINN-M2", "B: CKINN-M1"]}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()
