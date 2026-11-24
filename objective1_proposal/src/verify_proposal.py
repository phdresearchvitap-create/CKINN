import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import proposal as P
from verify import fake
from runner import Study


class ProposalChecks(unittest.TestCase):

    def test_one_se_rule_prefers_largest_lambda_within_one_se(self):
        base = np.linspace(0.5, 1.5, 400)
        wiggle = np.where(np.arange(400) % 2 == 0, 0.01, -0.01)
        losses = {0.0: base, 10.0: base + 0.0001 + wiggle, 100.0: base + 1.0}
        chosen, rows = P.one_se_choice(losses)
        self.assertEqual(chosen, 10.0)
        self.assertTrue(next((x for x in rows if x["lambda_value"] == 100.0))["within_one_se"] is False)

    def test_one_se_rule_keeps_best_when_others_worse(self):
        base = np.linspace(0, 1, 200)
        chosen, _ = P.one_se_choice({0.0: base, 10.0: base + 0.5})
        self.assertEqual(chosen, 0.0)

    def test_dersimonian_laird_hand_values(self):
        y, v = (np.array([1.0, 2.0, 4.0]), np.array([1.0, 1.0, 1.0]))
        res = P.dersimonian_laird(y, v)
        q = ((y - y.mean()) ** 2).sum()
        tau2 = max(0, (q - 2) / (3 - 3 / 3))
        self.assertAlmostEqual(res["tau2"], tau2)
        self.assertAlmostEqual(res["pooled"], y.mean())
        self.assertAlmostEqual(res["hi"] - res["pooled"], 1.959964 * np.sqrt((1 + tau2) / 3))

    def test_net_benefit_hand(self):
        y, w, p = (
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([2.0, 1.0, 1.0, 0.0]),
            np.array([0.3, 0.3, 0.01, 0.5]),
        )
        self.assertAlmostEqual(P._nb(y, w, p, 0.1), (2 - 0.1 / 0.9 * 1) / 4)

    def test_path_h_canonical_zero(self):
        h = P.path_h({"lr": 0.001}, ["fpg", "bmi"], 0)
        self.assertEqual(h["directions"], [])
        self.assertEqual(P.path_h({"lr": 0.001}, ["fpg"], 10)["directions"], ["fpg"])


def integration():
    import figures
    import xai

    cfg = json.loads((PKG / "config.json").read_text())
    cfg.update(profile="SMOKE", stage="ALL", device="cpu")
    cfg["profiles"]["SMOKE"].update(epochs=1, bootstrap=3, trials=1, comparator_trials=1, final_seeds=2)
    cfg["min_cases"] = cfg["min_controls"] = 3
    cfg["ckinn"]["lambda"] = [0, 1000]
    cfg["validation"].update(explain_n=30, background_n=10, shap_max_evals=17)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ident = root / "toy_input.txt"
        ident.write_text("toy records only")
        st = Study(root, cfg, paths={"toy": ident})
        rich = fake(1200, "RICH", "A", 7)
        rich["site"] = np.random.default_rng(5).integers(0, 3, len(rich))
        st.mem.update(
            RICH=rich, NAGALA=fake(600, "NAGALA", "A", 14), RICH_broad=rich.copy(), RICH_alt0=rich.copy()
        )
        st.mem.update(
            {"2017-2020": fake(400, "2017-2020", "B", 21), "2021-2023": fake(400, "2021-2023", "B", 28)}
        )
        for task in ["A", "R"]:
            P.nested(st, task)
            P.final_plans(st, task)
            st.freeze(task)
            st.score(task)
            P.external_report(st, task)
            xai.directions(st, task)
        P.final_plans(st, "B")
        st.freeze("B")
        st.score("B")
        P.external_report(st, "B")
        P.loso(st)
        P.sensitivity(st)
        P.fairness(st, "A")
        xai.explain(st, "A")
        P.size_curve(st)
        figures.all_figures(st)
        out = P.export(st, PKG)
        import zipfile

        names = zipfile.ZipFile(out).namelist()
        assert not any((n.endswith(".pkl") or "checkpoints/" in n for n in names))
        ta = P.table_dir(st, "A")
        for f in [
            "internal_pooled_oof.csv",
            "external_metrics_ci.csv",
            "contrast_ckinn_vs_m1.csv",
            "loso_pooled.csv",
            "fairness_gaps.csv",
            "shap_importance.csv",
            "counterfactual_summary.csv",
            "size_curve_summary.csv",
            "proposal_questions_evidence.csv",
            "lambda_path_final.csv",
        ]:
            assert (ta / f).exists(), f
        assert (P.table_dir(st, "R") / "contrast_ckinn_vs_m1.csv").exists()
        assert (st.fig / "A" / "v4" / "calibration.png").exists()
        plans = st.frozen("A")[1]["plans"]
        base = {k: v for k, v in plans["M1"]["h"].items() if k not in ("lambda", "directions")}
        ck = {k: v for k, v in plans["CKINN"]["h"].items() if k not in ("lambda", "directions")}
        assert base == ck and plans["CKINN"]["epochs"] == plans["M1"]["epochs"]
        return f"v4 integration passed on toy records: {len(names)} files in the aggregate ZIP"


def run_checks():
    res = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ProposalChecks)
    )
    if not res.wasSuccessful():
        raise AssertionError("Protocol v4 checks failed")
    return {"passed": res.testsRun}


if __name__ == "__main__":
    print(run_checks())
    print(integration())
