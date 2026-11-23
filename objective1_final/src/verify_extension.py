import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import extension as X
from verify import fake
from runner import Study
from stats import weights, brier


class ExtChecks(unittest.TestCase):

    def test_net_benefit_hand(self):
        y = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
        w = np.array([2.0, 1.0, 1.0, 1.0, 0.0])
        p = np.array([0.3, 0.3, 0.01, 0.02, 0.5])
        t = 0.1
        self.assertAlmostEqual(X._nb(y, w, p, t, 5), (2 - t / (1 - t) * 1) / 5)

    def test_ipa_reference(self):
        d = pd.DataFrame(dict(time=[1, 2, 4, 5, 2.5, 6], event=[1, 0, 0, 1, 1, 0]))
        y, w, _ = weights(d)
        ref = np.full(len(d), np.average(y, weights=w))
        self.assertAlmostEqual(1 - brier(y, w, ref) / brier(y, w, ref), 0.0)

    def test_plans_from_selection(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "s.csv"
            pd.DataFrame(
                [
                    dict(
                        model="M3",
                        lambda_value=10,
                        source_cv_brier=0.5,
                        threshold=0.0125,
                        candidate_index=2,
                        epochs=17.0,
                        role="",
                        settings=json.dumps({"lambda": 10, "lr": 0.001}),
                    ),
                    dict(
                        model="LR",
                        lambda_value=None,
                        source_cv_brier=0.6,
                        threshold=0.013,
                        candidate_index=1,
                        epochs=None,
                        role="",
                        settings=json.dumps({"C": 0.1}),
                    ),
                ]
            ).to_csv(f, index=False)
            p = X.plans_from_selection(f)
            self.assertEqual(p["M3"]["epochs"], 17)
            self.assertEqual(p["M3"]["h"]["lambda"], 10)
            self.assertEqual(p["LR"]["epochs"], 1)
            self.assertAlmostEqual(p["LR"]["threshold"], 0.013)


def integration():
    cfg = json.loads((PKG / "config.json").read_text())
    cfg.update(profile="SMOKE", stage="ALL", device="cpu")
    cfg["profiles"]["SMOKE"].update(epochs=1, bootstrap=2, trials=1)
    cfg["min_cases"] = cfg["min_controls"] = 3
    cfg["options"]["ablations"] = True
    X.SIZE_LAMBDAS = [10.0]
    X.LAMBDA_PATH = [0.0, 1000.0]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ident = root / "toy_input.txt"
        ident.write_text("toy records from verify.fake only")
        st = Study(root, cfg, paths={"toy": ident})
        st.mem["RICH"] = fake(3000, "RICH", "A", 7)
        st.mem["NAGALA"] = fake(600, "NAGALA", "A", 14)
        st.internal("A")
        st.freeze("A")
        st.score("A")
        t = X.run_all(st)
        files = sorted((p.name for p in (st.table / "A" / "EXT").glob("*.csv")))
        expected = {
            "E2_penalty_diagnostics.csv",
            "E3_paired_brier_vs_baselines.csv",
            "E4_index_of_prediction_accuracy.csv",
            "E5_net_benefit.csv",
            "E6_estimand_aligned_nagala.csv",
            "E7_nagala_hdl_missing.csv",
            "E8_directions_six.csv",
            "E9_lambda_path_expanded.csv",
            "E10_training_size_draws.csv",
            "E10_training_size_summary.csv",
            "E11_without_sites_14_16.csv",
            "EXT_external_subgroups.csv",
            "EXT_runtime.csv",
        }
        assert expected <= set(files), expected - set(files)
        e9 = pd.read_csv(st.table / "A" / "EXT" / "E9_lambda_path_expanded.csv")
        assert e9.selected_by_rich_inner_cv.sum() == 1
        before = pd.read_csv(st.table / "A" / "EXT" / "E10_training_size_draws.csv")
        X.run_all(st)
        after = pd.read_csv(st.table / "A" / "EXT" / "E10_training_size_draws.csv")
        pd.testing.assert_frame_equal(before, after)
        return f"EXT integration passed on toy records: {len(files)} tables; runtimes {t}"


def run_checks():
    res = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ExtChecks)
    )
    if not res.wasSuccessful():
        raise AssertionError("Extension checks failed")
    return {"passed": res.testsRun}


if __name__ == "__main__":
    print(run_checks())
    print(integration())
