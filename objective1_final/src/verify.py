import copy, inspect, json, tempfile, zipfile, unittest, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from data import *
from models import *
from stats import *
from runner import Study, model_spec
from analysis import grad_raw, ig, external_contrast, figures, subgroups, explain, updating, evidence


def fake(n=120, cohort="RICH", task="A", seed=42):
    r = np.random.default_rng(seed)
    d = pd.DataFrame(
        dict(
            age=r.integers(20, 76, n),
            sex=r.integers(0, 2, n),
            bmi=r.normal(25, 3, n),
            sbp=r.normal(120, 10, n),
            alt=r.uniform(10, 60, n),
            fpg=r.uniform(4, 6, n),
            tg=r.uniform(0.5, 3, n),
            hdl=r.uniform(0.6, 1.9, n),
            dbp=r.normal(75, 5, n),
            smoking=r.integers(0, 2, n),
            tc=r.normal(5, 0.5, n),
            ast=r.uniform(12, 45, n),
            site=np.arange(n) % 3,
            cohort=cohort,
            id=[f"{cohort}-{i}" for i in range(n)],
        )
    )
    d["event"] = np.tile([1, 0, 0, 1, 0, 0], n // 6 + 1)[:n]
    if task == "A":
        d["time"] = np.tile([2, 2, 4, 5, 5, 1], n // 6 + 1)[:n]
    d.loc[d.index[::9], "hdl"] = np.nan
    return d


H = dict(
    features=CORE,
    masks=True,
    branch=8,
    fusion=[32, 16],
    lr=0.001,
    decay=0.0001,
    batch=24,
    dropout=0.1,
    directions=["fpg", "bmi"],
)
H["lambda"] = 10.0


class Checks(unittest.TestCase):

    def test_boundaries_ties_hand_weights(self):
        d = pd.DataFrame(dict(time=[1, 1, 2, 3, 3, 4, 5], event=[1, 0, 0, 1, 0, 1, 0]))
        y, w, g = weights(d)
        np.testing.assert_allclose(w, [1, 0, 0, 1.5, 0, 2.25, 2.25])
        np.testing.assert_equal(status(d), [1, -1, -1, 1, -1, 0, 0])
        self.assertAlmostEqual(g, 4 / 9)
        self.assertAlmostEqual(ReverseKM(d.time, d.event).at(3, True), 2 / 3)

    def test_loss_denominator_and_penalty(self):
        y = np.array([1.0, 0.0, 0.0])
        w = np.array([2.0, 1.5, 0.0])
        self.assertAlmostEqual(brier(y, w, np.array([0.5, 0.5, 0.9])), 0.875 / 3)
        pp = Prep(CORE, "A").fit(fake())
        x = torch.zeros((3, 16), requires_grad=True)
        inside = torch.ones((3, 8), dtype=torch.bool)
        a = torch.tensor(-0.6, requires_grad=True)
        h = dict(H, directions=["bmi"])

        def objective(lam):
            logits = a * x[:, 2]
            return loss_terms(
                logits,
                x,
                inside,
                torch.tensor(y),
                torch.tensor(w),
                torch.ones(3),
                pp,
                dict(h, **{"lambda": lam}),
            )

        np.testing.assert_allclose(objective(0).item(), 3.5 * np.log(2) / 3)
        g0 = torch.autograd.grad(objective(0), a)[0]
        g1 = torch.autograd.grad(objective(3), a)[0]
        self.assertNotEqual(g0.item(), g1.item())
        inside[:, 2] = False
        self.assertAlmostEqual(objective(3).item(), objective(0).item())

    def test_derivative_sign_masks_and_raw_finite_difference(self):
        d = fake()
        pp = Prep(CORE, "A").fit(d)
        x = torch.tensor(pp.transform(d)[0], requires_grad=True)
        p = (x[:, 7] * 0.5).sigmoid()
        ok = torch.from_numpy(pp.transform(d)[1])
        observed = p.detach().numpy()
        expected = np.mean((0.5 * observed[ok[:, 7]] * (1 - observed[ok[:, 7]])) ** 2)
        self.assertAlmostEqual(clinical(p, x, ok, pp.names, ["hdl"]).item(), expected, places=6)
        seed_all(27)
        net = Fitted("net", pp, Net(pp, H).eval())
        dd = d[d.hdl.notna()].copy()
        up = dd.copy()
        down = dd.copy()
        up["hdl"] += 0.002
        down["hdl"] -= 0.002
        np.testing.assert_allclose(
            (net.predict(up) - net.predict(down)) / 0.004, grad_raw([net], dd, "hdl"), atol=4e-05
        )
        a, base = ig(net, dd.iloc[:4])
        np.testing.assert_allclose(base + a.sum(1), net.predict(dd.iloc[:4]), atol=2e-05)

    def test_training_preprocessing_and_forbidden(self):
        d = fake()
        tr = d.iloc[:90]
        va = d.iloc[90:].copy()
        va["bmi"] = 8000
        pp = Prep(CORE, "A").fit(tr)
        np.testing.assert_allclose(pp.med[2], tr.bmi.median())
        pp.transform(va)
        np.testing.assert_allclose(pp.med[2], tr.bmi.median())
        x, ok = pp.transform(d)
        self.assertEqual(x[0, 15], 1)
        self.assertFalse(ok[0, 7])
        for task, bad in [
            ("A", ["event", "time", "site", "id", "FPG of final visit(mmol/L)"]),
            ("B", ["a1c", "HbA1c", "fpg", "FastingGlucose", "Diabetes_self", "Diabetes_lab"]),
        ]:
            for name in bad:
                with self.assertRaises(ValueError):
                    Prep(["age", name], task)
        for a, b in split_indices(d, 3, 11):
            self.assertFalse(set(d.id.iloc[a]) & set(d.id.iloc[b]))

    def test_schema_units_and_flow(self):
        src = pd.DataFrame({k: [10.0, 11.0, 12.0, 13.0] for k in J.values()})
        src[J["age"]] = [19, 45, 50, 55]
        src[J["sex"]] = [1, 0, 1, 0]
        src[J["time"]] = [365.25, 730.5, 1095.75, 1461.0]
        src[J["event"]] = [0, 1, 3, 0]
        src[J["a1c"]] = [5, 5.5, 5.5, 6.5]
        for k, (f, m) in CONVERT.items():
            src[f] = 90.0
            src[J[k]] = 90 * m
        src.loc[1, J["hdl"]] = 0
        d, checks = harmonize(src, "NAGALA")
        self.assertEqual(d.time.tolist(), [1, 2, 3, 4])
        self.assertAlmostEqual(d.hdl.iloc[1], 90 * 0.02586)
        self.assertEqual(d.sex.tolist(), [1, 0, 1, 0])
        good, lg = eligible(d)
        self.assertEqual(len(good), 1)
        self.assertEqual(good.event.iloc[0], 1)
        self.assertEqual([x["n"] for x in lg], [4, 3, 3, 2, 2, 1, 1])
        with self.assertRaises(ValueError):
            mapped(src.drop(columns=J["event"]), J)
        nh = pd.DataFrame(
            dict(
                Age=[20, 19, 40, 52],
                Sex=[0, 1, 1, 0],
                BMI=[24] * 4,
                SystolicBP=[115] * 4,
                HbA1c=[6.5, 7, 6.4, np.nan],
                Diabetes_self=[0, 0, 1, 0],
            )
        )
        s, fl = screen_candidates(nh, "period", 0)
        self.assertEqual((len(s), s.event.iloc[0]), (1, 1))

    def test_integrity_and_zip(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.pkl"
            dump(f, {"a": 17})
            self.assertEqual(restore(f), {"a": 17})
            f.write_bytes(f.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                restore(f)
            z = Path(td) / "z.zip"
            for name in ["../esc.csv", "a/../../esc", "C:/esc", "/esc"]:
                with zipfile.ZipFile(z, "w") as a:
                    a.writestr(name, "x")
                with self.assertRaises(ValueError):
                    safe_unpack(z, Path(td) / "inside")

    def test_matched_zero_and_parameter_count(self):
        d = fake(60)
        pp = Prep(CORE, "A").fit(d)
        spec1 = model_spec("M1", dict(H), "A")
        spec0 = model_spec("fixed_zero", dict(H), "A")
        self.assertEqual(spec0, spec1)
        a, _ = train_net(d, None, pp, spec1, 10, 1, 2, "cpu", None, {})
        b, _ = train_net(d, None, pp, spec0, 10, 1, 2, "cpu", None, {})
        self.assertEqual(sum((p.numel() for p in a.parameters())), 1497)
        np.testing.assert_array_equal(torch_risk(a, pp.transform(d)[0]), torch_risk(b, pp.transform(d)[0]))

    def test_epoch_resume_identity(self):
        d = fake(60)
        pp = Prep(CORE, "A").fit(d)
        kwargs = dict(
            d=d,
            valid=None,
            prep=pp,
            h=H,
            seed=73,
            epochs=3,
            patience=2,
            device="cpu",
            identity={"fold": "artificial"},
        )
        full, _ = train_net(**kwargs, checkpoint=None)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "epoch.pkl"
            with self.assertRaises(InterruptedError):
                train_net(**kwargs, checkpoint=f, interrupt_epoch=1)
            with self.assertRaises(ValueError):
                train_net(**dict(kwargs, identity={"fold": "wrong"}), checkpoint=f)
            resumed, _ = train_net(**kwargs, checkpoint=f)
            for a, b in zip(full.parameters(), resumed.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_estimator_weight_response(self):
        d = fake(120)
        X = Prep(CORE, "A").fit(d).transform(d)[0]
        y = np.arange(len(d)) % 2
        h = dict(C=1.0, trees=10, depth=3)
        for n in ["LR", "RF", "XGB", "LGB", "SVM"]:

            def pred(w):
                m = fit_weighted(estimator(n, h, 10), X, y, w)
                return m.decision_function(X) if n == "SVM" else m.predict_proba(X)[:, 1]

            self.assertGreater(np.max(abs(pred(np.ones(len(y))) - pred(1 + 8 * y))), 1e-05, n)

    def test_site_weights_and_zero_support(self):
        d = fake()
        self.assertAlmostEqual(side_weights(d).mean(), 1)
        with self.assertRaises(ValueError):
            weights(d.assign(time=1.0))
        d["site"] = np.where(d.index < 5, "rare", "main")
        self.assertTrue(np.isfinite(weights(d, True)[1]).all())


def integration():
    import runner
    from analysis import export, internal_summary

    root0 = Path(__file__).resolve().parent
    cfg = json.loads((root0 / "config.json").read_text())
    cfg["profile"] = "SMOKE"
    cfg["device"] = "cpu"
    cfg["profiles"]["SMOKE"].update(epochs=1, bootstrap=2)
    cfg["profiles"]["SMOKE"]["trials"] = 1
    cfg["min_cases"] = 3
    cfg["min_controls"] = 3
    cfg["update_n"] = 60
    cfg["options"]["ablations"] = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ident = root / "synthetic_input.txt"
        ident.write_text("artificial only")
        st = Study(root, cfg, paths={"artificial": ident})
        for name, task, s in [
            ("RICH", "A", 7),
            ("NAGALA", "A", 14),
            (PERIODS[0], "B", 21),
            (PERIODS[1], "B", 28),
        ]:
            st.mem[name] = fake(180, name, task, s)
        for task in ["A", "B"]:
            if task == "A":
                st.internal(task)
                internal_summary(st)
            plans = st.final_plan(task)
            assert plans["M3"]["h"]["lambda"] == 0
            st.freeze(task)
            with patch.object(
                runner, "fit_model", side_effect=AssertionError("external scoring attempted training")
            ), patch.object(Prep, "fit", side_effect=AssertionError("external preprocessing refit")):
                st.score(task)
            d, p, _ = st.results(task)
            np.testing.assert_array_equal(p["M1"], p["M2"])
            selfsame = st.frozen(task)[1]
            assert len(d) == 180 and len(selfsame["files"]) >= 7
            external_contrast(st, task)
            if task == "A":
                zero, positive = [plans[n] for n in ["fixed_zero", "fixed_positive"]]
                assert zero["h"]["lambda"] == 0 and positive["h"]["lambda"] > 0
                assert zero["epochs"] == positive["epochs"]
                assert {k: v for k, v in positive["h"].items() if k != "lambda"} == {
                    k: v for k, v in zero["h"].items() if k != "lambda"
                }
                assert plans["expanded_prior"]["h"]["lambda"] == positive["h"]["lambda"]
                contrast = external_contrast(st, matched=True)
                assert contrast["delta_valid_boot"] == cfg["profiles"]["SMOKE"]["bootstrap"]
                assert (
                    abs(contrast["delta"] - (contrast["candidate_brier"] - contrast["baseline_brier"]))
                    < 1e-14
                )
                assert not np.array_equal(p["fixed_zero"], p["fixed_positive"])
            st.selection_table(task, frozen=True)
            figures(st, task)
            subgroups(st, task)
            explain(st, task)
        updating(st)
        for name in ["RICH_broad", "RICH_alt0"]:
            st.mem[name] = st.mem["RICH"].copy()
        st.internal("A", source="RICH_broad")
        st.internal("A", source="RICH_alt0")
        st.internal("A", site=True)
        out = export(st, root0)
        with zipfile.ZipFile(out) as z:
            assert not any(("checkpoints/" in n or n.endswith(".pkl") for n in z.namelist()))
    return "passed synthetic nested, frozen and matched positive penalty comparisons, subgroup, explanation, target update and export paths"


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Software checks failed")
    return {"passed": result.testsRun, "cohort_fits": 0}


if __name__ == "__main__":
    print(run_checks())
    if "--integration" in sys.argv:
        print(integration())
