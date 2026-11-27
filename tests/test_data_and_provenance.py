"""Regression tests for OP06 (data boundary/1-bar), OP07 (Greeks/offline demo), OP08 (provenance/compare)."""
from datetime import datetime
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bench
import data
import pricing
import run_ab
import scenarios

FIXTURE_PATH = os.path.join(REPO_ROOT, "fixtures", "sample_prices.json")


class _FakeSeries:
    """Minimal Series-like object whose .squeeze() returns a scalar (reproducing pandas 1-bar)."""

    def __init__(self, pairs):
        self._pairs = list(pairs)
        self.index = [k for k, _ in self._pairs]
        self.values = [v for _, v in self._pairs]

    def __len__(self):
        return len(self._pairs)

    def items(self):
        return iter(self._pairs)

    def squeeze(self):
        if len(self._pairs) == 1:
            return self._pairs[0][1]  # scalar float!
        return self


class _FakeSingleBarDF:
    def __init__(self, close_obj):
        self._close = close_obj

    def __len__(self):
        return len(self._close)

    def __getitem__(self, key):
        if key == "Close":
            return self._close
        raise KeyError(key)


class _FakeMultiIndexCloseDF:
    """Simulates yfinance >= 0.2.40 where df['Close'] is a 1-column DataFrame."""

    def __init__(self, ticker, pairs):
        self.columns = [ticker]
        self._series = _FakeSeries(pairs)
        self.iloc = self

    def __len__(self):
        return len(self._series)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._series
        return self._series


class TestDataAcquisitionOP06(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.env_patcher = mock.patch.dict(os.environ, {"BYRON_CACHE": self.tmpdir.name})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def test_single_bar_close_does_not_crash_on_squeeze(self):
        one_bar_df = _FakeSingleBarDF(_FakeSeries([(datetime(2024, 6, 10), 121.79)]))
        fake_yf = mock.MagicMock()
        fake_yf.download.return_value = one_bar_df

        with mock.patch.object(data, "yf", fake_yf):
            closes = data.get_closes("NVDA", "2024-06-10", "2024-06-10")

        self.assertEqual(closes, {"2024-06-10": 121.79})
        # Cold fetch MUST request end = hi + 1 day for inclusive [start, end]
        _, kwargs = fake_yf.download.call_args
        self.assertEqual(kwargs["start"], "2024-06-10")
        self.assertEqual(kwargs["end"], "2024-06-11")

    def test_multiindex_close_dataframe_and_warm_cache_inclusive_end(self):
        mi_close = _FakeMultiIndexCloseDF(
            "NVDA",
            [
                (datetime(2024, 6, 10), 121.79),
                (datetime(2024, 6, 11), 120.91),
                (datetime(2024, 6, 12), 125.20),
            ],
        )
        fake_yf = mock.MagicMock()
        fake_yf.download.return_value = _FakeSingleBarDF(mi_close)

        with mock.patch.object(data, "yf", fake_yf):
            cold = data.get_closes("NVDA", "2024-06-10", "2024-06-12")
            warm = data.get_closes("NVDA", "2024-06-10", "2024-06-12")

        self.assertEqual(fake_yf.download.call_count, 1)
        self.assertEqual(cold, warm)
        self.assertIn("2024-06-12", warm)
        _, kwargs = fake_yf.download.call_args
        self.assertEqual(kwargs["end"], "2024-06-13")


class TestSyntheticSimulationAndDemoOP07(unittest.TestCase):
    def test_equal_delta_repricing_at_different_dte_shows_strike_drift_and_greeks(self):
        k35 = pricing.strike_for_delta(100.0, 35 / 365.0, 0.40, 0.60, "call")
        k168 = pricing.strike_for_delta(100.0, 168 / 365.0, 0.40, 0.60, "call")
        self.assertNotAlmostEqual(k35, k168, places=1)

        g35 = pricing.bs_gamma(100.0, k35, 35 / 365.0, 0.40)
        g168 = pricing.bs_gamma(100.0, k168, 168 / 365.0, 0.40)
        th35 = pricing.bs_theta(100.0, k35, 35 / 365.0, 0.40, "call")
        th168 = pricing.bs_theta(100.0, k168, 168 / 365.0, 0.40, "call")
        self.assertGreater(g35, g168)
        self.assertLess(th35, th168)  # 35-DTE decays faster per calendar day

    def test_offline_demo_runs_deterministically_with_zero_network(self):
        with mock.patch("sys.stdout", new=io.StringIO()):
            report = bench.demo()

        self.assertEqual(report["mode"], "offline_synthetic_simulation_demo")
        self.assertEqual(len(report["fixture_sha256"]), 64)
        self.assertGreater(report["paired_dte_ab"]["n_setups"], 0)
        self.assertIn("pricing_decomposition", report["paired_dte_ab"])
        self.assertGreater(
            report["paired_dte_ab"]["pricing_decomposition"]["median_abs_strike_drift_pct_of_spot"],
            0.0,
        )
        self.assertEqual(report["book_valid_mock"]["verdict"], "VALID")
        self.assertEqual(report["book_all_failed_mock"]["verdict"], "INVALID")
        self.assertIsNone(report["book_all_failed_mock"]["alpha_vs_spy_cagr_pct"])


class TestCacheProvenanceAndCompareOP08(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_provenance_changes_invalidate_cache_key(self):
        base = "http://127.0.0.1:8082"
        id_base = bench._cache_identity("construction_alpha", "alias-v1", base)
        id_rev2 = bench._cache_identity("construction_alpha", "alias-v1", base, model_revision="sha256:rev2")
        id_code = bench._cache_identity("construction_alpha", "alias-v1", base, code_digest="deadbeef00000001")
        id_prompt = bench._cache_identity("construction_alpha", "alias-v1", base, prompt_digest="deadbeef00000002")
        id_data = bench._cache_identity("construction_alpha", "alias-v1", base, data_digest="deadbeef00000003")

        fps = {
            bench._cache_fingerprint(x)
            for x in (id_base, id_rev2, id_code, id_prompt, id_data)
        }
        self.assertEqual(len(fps), 5)

    def test_compare_surfaces_missing_and_incomparable_cards_explicitly(self):
        card_valid = {
            "name": "run_a",
            "config": {"start": "2024-02-01", "end": "2024-12-01", "step": 7},
            "provenance": {"book_backtest": {"code_digest": "aaa", "prompt_digest": "ppp", "data_digest": "ddd"}},
            "headline": {
                "verdict": "VALID",
                "book_coverage_pct": 100.0,
                "book_cagr_pct": 12.0,
                "book_sharpe": 1.1,
                "book_max_dd_pct": 10.0,
                "book_alpha_vs_spy_pct": 2.0,
            },
        }
        card_incomp = {
            "name": "run_b",
            "config": {"start": "2024-06-01", "end": "2024-12-01", "step": 14},
            "provenance": {"book_backtest": {"code_digest": "bbb", "prompt_digest": "ppp", "data_digest": "ddd"}},
            "headline": {
                "verdict": "INVALID",
                "book_coverage_pct": 0.0,
                "book_cagr_pct": None,
                "book_sharpe": None,
                "book_max_dd_pct": None,
                "book_alpha_vs_spy_pct": None,
            },
        }
        bench._write_json(os.path.join(self.tmpdir.name, "scorecard_run_a.json"), card_valid)
        bench._write_json(os.path.join(self.tmpdir.name, "scorecard_run_b.json"), card_incomp)

        buf = io.StringIO()
        with mock.patch("sys.stdout", new=buf):
            res = bench.compare(["run_a", "run_b", "run_missing"], cache_dir=self.tmpdir.name)

        out = buf.getvalue()
        self.assertEqual(res["missing"], ["run_missing"])
        self.assertIn("MISSING", out)
        self.assertIn("run_missing", out)
        self.assertIn("WARNING: cards are NOT directly comparable", out)
        self.assertTrue(any("INVALID" in r for r in res["incomparable_reasons"]))
        self.assertTrue(any("date windows" in r for r in res["incomparable_reasons"]))
        self.assertTrue(any("code_digest" in r for r in res["incomparable_reasons"]))
        self.assertTrue(any("model_revision" in r for r in res["incomparable_reasons"]))

    def test_post_hydration_data_digest_consistency_op08(self):
        """OP08 exact repro: cache hydration during bench.run() must bind POST-hydration data_digest."""
        import types

        px_dir = os.path.join(self.tmpdir.name, "px_cold")
        out_dir = os.path.join(self.tmpdir.name, "out")
        os.makedirs(px_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        with mock.patch.dict(os.environ, {"BYRON_CACHE": px_dir}):
            empty_digest = bench._data_digest(px_dir)
            # Empty cache SHA-256 prefix
            self.assertEqual(empty_digest, "e3b0c44298fc1c14")

            calls = {"construction": 0, "book": 0}

            def _hydrate_cache(ticker, payload):
                with open(os.path.join(px_dir, f"{ticker}.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f, sort_keys=True)

            scenarios_mod = types.ModuleType("scenarios")
            scenarios_mod.generate = lambda: [
                {"ticker": "NVDA", "entry_date": "2024-03-04", "horizon_days": 21, "mom20": 0.12}
            ]

            rma_mod = types.ModuleType("run_model_alpha")

            def fake_score(setups, base, model, **kwargs):
                calls["construction"] += 1
                # Simulate live hydration of price cache during track execution
                _hydrate_cache("NVDA", {"_start": "2024-01-01", "_end": "2024-06-01", "2024-03-04": 85.0})
                return [{"status": "ok", "model_ret": 0.15, "model_dte": 168, "doctrine_ret": 0.05}]

            def fake_summarize(rows, **kwargs):
                return {"verdict": "VALID", "qualifying": True, "n_ok": len(rows), "n_total": len(rows)}

            rma_mod.score = fake_score
            rma_mod.summarize = fake_summarize

            book_mod = types.ModuleType("book")

            class HydratingBook:
                def __init__(self, base, model, start, end, step, **kwargs):
                    self.equity_curve = [(start, 100000.0), (end, 105000.0)]

                def run(self):
                    calls["book"] += 1
                    _hydrate_cache("SPY", {"_start": "2024-01-01", "_end": "2024-06-01", "2024-03-04": 510.0})
                    return {"verdict": "VALID", "qualifying": True, "cagr_pct": 10.0}

            book_mod.Book = HydratingBook

            with mock.patch.dict(sys.modules, {
                "scenarios": scenarios_mod,
                "run_model_alpha": rma_mod,
                "book": book_mod,
            }):
                card = bench.run(
                    "hydrated_run",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2024-03-01",
                    "2024-06-01",
                    7,
                    cache_dir=out_dir,
                    model_revision="sha256:rev1",
                )
                post_digest = bench._data_digest(px_dir)

                # Post-hydration digest MUST differ from pre-run empty digest
                self.assertNotEqual(post_digest, empty_digest)
                # Neither track may record the stale pre-run empty hash!
                self.assertNotEqual(card["provenance"]["construction_alpha"]["data_digest"], empty_digest)
                self.assertEqual(card["provenance"]["book_backtest"]["data_digest"], post_digest)

                # Verify cached artifact on disk binds the post-hydration digest
                b_id = card["provenance"]["book_backtest"]
                b_path = os.path.join(
                    out_dir, f"{bench._cache_tag('book', 'hydrated_run', b_id)}_metrics.json"
                )
                self.assertTrue(os.path.exists(b_path))
                with open(b_path, encoding="utf-8") as f:
                    cached_book = json.load(f)
                self.assertEqual(cached_book[bench.CACHE_META_KEY]["data_digest"], post_digest)

                # Re-running with now-hydrated cache MUST hit book cache cleanly (0 extra calls)
                card2 = bench.run(
                    "hydrated_run",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2024-03-01",
                    "2024-06-01",
                    7,
                    skip_construction=True,
                    cache_dir=out_dir,
                    model_revision="sha256:rev1",
                )
                self.assertEqual(calls["book"], 1)
                self.assertEqual(card2["provenance"]["book_backtest"]["data_digest"], post_digest)


if __name__ == "__main__":
    unittest.main()
