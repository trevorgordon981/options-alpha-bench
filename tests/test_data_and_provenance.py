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

    def test_mid_run_cache_mutation_does_not_corrupt_construction_prices_or_digest_op08(self):
        """OP08 exact repro: mutating NVDA.json ($100 -> $200) inside model callback must NOT alter consumed prices or data_digest."""
        from datetime import date, timedelta
        import run_model_alpha

        px_dir = os.path.join(self.tmpdir.name, "px_mut_const")
        out_dir = os.path.join(self.tmpdir.name, "out_mut_const")
        os.makedirs(px_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # Build deterministic synthetic NVDA series: pre-window + entry 2025-01-02 @ $100 rising to $125 (+25%)
        def _make_series(entry_px: float, end_px: float) -> dict:
            bars = {}
            d = date(2024, 11, 15)
            while d <= date(2025, 2, 15):
                if d.weekday() < 5:
                    iso = d.isoformat()
                    if iso < "2025-01-02":
                        bars[iso] = entry_px
                    elif iso == "2025-01-02":
                        bars[iso] = entry_px
                    else:
                        bars[iso] = end_px
                d += timedelta(days=1)
            return {
                "_schema_version": 2,
                "_source": "synthetic_test",
                "_fixture_sha256": "pinned-test-fixture",
                "_start": "2024-11-15",
                "_end": "2025-02-15",
                **bars,
            }

        nvda_path = os.path.join(px_dir, "NVDA.json")
        with open(nvda_path, "w", encoding="utf-8") as f:
            json.dump(_make_series(100.0, 125.0), f, sort_keys=True)

        with mock.patch.dict(os.environ, {"BYRON_CACHE": px_dir}):
            digest_100 = data.cache_manifest_digest(px_dir)
            prompts_seen = []
            mutated = {"done": False}

            def _mutating_model_call(prompt, base, model, **kwargs):
                prompts_seen.append(prompt)
                if not mutated["done"]:
                    mutated["done"] = True
                    # Mutate on-disk cache mid-run: $100 -> $200 entry, $125 -> $150 end (-25% instead of +25%)
                    with open(nvda_path, "w", encoding="utf-8") as f:
                        json.dump(_make_series(200.0, 150.0), f, sort_keys=True)
                return json.dumps({
                    "direction": "long_call",
                    "entry_dte": 35,
                    "long_delta": 0.60,
                    "stop_pct": 0.30,
                    "conviction": "high",
                })

            test_setups = [
                {"ticker": "NVDA", "entry_date": "2025-01-02", "horizon_days": 21, "mom20": 0.10}
            ]
            with mock.patch("sys.stdout", new=io.StringIO()), \
                 mock.patch.object(scenarios, "generate", return_value=test_setups), \
                 mock.patch.object(run_model_alpha, "call_model", side_effect=_mutating_model_call):
                card1 = bench.run(
                    "mut_const",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2025-01-02",
                    "2025-01-24",
                    7,
                    skip_book=True,
                    cache_dir=out_dir,
                    model_revision="sha256:rev1",
                )

            digest_200 = data.cache_manifest_digest(px_dir)
            self.assertNotEqual(digest_100, digest_200)
            self.assertEqual(len(prompts_seen), 1)
            self.assertIn("Spot: $100.00", prompts_seen[0])

            # Provenance digest MUST match the exact $100 snapshot consumed, NOT mutated $200 disk cache!
            c_id = card1["provenance"]["construction_alpha"]
            self.assertEqual(c_id["data_digest"], digest_100)
            self.assertNotEqual(c_id["data_digest"], digest_200)

            ctag = bench._cache_tag("model_alpha", "mut_const", c_id)
            with open(os.path.join(out_dir, f"{ctag}_summary.json"), encoding="utf-8") as f:
                cached_summary = json.load(f)
            self.assertEqual(cached_summary[bench.CACHE_META_KEY]["data_digest"], digest_100)

            with open(os.path.join(out_dir, f"{ctag}_rows.json"), encoding="utf-8") as f:
                saved_rows = json.load(f)
            self.assertEqual(len(saved_rows), 1)
            # Underlying return MUST reflect the frozen $100 -> $125 (+25%) series, NOT mutated $200 -> $150 (-25%)!
            self.assertAlmostEqual(saved_rows[0]["underlying_ret"], 0.25, places=4)
            self.assertEqual(saved_rows[0]["data_digest"], digest_100)

            # A subsequent run AFTER the disk mutation must miss the $100 cache and bind the new $200 digest
            with mock.patch("sys.stdout", new=io.StringIO()), \
                 mock.patch.object(scenarios, "generate", return_value=test_setups), \
                 mock.patch.object(run_model_alpha, "call_model", side_effect=_mutating_model_call):
                card2 = bench.run(
                    "mut_const",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2025-01-02",
                    "2025-01-24",
                    7,
                    skip_book=True,
                    cache_dir=out_dir,
                    model_revision="sha256:rev1",
                )
            self.assertEqual(len(prompts_seen), 2)
            self.assertIn("Spot: $200.00", prompts_seen[1])
            self.assertEqual(card2["provenance"]["construction_alpha"]["data_digest"], digest_200)

    def test_mid_run_cache_mutation_does_not_corrupt_book_prices_or_digest_op08(self):
        """OP08 exact repro for Track 2: mutating on-disk cache ($100 -> $200) mid-Book run cannot drift consumed prices or data_digest."""
        from datetime import date, timedelta
        import book as real_book

        px_dir = os.path.join(self.tmpdir.name, "px_mut_book")
        out_dir = os.path.join(self.tmpdir.name, "out_mut_book")
        os.makedirs(px_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        def _make_series(val_start: float, val_end: float) -> dict:
            bars = {}
            d = date(2024, 9, 1)
            while d <= date(2025, 1, 20):
                if d.weekday() < 5:
                    iso = d.isoformat()
                    bars[iso] = val_start if iso <= "2025-01-03" else val_end
                d += timedelta(days=1)
            return {
                "_schema_version": 2,
                "_source": "synthetic_test",
                "_fixture_sha256": "pinned-test-fixture",
                "_start": "2024-09-01",
                "_end": "2025-01-20",
                **bars,
            }

        for tk in ("SPY", "NVDA"):
            with open(os.path.join(px_dir, f"{tk}.json"), "w", encoding="utf-8") as f:
                json.dump(_make_series(100.0, 110.0), f, sort_keys=True)

        with mock.patch.dict(os.environ, {"BYRON_CACHE": px_dir}):
            digest_100 = data.cache_manifest_digest(px_dir)
            prompts_seen = []

            def _mutating_book_call(self_book, prompt, max_tokens=1200):
                prompts_seen.append(prompt)
                if len(prompts_seen) == 1:
                    # Overwrite on-disk cache mid-run from $100/$110 to $200/$50!
                    for tk in ("SPY", "NVDA"):
                        with open(os.path.join(px_dir, f"{tk}.json"), "w", encoding="utf-8") as f:
                            json.dump(_make_series(200.0, 50.0), f, sort_keys=True)
                return '{"close": [], "open": []}'

            with mock.patch("sys.stdout", new=io.StringIO()), \
                 mock.patch.object(real_book.Book, "_call", _mutating_book_call):
                card = bench.run(
                    "mut_book",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2025-01-02",
                    "2025-01-16",
                    7,
                    skip_construction=True,
                    cache_dir=out_dir,
                    model_revision="sha256:rev1",
                )

            digest_200 = data.cache_manifest_digest(px_dir)
            self.assertNotEqual(digest_100, digest_200)
            self.assertGreaterEqual(len(prompts_seen), 2)
            # Even the SECOND step prompt (after disk mutation) MUST still see frozen $110, NOT mutated $50!
            self.assertIn("$110.0", prompts_seen[1])
            self.assertNotIn("$50.0", prompts_seen[1])
            # SPY comparator MUST reflect frozen +10% (100 -> 110), NOT mutated -75% (200 -> 50)!
            self.assertAlmostEqual(card["book_backtest"]["spy_buyhold_total_return_pct"], 10.0, places=1)
            # Provenance digest MUST equal the frozen $100 snapshot digest, NOT mutated disk digest!
            self.assertEqual(card["provenance"]["book_backtest"]["data_digest"], digest_100)
            self.assertNotEqual(card["provenance"]["book_backtest"]["data_digest"], digest_200)
            self.assertEqual(card["book_backtest"]["data_digest"], digest_100)

    def test_selection_time_cache_mutation_cannot_desync_prompt_scoring_or_digest_op08(self):
        """OP08 exact repro: cache revision ($110 -> $90) after setup selection cannot desync prompt, simulation, or digest."""
        from datetime import date, timedelta
        import run_model_alpha

        px_dir = os.path.join(self.tmpdir.name, "px_sel_mut")
        out_dir = os.path.join(self.tmpdir.name, "out_sel_mut")
        os.makedirs(px_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        entry = date(2025, 1, 2)
        base_bars = {(entry - timedelta(days=days)).isoformat(): 100.0 for days in range(1, 24)}
        before_bars = {**base_bars, "2025-01-02": 110.0, "2025-01-03": 112.0, "2025-01-06": 116.0}
        after_bars = {**before_bars, "2025-01-02": 90.0}

        meta = {
            "_schema_version": 2,
            "_source": "synthetic_test",
            "_fixture_sha256": "pinned-selection-fixture",
            "_start": "2024-01-01",
            "_end": "2026-01-01",
        }
        nvda_path = os.path.join(px_dir, "NVDA.json")
        with open(nvda_path, "w", encoding="utf-8") as f:
            json.dump({**meta, **before_bars}, f, sort_keys=True)

        digest_110 = data.PriceSnapshot({"NVDA": before_bars}).digest()
        digest_90 = data.PriceSnapshot({"NVDA": after_bars}).digest()
        self.assertNotEqual(digest_110, digest_90)

        native_generate = scenarios.generate
        prompts_run1 = []
        recomputed_inside_run1 = []

        def _mutating_generate():
            args = {"start": "2025-01-02", "end": "2025-01-02", "basket": ["NVDA"]}
            picked = native_generate(**args)
            self.assertEqual(len(picked), 1)
            self.assertEqual(picked[0]["mom20"], 0.10)
            self.assertEqual(picked[0]["direction"], "long_call")
            # Mutate on-disk cache immediately after setup selection ($110 -> $90)
            with open(nvda_path, "w", encoding="utf-8") as f:
                json.dump({**meta, **after_bars}, f, sort_keys=True)
            # Inside the same frozen construction session, re-running selection still sees $110
            recomputed = native_generate(**args)
            recomputed_inside_run1.extend(recomputed)
            return picked

        def _capture_call(prompt, base, model, **kwargs):
            prompts_run1.append(prompt)
            return json.dumps({
                "direction": "long_call",
                "entry_dte": 35,
                "long_delta": 0.60,
                "stop_pct": 0.30,
                "conviction": "high",
            })

        with mock.patch.dict(os.environ, {"BYRON_CACHE": px_dir}):
            fake_yf = mock.MagicMock()
            fake_yf.download.side_effect = AssertionError("unexpected price fetch")
            with mock.patch("sys.stdout", new=io.StringIO()), \
                 mock.patch.object(scenarios, "generate", side_effect=_mutating_generate), \
                 mock.patch.object(run_model_alpha, "call_model", side_effect=_capture_call), \
                 mock.patch.object(data, "yf", fake_yf):
                card1 = bench.run(
                    "sel_mut_audit",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2025-01-02",
                    "2025-01-06",
                    7,
                    skip_book=True,
                    cache_dir=out_dir,
                    model_revision="fixture-selection",
                )
            self.assertFalse(fake_yf.download.called)

            # Even the second native_generate inside the session remained pinned to $110 (+10%)
            self.assertEqual(len(recomputed_inside_run1), 1)
            self.assertEqual(recomputed_inside_run1[0]["mom20"], 0.10)
            self.assertEqual(recomputed_inside_run1[0]["direction"], "long_call")

            # Prompt MUST combine $110.00 spot with +10.0% momentum (NEVER $90.00 + +10.0%!)
            self.assertEqual(len(prompts_run1), 1)
            self.assertIn("Spot: $110.00", prompts_run1[0])
            self.assertIn("+10.0%", prompts_run1[0])
            self.assertNotIn("Spot: $90.00", prompts_run1[0])

            # Provenance digest MUST bind the exact $110 snapshot, NOT the mutated $90 disk file!
            c_id = card1["provenance"]["construction_alpha"]
            self.assertEqual(c_id["data_digest"], digest_110)
            self.assertNotEqual(c_id["data_digest"], digest_90)

            ctag = bench._cache_tag("model_alpha", "sel_mut_audit", c_id)
            with open(os.path.join(out_dir, f"{ctag}_rows.json"), encoding="utf-8") as f:
                saved_rows = json.load(f)
            self.assertEqual(len(saved_rows), 1)
            # Underlying return reflects $110 -> $116 (+5.45%), NOT $90 -> $116 (+28.89%)
            self.assertAlmostEqual(saved_rows[0]["underlying_ret"], (116.0 - 110.0) / 110.0, places=4)
            self.assertEqual(saved_rows[0]["data_digest"], digest_110)

            # A fresh subsequent run on the now-$90 disk cache must consistently select -10% & $90.00
            prompts_run2 = []

            def _capture_call_run2(prompt, base, model, **kwargs):
                prompts_run2.append(prompt)
                return json.dumps({
                    "direction": "long_put",
                    "entry_dte": 35,
                    "long_delta": 0.60,
                    "stop_pct": 0.30,
                    "conviction": "high",
                })

            with mock.patch("sys.stdout", new=io.StringIO()), \
                 mock.patch.object(
                     scenarios,
                     "generate",
                     side_effect=lambda **kw: native_generate(
                         start="2025-01-02", end="2025-01-02", basket=["NVDA"], **kw
                     ),
                 ), \
                 mock.patch.object(run_model_alpha, "call_model", side_effect=_capture_call_run2), \
                 mock.patch.object(data, "yf", fake_yf):
                card2 = bench.run(
                    "sel_mut_audit",
                    "http://127.0.0.1:8082",
                    "model-v1",
                    "2025-01-02",
                    "2025-01-06",
                    7,
                    skip_book=True,
                    cache_dir=out_dir,
                    model_revision="fixture-selection",
                )
            self.assertFalse(fake_yf.download.called)
            self.assertEqual(len(prompts_run2), 1)
            self.assertIn("Spot: $90.00", prompts_run2[0])
            self.assertIn("-10.0%", prompts_run2[0])
            self.assertEqual(card2["provenance"]["construction_alpha"]["data_digest"], digest_90)


if __name__ == "__main__":
    unittest.main()
