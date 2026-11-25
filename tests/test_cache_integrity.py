import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bench


def _make_fake_modules(construction_calls, book_calls):
    """Build lightweight stub modules so tests run without network or external deps."""
    scenarios_mod = types.ModuleType("scenarios")
    scenarios_mod.generate = lambda: [{"ticker": "NVDA", "entry_date": "2024-03-01", "horizon_days": 21, "mom20": 0.12}]

    run_model_alpha_mod = types.ModuleType("run_model_alpha")

    def fake_score(setups, base, model, limit=None):
        construction_calls.append((base, model))
        if model == "model-A":
            return [{"model_ret": 0.10, "model_dte": 45, "doctrine_ret": 0.05}]
        if model == "model-B":
            return [{"model_ret": 0.42, "model_dte": 180, "doctrine_ret": 0.05}]
        return [{"model_ret": 0.25, "model_dte": 120, "doctrine_ret": 0.05}]

    def fake_summarize(rows):
        r = rows[0]
        return {
            "n_ok": len(rows),
            "n_err": 0,
            "model_mean_ret_pct": round(r["model_ret"] * 100, 2),
            "model_median_dte": r["model_dte"],
            "model_dte_ge_120_pct": 100.0 if r["model_dte"] >= 120 else 0.0,
        }

    run_model_alpha_mod.score = fake_score
    run_model_alpha_mod.summarize = fake_summarize

    book_mod = types.ModuleType("book")

    class FakeBook:
        def __init__(self, base, model, start, end, step):
            self.base = base
            self.model = model
            self.start = start
            self.end = end
            self.step = step
            self.equity_curve = [(start, 100000.0), (end, 125000.0 if model == "model-B" else 95000.0)]

        def run(self):
            book_calls.append((self.base, self.model, self.start, self.end, self.step))
            if self.model == "model-A":
                return {
                    "cagr_pct": -5.0,
                    "sharpe": -0.2,
                    "max_drawdown_pct": 30.0,
                    "alpha_vs_spy_cagr_pct": -20.0,
                }
            if self.model == "model-B":
                return {
                    "cagr_pct": 25.0,
                    "sharpe": 1.8,
                    "max_drawdown_pct": 8.5,
                    "alpha_vs_spy_cagr_pct": 6.0,
                }
            return {
                "cagr_pct": 12.0,
                "sharpe": 0.9,
                "max_drawdown_pct": 15.0,
                "alpha_vs_spy_cagr_pct": 1.0,
            }

    book_mod.Book = FakeBook

    return {
        "scenarios": scenarios_mod,
        "run_model_alpha": run_model_alpha_mod,
        "book": book_mod,
    }


class TestBenchCacheIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.construction_calls = []
        self.book_calls = []
        self.fake_modules = _make_fake_modules(self.construction_calls, self.book_calls)
        self.modules_patcher = mock.patch.dict(sys.modules, self.fake_modules)
        self.modules_patcher.start()
        self.addCleanup(self.modules_patcher.stop)
        self.here_patcher = mock.patch.object(bench, "HERE", self.tmpdir.name)
        self.here_patcher.start()
        self.addCleanup(self.here_patcher.stop)

    def test_same_run_name_under_different_models_does_not_reuse_or_relabel(self):
        base = "http://127.0.0.1:8082"
        card_a = bench.run("shared_run", base, "model-A", "2024-02-01", "2026-04-15", 7)
        card_b = bench.run("shared_run", base, "model-B", "2024-02-01", "2026-04-15", 7)

        self.assertEqual(self.construction_calls, [(base, "model-A"), (base, "model-B")])
        self.assertEqual(
            self.book_calls,
            [
                (base, "model-A", "2024-02-01", "2026-04-15", 7),
                (base, "model-B", "2024-02-01", "2026-04-15", 7),
            ],
        )

        self.assertEqual(card_a["model"], "model-A")
        self.assertEqual(card_a["construction_alpha"]["model_mean_ret_pct"], 10.0)
        self.assertEqual(card_a["book_backtest"]["cagr_pct"], -5.0)

        self.assertEqual(card_b["model"], "model-B")
        self.assertEqual(card_b["construction_alpha"]["model_mean_ret_pct"], 42.0)
        self.assertEqual(card_b["book_backtest"]["cagr_pct"], 25.0)

        # Verify distinct fingerprinted cache files were persisted with identity metadata.
        c_id_a = bench._cache_identity("construction_alpha", "model-A", base)
        c_id_b = bench._cache_identity("construction_alpha", "model-B", base)
        c_path_a = os.path.join(self.tmpdir.name, f"{bench._cache_tag('model_alpha', 'shared_run', c_id_a)}_summary.json")
        c_path_b = os.path.join(self.tmpdir.name, f"{bench._cache_tag('model_alpha', 'shared_run', c_id_b)}_summary.json")
        self.assertNotEqual(c_path_a, c_path_b)
        self.assertTrue(os.path.exists(c_path_a))
        self.assertTrue(os.path.exists(c_path_b))

        with open(c_path_a) as f:
            self.assertEqual(json.load(f)[bench.CACHE_META_KEY], c_id_a)
        with open(c_path_b) as f:
            self.assertEqual(json.load(f)[bench.CACHE_META_KEY], c_id_b)

    def test_legitimate_same_model_cache_hit_reuses_without_recomputing(self):
        base = "http://127.0.0.1:8082"
        first = bench.run("steady_run", base, "model-A", "2024-02-01", "2026-04-15", 7)
        second = bench.run("steady_run", base, "model-A", "2024-02-01", "2026-04-15", 7)

        self.assertEqual(len(self.construction_calls), 1)
        self.assertEqual(len(self.book_calls), 1)
        self.assertEqual(first, second)
        self.assertNotIn(bench.CACHE_META_KEY, second["construction_alpha"])
        self.assertNotIn(bench.CACHE_META_KEY, second["book_backtest"])

    def test_changed_runner_settings_invalidate_cache_key(self):
        base1 = "http://127.0.0.1:8082"
        base2 = "http://127.0.0.1:9090"

        bench.run("cfg_run", base1, "model-A", "2024-02-01", "2026-04-15", 7)
        # Changing book window/step should reuse construction cache, but recompute book.
        bench.run("cfg_run", base1, "model-A", "2024-06-01", "2026-04-15", 14)
        self.assertEqual(len(self.construction_calls), 1)
        self.assertEqual(len(self.book_calls), 2)

        # Changing endpoint base should recompute both tracks.
        bench.run("cfg_run", base2, "model-A", "2024-06-01", "2026-04-15", 14)
        self.assertEqual(len(self.construction_calls), 2)
        self.assertEqual(len(self.book_calls), 3)

    def test_mismatched_embedded_model_identity_in_construction_cache_raises(self):
        base = "http://127.0.0.1:8082"
        c_id_b = bench._cache_identity("construction_alpha", "model-B", base)
        c_path_b = os.path.join(self.tmpdir.name, f"{bench._cache_tag('model_alpha', 'tampered', c_id_b)}_summary.json")

        # Plant a cache file at model-B's path whose embedded metadata claims model-A.
        poisoned = {
            "n_ok": 1,
            "n_err": 0,
            "model_mean_ret_pct": 10.0,
            "model_median_dte": 45,
            "model_dte_ge_120_pct": 0.0,
            bench.CACHE_META_KEY: bench._cache_identity("construction_alpha", "model-A", base),
        }
        with open(c_path_b, "w") as f:
            json.dump(poisoned, f)

        with self.assertRaises(bench.CacheIntegrityError) as ctx:
            bench._load_cached(c_path_b, c_id_b)
        self.assertIn("model-A", str(ctx.exception))
        self.assertIn("model-B", str(ctx.exception))

        with self.assertRaises(bench.CacheIntegrityError):
            bench.run("tampered", base, "model-B", "2024-02-01", "2026-04-15", 7)

        self.assertEqual(self.construction_calls, [])

    def test_mismatched_embedded_model_identity_in_book_cache_raises(self):
        base = "http://127.0.0.1:8082"
        b_id_b = bench._cache_identity(
            "book_backtest", "model-B", base, start="2024-02-01", end="2026-04-15", step=7
        )
        b_path_b = os.path.join(self.tmpdir.name, f"{bench._cache_tag('book', 'tampered_book', b_id_b)}_metrics.json")

        poisoned = {
            "cagr_pct": -5.0,
            "sharpe": -0.2,
            "max_drawdown_pct": 30.0,
            "alpha_vs_spy_cagr_pct": -20.0,
            bench.CACHE_META_KEY: bench._cache_identity(
                "book_backtest", "model-A", base, start="2024-02-01", end="2026-04-15", step=7
            ),
        }
        with open(b_path_b, "w") as f:
            json.dump(poisoned, f)

        with self.assertRaises(bench.CacheIntegrityError) as ctx:
            bench.run("tampered_book", base, "model-B", "2024-02-01", "2026-04-15", 7, skip_construction=True)
        self.assertIn("model-A", str(ctx.exception))
        self.assertIn("model-B", str(ctx.exception))
        self.assertEqual(self.book_calls, [])

    def test_missing_metadata_in_cache_file_raises(self):
        base = "http://127.0.0.1:8082"
        c_id = bench._cache_identity("construction_alpha", "model-B", base)
        c_path = os.path.join(self.tmpdir.name, f"{bench._cache_tag('model_alpha', 'no_meta', c_id)}_summary.json")

        with open(c_path, "w") as f:
            json.dump({"n_ok": 1, "model_mean_ret_pct": 99.0}, f)

        with self.assertRaises(bench.CacheIntegrityError) as ctx:
            bench._load_cached(c_path, c_id)
        self.assertIn(bench.CACHE_META_KEY, str(ctx.exception))

        with self.assertRaises(bench.CacheIntegrityError):
            bench.run("no_meta", base, "model-B", "2024-02-01", "2026-04-15", 7)

    def test_legacy_unfingerprinted_cache_files_are_not_silently_reused(self):
        base = "http://127.0.0.1:8082"
        # Old cache filenames keyed only by run name.
        with open(os.path.join(self.tmpdir.name, "model_alpha_legacy_summary.json"), "w") as f:
            json.dump({"n_ok": 1, "model_mean_ret_pct": 10.0, "model_median_dte": 45, "model_dte_ge_120_pct": 0.0}, f)
        with open(os.path.join(self.tmpdir.name, "book_legacy_metrics.json"), "w") as f:
            json.dump({"cagr_pct": -5.0, "sharpe": -0.2, "max_drawdown_pct": 30.0, "alpha_vs_spy_cagr_pct": -20.0}, f)

        card = bench.run("legacy", base, "model-B", "2024-02-01", "2026-04-15", 7)
        self.assertEqual(self.construction_calls, [(base, "model-B")])
        self.assertEqual(self.book_calls, [(base, "model-B", "2024-02-01", "2026-04-15", 7)])
        self.assertEqual(card["construction_alpha"]["model_mean_ret_pct"], 42.0)
        self.assertEqual(card["book_backtest"]["cagr_pct"], 25.0)

    def test_skip_flags_do_not_inherit_other_model_cache(self):
        base = "http://127.0.0.1:8082"
        bench.run("skip_run", base, "model-A", "2024-02-01", "2026-04-15", 7)

        card_b = bench.run(
            "skip_run", base, "model-B", "2024-02-01", "2026-04-15", 7,
            skip_construction=True, skip_book=True,
        )
        self.assertIsNone(card_b["construction_alpha"])
        self.assertIsNone(card_b["book_backtest"])


if __name__ == "__main__":
    unittest.main()
