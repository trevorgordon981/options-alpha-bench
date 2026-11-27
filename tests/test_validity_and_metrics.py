"""Regression tests for OP04 (validity gating) and OP05 (dedup marks + annualization)."""
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bench
import book
import run_model_alpha


class TestValidityGatingOP04(unittest.TestCase):
    def test_all_failed_book_over_falling_spy_is_invalid_and_suppresses_alpha(self):
        """OP04 exact repro: 3 timeouts over 14 days while SPY drops 100 -> 98."""
        fake_spy = {
            "2024-03-01": 100.0,
            "2024-03-08": 99.0,
            "2024-03-15": 98.0,
        }
        b = book.Book(
            "http://127.0.0.1:8000",
            "timeout-model",
            start="2024-03-01",
            end="2024-03-15",
            step_days=7,
            verbose=False,
        )
        with mock.patch.object(book, "get_closes", return_value=fake_spy), \
             mock.patch("market.get_closes", return_value=fake_spy), \
             mock.patch.object(b, "_call", side_effect=TimeoutError("timed out")):
            m = b.run()

        self.assertEqual(m["verdict"], "INVALID")
        self.assertFalse(m["qualifying"])
        self.assertEqual(m["decision_steps"], 3)
        self.assertEqual(m["valid_decisions"], 0)
        self.assertEqual(m["transport_errors"], 3)
        self.assertEqual(m["failed_steps"], 3)
        self.assertEqual(m["coverage_pct"], 0.0)
        self.assertEqual(m["duration_days"], 14)
        self.assertEqual(m["cash_comparator_cagr_pct"], 0.0)
        self.assertLess(m["spy_buyhold_cagr_pct"], 0.0)
        # CRITICAL: Must NEVER publish favorable alpha/CAGR/Sharpe from defaulting to cash!
        self.assertIsNone(m["alpha_vs_spy_cagr_pct"])
        self.assertIsNone(m["alpha_vs_cash_cagr_pct"])
        self.assertIsNone(m["cagr_pct"])
        self.assertIsNone(m["sharpe"])
        self.assertIn("zero_valid_decisions", m["invalid_reason"])

        # Headline must propagate INVALID verdict, coverage, cash/SPY comparators, and duration
        headline = bench._headline({"name": "bad_run", "book_backtest": m})
        self.assertEqual(headline["verdict"], "INVALID")
        self.assertFalse(headline["qualifying"])
        self.assertIsNone(headline["book_alpha_vs_spy_pct"])
        self.assertIsNone(headline["book_cagr_pct"])
        self.assertEqual(headline["book_coverage_pct"], 0.0)
        self.assertEqual(headline["book_cash_cagr_pct"], 0.0)
        self.assertEqual(headline["book_duration_days"], 14)

    def test_deliberate_abstentions_are_valid_and_distinct_from_errors(self):
        fake_spy = {
            "2024-03-01": 100.0,
            "2024-03-08": 99.0,
            "2024-03-15": 98.0,
        }
        b = book.Book(
            "http://127.0.0.1:8000",
            "cautious-model",
            start="2024-03-01",
            end="2024-03-15",
            step_days=7,
            verbose=False,
        )
        with mock.patch.object(book, "get_closes", return_value=fake_spy), \
             mock.patch("market.get_closes", return_value=fake_spy), \
             mock.patch.object(b, "_call", return_value='{"close": [], "open": []}'):
            m = b.run()

        self.assertEqual(m["verdict"], "VALID")
        self.assertTrue(m["qualifying"])
        self.assertEqual(m["valid_decisions"], 3)
        self.assertEqual(m["deliberate_abstentions"], 3)
        self.assertEqual(m["failed_steps"], 0)
        self.assertEqual(m["coverage_pct"], 100.0)
        self.assertIsNotNone(m["alpha_vs_spy_cagr_pct"])
        self.assertEqual(m["alpha_vs_cash_cagr_pct"], 0.0)

    def test_below_threshold_coverage_flags_invalid_in_both_tracks(self):
        rows = [
            {"status": "ok", "model_ret": 0.50, "model_dte": 180, "doctrine_ret": 0.05},
            {"status": "call_error", "err": "call:timeout"},
            {"status": "parse_fail", "err": "parse_fail:none"},
            {"status": "schema_fail", "err": "schema_fail:bad"},
        ]
        summ = run_model_alpha.summarize(rows, min_coverage=0.80)
        self.assertEqual(summ["verdict"], "INVALID")
        self.assertFalse(summ["qualifying"])
        self.assertEqual(summ["coverage_pct"], 25.0)
        self.assertIsNone(summ["model_mean_ret_pct"])
        self.assertIsNone(summ["alpha_vs_doctrine_pct"])


class TestMetricsAccuracyOP05(unittest.TestCase):
    def test_duplicate_terminal_sample_does_not_distort_sharpe(self):
        """OP05 exact repro: two equal +10% weekly returns + duplicate final date must yield Sharpe 0.0."""
        curve_clean = [
            ("2024-02-01", 100_000.0),
            ("2024-02-08", 110_000.0),
            ("2024-02-15", 121_000.0),
        ]
        curve_with_dup = curve_clean + [("2024-02-15", 121_000.0)]

        s_clean = book.compute_equity_metrics(curve_clean, "2024-02-01", "2024-02-15", step_days=7)
        s_dup = book.compute_equity_metrics(curve_with_dup, "2024-02-01", "2024-02-15", step_days=7)

        self.assertEqual(s_clean["sharpe"], 0.0)
        self.assertEqual(s_dup["sharpe"], 0.0)
        self.assertEqual(s_clean, s_dup)

    def test_book_run_replaces_terminal_mark_on_exact_end_date(self):
        fake_spy = {
            "2024-02-01": 100.0,
            "2024-02-08": 102.0,
            "2024-02-15": 104.0,
        }
        b = book.Book(
            "http://127.0.0.1:8000",
            "m",
            start="2024-02-01",
            end="2024-02-15",
            step_days=7,
            verbose=False,
        )
        with mock.patch.object(book, "get_closes", return_value=fake_spy), \
             mock.patch("market.get_closes", return_value=fake_spy), \
             mock.patch.object(b, "_call", return_value='{"close": [], "open": []}'):
            b.run()

        dates = [d for d, _ in b.equity_curve]
        self.assertEqual(dates, ["2024-02-01", "2024-02-08", "2024-02-15"])
        self.assertEqual(len(dates), len(set(dates)))

    def test_annualization_scales_accurately_with_configured_sampling_frequency(self):
        curve = [
            ("2024-01-01", 100_000.0),
            ("2024-01-15", 105_000.0),
            ("2024-01-29", 102_000.0),
            ("2024-02-12", 108_000.0),
            ("2024-02-26", 112_000.0),
        ]
        s_weekly = book.compute_equity_metrics(curve, "2024-01-01", "2024-02-26", step_days=7, mark_frequency="step")
        s_biweekly = book.compute_equity_metrics(curve, "2024-01-01", "2024-02-26", step_days=14, mark_frequency="step")
        s_daily = book.compute_equity_metrics(curve, "2024-01-01", "2024-02-26", step_days=7, mark_frequency="daily")

        self.assertAlmostEqual(s_weekly["periods_per_year"], 365.25 / 7.0, places=3)
        self.assertAlmostEqual(s_biweekly["periods_per_year"], 365.25 / 14.0, places=3)
        self.assertEqual(s_daily["periods_per_year"], 252.0)

        # Weekly Sharpe must be sqrt(2) times biweekly Sharpe (never fixed sqrt(52) for both!)
        ratio = s_weekly["sharpe"] / s_biweekly["sharpe"]
        self.assertAlmostEqual(ratio, math.sqrt(2.0), delta=0.05)
        self.assertGreater(s_daily["sharpe"], s_weekly["sharpe"])

    def test_daily_equity_curve_captures_intraperiod_drawdown(self):
        step_curve = [
            ("2024-02-01", 100_000.0),
            ("2024-02-08", 102_000.0),
        ]
        daily_curve = [
            ("2024-02-01", 100_000.0),
            ("2024-02-05", 80_000.0),  # 20% intraperiod drawdown missed by weekly marks
            ("2024-02-08", 102_000.0),
        ]
        stats = book.compute_equity_metrics(
            step_curve, "2024-02-01", "2024-02-08", step_days=7, daily_equity_curve=daily_curve
        )
        self.assertEqual(stats["step_max_drawdown_pct"], 0.0)
        self.assertEqual(stats["max_drawdown_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
