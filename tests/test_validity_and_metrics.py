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


class TestChronologicalTimelineAndSemanticActionsOP09OP10(unittest.TestCase):
    def test_future_exit_shock_never_changes_earlier_daily_mark_op10(self):
        """OP10 exact counterfactual: Jan 6 crash stop-exit MUST NOT alter Jan 3 equity mark."""
        base_dates = {
            "2024-12-02": 100.0,
            "2024-12-09": 102.0,
            "2024-12-16": 104.0,
            "2024-12-23": 103.0,
            "2024-12-30": 105.0,
            "2025-01-02": 100.0,  # Thu: entry
            "2025-01-03": 115.0,  # Fri: strong rally (option gains materially)
        }
        spy_series = {**base_dates, "2025-01-06": 100.0}
        nvda_shock = {**base_dates, "2025-01-06": 60.0}     # Mon: crash triggers -30% stop
        nvda_no_shock = {**base_dates, "2025-01-06": 115.0} # Mon: flat, stays open

        open_reply = (
            '{"close": [], "open": [{"ticker": "NVDA", "direction": "long_call", '
            '"entry_dte": 90, "long_delta": 0.60, "size_pct": 0.25}]}'
        )

        def _run_case(nvda_px, mark_frequency):
            def _fake_get_closes(ticker, start, end):
                src = spy_series if ticker.upper() == "SPY" else nvda_px
                return {k: v for k, v in src.items() if start <= k <= end}

            b = book.Book(
                "http://127.0.0.1:8000",
                "m",
                start="2025-01-02",
                end="2025-01-06",
                step_days=7,
                mark_frequency=mark_frequency,
                verbose=False,
                basket=["NVDA", "SPY"],
            )
            with mock.patch.object(book, "get_closes", side_effect=_fake_get_closes), \
                 mock.patch("market.get_closes", side_effect=_fake_get_closes), \
                 mock.patch.object(b, "_call", return_value=open_reply):
                m = b.run()
            return b, m

        for freq in ("daily", "step"):
            b_shock, _ = _run_case(nvda_shock, freq)
            b_calm, _ = _run_case(nvda_no_shock, freq)

            # Confirm the Jan 6 stop actually fired in the shock scenario and NOT in calm
            self.assertEqual(len(b_shock.closed), 1)
            self.assertEqual(b_shock.closed[0].exit_reason, "stop")
            self.assertEqual(b_shock.closed[0].exit_date, "2025-01-06")
            self.assertEqual(b_calm.closed[0].exit_reason, "final_liquidation")

            daily_shock = dict(b_shock.daily_equity_curve)
            daily_calm = dict(b_calm.daily_equity_curve)

            # CRITICAL: Jan 3 mark MUST be identical regardless of the future Jan 6 shock!
            self.assertIn("2025-01-03", daily_shock)
            self.assertIn("2025-01-03", daily_calm)
            self.assertEqual(daily_shock["2025-01-03"], daily_calm["2025-01-03"])
            self.assertGreater(daily_shock["2025-01-03"], book.START_EQUITY)
            self.assertLess(daily_shock["2025-01-06"], book.START_EQUITY)

            # Direct point-in-time valuation check on Jan 3 (using only info available Jan 3)
            pos = b_shock.positions[0]
            cash_after_entry = book.START_EQUITY - pos.entry_cost_ps * 100 * pos.contracts
            from datetime import date as _date
            expected_jan3 = round(
                cash_after_entry + pos.value_ps(115.0, _date(2025, 1, 3).toordinal()) * 100 * pos.contracts,
                2,
            )
            self.assertAlmostEqual(daily_shock["2025-01-03"], expected_jan3, places=2)
            if freq == "daily":
                self.assertEqual(dict(b_shock.equity_curve)["2025-01-03"], expected_jan3)

    def test_daily_mode_excludes_weekend_decision_and_end_marks_op05_op10(self):
        """OP05/OP10 exact repro: 2-day cadence landing on Saturday 2025-01-04 must NOT pollute daily series."""
        trading_px = {
            "2025-01-02": 100.0,  # Thu
            "2025-01-03": 101.0,  # Fri
            "2025-01-06": 102.0,  # Mon
        }

        def _fake_get_closes(ticker, start, end):
            return {k: v for k, v in trading_px.items() if start <= k <= end}

        b = book.Book(
            "http://127.0.0.1:8000",
            "m",
            start="2025-01-02",
            end="2025-01-06",
            step_days=2,  # Decision dates: Thu Jan 2, Sat Jan 4, Mon Jan 6
            mark_frequency="daily",
            verbose=False,
            basket=["SPY"],
        )
        with mock.patch.object(book, "get_closes", side_effect=_fake_get_closes), \
             mock.patch("market.get_closes", side_effect=_fake_get_closes), \
             mock.patch.object(b, "_call", return_value='{"close": [], "open": []}'):
            m = b.run()

        self.assertEqual(m["periods_per_year"], 252.0)
        self.assertEqual(m["sampling_frequency"], "daily_trading_sessions")
        # Saturday 2025-01-04 MUST NOT appear in daily equity_curve or daily_equity_curve
        eq_dates = [d for d, _ in b.equity_curve]
        daily_dates = [d for d, _ in b.daily_equity_curve]
        self.assertEqual(eq_dates, ["2025-01-02", "2025-01-03", "2025-01-06"])
        self.assertEqual(daily_dates, ["2025-01-02", "2025-01-03", "2025-01-06"])
        self.assertNotIn("2025-01-04", eq_dates)
        self.assertNotIn("2025-01-04", daily_dates)
        # Saturday 2025-01-04 defers to Monday 2025-01-06 (deduplicated) — never executed on Saturday (OP11)
        self.assertEqual([d for d, _ in b.decision_equity_curve], ["2025-01-02", "2025-01-06"])
        self.assertNotIn("2025-01-04", [d for d, _ in b.decision_equity_curve])

        # Weekend end date (Sunday 2025-01-05) must also be excluded in daily mode
        b_wknd_end = book.Book(
            "http://127.0.0.1:8000",
            "m",
            start="2025-01-02",
            end="2025-01-05",
            step_days=2,
            mark_frequency="daily",
            verbose=False,
            basket=["SPY"],
        )
        with mock.patch.object(book, "get_closes", side_effect=_fake_get_closes), \
             mock.patch("market.get_closes", side_effect=_fake_get_closes), \
             mock.patch.object(b_wknd_end, "_call", return_value='{"close": [], "open": []}'):
            b_wknd_end.run()
        self.assertEqual([d for d, _ in b_wknd_end.equity_curve], ["2025-01-02", "2025-01-03"])
        self.assertEqual([d for d, _ in b_wknd_end.daily_equity_curve], ["2025-01-02", "2025-01-03"])

    def test_invalid_close_position_ids_rejected_as_unusable_actions_op09(self):
        """OP09 exact repro: 3 replies closing 'never_opened' over falling SPY MUST fail validation."""
        fake_spy = {
            "2024-03-01": 100.0,
            "2024-03-08": 99.0,
            "2024-03-15": 98.0,
        }
        b = book.Book(
            "http://127.0.0.1:8000",
            "impossible-close-model",
            start="2024-03-01",
            end="2024-03-15",
            step_days=7,
            verbose=False,
        )
        with mock.patch.object(book, "get_closes", return_value=fake_spy), \
             mock.patch("market.get_closes", return_value=fake_spy), \
             mock.patch.object(b, "_call", return_value='{"close": ["never_opened"], "open": []}'):
            m = b.run()

        self.assertEqual(m["verdict"], "INVALID")
        self.assertFalse(m["qualifying"])
        self.assertEqual(m["decision_steps"], 3)
        self.assertEqual(m["valid_decisions"], 0)
        self.assertEqual(m["deliberate_abstentions"], 0)
        self.assertEqual(m["invalid_actions"], 3)
        self.assertEqual(m["failed_steps"], 3)
        self.assertEqual(m["coverage_pct"], 0.0)
        self.assertIsNone(m["alpha_vs_spy_cagr_pct"])
        self.assertIsNone(m["cagr_pct"])
        self.assertIsNone(m["sharpe"])
        self.assertTrue(all(s["status"] == "invalid_action" for s in b.step_log))
        self.assertTrue(all("never_opened" in s["error"] for s in b.step_log))

    def test_unknown_ticker_duplicate_and_already_closed_id_rejected_op09(self):
        fake_spy = {
            "2024-02-01": 95.0,
            "2024-02-08": 96.0,
            "2024-02-15": 97.0,
            "2024-02-22": 98.0,
            "2024-03-01": 100.0,
            "2024-03-08": 101.0,
            "2024-03-15": 102.0,
            "2024-03-22": 103.0,
            "2024-03-29": 104.0,
        }
        b = book.Book(
            "http://127.0.0.1:8000",
            "mixed-action-model",
            start="2024-03-01",
            end="2024-03-29",
            step_days=7,
            verbose=False,
            basket=["SPY"],
        )
        replies = [
            # Step 1: unknown ticker -> invalid_action
            '{"close": [], "open": [{"ticker": "NOT_IN_BASKET", "direction": "long_call", "entry_dte": 90, "long_delta": 0.6, "size_pct": 0.1}]}',
            # Step 2: duplicate close id -> schema_fail
            '{"close": ["p1", "p1"], "open": []}',
            # Step 3: valid open p1 -> ok
            '{"close": [], "open": [{"ticker": "SPY", "direction": "long_call", "entry_dte": 90, "long_delta": 0.6, "size_pct": 0.1}]}',
            # Step 4: valid close p1 -> ok
            '{"close": ["p1"], "open": []}',
            # Step 5: re-closing already-closed p1 -> invalid_action
            '{"close": ["p1"], "open": []}',
        ]
        with mock.patch.object(book, "get_closes", return_value=fake_spy), \
             mock.patch("market.get_closes", return_value=fake_spy), \
             mock.patch.object(b, "_call", side_effect=replies):
            m = b.run()

        self.assertEqual(m["decision_steps"], 5)
        self.assertEqual(m["valid_decisions"], 2)
        self.assertEqual(m["invalid_actions"], 2)
        self.assertEqual(m["schema_fails"], 1)
        self.assertEqual(m["failed_steps"], 3)
        self.assertEqual(m["verdict"], "INVALID")  # 40% < 80% coverage
        self.assertEqual(
            [s["status"] for s in b.step_log],
            ["invalid_action", "schema_fail", "ok", "ok", "invalid_action"],
        )


class TestTradingSessionSchedulingAndChronologyOP11(unittest.TestCase):
    def test_friday_to_sunday_two_day_cadence_never_executes_sunday_or_inverts_exit_op11(self):
        """OP11 exact repro: Fri 2025-01-03 -> Sun 2025-01-05 (step=2) must NOT execute Sunday or set exit < entry."""
        px_series = {
            "2024-12-02": 95.0,
            "2024-12-09": 96.0,
            "2024-12-16": 97.0,
            "2024-12-23": 98.0,
            "2024-12-30": 99.0,
            "2025-01-02": 100.0,
            "2025-01-03": 102.0,  # Fri: sole valid trading session in [2025-01-03, 2025-01-05]
        }

        def _fake_get_closes(ticker, start, end):
            return {k: v for k, v in px_series.items() if start <= k <= end}

        open_reply = (
            '{"close": [], "open": [{"ticker": "NVDA", "direction": "long_call", '
            '"entry_dte": 90, "long_delta": 0.60, "size_pct": 0.15}]}'
        )
        abstain_reply = '{"close": [], "open": []}'

        # Exact audit scenario: abstain Friday, would have opened Sunday if Sunday were called
        for freq in ("daily", "step"):
            prompts_seen = []

            def _responder_abstain_fri_open_second(prompt, max_tokens=1200):
                prompts_seen.append(prompt)
                return abstain_reply if len(prompts_seen) == 1 else open_reply

            b = book.Book(
                "http://127.0.0.1:8000",
                "m",
                start="2025-01-03",
                end="2025-01-05",
                step_days=2,
                mark_frequency=freq,
                verbose=False,
                basket=["NVDA", "SPY"],
            )
            with mock.patch.object(book, "get_closes", side_effect=_fake_get_closes), \
                 mock.patch.object(b, "_call", side_effect=_responder_abstain_fri_open_second):
                m = b.run()

            # Sunday 2025-01-05 MUST NOT be executed
            self.assertEqual(len(prompts_seen), 1)
            self.assertIn("Date: 2025-01-03", prompts_seen[0])
            self.assertEqual(b.decision_schedule, ["2025-01-03"])
            self.assertEqual(m["decision_steps"], 1)
            self.assertEqual(m["n_trades"], 0)
            self.assertEqual([s["date"] for s in b.step_log], ["2025-01-03"])
            self.assertNotIn("2025-01-05", [d for d, _ in b.equity_curve])
            self.assertNotIn("2025-01-05", [d for d, _ in b.daily_equity_curve])
            self.assertNotIn("2025-01-05", [d for d, _ in b.decision_equity_curve])
            self.assertTrue(all(p.exit_date >= p.entry_date for p in b.closed))

            # Opening on Friday 2025-01-03 must also liquidate on >= entry_date (never earlier, never Sunday)
            b_open = book.Book(
                "http://127.0.0.1:8000",
                "m",
                start="2025-01-03",
                end="2025-01-05",
                step_days=2,
                mark_frequency=freq,
                verbose=False,
                basket=["NVDA", "SPY"],
            )
            with mock.patch.object(book, "get_closes", side_effect=_fake_get_closes), \
                 mock.patch.object(b_open, "_call", return_value=open_reply):
                m_open = b_open.run()

            self.assertEqual(m_open["n_trades"], 1)
            p = b_open.closed[0]
            self.assertEqual(p.entry_date, "2025-01-03")
            self.assertEqual(p.exit_date, "2025-01-03")
            self.assertGreaterEqual(p.exit_date, p.entry_date)

    def test_weekend_cadence_defers_to_next_valid_session_within_window_op11(self):
        """Nominal Sunday 2025-01-05 defers to Monday 2025-01-06 when Monday is inside [start, end]."""
        px_series = {
            "2024-12-02": 95.0,
            "2024-12-09": 96.0,
            "2024-12-16": 97.0,
            "2024-12-23": 98.0,
            "2024-12-30": 99.0,
            "2025-01-02": 100.0,
            "2025-01-03": 100.0,  # Fri
            "2025-01-06": 110.0,  # Mon
        }

        def _fake_get_closes(ticker, start, end):
            return {k: v for k, v in px_series.items() if start <= k <= end}

        replies = [
            '{"close": [], "open": []}',
            '{"close": [], "open": [{"ticker": "NVDA", "direction": "long_call", "entry_dte": 90, "long_delta": 0.60, "size_pct": 0.15}]}',
        ]
        b = book.Book(
            "http://127.0.0.1:8000",
            "m",
            start="2025-01-03",
            end="2025-01-06",
            step_days=2,  # Nominal: Fri Jan 3, Sun Jan 5 -> defers to Mon Jan 6
            mark_frequency="daily",
            verbose=False,
            basket=["NVDA", "SPY"],
        )
        with mock.patch.object(book, "get_closes", side_effect=_fake_get_closes), \
             mock.patch.object(b, "_call", side_effect=replies):
            m = b.run()

        self.assertEqual(b.decision_schedule, ["2025-01-03", "2025-01-06"])
        self.assertEqual(m["decision_steps"], 2)
        self.assertEqual(b.step_log[1]["date"], "2025-01-06")
        self.assertEqual(b.step_log[1]["nominal_date"], "2025-01-05")
        self.assertEqual(len(b.closed), 1)
        self.assertEqual(b.closed[0].entry_date, "2025-01-06")
        self.assertEqual(b.closed[0].exit_date, "2025-01-06")
        self.assertGreaterEqual(b.closed[0].exit_date, b.closed[0].entry_date)

    def test_close_guard_rejects_exit_date_earlier_than_entry_date_op11(self):
        b = book.Book("http://127.0.0.1:8000", "m", verbose=False)
        pos = book.Position(
            pid="p1",
            ticker="NVDA",
            direction="long_call",
            entry_date="2025-01-06",
            expiry_ord=739300,
            iv=0.40,
            contracts=1,
            entry_cost_ps=5.0,
            legs={"long_k": 100.0},
        )
        with self.assertRaises(ValueError):
            b._close(pos, "2025-01-03", 5.0, "final_liquidation")


if __name__ == "__main__":
    unittest.main()
