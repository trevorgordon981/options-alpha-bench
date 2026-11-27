"""Regression tests for OP01 (fixed-stop policy) and OP02 (strict schema + durable replay)."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import book
import data
import engine
import run_model_alpha
import schema

FIXTURE_PATH = os.path.join(REPO_ROOT, "fixtures", "sample_prices.json")


class TestStopPolicyOP01(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        px_dir = os.path.join(self.tmpdir.name, "px")
        data.load_fixture_into_cache(FIXTURE_PATH, px_dir)
        self.env_patcher = mock.patch.dict(os.environ, {"BYRON_CACHE": px_dir})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def test_default_fixed_stop_prevents_model_stop_confounding_dte_comparison(self):
        captured_trades = []
        real_simulate = engine.simulate

        def spy_simulate(trade):
            captured_trades.append(trade)
            return real_simulate(trade)

        setups = [{"ticker": "NVDA", "entry_date": "2024-03-04", "horizon_days": 21, "mom20": 0.12}]
        model_reply = json.dumps({
            "direction": "long_call",
            "entry_dte": 168,
            "long_delta": 0.60,
            "stop_pct": 1.0,
            "conviction": "high",
        })

        with mock.patch.object(run_model_alpha, "call_model", return_value=model_reply), \
             mock.patch.object(run_model_alpha, "simulate", side_effect=spy_simulate):
            rows = run_model_alpha.score(setups, "http://127.0.0.1:8000", "m", verbose=False)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(captured_trades), 2)
        model_trade, baseline_trade = captured_trades
        # Both candidate and baseline MUST share the exact same 0.30 stop and 3-DTE time-stop
        self.assertEqual(model_trade.premium_stop, 0.30)
        self.assertEqual(baseline_trade.premium_stop, 0.30)
        self.assertEqual(model_trade.time_stop_dte, baseline_trade.time_stop_dte)

        row = rows[0]
        self.assertEqual(row["stop_policy"], "fixed")
        self.assertEqual(row["requested_stop_pct"], 1.0)
        self.assertEqual(row["applied_stop_pct"], 0.30)
        self.assertTrue(row["stop_override_applied"])
        self.assertFalse(row["policy_confounded"])

        summ = run_model_alpha.summarize(rows)
        self.assertEqual(summ["stop_policy"], "fixed")
        self.assertEqual(summ["stop_overrides"], 1)
        self.assertEqual(summ["policy_confounded_cases"], 0)

    def test_explicit_allow_model_stop_labels_policy_variation(self):
        captured_trades = []
        real_simulate = engine.simulate

        def spy_simulate(trade):
            captured_trades.append(trade)
            return real_simulate(trade)

        setups = [{"ticker": "NVDA", "entry_date": "2024-03-04", "horizon_days": 21, "mom20": 0.12}]
        model_reply = json.dumps({
            "direction": "long_call",
            "entry_dte": 168,
            "long_delta": 0.60,
            "stop_pct": 1.0,
            "conviction": "high",
        })

        with mock.patch.object(run_model_alpha, "call_model", return_value=model_reply), \
             mock.patch.object(run_model_alpha, "simulate", side_effect=spy_simulate):
            rows = run_model_alpha.score(
                setups, "http://127.0.0.1:8000", "m", allow_model_stop=True, verbose=False
            )

        model_trade, baseline_trade = captured_trades
        self.assertEqual(model_trade.premium_stop, 1.0)
        self.assertEqual(baseline_trade.premium_stop, 0.30)
        self.assertEqual(rows[0]["stop_policy"], "model_controlled")
        self.assertTrue(rows[0]["policy_confounded"])
        summ = run_model_alpha.summarize(rows)
        self.assertEqual(summ["stop_policy"], "model_controlled")
        self.assertEqual(summ["policy_confounded_cases"], 1)


class TestStrictSchemaAndDurableReplayOP02(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        px_dir = os.path.join(self.tmpdir.name, "px")
        data.load_fixture_into_cache(FIXTURE_PATH, px_dir)
        self.env_patcher = mock.patch.dict(os.environ, {"BYRON_CACHE": px_dir})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def test_construction_schema_rejects_malformed_types_and_bounds(self):
        bad_payloads = [
            {"direction": "long_call", "entry_dte": "bad"},
            {"direction": "long_call", "entry_dte": True},
            {"direction": "long_call", "entry_dte": 0},
            {"direction": "long_call", "entry_dte": 5000},
            {"direction": "long_call", "entry_dte": 90, "long_delta": "0.6"},
            {"direction": "long_call", "entry_dte": 90, "long_delta": float("nan")},
            {"direction": "long_call", "entry_dte": 90, "long_delta": 1.5},
            {"direction": "long_call", "entry_dte": 90, "stop_pct": -0.2},
            {"direction": "long_call", "entry_dte": 90, "stop_pct": 1.5},
            {"direction": "naked_short_put", "entry_dte": 90},
            {"direction": "long_call", "entry_dte": 90, "conviction": "super"},
        ]
        for bad in bad_payloads:
            valid, err = schema.validate_construction_trade(bad)
            self.assertIsNone(valid, f"Should have rejected {bad!r}")
            self.assertIsNotNone(err)

    def test_book_schema_rejects_null_open_and_malformed_types(self):
        bad_decisions = [
            {"open": [None]},
            {"open": "not-a-list"},
            {"close": [123]},
            {"close": None, "open": []},
            {"open": [{"ticker": "NVDA", "direction": "long_call", "entry_dte": "bad"}]},
            {"open": [{"ticker": "NVDA", "direction": "long_call", "entry_dte": True}]},
            {"open": [{"ticker": "NVDA", "direction": "long_call", "entry_dte": 90, "size_pct": 0.9}]},
            {"open": [{"ticker": "NVDA", "direction": "long_call", "entry_dte": 90, "size_pct": 0.0}]},
            {"open": [{"ticker": 123, "direction": "long_call", "entry_dte": 90}]},
        ]
        for bad in bad_decisions:
            valid, err = schema.validate_book_decision(bad)
            self.assertIsNone(valid, f"Should have rejected {bad!r}")
            self.assertIsNotNone(err)

    def test_construction_replay_survives_bad_field_types_and_persists_jsonl(self):
        setups = [
            {"ticker": "NVDA", "entry_date": "2024-03-04", "horizon_days": 21, "mom20": 0.12},
            {"ticker": "MSFT", "entry_date": "2024-03-18", "horizon_days": 21, "mom20": 0.09},
            {"ticker": "AAPL", "entry_date": "2024-04-01", "horizon_days": 21, "mom20": 0.08},
        ]
        replies = [
            '{"direction": "long_call", "entry_dte": "bad", "long_delta": 0.6}',
            '{"direction": "long_call", "entry_dte": 168, "long_delta": 0.6, "stop_pct": 0.30}',
            "I refuse to output JSON",
        ]
        ckpt = os.path.join(self.tmpdir.name, "cases.jsonl")

        with mock.patch.object(run_model_alpha, "call_model", side_effect=replies):
            rows = run_model_alpha.score(
                setups, "http://127.0.0.1:8000", "m", checkpoint_path=ckpt, verbose=False
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["status"], "schema_fail")
        self.assertEqual(rows[1]["status"], "ok")
        self.assertEqual(rows[2]["status"], "parse_fail")

        # Verify durable per-case JSONL persisted all 3 terminal outcomes
        with open(ckpt, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual([l["status"] for l in lines], ["schema_fail", "ok", "parse_fail"])

    def test_book_replay_survives_open_null_and_persists_step_jsonl(self):
        trace_path = os.path.join(self.tmpdir.name, "book_steps.jsonl")
        b = book.Book(
            "http://127.0.0.1:8000",
            "m",
            start="2024-03-04",
            end="2024-03-18",
            step_days=7,
            verbose=False,
            trace_path=trace_path,
        )
        replies = [
            '{"close": [], "open": [null]}',  # OP02 repro: used to crash with AttributeError
            '{"close": [], "open": [{"ticker": "NVDA", "direction": "long_call", "entry_dte": 168, "long_delta": 0.6, "size_pct": 0.10}]}',
            '{"close": [], "open": []}',
        ]
        with mock.patch.object(b, "_call", side_effect=replies):
            m = b.run()

        self.assertEqual(m["decision_steps"], 3)
        self.assertEqual(m["schema_fails"], 1)
        self.assertEqual(m["valid_decisions"], 2)
        self.assertEqual(m["deliberate_abstentions"], 1)
        self.assertEqual(m["failed_steps"], 1)
        # Coverage = 2/3 = 66.7% < 80% default -> INVALID
        self.assertEqual(m["verdict"], "INVALID")

        with open(trace_path, encoding="utf-8") as f:
            steps = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([s["status"] for s in steps], ["schema_fail", "ok", "abstain"])


if __name__ == "__main__":
    unittest.main()
