"""
Autonomous book backtest — Track 2 of Byron Bench.

The model runs a simulated options book over a date window:
  - decision steps every `step_days` calendar days; between steps, fixed exit
    discipline runs continuously (-30% premium stop + 3-DTE time-stop on daily closes)
  - at each decision point the model sees equity, cash, open positions, and a
    market slate, and returns strict JSON: {"close": [...], "open": [...]}
  - options are BS-marked on daily underlying closes (synthetic pricing, not
    historical option chains)
  - overall validity is gated on usable model decision coverage; deliberate
    abstentions are tracked separately from transport, parse, and schema errors
  - equity marks are deduplicated by date (replacing the terminal mark after
    liquidation) and annualized strictly from the configured sampling frequency
"""
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
import os

from data import get_closes
from llm_client import chat_completion
from market import slate
from pricing import bs_price, estimate_iv, strike_for_delta
from scenarios import BASKET
from schema import (
    append_jsonl_record,
    extract_json_objects,
    validate_book_decision,
    validate_book_open_spec,
)

START_EQUITY = 100_000.0
MAX_CONCURRENT = 8
MAX_SIZE_PCT = 0.25       # cap any single new position at 25% of equity
PREMIUM_STOP = 0.30
TIME_STOP_DTE = 3
MIN_VALID_COVERAGE = 0.80
TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class Position:
    pid: str
    ticker: str
    direction: str
    entry_date: str
    expiry_ord: int
    iv: float
    contracts: int
    entry_cost_ps: float       # per-share premium paid
    legs: dict
    peak_val_ps: float = 0.0   # for reporting
    closed: bool = False
    exit_date: str = ""
    exit_val_ps: float = 0.0
    exit_reason: str = ""

    def kind(self):
        return "call" if "call" in self.direction else "put"

    def value_ps(self, spot, on_ord):
        t = max(self.expiry_ord - on_ord, 0) / 365.0
        if "spread" in self.direction:
            return (bs_price(spot, self.legs["long_k"], t, self.iv, self.kind())
                    - bs_price(spot, self.legs["short_k"], t, self.iv, self.kind()))
        return bs_price(spot, self.legs["long_k"], t, self.iv, self.kind())


def _closes_cached(ticker, start, end):
    return get_closes(ticker, start, end)


def _spot_on_or_before(ticker, iso, cache, window_start=None, window_end=None):
    """Most recent close at/before iso (handles weekends/holidays)."""
    if ticker not in cache:
        d0 = date.fromisoformat(iso)
        lo = window_start if window_start else (d0 - timedelta(days=90)).isoformat()
        hi = window_end if window_end else (d0 + timedelta(days=400)).isoformat()
        lo = min(lo, (d0 - timedelta(days=45)).isoformat())
        hi = max(hi, iso)
        cache[ticker] = get_closes(ticker, lo, hi)
    items = [(k, v) for k, v in cache[ticker].items() if k <= iso]
    return max(items)[1] if items else None


def _daily_between(ticker, a_iso, b_iso, cache, window_start=None, window_end=None):
    if ticker not in cache:
        _spot_on_or_before(ticker, b_iso, cache, window_start=window_start, window_end=window_end)
    return sorted((k, v) for k, v in cache[ticker].items() if a_iso < k <= b_iso)


def deduplicate_equity_curve(curve):
    """Collapse duplicate date marks by keeping the final mark per ISO date, sorted chronologically."""
    by_date = {}
    for iso, eq in curve:
        by_date[iso] = float(eq)
    return [(d, by_date[d]) for d in sorted(by_date)]


def periods_per_year_for(step_days: int, mark_frequency: str = "step") -> float:
    """Return the documented annualization periods/year for the configured sampling frequency."""
    if mark_frequency == "daily":
        return TRADING_DAYS_PER_YEAR
    if step_days <= 0:
        raise ValueError(f"step_days must be positive, got {step_days}")
    return 365.25 / float(step_days)


def compute_equity_metrics(
    equity_curve,
    start: str,
    end: str,
    *,
    step_days: int = 7,
    mark_frequency: str = "step",
    daily_equity_curve=None,
) -> dict:
    """Compute deduplicated return, CAGR, Sharpe, and drawdown statistics."""
    dedup = deduplicate_equity_curve(equity_curve)
    if not dedup:
        return {
            "final_equity": START_EQUITY,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "step_max_drawdown_pct": 0.0,
            "periods_per_year": round(periods_per_year_for(step_days, mark_frequency), 4),
            "sampling_frequency": "daily_trading_sessions" if mark_frequency == "daily" else f"every_{step_days}_days",
            "duration_days": max((date.fromisoformat(end) - date.fromisoformat(start)).days, 0),
        }

    eq = [e for _, e in dedup]
    rets = [(eq[i] / eq[i - 1] - 1.0) for i in range(1, len(eq)) if eq[i - 1] > 0]
    ppy = periods_per_year_for(step_days, mark_frequency)
    if len(rets) >= 2:
        mean_r = sum(rets) / len(rets)
        var_r = sum((x - mean_r) ** 2 for x in rets) / (len(rets) - 1)
        sd_r = math.sqrt(var_r)
        # Guard against floating-point noise on constant-return series
        sharpe = (mean_r / sd_r * math.sqrt(ppy)) if sd_r > 1e-12 else 0.0
    else:
        sharpe = 0.0

    def _max_dd(series):
        if not series:
            return 0.0
        peak, mdd = series[0], 0.0
        for val in series:
            peak = max(peak, val)
            if peak > 0:
                mdd = max(mdd, (peak - val) / peak)
        return mdd

    step_mdd = _max_dd(eq)
    if daily_equity_curve:
        daily_dedup = deduplicate_equity_curve(daily_equity_curve)
        mdd = max(step_mdd, _max_dd([e for _, e in daily_dedup]))
    else:
        mdd = step_mdd

    duration_days = max((date.fromisoformat(end) - date.fromisoformat(start)).days, 0)
    yrs = duration_days / 365.25
    total_ret = (eq[-1] / eq[0] - 1.0) if eq[0] > 0 else 0.0
    cagr = (eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0 if eq[0] > 0 and yrs > 0 else 0.0

    return {
        "final_equity": round(eq[-1], 2),
        "total_return_pct": round(total_ret * 100.0, 1),
        "cagr_pct": round(cagr * 100.0, 1),
        "cagr_raw": cagr,
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd * 100.0, 1),
        "step_max_drawdown_pct": round(step_mdd * 100.0, 1),
        "periods_per_year": round(ppy, 4),
        "sampling_frequency": "daily_trading_sessions" if mark_frequency == "daily" else f"every_{step_days}_days",
        "duration_days": duration_days,
    }


class Book:
    def __init__(
        self,
        base,
        model,
        start="2024-02-01",
        end="2026-04-15",
        step_days=7,
        verbose=True,
        *,
        api_key=None,
        api_key_env=None,
        min_coverage=MIN_VALID_COVERAGE,
        mark_frequency="step",
        trace_path=None,
        basket=None,
    ):
        if step_days <= 0:
            raise ValueError(f"step_days must be positive, got {step_days}")
        if mark_frequency not in ("step", "daily"):
            raise ValueError(f"mark_frequency must be 'step' or 'daily', got {mark_frequency!r}")
        self.base, self.model = base, model
        self.start, self.end, self.step_days = start, end, step_days
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.min_coverage = min_coverage
        self.mark_frequency = mark_frequency
        self.trace_path = trace_path
        self.basket = list(BASKET if basket is None else basket)

        self.cash = START_EQUITY
        self.positions = []
        self.closed = []
        self.equity_curve = []          # [(date, equity)]
        self.daily_equity_curve = []    # [(trading_date, equity)]
        self.step_log = []
        self.pcount = 0
        self.px = {}                    # ticker -> {date: close}
        self.verbose = verbose

        # Distinct outcome counters (OP04)
        self.calls = 0
        self.decision_steps_run = 0
        self.valid_decisions = 0
        self.deliberate_abstentions = 0
        self.transport_errors = 0
        self.parse_fails = 0
        self.schema_fails = 0

    # ---- model I/O ----
    def _call(self, prompt, max_tokens=1200):
        out = chat_completion(
            self.base,
            self.model,
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            timeout=200,
            api_key=self.api_key,
            api_key_env=self.api_key_env,
        )
        self.calls += 1
        return out

    @staticmethod
    def _parse_and_validate(text):
        """Parse and strictly validate a book decision JSON object.

        Returns (validated_dict | None, error_kind | None, error_detail | None).
        """
        objs = extract_json_objects(text)
        candidates = [o for o in objs if "open" in o or "close" in o]
        if not candidates:
            return None, "parse_fail", "no JSON object containing 'open' or 'close' found"
        last_err = None
        for cand in reversed(candidates):
            valid, err = validate_book_decision(cand)
            if valid is not None:
                return valid, None, None
            if last_err is None:
                last_err = err
        return None, "schema_fail", last_err

    @staticmethod
    def _parse(text):
        valid, _, _ = Book._parse_and_validate(text)
        return valid

    # ---- equity recording ----
    def _record_equity(self, iso, equity, curve=None):
        """Record an equity mark, replacing an existing mark on the same date."""
        target = self.equity_curve if curve is None else curve
        rounded = round(float(equity), 2)
        if target and target[-1][0] == iso:
            target[-1] = (iso, rounded)
        else:
            target.append((iso, rounded))

    def _trading_sessions_between(self, a_iso, b_iso):
        try:
            spy = get_closes("SPY", a_iso, b_iso)
            dates = sorted(d for d in spy.keys() if a_iso <= d <= b_iso)
            if dates:
                return dates
        except Exception:
            pass
        # Fallback to any cached ticker dates
        all_dates = set()
        for series in self.px.values():
            for d in series:
                if a_iso <= d <= b_iso:
                    all_dates.add(d)
        return sorted(all_dates)

    def _mark_daily_between(self, a_iso, b_iso):
        """Record one equity mark per trading session in (a_iso, b_iso] (and a_iso if empty)."""
        sessions = self._trading_sessions_between(a_iso, b_iso)
        for s_iso in sessions:
            if self.daily_equity_curve and s_iso < self.daily_equity_curve[-1][0]:
                continue
            if self.daily_equity_curve and s_iso == self.daily_equity_curve[-1][0]:
                continue
            eq = self._mark_equity(s_iso)
            self._record_equity(s_iso, eq, self.daily_equity_curve)
            if self.mark_frequency == "daily":
                self._record_equity(s_iso, eq, self.equity_curve)

    # ---- mechanics ----
    def _auto_exits(self, a_iso, b_iso):
        """Run -30% stop + 3-DTE time-stop continuously between decision dates."""
        for p in self.positions:
            if p.closed:
                continue
            for d_iso, spot in _daily_between(p.ticker, a_iso, b_iso, self.px):
                d_ord = date.fromisoformat(d_iso).toordinal()
                dte = p.expiry_ord - d_ord
                val = p.value_ps(spot, d_ord)
                p.peak_val_ps = max(p.peak_val_ps, val)
                pnl = (val - p.entry_cost_ps) / p.entry_cost_ps if p.entry_cost_ps else 0
                if dte <= TIME_STOP_DTE:
                    self._close(p, d_iso, val, "time_stop"); break
                if pnl <= -PREMIUM_STOP:
                    self._close(p, d_iso, val, "stop"); break

    def _close(self, p, iso, val_ps, reason):
        p.closed = True
        p.exit_date, p.exit_val_ps, p.exit_reason = iso, val_ps, reason
        self.cash += val_ps * 100 * p.contracts
        self.closed.append(p)

    def _mark_equity(self, iso):
        eq = self.cash
        for p in self.positions:
            if p.closed:
                continue
            spot = _spot_on_or_before(p.ticker, iso, self.px)
            if spot is None:
                continue
            eq += p.value_ps(spot, date.fromisoformat(iso).toordinal()) * 100 * p.contracts
        return eq

    def _open(self, iso, spec, equity):
        valid_spec, err = validate_book_open_spec(spec)
        if err is not None:
            return None
        tk = valid_spec["ticker"]
        direction = valid_spec["direction"]
        if tk not in self.basket:
            return None
        spot = _spot_on_or_before(tk, iso, self.px, window_start=self.start, window_end=self.end)
        if not spot:
            return None
        dte = valid_spec["entry_dte"]
        delta = valid_spec["long_delta"]
        size_pct = valid_spec["size_pct"]
        # IV from that ticker's realized vol at entry
        d0 = date.fromisoformat(iso)
        hist = get_closes(tk, (d0 - timedelta(days=45)).isoformat(), iso)
        iv = estimate_iv([v for _, v in sorted(hist.items())], 20)
        t0 = dte / 365.0
        k = "call" if "call" in direction else "put"
        legs = {"long_k": strike_for_delta(spot, t0, iv, delta, k)}
        if "spread" in direction:
            legs["short_k"] = strike_for_delta(spot, t0, iv, 0.30, k)
            cost_ps = (bs_price(spot, legs["long_k"], t0, iv, k)
                       - bs_price(spot, legs["short_k"], t0, iv, k))
        else:
            cost_ps = bs_price(spot, legs["long_k"], t0, iv, k)
        if cost_ps <= 0.05:
            return None
        budget = size_pct * equity
        contracts = int(budget // (cost_ps * 100))
        if contracts < 1 or cost_ps * 100 * contracts > self.cash:
            contracts = int(min(budget, self.cash) // (cost_ps * 100))
        if contracts < 1:
            return None
        self.pcount += 1
        p = Position(pid=f"p{self.pcount}", ticker=tk, direction=direction, entry_date=iso,
                     expiry_ord=d0.toordinal() + dte, iv=iv, contracts=contracts,
                     entry_cost_ps=cost_ps, legs={kk: round(vv, 2) for kk, vv in legs.items()})
        self.cash -= cost_ps * 100 * contracts
        self.positions.append(p)
        return p

    def _prompt(self, iso, equity):
        opens = [p for p in self.positions if not p.closed]
        lines = []
        for p in opens:
            spot = _spot_on_or_before(p.ticker, iso, self.px)
            val = p.value_ps(spot, date.fromisoformat(iso).toordinal())
            pnl = (val - p.entry_cost_ps) / p.entry_cost_ps * 100 if p.entry_cost_ps else 0
            dte = p.expiry_ord - date.fromisoformat(iso).toordinal()
            lines.append(f'  {p.pid}: {p.ticker} {p.direction} x{p.contracts}, {dte} DTE, P&L {pnl:+.0f}%')
        pos_block = "\n".join(lines) if lines else "  (none)"
        sl = slate(self.basket, iso, top=8)
        slate_block = "\n".join(
            f'  {c["ticker"]}: ${c["spot"]}, 5d {c["ret_5d"]*100:+.0f}%, 20d {c["ret_20d"]*100:+.0f}%, '
            f'{c["pct_below_high"]*100:.0f}% below high, rvol {c["rvol"]:.0%}' for c in sl)
        return f"""You are running a defined-risk options book. Date: {iso}.
Equity: ${equity:,.0f} | Cash: ${self.cash:,.0f} | Open positions: {len(opens)}/{MAX_CONCURRENT}

OPEN POSITIONS:
{pos_block}

MARKET SLATE (daily-close data as of today):
{slate_block}

Rules: small defined-risk debits; fixed -30% stop + 3-DTE time-stop run automatically.
Decide for this period. Output ONE strict JSON object on its own line:
{{"close": ["p1", ...], "open": [{{"ticker": "NVDA", "direction": "long_call|long_put|call_debit_spread|put_debit_spread", "entry_dte": <int>, "long_delta": <0.05-0.95>, "size_pct": <0.02-0.25>}}]}}
Open [] to sit in cash. Never exceed {MAX_CONCURRENT} open positions."""

    def _record_step(self, record):
        self.step_log.append(record)
        if self.trace_path:
            append_jsonl_record(self.trace_path, record)

    # ---- run ----
    def run(self):
        d = date.fromisoformat(self.start)
        end_d = date.fromisoformat(self.end)
        prev_iso = self.start
        while d <= end_d:
            iso = d.isoformat()
            self._auto_exits(prev_iso, iso)
            self._mark_daily_between(prev_iso, iso)
            equity = self._mark_equity(iso)
            if self.mark_frequency == "step":
                self._record_equity(iso, equity, self.equity_curve)
            else:
                self._record_equity(iso, equity, self.equity_curve)
            self._record_equity(iso, equity, self.daily_equity_curve)

            self.decision_steps_run += 1
            step_record = {
                "step": self.decision_steps_run,
                "date": iso,
                "equity_before": round(equity, 2),
                "cash_before": round(self.cash, 2),
            }

            prompt_text = self._prompt(iso, equity)
            try:
                out = self._call(prompt_text)
            except Exception as e:
                self.transport_errors += 1
                step_record["status"] = "transport_error"
                step_record["error"] = str(e)
                self._record_step(step_record)
                if self.verbose:
                    print(f"  {iso} call err: {e}", flush=True)
            else:
                dec, err_kind, err_detail = self._parse_and_validate(out)
                if dec is None:
                    if err_kind == "schema_fail":
                        self.schema_fails += 1
                    else:
                        self.parse_fails += 1
                    step_record["status"] = err_kind
                    step_record["error"] = err_detail
                    self._record_step(step_record)
                    if self.verbose:
                        print(f"  {iso} {err_kind}: {err_detail}", flush=True)
                else:
                    self.valid_decisions += 1
                    is_abstention = (not dec["close"]) and (not dec["open"])
                    if is_abstention:
                        self.deliberate_abstentions += 1

                    closed_pids = []
                    for pid in dec["close"]:
                        for p in self.positions:
                            if p.pid == pid and not p.closed:
                                spot = _spot_on_or_before(p.ticker, iso, self.px)
                                if spot is not None:
                                    self._close(p, iso, p.value_ps(spot, d.toordinal()), "model_close")
                                    closed_pids.append(pid)

                    opened_pids = []
                    nopen = len([p for p in self.positions if not p.closed])
                    for spec in dec["open"]:
                        if nopen >= MAX_CONCURRENT:
                            break
                        opened = self._open(iso, spec, equity)
                        if opened:
                            opened_pids.append(opened.pid)
                            nopen += 1

                    step_record["status"] = "abstain" if is_abstention else "ok"
                    step_record["closed"] = closed_pids
                    step_record["opened"] = opened_pids
                    self._record_step(step_record)

            if self.verbose:
                print(
                    f"  {iso}  eq=${equity:,.0f}  open={len([p for p in self.positions if not p.closed])} "
                    f"closed={len(self.closed)}",
                    flush=True,
                )
            prev_iso = iso
            d += timedelta(days=self.step_days)

        # Liquidate remaining open positions at final marks and REPLACE terminal mark
        fin = end_d.isoformat()
        self._auto_exits(prev_iso, fin)
        self._mark_daily_between(prev_iso, fin)
        for p in self.positions:
            if not p.closed:
                spot = _spot_on_or_before(p.ticker, fin, self.px)
                if spot is not None:
                    self._close(p, fin, p.value_ps(spot, end_d.toordinal()), "final_liquidation")
        final_eq = self.cash
        self._record_equity(fin, final_eq, self.equity_curve)
        self._record_equity(fin, final_eq, self.daily_equity_curve)
        return self.metrics()

    def metrics(self):
        stats = compute_equity_metrics(
            self.equity_curve,
            self.start,
            self.end,
            step_days=self.step_days,
            mark_frequency=self.mark_frequency,
            daily_equity_curve=self.daily_equity_curve if self.daily_equity_curve else None,
        )

        duration_days = stats["duration_days"]
        yrs = duration_days / 365.25
        wins = [p for p in self.closed if p.exit_val_ps > p.entry_cost_ps]

        # Comparators: uninvested cash (0%) and SPY buy-and-hold
        spy = get_closes("SPY", self.start, self.end)
        spy_items = sorted(spy.items())
        spy_ret = (spy_items[-1][1] / spy_items[0][1] - 1.0) if len(spy_items) > 1 else 0.0
        spy_cagr = (1.0 + spy_ret) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
        cash_cagr = 0.0

        # Coverage & validity gate (OP04)
        failed_steps = self.transport_errors + self.parse_fails + self.schema_fails
        total_steps = self.decision_steps_run
        if total_steps == 0:
            total_steps = self.valid_decisions + failed_steps
        coverage_ratio = (self.valid_decisions / total_steps) if total_steps > 0 else 0.0
        coverage_pct = round(100.0 * coverage_ratio, 1)

        valid = (total_steps > 0) and (self.valid_decisions > 0) and (coverage_ratio >= self.min_coverage)
        if total_steps == 0:
            invalid_reason = "no_decision_steps_recorded"
        elif self.valid_decisions == 0:
            invalid_reason = f"zero_valid_decisions (0/{total_steps})"
        elif coverage_ratio < self.min_coverage:
            invalid_reason = (
                f"coverage_below_threshold ({coverage_pct:.1f}% < {self.min_coverage*100:.1f}%)"
            )
        else:
            invalid_reason = None

        verdict = "VALID" if valid else "INVALID"
        cagr_raw = stats.get("cagr_raw", 0.0)

        return {
            "verdict": verdict,
            "qualifying": valid,
            "invalid_reason": invalid_reason,
            "final_equity": stats["final_equity"],
            "total_return_pct": stats["total_return_pct"],
            "cagr_pct": stats["cagr_pct"] if valid else None,
            "sharpe": stats["sharpe"] if valid else None,
            "max_drawdown_pct": stats["max_drawdown_pct"] if valid else None,
            "step_max_drawdown_pct": stats["step_max_drawdown_pct"] if valid else None,
            "sampling_frequency": stats["sampling_frequency"],
            "periods_per_year": stats["periods_per_year"],
            "duration_days": duration_days,
            "n_trades": len(self.closed),
            "hit_rate_pct": round(100 * len(wins) / len(self.closed), 1) if self.closed else 0.0,
            "decision_steps": total_steps,
            "valid_decisions": self.valid_decisions,
            "deliberate_abstentions": self.deliberate_abstentions,
            "failed_steps": failed_steps,
            "transport_errors": self.transport_errors,
            "parse_fails": self.parse_fails,
            "schema_fails": self.schema_fails,
            "coverage_pct": coverage_pct,
            "min_coverage_pct": round(self.min_coverage * 100.0, 1),
            "model_calls": self.calls,
            "cash_comparator_cagr_pct": round(cash_cagr * 100.0, 1),
            "cash_comparator_total_return_pct": 0.0,
            "spy_buyhold_cagr_pct": round(spy_cagr * 100.0, 1),
            "spy_buyhold_total_return_pct": round(spy_ret * 100.0, 1),
            "alpha_vs_spy_cagr_pct": round((cagr_raw - spy_cagr) * 100.0, 1) if valid else None,
            "alpha_vs_cash_cagr_pct": round((cagr_raw - cash_cagr) * 100.0, 1) if valid else None,
            "exit_reasons": {
                r: sum(1 for p in self.closed if p.exit_reason == r)
                for r in sorted(set(p.exit_reason for p in self.closed))
            },
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "local-model"))
    ap.add_argument("--start", default="2024-02-01")
    ap.add_argument("--end", default="2026-04-15")
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--mark-frequency", choices=["step", "daily"], default="step")
    ap.add_argument("--min-coverage", type=float, default=MIN_VALID_COVERAGE)
    ap.add_argument("--out", default="book_run")
    a = ap.parse_args()
    b = Book(
        a.base,
        a.model,
        a.start,
        a.end,
        a.step,
        mark_frequency=a.mark_frequency,
        min_coverage=a.min_coverage,
        trace_path=f"{a.out}_steps.jsonl",
    )
    print(f"running book {a.start}->{a.end} step {a.step}d via {a.model}", flush=True)
    m = b.run()
    with open(f"{a.out}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    with open(f"{a.out}_equity.json", "w") as f:
        json.dump(b.equity_curve, f)
    print(json.dumps(m, indent=2))
