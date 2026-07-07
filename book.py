"""
Autonomous book backtest — the realism track of BotTrade Bench.

The model RUNS a simulated options book over a real window (default 2024-2026):
  - weekly decision points; between them, exit discipline runs continuously
    (fixed -30% premium stop + 3-DTE time-stop close positions automatically)
  - at each decision point the model sees current equity, cash, open positions
    (with live P&L) and a real market slate, and returns JSON: which positions to
    close + which new trades to open (ticker, direction, DTE, delta, size%)
  - every option is BS-marked on the REAL underlying close each day
Scored on the equity curve: CAGR, Sharpe, max drawdown, hit rate, vs SPY buy-hold.

This measures what a rubric never can: would a book actually run by THIS model have
made money, and how much of the P&L is the DTE/exit discipline vs the model's picks.
"""
import json
import math
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta

from data import get_closes
from pricing import bs_price, estimate_iv, strike_for_delta
from scenarios import BASKET
from market import slate

START_EQUITY = 100_000.0
MAX_CONCURRENT = 8
MAX_SIZE_PCT = 0.25       # cap any single new position at 25% of equity
PREMIUM_STOP = 0.30
TIME_STOP_DTE = 3


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


def _spot_on_or_before(ticker, iso, cache):
    """Most recent close at/before iso (handles weekends/holidays)."""
    if ticker not in cache:
        d0 = date.fromisoformat(iso)
        cache[ticker] = get_closes(ticker, (d0 - timedelta(days=400)).isoformat(),
                                   (d0 + timedelta(days=900)).isoformat())
    items = [(k, v) for k, v in cache[ticker].items() if k <= iso]
    return max(items)[1] if items else None


def _daily_between(ticker, a_iso, b_iso, cache):
    if ticker not in cache:
        _spot_on_or_before(ticker, b_iso, cache)
    return sorted((k, v) for k, v in cache[ticker].items() if a_iso < k <= b_iso)


class Book:
    def __init__(self, base, model, start="2024-02-01", end="2026-04-15",
                 step_days=7, verbose=True):
        self.base, self.model = base, model
        self.start, self.end, self.step_days = start, end, step_days
        self.cash = START_EQUITY
        self.positions = []
        self.closed = []
        self.equity_curve = []      # (date, equity)
        self.pcount = 0
        self.px = {}                # ticker -> {date: close}
        self.verbose = verbose
        self.calls = 0
        self.parse_fails = 0

    # ---- model I/O ----
    def _call(self, prompt, max_tokens=1200):
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0.3}).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=200) as r:
            self.calls += 1
            return json.loads(r.read())["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(text):
        m = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text)
        for c in reversed(m):
            try:
                j = json.loads(c)
                if "open" in j or "close" in j:
                    return j
            except Exception:
                continue
        return None

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
        tk = spec.get("ticker", "").upper()
        direction = spec.get("direction", "")
        if tk not in BASKET or direction not in (
                "long_call", "long_put", "call_debit_spread", "put_debit_spread"):
            return None
        spot = _spot_on_or_before(tk, iso, self.px)
        if not spot:
            return None
        dte = int(spec.get("entry_dte", 45))
        delta = max(0.05, min(0.95, float(spec.get("long_delta", 0.6))))
        size_pct = max(0.0, min(MAX_SIZE_PCT, float(spec.get("size_pct", 0.1))))
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
        sl = slate(BASKET, iso, top=8)
        slate_block = "\n".join(
            f'  {c["ticker"]}: ${c["spot"]}, 5d {c["ret_5d"]*100:+.0f}%, 20d {c["ret_20d"]*100:+.0f}%, '
            f'{c["pct_below_high"]*100:.0f}% below high, rvol {c["rvol"]:.0%}' for c in sl)
        return f"""You are Walter, running a defined-risk options book. Date: {iso}.
Equity: ${equity:,.0f} | Cash: ${self.cash:,.0f} | Open positions: {len(opens)}/{MAX_CONCURRENT}

OPEN POSITIONS:
{pos_block}

MARKET SLATE (real data as of today):
{slate_block}

Rules: small defined-risk debits; fixed -30% stop + 3-DTE time-stop run automatically.
Decide this week. Output ONE strict JSON object on its own line:
{{"close": ["p1", ...], "open": [{{"ticker": "NVDA", "direction": "long_call|long_put|call_debit_spread|put_debit_spread", "entry_dte": <int>, "long_delta": <0.05-0.95>, "size_pct": <0.02-0.25>}}]}}
Open [] to sit in cash. Never exceed {MAX_CONCURRENT} open positions."""

    # ---- run ----
    def run(self):
        d = date.fromisoformat(self.start)
        end_d = date.fromisoformat(self.end)
        prev_iso = self.start
        while d <= end_d:
            iso = d.isoformat()
            self._auto_exits(prev_iso, iso)
            equity = self._mark_equity(iso)
            self.equity_curve.append((iso, round(equity, 2)))
            try:
                out = self._call(self._prompt(iso, equity))
                dec = self._parse(out)
            except Exception as e:
                dec = None
                if self.verbose:
                    print(f"  {iso} call err: {e}", flush=True)
            if dec is None:
                self.parse_fails += 1
            else:
                for pid in dec.get("close", []) or []:
                    for p in self.positions:
                        if p.pid == pid and not p.closed:
                            spot = _spot_on_or_before(p.ticker, iso, self.px)
                            self._close(p, iso, p.value_ps(spot, d.toordinal()), "model_close")
                nopen = len([p for p in self.positions if not p.closed])
                for spec in (dec.get("open", []) or []):
                    if nopen >= MAX_CONCURRENT:
                        break
                    if self._open(iso, spec, equity):
                        nopen += 1
            if self.verbose:
                print(f"  {iso}  eq=${equity:,.0f}  open={len([p for p in self.positions if not p.closed])} "
                      f"closed={len(self.closed)}", flush=True)
            prev_iso = iso
            d += timedelta(days=self.step_days)
        # liquidate at final marks
        fin = end_d.isoformat()
        self._auto_exits(prev_iso, fin)
        for p in self.positions:
            if not p.closed:
                spot = _spot_on_or_before(p.ticker, fin, self.px)
                self._close(p, fin, p.value_ps(spot, end_d.toordinal()), "final_liquidation")
        final_eq = self.cash
        self.equity_curve.append((fin, round(final_eq, 2)))
        return self.metrics()

    def metrics(self):
        eq = [e for _, e in self.equity_curve]
        rets = [(eq[i] / eq[i - 1] - 1) for i in range(1, len(eq)) if eq[i - 1] > 0]
        if len(rets) >= 2:
            m = sum(rets) / len(rets)
            sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
            sharpe = (m / sd * math.sqrt(52)) if sd > 0 else 0.0
        else:
            sharpe = 0.0
        peak, mdd = eq[0], 0.0
        for e in eq:
            peak = max(peak, e)
            mdd = max(mdd, (peak - e) / peak)
        yrs = (date.fromisoformat(self.end) - date.fromisoformat(self.start)).days / 365.25
        cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if eq[0] > 0 and yrs > 0 else 0.0
        wins = [p for p in self.closed if p.exit_val_ps > p.entry_cost_ps]
        # SPY benchmark
        spy = get_closes("SPY", self.start, self.end)
        spy_items = sorted(spy.items())
        spy_ret = (spy_items[-1][1] / spy_items[0][1] - 1) if len(spy_items) > 1 else 0.0
        spy_cagr = (1 + spy_ret) ** (1 / yrs) - 1 if yrs > 0 else 0.0
        return {
            "final_equity": round(eq[-1], 2),
            "total_return_pct": round((eq[-1] / eq[0] - 1) * 100, 1),
            "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(mdd * 100, 1),
            "n_trades": len(self.closed),
            "hit_rate_pct": round(100 * len(wins) / len(self.closed), 1) if self.closed else 0.0,
            "decision_steps": len(self.equity_curve) - 1,
            "model_calls": self.calls, "parse_fails": self.parse_fails,
            "spy_buyhold_cagr_pct": round(spy_cagr * 100, 1),
            "alpha_vs_spy_cagr_pct": round((cagr - spy_cagr) * 100, 1),
            "exit_reasons": {r: sum(1 for p in self.closed if p.exit_reason == r)
                             for r in set(p.exit_reason for p in self.closed)},
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default="/Users/trevorgordon/models/gemma-4-31b-derefused-a03nt-8bit")
    ap.add_argument("--start", default="2024-02-01")
    ap.add_argument("--end", default="2026-04-15")
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--out", default="book_walter")
    a = ap.parse_args()
    b = Book(a.base, a.model, a.start, a.end, a.step)
    print(f"running book {a.start}->{a.end} step {a.step}d via {a.model}", flush=True)
    m = b.run()
    with open(f"{a.out}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    with open(f"{a.out}_equity.json", "w") as f:
        json.dump(b.equity_curve, f)
    print(json.dumps(m, indent=2))
