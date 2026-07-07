"""
Driver: run the disciplined RULES-EXPERT (expert_policy.expert_decision) through the
REAL book.py engine over the full window, capturing metrics + a full per-step trace
(state handed in -> expert decision + reasoning). Nothing about accounting, pricing,
stops, or equity is reimplemented -- we reuse Book._auto_exits / _mark_equity / _open
/ _close / metrics exactly (same pattern as book_opus_driver.py), replacing only the
model-I/O step with the python expert.

Outputs:
  book_expert_metrics.json   -- CAGR/Sharpe/DD/hit-rate/alpha vs SPY
  book_expert_equity.json    -- equity curve
  book_expert_trace.json     -- [{iso, state, decision, reasoning, applied}]
"""
import json
import os
from datetime import date, timedelta

from book import Book, MAX_CONCURRENT, _spot_on_or_before
from market import slate as market_slate, context as market_context
from scenarios import BASKET
from expert_policy import expert_decision, cluster_of
from data import get_closes

START, END, STEP = "2024-02-01", "2026-04-15", 7

# SPY trend filter window (trading days) for the confirmed-risk-off cash gate.
# Defensible, general trend rule (SPY below its N-day SMA = risk-off), NOT tuned to
# specific 2024-2026 crash dates. Env-tunable for iteration; default 100d (~5-month
# trend, a classic medium-term timing filter).
SPY_MA_DAYS = int(os.environ.get("EX_SPY_MA_DAYS", 100))


def spy_trend(iso):
    """SPY vs its N-day simple moving average as of `iso` (inclusive, no lookahead).
    Returns (spot, sma, below_ma_bool, gap_frac). All from real cached closes."""
    d0 = date.fromisoformat(iso)
    start = (d0 - timedelta(days=int(SPY_MA_DAYS * 1.7) + 40)).isoformat()
    closes = get_closes("SPY", start, iso)
    px = [v for k, v in sorted(closes.items()) if k <= iso]
    if len(px) < SPY_MA_DAYS:
        return (px[-1] if px else 0.0, None, False, 0.0)  # too early -> no gate
    spot = px[-1]
    sma = sum(px[-SPY_MA_DAYS:]) / SPY_MA_DAYS
    gap = (spot - sma) / sma if sma else 0.0
    return (spot, sma, spot < sma, round(gap, 4))


def build_state(b, iso, equity, peak_equity):
    """Assemble the doctrine state the expert sees, from book.py's live positions."""
    d_ord = date.fromisoformat(iso).toordinal()
    positions = []
    pos_mktval = 0.0
    for p in b.positions:
        if p.closed:
            continue
        spot = _spot_on_or_before(p.ticker, iso, b.px)
        val_ps = p.value_ps(spot, d_ord)
        mktval = val_ps * 100 * p.contracts
        pos_mktval += mktval
        pnl = (val_ps - p.entry_cost_ps) / p.entry_cost_ps * 100 if p.entry_cost_ps else 0.0
        peak_pnl = (p.peak_val_ps - p.entry_cost_ps) / p.entry_cost_ps * 100 if p.entry_cost_ps else 0.0
        positions.append({
            "pid": p.pid, "ticker": p.ticker, "direction": p.direction,
            "contracts": p.contracts, "dte": p.expiry_ord - d_ord,
            "pnl_pct": round(pnl, 1), "peak_pnl_pct": round(max(peak_pnl, pnl), 1),
            "mktval_frac": round(mktval / equity, 4) if equity else 0.0,
            "cluster": cluster_of(p.ticker),
        })
    sl = market_slate(BASKET, iso, top=8)
    for c in sl:
        c["cluster"] = cluster_of(c["ticker"])
    spy_ctx = market_context("SPY", iso)
    spy_r20 = spy_ctx["ret_20d"] if spy_ctx else 0.0
    _spot, _sma, spy_below_ma, spy_ma_gap = spy_trend(iso)
    return {
        "iso": iso, "equity": round(equity, 2), "cash": round(b.cash, 2),
        "peak_equity": round(peak_equity, 2),
        "drawdown": round((peak_equity - equity) / peak_equity, 4) if peak_equity > 0 else 0.0,
        "deployment_frac": round(pos_mktval / equity, 4) if equity else 0.0,
        "n_open": len(positions), "max_concurrent": MAX_CONCURRENT,
        "positions": positions, "spy_r20": round(spy_r20, 4), "slate": sl,
        "spy_below_ma": spy_below_ma, "spy_ma_gap": spy_ma_gap,
        "spy_ma_days": SPY_MA_DAYS,
    }


def main():
    b = Book(base="", model="rules_expert", start=START, end=END, step_days=STEP, verbose=False)
    d = date.fromisoformat(START)
    end_d = date.fromisoformat(END)
    prev_iso = START
    peak_equity = 0.0
    trace = []

    while d <= end_d:
        iso = d.isoformat()
        b._auto_exits(prev_iso, iso)
        equity = b._mark_equity(iso)
        b.equity_curve.append((iso, round(equity, 2)))
        peak_equity = max(peak_equity, equity)

        state = build_state(b, iso, equity, peak_equity)
        decision, reasoning = expert_decision(state)

        # ---- apply closes ----
        closed_report = []
        for pid in decision.get("close", []) or []:
            for p in b.positions:
                if p.pid == pid and not p.closed:
                    spot = _spot_on_or_before(p.ticker, iso, b.px)
                    val = p.value_ps(spot, d.toordinal())
                    pnl = (val - p.entry_cost_ps) / p.entry_cost_ps * 100 if p.entry_cost_ps else 0
                    b._close(p, iso, val, "model_close")
                    closed_report.append(f"{pid} {p.ticker} P&L {pnl:+.0f}%")

        # ---- apply opens ----
        nopen = len([p for p in b.positions if not p.closed])
        opened_report = []
        for spec in (decision.get("open", []) or []):
            if nopen >= MAX_CONCURRENT:
                opened_report.append(f"SKIP {spec.get('ticker')} (max)")
                continue
            p = b._open(iso, spec, equity)
            if p:
                cost = p.entry_cost_ps * 100 * p.contracts
                opened_report.append(f"{p.pid} {p.ticker} {p.direction} x{p.contracts} "
                                     f"cost=${cost:,.0f} ({cost/equity*100:.0f}%)")
                nopen += 1
            else:
                opened_report.append(f"REJECT {spec.get('ticker')} {spec.get('direction')}")

        trace.append({
            "step": len(trace) + 1, "iso": iso, "state": state,
            "decision": decision, "reasoning": reasoning,
            "applied": {"closed": closed_report, "opened": opened_report,
                        "cash_after": round(b.cash, 2)},
        })
        prev_iso = iso
        d += timedelta(days=STEP)

    # ---- final liquidation (same as Book.run) ----
    fin = end_d.isoformat()
    b._auto_exits(prev_iso, fin)
    for p in b.positions:
        if not p.closed:
            spot = _spot_on_or_before(p.ticker, fin, b.px)
            b._close(p, fin, p.value_ps(spot, end_d.toordinal()), "final_liquidation")
    b.equity_curve.append((fin, round(b.cash, 2)))

    m = b.metrics()
    m["model"] = "rules_expert"
    with open("book_expert_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    with open("book_expert_equity.json", "w") as f:
        json.dump(b.equity_curve, f)
    with open("book_expert_trace.json", "w") as f:
        json.dump(trace, f, indent=2)

    print(json.dumps(m, indent=2))
    n_open_steps = sum(1 for t in trace if t["applied"]["opened"])
    n_close_steps = sum(1 for t in trace if t["applied"]["closed"])
    print(f"\nsteps={len(trace)}  open-steps={n_open_steps}  close-steps={n_close_steps}")


if __name__ == "__main__":
    main()
