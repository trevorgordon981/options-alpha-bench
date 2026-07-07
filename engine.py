"""
Forward-walk P&L engine. Takes a structured Trade + a real underlying path and
returns realized P&L, walking day-by-day and honoring Trevor's exit discipline:
  - fixed -30% premium stop (his airtight stop)
  - 3-DTE time-stop
  - thesis-horizon exit
  - optional profit target
Options are repriced each day via Black-Scholes on the REAL close (see pricing.py).
"""
from dataclasses import dataclass, field
from datetime import date

from data import path_from
from pricing import bs_price, estimate_iv, strike_for_delta


@dataclass
class Trade:
    ticker: str
    entry_date: str            # ISO
    direction: str             # long_call | long_put | call_debit_spread | put_debit_spread
    entry_dte: int             # calendar days to expiry at entry
    horizon_days: int          # thesis window in CALENDAR days (exit if reached)
    long_delta: float = 0.60   # target delta of the long leg (abs)
    short_delta: float = 0.30  # target delta of the short leg (spreads only)
    premium_stop: float = 0.30 # exit if position down this fraction
    time_stop_dte: int = 3
    target_pct: float | None = None   # exit if up this fraction (None = ride to horizon)
    note: str = ""


@dataclass
class Result:
    trade: Trade
    ok: bool
    entry_spot: float = 0.0
    exit_spot: float = 0.0
    iv: float = 0.0
    entry_cost: float = 0.0
    exit_val: float = 0.0
    return_pct: float = 0.0         # P&L on premium deployed
    hold_trading_days: int = 0
    exit_dte: int = 0
    exit_reason: str = ""
    underlying_return_pct: float = 0.0
    err: str = ""
    legs: dict = field(default_factory=dict)


def _kind(direction):
    return "call" if "call" in direction else "put"


def _value(direction, spot, legs, t, iv):
    k = _kind(direction)
    if "spread" in direction:
        return bs_price(spot, legs["long_k"], t, iv, k) - bs_price(spot, legs["short_k"], t, iv, k)
    return bs_price(spot, legs["long_k"], t, iv, k)


def simulate(trade: Trade) -> Result:
    try:
        pre, fwd = path_from(trade.ticker, trade.entry_date, days_forward=trade.entry_dte + 30)
    except Exception as e:
        return Result(trade, False, err=f"data:{e}")
    if len(fwd) < 3:
        return Result(trade, False, err="no forward path")

    entry_iso, entry_spot = fwd[0]
    pre_closes = [c for _, c in pre] + [entry_spot]
    iv = estimate_iv(pre_closes, lookback=20)
    k = _kind(trade.direction)
    t0 = trade.entry_dte / 365.0

    legs = {"long_k": strike_for_delta(entry_spot, t0, iv, trade.long_delta, k)}
    if "spread" in trade.direction:
        legs["short_k"] = strike_for_delta(entry_spot, t0, iv, trade.short_delta, k)

    entry_cost = _value(trade.direction, entry_spot, legs, t0, iv)
    if entry_cost <= 0.01:
        return Result(trade, False, err="degenerate entry cost", entry_spot=entry_spot)

    expiry = date.fromisoformat(entry_iso).toordinal() + trade.entry_dte
    horizon_end = date.fromisoformat(entry_iso).toordinal() + trade.horizon_days

    exit_val, exit_reason, exit_spot, exit_dte, held = entry_cost, "horizon", entry_spot, trade.entry_dte, 0
    for i, (d_iso, spot) in enumerate(fwd[1:], start=1):
        d_ord = date.fromisoformat(d_iso).toordinal()
        dte = expiry - d_ord
        t = max(dte, 0) / 365.0
        val = _value(trade.direction, spot, legs, t, iv)
        pnl = (val - entry_cost) / entry_cost
        held, exit_spot, exit_dte = i, spot, dte
        # exits in priority order
        if dte <= trade.time_stop_dte:
            exit_val, exit_reason = val, "time_stop"; break
        if pnl <= -trade.premium_stop:
            exit_val, exit_reason = val, "stop"; break
        if trade.target_pct is not None and pnl >= trade.target_pct:
            exit_val, exit_reason = val, "target"; break
        if d_ord >= horizon_end:
            exit_val, exit_reason = val, "horizon"; break
    else:
        exit_val, exit_reason = val, "path_end"

    return Result(
        trade=trade, ok=True, entry_spot=entry_spot, exit_spot=exit_spot, iv=iv,
        entry_cost=entry_cost, exit_val=exit_val,
        return_pct=(exit_val - entry_cost) / entry_cost,
        hold_trading_days=held, exit_dte=exit_dte, exit_reason=exit_reason,
        underlying_return_pct=(exit_spot - entry_spot) / entry_spot,
        legs={kk: round(vv, 2) for kk, vv in legs.items()},
    )


if __name__ == "__main__":
    # One trade both ways: NVDA bullish June 2024, 3-week thesis.
    for dte, tag in [(35, "doctrine 35-DTE"), (168, "8x rule ~168-DTE")]:
        t = Trade("NVDA", "2024-06-10", "long_call", dte, horizon_days=21, long_delta=0.60)
        r = simulate(t)
        print(f"{tag:20} cost={r.entry_cost:6.2f}  exit={r.exit_val:6.2f}  "
              f"ret={r.return_pct*100:+6.1f}%  reason={r.exit_reason:9} "
              f"held={r.hold_trading_days}d  underlying={r.underlying_return_pct*100:+.1f}%")
