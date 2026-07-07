"""
MECHANICAL setup generator — removes hindsight bias.

We do NOT hand-pick winning trades. We walk a basket of liquid names on a fixed
cadence and apply one simple, public momentum rule to assign a direction:
  20-day return > +TREND  -> bullish (long_call)
  20-day return < -TREND  -> bearish (long_put)
  otherwise               -> no clear thesis, skip
Every emitted setup is thus determined by data available AT the entry date only.
The DTE A/B then varies ONLY the expiry, so any P&L difference is attributable to
the DTE choice, not to trade selection.
"""
from datetime import date, timedelta

from data import path_from

BASKET = ["NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AVGO", "AMZN",
          "SPY", "QQQ", "NFLX", "CRM", "MU", "COIN"]

TREND = 0.05          # 20d move threshold to call a direction
HORIZON_DAYS = 21     # thesis window (calendar) — the "intended trade window"


def _twenty_day_ret(pre, entry_spot):
    closes = [c for _, c in pre] + [entry_spot]
    if len(closes) < 21:
        return None
    return (closes[-1] - closes[-21]) / closes[-21]


def generate(start="2024-01-15", end="2026-04-15", cadence_days=14, basket=BASKET):
    """Yield dicts: {ticker, entry_date, direction, horizon_days, mom20}."""
    setups = []
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    dates = []
    while d <= end_d:
        dates.append(d.isoformat())
        d += timedelta(days=cadence_days)
    for tk in basket:
        for ed in dates:
            try:
                pre, fwd = path_from(tk, ed, days_forward=HORIZON_DAYS + 10)
            except Exception:
                continue
            if len(fwd) < 3 or len(pre) < 20:
                continue
            entry_spot = fwd[0][1]
            mom = _twenty_day_ret(pre, entry_spot)
            if mom is None:
                continue
            if mom > TREND:
                direction = "long_call"
            elif mom < -TREND:
                direction = "long_put"
            else:
                continue
            setups.append({
                "ticker": tk, "entry_date": fwd[0][0], "direction": direction,
                "horizon_days": HORIZON_DAYS, "mom20": round(mom, 4),
            })
    return setups


if __name__ == "__main__":
    s = generate()
    import collections
    print("total mechanical setups:", len(s))
    print("by direction:", dict(collections.Counter(x["direction"] for x in s)))
    print("by ticker:", dict(collections.Counter(x["ticker"] for x in s)))
    print("sample:", s[0], s[len(s)//2], s[-1])
