"""Real market-context snapshots, computed from actual price history at a given date.
Used to build the candidate slate the bot sees at each decision point — so the bot
decides on data that was genuinely available then, no lookahead."""
from datetime import date, timedelta

from data import get_closes


def context(ticker, asof, lookback_days=90):
    """Snapshot of what was knowable at `asof` (inclusive). Returns None if insufficient."""
    d0 = date.fromisoformat(asof)
    start = (d0 - timedelta(days=lookback_days)).isoformat()
    closes = get_closes(ticker, start, asof)
    items = sorted((k, v) for k, v in closes.items() if k <= asof)
    if len(items) < 25:
        return None
    px = [v for _, v in items]
    spot = px[-1]
    hi = max(px)
    r5 = (px[-1] - px[-6]) / px[-6] if len(px) > 6 else 0.0
    r20 = (px[-1] - px[-21]) / px[-21] if len(px) > 21 else 0.0
    # realized vol (20d)
    import math
    rets = [math.log(px[i] / px[i - 1]) for i in range(max(1, len(px) - 20), len(px)) if px[i - 1] > 0]
    if len(rets) >= 2:
        m = sum(rets) / len(rets)
        rv = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) * math.sqrt(252)
    else:
        rv = 0.3
    return {
        "ticker": ticker, "asof": items[-1][0], "spot": round(spot, 2),
        "ret_5d": round(r5, 4), "ret_20d": round(r20, 4),
        "pct_below_high": round((hi - spot) / hi, 4), "rvol": round(rv, 3),
    }


def slate(basket, asof, top=8):
    """Candidate slate at `asof`: contexts sorted by |20d momentum| (most actionable first)."""
    ctx = [c for c in (context(t, asof) for t in basket) if c]
    ctx.sort(key=lambda c: abs(c["ret_20d"]), reverse=True)
    return ctx[:top]


if __name__ == "__main__":
    from scenarios import BASKET
    s = slate(BASKET, "2024-06-10")
    for c in s:
        print(c)
