"""Cached daily-close price data via yfinance. Cache to disk so re-runs are free
and we don't get rate-limited mid-sweep."""
import json
import os

import yfinance as yf

CACHE_DIR = os.environ.get("BYRON_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pxcache"))
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"{ticker.upper()}.json")


def get_closes(ticker, start, end):
    """Return {isodate: close} for [start, end], split/div-adjusted. Cached per ticker
    over a wide window so overlapping requests reuse one download."""
    ticker = ticker.upper()
    cp = _cache_path(ticker)
    cache = {}
    if os.path.exists(cp):
        with open(cp) as f:
            cache = json.load(f)
    need = not cache or start < cache.get("_start", "9999") or end > cache.get("_end", "0000")
    if need:
        lo = min(start, cache.get("_start", start))
        hi = max(end, cache.get("_end", end))
        df = yf.download(ticker, start=lo, end=hi, progress=False, auto_adjust=True)
        c = df["Close"].squeeze()
        data = {d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in c.items()}
        cache = {"_start": lo, "_end": hi, **data}
        with open(cp, "w") as f:
            json.dump(cache, f)
    return {k: v for k, v in cache.items() if not k.startswith("_") and start <= k <= end}


def path_from(ticker, entry_date, days_forward=220):
    """Ordered list of (isodate, close) starting at/after entry_date, plus the
    trailing window before entry_date (for realized-vol / IV estimation)."""
    from datetime import date, timedelta
    d0 = date.fromisoformat(entry_date)
    start = (d0 - timedelta(days=45)).isoformat()
    end = (d0 + timedelta(days=days_forward)).isoformat()
    closes = get_closes(ticker, start, end)
    items = sorted(closes.items())
    pre = [(d, c) for d, c in items if d < entry_date]
    fwd = [(d, c) for d, c in items if d >= entry_date]
    return pre, fwd


if __name__ == "__main__":
    pre, fwd = path_from("NVDA", "2024-06-10", 40)
    print(f"pre-window bars: {len(pre)}  forward bars: {len(fwd)}")
    print("entry:", fwd[0], " ~40d later:", fwd[-1])
