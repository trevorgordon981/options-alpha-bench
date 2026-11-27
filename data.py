"""Cached daily-close price data via yfinance (or pinned offline fixtures).

Cache files store inclusive [start, end] metadata (`_start`, `_end`) and handle
yfinance's end-exclusive download semantics by requesting `hi + 1 day`. Single-bar
and MultiIndex DataFrame payloads are normalized without scalar `.squeeze()` defects.
"""
from datetime import date, timedelta
import hashlib
import json
import math
import os
import tempfile

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional for offline/unit-test environments
    yf = None

CACHE_SCHEMA_VERSION = 2
CACHE_DIR = os.environ.get(
    "BYRON_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pxcache"),
)


def _get_cache_dir() -> str:
    path = os.environ.get("BYRON_CACHE") or CACHE_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(ticker: str) -> str:
    return os.path.join(_get_cache_dir(), f"{ticker.upper()}.json")


def _atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".pxcache_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _format_iso_date(idx_val) -> str:
    if hasattr(idx_val, "strftime"):
        return idx_val.strftime("%Y-%m-%d")
    return str(idx_val)[:10]


def _extract_close_series(df, ticker: str | None = None) -> dict[str, float]:
    """Normalize a yfinance-like Close payload (1-bar, multi-bar, or MultiIndex) to {iso: close}."""
    if df is None:
        return {}
    try:
        if len(df) == 0:
            return {}
    except TypeError:
        return {}

    close_obj = df["Close"] if hasattr(df, "__getitem__") else df

    # Handle 2D DataFrame (e.g. yfinance MultiIndex columns where df["Close"] is a DataFrame)
    if hasattr(close_obj, "columns") and hasattr(close_obj, "iloc"):
        cols = list(close_obj.columns)
        if not cols:
            return {}
        if ticker and ticker.upper() in cols:
            series = close_obj[ticker.upper()]
        else:
            series = close_obj.iloc[:, 0]
    else:
        series = close_obj

    if hasattr(series, "items"):
        pairs = list(series.items())
    elif hasattr(series, "index") and hasattr(series, "values"):
        pairs = list(zip(series.index, series.values))
    else:
        raise TypeError(f"Unsupported Close series type: {type(series).__name__}")

    out = {}
    for d_val, px in pairs:
        if px is None:
            continue
        val = float(px)
        if not math.isfinite(val):
            continue
        out[_format_iso_date(d_val)] = round(val, 4)
    return out


def get_closes(ticker: str, start: str, end: str) -> dict[str, float]:
    """Return {isodate: close} for inclusive [start, end], split/div-adjusted.

    Cached per ticker over a wide window so overlapping requests reuse one download.
    """
    ticker = ticker.upper()
    cp = _cache_path(ticker)
    cache = {}
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            cache = json.load(f)

    is_pinned_fixture = bool(cache.get("_fixture_sha256"))
    need = (
        not cache
        or (
            not is_pinned_fixture
            and (start < cache.get("_start", "9999") or end > cache.get("_end", "0000"))
        )
    )
    if need:
        if yf is None:
            raise RuntimeError(
                "yfinance is not installed and requested dates are not in cache; "
                "install yfinance or populate BYRON_CACHE from fixtures."
            )
        lo = min(start, cache.get("_start", start))
        hi = max(end, cache.get("_end", end))
        # yfinance `end` is exclusive; request hi + 1 calendar day so `hi` is inclusive.
        end_exclusive = (date.fromisoformat(hi) + timedelta(days=1)).isoformat()
        df = yf.download(
            ticker,
            start=lo,
            end=end_exclusive,
            progress=False,
            auto_adjust=True,
        )
        data = _extract_close_series(df, ticker=ticker)
        # Preserve any existing bars within [lo, hi] if provider omits partial overlap
        merged = {k: v for k, v in cache.items() if not k.startswith("_")}
        merged.update(data)
        cache = {
            "_schema_version": CACHE_SCHEMA_VERSION,
            "_source": "yfinance",
            "_auto_adjust": True,
            "_start": lo,
            "_end": hi,
            **merged,
        }
        _atomic_write_json(cp, cache)
    return {k: v for k, v in cache.items() if not k.startswith("_") and start <= k <= end}


def path_from(ticker: str, entry_date: str, days_forward: int = 220):
    """Ordered list of (isodate, close) starting at/after entry_date, plus the
    trailing window before entry_date (for realized-vol / IV estimation)."""
    d0 = date.fromisoformat(entry_date)
    start = (d0 - timedelta(days=45)).isoformat()
    end = (d0 + timedelta(days=days_forward)).isoformat()
    closes = get_closes(ticker, start, end)
    items = sorted(closes.items())
    pre = [(d, c) for d, c in items if d < entry_date]
    fwd = [(d, c) for d, c in items if d >= entry_date]
    return pre, fwd


def load_fixture_into_cache(fixture_path: str, cache_dir: str) -> str:
    """Populate a cache directory from a pinned JSON fixture and return its SHA-256."""
    with open(fixture_path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    tickers = payload.get("tickers", {})
    os.makedirs(cache_dir, exist_ok=True)
    for tk, closes in tickers.items():
        if not closes:
            continue
        dates = sorted(closes.keys())
        entry = {
            "_schema_version": CACHE_SCHEMA_VERSION,
            "_source": payload.get("source", "pinned_fixture"),
            "_fixture_sha256": digest,
            "_auto_adjust": True,
            "_start": payload.get("start", dates[0]),
            "_end": payload.get("end", dates[-1]),
            **{k: round(float(v), 4) for k, v in closes.items()},
        }
        _atomic_write_json(os.path.join(cache_dir, f"{tk.upper()}.json"), entry)
    return digest


def cache_manifest_digest(cache_dir: str | None = None) -> str:
    """Return a deterministic SHA-256 prefix over all price cache files in cache_dir."""
    root = cache_dir if cache_dir is not None else _get_cache_dir()
    h = hashlib.sha256()
    if not os.path.isdir(root):
        return h.hexdigest()[:16]
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as f:
            h.update(f.read())
        h.update(b"\n")
    return h.hexdigest()[:16]


if __name__ == "__main__":
    pre, fwd = path_from("NVDA", "2024-06-10", 40)
    print(f"pre-window bars: {len(pre)}  forward bars: {len(fwd)}")
    print("entry:", fwd[0], " ~40d later:", fwd[-1])
