"""Cached daily-close price data via yfinance (or pinned offline fixtures).

Cache files store inclusive [start, end] metadata (`_start`, `_end`) and handle
yfinance's end-exclusive download semantics by requesting `hi + 1 day`. Single-bar
and MultiIndex DataFrame payloads are normalized without scalar `.squeeze()` defects.
Runs can freeze an immutable in-memory `PriceSnapshot` so mid-run mutations to on-disk
cache files cannot cause mixed price consumption or provenance digest drift.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, timedelta
import hashlib
import json
import math
import os
import tempfile
from types import MappingProxyType

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional for offline/unit-test environments
    yf = None

CACHE_SCHEMA_VERSION = 2
CACHE_DIR = os.environ.get(
    "BYRON_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pxcache"),
)

_ACTIVE_SNAPSHOT: ContextVar["PriceSnapshot | None"] = ContextVar(
    "_ACTIVE_SNAPSHOT", default=None
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


def price_series_digest(series_by_ticker: dict[str, dict[str, float]]) -> str:
    """Return deterministic SHA-256[:16] over canonical consumed {TICKER: {iso_date: close}}."""
    h = hashlib.sha256()
    for tk in sorted(series_by_ticker):
        bars = series_by_ticker[tk]
        if not bars:
            continue
        clean = {
            str(d): round(float(v), 4)
            for d, v in sorted(bars.items())
            if not str(d).startswith("_")
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        }
        if not clean:
            continue
        h.update(tk.upper().encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


class PriceSnapshot:
    """Immutable in-memory snapshot of daily-close series with an exact content digest.

    Once hydrated, all lookups (`get_closes`, `path_from`) read strictly from the frozen
    in-memory bars — immune to any mid-run mutation of on-disk cache files.
    """

    def __init__(self, series_by_ticker: dict[str, dict[str, float]] | None = None):
        frozen: dict[str, MappingProxyType] = {}
        if series_by_ticker:
            for tk, bars in series_by_ticker.items():
                if not bars:
                    continue
                clean = {
                    str(d): round(float(v), 4)
                    for d, v in sorted(bars.items())
                    if not str(d).startswith("_")
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(float(v))
                }
                if clean:
                    frozen[tk.upper()] = MappingProxyType(clean)
        self._series: MappingProxyType = MappingProxyType(frozen)
        self._digest: str = price_series_digest(frozen)

    def __len__(self) -> int:
        return len(self._series)

    def __contains__(self, ticker: str) -> bool:
        return str(ticker).upper() in self._series

    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self._series))

    def digest(self) -> str:
        return self._digest

    def get_closes(self, ticker: str, start: str, end: str) -> dict[str, float]:
        bars = self._series.get(ticker.upper())
        if not bars:
            return {}
        return {k: v for k, v in bars.items() if start <= k <= end}

    def path_from(self, ticker: str, entry_date: str, days_forward: int = 220):
        d0 = date.fromisoformat(entry_date)
        start = (d0 - timedelta(days=45)).isoformat()
        end = (d0 + timedelta(days=days_forward)).isoformat()
        closes = self.get_closes(ticker, start, end)
        items = sorted(closes.items())
        pre = [(d, c) for d, c in items if d < entry_date]
        fwd = [(d, c) for d, c in items if d >= entry_date]
        return pre, fwd

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {tk: dict(bars) for tk, bars in self._series.items()}

    def subset(self, tickers) -> "PriceSnapshot":
        wanted = {str(tk).upper() for tk in tickers}
        return PriceSnapshot({tk: dict(self._series[tk]) for tk in sorted(wanted) if tk in self._series})


class PriceSnapshotSession:
    """Single-run write-once price session frozen BEFORE setup selection (OP08).

    Pre-loads any currently cached on-disk ticker series once at session initialization
    so subsequent cache mutations on disk cannot desynchronize setup selection,
    hydration, prompt construction, scoring, simulation, or provenance digests.
    For uncached tickers (e.g. cold live runs), bars are fetched on demand and frozen
    write-once in memory (already-observed dates can never be overwritten).
    """

    def __init__(self, *, cache_dir: str | None = None, get_closes_fn=None):
        self._fn = get_closes_fn if get_closes_fn is not None else get_closes
        self._is_default_loader = (
            (self._fn is _DEFAULT_GET_CLOSES) and (get_closes is _DEFAULT_GET_CLOSES)
        )
        self._cache_dir = cache_dir
        self._series: dict[str, dict[str, float]] = {}
        self._pinned: set[str] = set()
        self._bounds: dict[str, tuple[str, str]] = {}
        self._consumed: set[str] = set()

        if self._is_default_loader:
            root = cache_dir if cache_dir is not None else _get_cache_dir()
            if os.path.isdir(root):
                for name in sorted(os.listdir(root)):
                    if not name.endswith(".json"):
                        continue
                    path = os.path.join(root, name)
                    if not os.path.isfile(path):
                        continue
                    tk = os.path.splitext(name)[0].upper()
                    try:
                        with open(path, encoding="utf-8") as f:
                            payload = json.load(f)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    bars = {
                        str(k): round(float(v), 4)
                        for k, v in payload.items()
                        if not str(k).startswith("_")
                        and isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and math.isfinite(float(v))
                    }
                    if not bars:
                        continue
                    self._series[tk] = bars
                    if payload.get("_fixture_sha256"):
                        self._pinned.add(tk)
                    lo = payload.get("_start")
                    hi = payload.get("_end")
                    if isinstance(lo, str) and isinstance(hi, str):
                        self._bounds[tk] = (lo, hi)

    def _is_covered(self, tk: str, start: str, end: str) -> bool:
        if tk not in self._series:
            return False
        if tk in self._pinned:
            return True
        bounds = self._bounds.get(tk)
        if bounds is None:
            return False
        return bounds[0] <= start and bounds[1] >= end

    def get_closes(self, ticker: str, start: str, end: str) -> dict[str, float]:
        tk = str(ticker).upper()
        self._consumed.add(tk)
        if not self._is_covered(tk, start, end):
            with use_price_snapshot(None):
                fetched = self._fn(tk, start, end)
            dest = self._series.setdefault(tk, {})
            if isinstance(fetched, dict):
                for d, v in fetched.items():
                    if (
                        not str(d).startswith("_")
                        and isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and math.isfinite(float(v))
                    ):
                        dest.setdefault(str(d), round(float(v), 4))
            if self._is_default_loader:
                try:
                    cp = (
                        os.path.join(self._cache_dir, f"{tk}.json")
                        if self._cache_dir is not None
                        else _cache_path(tk)
                    )
                    for d, v in _read_cached_bars_file(cp).items():
                        dest.setdefault(d, v)
                except Exception:
                    pass
            prev = self._bounds.get(tk)
            if prev is None:
                self._bounds[tk] = (start, end)
            else:
                self._bounds[tk] = (min(prev[0], start), max(prev[1], end))
        bars = self._series.get(tk)
        if not bars:
            return {}
        return {k: v for k, v in bars.items() if start <= k <= end}

    def path_from(self, ticker: str, entry_date: str, days_forward: int = 220):
        d0 = date.fromisoformat(entry_date)
        start = (d0 - timedelta(days=45)).isoformat()
        end = (d0 + timedelta(days=days_forward)).isoformat()
        closes = self.get_closes(ticker, start, end)
        items = sorted(closes.items())
        pre = [(d, c) for d, c in items if d < entry_date]
        fwd = [(d, c) for d, c in items if d >= entry_date]
        return pre, fwd

    def hydrate_windows(self, ticker_windows: dict[str, tuple[str, str]]) -> None:
        for raw_tk in sorted(ticker_windows):
            lo, hi = ticker_windows[raw_tk]
            try:
                self.get_closes(raw_tk, lo, hi)
            except Exception:
                continue

    def freeze(self, tickers=None) -> PriceSnapshot:
        if tickers is not None:
            wanted = {str(tk).upper() for tk in tickers}
        elif self._consumed:
            wanted = set(self._consumed)
        else:
            wanted = set(self._series.keys())
        return PriceSnapshot(
            {tk: dict(self._series[tk]) for tk in sorted(wanted) if self._series.get(tk)}
        )


@contextmanager
def use_price_snapshot(snapshot: "PriceSnapshot | None"):
    """Bind an immutable PriceSnapshot for all data lookups in this execution context."""
    token = _ACTIVE_SNAPSHOT.set(snapshot)
    try:
        yield snapshot
    finally:
        _ACTIVE_SNAPSHOT.reset(token)


def active_price_snapshot() -> "PriceSnapshot | None":
    return _ACTIVE_SNAPSHOT.get()


def _read_cached_bars_file(path: str) -> dict[str, float]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return {
        str(k): round(float(v), 4)
        for k, v in payload.items()
        if not str(k).startswith("_")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    }


def get_closes(
    ticker: str,
    start: str,
    end: str,
    *,
    price_snapshot: "PriceSnapshot | None" = None,
) -> dict[str, float]:
    """Return {isodate: close} for inclusive [start, end], split/div-adjusted.

    When a `PriceSnapshot` is provided or active via `use_price_snapshot()`, serves
    strictly from the frozen in-memory snapshot without touching on-disk cache files.
    """
    snap = price_snapshot if price_snapshot is not None else _ACTIVE_SNAPSHOT.get()
    if snap is not None:
        return snap.get_closes(ticker, start, end)

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


_DEFAULT_GET_CLOSES = get_closes


def path_from(
    ticker: str,
    entry_date: str,
    days_forward: int = 220,
    *,
    price_snapshot: "PriceSnapshot | None" = None,
):
    """Ordered list of (isodate, close) starting at/after entry_date, plus the
    trailing window before entry_date (for realized-vol / IV estimation)."""
    snap = price_snapshot if price_snapshot is not None else _ACTIVE_SNAPSHOT.get()
    if snap is not None:
        return snap.path_from(ticker, entry_date, days_forward=days_forward)
    d0 = date.fromisoformat(entry_date)
    start = (d0 - timedelta(days=45)).isoformat()
    end = (d0 + timedelta(days=days_forward)).isoformat()
    closes = get_closes(ticker, start, end)
    items = sorted(closes.items())
    pre = [(d, c) for d, c in items if d < entry_date]
    fwd = [(d, c) for d, c in items if d >= entry_date]
    return pre, fwd


def load_cache_snapshot(cache_dir: str | None = None) -> PriceSnapshot:
    """Load all cached price series from `cache_dir` into an immutable PriceSnapshot."""
    root = cache_dir if cache_dir is not None else _get_cache_dir()
    series: dict[str, dict[str, float]] = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            tk = os.path.splitext(name)[0].upper()
            try:
                bars = _read_cached_bars_file(path)
            except Exception:
                continue
            if bars:
                series[tk] = bars
    return PriceSnapshot(series)


def hydrate_price_snapshot(
    ticker_windows: dict[str, tuple[str, str]],
    *,
    get_closes_fn=None,
) -> PriceSnapshot:
    """Hydrate requested ticker windows ONCE and freeze them into an immutable PriceSnapshot.

    When using the default disk/yfinance loader, all cached bars for each hydrated ticker
    are captured once so subsequent lookups (even at wider DTEs) never re-read mutable
    on-disk cache files.
    """
    fn = get_closes_fn if get_closes_fn is not None else get_closes
    is_default_loader = (fn is _DEFAULT_GET_CLOSES) and (get_closes is _DEFAULT_GET_CLOSES)
    series: dict[str, dict[str, float]] = {}

    with use_price_snapshot(None):
        for raw_tk in sorted(ticker_windows):
            lo, hi = ticker_windows[raw_tk]
            tk = str(raw_tk).upper()
            try:
                fetched = fn(tk, lo, hi)
            except Exception:
                continue
            merged: dict[str, float] = {}
            if is_default_loader:
                try:
                    merged.update(_read_cached_bars_file(_cache_path(tk)))
                except Exception:
                    pass
            if isinstance(fetched, dict):
                merged.update(fetched)
            if merged:
                series[tk] = merged
    return PriceSnapshot(series)


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
    """Return a deterministic SHA-256 prefix over canonical cached price series in cache_dir."""
    return load_cache_snapshot(cache_dir).digest()


if __name__ == "__main__":
    pre, fwd = path_from("NVDA", "2024-06-10", 40)
    print(f"pre-window bars: {len(pre)}  forward bars: {len(fwd)}")
    print("entry:", fwd[0], " ~40d later:", fwd[-1])
