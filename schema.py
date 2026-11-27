"""Strict finite bounded typed schema validation and durable JSONL persistence."""
import json
import math
import os

VALID_DIRECTIONS = frozenset({
    "long_call",
    "long_put",
    "call_debit_spread",
    "put_debit_spread",
})
VALID_CONVICTIONS = frozenset({"low", "med", "high"})

MIN_ENTRY_DTE = 1
MAX_ENTRY_DTE = 1095
MIN_DELTA = 0.05
MAX_DELTA = 0.95
MIN_STOP_PCT = 0.01
MAX_STOP_PCT = 1.0
MAX_BOOK_SIZE_PCT = 0.25


def _is_strict_int(val) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_number(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def extract_json_objects(text: str) -> list[dict]:
    """Extract top-level JSON dict objects from arbitrary model text using JSONDecoder."""
    if not isinstance(text, str) or not text:
        return []
    decoder = json.JSONDecoder()
    objs = []
    idx = 0
    n = len(text)
    while idx < n:
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
            if isinstance(obj, dict):
                objs.append(obj)
                idx = end
                continue
        except json.JSONDecodeError:
            pass
        idx = brace + 1
    return objs


def validate_construction_trade(obj) -> tuple[dict | None, str | None]:
    """Strictly validate a single construction-alpha trade JSON object.

    Malformed field types, booleans, non-finite numbers, or out-of-range values are
    rejected explicitly rather than silently coerced.
    """
    if not isinstance(obj, dict):
        return None, f"expected JSON object, got {type(obj).__name__}"

    direction = obj.get("direction")
    if not isinstance(direction, str) or direction not in VALID_DIRECTIONS:
        return None, f"invalid direction: {direction!r}"

    if "entry_dte" not in obj:
        return None, "missing required field 'entry_dte'"
    entry_dte = obj["entry_dte"]
    if not _is_strict_int(entry_dte):
        return None, f"entry_dte must be an int, got {type(entry_dte).__name__} ({entry_dte!r})"
    if not (MIN_ENTRY_DTE <= entry_dte <= MAX_ENTRY_DTE):
        return None, f"entry_dte {entry_dte} out of bounds [{MIN_ENTRY_DTE}, {MAX_ENTRY_DTE}]"

    raw_delta = obj.get("long_delta", 0.60)
    if not _is_finite_number(raw_delta):
        return None, f"long_delta must be a finite number, got {raw_delta!r}"
    long_delta = float(raw_delta)
    if not (MIN_DELTA <= long_delta <= MAX_DELTA):
        return None, f"long_delta {long_delta} out of bounds [{MIN_DELTA}, {MAX_DELTA}]"

    raw_stop = obj.get("stop_pct", 0.30)
    if not _is_finite_number(raw_stop):
        return None, f"stop_pct must be a finite number, got {raw_stop!r}"
    stop_pct = float(raw_stop)
    if not (MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT):
        return None, f"stop_pct {stop_pct} out of bounds [{MIN_STOP_PCT}, {MAX_STOP_PCT}]"

    conviction = obj.get("conviction")
    if conviction is not None:
        if not isinstance(conviction, str) or conviction.lower() not in VALID_CONVICTIONS:
            return None, f"invalid conviction: {conviction!r}"
        conviction = conviction.lower()

    return {
        "direction": direction,
        "entry_dte": int(entry_dte),
        "long_delta": long_delta,
        "stop_pct": stop_pct,
        "conviction": conviction,
    }, None


def validate_book_open_spec(spec, *, valid_tickers=None) -> tuple[dict | None, str | None]:
    """Strictly validate one item in a book decision's 'open' list."""
    if not isinstance(spec, dict):
        return None, f"open item must be an object, got {type(spec).__name__}"

    ticker = spec.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        return None, f"ticker must be a non-empty string, got {ticker!r}"
    ticker_norm = ticker.strip().upper()
    if valid_tickers is not None and ticker_norm not in valid_tickers:
        return None, f"unknown ticker: {ticker_norm!r}"

    direction = spec.get("direction")
    if not isinstance(direction, str) or direction not in VALID_DIRECTIONS:
        return None, f"invalid direction: {direction!r}"

    if "entry_dte" not in spec:
        return None, "missing required field 'entry_dte'"
    entry_dte = spec["entry_dte"]
    if not _is_strict_int(entry_dte):
        return None, f"entry_dte must be an int, got {type(entry_dte).__name__} ({entry_dte!r})"
    if not (MIN_ENTRY_DTE <= entry_dte <= MAX_ENTRY_DTE):
        return None, f"entry_dte {entry_dte} out of bounds [{MIN_ENTRY_DTE}, {MAX_ENTRY_DTE}]"

    raw_delta = spec.get("long_delta", 0.60)
    if not _is_finite_number(raw_delta):
        return None, f"long_delta must be a finite number, got {raw_delta!r}"
    long_delta = float(raw_delta)
    if not (MIN_DELTA <= long_delta <= MAX_DELTA):
        return None, f"long_delta {long_delta} out of bounds [{MIN_DELTA}, {MAX_DELTA}]"

    raw_size = spec.get("size_pct", 0.10)
    if not _is_finite_number(raw_size):
        return None, f"size_pct must be a finite number, got {raw_size!r}"
    size_pct = float(raw_size)
    if not (0.0 < size_pct <= MAX_BOOK_SIZE_PCT):
        return None, f"size_pct {size_pct} out of bounds (0.0, {MAX_BOOK_SIZE_PCT}]"

    return {
        "ticker": ticker_norm,
        "direction": direction,
        "entry_dte": int(entry_dte),
        "long_delta": long_delta,
        "size_pct": size_pct,
    }, None


def validate_book_decision(
    obj,
    *,
    valid_tickers=None,
    open_position_ids=None,
) -> tuple[dict | None, str | None]:
    """Strictly validate a book decision JSON object {'close': [...], 'open': [...]}."""
    if not isinstance(obj, dict):
        return None, f"expected JSON object, got {type(obj).__name__}"
    if "open" not in obj and "close" not in obj:
        return None, "decision object must contain 'open' or 'close'"

    raw_close = obj.get("close", [])
    if not isinstance(raw_close, list):
        return None, f"'close' must be a list, got {type(raw_close).__name__}"
    closes = []
    seen_closes = set()
    open_id_set = set(open_position_ids) if open_position_ids is not None else None
    for idx, pid in enumerate(raw_close):
        if not isinstance(pid, str) or not pid.strip():
            return None, f"close[{idx}] must be a non-empty string, got {pid!r}"
        pid_norm = pid.strip()
        if pid_norm in seen_closes:
            return None, f"close[{idx}]: duplicate position id {pid_norm!r}"
        if open_id_set is not None and pid_norm not in open_id_set:
            return None, f"close[{idx}]: unknown or non-open position id {pid_norm!r}"
        seen_closes.add(pid_norm)
        closes.append(pid_norm)

    raw_open = obj.get("open", [])
    if not isinstance(raw_open, list):
        return None, f"'open' must be a list, got {type(raw_open).__name__}"
    opens = []
    for idx, spec in enumerate(raw_open):
        valid_spec, err = validate_book_open_spec(spec, valid_tickers=valid_tickers)
        if err is not None:
            return None, f"open[{idx}]: {err}"
        opens.append(valid_spec)

    return {"close": closes, "open": opens}, None


def append_jsonl_record(path: str, record: dict) -> None:
    """Append one JSON record to a JSONL file and fsync for crash-safe durability."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
