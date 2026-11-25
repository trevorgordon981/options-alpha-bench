"""
Byron Bench — point it at any served model, get a trading scorecard.

Three tracks (a model-picker, not a vibe check):
  1. construction_alpha : 524 mechanical setups; model builds one trade each; realized
                          return-on-premium + the DTE it actually chooses.
  2. book_backtest      : model runs a $100k options book over 2024-2026; CAGR, Sharpe,
                          max drawdown, hit rate, alpha vs SPY buy-hold.
  3. doctrine (optional): gordon-gauntlet rubric score, attached if provided.

Usage:
  bench.py run --name walter_v1 --model <path> --base http://127.0.0.1:8082
  bench.py compare walter_v1 m3_prod ...
Cached track outputs are reused so re-runs are cheap.
"""
import argparse
import hashlib
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_META_KEY = "_cache_meta"


class CacheIntegrityError(ValueError):
    """Raised when a cached track artifact does not match the requested identity."""


def _load(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _write_json(path, payload, indent=2):
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def _cache_identity(track, model, base, **params):
    identity = {
        "track": track,
        "model": model,
        "base": base,
    }
    for key, value in sorted(params.items()):
        if value is not None:
            identity[key] = value
    return identity


def _cache_fingerprint(identity):
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def _cache_tag(prefix, name, identity):
    return f"{prefix}_{name}_{_cache_fingerprint(identity)}"


def _validate_cache_entry(data, expected_identity, path):
    if not isinstance(data, dict):
        raise CacheIntegrityError(
            f"Cache file {path} has invalid payload type {type(data).__name__}; expected dict"
        )
    meta = data.get(CACHE_META_KEY)
    if not isinstance(meta, dict):
        raise CacheIntegrityError(
            f"Cache file {path} is missing required '{CACHE_META_KEY}' model identity metadata"
        )
    mismatches = []
    for key in sorted(set(expected_identity) | set(meta)):
        if meta.get(key) != expected_identity.get(key):
            mismatches.append(
                f"{key}: cached={meta.get(key)!r} != requested={expected_identity.get(key)!r}"
            )
    if mismatches:
        raise CacheIntegrityError(
            f"Cache identity mismatch in {path}: " + "; ".join(mismatches)
        )
    return {k: v for k, v in data.items() if k != CACHE_META_KEY}


def _load_cached(path, expected_identity):
    data = _load(path)
    if data is None:
        return None
    return _validate_cache_entry(data, expected_identity, path)


def _save_cached(path, payload, identity):
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict cache payload for {path}, got {type(payload).__name__}")
    entry = dict(payload)
    entry[CACHE_META_KEY] = dict(identity)
    _write_json(path, entry, indent=2)


def run(name, base, model, start, end, step, skip_construction=False, skip_book=False,
        doctrine_score=None, cache_dir=None):
    out_dir = HERE if cache_dir is None else cache_dir
    card = {"name": name, "model": model, "base": base}

    # --- track 1: construction alpha (reuse cache only on exact model/config match) ---
    c_identity = _cache_identity("construction_alpha", model, base)
    ctag = _cache_tag("model_alpha", name, c_identity)
    csum_path = os.path.join(out_dir, f"{ctag}_summary.json")
    csum = _load_cached(csum_path, c_identity)
    if csum is None and not skip_construction:
        from run_model_alpha import score as construction_score, summarize as construction_summ
        from scenarios import generate

        setups = generate()
        rows = construction_score(setups, base, model)
        csum = construction_summ(rows)
        _write_json(os.path.join(out_dir, f"{ctag}_rows.json"), rows, indent=2)
        _save_cached(csum_path, csum, c_identity)
    card["construction_alpha"] = csum

    # --- track 2: book backtest (reuse cache only on exact model/config match) ---
    b_identity = _cache_identity("book_backtest", model, base, start=start, end=end, step=step)
    btag = _cache_tag("book", name, b_identity)
    bsum_path = os.path.join(out_dir, f"{btag}_metrics.json")
    bsum = _load_cached(bsum_path, b_identity)
    if bsum is None and not skip_book:
        from book import Book

        b = Book(base, model, start, end, step)
        bsum = b.run()
        _save_cached(bsum_path, bsum, b_identity)
        _write_json(os.path.join(out_dir, f"{btag}_equity.json"), b.equity_curve, indent=None)
    card["book_backtest"] = bsum

    # --- track 3: doctrine rubric (attach if supplied) ---
    if doctrine_score is not None:
        card["doctrine_pct"] = doctrine_score

    card["headline"] = _headline(card)
    _write_json(os.path.join(out_dir, f"scorecard_{name}.json"), card, indent=2)
    return card


def _headline(card):
    ca = card.get("construction_alpha") or {}
    bb = card.get("book_backtest") or {}
    return {
        "book_cagr_pct": bb.get("cagr_pct"),
        "book_sharpe": bb.get("sharpe"),
        "book_max_dd_pct": bb.get("max_drawdown_pct"),
        "book_alpha_vs_spy_pct": bb.get("alpha_vs_spy_cagr_pct"),
        "construction_mean_ret_pct": ca.get("model_mean_ret_pct"),
        "construction_median_dte": ca.get("model_median_dte"),
        "construction_dte_ge120_pct": ca.get("model_dte_ge_120_pct"),
        "doctrine_pct": card.get("doctrine_pct"),
    }


def compare(names):
    cards = [(_load(os.path.join(HERE, f"scorecard_{n}.json"))) for n in names]
    cards = [c for c in cards if c]
    cols = [("model", "name"), ("bookCAGR%", "book_cagr_pct"), ("Sharpe", "book_sharpe"),
            ("maxDD%", "book_max_dd_pct"), ("αvsSPY%", "book_alpha_vs_spy_pct"),
            ("constRet%", "construction_mean_ret_pct"), ("medDTE", "construction_median_dte"),
            ("DTE≥120%", "construction_dte_ge120_pct"), ("doctrine%", "doctrine_pct")]
    w = {h: max(len(h), 9) for h, _ in cols}
    print("  ".join(h.ljust(w[h]) for h, _ in cols))
    print("  ".join("-" * w[h] for h, _ in cols))
    for c in cards:
        h = c.get("headline", {})
        row = []
        for head, key in cols:
            v = c["name"] if key == "name" else h.get(key)
            row.append(("" if v is None else str(v)).ljust(w[head]))
        print("  ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--base", default="http://127.0.0.1:8082")
    r.add_argument("--model", default="/Users/trevorgordon/models/gemma-4-31b-derefused-a03nt-8bit")
    r.add_argument("--start", default="2024-02-01")
    r.add_argument("--end", default="2026-04-15")
    r.add_argument("--step", type=int, default=7)
    r.add_argument("--skip-construction", action="store_true")
    r.add_argument("--skip-book", action="store_true")
    r.add_argument("--doctrine", type=float, default=None)
    c = sub.add_parser("compare")
    c.add_argument("names", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        card = run(a.name, a.base, a.model, a.start, a.end, a.step,
                   a.skip_construction, a.skip_book, a.doctrine)
        print(json.dumps(card["headline"], indent=2))
    else:
        compare(a.names)
