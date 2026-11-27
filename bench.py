"""
Byron Bench — synthetic Black-Scholes options simulation & model scorecard harness.

Three tracks:
  1. construction_alpha : mechanical momentum setups; model builds one defined-risk
                          trade per setup; scored under fixed stop/time-stop rules
                          against a paired 35-DTE baseline.
  2. book_backtest      : model runs a simulated $100k options book over a date window;
                          CAGR, Sharpe, max drawdown, hit rate, cash & SPY comparators.
  3. doctrine (optional): attach an external rubric score via --doctrine <pct>.

All options P&L is simulated via constant-IV Black-Scholes on daily underlying closes
(not historical option chains). Cached track outputs are keyed by full model, data,
prompt, and implementation provenance and written atomically.

Usage:
  bench.py demo
  bench.py run --name myrun --model <model-id> --base http://127.0.0.1:8082
  bench.py compare myrun otherrun
"""
import argparse
import hashlib
import inspect
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_META_KEY = "_cache_meta"
DEFAULT_MIN_COVERAGE = 0.80

_TRACK_SOURCE_FILES = {
    "construction_alpha": (
        "pricing.py",
        "engine.py",
        "data.py",
        "schema.py",
        "llm_client.py",
        "scenarios.py",
        "run_model_alpha.py",
    ),
    "book_backtest": (
        "pricing.py",
        "data.py",
        "market.py",
        "schema.py",
        "llm_client.py",
        "book.py",
    ),
}


class CacheIntegrityError(ValueError):
    """Raised when a cached track artifact does not match the requested identity."""


def _load(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _write_json(path, payload, indent=2):
    """Atomically write JSON to path via temp file + fsync + os.replace."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".bench_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _code_digest(track: str) -> str:
    """SHA-256 prefix over the source files governing `track`."""
    files = _TRACK_SOURCE_FILES.get(track, ("bench.py",))
    h = hashlib.sha256()
    for rel in files:
        path = os.path.join(SRC_DIR, rel)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _prompt_digest(track: str) -> str:
    """SHA-256 prefix over the active prompt template for `track`."""
    h = hashlib.sha256()
    if track == "construction_alpha":
        src_path = os.path.join(SRC_DIR, "run_model_alpha.py")
        try:
            import run_model_alpha as rma
            prompt_val = getattr(rma, "PROMPT", None)
        except Exception:
            prompt_val = None
        if isinstance(prompt_val, str):
            h.update(prompt_val.encode("utf-8"))
            return h.hexdigest()[:16]
        if os.path.isfile(src_path):
            with open(src_path, "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:16]
    if track == "book_backtest":
        src_path = os.path.join(SRC_DIR, "book.py")
        try:
            import book as book_mod
            book_cls = getattr(book_mod, "Book", None)
            prompt_fn = getattr(book_cls, "_prompt", None)
            if prompt_fn is not None:
                h.update(inspect.getsource(prompt_fn).encode("utf-8"))
                return h.hexdigest()[:16]
        except Exception:
            pass
        if os.path.isfile(src_path):
            with open(src_path, "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:16]
    h.update(track.encode("utf-8"))
    return h.hexdigest()[:16]


def _data_digest(data_cache_dir: str | None = None) -> str:
    """SHA-256 prefix over cached price data files."""
    try:
        from data import cache_manifest_digest
        return cache_manifest_digest(data_cache_dir)
    except Exception:
        return hashlib.sha256(b"no-data").hexdigest()[:16]


def _cache_identity(track, model, base, **params):
    """Build a full provenance-bound cache identity for a benchmark track."""
    identity = {
        "track": track,
        "model": model,
        "base": base,
        "code_digest": params.pop("code_digest", None) or _code_digest(track),
        "prompt_digest": params.pop("prompt_digest", None) or _prompt_digest(track),
        "data_digest": params.pop("data_digest", None) or _data_digest(),
    }
    if track == "construction_alpha":
        identity["stop_policy"] = params.pop("stop_policy", "fixed")
    elif track == "book_backtest":
        identity["mark_frequency"] = params.pop("mark_frequency", "step")
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


def _accepts_kwarg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def run(
    name,
    base,
    model,
    start,
    end,
    step,
    skip_construction=False,
    skip_book=False,
    doctrine_score=None,
    cache_dir=None,
    *,
    model_revision=None,
    allow_model_stop=False,
    mark_frequency="step",
    min_coverage=DEFAULT_MIN_COVERAGE,
    api_key=None,
    api_key_env=None,
):
    out_dir = HERE if cache_dir is None else cache_dir
    stop_policy = "model_controlled" if allow_model_stop else "fixed"
    card = {
        "name": name,
        "model": model,
        "model_revision": model_revision,
        "base": base,
        "config": {
            "start": start,
            "end": end,
            "step": step,
            "stop_policy": stop_policy,
            "mark_frequency": mark_frequency,
            "min_coverage": min_coverage,
        },
        "provenance": {},
    }

    # --- track 1: construction alpha (reuse cache only on exact provenance match) ---
    c_params = {"stop_policy": stop_policy}
    if model_revision is not None:
        c_params["model_revision"] = model_revision
    if min_coverage != DEFAULT_MIN_COVERAGE:
        c_params["min_coverage"] = min_coverage
    pre_c_identity = _cache_identity("construction_alpha", model, base, **c_params)
    pre_ctag = _cache_tag("model_alpha", name, pre_c_identity)
    pre_csum_path = os.path.join(out_dir, f"{pre_ctag}_summary.json")
    csum = _load_cached(pre_csum_path, pre_c_identity)
    c_identity = pre_c_identity
    if csum is None and not skip_construction:
        from run_model_alpha import score as construction_score, summarize as construction_summ
        from scenarios import generate

        setups = generate()
        pre_ckpt_path = os.path.join(out_dir, f"{pre_ctag}_cases.jsonl")
        score_kwargs = {}
        if _accepts_kwarg(construction_score, "allow_model_stop"):
            score_kwargs["allow_model_stop"] = allow_model_stop
        if _accepts_kwarg(construction_score, "checkpoint_path"):
            score_kwargs["checkpoint_path"] = pre_ckpt_path
        if _accepts_kwarg(construction_score, "api_key"):
            score_kwargs["api_key"] = api_key
        if _accepts_kwarg(construction_score, "api_key_env"):
            score_kwargs["api_key_env"] = api_key_env
        rows = construction_score(setups, base, model, **score_kwargs)
        summ_kwargs = {}
        if _accepts_kwarg(construction_summ, "min_coverage"):
            summ_kwargs["min_coverage"] = min_coverage
        csum = construction_summ(rows, **summ_kwargs)
        # Recompute provenance digest AFTER cache hydration/updates (OP08)
        c_identity = _cache_identity("construction_alpha", model, base, **c_params)
        ctag = _cache_tag("model_alpha", name, c_identity)
        csum_path = os.path.join(out_dir, f"{ctag}_summary.json")
        if ctag != pre_ctag and os.path.exists(pre_ckpt_path):
            os.replace(pre_ckpt_path, os.path.join(out_dir, f"{ctag}_cases.jsonl"))
        _write_json(os.path.join(out_dir, f"{ctag}_rows.json"), rows, indent=2)
        _save_cached(csum_path, csum, c_identity)
    card["construction_alpha"] = csum
    card["provenance"]["construction_alpha"] = c_identity

    # --- track 2: book backtest (reuse cache only on exact provenance match) ---
    b_params = {"start": start, "end": end, "step": step, "mark_frequency": mark_frequency}
    if model_revision is not None:
        b_params["model_revision"] = model_revision
    if min_coverage != DEFAULT_MIN_COVERAGE:
        b_params["min_coverage"] = min_coverage
    pre_b_identity = _cache_identity("book_backtest", model, base, **b_params)
    pre_btag = _cache_tag("book", name, pre_b_identity)
    pre_bsum_path = os.path.join(out_dir, f"{pre_btag}_metrics.json")
    bsum = _load_cached(pre_bsum_path, pre_b_identity)
    b_identity = pre_b_identity
    if bsum is None and not skip_book:
        from book import Book

        pre_trace_path = os.path.join(out_dir, f"{pre_btag}_steps.jsonl")
        book_kwargs = {}
        if _accepts_kwarg(Book.__init__, "api_key"):
            book_kwargs["api_key"] = api_key
        if _accepts_kwarg(Book.__init__, "api_key_env"):
            book_kwargs["api_key_env"] = api_key_env
        if _accepts_kwarg(Book.__init__, "min_coverage"):
            book_kwargs["min_coverage"] = min_coverage
        if _accepts_kwarg(Book.__init__, "mark_frequency"):
            book_kwargs["mark_frequency"] = mark_frequency
        if _accepts_kwarg(Book.__init__, "trace_path"):
            book_kwargs["trace_path"] = pre_trace_path
        b = Book(base, model, start, end, step, **book_kwargs)
        bsum = b.run()
        # Recompute provenance digest AFTER cache hydration/updates (OP08)
        b_identity = _cache_identity("book_backtest", model, base, **b_params)
        btag = _cache_tag("book", name, b_identity)
        bsum_path = os.path.join(out_dir, f"{btag}_metrics.json")
        if btag != pre_btag and os.path.exists(pre_trace_path):
            os.replace(pre_trace_path, os.path.join(out_dir, f"{btag}_steps.jsonl"))
        _save_cached(bsum_path, bsum, b_identity)
        _write_json(os.path.join(out_dir, f"{btag}_equity.json"), b.equity_curve, indent=None)
    card["book_backtest"] = bsum
    card["provenance"]["book_backtest"] = b_identity

    # --- track 3: doctrine rubric (attach if supplied) ---
    if doctrine_score is not None:
        card["doctrine_pct"] = doctrine_score

    card["headline"] = _headline(card)
    _write_json(os.path.join(out_dir, f"scorecard_{name}.json"), card, indent=2)
    return card


def _track_verdict(track_data):
    if not track_data:
        return None
    if "verdict" in track_data:
        return track_data["verdict"]
    if "qualifying" in track_data:
        return "VALID" if track_data["qualifying"] else "INVALID"
    # Backwards-compatible fallback for minimal stub dicts in tests
    if "valid_decisions" in track_data and "decision_steps" in track_data:
        return "VALID" if track_data["valid_decisions"] > 0 else "INVALID"
    if "n_ok" in track_data and track_data.get("n_ok") == 0:
        return "INVALID"
    return "VALID"


def _headline(card):
    ca = card.get("construction_alpha") or {}
    bb = card.get("book_backtest") or {}

    ca_verdict = _track_verdict(ca)
    bb_verdict = _track_verdict(bb)
    present_verdicts = [v for v in (ca_verdict, bb_verdict) if v is not None]
    if not present_verdicts:
        overall_verdict = "INCOMPLETE"
    elif any(v == "INVALID" for v in present_verdicts):
        overall_verdict = "INVALID"
    else:
        overall_verdict = "VALID"

    bb_valid = bb_verdict == "VALID"
    ca_valid = ca_verdict == "VALID"

    return {
        "verdict": overall_verdict,
        "qualifying": overall_verdict == "VALID",
        "book_verdict": bb_verdict,
        "book_coverage_pct": bb.get("coverage_pct"),
        "book_valid_decisions": bb.get("valid_decisions"),
        "book_failed_steps": bb.get("failed_steps"),
        "book_invalid_actions": bb.get("invalid_actions"),
        "book_deliberate_abstentions": bb.get("deliberate_abstentions"),
        "book_duration_days": bb.get("duration_days"),
        "book_cash_cagr_pct": bb.get("cash_comparator_cagr_pct"),
        "book_spy_cagr_pct": bb.get("spy_buyhold_cagr_pct"),
        "book_cagr_pct": bb.get("cagr_pct") if bb_valid else None,
        "book_sharpe": bb.get("sharpe") if bb_valid else None,
        "book_max_dd_pct": bb.get("max_drawdown_pct") if bb_valid else None,
        "book_alpha_vs_spy_pct": bb.get("alpha_vs_spy_cagr_pct") if bb_valid else None,
        "construction_verdict": ca_verdict,
        "construction_coverage_pct": ca.get("coverage_pct"),
        "construction_n_ok": ca.get("n_ok"),
        "construction_n_total": ca.get("n_total"),
        "construction_mean_ret_pct": ca.get("model_mean_ret_pct") if ca_valid else None,
        "construction_median_dte": ca.get("model_median_dte") if ca_valid else None,
        "construction_dte_ge120_pct": ca.get("model_dte_ge_120_pct") if ca_valid else None,
        "construction_alpha_vs_doctrine_pct": ca.get("alpha_vs_doctrine_pct") if ca_valid else None,
        "doctrine_pct": card.get("doctrine_pct"),
    }


def _check_comparability(cards):
    reasons = []
    if not cards:
        return reasons
    for c in cards:
        h = c.get("headline") or {}
        if h.get("verdict") == "INVALID":
            reasons.append(f"{c['name']}: non-qualifying (verdict=INVALID)")

    if len(cards) < 2:
        return reasons

    windows = {
        (
            (c.get("config") or {}).get("start"),
            (c.get("config") or {}).get("end"),
            (c.get("config") or {}).get("step"),
        )
        for c in cards
        if c.get("config")
    }
    if len(windows) > 1:
        reasons.append(f"book date windows/cadences differ across cards: {sorted(windows)}")

    mark_freqs = {
        (c.get("config") or {}).get("mark_frequency")
        for c in cards
        if (c.get("config") or {}).get("mark_frequency") is not None
    }
    if len(mark_freqs) > 1:
        reasons.append(f"book mark_frequency differs across cards: {sorted(mark_freqs)}")

    stop_policies = {
        (c.get("config") or {}).get("stop_policy")
        for c in cards
        if (c.get("config") or {}).get("stop_policy") is not None
    }
    if len(stop_policies) > 1:
        reasons.append(f"construction stop_policy differs across cards: {sorted(stop_policies)}")

    unverified_revs = [c["name"] for c in cards if not c.get("model_revision")]
    if unverified_revs:
        reasons.append(
            f"unverified model_revision (alias-only build): {', '.join(unverified_revs)}"
        )

    for track in ("construction_alpha", "book_backtest"):
        for field in ("code_digest", "prompt_digest", "data_digest"):
            vals = {
                ((c.get("provenance") or {}).get(track) or {}).get(field)
                for c in cards
                if ((c.get("provenance") or {}).get(track) or {}).get(field) is not None
            }
            if len(vals) > 1:
                reasons.append(f"{track}.{field} differs across cards: {sorted(vals)}")
    return reasons


def compare(names, cache_dir=None):
    out_dir = HERE if cache_dir is None else cache_dir
    loaded = []
    missing = []
    for n in names:
        c = _load(os.path.join(out_dir, f"scorecard_{n}.json"))
        if c is None:
            missing.append(n)
        else:
            loaded.append(c)

    cols = [
        ("model", "name"),
        ("verdict", "verdict"),
        ("bookCov%", "book_coverage_pct"),
        ("bookCAGR%", "book_cagr_pct"),
        ("Sharpe", "book_sharpe"),
        ("maxDD%", "book_max_dd_pct"),
        ("αvsSPY%", "book_alpha_vs_spy_pct"),
        ("constCov%", "construction_coverage_pct"),
        ("constRet%", "construction_mean_ret_pct"),
        ("medDTE", "construction_median_dte"),
        ("doctrine%", "doctrine_pct"),
    ]
    w = {h: max(len(h), 9) for h, _ in cols}
    print("  ".join(h.ljust(w[h]) for h, _ in cols))
    print("  ".join("-" * w[h] for h, _ in cols))
    rows = []
    for c in loaded:
        h = c.get("headline", {})
        row_vals = {}
        row = []
        for head, key in cols:
            v = c["name"] if key == "name" else h.get(key)
            row_vals[key] = v
            row.append(("" if v is None else str(v)).ljust(w[head]))
        rows.append(row_vals)
        print("  ".join(row))
    for m in missing:
        row = [m.ljust(w["model"]), "MISSING".ljust(w["verdict"])] + [
            "".ljust(w[h]) for h, _ in cols[2:]
        ]
        print("  ".join(row))

    incomparable = _check_comparability(loaded)
    if missing:
        print(f"\nWARNING: missing scorecards: {', '.join(missing)}")
    if incomparable:
        print("\nWARNING: cards are NOT directly comparable:")
        for reason in incomparable:
            print(f"  - {reason}")

    return {
        "loaded": [c["name"] for c in loaded],
        "missing": missing,
        "incomparable_reasons": incomparable,
        "rows": rows,
    }


def demo():
    """Run a 100% offline, deterministic demonstration on pinned synthetic daily closes."""
    from unittest import mock

    from book import Book
    from data import load_fixture_into_cache
    import run_ab
    import scenarios

    fixture_path = os.path.join(SRC_DIR, "fixtures", "sample_prices.json")
    if not os.path.isfile(fixture_path):
        raise FileNotFoundError(f"Offline demo fixture missing: {fixture_path}")

    with tempfile.TemporaryDirectory(prefix="byron_demo_") as tmp:
        px_dir = os.path.join(tmp, "pxcache")
        fixture_sha = load_fixture_into_cache(fixture_path, px_dir)
        old_cache = os.environ.get("BYRON_CACHE")
        os.environ["BYRON_CACHE"] = px_dir
        try:
            demo_basket = ["NVDA", "AAPL", "MSFT", "META", "SPY"]
            setups = scenarios.generate(
                start="2024-02-15",
                end="2024-11-15",
                cadence_days=14,
                basket=demo_basket,
            )
            ab_rows = run_ab.run(setups)
            ab_summary = run_ab.summarize(ab_rows)

            # Valid mock book run (opens long-dated call when capacity/cash allows, else deliberately abstains)
            def _valid_responder(prompt, max_tokens=1200):
                if "Open positions: 8/8" in prompt:
                    return json.dumps({"close": [], "open": []})
                return json.dumps({
                    "close": [],
                    "open": [{
                        "ticker": "NVDA",
                        "direction": "long_call",
                        "entry_dte": 168,
                        "long_delta": 0.60,
                        "size_pct": 0.10,
                    }],
                })

            b_valid = Book(
                base="http://127.0.0.1:8000",
                model="demo-mock-valid",
                start="2024-03-01",
                end="2024-09-01",
                step_days=14,
                verbose=False,
                basket=demo_basket,
            )
            with mock.patch.object(b_valid, "_call", side_effect=_valid_responder):
                m_valid = b_valid.run()

            # All-failed mock book run (every call times out -> MUST be verdict=INVALID, alpha=None)
            b_failed = Book(
                base="http://127.0.0.1:8000",
                model="demo-mock-timeout",
                start="2024-03-01",
                end="2024-09-01",
                step_days=14,
                verbose=False,
                basket=demo_basket,
            )
            with mock.patch.object(b_failed, "_call", side_effect=TimeoutError("simulated timeout")):
                m_failed = b_failed.run()

            report = {
                "mode": "offline_synthetic_simulation_demo",
                "fixture": "fixtures/sample_prices.json",
                "fixture_sha256": fixture_sha,
                "paired_dte_ab": ab_summary,
                "book_valid_mock": {
                    "verdict": m_valid["verdict"],
                    "coverage_pct": m_valid["coverage_pct"],
                    "duration_days": m_valid["duration_days"],
                    "sampling_frequency": m_valid["sampling_frequency"],
                    "cagr_pct": m_valid["cagr_pct"],
                    "sharpe": m_valid["sharpe"],
                    "max_drawdown_pct": m_valid["max_drawdown_pct"],
                    "cash_comparator_cagr_pct": m_valid["cash_comparator_cagr_pct"],
                    "spy_buyhold_cagr_pct": m_valid["spy_buyhold_cagr_pct"],
                    "alpha_vs_spy_cagr_pct": m_valid["alpha_vs_spy_cagr_pct"],
                },
                "book_all_failed_mock": {
                    "verdict": m_failed["verdict"],
                    "invalid_reason": m_failed["invalid_reason"],
                    "coverage_pct": m_failed["coverage_pct"],
                    "transport_errors": m_failed["transport_errors"],
                    "alpha_vs_spy_cagr_pct": m_failed["alpha_vs_spy_cagr_pct"],
                },
            }
            print(json.dumps(report, indent=2))
            return report
        finally:
            if old_cache is None:
                os.environ.pop("BYRON_CACHE", None)
            else:
                os.environ["BYRON_CACHE"] = old_cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--base", default="http://127.0.0.1:8082")
    r.add_argument("--model", default=os.environ.get("BENCH_MODEL", "local-model"))
    r.add_argument("--model-revision", default=None)
    r.add_argument("--start", default="2024-02-01")
    r.add_argument("--end", default="2026-04-15")
    r.add_argument("--step", type=int, default=7)
    r.add_argument("--mark-frequency", choices=["step", "daily"], default="step")
    r.add_argument("--allow-model-stop", action="store_true")
    r.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    r.add_argument("--skip-construction", action="store_true")
    r.add_argument("--skip-book", action="store_true")
    r.add_argument("--doctrine", type=float, default=None)
    c = sub.add_parser("compare")
    c.add_argument("names", nargs="+")
    sub.add_parser("demo")
    a = ap.parse_args()
    if a.cmd == "run":
        card = run(
            a.name,
            a.base,
            a.model,
            a.start,
            a.end,
            a.step,
            a.skip_construction,
            a.skip_book,
            a.doctrine,
            model_revision=a.model_revision,
            allow_model_stop=a.allow_model_stop,
            mark_frequency=a.mark_frequency,
            min_coverage=a.min_coverage,
        )
        print(json.dumps(card["headline"], indent=2))
    elif a.cmd == "compare":
        compare(a.names)
    elif a.cmd == "demo":
        demo()
