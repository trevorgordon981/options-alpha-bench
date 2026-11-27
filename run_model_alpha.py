"""
MODEL-IN-THE-LOOP construction scoring — Track 1 of Byron Bench.

For each mechanical setup we hand the model the market context available at the
entry date and ask it to construct ONE defined-risk options trade as strict JSON.
We validate the typed schema strictly, walk the trade forward on daily closes
(engine.py), and compare against a paired 35-DTE baseline under common stop and
time-stop rules (unless model-controlled stop variation is explicitly enabled).
"""
import argparse
from datetime import date, timedelta
import json
import os
import statistics as st

from data import PriceSnapshot, hydrate_price_snapshot, path_from, use_price_snapshot
from engine import Trade, simulate
from llm_client import chat_completion
from scenarios import generate
from schema import (
    append_jsonl_record,
    extract_json_objects,
    validate_construction_trade,
)

FIXED_PREMIUM_STOP = 0.30
FIXED_TIME_STOP_DTE = 3
MIN_VALID_COVERAGE = 0.80

PROMPT = """You are a defined-risk options swing trader. A setup:

Ticker: {ticker}
Spot: ${spot:.2f}
20-day momentum: {mom:+.1%}
Intended trade window (thesis horizon): ~{hz} calendar days
Account is small and cash-only (no naked assignment; a 3-DTE time-stop applies; fixed -30% premium stop).

Construct ONE defined-risk options trade for this setup. Think briefly, then output
your trade as a strict JSON object ON ITS OWN LINE, exactly this schema:
{{"direction": "long_call|long_put|call_debit_spread|put_debit_spread", "entry_dte": <int days to expiry>, "long_delta": <0.05-0.95>, "stop_pct": 0.30, "conviction": "low|med|high"}}
"""


def call_model(prompt, base, model, max_tokens=900, api_key=None, api_key_env=None):
    return chat_completion(
        base,
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        timeout=180,
        api_key=api_key,
        api_key_env=api_key_env,
    )


def parse_and_validate_trade(text):
    """Extract the last candidate trade JSON object and strictly validate its typed schema.

    Returns (validated_trade_dict | None, error_status | None, error_detail | None).
    """
    objs = extract_json_objects(text)
    candidates = [o for o in objs if "direction" in o or "entry_dte" in o]
    if not candidates:
        return None, "parse_fail", "no candidate JSON object with direction/entry_dte found"

    last_err = None
    for cand in reversed(candidates):
        valid, err = validate_construction_trade(cand)
        if valid is not None:
            return valid, None, None
        if last_err is None:
            last_err = err
    return None, "schema_fail", last_err


def parse_trade(text):
    """Pull and strictly validate the last trade JSON object from model output."""
    valid, _, _ = parse_and_validate_trade(text)
    return valid


def hydrate_snapshot_for_setups(setups, *, get_closes_fn=None) -> PriceSnapshot:
    """Hydrate all price series needed for `setups` ONCE into an immutable PriceSnapshot."""
    windows: dict[str, tuple[str, str]] = {}
    for s in setups:
        tk = str(s["ticker"]).upper()
        d0 = date.fromisoformat(s["entry_date"])
        hz = int(s.get("horizon_days", 21))
        lo = (d0 - timedelta(days=45)).isoformat()
        hi = (d0 + timedelta(days=hz + 30)).isoformat()
        if tk in windows:
            prev_lo, prev_hi = windows[tk]
            windows[tk] = (min(prev_lo, lo), max(prev_hi, hi))
        else:
            windows[tk] = (lo, hi)
    return hydrate_price_snapshot(windows, get_closes_fn=get_closes_fn)


def score(
    setups,
    base,
    model,
    limit=None,
    *,
    allow_model_stop=False,
    checkpoint_path=None,
    api_key=None,
    api_key_env=None,
    verbose=True,
    price_snapshot=None,
    provenance_out=None,
):
    """Score setups one-by-one against a single frozen in-memory price snapshot."""
    rows = []
    stop_policy = "model_controlled" if allow_model_stop else "fixed"
    selected = setups if limit is None else setups[:limit]
    snapshot = (
        price_snapshot
        if price_snapshot is not None
        else hydrate_snapshot_for_setups(selected)
    )
    if provenance_out is not None:
        provenance_out["price_snapshot"] = snapshot
        provenance_out["data_digest"] = snapshot.digest()
        provenance_out["tickers"] = list(snapshot.tickers())

    def _emit(row):
        rows.append(row)
        if checkpoint_path:
            append_jsonl_record(checkpoint_path, row)

    with use_price_snapshot(snapshot):
        for i, s in enumerate(selected):
            case_id = f"{s['ticker']}:{s['entry_date']}:{s['horizon_days']}"
            case_digest = snapshot.subset([s["ticker"]]).digest()
            try:
                _, fwd = path_from(
                    s["ticker"],
                    s["entry_date"],
                    days_forward=s["horizon_days"] + 10,
                    price_snapshot=snapshot,
                )
                if not fwd:
                    raise ValueError("empty forward path")
                spot = fwd[0][1]
            except Exception as e:
                _emit({
                    "case_id": case_id,
                    "ticker": s["ticker"],
                    "entry": s["entry_date"],
                    "status": "data_error",
                    "err": f"data:{e}",
                    "data_digest": case_digest,
                })
                continue

            prompt = PROMPT.format(ticker=s["ticker"], spot=spot, mom=s["mom20"], hz=s["horizon_days"])
            try:
                out = call_model(prompt, base, model, api_key=api_key, api_key_env=api_key_env)
            except Exception as e:
                _emit({
                    "case_id": case_id,
                    "ticker": s["ticker"],
                    "entry": s["entry_date"],
                    "status": "call_error",
                    "err": f"call:{e}",
                    "data_digest": case_digest,
                })
                continue

            j, err_status, err_detail = parse_and_validate_trade(out)
            if j is None:
                _emit({
                    "case_id": case_id,
                    "ticker": s["ticker"],
                    "entry": s["entry_date"],
                    "status": err_status,
                    "err": f"{err_status}:{err_detail}",
                    "raw": out[-200:] if isinstance(out, str) else repr(out)[:200],
                    "data_digest": case_digest,
                })
                continue

            requested_stop = j["stop_pct"]
            applied_stop = requested_stop if allow_model_stop else FIXED_PREMIUM_STOP
            stop_override_applied = (not allow_model_stop) and (abs(requested_stop - FIXED_PREMIUM_STOP) > 1e-9)
            policy_confounded = allow_model_stop and (abs(requested_stop - FIXED_PREMIUM_STOP) > 1e-9)

            # Model trade: under default fixed policy, stop & time-stop match baseline identically.
            mt = Trade(
                s["ticker"],
                s["entry_date"],
                j["direction"],
                j["entry_dte"],
                horizon_days=s["horizon_days"],
                long_delta=j["long_delta"],
                premium_stop=applied_stop,
                time_stop_dte=FIXED_TIME_STOP_DTE,
            )
            mr = simulate(mt)

            # Paired 35-DTE baseline with identical direction, delta, and common fixed stop policy.
            dt = Trade(
                s["ticker"],
                s["entry_date"],
                j["direction"],
                35,
                horizon_days=s["horizon_days"],
                long_delta=j["long_delta"],
                premium_stop=FIXED_PREMIUM_STOP,
                time_stop_dte=FIXED_TIME_STOP_DTE,
            )
            dr = simulate(dt)

            if not mr.ok:
                _emit({
                    "case_id": case_id,
                    "ticker": s["ticker"],
                    "entry": s["entry_date"],
                    "status": "sim_error",
                    "err": mr.err,
                    "model_dte": j["entry_dte"],
                    "data_digest": case_digest,
                })
                continue

            row = {
                "case_id": case_id,
                "ticker": s["ticker"],
                "entry": s["entry_date"],
                "status": "ok",
                "data_digest": case_digest,
                "dir": j["direction"],
                "model_dte": j["entry_dte"],
                "long_delta": j["long_delta"],
                "conviction": j["conviction"],
                "stop_policy": stop_policy,
                "requested_stop_pct": requested_stop,
                "applied_stop_pct": applied_stop,
                "baseline_stop_pct": FIXED_PREMIUM_STOP,
                "stop_override_applied": stop_override_applied,
                "policy_confounded": policy_confounded,
                "model_ret": round(mr.return_pct, 4),
                "model_reason": mr.exit_reason,
                "doctrine_ret": round(dr.return_pct, 4) if dr.ok else None,
                "underlying_ret": round(mr.underlying_return_pct, 4),
            }
            _emit(row)
            if verbose:
                print(
                    f"[{i+1}] {s['ticker']:5} {s['entry_date']} {j['direction']:16} "
                    f"dte={j['entry_dte']:>4} stop={applied_stop:.2f} -> "
                    f"ret={mr.return_pct*100:+6.1f}% ({mr.exit_reason})",
                    flush=True,
                )
    return rows


def summarize(rows, min_coverage=MIN_VALID_COVERAGE):
    """Aggregate scored rows and gate headline performance claims on valid coverage."""
    n_total = len(rows)
    ok = [r for r in rows if "model_ret" in r]
    n_ok = len(ok)
    n_err = n_total - n_ok
    coverage_ratio = (n_ok / n_total) if n_total > 0 else 0.0
    coverage_pct = round(100.0 * coverage_ratio, 1)

    n_call_err = sum(1 for r in rows if r.get("status") == "call_error" or str(r.get("err", "")).startswith("call:"))
    n_parse_fail = sum(1 for r in rows if r.get("status") == "parse_fail" or r.get("err") == "parse_fail")
    n_schema_fail = sum(1 for r in rows if r.get("status") == "schema_fail" or str(r.get("err", "")).startswith("schema_fail"))
    n_sim_err = sum(1 for r in rows if r.get("status") == "sim_error")
    n_data_err = sum(1 for r in rows if r.get("status") == "data_error")

    stop_policies = sorted({r.get("stop_policy", "fixed") for r in ok})
    stop_policy = stop_policies[0] if len(stop_policies) == 1 else ("mixed" if stop_policies else "fixed")
    stop_overrides = sum(1 for r in ok if r.get("stop_override_applied"))
    policy_confounded_n = sum(1 for r in ok if r.get("policy_confounded"))

    valid = n_total > 0 and n_ok > 0 and coverage_ratio >= min_coverage
    if n_total == 0:
        invalid_reason = "no_cases_evaluated"
    elif n_ok == 0:
        invalid_reason = f"zero_valid_responses (0/{n_total})"
    elif coverage_ratio < min_coverage:
        invalid_reason = f"coverage_below_threshold ({coverage_pct:.1f}% < {min_coverage*100:.1f}%)"
    else:
        invalid_reason = None

    verdict = "VALID" if valid else "INVALID"

    base_summary = {
        "verdict": verdict,
        "qualifying": valid,
        "invalid_reason": invalid_reason,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_call_err": n_call_err,
        "n_parse_fail": n_parse_fail,
        "n_schema_fail": n_schema_fail,
        "n_sim_err": n_sim_err,
        "n_data_err": n_data_err,
        "coverage_pct": coverage_pct,
        "min_coverage_pct": round(min_coverage * 100.0, 1),
        "stop_policy": stop_policy,
        "fixed_stop_pct": FIXED_PREMIUM_STOP,
        "stop_overrides": stop_overrides,
        "policy_confounded_cases": policy_confounded_n,
    }

    if not valid:
        base_summary.update({
            "model_mean_ret_pct": None,
            "model_median_ret_pct": None,
            "model_win_rate_pct": None,
            "doctrine35_mean_ret_pct": None,
            "model_median_dte": None,
            "model_dte_ge_120_pct": None,
            "alpha_vs_doctrine_pct": None,
        })
        return base_summary

    mr = [r["model_ret"] for r in ok]
    dr = [r["doctrine_ret"] for r in ok if r.get("doctrine_ret") is not None]
    dtes = [r["model_dte"] for r in ok]
    base_summary.update({
        "model_mean_ret_pct": round(100 * st.mean(mr), 2),
        "model_median_ret_pct": round(100 * st.median(mr), 2),
        "model_win_rate_pct": round(100 * sum(1 for x in mr if x > 0) / len(mr), 1),
        "doctrine35_mean_ret_pct": round(100 * st.mean(dr), 2) if dr else None,
        "model_median_dte": int(st.median(dtes)),
        "model_dte_ge_120_pct": round(100 * sum(1 for d in dtes if d >= 120) / len(dtes), 1),
        "alpha_vs_doctrine_pct": round(100 * (st.mean(mr) - st.mean(dr)), 2) if dr else None,
    })
    return base_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "local-model"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="model_alpha")
    ap.add_argument("--allow-model-stop", action="store_true",
                    help="Allow model-specified stop_pct (labels policy variation explicitly)")
    ap.add_argument("--min-coverage", type=float, default=MIN_VALID_COVERAGE)
    a = ap.parse_args()
    setups = generate()
    print(f"{len(setups)} setups; scoring {a.limit or len(setups)} via {a.model}", flush=True)
    prov = {}
    rows = score(
        setups,
        a.base,
        a.model,
        a.limit,
        allow_model_stop=a.allow_model_stop,
        checkpoint_path=f"{a.out}_cases.jsonl",
        provenance_out=prov,
    )
    summ = summarize(rows, min_coverage=a.min_coverage)
    if "data_digest" in prov:
        summ["data_digest"] = prov["data_digest"]
    with open(f"{a.out}_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(f"{a.out}_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
