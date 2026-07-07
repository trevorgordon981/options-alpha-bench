"""
MODEL-IN-THE-LOOP alpha scoring — the alpha track for the gauntlet.

Ties the whole thing together:
  - rubric track (existing gauntlet): does the model REASON like the doctrine?
  - alpha track (this):               does the model's ACTUAL construction make money?

For each real setup we hand the model the market context available at the entry date
and ask it to construct ONE defined-risk options trade as strict JSON. We parse that
into a Trade, walk it forward on the real underlying path (engine.py), and score
realized return-on-premium. We compare the model against two baselines on the SAME
setups: a fixed doctrine-35-DTE construction, and buy-and-hold the underlying.
"""
import argparse
import json
import os
import re
import statistics as st
import urllib.request

from engine import Trade, simulate
from scenarios import generate

PROMPT = """You are Walter, a defined-risk options swing trader. A setup:

Ticker: {ticker}
Spot: ${spot:.2f}
20-day momentum: {mom:+.1%}
Intended trade window (thesis horizon): ~{hz} calendar days
Account is small and cash-only (no naked assignment; a 3-DTE time-stop applies; fixed -30% premium stop).

Construct ONE defined-risk options trade for this setup. Think briefly, then output
your trade as a strict JSON object ON ITS OWN LINE, exactly this schema:
{{"direction": "long_call|long_put|call_debit_spread|put_debit_spread", "entry_dte": <int days to expiry>, "long_delta": <0.05-0.95>, "stop_pct": 0.30, "conviction": "low|med|high"}}
"""


def call_model(prompt, base, model, max_tokens=900):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.3}
    headers = {"Content-Type": "application/json"}
    if "openrouter" in base:                       # stock-M3-via-OpenRouter baseline
        payload["reasoning"] = {"enabled": False}  # else reasoning eats the budget -> null content
        payload["provider"] = {"quantizations": ["fp8", "int8"]}
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_trade(text):
    """Pull the last JSON object with a 'direction' key from the model output."""
    cands = re.findall(r"\{[^{}]*\"direction\"[^{}]*\}", text)
    for c in reversed(cands):
        try:
            j = json.loads(c)
            if "direction" in j and "entry_dte" in j:
                return j
        except Exception:
            continue
    return None


def score(setups, base, model, limit=None):
    rows = []
    for i, s in enumerate(setups if limit is None else setups[:limit]):
        # entry spot from the engine's own path (single source of truth)
        from data import path_from
        try:
            _, fwd = path_from(s["ticker"], s["entry_date"], days_forward=s["horizon_days"] + 10)
            spot = fwd[0][1]
        except Exception:
            continue
        prompt = PROMPT.format(ticker=s["ticker"], spot=spot, mom=s["mom20"], hz=s["horizon_days"])
        try:
            out = call_model(prompt, base, model)
        except Exception as e:
            rows.append({"ticker": s["ticker"], "entry": s["entry_date"], "err": f"call:{e}"})
            continue
        j = parse_trade(out)
        if not j:
            rows.append({"ticker": s["ticker"], "entry": s["entry_date"], "err": "parse_fail",
                         "raw": out[-200:]})
            continue
        # model trade
        mt = Trade(s["ticker"], s["entry_date"], j["direction"], int(j["entry_dte"]),
                   horizon_days=s["horizon_days"],
                   long_delta=float(j.get("long_delta", 0.6)),
                   premium_stop=float(j.get("stop_pct", 0.30)))
        mr = simulate(mt)
        # doctrine baseline: same direction the model chose, but forced 35-DTE
        dt = Trade(s["ticker"], s["entry_date"], j["direction"], 35,
                   horizon_days=s["horizon_days"], long_delta=float(j.get("long_delta", 0.6)))
        dr = simulate(dt)
        if not mr.ok:
            rows.append({"ticker": s["ticker"], "entry": s["entry_date"], "err": mr.err,
                         "model_dte": j.get("entry_dte")})
            continue
        rows.append({
            "ticker": s["ticker"], "entry": s["entry_date"], "dir": j["direction"],
            "model_dte": int(j["entry_dte"]), "conviction": j.get("conviction"),
            "model_ret": round(mr.return_pct, 4), "model_reason": mr.exit_reason,
            "doctrine_ret": round(dr.return_pct, 4) if dr.ok else None,
            "underlying_ret": round(mr.underlying_return_pct, 4),
        })
        print(f"[{i+1}] {s['ticker']:5} {s['entry_date']} {j['direction']:16} "
              f"dte={j['entry_dte']:>4} -> ret={mr.return_pct*100:+6.1f}% ({mr.exit_reason})",
              flush=True)
    return rows


def summarize(rows):
    ok = [r for r in rows if "model_ret" in r]
    if not ok:
        return {"n_ok": 0, "n_err": len(rows)}
    mr = [r["model_ret"] for r in ok]
    dr = [r["doctrine_ret"] for r in ok if r["doctrine_ret"] is not None]
    dtes = [r["model_dte"] for r in ok]
    return {
        "n_ok": len(ok), "n_err": len(rows) - len(ok),
        "model_mean_ret_pct": round(100 * st.mean(mr), 2),
        "model_median_ret_pct": round(100 * st.median(mr), 2),
        "model_win_rate_pct": round(100 * sum(1 for x in mr if x > 0) / len(mr), 1),
        "doctrine35_mean_ret_pct": round(100 * st.mean(dr), 2) if dr else None,
        "model_median_dte": int(st.median(dtes)),
        "model_dte_ge_120_pct": round(100 * sum(1 for d in dtes if d >= 120) / len(dtes), 1),
        "alpha_vs_doctrine_pct": round(100 * (st.mean(mr) - st.mean(dr)), 2) if dr else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default="/Users/trevorgordon/models/gemma-4-31b-derefused-a03nt-8bit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="model_alpha")
    a = ap.parse_args()
    setups = generate()
    print(f"{len(setups)} setups; scoring {a.limit or len(setups)} via {a.model}", flush=True)
    rows = score(setups, a.base, a.model, a.limit)
    summ = summarize(rows)
    with open(f"{a.out}_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(f"{a.out}_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
