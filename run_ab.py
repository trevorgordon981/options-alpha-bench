"""
DTE A/B backtest — the direct answer to "does 8x DTE produce alpha?"

For every mechanical setup we construct the SAME directional trade two ways:
  A) doctrine   : 35-DTE (the 25-45 band dataset-v1 taught)
  B) 8x rule    : 8 x horizon DTE (Trevor's theta rule)
...walk both forward on the real underlying path, apply identical exit discipline,
and compare realized return-on-premium. Because only the expiry differs, the paired
difference (B - A) IS the alpha attributable to the DTE doctrine.
"""
import json
import statistics as st
import sys

from engine import Trade, simulate
from scenarios import generate

DOCTRINE_DTE = 35


def run(setups, target_pct=None):
    rows = []
    for s in setups:
        hz = s["horizon_days"]
        eight_x = 8 * hz
        legs = {}
        for tag, dte in [("doctrine", DOCTRINE_DTE), ("eightx", eight_x)]:
            t = Trade(s["ticker"], s["entry_date"], s["direction"], dte,
                      horizon_days=hz, long_delta=0.60, target_pct=target_pct)
            legs[tag] = simulate(t)
        a, b = legs["doctrine"], legs["eightx"]
        if not (a.ok and b.ok):
            continue
        rows.append({
            "ticker": s["ticker"], "entry": s["entry_date"], "dir": s["direction"],
            "mom20": s["mom20"], "underlying_ret": round(a.underlying_return_pct, 4),
            "doctrine_ret": round(a.return_pct, 4), "doctrine_reason": a.exit_reason,
            "eightx_ret": round(b.return_pct, 4), "eightx_reason": b.exit_reason,
            "delta": round(b.return_pct - a.return_pct, 4),
        })
    return rows


def summarize(rows):
    def agg(key):
        v = [r[key] for r in rows]
        wins = sum(1 for x in v if x > 0)
        return {
            "n": len(v), "mean_ret_pct": round(100 * st.mean(v), 2),
            "median_ret_pct": round(100 * st.median(v), 2),
            "win_rate_pct": round(100 * wins / len(v), 1),
            "total_ret_units": round(sum(v), 3),
        }
    deltas = [r["delta"] for r in rows]
    b_wins = sum(1 for d in deltas if d > 0)
    out = {
        "n_setups": len(rows),
        "doctrine_35dte": agg("doctrine_ret"),
        "eightx_rule": agg("eightx_ret"),
        "paired_delta_8x_minus_doctrine": {
            "mean_pct": round(100 * st.mean(deltas), 2),
            "median_pct": round(100 * st.median(deltas), 2),
            "8x_better_pct": round(100 * b_wins / len(deltas), 1),
        },
    }
    # exit-reason mix (theta story shows up here)
    import collections
    out["doctrine_exit_mix"] = dict(collections.Counter(r["doctrine_reason"] for r in rows))
    out["eightx_exit_mix"] = dict(collections.Counter(r["eightx_reason"] for r in rows))
    return out


if __name__ == "__main__":
    print("generating mechanical setups...", flush=True)
    setups = generate()
    print(f"  {len(setups)} setups", flush=True)
    tgt = None
    if len(sys.argv) > 1 and sys.argv[1] != "none":
        tgt = float(sys.argv[1])
    rows = run(setups, target_pct=tgt)
    summ = summarize(rows)
    with open("ab_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open("ab_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
