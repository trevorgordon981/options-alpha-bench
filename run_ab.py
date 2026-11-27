"""
Paired DTE A/B synthetic-pricing experiment.

For every mechanical setup we construct the same directional 0.60-delta option idea
two ways under Black-Scholes daily-close simulation with identical exit rules:
  A) short leg  : 35-DTE
  B) longer leg : 8 x horizon DTE

Because target delta is held constant across different expiries, the paired comparison
captures combined strike drift, premium/leverage, gamma, path, and theta differences —
not an isolated theta-only effect or executable option-chain P&L.
"""
import collections
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
        strike_a = a.legs.get("long_k", 0.0)
        strike_b = b.legs.get("long_k", 0.0)
        rows.append({
            "ticker": s["ticker"], "entry": s["entry_date"], "dir": s["direction"],
            "mom20": s["mom20"], "underlying_ret": round(a.underlying_return_pct, 4),
            "doctrine_ret": round(a.return_pct, 4), "doctrine_reason": a.exit_reason,
            "eightx_ret": round(b.return_pct, 4), "eightx_reason": b.exit_reason,
            "delta": round(b.return_pct - a.return_pct, 4),
            "doctrine_strike": strike_a, "eightx_strike": strike_b,
            "strike_diff_pct": round((strike_b - strike_a) / a.entry_spot * 100, 2) if a.entry_spot else 0.0,
            "doctrine_cost": round(a.entry_cost, 4), "eightx_cost": round(b.entry_cost, 4),
            "cost_ratio_8x_to_35": round(b.entry_cost / a.entry_cost, 3) if a.entry_cost else None,
            "doctrine_gamma": a.entry_gamma, "eightx_gamma": b.entry_gamma,
            "doctrine_theta_day": a.entry_theta_per_day, "eightx_theta_day": b.entry_theta_per_day,
        })
    return rows


def summarize(rows):
    if not rows:
        return {
            "n_setups": 0,
            "simulation_model": "constant_iv_black_scholes_daily_close",
            "doctrine_35dte": None,
            "eightx_rule": None,
            "paired_delta_8x_minus_doctrine": None,
            "pricing_decomposition": None,
            "doctrine_exit_mix": {},
            "eightx_exit_mix": {},
        }

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

    strike_diffs = [abs(r["strike_diff_pct"]) for r in rows if "strike_diff_pct" in r]
    cost_ratios = [r["cost_ratio_8x_to_35"] for r in rows if r.get("cost_ratio_8x_to_35") is not None]
    gamma_35 = [r["doctrine_gamma"] for r in rows if "doctrine_gamma" in r]
    gamma_8x = [r["eightx_gamma"] for r in rows if "eightx_gamma" in r]
    theta_35 = [r["doctrine_theta_day"] for r in rows if "doctrine_theta_day" in r]
    theta_8x = [r["eightx_theta_day"] for r in rows if "eightx_theta_day" in r]

    out = {
        "n_setups": len(rows),
        "simulation_model": "constant_iv_black_scholes_daily_close",
        "doctrine_35dte": agg("doctrine_ret"),
        "eightx_rule": agg("eightx_ret"),
        "paired_delta_8x_minus_doctrine": {
            "mean_pct": round(100 * st.mean(deltas), 2),
            "median_pct": round(100 * st.median(deltas), 2),
            "8x_better_pct": round(100 * b_wins / len(deltas), 1),
        },
        "pricing_decomposition": {
            "note": (
                "Repricing equal-delta options at different DTE changes strike, entry premium, "
                "gamma, and path sensitivity alongside theta; paired differences are not a pure "
                "theta-only causal effect."
            ),
            "median_abs_strike_drift_pct_of_spot": round(st.median(strike_diffs), 2) if strike_diffs else None,
            "median_entry_cost_ratio_8x_over_35": round(st.median(cost_ratios), 3) if cost_ratios else None,
            "median_entry_gamma_35dte": round(st.median(gamma_35), 6) if gamma_35 else None,
            "median_entry_gamma_8x": round(st.median(gamma_8x), 6) if gamma_8x else None,
            "median_entry_theta_day_35dte": round(st.median(theta_35), 4) if theta_35 else None,
            "median_entry_theta_day_8x": round(st.median(theta_8x), 4) if theta_8x else None,
        },
    }
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
