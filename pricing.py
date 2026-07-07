"""
Black-Scholes option pricing + greeks, and an IV estimator.

Why BS (not real option chains): free historical option-chain data is scarce and
unreliable, but the question we care about — "does a longer DTE produce alpha, or
does theta bleed short-DTED trades to death?" — is a THETA question, and theta is
exactly what BS models faithfully given the real underlying path. We reconstruct
each option's value from the REAL underlying close each day, an IV estimated from
the underlying's own realized vol, and the shrinking time-to-expiry. This isolates
the delta+theta tradeoff that the DTE decision actually turns on.

Assumptions (stated so they're honest, not hidden):
  - IV held constant over the hold at entry-IV (isolates theta/delta; no vega noise)
  - IV proxy = trailing realized vol * IV_PREMIUM, floored (implied usually > realized)
  - European-style BS (fine for the P&L question; American early-ex premium is tiny
    for the non-dividend growth names we test)
  - r = flat risk-free
"""
import math

R_FREE = 0.045          # flat risk-free (2024-25 T-bill ballpark)
IV_PREMIUM = 1.10       # implied typically ~10% over trailing realized
IV_FLOOR = 0.15
IV_CAP = 1.50


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t_years, iv, kind="call", r=R_FREE):
    """Black-Scholes price of one option (per share). kind in {'call','put'}."""
    if t_years <= 0:
        # intrinsic at/after expiry
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    if iv <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    else:
        return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot, strike, t_years, iv, kind="call", r=R_FREE):
    if t_years <= 0 or iv <= 0:
        # step function at expiry
        itm = (spot > strike) if kind == "call" else (spot < strike)
        return (1.0 if kind == "call" else -1.0) if itm else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0


def strike_for_delta(spot, t_years, iv, target_delta, kind="call", r=R_FREE):
    """Solve for the strike whose delta == target_delta (bisection). target_delta>0."""
    lo, hi = spot * 0.3, spot * 3.0
    td = abs(target_delta)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        d = abs(bs_delta(spot, mid, t_years, iv, kind, r))
        # calls: higher strike -> lower delta; puts (abs): higher strike -> higher |delta|
        if kind == "call":
            if d > td:
                lo = mid
            else:
                hi = mid
        else:
            if d > td:
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


def realized_vol(closes, lookback=20):
    """Annualized realized vol from a list/Series of daily closes (most recent last)."""
    c = [float(x) for x in list(closes)[-(lookback + 1):]]
    if len(c) < 3:
        return IV_FLOOR
    rets = [math.log(c[i] / c[i - 1]) for i in range(1, len(c)) if c[i - 1] > 0]
    if len(rets) < 2:
        return IV_FLOOR
    m = sum(rets) / len(rets)
    var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def estimate_iv(closes, lookback=20):
    rv = realized_vol(closes, lookback)
    return max(IV_FLOOR, min(IV_CAP, rv * IV_PREMIUM))


if __name__ == "__main__":
    # sanity: ATM call, 30 DTE, 40% IV
    p = bs_price(100, 100, 30 / 365, 0.40, "call")
    d = bs_delta(100, 100, 30 / 365, 0.40, "call")
    print(f"ATM 30d call @40% IV: price={p:.2f} (expect ~4.6), delta={d:.3f} (expect ~0.53)")
    # theta check: same call, 5 DTE should be much cheaper
    p5 = bs_price(100, 100, 5 / 365, 0.40, "call")
    print(f"ATM 5d call @40% IV:  price={p5:.2f} (expect ~1.9) -> theta bleed visible")
    k = strike_for_delta(100, 120 / 365, 0.40, 0.60, "call")
    print(f"strike for 0.60-delta 120d call: {k:.2f} (expect slightly ITM, <100)")
