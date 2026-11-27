"""
Black-Scholes option pricing + greeks, and an IV estimator.

This module provides a synthetic pricing model on daily underlying closes (not
historical option chains). It enables counterfactual paired experiments over
real or pinned underlying paths while keeping pricing assumptions explicit:

  - IV held constant over the hold at entry-IV (omits vega / IV crush & skew)
  - IV proxy = trailing realized vol * IV_PREMIUM, clamped to [IV_FLOOR, IV_CAP]
  - European-style BS (omits American early exercise and discrete dividends)
  - Flat risk-free rate r = R_FREE; zero bid-ask spread, slippage, or commissions
  - Holding target delta constant across different expiries changes strike,
    premium/leverage, gamma, and path sensitivity alongside theta.
"""
import math

R_FREE = 0.045          # flat risk-free (T-bill ballpark)
IV_PREMIUM = 1.10       # implied typically ~10% over trailing realized
IV_FLOOR = 0.15
IV_CAP = 1.50


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


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
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        # step function at expiry
        itm = (spot > strike) if kind == "call" else (spot < strike)
        return (1.0 if kind == "call" else -1.0) if itm else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0


def bs_gamma(spot, strike, t_years, iv, r=R_FREE):
    """Black-Scholes gamma (d^2V/dS^2, per share)."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    return _norm_pdf(d1) / (spot * iv * sqrt_t)


def bs_theta(spot, strike, t_years, iv, kind="call", r=R_FREE):
    """Black-Scholes theta per calendar day (dV/dt / 365, per share)."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    term1 = -(spot * _norm_pdf(d1) * iv) / (2.0 * sqrt_t)
    if kind == "call":
        annual = term1 - r * strike * math.exp(-r * t_years) * _norm_cdf(d2)
    else:
        annual = term1 + r * strike * math.exp(-r * t_years) * _norm_cdf(-d2)
    return annual / 365.0


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
    g = bs_gamma(100, 100, 30 / 365, 0.40)
    th = bs_theta(100, 100, 30 / 365, 0.40, "call")
    print(f"ATM 30d call @40% IV: price={p:.2f}, delta={d:.3f}, gamma={g:.4f}, theta/day={th:.4f}")
    p5 = bs_price(100, 100, 5 / 365, 0.40, "call")
    print(f"ATM 5d call @40% IV:  price={p5:.2f}")
    k = strike_for_delta(100, 120 / 365, 0.40, 0.60, "call")
    print(f"strike for 0.60-delta 120d call: {k:.2f}")
