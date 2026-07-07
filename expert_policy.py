"""
Disciplined RULES-EXPERT policy for the autonomous options book (book.py).

This is the "expert teacher" whose (book-state -> decision) trace we distill into
training rows for Alfred. It implements Trevor's stated doctrine EXACTLY:

  - Conviction-scaled sizing (base target by conviction 10/8->15%, 5->10%, 4->8%,
    3->6% of net-liq). NEVER a blanket per-trade cap.
  - Total-deployment CEILING ~57% of net-liq (keep ~40% cash) -- never open past it;
    push toward the ceiling only when multiple genuine A+ setups align, else hold cash.
  - Correlation cap 20% net-liq per correlated cluster (sector/factor groups = one bet).
  - 8x-DTE entries (~168 DTE for the ~21-day cadence) so theta doesn't bleed you.
  - -30% stop + 3-DTE time-stop are enforced by book.py; the policy manages
    opens/closes and does NOT fight the stops.
  - De-risk after real drawdown (>=~20% off peak): smaller/fewer opens, raise cash.
  - Circuit-breaker on a correlated vol spike: go to cash (no new opens).
  - Sit-out discipline: no forced opens when nothing qualifies. Cash is a position.

expert_decision(state) -> (decision_json, reasoning_str)

`state` is built by the driver from book.py's REAL live state each week and carries:
  iso, equity, cash, peak_equity, drawdown, deployment_frac, n_open, max_concurrent,
  positions:[{pid,ticker,direction,dte,pnl_pct,mktval_frac,cluster}],
  spy_r20, regime, slate:[{ticker,spot,ret_5d,ret_20d,pct_below_high,rvol,cluster}]
"""

# --- correlated clusters: treat each group as ONE bet for the 20% cap ---
CLUSTER = {
    "NVDA": "semis", "AMD": "semis", "AVGO": "semis", "MU": "semis",
    "AAPL": "megacap", "MSFT": "megacap", "GOOGL": "megacap", "META": "megacap",
    "AMZN": "megacap", "NFLX": "megacap", "CRM": "megacap",
    "SPY": "index", "QQQ": "index",
    "TSLA": "highbeta", "COIN": "highbeta",
}

import os as _os
def _p(name, default):
    return float(_os.environ.get(name, default))

# doctrine anchors (env-tunable for iteration; defaults are the CORRECTED canonical
# constants: deploy ceiling ~57% net-liq / ~40% cash, and a 20% per-cluster cap.)
DEPLOY_CAP = _p("EX_DEPLOY_CAP", 0.57)     # total premium-at-risk vs net-liq CEILING (~57%, keep ~40% cash)
CLUSTER_CAP = _p("EX_CLUSTER_CAP", 0.20)   # per correlated cluster vs net-liq (20% doctrine)
ENTRY_DTE = int(_p("EX_DTE", 168))         # 8x the ~21-day trade window
LONG_DELTA = 0.60          # slightly-ITM long leg, clean defined risk
DD_DERISK = _p("EX_DD_DERISK", 0.12)       # book down this much off peak -> de-risk
MIN_THESIS = 0.05          # |20d move| under this = no clear thesis, skip
VOL_SPIKE_RVOL = 0.55      # median slate rvol above this = correlated vol spike
WINNER_KEEP = _p("EX_WINNER_KEEP", 0.0)    # in the risk-off cash gate, keep only positions up >= this % (green); cut the rest
TAKE_PROFIT = _p("EX_TAKE_PROFIT", 75.0)   # lock a position up >= this %
TRAIL_ARM = _p("EX_TRAIL_ARM", 30.0)       # once a position has peaked >= this %, protect it
TRAIL_GIVEBACK = _p("EX_TRAIL_GIVEBACK", 0.40)  # close if it gives back this frac of peak gain

# conviction -> base target size (% net-liq). Discrete doctrine anchors.
SIZE_BY_CONV = {10: 0.15, 8: 0.15, 5: 0.10, 4: 0.08, 3: 0.06}


def cluster_of(ticker):
    return CLUSTER.get(ticker.upper(), "other")


def conviction(c):
    """Conviction proxy 3..10 from momentum strength + 5d/20d alignment.
    0 = no thesis (skip). Higher when the 20d trend is strong AND the last week
    confirms it (5d in the same direction)."""
    r20, r5 = c["ret_20d"], c["ret_5d"]
    a = abs(r20)
    if a < MIN_THESIS:
        return 0
    aligned = (r20 >= 0) == (r5 >= 0)
    if a >= 0.15 and aligned:
        return 10
    if a >= 0.12 and aligned:
        return 8
    if a >= 0.09:
        return 5
    if a >= 0.07:
        return 4
    return 3


def regime_of(spy_r20, slate):
    """Coarse regime from SPY 20d + slate breadth/vol. Drives long/put gating and
    the circuit-breaker."""
    rvols = sorted(c["rvol"] for c in slate) if slate else [0.3]
    med_rvol = rvols[len(rvols) // 2]
    neg_breadth = (sum(1 for c in slate if c["ret_20d"] < 0) / len(slate)) if slate else 0.5
    # circuit-breaker: correlated vol spike -> cash
    if med_rvol >= VOL_SPIKE_RVOL or spy_r20 <= -0.09:
        return "vol_spike", med_rvol, neg_breadth
    if spy_r20 >= 0.02:
        return "risk_on", med_rvol, neg_breadth
    if spy_r20 <= -0.04:
        return "risk_off", med_rvol, neg_breadth
    return "neutral", med_rvol, neg_breadth


import os as _os_struct
_STRUCTURE = _os_struct.environ.get("EX_STRUCTURE", "naked")   # "naked" | "spread" | "gated"
_GATE_CONV = int(_os_struct.environ.get("EX_GATE_CONV", "8"))  # gated: conv>=this -> naked LEAP (A+ tail), else bull spread

def _apply_structure(direction, conv):
    """Pick naked long vs bull debit spread. naked/spread fixed; GATED = conviction-scaled:
    A+ conviction keeps the uncapped naked LEAP, everything else takes a cheaper spread
    (enter the LEAP for less money). Long-dated either way."""
    if direction is None:
        return None
    kind = "call" if "call" in direction else "put"
    if _STRUCTURE == "spread":
        return f"{kind}_debit_spread"
    if _STRUCTURE == "gated":
        return f"long_{kind}" if conv >= _GATE_CONV else f"{kind}_debit_spread"
    return f"long_{kind}"

def _direction_for(c, regime):
    """Momentum sign -> option KIND (call/put), gated by regime (puts only in a
    confirmed downtrend). Structure (naked vs spread) is applied later by conviction."""
    if c["ret_20d"] > 0:
        return "long_call"
    if regime == "risk_off":
        return "long_put"
    return None


def expert_decision(state):
    equity = state["equity"]
    positions = state["positions"]
    slate = state["slate"]
    dd = state["drawdown"]
    deploy = state["deployment_frac"]
    n_open = state["n_open"]
    max_conc = state["max_concurrent"]
    spy_r20 = state.get("spy_r20", 0.0)

    regime, med_rvol, neg_breadth = regime_of(spy_r20, slate)
    notes = []
    closes = []

    # when the BOOK itself is in drawdown, protect winners harder (lock earlier)
    derisk = dd >= DD_DERISK
    tp = TAKE_PROFIT * 0.7 if derisk else TAKE_PROFIT
    tg = min(TRAIL_GIVEBACK, 0.30) if derisk else TRAIL_GIVEBACK
    arm = TRAIL_ARM * 0.7 if derisk else TRAIL_ARM

    # current cluster exposure (by market value fraction of net-liq)
    cluster_exp = {}
    open_tickers = set()
    for p in positions:
        cluster_exp[p["cluster"]] = cluster_exp.get(p["cluster"], 0.0) + p["mktval_frac"]
        open_tickers.add(p["ticker"])

    # ---- discretionary CLOSES ----
    slate_by_tk = {c["ticker"]: c for c in slate}
    for p in positions:
        pnl = p["pnl_pct"]
        peak = p.get("peak_pnl_pct", pnl)
        # (1) take profit: lock an outsized winner instead of round-tripping it to the stop
        if pnl >= tp:
            closes.append(p["pid"])
            notes.append(f"TAKE PROFIT {p['pid']} {p['ticker']}: +{pnl:.0f}% — lock the multiple, "
                         f"don't round-trip a 2x+ back to the -30% stop.")
            continue
        # (2) trailing give-back: a big winner that has surrendered part of its peak gain
        if peak >= arm and pnl <= peak * tg:
            closes.append(p["pid"])
            notes.append(f"TRAIL-STOP {p['pid']} {p['ticker']}: peaked +{peak:.0f}%, now +{pnl:.0f}% "
                         f"(gave back half the run) — protect the gain.")
            continue
        # (3) thesis invalidated (momentum flipped hard against the position)
        c = slate_by_tk.get(p["ticker"])
        if c:
            r20 = c["ret_20d"]
            flipped = (p["direction"] == "long_call" and r20 <= -0.07) or \
                      (p["direction"] == "long_put" and r20 >= 0.07)
            if flipped:
                closes.append(p["pid"])
                notes.append(f"CLOSE {p['pid']} {p['ticker']}: thesis dead — 20d flipped to "
                             f"{r20*100:+.0f}% against the {p['direction']}; don't wait for the -30% stop.")

    # ---- CONFIRMED-RISK-OFF CASH GATE ----------------------------------------
    # Two independent confirmations of risk-off both route to the same action:
    # stop opening new premium, move toward heavy cash, hold only existing winners
    # (their -30%/time stops manage them). Resume new opens ONLY when risk-on
    # re-confirms (SPY back above its trend filter and no vol spike).
    #   (a) correlated vol spike (fast panic) — median slate rvol / SPY 20d crash.
    #   (b) trend break — SPY below its N-day SMA (a simple, general trend filter,
    #       NOT hardcoded crash dates). This is the DD-killer the frontier run used:
    #       sit in cash through confirmed downtrends instead of deploying into them.
    spy_below_ma = state.get("spy_below_ma", False)
    spy_ma_days = state.get("spy_ma_days", 0)
    spy_ma_gap = state.get("spy_ma_gap", 0.0)
    if regime == "vol_spike":
        notes.append(f"CIRCUIT-BREAKER: correlated vol spike (median slate rvol "
                     f"{med_rvol:.0%}, SPY 20d {spy_r20*100:+.0f}%). No new opens — "
                     f"stops manage the book, cash is the position.")
        reasoning = _render_reasoning(state, regime, deploy, dd, closes, [], notes)
        return {"close": closes, "open": []}, reasoning
    if spy_below_ma:
        # move toward HEAVY CASH: cut every position that isn't a clear winner
        # (flat/red premium bleeds to its -30% stop in a downtrend — take the small
        # loss now, don't ride it down). Keep genuine winners riding on their stops /
        # trail (the profit-mgmt pass above already protects those).
        cut = []
        for p in positions:
            if p["pid"] in closes:
                continue
            if p["pnl_pct"] <= WINNER_KEEP:
                cut.append(p["pid"])
        if cut:
            notes.append(f"CASH-RAISE: closing {len(cut)} non-winner "
                         f"position(s) (P&L <= +{WINNER_KEEP:.0f}%) to raise cash into the "
                         f"downtrend — take the small loss, don't ride it to the -30% stop.")
        closes = list(closes) + cut
        notes.append(f"RISK-OFF CASH GATE: SPY is {abs(spy_ma_gap)*100:.0f}% BELOW its "
                     f"{spy_ma_days}d trend (confirmed downtrend). No new opens — heavy "
                     f"cash, hold only existing winners on their stops. Resume opening "
                     f"only when SPY reclaims its {spy_ma_days}d trend (risk-on).")
        reasoning = _render_reasoning(state, "risk_off_cash_gate", deploy, dd, closes, [], notes)
        return {"close": closes, "open": []}, reasoning

    # ---- de-risk multiplier after real drawdown ----
    size_mult = 1.0
    if derisk:
        size_mult = 0.5
        notes.append(f"DE-RISK: book down {dd*100:.0f}% off peak (>= {DD_DERISK*100:.0f}%). "
                     f"Halving new sizes, raising cash.")

    # available deployment headroom (account for positions we're about to close)
    freed = sum(p["mktval_frac"] for p in positions if p["pid"] in closes)
    deploy_now = max(0.0, deploy - freed)
    for cl in list(cluster_exp):
        cluster_exp[cl] -= sum(p["mktval_frac"] for p in positions
                               if p["pid"] in closes and p["cluster"] == cl)

    opens = []
    n_after_close = n_open - len(closes)
    max_new = max_conc - n_after_close
    if derisk:
        max_new = min(max_new, 2)   # fewer opens when wounded

    # rank candidates by conviction (slate already ~sorted by |momentum|)
    ranked = []
    for c in slate:
        if c["ticker"] in open_tickers:
            continue                # already hold it — one bet per name
        direction = _direction_for(c, regime)
        if direction is None:
            continue
        conv = conviction(c)
        if conv == 0:
            continue
        # neutral regime: only take high-conviction longs
        if regime == "neutral" and conv < 5:
            continue
        ranked.append((conv, c, direction))
    ranked.sort(key=lambda x: -x[0])

    for conv, c, direction in ranked:
        if len(opens) >= max_new:
            break
        cl = c["cluster"]
        target = SIZE_BY_CONV[conv] * size_mult
        # deployment cap
        room_deploy = DEPLOY_CAP - deploy_now
        if room_deploy <= 0.02:
            break
        # correlation cap
        room_cluster = CLUSTER_CAP - cluster_exp.get(cl, 0.0)
        if room_cluster <= 0.02:
            notes.append(f"SKIP {c['ticker']}: {cl} cluster already ~{cluster_exp.get(cl,0)*100:.0f}% "
                         f"of net-liq (>= {CLUSTER_CAP*100:.0f}% cap) — same bet, no double-up.")
            continue
        size = round(min(target, room_deploy, room_cluster), 3)
        if size < 0.03:
            continue
        direction = _apply_structure(direction, conv)
        opens.append({"ticker": c["ticker"], "direction": direction,
                      "entry_dte": ENTRY_DTE, "long_delta": LONG_DELTA, "size_pct": size})
        deploy_now += size
        cluster_exp[cl] = cluster_exp.get(cl, 0.0) + size
        notes.append(f"OPEN {c['ticker']} {direction} {size*100:.0f}% @ {ENTRY_DTE}DTE (8x rule): "
                     f"conv {conv} (20d {c['ret_20d']*100:+.0f}%, 5d {c['ret_5d']*100:+.0f}%); "
                     f"base {SIZE_BY_CONV[conv]*100:.0f}%"
                     + (" x0.5 de-risk" if derisk else "")
                     + f", fits deploy (now {deploy_now*100:.0f}%/{DEPLOY_CAP*100:.0f}%) & {cl} cap.")

    if not opens and regime != "vol_spike":
        why = []
        if n_after_close >= max_conc:
            why.append("at max concurrent")
        if deploy_now >= DEPLOY_CAP - 0.02:
            why.append(f"deployment ~{deploy_now*100:.0f}% at the {DEPLOY_CAP*100:.0f}% cap")
        if not ranked:
            why.append("no candidate clears the thesis/regime gate")
        notes.append("SIT OUT: " + (", ".join(why) if why else "nothing qualifies") +
                     ". Cash is a position — no forced trades.")

    reasoning = _render_reasoning(state, regime, deploy_now, dd, closes, opens, notes)
    return {"close": closes, "open": opens}, reasoning


def _render_reasoning(state, regime, deploy_after, dd, closes, opens, notes):
    head = (f"Equity ${state['equity']:,.0f}, cash {state['cash']/state['equity']*100:.0f}%, "
            f"deployed {state['deployment_frac']*100:.0f}%->{deploy_after*100:.0f}% of net-liq "
            f"(cap {DEPLOY_CAP*100:.0f}%), drawdown {dd*100:.0f}%. Regime: {regime} "
            f"(SPY 20d {state.get('spy_r20',0)*100:+.0f}%).")
    body = " ".join(notes) if notes else "Hold — book is within all caps, let winners run to the stops."
    return head + " " + body
