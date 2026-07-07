"""Substantiate the DTE doctrine (1-3 months for short-term, LEAPs for longer) with real
BS-marked outcomes on all 524 setups. Bull/bear DEBIT SPREADS, -30% stop, momentum direction
held fixed; only DTE varies. Proves which tenor buckets actually pay before we teach them."""
import statistics as st, collections
from engine import Trade, simulate
from scenarios import generate

STOP = 0.30
BUCKETS = [("short 14d", 14), ("short 21d", 21), ("1mo 30d", 30), ("2mo 60d", 60),
           ("3mo 90d", 90), ("mid 180d", 180), ("LEAP 365d", 365), ("LEAP 545d", 545), ("LEAP 730d", 730)]

def run():
    setups = generate()
    legs = {k: [] for k, _ in BUCKETS}; reasons = {k: collections.Counter() for k, _ in BUCKETS}
    for s in setups:
        kind = "call" if s["mom20"] > 0 else "put"
        d = f"{kind}_debit_spread"
        res = {}
        for tag, dte in BUCKETS:
            r = simulate(Trade(s["ticker"], s["entry_date"], d, dte, horizon_days=s["horizon_days"],
                               long_delta=0.60, short_delta=0.30, premium_stop=STOP))
            res[tag] = r
        if not all(r.ok for r in res.values()):
            continue
        for tag, _ in BUCKETS:
            legs[tag].append(res[tag].return_pct); reasons[tag][res[tag].exit_reason] += 1
    return legs, reasons

legs, reasons = run()
n = len(next(iter(legs.values())))
print(f"n={n} setups (both-ok across all buckets)\n")
print(f"{'DTE bucket':12}{'mean%':>8}{'median%':>9}{'win%':>7}{'total(u)':>10}  exits")
for tag, _ in BUCKETS:
    v = legs[tag]; wins = sum(1 for x in v if x > 0)
    ex = dict(reasons[tag])
    print(f"{tag:12}{100*st.mean(v):>+8.1f}{100*st.median(v):>+9.1f}{100*wins/len(v):>6.0f}%{sum(v):>+10.2f}  {ex}")
# paired: 1-3mo avg vs short avg vs LEAP avg
def avg(tags):
    import statistics; return statistics.mean([st.mean(legs[t]) for t in tags])
print("\n--- doctrine check (mean return-on-premium) ---")
print(f"  short (14-21d):   {100*avg(['short 14d','short 21d']):+.1f}%")
print(f"  1-3mo (30-90d):   {100*avg(['1mo 30d','2mo 60d','3mo 90d']):+.1f}%")
print(f"  mid (180d):       {100*st.mean(legs['mid 180d']):+.1f}%")
print(f"  LEAP (365-730d):  {100*avg(['LEAP 365d','LEAP 545d','LEAP 730d']):+.1f}%")
