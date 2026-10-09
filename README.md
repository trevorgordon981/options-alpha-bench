# Byron Bench

Point it at any served model → get a **trading scorecard**. Not a rubric vibe-check —
it measures whether a model's *actual constructed trades make money*, walked forward on
**real** price history. Built to answer: *which model should run as your trading assistant?*

## The three tracks

| track | what it measures | file |
|---|---|---|
| **construction_alpha** | 524 mechanical setups (no hindsight); model builds ONE trade each; realized return-on-premium + the DTE it actually picks | `run_model_alpha.py` |
| **book_backtest** | model runs a $100k options book over 2024–2026 (weekly decisions, continuous −30% stop + 3-DTE time-stop); CAGR, Sharpe, max drawdown, hit rate, alpha vs SPY buy-hold | `book.py` |
| **doctrine** *(optional)* | attach an external rubric score via `--doctrine <pct>` | (bring your own) |

## How options P&L is computed (honest about the model)
Free historical option chains are unreliable, so every option is **Black-Scholes-marked
on the REAL underlying close each day** — IV from the underlying's own trailing realized
vol, held constant over the hold (isolates delta+theta, no vega noise). This is faithful
for the question that matters — theta/DTE — which is exactly where short-DTE trades bleed.
Assumptions live at the top of `pricing.py`.

Why "mechanical" setups: direction is assigned by a public 20-day momentum rule using only
data available at entry. No cherry-picking → the alpha number is defensible.

## Run it

```bash
# full scorecard for a served model (reuses any cached track outputs):
python bench.py run --name myrun --model <path-or-id> --base http://127.0.0.1:8000
# just the DTE-doctrine backtest (no model needed — proves longer DTE = alpha):
python run_ab.py none
# compare models side by side:
python bench.py compare myrun otherrun
```

Any OpenAI-compatible `/v1/chat/completions` endpoint works: a local server, or a hosted
provider (e.g. `--base https://openrouter.ai/api/v1 --model ...`, set `OPENROUTER_API_KEY`).
Price history is cached yfinance closes (`data.py`) — first run fetches, then reuses.

## Key finding baked in
- **DTE A/B (524 setups):** same trade at a short 35-DTE vs a longer ~168-DTE (an "8×
  the trade window" rule). The longer DTE wins **66.6%** of the time, **+10.4 pt median** —
  the whole gap is theta: the short-DTE version gets stopped out **67%** of the time vs 30%.
  → **longer DTE is real, measurable alpha**, not just doctrine.
- **The point of the bench:** one baseline model we tested scored **83.6% "good"** on a
  rubric — but run as a real book (2024→2026) it posted **−3.4% CAGR, −0.04 Sharpe, 40.7%
  max drawdown, −22.6 pt alpha vs SPY buy-hold** (SPY +19.1% CAGR). It *lost money while the
  market ran up 40%+*. That gap — looks-good-on-a-rubric vs actually-makes-money — is exactly
  why this bench exists.

## Files
`pricing.py` BS + greeks · `data.py` cached yfinance closes · `market.py` real context
snapshots · `scenarios.py` mechanical setup generator · `engine.py` single-trade
forward-walk · `run_ab.py` DTE A/B · `run_model_alpha.py` track 1 (construction alpha) ·
`book.py` track 2 (book backtest) · `bench.py` orchestrator + `compare` ·
`expert_policy.py` / `run_book_expert.py` fixed-policy (non-LLM) baselines.

## The trade contract
`run_model_alpha.py` prompts the model for a single trade as JSON and parses fields like
`direction`, `entry_dte`, `long_delta`, `stop_pct`, `conviction`. Adapt the prompt/parse in
`call_model` / the parsing helper to match whatever schema your model emits.

## License

MIT, see [LICENSE](LICENSE).
