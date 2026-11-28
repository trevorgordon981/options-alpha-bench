# Byron Bench (`options-alpha-bench`)

A **synthetic Black-Scholes options simulation and model-evaluation harness** for
scoring OpenAI-compatible LLMs on structured options trade construction and
periodic portfolio decisions over daily underlying close series.

> **Important scope note:** Every option price and P&L figure produced by this
> repository is a **synthetic daily-close Black-Scholes simulation**, not
> executable historical option-chain P&L. Results are suitable for controlled,
> counterfactual model comparisons and sensitivity studies — never as proof of
> live tradable alpha or a pure single-Greek causal effect.

## Prerequisites

- **Python >= 3.11** (required for modern typing and stdlib behavior; stock macOS Python 3.9 is unsupported).
- Offline tests and `bench.py demo` require **no external packages, network access, or API keys**.
- Live market-data fetching (`run_ab.py` / uncached `bench.py run`) additionally requires `yfinance` (`pip install -r requirements.txt`).

## Reproducible offline demo (no network, no API keys)

Create an isolated Python 3.11+ virtual environment and run the deterministic synthetic demonstration on the pinned fixture (`fixtures/sample_prices.json`) to verify the paired DTE A/B simulation, Greeks and strike-drift decomposition, valid mock-book scoring, and invalid-run gating:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python bench.py demo
```

The demo prints the SHA-256 hash of the pinned fixture, setup denominator,
pricing decomposition, and both a qualifying (`verdict: "VALID"`) and an
all-failed (`verdict: "INVALID"`, `alpha_vs_spy_cagr_pct: null`) book readout.

## Measured Results

Verified locally in an isolated `Python 3.11.14` virtual environment (stdlib only, no network or API keys):

| Check / Command | Outcome | Wall Time |
|---|---|---|
| `python -m compileall -q .` | Passed | `0.072s` |
| `python -m unittest discover -s tests -p "test_*.py" -v` | **48 / 48 tests passed** (`0.159s` unittest) | `0.229s` |
| `python bench.py demo` | Passed (deterministic fixture `4131da77...2ea5e`) | `0.113s` |

Key verified outputs from `python bench.py demo` on `fixtures/sample_prices.json` (`n_setups = 22`, synthetic constant-IV Black-Scholes daily-close simulation):

- **Paired DTE A/B (`35 DTE` vs `8x` rule):**
  - `doctrine_35dte`: mean `-8.75%`, median `-36.28%`, win rate `13.6%` (`18` stop / `4` horizon exits)
  - `eightx_rule`: mean `-3.11%`, median `-9.86%`, win rate `27.3%` (`19` horizon / `3` stop exits)
  - Paired delta (`8x - 35 DTE`): mean `+5.63%`, median `+16.59%`, `8x` better on `86.4%` of setups
  - Pricing decomposition: median `|strike drift|` `3.57%` of spot, median entry cost ratio `2.278x`, median entry gamma `0.034891` (`35 DTE`) vs `0.015926` (`8x`), median daily theta `-0.0469` (`35 DTE`) vs `-0.014` (`8x`)
- **Mock Book — valid run (`book_valid_mock`):** `verdict: "VALID"`, coverage `100.0%` over `184` days (`every_14_days`), CAGR `-21.4%`, Sharpe `0.29`, max drawdown `69.2%`, cash CAGR `0.0%`, SPY buy-and-hold CAGR `-18.5%`, alpha vs SPY `-2.9%`.
- **Mock Book — all-failed run (`book_all_failed_mock`):** `verdict: "INVALID"` (`zero_valid_decisions (0/14)`), coverage `0.0%`, `14` transport errors, `alpha_vs_spy_cagr_pct: null`.

## The three tracks

| Track | What it evaluates | Module |
|---|---|---|
| **construction_alpha** | Mechanical momentum setups; model emits one strict JSON defined-risk trade per setup; scored under **common fixed stop (`-30%`) and time-stop (`3 DTE`) rules** against a paired 35-DTE baseline (optional `--allow-model-stop` labels policy variation explicitly). | `run_model_alpha.py` |
| **book_backtest** | Model manages a simulated `$100k` options book over a date window at a configured cadence (`--step`, `--mark-frequency step|daily`) with continuous daily-close `-30%` stop and `3-DTE` time-stop; reports coverage, deliberate abstentions, error breakdown, cash & SPY comparators, CAGR, Sharpe, and drawdown. | `book.py` |
| **doctrine** *(optional)* | Attach an external rubric percentage via `--doctrine <pct>`. | external |

## Running against a local or hosted model

```bash
# Local OpenAI-compatible server (normalized to /v1/chat/completions):
python bench.py run --name local_run --model <model-id> --base http://127.0.0.1:8000

# Hosted OpenRouter endpoint (normalized to /api/v1/chat/completions; auth verified on official HTTPS origin):
export OPENROUTER_API_KEY="your-key"
python bench.py run --name hosted_run --model <provider/model> --base https://openrouter.ai/api/v1

# Compare scorecards side by side (explicitly surfaces missing or incomparable cards):
python bench.py compare local_run hosted_run

# Paired synthetic DTE A/B counterfactual study (requires yfinance cache or live fetch):
python run_ab.py none
```

### Endpoint & authentication behavior (`llm_client.py`)

Both `run_model_alpha.py` and `book.py` share `llm_client.py`:
- Endpoints ending in `/v1` (such as `https://openrouter.ai/api/v1`) normalize to
  `.../v1/chat/completions` rather than duplicating `/v1/v1/`.
- `OPENROUTER_API_KEY` is attached automatically **only** for verified `https://openrouter.ai`
  / `https://api.openrouter.ai` origins (`OPENAI_API_KEY` only for `https://api.openai.com`),
  never via URL substring matching or over plain HTTP.

## Evaluation integrity guarantees

- **Coverage-gated validity & semantic action checks:** Runs with zero valid decisions
  or coverage below `--min-coverage` (default `80%`) are flagged `verdict: "INVALID"`
  (`qualifying: false`), and performance/alpha fields (`cagr_pct`, `sharpe`,
  `alpha_vs_spy_cagr_pct`, `alpha_vs_doctrine_pct`) are suppressed to `null`.
  Deliberate cash abstentions (`{"close": [], "open": []}`) are tracked separately from
  transport, parse, schema, and semantically invalid/impossible action failures (such as
  closing unknown/non-open position IDs or opening tickers outside the basket).
- **Strict typed schema & durable replay (`schema.py`):** Malformed JSON replies
  (wrong types, `NaN`/infinite numbers, booleans where integers are required,
  `open: [null]`, out-of-range deltas/sizes/DTEs) become explicit failed rows/steps
  instead of aborting a run or being silently coerced. Terminal case/step outcomes
  are appended and fsynced to JSONL checkpoints.
- **Chronological trading-session execution & exact-bar fills (`book.py`):** Decision
  schedules map nominal cadences strictly onto valid trading sessions within `[start, end]`
  (deferring weekend/holiday dates to the next valid session in window, or skipping
  cleanly when none remain), auto-exits and daily equity marks advance together along
  one chronological trading-session timeline (zero lookahead from future exits), and
  every close/liquidation enforces `exit_date >= entry_date`. Newly executed synthetic
  fills require an exact-date symbol close bar (`date == entry_date`); symbols missing a
  close on the decision session fail closed (`missing_entry_bar` / `stale_bar_rejected`)
  rather than filling on a stale prior-day close, while prior-bar carry-forward is
  restricted to non-fill mark-to-market and explicitly counted forced liquidations
  (`stale_liquidation_marks`). Final liquidation occurs on the final valid trading session
  and replaces the terminal mark cleanly; annualization uses the configured sampling
  frequency (`365.25 / step_days` for step marks, `252` for daily trading-session marks).
- **Frozen consumed-data provenance caching (`bench.py`, `data.py`):** Each track
  initializes a single frozen in-memory `PriceSnapshot` / `PriceSnapshotSession`
  **before** setup selection and model calls begin, threads that exact snapshot through
  setup generation, hydration, prompt construction, scoring, and simulation, and hashes
  the exact consumed in-memory price bytes into `data_digest` (immune to pre-hydration or
  mid-run mutation of on-disk cache files). Track caches bind model ID, optional
  `--model-revision`, endpoint base, run parameters, and SHA-256 digests of source code,
  prompt templates, and consumed prices, written atomically. `bench.py compare` flags
  unverified alias-only builds lacking `model_revision`.

## Methodology & simulation limits

All simulated outcomes carry the following explicit limitations (see `pricing.py`
and `engine.py`):

1. **Synthetic pricing, not option chains:** Option prices are modeled via
   European Black-Scholes from split/dividend-adjusted daily underlying closes
   (`data.py`), not historical bid/ask quotes or OPRA chains.
2. **Constant entry-IV proxy:** Implied volatility is approximated as
   `1.10 × trailing 20-day realized volatility` (clamped to `[0.15, 1.50]`) and
   held constant over the life of each trade. This omits implied-volatility skew,
   term structure, earnings IV crush, and vega P&L.
3. **Confounded DTE effects (not theta-only):** Repricing the same target delta
   (e.g., `0.60` delta) at `35 DTE` versus `168 DTE` selects **different strikes**
   and **different entry premiums**. Paired P&L differences therefore combine
   strike drift, dollar leverage, gamma/delta curvature, path dependency, and
   theta decay — not an isolated theta-only causal effect. `run_ab.py` reports
   entry strike drift, cost ratio, gamma, and daily theta alongside returns.
4. **Daily-close stop evaluation & gap-through risk:** Stops are checked only at
   daily closes. An overnight gap can fill materially worse than `-30%` (down to
   `-100%` on a large adverse gap).
5. **No execution friction or corporate-action nuances:** Simulations omit
   bid-ask spreads, slippage, commissions, assignment/pin risk, discrete
   dividends, and American early-exercise premia.
6. **Data provenance:** Live `yfinance` history can be revised retroactively by
   upstream providers. For reproducible comparisons, pin input caches and verify
   the provenance digests embedded in each `scorecard_<name>.json`.

## Running tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Files

- `llm_client.py` — centralized endpoint normalization and origin-checked auth
- `schema.py` — strict bounded typed validators and fsynced JSONL writer
- `pricing.py` — Black-Scholes price, delta, gamma, daily theta, and IV proxy
- `data.py` — inclusive daily-close fetcher/cache and fixture loader
- `market.py` — point-in-time market context snapshots
- `scenarios.py` — mechanical 20-day momentum setup generator
- `engine.py` — single-trade daily-close forward-walk simulator
- `run_ab.py` — paired DTE synthetic counterfactual study with pricing decomposition
- `run_model_alpha.py` — Track 1 (construction alpha)
- `book.py` — Track 2 (autonomous book simulator + metrics)
- `bench.py` — provenance-bound orchestrator, `compare`, and offline `demo`
- `expert_policy.py` / `run_book_expert.py` — deterministic non-LLM baseline policy
- `fixtures/sample_prices.json` — pinned synthetic daily closes for offline demos

## License

MIT, see [LICENSE](LICENSE).
