# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-11-27

### Added
- **Three-track options evaluation harness (`bench.py`, `run_model_alpha.py`, `book.py`):**
  - `construction_alpha` track scoring structured JSON defined-risk options trades against a paired 35-DTE baseline under common `-30%` stop and `3-DTE` time-stop rules.
  - `book_backtest` track simulating a `$100k` options book over configurable cadences (`--step`, `--mark-frequency step|daily`) with cash and SPY buy-and-hold comparators, CAGR, Sharpe, and drawdown metrics.
  - Deterministic offline demo (`python bench.py demo`) over pinned `fixtures/sample_prices.json` with paired DTE A/B pricing decomposition, valid mock-book scoring, and invalid-run gating.
- **Evaluation integrity & provenance:**
  - Coverage-gated validity (`verdict: "INVALID"`, `qualifying: false`, nullified alpha/performance fields) and distinct tracking of deliberate cash abstentions vs transport, parse, schema, and semantic action errors.
  - Strict typed JSON schema validation and fsynced JSONL replay checkpoints (`schema.py`).
  - Chronological trading-session decision scheduling, `exit_date >= entry_date` enforcement, and exact-date symbol close-bar requirement for new synthetic fills (`book.py`).
  - Frozen in-memory `PriceSnapshot` / `PriceSnapshotSession` bound before setup selection and hashed into `data_digest`, plus provenance-keyed atomic track caches (`bench.py`, `data.py`).
  - Centralized endpoint normalization and origin-verified `OPENROUTER_API_KEY` / `OPENAI_API_KEY` attachment (`llm_client.py`).
- **Verification & documentation:**
  - 48 offline unit tests covering cache integrity, endpoint auth, schema replay, validity gating, metrics annualization, chronology, exact-bar execution, and frozen data provenance, wired into GitHub Actions CI.
  - Explicit `Python >= 3.11` quickstart instructions and verified measured results in `README.md`.
