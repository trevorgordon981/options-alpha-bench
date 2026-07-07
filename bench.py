"""
Byron Bench — point it at any served model, get a trading scorecard.

Three tracks (a model-picker, not a vibe check):
  1. construction_alpha : 524 mechanical setups; model builds one trade each; realized
                          return-on-premium + the DTE it actually chooses.
  2. book_backtest      : model runs a $100k options book over 2024-2026; CAGR, Sharpe,
                          max drawdown, hit rate, alpha vs SPY buy-hold.
  3. doctrine (optional): gordon-gauntlet rubric score, attached if provided.

Usage:
  bench.py run --name walter_v1 --model <path> --base http://127.0.0.1:8082
  bench.py compare walter_v1 m3_prod ...
Cached track outputs are reused so re-runs are cheap.
"""
import argparse
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def run(name, base, model, start, end, step, skip_construction=False, skip_book=False,
        doctrine_score=None):
    from run_model_alpha import score as construction_score, summarize as construction_summ
    from scenarios import generate
    from book import Book

    card = {"name": name, "model": model, "base": base}

    # --- track 1: construction alpha (reuse cache if present) ---
    ctag = f"model_alpha_{name}"
    csum = _load(os.path.join(HERE, f"{ctag}_summary.json"))
    if csum is None and not skip_construction:
        setups = generate()
        rows = construction_score(setups, base, model)
        csum = construction_summ(rows)
        json.dump(rows, open(os.path.join(HERE, f"{ctag}_rows.json"), "w"), indent=2)
        json.dump(csum, open(os.path.join(HERE, f"{ctag}_summary.json"), "w"), indent=2)
    card["construction_alpha"] = csum

    # --- track 2: book backtest (reuse cache if present) ---
    btag = f"book_{name}"
    bsum = _load(os.path.join(HERE, f"{btag}_metrics.json"))
    if bsum is None and not skip_book:
        b = Book(base, model, start, end, step)
        bsum = b.run()
        json.dump(bsum, open(os.path.join(HERE, f"{btag}_metrics.json"), "w"), indent=2)
        json.dump(b.equity_curve, open(os.path.join(HERE, f"{btag}_equity.json"), "w"))
    card["book_backtest"] = bsum

    # --- track 3: doctrine rubric (attach if supplied) ---
    if doctrine_score is not None:
        card["doctrine_pct"] = doctrine_score

    card["headline"] = _headline(card)
    json.dump(card, open(os.path.join(HERE, f"scorecard_{name}.json"), "w"), indent=2)
    return card


def _headline(card):
    ca = card.get("construction_alpha") or {}
    bb = card.get("book_backtest") or {}
    return {
        "book_cagr_pct": bb.get("cagr_pct"),
        "book_sharpe": bb.get("sharpe"),
        "book_max_dd_pct": bb.get("max_drawdown_pct"),
        "book_alpha_vs_spy_pct": bb.get("alpha_vs_spy_cagr_pct"),
        "construction_mean_ret_pct": ca.get("model_mean_ret_pct"),
        "construction_median_dte": ca.get("model_median_dte"),
        "construction_dte_ge120_pct": ca.get("model_dte_ge_120_pct"),
        "doctrine_pct": card.get("doctrine_pct"),
    }


def compare(names):
    cards = [(_load(os.path.join(HERE, f"scorecard_{n}.json"))) for n in names]
    cards = [c for c in cards if c]
    cols = [("model", "name"), ("bookCAGR%", "book_cagr_pct"), ("Sharpe", "book_sharpe"),
            ("maxDD%", "book_max_dd_pct"), ("αvsSPY%", "book_alpha_vs_spy_pct"),
            ("constRet%", "construction_mean_ret_pct"), ("medDTE", "construction_median_dte"),
            ("DTE≥120%", "construction_dte_ge120_pct"), ("doctrine%", "doctrine_pct")]
    w = {h: max(len(h), 9) for h, _ in cols}
    print("  ".join(h.ljust(w[h]) for h, _ in cols))
    print("  ".join("-" * w[h] for h, _ in cols))
    for c in cards:
        h = c.get("headline", {})
        row = []
        for head, key in cols:
            v = c["name"] if key == "name" else h.get(key)
            row.append(("" if v is None else str(v)).ljust(w[head]))
        print("  ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--base", default="http://127.0.0.1:8082")
    r.add_argument("--model", default="/Users/trevorgordon/models/gemma-4-31b-derefused-a03nt-8bit")
    r.add_argument("--start", default="2024-02-01")
    r.add_argument("--end", default="2026-04-15")
    r.add_argument("--step", type=int, default=7)
    r.add_argument("--skip-construction", action="store_true")
    r.add_argument("--skip-book", action="store_true")
    r.add_argument("--doctrine", type=float, default=None)
    c = sub.add_parser("compare")
    c.add_argument("names", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        card = run(a.name, a.base, a.model, a.start, a.end, a.step,
                   a.skip_construction, a.skip_book, a.doctrine)
        print(json.dumps(card["headline"], indent=2))
    else:
        compare(a.names)
