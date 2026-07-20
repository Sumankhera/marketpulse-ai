"""
Day 7 — End-to-End Pipeline (now folding in Day 8's news sentiment)

In simple terms: Days 1-6 were separate scripts you had to run by hand,
in order, and Day 6 only printed its summary to the terminal. Today
wires them together into one command that pulls fresh data, rebuilds
the features, retrains the model, fetches each ticker's news sentiment
(Day 8), asks Claude for a summary that weighs both together, and
saves everything to a dated report file instead of just printing it.

Run: python src/day07_pipeline.py
Output: outputs/reports/YYYY-MM-DD.md (plus the same files day01/day03/day08
already write to data/, data/features/, and data/news/, refreshed with
today's data).
"""

import glob
import os
from datetime import datetime

from day01_data_ingestion import WATCHLIST, fetch_and_save
from day03_features import build_features, load_ticker_data
from day06_llm_summary import build_prompt, build_ticker_report
from day08_news_sentiment import fetch_headlines, get_sentiment, save_ticker_news

import anthropic

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")

MODEL = "claude-opus-4-8"


def run_ingestion() -> list[str]:
    """Day 1 step: pull fresh OHLCV data for the watchlist."""
    saved = []
    for ticker in WATCHLIST:
        try:
            saved.append(fetch_and_save(ticker))
        except Exception as e:
            print(f"  {ticker}: FAILED to fetch ({e})")
    return saved


def run_features() -> list[str]:
    """Day 3 step: rebuild the feature CSVs from whatever is in data/."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    written = []
    for filepath in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        df = load_ticker_data(filepath)
        features = build_features(df)
        out_path = os.path.join(FEATURES_DIR, f"{ticker}_features.csv")
        features.to_csv(out_path)
        written.append(out_path)
    return written


def run_model_and_llm() -> tuple[dict, str]:
    """Day 4/5/6/8 step: retrain per ticker, predict, fetch+score news sentiment,
    and ask Claude to summarize both together."""
    client = anthropic.Anthropic()
    reports = {}
    for filepath in sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.csv"))):
        ticker = os.path.basename(filepath).replace("_features.csv", "")
        df = load_ticker_data(filepath)
        reports[ticker] = build_ticker_report(df)

        try:
            headlines = fetch_headlines(ticker)
            sentiment = get_sentiment(ticker, headlines, client)
        except Exception as e:
            headlines, sentiment = [], {"label": "neutral", "rationale": f"news fetch failed ({e})"}
        save_ticker_news(ticker, headlines, sentiment)
        reports[ticker]["news_label"] = sentiment["label"]
        reports[ticker]["news_rationale"] = sentiment["rationale"]

    prompt = build_prompt(reports)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = next(block.text for block in response.content if block.type == "text")
    return reports, summary


def write_report(reports: dict, summary: str) -> str:
    today = datetime.now().date().isoformat()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"{today}.md")

    lines = [f"# MarketPulse AI — Daily Report ({today})\n"]
    lines.append(summary.strip() + "\n")
    lines.append("## Raw model output\n")
    lines.append("| Ticker | Close | RSI14 | Predicted | Precision | Recall | Strategy | Buy & Hold | News |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ticker, r in reports.items():
        lines.append(
            f"| {ticker} | ${r['latest_close']} | {r['latest_rsi14']} | "
            f"{r['predicted_direction']} | {r['precision']:.0%} | {r['recall']:.0%} | "
            f"{r['strategy_return']:+.1%} | {r['buy_and_hold_return']:+.1%} | "
            f"{r.get('news_label', 'n/a')} |"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    print(f"MarketPulse AI — Day 7: end-to-end pipeline ({datetime.now().date()})\n")

    print("Step 1/3: pulling fresh data...")
    saved = run_ingestion()
    print(f"  {len(saved)}/{len(WATCHLIST)} tickers fetched\n")

    print("Step 2/3: rebuilding features...")
    written = run_features()
    print(f"  {len(written)} feature files written\n")

    print("Step 3/3: retraining models, fetching news sentiment, and generating the LLM summary...")
    reports, summary = run_model_and_llm()

    out_path = write_report(reports, summary)
    print(f"\nDone. Report saved to {out_path}")
    print("Next: consider scheduling this to run automatically each trading day.")


if __name__ == "__main__":
    main()
