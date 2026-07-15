"""
Day 6 — LLM Layer: Plain-English Daily Summary

In simple terms: Days 4-5 gave us a model, its accuracy, and a backtest —
all useful, but not something a non-technical person would want to read.
Today we hand that same data to Claude and ask it to write a short,
honest, plain-English summary per ticker: what the model expects for the
next trading day, how much to trust it, and how it's performed so far.

The LLM is not told to be optimistic — it's explicitly asked to flag
low-confidence signals and mention when a ticker's model has no real edge,
so the summary stays honest about a first-pass model's limits.

Run this after src/day03_features.py has populated data/features/.
Output: printed plain-English summary, one paragraph per ticker.
"""

import glob
import os

import anthropic
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

FEATURE_COLUMNS = [
    "RSI14",
    "BB_Mid",
    "BB_Upper",
    "BB_Lower",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
]

# Same chronological split as Day 4/5.
TEST_FRACTION = 0.2

MODEL = "claude-opus-4-8"


def load_ticker_features(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def build_ticker_report(df: pd.DataFrame) -> dict:
    """Train on labeled history, evaluate on the held-out test set, and
    predict tomorrow's direction from the most recent row (the one row
    that has no Target yet, since there's no next day to compare against)."""
    labeled = df.dropna(subset=FEATURE_COLUMNS + ["Target"])

    split_at = int(len(labeled) * (1 - TEST_FRACTION))
    train, test = labeled.iloc[:split_at], labeled.iloc[split_at:]

    X_train, y_train = train[FEATURE_COLUMNS], train["Target"].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_test = test["Target"].astype(int)
    y_pred = model.predict(test[FEATURE_COLUMNS])
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    strategy_returns = test["Return"].where(y_pred == 1, 0.0)
    strategy_return = (1 + strategy_returns).prod() - 1
    buy_and_hold_return = (1 + test["Return"]).prod() - 1

    latest_row = df.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    next_day_prediction = model.predict(latest_row[FEATURE_COLUMNS].to_frame().T)[0]

    return {
        "as_of": latest_row.name.date().isoformat(),
        "latest_close": round(float(latest_row["Close"]), 2),
        "latest_rsi14": round(float(latest_row["RSI14"]), 1),
        "predicted_direction": "up" if next_day_prediction == 1 else "down",
        "test_days": len(test),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "strategy_return": round(strategy_return, 4),
        "buy_and_hold_return": round(buy_and_hold_return, 4),
    }


def build_prompt(reports: dict[str, dict]) -> str:
    lines = [
        "Here is model output for a small stock watchlist. For each ticker, "
        "write one short plain-English paragraph (3-4 sentences) covering: "
        "what the model predicts for the next trading day, how much to trust "
        "that prediction given its precision/recall on held-out data, and how "
        "the trading strategy has done versus simple buy-and-hold. Be honest: "
        "if precision or recall is weak (e.g. under ~55%) or the strategy "
        "didn't beat buy-and-hold, say so plainly instead of hedging around it. "
        "Do not give financial advice or tell the reader to buy/sell — describe "
        "what the model says and how reliable it has been.\n",
    ]
    for ticker, r in reports.items():
        lines.append(
            f"{ticker} (as of {r['as_of']}): close ${r['latest_close']}, "
            f"RSI14 {r['latest_rsi14']}, model predicts '{r['predicted_direction']}' "
            f"for the next trading day. On {r['test_days']} held-out test days: "
            f"precision {r['precision']:.0%}, recall {r['recall']:.0%}. "
            f"Backtest strategy return {r['strategy_return']:+.1%} vs. "
            f"buy-and-hold {r['buy_and_hold_return']:+.1%}."
        )
    return "\n".join(lines)


def main():
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.csv")))
    if not files:
        print("No feature files found in data/features/. Run src/day03_features.py first.")
        return

    print("MarketPulse AI — Day 6: turning model output into a plain-English daily summary\n")

    reports = {}
    for filepath in files:
        ticker = os.path.basename(filepath).replace("_features.csv", "")
        df = load_ticker_features(filepath)
        reports[ticker] = build_ticker_report(df)

    prompt = build_prompt(reports)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = next(block.text for block in response.content if block.type == "text")

    print(summary)
    print("\nDone. Next: automate the pipeline end-to-end (Day 7).")


if __name__ == "__main__":
    main()
