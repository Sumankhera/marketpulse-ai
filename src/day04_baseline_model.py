"""
Day 4 — Baseline ML Model

In simple terms: now that Day 3 gave us "clues" (RSI, Bollinger Bands,
past returns) and an "answer key" (did the price go up or down the
next day?), we train a simple model to guess the answer from the
clues alone, then check how often it's right on data it hasn't seen.

We compare it against the "dumb" baseline of always guessing the more
common outcome — if our model can't beat that, it isn't learning
anything useful.

Run this after src/day03_features.py has populated data/features/.
Output: printed accuracy report per ticker.
"""

import glob
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Fraction of rows (in time order) held out as the test set. Chronological,
# not random — a model that predicts the past from the future doesn't count.
TEST_FRACTION = 0.2


def load_ticker_features(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def train_and_evaluate(df: pd.DataFrame) -> dict:
    ready = df.dropna(subset=FEATURE_COLUMNS + ["Target"])

    split_at = int(len(ready) * (1 - TEST_FRACTION))
    train, test = ready.iloc[:split_at], ready.iloc[split_at:]

    X_train, y_train = train[FEATURE_COLUMNS], train["Target"].astype(int)
    X_test, y_test = test[FEATURE_COLUMNS], test["Target"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    model_accuracy = accuracy_score(y_test, predictions)
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())

    return {
        "rows_train": len(train),
        "rows_test": len(test),
        "model_accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
    }


def main():
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.csv")))
    if not files:
        print("No feature files found in data/features/. Run src/day03_features.py first.")
        return

    print("MarketPulse AI — Day 4: training a baseline up/down predictor\n")

    for filepath in files:
        ticker = os.path.basename(filepath).replace("_features.csv", "")
        df = load_ticker_features(filepath)
        result = train_and_evaluate(df)

        beat_baseline = "yes" if result["model_accuracy"] > result["baseline_accuracy"] else "no"
        print(f"  {ticker}: trained on {result['rows_train']} days, tested on {result['rows_test']} days")
        print(f"           model accuracy:    {result['model_accuracy']:.1%}")
        print(f"           baseline accuracy:  {result['baseline_accuracy']:.1%} (always guess the more common direction)")
        print(f"           beats baseline: {beat_baseline}\n")

    print("Done. Next: model evaluation + a simple backtest (Day 5).")


if __name__ == "__main__":
    main()
