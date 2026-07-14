"""
Day 5 — Model Evaluation + Simple Backtest

In simple terms: Day 4 told us how often the model guessed the right
direction. That's not the same as whether it would have made money.
Today we check both:
  - Precision/recall, not just accuracy (accuracy alone can hide a
    model that's only good at calling "up" and useless at calling "down").
  - A backtest: pretend we traded on the model's calls during the test
    period, and compare the result to simply buying and holding.

Run this after src/day03_features.py has populated data/features/.
Output: printed evaluation + backtest report per ticker.
"""

import glob
import os

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

# Same chronological split as Day 4 — the test set here must match the one
# the model was evaluated on, so the backtest reflects the same "unseen" days.
TEST_FRACTION = 0.2


def load_ticker_features(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Train on the first 80% of days, return the test-set rows with predictions attached."""
    ready = df.dropna(subset=FEATURE_COLUMNS + ["Target"])

    split_at = int(len(ready) * (1 - TEST_FRACTION))
    train, test = ready.iloc[:split_at], ready.iloc[split_at:]

    X_train, y_train = train[FEATURE_COLUMNS], train["Target"].astype(int)
    X_test = test[FEATURE_COLUMNS]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    test = test.copy()
    test["Predicted"] = model.predict(X_test)
    return test


def backtest(test: pd.DataFrame) -> dict:
    """
    Strategy: on days the model predicts "up", hold the stock (earn that
    day's actual return). On days it predicts "down", sit in cash (earn 0).
    Compare the compounded result to simply buying and holding the whole
    test period.
    """
    strategy_returns = test["Return"].where(test["Predicted"] == 1, 0.0)

    strategy_total_return = (1 + strategy_returns).prod() - 1
    buy_and_hold_return = (1 + test["Return"]).prod() - 1

    return {
        "strategy_return": strategy_total_return,
        "buy_and_hold_return": buy_and_hold_return,
    }


def main():
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.csv")))
    if not files:
        print("No feature files found in data/features/. Run src/day03_features.py first.")
        return

    print("MarketPulse AI — Day 5: evaluating the model beyond accuracy + a simple backtest\n")

    for filepath in files:
        ticker = os.path.basename(filepath).replace("_features.csv", "")
        df = load_ticker_features(filepath)
        test = train_and_predict(df)

        y_true = test["Target"].astype(int)
        y_pred = test["Predicted"]

        # zero_division=0: if the model never predicts one class in a short
        # test window, report 0 instead of crashing.
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        result = backtest(test)

        beat_market = "yes" if result["strategy_return"] > result["buy_and_hold_return"] else "no"
        print(f"  {ticker}: {len(test)} test days")
        print(f"           precision: {precision:.1%} (of the days it called 'up', how often it was right)")
        print(f"           recall:    {recall:.1%} (of the days that were actually 'up', how many it caught)")
        print(f"           strategy return:     {result['strategy_return']:+.1%}")
        print(f"           buy-and-hold return: {result['buy_and_hold_return']:+.1%}")
        print(f"           beat buy-and-hold: {beat_market}\n")

    print("Done. Next: LLM layer — turn model output + data into a plain-English daily summary (Day 6).")


if __name__ == "__main__":
    main()
