"""
Day 3 — Feature Engineering

In simple terms: we take the stock price history from Day 1 and add a
few extra "clues" (columns) that help spot patterns. These clues will
be fed into a prediction model on Day 4.

The clues we add:
  - RSI (14-day)      -> is the stock "overbought" or "oversold"?
  - Bollinger Bands    -> is the price moving outside its normal range?
  - Past returns       -> how much did the stock move on recent days?
  - Target             -> did the stock go UP (1) or DOWN (0) the next day?
                          (this is the "answer key" Day 4 will learn to predict)

Run this after src/day01_data_ingestion.py has populated data/*.csv.
Output: one CSV per ticker in data/features/, e.g. data/features/AAPL_features.csv
"""

import glob
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

# How many days back to look when building "past returns" columns
LAG_DAYS = (1, 2, 3, 5)


def load_ticker_data(filepath: str) -> pd.DataFrame:
    """Read one ticker's price CSV and make sure it's sorted oldest to newest."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI = a 0-100 score.
    High (e.g. above 70) = stock has gone up a lot lately, may be "overbought".
    Low (e.g. below 30) = stock has gone down a lot lately, may be "oversold".
    """
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily percentage change in price
    df["Return"] = df["Close"].pct_change(fill_method=None)

    # Clue 1: RSI (overbought / oversold score)
    df["RSI14"] = compute_rsi(df["Close"], window=14)

    # Clue 2: Bollinger Bands (normal price range based on recent volatility)
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Mid"] = ma20
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20

    # Clue 3: what happened on recent past days (model can't see the future,
    # so we line up past values as separate columns)
    for lag in LAG_DAYS:
        df[f"Return_Lag{lag}"] = df["Return"].shift(lag)

    # The "answer key" for Day 4: did price go up the next day?
    # (the very last row has no "next day" yet, so its answer stays blank
    # rather than being wrongly marked as "down")
    next_close = df["Close"].shift(-1)
    df["Target"] = (next_close > df["Close"]).where(next_close.notna())

    return df


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print("No data found in data/. Run src/day01_data_ingestion.py first.")
        return

    print("MarketPulse AI — Day 3: adding prediction clues to the data\n")

    os.makedirs(FEATURES_DIR, exist_ok=True)

    for filepath in files:
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        df = load_ticker_data(filepath)
        features = build_features(df)

        out_path = os.path.join(FEATURES_DIR, f"{ticker}_features.csv")
        features.to_csv(out_path)

        ready_rows = features.dropna().shape[0]
        print(f"  {ticker}: {features.shape[0]} rows saved -> {out_path}")
        print(f"           ({ready_rows} of those rows have every clue filled in and are ready for modeling)")

    print("\nDone. Next: Day 4 will use these clues to train a model that predicts up/down.")


if __name__ == "__main__":
    main()
