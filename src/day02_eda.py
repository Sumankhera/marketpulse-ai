"""
Day 2 — Exploratory Data Analysis
Reads the OHLCV data pulled in Day 1 and produces:
  - a price chart with a 20-day moving average
  - a daily returns histogram
  - a rolling 20-day volatility chart
  - a printed summary stats table

Run this after src/day01_data_ingestion.py has populated data/*.csv.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")


def load_ticker_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def analyze_ticker(ticker: str, df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Return"] = df["Close"].pct_change(fill_method=None)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility20"] = df["Return"].rolling(20).std()

    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Price + moving average
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["MA20"], label="20-day MA", linestyle="--")
    ax.set_title(f"{ticker} — Price & 20-day Moving Average")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{ticker}_price.png"))
    plt.close(fig)

    # Returns histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Return"].dropna().hist(bins=40, ax=ax)
    ax.set_title(f"{ticker} — Daily Returns Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{ticker}_returns_hist.png"))
    plt.close(fig)

    # Rolling volatility
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["Volatility20"])
    ax.set_title(f"{ticker} — 20-day Rolling Volatility")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{ticker}_volatility.png"))
    plt.close(fig)

    return {
        "ticker": ticker,
        "mean_daily_return": df["Return"].mean(),
        "std_daily_return": df["Return"].std(),
        "max_daily_return": df["Return"].max(),
        "min_daily_return": df["Return"].min(),
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print("No data found in data/. Run src/day01_data_ingestion.py first.")
        return

    print("MarketPulse AI — Day 2 exploratory analysis\n")

    summary_rows = []
    for filepath in files:
        ticker = os.path.splitext(os.path.basename(filepath))[0]
        df = load_ticker_data(filepath)
        summary_rows.append(analyze_ticker(ticker, df))
        print(f"  {ticker}: charts saved to {CHARTS_DIR}")

    summary = pd.DataFrame(summary_rows).set_index("ticker")
    print("\nSummary stats (daily returns):")
    print(summary.round(4))
    print("\nDone. Next: Day 3 will engineer features (RSI, volatility bands) for the model.")


if __name__ == "__main__":
    main()
