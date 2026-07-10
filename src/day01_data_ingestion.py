"""
Day 1 — Data Ingestion
Pulls daily OHLCV (open/high/low/close/volume) data for a small watchlist
of stocks/ETFs and saves each to a CSV in data/.

This is the foundation the rest of the project builds on: Day 2 will
analyze this data, Day 4 will train a model on it, Day 6 will summarize
it with an LLM.
"""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf

# Watchlist: a mix of individual stocks and a broad market ETF.
WATCHLIST = ["AAPL", "MSFT", "GOOGL", "SPY"]

# How far back to pull history (enough for moving averages later).
PERIOD = "6mo"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_and_save(ticker: str, period: str = PERIOD) -> str:
    """Download OHLCV data for one ticker and save it as CSV. Returns the file path."""
    df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol or your connection.")

    # Newer yfinance versions return MultiIndex columns (e.g. ("Close", "AAPL"))
    # even for a single ticker. Flatten to plain column names so the CSV has
    # one clean header row instead of two, which would break reading it back.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(filepath)
    return filepath


def main():
    print(f"MarketPulse AI — Day 1 data pull ({datetime.now().date()})")
    print(f"Watchlist: {', '.join(WATCHLIST)}\n")

    for ticker in WATCHLIST:
        try:
            path = fetch_and_save(ticker)
            print(f"  {ticker}: saved to {path}")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    print("\nDone. Next: Day 2 will explore this data and build the first charts.")


if __name__ == "__main__":
    main()
