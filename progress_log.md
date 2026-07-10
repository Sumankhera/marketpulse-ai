# Progress Log

## Day 1 — 2026-07-08
Set up project scaffold and built the data ingestion script. Pulls daily OHLCV data for a small watchlist (AAPL, MSFT, GOOGL, SPY) via yfinance and saves it to `data/`.

Verified: script syntax is valid and the CSV-save logic works correctly (tested with mock data). Live Yahoo Finance calls couldn't be tested from this sandbox (network is restricted here) — run `python src/day01_data_ingestion.py` on your own machine to pull real data.

Next: exploratory analysis and first visualizations (Day 2).

## Day 2 — 2026-07-10
Built the EDA script: reads Day 1's CSVs, computes daily returns, 20-day moving average, and 20-day rolling volatility, and saves three charts per ticker (price+MA, returns histogram, volatility) to `outputs/charts/`. Also prints a summary stats table.

Verified: tested end-to-end with synthetic mock data (same shape as real yfinance output) — script runs cleanly and produces all 6 chart files plus the summary table. Added `.gitignore` so raw `data/*.csv` isn't tracked (regenerates daily), while `outputs/charts/` PNGs are tracked as the visible portfolio evidence.

Next: feature engineering — RSI, volatility bands (Day 3).
