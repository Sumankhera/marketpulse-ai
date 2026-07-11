# Progress Log

## Day 1 — 2026-07-08
Set up project scaffold and built the data ingestion script. Pulls daily OHLCV data for a small watchlist (AAPL, MSFT, GOOGL, SPY) via yfinance and saves it to `data/`.

Verified: script syntax is valid and the CSV-save logic works correctly (tested with mock data). Live Yahoo Finance calls couldn't be tested from this sandbox (network is restricted here) — run `python src/day01_data_ingestion.py` on your own machine to pull real data.

Next: exploratory analysis and first visualizations (Day 2).

## Day 2 — 2026-07-10
Built the EDA script: reads Day 1's CSVs, computes daily returns, 20-day moving average, and 20-day rolling volatility, and saves three charts per ticker (price+MA, returns histogram, volatility) to `outputs/charts/`. Also prints a summary stats table.

Verified: tested end-to-end with synthetic mock data (same shape as real yfinance output) — script runs cleanly and produces all 6 chart files plus the summary table. Added `.gitignore` so raw `data/*.csv` isn't tracked (regenerates daily), while `outputs/charts/` PNGs are tracked as the visible portfolio evidence.

Next: feature engineering — RSI, volatility bands (Day 3).

## Day 3 — 2026-07-11
Built the feature engineering script: reads Day 1's CSVs and adds the "clues" a prediction model will need — a 14-day RSI (overbought/oversold score), Bollinger Bands (normal price range based on recent volatility), lagged returns (1/2/3/5 days back), and a target column marking whether the price went up the next day. Saves one enriched CSV per ticker to `data/features/`.

Verified: ran end-to-end on the real data already pulled in Day 1 (not mock data this time). Caught and fixed a real bug along the way — the very last row of each ticker has no "next day" yet, so its target was silently coming out as a fake "down" instead of blank; fixed by leaving it blank (`NaN`) when there's no next-day price to compare against. Added `data/features/*.csv` to `.gitignore` since it regenerates from `data/*.csv`, same treatment as the raw data.

Next: baseline ML model — predict next-day direction (up/down) using these features (Day 4).
