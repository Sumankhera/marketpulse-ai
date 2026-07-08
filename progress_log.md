# Progress Log

## Day 1 — 2026-07-08
Set up project scaffold and built the data ingestion script. Pulls daily OHLCV data for a small watchlist (AAPL, MSFT, GOOGL, SPY) via yfinance and saves it to `data/`.

Verified: script syntax is valid and the CSV-save logic works correctly (tested with mock data). Live Yahoo Finance calls couldn't be tested from this sandbox (network is restricted here) — run `python src/day01_data_ingestion.py` on your own machine to pull real data.

Next: exploratory analysis and first visualizations (Day 2).
