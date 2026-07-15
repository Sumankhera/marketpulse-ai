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

## Day 4 — 2026-07-13
Built the baseline model: a logistic regression trained per ticker on Day 3's clues (RSI, Bollinger Bands, lagged returns) to predict next-day up/down. Split each ticker's history chronologically (80/20) rather than randomly, since shuffling would let the model "see the future." Compared against a dumb baseline (always guess the more common direction).

Verified: ran end-to-end on the real feature data. Results are mixed and reported honestly — MSFT beat the baseline (66.7% vs 52.4%), but AAPL, GOOGL, and SPY did not. That's expected for a first-pass model on ~100 rows per ticker; the point of Day 4 was a working, honest baseline to improve on, not a winning model yet.

Next: model evaluation + a simple backtest (Day 5).

## Day 5 — 2026-07-14
Built the evaluation + backtest script: reports precision/recall (not just accuracy, which hides a model that only calls "up") on the same chronological test split as Day 4, then runs a simple backtest — hold the stock on days the model predicts "up", sit in cash otherwise — and compares the compounded result to plain buy-and-hold.

Verified: ran end-to-end on the real feature data. MSFT showed real signal (75% precision, beat buy-and-hold by ~9.7 points); AAPL and GOOGL never predicted "up" at all on this test window (0% precision/recall), so they just sat in cash; SPY had 100% recall but only 48% precision and tied buy-and-hold. Reported honestly rather than cherry-picked.

Next: LLM layer — turn model output + data into a plain-English daily summary (Day 6).

## Day 6 — 2026-07-14
Built the LLM layer: retrains the Day 4/5 model per ticker, predicts tomorrow's direction from the latest (unlabeled) row, and hands Claude (`claude-opus-4-8`) a compact per-ticker data summary — price, RSI, predicted direction, precision/recall, and backtest vs. buy-and-hold — with instructions to write one honest plain-English paragraph per ticker and flag weak signals instead of hedging.

Verified: the data-prep half (retraining, next-day prediction, prompt construction) ran end-to-end on the real feature data and produced a correct per-ticker prompt. The actual Claude API call is unverified in this sandbox — no `ANTHROPIC_API_KEY` is set here; the user will run it with their own key.

Next: automate the pipeline end-to-end, writing a daily report file (Day 7).

## Day 7 — 2026-07-15
Wired Days 1, 3, 4/5, and 6 together into one script (`day07_pipeline.py`): pulls fresh OHLCV data, rebuilds the feature CSVs, retrains each ticker's model, asks Claude for the summary, and writes everything to a dated report file (`outputs/reports/YYYY-MM-DD.md`) instead of just printing to the terminal.

Verified: ran the ingestion and feature-rebuild steps end-to-end with live Yahoo Finance data (network was reachable from this sandbox today, unlike Day 1). Ran the model-retraining, prediction, and report-writing steps end-to-end too, standing in a placeholder for the one piece this sandbox can't do — the live Claude API call, since no `ANTHROPIC_API_KEY` is set here — and confirmed the generated report file has the right shape (summary + per-ticker table). Deleted that placeholder report afterward so the committed repo doesn't carry a fake summary; the user will run `python src/day07_pipeline.py` with their own key to produce the first real one.

Next: news sentiment (RAG over headlines) and a simple dashboard (Day 8).
