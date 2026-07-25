"""
Day 9 — Simple Dashboard

In simple terms: everything so far (price data, features, model predictions,
news sentiment, daily reports) only existed as CSVs, JSON files, and printed
terminal output. Today wraps all of it in one interactive page — pick a
ticker from the watchlist and see its price chart, latest indicators, model
prediction + backtest, news sentiment, and the most recent daily report,
without digging through files by hand. This is a viewer only; it reads
whatever day07_pipeline.py last wrote, it doesn't refetch or retrain.

Run: streamlit run src/day09_dashboard.py
(Requires day01/day03 to have populated data/ and data/features/ at least
once; news sentiment and the daily report sections show a friendly message
if day07/day08 haven't been run yet.)
"""

import glob
import json
import os

import pandas as pd
import streamlit as st

from day01_data_ingestion import WATCHLIST

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
NEWS_DIR = os.path.join(DATA_DIR, "news")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")


def load_price_data(ticker: str) -> pd.DataFrame | None:
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def load_features(ticker: str) -> pd.DataFrame | None:
    filepath = os.path.join(FEATURES_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df.sort_index()


def load_news(ticker: str) -> dict | None:
    filepath = os.path.join(NEWS_DIR, f"{ticker}_news.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def latest_report_path() -> str | None:
    reports = sorted(glob.glob(os.path.join(REPORTS_DIR, "*.md")))
    return reports[-1] if reports else None


def render_ticker(ticker: str) -> None:
    price_df = load_price_data(ticker)
    features_df = load_features(ticker)

    st.subheader(f"{ticker}")

    if price_df is None:
        st.warning("No price data found. Run src/day01_data_ingestion.py first.")
        return

    price_df = price_df.copy()
    price_df["MA20"] = price_df["Close"].rolling(20).mean()
    st.line_chart(price_df[["Close", "MA20"]])

    col1, col2, col3 = st.columns(3)
    latest_close = price_df["Close"].iloc[-1]
    col1.metric("Latest close", f"${latest_close:.2f}")

    if features_df is not None:
        latest_features = features_df.dropna(subset=["RSI14"]).iloc[-1]
        col2.metric("RSI14", f"{latest_features['RSI14']:.1f}")
        col3.metric("As of", str(latest_features.name.date()))
    else:
        col2.info("No feature data. Run src/day03_features.py.")

    news = load_news(ticker)
    st.markdown("**News sentiment**")
    if news is None:
        st.info("No news sentiment yet. Run src/day08_news_sentiment.py or src/day07_pipeline.py.")
    else:
        label = news["sentiment"]["label"]
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(label, "⚪")
        st.write(f"{emoji} **{label.capitalize()}** — {news['sentiment']['rationale']}")
        st.caption(f"As of {news['as_of']}, based on {len(news['headlines'])} headline(s).")
        with st.expander("Show retrieved headlines"):
            for h in news["headlines"]:
                st.write(f"- {h['title']} ({h['publisher']})")


def main():
    st.set_page_config(page_title="MarketPulse AI", layout="wide")
    st.title("MarketPulse AI — Dashboard")
    st.caption(
        "A daily-built project turning market data into plain-English insights "
        "via data pipelines, ML, and an LLM layer."
    )

    report_path = latest_report_path()
    if report_path is None:
        st.info(
            "No daily report yet. Run `python src/day07_pipeline.py` "
            "(needs an ANTHROPIC_API_KEY) to generate the first one."
        )
    else:
        with open(report_path) as f:
            report_text = f.read()
        with st.expander(f"Latest daily report ({os.path.basename(report_path)})", expanded=True):
            st.markdown(report_text)

    st.divider()

    tabs = st.tabs(WATCHLIST)
    for tab, ticker in zip(tabs, WATCHLIST):
        with tab:
            render_ticker(ticker)


if __name__ == "__main__":
    main()
