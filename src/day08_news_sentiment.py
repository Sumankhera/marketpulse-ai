"""
Day 8 — News Sentiment (RAG over headlines)

In simple terms: the model so far only sees price/volume history. It has
no idea a company just got sued, beat earnings, or is the subject of a
sell-off in its whole sector. Today we pull each ticker's recent news
headlines and hand them to Claude, asking it to read *only* those
headlines (nothing from its own training data) and call the near-term
tone bullish, neutral, or bearish, quoting the specific headline that
drove the read. That "retrieve real, current headlines, then ground the
generation in them" is the RAG pattern the roadmap called for.

This is deliberately a standalone script, same as Day 6 was before Day 7
wired it into the pipeline — next step is folding this into day07's run.

Run this any time after Day 1 has been run at least once (only needs a
ticker symbol, not the price CSVs).
Output: printed sentiment call + rationale per ticker, and the raw
headlines + calls saved to data/news/{ticker}_news.json.
"""

import json
import os
from datetime import datetime

import anthropic
import yfinance as yf

from day01_data_ingestion import WATCHLIST

NEWS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "news")

MODEL = "claude-opus-4-8"

# How many of the most recent headlines to retrieve per ticker.
HEADLINE_COUNT = 8


def fetch_headlines(ticker: str, count: int = HEADLINE_COUNT) -> list[dict]:
    """Retrieve the most recent news items for a ticker via yfinance.
    Returns a list of {title, publisher, published, link}, newest first."""
    raw = yf.Ticker(ticker).news or []

    headlines = []
    for item in raw[:count]:
        content = item.get("content", item)  # yfinance has nested "content" on newer payloads
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider") or {}
        link = (content.get("canonicalUrl") or {}).get("url", "")
        headlines.append(
            {
                "title": title,
                "publisher": provider.get("displayName", "unknown"),
                "published": content.get("pubDate", ""),
                "link": link,
            }
        )
    return headlines


def build_sentiment_prompt(ticker: str, headlines: list[dict]) -> str:
    lines = [
        f"Here are the {len(headlines)} most recent news headlines for {ticker}, "
        "retrieved just now. Base your answer only on these headlines — do not "
        "use anything else you know about this company. Some headlines may be "
        "generic market news (not company-specific) or irrelevant; ignore those.\n",
    ]
    for i, h in enumerate(headlines, 1):
        lines.append(f"{i}. \"{h['title']}\" — {h['publisher']}")

    lines.append(
        "\nRespond in exactly this format:\n"
        "Label: bullish | neutral | bearish\n"
        "Rationale: one sentence, quoting or referencing the specific headline(s) "
        "that drove the call. If no headline is actually relevant to this ticker, "
        "say Label: neutral and say why (e.g. \"no company-specific news in this batch\")."
    )
    return "\n".join(lines)


def parse_sentiment_response(text: str) -> dict:
    """Pull the Label/Rationale lines out of Claude's reply. Falls back to
    'neutral' with the raw text as rationale if the format wasn't followed."""
    label = "neutral"
    rationale = text.strip()

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            label = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("rationale:"):
            rationale = line.split(":", 1)[1].strip()

    if label not in ("bullish", "neutral", "bearish"):
        label = "neutral"

    return {"label": label, "rationale": rationale}


def get_sentiment(ticker: str, headlines: list[dict], client: anthropic.Anthropic) -> dict:
    if not headlines:
        return {"label": "neutral", "rationale": "No headlines retrieved for this ticker."}

    prompt = build_sentiment_prompt(ticker, headlines)
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return parse_sentiment_response(text)


def save_ticker_news(ticker: str, headlines: list[dict], sentiment: dict) -> str:
    os.makedirs(NEWS_DIR, exist_ok=True)
    out_path = os.path.join(NEWS_DIR, f"{ticker}_news.json")
    payload = {
        "ticker": ticker,
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "sentiment": sentiment,
        "headlines": headlines,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main():
    print(f"MarketPulse AI — Day 8: news sentiment via RAG over headlines ({datetime.now().date()})\n")

    client = anthropic.Anthropic()

    for ticker in WATCHLIST:
        print(f"{ticker}:")
        try:
            headlines = fetch_headlines(ticker)
        except Exception as e:
            print(f"  FAILED to fetch headlines ({e})\n")
            continue

        print(f"  retrieved {len(headlines)} headline(s)")
        sentiment = get_sentiment(ticker, headlines, client)
        out_path = save_ticker_news(ticker, headlines, sentiment)

        print(f"  Label: {sentiment['label']}")
        print(f"  Rationale: {sentiment['rationale']}")
        print(f"  Saved to {out_path}\n")

    print("Done. Next: fold this into the Day 7 pipeline, then a simple dashboard.")


if __name__ == "__main__":
    main()
