import pandas as pd
import re

B3_CSV_PATH = "data/b3_tickers.csv"

def load_b3_tickers() -> pd.DataFrame:
    """Load curated B3 tickers (Yahoo Finance format, with .SA suffix)."""
    df = pd.read_csv(B3_CSV_PATH)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["name"] = df["name"].astype(str)
    return df

def ensure_sa_suffix(ticker: str) -> str:
    """Normalize user input to Yahoo Finance .SA format (e.g., PETR4 -> PETR4.SA)."""
    if not isinstance(ticker, str) or not ticker.strip():
        return ""
    t = ticker.strip().upper()
    if not t.endswith(".SA"):
        t = f"{t}.SA"
    return t

def is_known_b3_ticker(ticker: str) -> bool:
    """Check if ticker (normalized) exists in our curated list."""
    df = load_b3_tickers()
    t = ensure_sa_suffix(ticker)
    return t in set(df["ticker"].tolist())

def search_b3(query: str, limit: int = 20) -> pd.DataFrame:
    """Simple case-insensitive search by code or name."""
    df = load_b3_tickers()
    if not query:
        return df.head(limit)
    q = query.strip().lower()
    mask = df["ticker"].str.lower().str.contains(q) | df["name"].str.lower().str.contains(q)
    return df[mask].head(limit)