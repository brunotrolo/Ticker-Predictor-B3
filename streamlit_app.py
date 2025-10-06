import os
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3

st.set_page_config(page_title="B3 Ticker App", page_icon="📈", layout="wide")

def _collapse_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    """If yfinance returns duplicate column names (e.g., two 'Close'), keep the first non-null value."""
    out = df.copy()
    # If MultiIndex columns, try to flatten by taking last level
    if isinstance(out.columns, pd.MultiIndex):
        # Common case: level 0 = attribute, level 1 = ticker OR vice-versa.
        # We'll try using the level that contains OHLCV names.
        levels = [list(l) for l in out.columns.levels]
        lvl0 = set(out.columns.get_level_values(0))
        lvl1 = set(out.columns.get_level_values(1))
        if {'Open','High','Low','Close','Volume'}.issubset(lvl0):
            out.columns = out.columns.get_level_values(0)
        elif {'Open','High','Low','Close','Volume'}.issubset(lvl1):
            out.columns = out.columns.get_level_values(1)
        else:
            # Fallback: flatten using join
            out.columns = ['_'.join([str(x) for x in tup if x!='']).strip('_') for tup in out.columns.to_list()]
    # Handle duplicated names (e.g., two 'Close')
    if out.columns.duplicated().any():
        # Group duplicated columns and take first non-null per row
        new_cols = {}
        for col in out.columns.unique():
            same = out.loc[:, out.columns == col]
            if isinstance(same, pd.DataFrame) and same.shape[1] > 1:
                # first non-null across duplicates
                s = same.apply(pd.to_numeric, errors='coerce').bfill(axis=1).iloc[:, 0]
                new_cols[col] = s
            else:
                new_cols[col] = out[col]
        out = pd.DataFrame(new_cols, index=out.index)
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    t = ensure_sa_suffix(ticker)
    df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _collapse_duplicate_cols(df)
    df = df.rename_axis("Date").reset_index()
    # Ensure proper dtypes (guard against accidental 2-D selections)
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            val = df[col]
            if isinstance(val, pd.DataFrame):
                s = val.apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:,0]
                df[col] = s
            else:
                df[col] = pd.to_numeric(val, errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    close = out["Close"].astype(float)
    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["EMA20"] = ema(close, 20)
    out["RSI14"] = rsi(close, 14)
    return out

def price_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preço"
    ))
    for col, nm in [("SMA20","SMA 20"), ("SMA50","SMA 50"), ("SMA200","SMA 200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=nm))
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title="Preço (BRL)", height=600)
    st.plotly_chart(fig, use_container_width=True)

def rsi_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI 14"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title="RSI", height=250)
    st.plotly_chart(fig, use_container_width=True)

# Sidebar – choose ticker (B3 only)
st.sidebar.header("⚙️ Configurações")
b3 = load_b3_tickers()
q = st.sidebar.text_input("Buscar empresa ou ticker (ex.: PETR4, VALE3)", value="")
res = search_b3(q, limit=50) if q else b3.head(50)
choice = st.sidebar.selectbox("Selecione o ticker (B3)", options=res["ticker"].tolist(), format_func=lambda t: f"{t} — {b3.loc[b3['ticker']==t,'name'].values[0]}")

# Dates
col1, col2 = st.sidebar.columns(2)
default_end = date.today()
default_start = default_end - timedelta(days=365*2)
start = col1.date_input("Início", value=default_start)
end = col2.date_input("Fim", value=default_end)

st.title("📈 B3 – Análise de Ações (Yahoo Finance)")
st.caption("Somente tickers da B3 (.SA). Os preços são ajustados por proventos.")

ticker = choice
if not is_known_b3_ticker(ticker):
    st.error("Ticker fora da B3. Use apenas códigos .SA.")
    st.stop()

with st.spinner("Baixando histórico..."):
    df = fetch_history(ticker, start, end)

if df.empty:
    st.warning("Não foi possível obter dados para este período.")
    st.stop()

dfi = add_indicators(df)

# Top metrics
c1, c2, c3, c4 = st.columns(4)
last_close = float(dfi['Close'].iloc[-1])
pct_20 = (last_close / float(dfi['SMA20'].iloc[-1]) - 1) * 100 if pd.notna(dfi['SMA20'].iloc[-1]) else np.nan
pct_50 = (last_close / float(dfi['SMA50'].iloc[-1]) - 1) * 100 if pd.notna(dfi['SMA50'].iloc[-1]) else np.nan
rsi_last = float(dfi['RSI14'].iloc[-1]) if pd.notna(dfi['RSI14'].iloc[-1]) else np.nan
c1.metric("Ticker", ticker)
c2.metric("Fechamento", f"R$ {last_close:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
c3.metric("Δ vs SMA20", f"{pct_20:+.2f}%" if pd.notna(pct_20) else "—")
c4.metric("RSI(14)", f"{rsi_last:.1f}" if pd.notna(rsi_last) else "—")

# Charts
price_chart(dfi, f"{ticker} • Preço e Médias Móveis")
rsi_chart(dfi, f"{ticker} • RSI (14)")

st.info("Dica: você pode colar **PETR4**, **VALE3**, **ITUB4**, etc. Se digitar sem **.SA**, a aplicação adiciona automaticamente.")