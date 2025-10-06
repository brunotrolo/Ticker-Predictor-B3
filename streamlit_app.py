import os
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3

st.set_page_config(page_title="B3 Ticker App", page_icon="📈", layout="wide")

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    t = ensure_sa_suffix(ticker)
    df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    df = df.rename_axis("Date").reset_index()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Moving averages
    out["SMA20"] = SMAIndicator(out["Close"], window=20).sma_indicator()
    out["SMA50"] = SMAIndicator(out["Close"], window=50).sma_indicator()
    out["SMA200"] = SMAIndicator(out["Close"], window=200).sma_indicator()
    out["EMA20"] = EMAIndicator(out["Close"], window=20).ema_indicator()
    # RSI
    out["RSI14"] = RSIIndicator(out["Close"], window=14).rsi()
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
pct_20 = (last_close / float(dfi['SMA20'].iloc[-1]) - 1) * 100 if not np.isnan(dfi['SMA20'].iloc[-1]) else np.nan
pct_50 = (last_close / float(dfi['SMA50'].iloc[-1]) - 1) * 100 if not np.isnan(dfi['SMA50'].iloc[-1]) else np.nan
rsi = float(dfi['RSI14'].iloc[-1])
c1.metric("Ticker", ticker)
c2.metric("Fechamento", f"R$ {last_close:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
c3.metric("Δ vs SMA20", f"{pct_20:+.2f}%")
c4.metric("RSI(14)", f"{rsi:.1f}")

# Charts
price_chart(dfi, f"{ticker} • Preço e Médias Móveis")
rsi_chart(dfi, f"{ticker} • RSI (14)")

st.info("Dica: você pode colar **PETR4**, **VALE3**, **ITUB4**, etc. Se digitar sem **.SA**, a aplicação adiciona automaticamente.")