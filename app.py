# ============================================================
# VIN QUANT | QUANTITATIVE MARKET TERMINAL
# Real market data | Technical analytics | Backtesting
# Data sources: Coinbase Exchange API + Yahoo Finance
# ============================================================

import io
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# -------------------- CONFIGURATION --------------------------

st.set_page_config(
    page_title="VIN QUANT | Market Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "VIN QUANT"
COINBASE_URL = "https://api.exchange.coinbase.com"

ASSETS = {
    "BTC / USD": {"source": "coinbase", "symbol": "BTC-USD"},
    "ETH / USD": {"source": "coinbase", "symbol": "ETH-USD"},
    "SOL / USD": {"source": "coinbase", "symbol": "SOL-USD"},
    "GOLD / USD": {"source": "yahoo", "symbol": "GC=F"},
    "S&P 500": {"source": "yahoo", "symbol": "^GSPC"},
    "NASDAQ 100": {"source": "yahoo", "symbol": "^NDX"},
    "US DOLLAR INDEX": {"source": "yahoo", "symbol": "DX-Y.NYB"},
    "US TREASURY 20Y": {"source": "yahoo", "symbol": "TLT"},
}

PERIOD_OPTIONS = {
    "1H": ["7 hari", "30 hari", "60 hari"],
    "1D": ["1 bulan", "3 bulan", "6 bulan", "1 tahun", "2 tahun"],
}

PERIOD_DAYS = {
    "7 hari": 7,
    "30 hari": 30,
    "60 hari": 60,
    "1 bulan": 30,
    "3 bulan": 90,
    "6 bulan": 180,
    "1 tahun": 365,
    "2 tahun": 730,
}

# -------------------- TERMINAL DESIGN ------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #090B0E;
        color: #E6EAF0;
    }
    [data-testid="stSidebar"] {
        background: #101318;
        border-right: 1px solid #292D35;
    }
    h1, h2, h3 {
        color: #F2F4F8 !important;
        letter-spacing: -0.4px;
    }
    p, label, .stMarkdown {
        color: #C9CED8;
    }
    div[data-testid="stMetric"] {
        background: #11151B;
        border: 1px solid #292E37;
        padding: 14px 16px;
        border-radius: 9px;
    }
    div[data-testid="stMetricLabel"] {
        color: #929BA9;
    }
    div[data-testid="stMetricValue"] {
        color: #F5F7FA;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    .terminal-header {
        background: linear-gradient(110deg, #141920, #101318);
        border: 1px solid #303640;
        padding: 20px 22px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
    .brand {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #F59E0B;
    }
    .subtitle {
        color: #8F99A8;
        font-size: 12px;
        letter-spacing: 1.3px;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid #246B4B;
        background: #10281E;
        color: #65E6A1;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 11px;
    }
    .section-title {
        color: #F59E0B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.4px;
        padding: 8px 0 12px 0;
    }
    .small-note {
        color: #858E9C;
        font-size: 11px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #292E37;
        border-radius: 8px;
    }
    button[kind="primary"] {
        background: #F59E0B;
        color: #101318;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- DATA FUNCTIONS -------------------------

def utc_now():
    return datetime.now(timezone.utc)


def normalize_ohlcv(df):
    """Clean and standardize OHLCV data."""
    if df is None or df.empty:
        raise ValueError("Sumber data tidak mengembalikan candle.")

    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Kolom OHLC tidak lengkap: {missing}")

    if "volume" not in out.columns:
        out["volume"] = np.nan

    out = out[["open", "high", "low", "close", "volume"]].copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    # Remove impossible or non-positive prices.
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]

    if out.empty:
        raise ValueError("Tidak ada candle valid setelah proses pembersihan.")

    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_coinbase(product, days, timeframe):
    """Fetch real candles from Coinbase public Exchange API."""
    granularity = 3600 if timeframe == "1H" else 86400
    end_time = utc_now()
    start_time = end_time - timedelta(days=int(days))

    all_rows = []
    chunk_size = 280 * granularity
    cursor = start_time

    headers = {
        "User-Agent": "VIN-QUANT-Research/1.0",
        "Accept": "application/json",
    }

    while cursor < end_time:
        chunk_end = min(cursor + timedelta(seconds=chunk_size), end_time)

        params = {
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": granularity,
        }

        response = requests.get(
            f"{COINBASE_URL}/products/{product}/candles",
            params=params,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json()

        if isinstance(rows, list):
            all_rows.extend(rows)

        cursor = chunk_end
        time.sleep(0.12)

    if not all_rows:
        raise ValueError(
            f"Coinbase tidak mengembalikan candle untuk {product}. "
            "Coba refresh atau pilih periode lain."
        )

    # Coinbase format: [time, low, high, open, close, volume]
    df = pd.DataFrame(
        all_rows,
        columns=["epoch", "low", "high", "open", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.drop(columns=["epoch"]).set_index("timestamp")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return normalize_ohlcv(df)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo(symbol, period_label, timeframe):
    """Fetch real market data from Yahoo Finance via yfinance."""
    if timeframe == "1H":
        period_map = {
            "7 hari": "7d",
            "30 hari": "1mo",
            "60 hari": "60d",
        }
        interval = "1h"
    else:
        period_map = {
            "1 bulan": "1mo",
            "3 bulan": "3mo",
            "6 bulan": "6mo",
            "1 tahun": "1y",
            "2 tahun": "2y",
        }
        interval = "1d"

    period = period_map[period_label]

    data = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )

    if data is None or data.empty:
        raise ValueError(
            f"Yahoo Finance tidak mengembalikan data untuk {symbol}. "
            "Coba timeframe harian atau periode yang lebih pendek."
        )

    # Newer yfinance versions may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).lower() for c in data.columns]

    rename_map = {
        "adj close": "adj_close",
        "stock splits": "stock_splits",
    }
    data = data.rename(columns=rename_map)

    keep = ["open", "high", "low", "close"]
    if "volume" in data.columns:
        keep.append("volume")

    data = data[keep].copy()
    return normalize_ohlcv(data)


def fetch_market_data(asset_name, period_label, timeframe):
    asset = ASSETS[asset_name]
    days = PERIOD_DAYS[period_label]

    if asset["source"] == "coinbase":
        return fetch_coinbase(asset["symbol"], days, timeframe)

    return fetch_yahoo(asset["symbol"], period_label, timeframe)


# -------------------- QUANTITATIVE FEATURES ------------------

def calculate_features(raw):
    df = raw.copy()

    # Returns
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Trend indicators
    df["sma20"] = df["close"].rolling(20, min_periods=20).mean()
    df["sma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Rolling volatility; annualization assumes 24/7 for hourly crypto
    # and 252 daily periods for daily market data.
    periods_per_year = 24 * 365 if df.index.to_series().diff().median() < pd.Timedelta("2D") else 252
    df["volatility20"] = (
        df["log_return"].rolling(20, min_periods=10).std()
        * np.sqrt(periods_per_year)
    )

    # Z-score of price relative to 20-period rolling mean.
    rolling_std = df["close"].rolling(20, min_periods=20).std()
    df["zscore20"] = (df["close"] - df["sma20"]) / rolling_std.replace(0, np.nan)

    # Drawdown from running peak.
    df["running_peak"] = df["close"].cummax()
    df["drawdown"] = df["close"] / df["running_peak"] - 1

    # Simple SMA crossover position; 1 = long, 0 = cash.
    df["position"] = (df["sma20"] > df["sma50"]).astype(int)

    # Lag position one bar to avoid using same-bar information.
    df["strategy_return"] = df["position"].shift(1) * df["return"]
    df["strategy_return"] = df["strategy_return"].fillna(0.0)

    df["buy_hold_equity"] = (1 + df["return"].fillna(0)).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    df["strategy_peak"] = df["strategy_equity"].cummax()
    df["strategy_drawdown"] = (
        df["strategy_equity"] / df["strategy_peak"] - 1
    )

    return df.replace([np.inf, -np.inf], np.nan)


def calculate_performance(df):
    r = df["strategy_return"].dropna()
    equity = df["strategy_equity"].dropna()

    if len(r) < 2 or len(equity) < 2:
        return {}

    hourly = df.index.to_series().diff().median() < pd.Timedelta("2D")
    periods_per_year = 24 * 365 if hourly else 252

    total_return = float(equity.iloc[-1] - 1)
    years = max(len(r) / periods_per_year, 1 / periods_per_year)

    if total_return > -1:
        cagr = float((equity.iloc[-1] ** (1 / years)) - 1)
    else:
        cagr = -1.0

    std = r.std(ddof=1)
    sharpe = (
        float(np.sqrt(periods_per_year) * r.mean() / std)
        if pd.notna(std) and std > 0
        else np.nan
    )

    peak = df["strategy_equity"].cummax()
    dd = df["strategy_equity"] / peak - 1
    max_dd = float(dd.min())

    active = df["position"].shift(1).fillna(0).astype(bool)
    trade_returns = df.loc[active, "strategy_return"]
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else np.nan

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0
        else (np.inf if gross_profit > 0 else np.nan)
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Observations": len(df),
    }


def make_excel_bytes(dataframe):
    """Convert DataFrame to a downloadable Excel workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="VIN_QUANT_DATA")
    buffer.seek(0)
    return buffer.getvalue()


def fmt_price(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def fmt_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:,.2f}%"


# -------------------- SIDEBAR --------------------------------

st.sidebar.markdown(
    """
    <div style="padding:8px 0 20px 0">
      <div class="brand">VIN QUANT</div>
      <div class="subtitle">QUANTITATIVE RESEARCH TERMINAL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Market Overview",
        "Quant Analytics",
        "Backtest Lab",
        "Data Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### MARKET CONFIGURATION")

asset_name = st.sidebar.selectbox(
    "Pilih aset",
    list(ASSETS.keys()),
    index=0,
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1H", "1D"],
    index=0,
)

period_label = st.sidebar.selectbox(
    "Periode observasi",
    PERIOD_OPTIONS[timeframe],
    index=1 if timeframe == "1H" else 1,
)

st.sidebar.caption("Data publik • Tidak memerlukan API key")

if st.sidebar.button("↻ Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### RESEARCH SETTINGS")
sma_fast = st.sidebar.slider("SMA cepat", 5, 50, 20)
sma_slow = st.sidebar.slider("SMA lambat", 20, 200, 50)
initial_capital = st.sidebar.number_input(
    "Modal simulasi ($)",
    min_value=100.0,
    max_value=1_000_000_000.0,
    value=10_000.0,
    step=500.0,
)
fee_bps = st.sidebar.number_input(
    "Biaya per pergantian posisi (bps)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
)

if sma_fast >= sma_slow:
    st.sidebar.warning("SMA cepat sebaiknya lebih kecil dari SMA lambat.")

# -------------------- HEADER ---------------------------------

st.markdown(
    f"""
    <div class="terminal-header">
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">
        <div>
          <div class="brand">VIN QUANT <span style="color:#F3F4F6">/ TERMINAL</span></div>
          <div class="subtitle">MARKET DATA • QUANTITATIVE ANALYTICS • STRATEGY RESEARCH</div>
        </div>
        <div style="text-align:right">
          <span class="status-pill">● DATA CONNECTORS ENABLED</span>
          <div class="small-note" style="margin-top:8px">
            Local time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- LOAD DATA ------------------------------

try:
    with st.spinner(f"Mengambil data {asset_name} dari sumber pasar..."):
        raw_data = fetch_market_data(asset_name, period_label, timeframe)

    # Allow research settings to change the crossover windows.
    data = raw_data.copy()
    data["return"] = data["close"].pct_change()
    data["log_return"] = np.log(data["close"] / data["close"].shift(1))
    data["sma20"] = data["close"].rolling(sma_fast, min_periods=sma_fast).mean()
    data["sma50"] = data["close"].rolling(sma_slow, min_periods=sma_slow).mean()
    data["ema20"] = data["close"].ewm(span=sma_fast, adjust=False).mean()
    data["ema50"] = data["close"].ewm(span=sma_slow, adjust=False).mean()

    periods_per_year = 24 * 365 if timeframe == "1H" else 252
    data["volatility20"] = (
        data["log_return"].rolling(20, min_periods=10).std()
        * np.sqrt(periods_per_year)
    )
    rolling_std = data["close"].rolling(20, min_periods=20).std()
    data["zscore20"] = (data["close"] - data["close"].rolling(20).mean()) / rolling_std.replace(0, np.nan)

    data["running_peak"] = data["close"].cummax()
    data["drawdown"] = data["close"] / data["running_peak"] - 1

    # Strategy: long when fast SMA > slow SMA; otherwise cash.
    data["position"] = (data["sma20"] > data["sma50"]).astype(int)
    data["position_lag"] = data["position"].shift(1).fillna(0)

    # Charge transaction cost when position changes.
    turnover = data["position"].diff().abs().fillna(0)
    fee_rate = fee_bps / 10000.0
    data["strategy_return"] = (
        data["position_lag"] * data["return"].fillna(0)
        - turnover * fee_rate
    )
    data["strategy_equity"] = (1 + data["strategy_return"]).cumprod()
    data["buy_hold_equity"] = (1 + data["return"].fillna(0)).cumprod()
    data["strategy_peak"] = data["strategy_equity"].cummax()
    data["strategy_drawdown"] = data["strategy_equity"] / data["strategy_peak"] - 1

    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    price_change = latest["close"] / previous["close"] - 1 if previous["close"] else np.nan

    if latest["sma20"] > latest["sma50"]:
        trend_status = "BULLISH TREND"
        trend_color = "#5BE39A"
    elif latest["sma20"] < latest["sma50"]:
        trend_status = "BEARISH TREND"
        trend_color = "#FF6B6B"
    else:
        trend_status = "NEUTRAL / INSUFFICIENT DATA"
        trend_color = "#F4B860"

    # -------------------- MARKET TICKER ----------------------

    st.markdown('<div class="section-title">01 / MARKET SNAPSHOT</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("LAST PRICE", fmt_price(latest["close"]), fmt_pct(price_change))
    m2.metric("OBSERVATIONS", f"{len(data):,}")
    m3.metric("LOG RETURN", fmt_pct(latest["log_return"]))
    m4.metric("VOLATILITY (20)", fmt_pct(latest["volatility20"]))
    m5.metric("MAX DRAWDOWN", fmt_pct(data["drawdown"].min()))

    st.markdown(
        f"""
        <div style="background:#11151B;border:1px solid #292E37;
                    border-radius:8px;padding:12px 16px;margin:8px 0 18px 0">
          <span class="small-note">QUANT TREND STATUS</span>
          <span style="color:{trend_color};font-weight:800;margin-left:12px">
            {trend_status}
          </span>
          <span class="small-note" style="float:right">
            Source: {ASSETS[asset_name]["source"].upper()}
            | Symbol: {ASSETS[asset_name]["symbol"]}
            | Timeframe: {timeframe}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- PRICE CHART ------------------------

    if page in ["Market Overview", "Quant Analytics"]:
        st.markdown('<div class="section-title">02 / PRICE & TREND STRUCTURE</div>', unsafe_allow_html=True)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.74, 0.26],
            subplot_titles=("OHLC CANDLESTICKS + MOVING AVERAGES", "VOLUME"),
        )

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="OHLC",
                increasing_line_color="#29C77A",
                decreasing_line_color="#F05D5E",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["sma20"],
                mode="lines",
                name=f"SMA {sma_fast}",
                line=dict(color="#F59E0B", width=1.5),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["sma50"],
                mode="lines",
                name=f"SMA {sma_slow}",
                line=dict(color="#65A5FF", width=1.5),
            ),
            row=1,
            col=1,
        )

        volume_colors = np.where(
            data["close"] >= data["open"], "#29C77A", "#F05D5E"
        )

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                name="Volume",
                marker_color=volume_colors,
                opacity=0.65,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=650,
            template="plotly_dark",
            paper_bgcolor="#090B0E",
            plot_bgcolor="#090B0E",
            font=dict(color="#C9CED8", size=11),
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        fig.update_xaxes(
            gridcolor="#252A32",
            showgrid=True,
            zeroline=False,
        )
        fig.update_yaxes(
            gridcolor="#252A32",
            showgrid=True,
            zeroline=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------- QUANT ANALYTICS --------------------

    if page == "Quant Analytics":
        st.markdown('<div class="section-title">03 / QUANTITATIVE FEATURES</div>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            zfig = go.Figure()
            zfig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["zscore20"],
                    mode="lines",
                    name="Z-Score",
                    line=dict(color="#B78AFF", width=1.6),
                )
            )
            zfig.add_hline(y=2, line_dash="dash", line_color="#F05D5E")
            zfig.add_hline(y=-2, line_dash="dash", line_color="#29C77A")
            zfig.add_hline(y=0, line_color="#555B66")
            zfig.update_layout(
                title="Rolling Price Z-Score",
                height=330,
                template="plotly_dark",
                paper_bgcolor="#090B0E",
                plot_bgcolor="#090B0E",
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(zfig, use_container_width=True)

        with right:
            vfig = go.Figure()
            vfig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["volatility20"] * 100,
                    mode="lines",
                    name="Annualized Volatility",
                    line=dict(color="#F59E0B", width=1.6),
                    fill="tozeroy",
                )
            )
            vfig.update_layout(
                title="Rolling Volatility (Annualized %)",
                height=330,
                template="plotly_dark",
                paper_bgcolor="#090B0E",
                plot_bgcolor="#090B0E",
                margin=dict(l=10, r=10, t=45, b=10),
                yaxis_title="Percent",
            )
            st.plotly_chart(vfig, use_container_width=True)

        st.markdown("#### Quantitative interpretation")
        z = latest["zscore20"]
        vol = latest["volatility20"]

        if pd.isna(z):
            z_text = "Z-Score belum tersedia; data belum cukup untuk rolling window."
        elif z >= 2:
            z_text = "Harga berada sekitar 2 standar deviasi atau lebih di atas rata-rata 20 candle."
        elif z <= -2:
            z_text = "Harga berada sekitar 2 standar deviasi atau lebih di bawah rata-rata 20 candle."
        else:
            z_text = "Harga masih berada dalam rentang ±2 standar deviasi dari rata-rata 20 candle."

        st.write(f"**Z-Score:** {z_text}")
        st.write(
            "**Volatilitas:** "
            + (
                f"estimasi annualized volatility {vol:.2%}."
                if pd.notna(vol)
                else "belum tersedia."
            )
        )
        st.caption(
            "Z-Score bukan sinyal BUY/SELL yang berdiri sendiri. Nilai ekstrem dapat bertahan "
            "lama saat tren kuat."
        )

    # -------------------- BACKTEST LAB -----------------------

    if page == "Backtest Lab":
        st.markdown('<div class="section-title">03 / STRATEGY BACKTEST LAB</div>', unsafe_allow_html=True)
        st.caption(
            f"Strategi: long saat SMA {sma_fast} > SMA {sma_slow}; cash saat SMA cepat <= SMA lambat. "
            f"Biaya simulasi {fee_bps:.1f} bps ketika posisi berubah."
        )

        perf = calculate_performance(data)

        if not perf:
            st.warning("Data belum cukup untuk menghitung metrik backtest.")
        else:
            final_value = initial_capital * float(data["strategy_equity"].iloc[-1])
            buy_hold_value = initial_capital * float(data["buy_hold_equity"].iloc[-1])

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("STRATEGY FINAL VALUE", f"${final_value:,.2f}", fmt_pct(perf["Total Return"]))
            b2.metric("BUY & HOLD VALUE", f"${buy_hold_value:,.2f}")
            b3.metric("SHARPE RATIO", f"{perf['Sharpe Ratio']:.2f}" if pd.notna(perf["Sharpe Ratio"]) else "N/A")
            b4.metric("MAX STRATEGY DD", fmt_pct(perf["Max Drawdown"]))

            b5, b6, b7 = st.columns(3)
            b5.metric("WIN RATE", fmt_pct(perf["Win Rate"]))
            pf = perf["Profit Factor"]
            b6.metric("PROFIT FACTOR", f"{pf:.2f}" if pd.notna(pf) and np.isfinite(pf) else "N/A")
            b7.metric("CAGR (APPROX.)", fmt_pct(perf["CAGR"]))

            eqfig = go.Figure()
            eqfig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=initial_capital * data["strategy_equity"],
                    mode="lines",
                    name="SMA Strategy",
                    line=dict(color="#F59E0B", width=2),
                )
            )
            eqfig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=initial_capital * data["buy_hold_equity"],
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#65A5FF", width=1.5),
                )
            )
            eqfig.update_layout(
                title="Equity Curve",
                height=420,
                template="plotly_dark",
                paper_bgcolor="#090B0E",
                plot_bgcolor="#090B0E",
                yaxis_title="Portfolio value ($)",
                margin=dict(l=10, r=10, t=50, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(eqfig, use_container_width=True)

            ddfig = go.Figure()
            ddfig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["strategy_drawdown"] * 100,
                    mode="lines",
                    name="Strategy Drawdown",
                    line=dict(color="#F05D5E", width=1.4),
                    fill="tozeroy",
                )
            )
            ddfig.update_layout(
                title="Strategy Drawdown (%)",
                height=280,
                template="plotly_dark",
                paper_bgcolor="#090B0E",
                plot_bgcolor="#090B0E",
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(ddfig, use_container_width=True)

            st.markdown("#### Performance metrics")
            perf_table = pd.DataFrame(
                {
                    "Metric": list(perf.keys()),
                    "Value": [
                        fmt_pct(v) if k in ["Total Return", "CAGR", "Max Drawdown", "Win Rate"]
                        else (f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v)
                              else str(v))
                        for k, v in perf.items()
                    ],
                }
            )
            st.dataframe(perf_table, use_container_width=True, hide_index=True)

            st.warning(
                "Backtest ini hanya riset historis. Belum mencakup slippage, spread aktual, "
                "pajak, funding, short selling, atau eksekusi broker. Hasil masa lalu tidak "
                "menjamin hasil di masa depan."
            )

    # -------------------- DATA EXPLORER ----------------------

    if page == "Data Explorer":
        st.markdown('<div class="section-title">02 / HISTORICAL DATA EXPLORER</div>', unsafe_allow_html=True)

        st.write(
            f"Dataset: **{asset_name}** | **{timeframe}** | "
            f"{data.index.min().strftime('%Y-%m-%d %H:%M UTC')} — "
            f"{data.index.max().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        clean_count = int(data.isna().sum().sum())
        inf_count = int(np.isinf(data.select_dtypes(include=[np.number])).sum().sum())

        d1, d2, d3 = st.columns(3)
        d1.metric("ROWS", f"{len(data):,}")
        d2.metric("MISSING CELLS", f"{clean_count:,}")
        d3.metric("INFINITE VALUES", f"{inf_count:,}")

        st.dataframe(
            data.sort_index(ascending=False),
            use_container_width=True,
            height=480,
        )

        csv_data = data.to_csv(index=True).encode("utf-8")
        excel_data = make_excel_bytes(data)

        x1, x2 = st.columns(2)
        with x1:
            st.download_button(
                "⬇ Download CSV",
                data=csv_data,
                file_name="VIN_QUANT_market_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with x2:
            st.download_button(
                "⬇ Download Excel",
                data=excel_data,
                file_name="VIN_QUANT_market_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # -------------------- FOOTER --------------------------------

    st.markdown("---")
    st.markdown(
        """
        <div class="small-note">
        VIN QUANT • Quantitative Research Terminal<br>
        Informasi untuk riset dan edukasi, bukan rekomendasi investasi.
        Data publik dapat mengalami keterlambatan, pembatasan akses, atau gangguan.
        </div>
        """,
        unsafe_allow_html=True,
    )

except requests.exceptions.RequestException as exc:
    st.error("Gagal mengambil data dari sumber pasar.")
    st.code(str(exc))
    st.info(
        "Periksa koneksi internet, lalu klik Refresh data. Jika aset kripto gagal, "
        "coba BTC / USD atau ETH / USD. Jika aset Yahoo gagal, coba lagi atau pilih aset lain."
    )

except Exception as exc:
    st.error("Aplikasi menemui kendala saat mengambil atau mengolah data.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Coba refresh data. Bila masih gagal, catat pesan error di atas agar bisa diperbaiki "
        "berdasarkan error yang sebenarnya."
    )
