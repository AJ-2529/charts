# streamlit_ticker_chart_app.py
# Streamlit app that lets you enter a ticker, download CSV via yfinance,
# or upload a CSV; then it plots candlestick charts with SMA and volume.

import io
import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Ticker → Candlestick Chart", layout="wide")


@st.cache_data
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, parse dates, convert numeric columns.
    Expects CSV with Date, Open, High, Low, Close, optional Adj Close and Volume.
    """
    df = df.copy()
    # strip whitespace from headers
    df.columns = [c.strip() for c in df.columns]

    # map common names to canonical names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "date" in lc:
            col_map[c] = "Date"
        elif lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc in ("close", "adj close", "adj_close", "adjclose"):
            # prefer 'Adj Close' if explicitly labeled; otherwise 'Close'
            if "adj" in lc:
                col_map[c] = "Adj Close"
            else:
                col_map[c] = "Close"
        elif "volume" in lc:
            col_map[c] = "Volume"

    df = df.rename(columns=col_map)

    # if Adj Close exists but Close missing, copy Adj Close to Close
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]

    # validate required columns
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    # parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # numeric conversion: remove commas, coerce errors to NaN
    for col in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").replace(["", "nan", "None", "none"], pd.NA)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_figure(df: pd.DataFrame, show_volume: bool = True, sma: Optional[int] = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            hovertemplate="%{x|%Y-%m-%d}<br>O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}<extra></extra>",
        )
    )

    if sma is not None and sma >= 1:
        df[f"SMA_{sma}"] = df["Close"].rolling(sma).mean()
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[f"SMA_{sma}"],
                mode="lines",
                line=dict(width=1.6, color="#FFD54F"),
                name=f"SMA {sma}",
            )
        )

    if show_volume and "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"].fillna(0),
                name="Volume",
                marker_color="#4c8cff",
                opacity=0.6,
                yaxis="y2",
                hovertemplate="%{x|%Y-%m-%d}<br>Volume: %{y:.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Price", domain=[0.25, 1.0]),
            yaxis2=dict(title="Volume", domain=[0.0, 0.22], overlaying=None, anchor="x"),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=10, b=30, l=60, r=30),
        )
    else:
        fig.update_layout(
            yaxis=dict(title="Price"),
            xaxis_rangeslider_visible=False,
            margin=dict(t=10, b=30, l=60, r=30),
        )

    fig.update_layout(autosize=True, template="plotly_dark")
    return fig


@st.cache_data
def download_ticker_csv(ticker: str, period: str = "max", filename: Optional[str] = None) -> str:
    """
    Download ticker via yfinance and save to CSV. Returns the path to the CSV.
    """
    ticker = ticker.strip()
    if not ticker:
        raise ValueError("Ticker empty")

    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}")

    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols]

    if filename is None:
        filename = f"{ticker.replace('^','').replace('/','_')}.csv"

    df.to_csv(filename, index=False)
    return filename


def load_csv_bytes(b: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b))
    return clean_dataframe(df)


def load_csv_path(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return clean_dataframe(df)


def main():
    # Page header
    st.title("Money Maker 101")
    st.markdown("***")

    st.markdown(
        """
        **Enter a ticker symbol below and click _Download_.**
        The app will fetch the historical OHLCV data, display a clean candlestick chart, and offer SMA & Volume toggles.
        """
    )

    # Layout: large left column for chart, smaller right column for controls
    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.subheader("Chart controls")
        period = st.selectbox("Download period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=7)
        st.checkbox("Show Volume", value=True, key="show_volume")
        st.checkbox("Show SMA", value=True, key="show_sma")
        sma_window = st.slider("SMA window", min_value=5, max_value=200, value=20, step=1)

    # Prominent ticker input placed under the header in the left column
    with col_left:
        st.subheader("Ticker")
        ticker = st.text_input("Enter ticker (eg. MSFT or ^NSEI)", placeholder="Type ticker and press Download")
        download_btn = st.button("Load Ticker")

    df = None
    load_error = None
    csv_path = None

    # Download flow (no CSV upload or local CSV fallback)
    if download_btn:
        if not ticker or not ticker.strip():
            st.error("Please enter a valid ticker before clicking Download.")
            return
        try:
            with st.spinner(f"Downloading {ticker} ..."):
                csv_path = download_ticker_csv(ticker, period=period)
                df = load_csv_path(csv_path)
                st.success(f"All The Best!!!")
        except Exception as e:
            load_error = f"Download failed: {e}"

    if load_error:
        st.error(load_error)
        return

    if df is None:
        st.info("Enter a ticker and click Download to fetch data.")
        return

    # date range selector
    min_date, max_date = df["Date"].min(), df["Date"].max()
    start_date, end_date = st.date_input("Date range", [min_date.date(), max_date.date()], min_value=min_date.date(), max_value=max_date.date())
    if isinstance(start_date, (list, tuple)):
        start_date, end_date = start_date[0], start_date[1]

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_f = df.loc[mask].copy()
    st.write(f"Showing {len(df_f)} rows from {start_date} to {end_date}")

    # plot
    sma_val = sma_window if st.session_state.get("show_sma", True) else None
    show_volume = st.session_state.get("show_volume", True)
    fig = make_figure(df_f, show_volume=show_volume, sma=sma_val)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    # data table & download
    with st.expander("Data & export"):
        st.dataframe(df_f.head(500))
        csv_bytes = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name=(csv_path or "filtered.csv"), mime="text/csv")

    # also offer the full CSV if we downloaded
    if csv_path and os.path.exists(csv_path):
        with st.expander("Original CSV"):
            st.write(csv_path)
            with open(csv_path, "rb") as f:
                st.download_button("Download original CSV", data=f, file_name=os.path.basename(csv_path), mime="text/csv")


if __name__ == "__main__":
    main()
