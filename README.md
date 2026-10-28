# Darvas Box 2.0 (weekly, Nifty 50)

Replica of the "Advanced Darvas Box 2.0 - FIFO & LIFO" strategy (Fire in India), with a backtest, parameter sweep and Streamlit app. Educational only, not investment advice.

| File | What it is |
|---|---|
| `streamlit_app.py` | Streamlit app (backtest FIFO vs LIFO, sweep, trade log) |
| `darvas2_india.py` | Backtest engine + CLI (`py darvas2_india.py --sweep --trail`) |
| `darvas2_weekly.pine` | Single-symbol TradingView strategy (daily chart) |
| `data_cache/nifty50.pkl` | Cached daily OHLC for Nifty 50 (so the app starts without Yahoo) |
| `darvas_backtest.py`, `darvas_box_2_fifo_lifo.pine` | Early daily-box version; superseded, kept for reference |

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. share.streamlit.io -> New app -> repo `yash1709/darwas-box`, branch `main`, main file `streamlit_app.py`.
2. Deploy. First load computes scanner metrics once (~20-40 s).
