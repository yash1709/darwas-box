"""Streamlit app: Advanced Darvas Box 2.0 (weekly, Nifty 50) - FIFO vs LIFO backtest and parameter sweep."""
from dataclasses import replace
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import darvas2_india as dv

st.set_page_config(page_title="Darvas Box 2.0 - Nifty 50", layout="wide")


@st.cache_resource(show_spinner="Loading Nifty 50 data...")
def get_data(refresh_token: int):
    idx, names, arr = dv.load_panel("2014-01-01", refresh=refresh_token > 0)
    return idx, names, arr


@st.cache_resource(show_spinner="Computing scanner metrics (one-off)...")
def get_pre(refresh_token: int):
    idx, names, arr = get_data(refresh_token)
    return dv.prepare(idx, arr)


@st.cache_data(show_spinner=False)
def run_one(token, mode, params):
    idx, names, arr = get_data(token)
    r, tr = dv.simulate(idx, names, arr, get_pre(token), mode, params)
    return r, tr, tr.attrs["equity"]


st.title("Advanced Darvas Box 2.0 - Nifty 50")
st.caption("Weekly Darvas: buy above last week's high, 2.75% ignorable range, FIFO or LIFO exits. "
           "Educational backtest only - not investment advice.")

if "token" not in st.session_state:
    st.session_state.token = 0

with st.sidebar:
    st.header("Parameters")
    capital = st.number_input("Capital (Rs)", 50_000, 10_000_000, 200_000, 10_000)
    max_stocks = st.slider("Max stocks held", 1, 10, 4)
    max_lots = st.slider("Max lots per stock", 1, 12, 6)
    target = st.slider("Target %", 1.0, 10.0, 5.0, 0.25) / 100
    ignore = st.slider("Ignorable range %", 0.0, 6.0, 2.75, 0.25) / 100
    incentive = st.slider("Incentive % (profit not compounded)", 0, 50, 0) / 100
    trail = st.checkbox("Trailing profit (sell if low < yesterday's low after target)", value=False)
    cost = st.number_input("Cost per side (%)", 0.0, 1.0, 0.05, 0.01) / 100
    if st.button("Re-download data"):
        st.session_state.token += 1
        run_one.clear()
        st.rerun()

token = st.session_state.token
idx, names, arr = get_data(token)
st.sidebar.caption(f"{len(names)} stocks | {idx[0].date()} -> {idx[-1].date()}")
params = dv.P(capital=float(capital), max_stocks=max_stocks, max_lots=max_lots, target=target,
              ignore=ignore, incentive=incentive, cost=cost, trail=trail)

tab1, tab2 = st.tabs(["Backtest: FIFO vs LIFO", "Parameter sweep"])

with tab1:
    res = {m: run_one(token, m, params) for m in ("FIFO", "LIFO")}
    summary = pd.DataFrame([r for r, _, _ in res.values()]).set_index("Mode")
    c = st.columns(4)
    for k, m in enumerate(("FIFO", "LIFO")):
        r = res[m][0]
        c[2 * k].metric(f"{m} XIRR", f"{r['XIRR']:.1%}" if not np.isnan(r["XIRR"]) else "n/a")
        c[2 * k + 1].metric(f"{m} max drawdown", f"{r['MaxDD']:.1%}")
    st.dataframe(summary.style.format("{:,.3f}"), width="stretch")

    st.subheader("Equity (start = capital)")
    st.line_chart(pd.DataFrame({m: res[m][2] for m in res}))

    m = st.radio("Trade log", ["FIFO", "LIFO"], horizontal=True)
    tr = res[m][1].copy()
    tr.attrs = {}
    st.dataframe(tr.reset_index(drop=True).copy(), width="stretch", height=300)
    st.download_button(f"Download {m} trades (CSV)", tr.to_csv(index=False), f"{m}_trades.csv")

with tab2:
    st.write("Sweeps max lots per stock x max stocks (position size) for both modes. "
             "Other parameters come from the sidebar. Each run takes about a second.")
    lots = st.multiselect("Max lots", list(range(1, 13)), [3, 4, 6, 8, 10])
    stocks = st.multiselect("Max stocks", list(range(1, 11)), [2, 4, 6, 8])
    metric = st.selectbox("Heatmap metric", ["XIRR", "CAGR_cap", "MaxDD", "WinRate", "AvgDays"])
    if st.button("Run sweep") and lots and stocks:
        combos = list(product(lots, stocks, ("FIFO", "LIFO")))
        bar, rows = st.progress(0.0), []
        for i, (ml, ms, m) in enumerate(combos):
            r, _, _ = run_one(token, m, replace(params, max_lots=ml, max_stocks=ms))
            rows.append(dict(MaxLots=ml, MaxStocks=ms, **r))
            bar.progress((i + 1) / len(combos))
        st.session_state.sweep = pd.DataFrame(rows)
    if "sweep" in st.session_state:
        sw = st.session_state.sweep
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for k, m in enumerate(("FIFO", "LIFO")):
            pv = sw[sw.Mode == m].pivot(index="MaxStocks", columns="MaxLots", values=metric)
            im = ax[k].imshow(pv.values, cmap="RdYlGn", aspect="auto")
            ax[k].set_xticks(range(len(pv.columns)), pv.columns)
            ax[k].set_yticks(range(len(pv.index)), pv.index)
            ax[k].set_title(f"{m} {metric}")
            ax[k].set_xlabel("max lots")
            ax[k].set_ylabel("max stocks")
            for (i, j), v in np.ndenumerate(pv.values):
                ax[k].text(j, i, f"{v:.1%}" if metric != "AvgDays" else f"{v:.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax[k])
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(sw.sort_values("XIRR", ascending=False), width="stretch")
        st.download_button("Download sweep (CSV)", sw.to_csv(index=False), "sweep.csv")

