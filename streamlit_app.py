"""Streamlit app: Advanced Darvas Box 2.0 (weekly, Nifty 50) - FIFO vs LIFO backtest and parameter sweep."""
from dataclasses import replace
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import darvas2_india as dv

st.set_page_config(page_title="Darvas Box 2.0 - Nifty 50", layout="wide")


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading Nifty 50 data (downloads latest prices if stale)...")
def get_data(refresh_token: int):
    idx, names, arr = dv.load_panel("2014-01-01")
    expected = pd.Timestamp.today().normalize() - pd.offsets.BDay(1)
    if refresh_token > 0 or idx[-1] < expected:
        try:
            idx, names, arr = dv.load_panel("2014-01-01", refresh=True)
        except Exception as e:      # Yahoo down / rate-limited: keep the bundled cache
            st.session_state["refresh_error"] = str(e)
    return idx, names, arr


@st.cache_resource(show_spinner="Computing scanner metrics (one-off)...")
def get_pre(refresh_token: int, data_key: str):
    idx, names, arr = get_data(refresh_token)
    return dv.prepare(idx, arr)


@st.cache_data(show_spinner=False)
def run_one(token, data_key, mode, params):
    idx, names, arr = get_data(token)
    r, tr = dv.simulate(idx, names, arr, get_pre(token, data_key), mode, params)
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
data_key = str(idx[-1].date())
pre = get_pre(token, data_key)
if "refresh_error" in st.session_state:
    st.sidebar.warning(f"Could not refresh prices, using bundled data: {st.session_state['refresh_error']}")
st.sidebar.caption(f"{len(names)} stocks | {idx[0].date()} -> {idx[-1].date()}")
params = dv.P(capital=float(capital), max_stocks=max_stocks, max_lots=max_lots, target=target,
              ignore=ignore, incentive=incentive, cost=cost, trail=trail)

tab0, tab1, tab2 = st.tabs(["Signals / Scanner", "Backtest: FIFO vs LIFO", "Parameter sweep"])

with tab0:
    st.subheader(f"Signals as of {idx[-1].date()} close")
    st.caption("GTT buy = last completed week's high + Rs 0.05. Candidates = stocks you don't hold, 52w-low more recent "
               "than 52w-high (BOH eligible), ranked by biggest fall from the 52w-high close. "
               "Prices come from Yahoo Finance and can lag; verify with your broker before placing orders.")
    st.markdown("**Your current holdings** (one row per lot; edit, paste or upload a CSV with "
                "`Ticker, Date, BuyPrice, Qty`). Needed for the ignore-range check and sell signals. "
                "Not saved between sessions - download it to keep it.")
    up = st.file_uploader("Upload holdings CSV", type="csv")
    if up is not None and st.session_state.get("hold_file") != up.name:
        st.session_state.holdings = pd.read_csv(up)
        st.session_state.hold_file = up.name
    if "holdings" not in st.session_state:
        st.session_state.holdings = pd.DataFrame({"Ticker": pd.Series(dtype=str), "Date": pd.Series(dtype=str),
                                                  "BuyPrice": pd.Series(dtype=float), "Qty": pd.Series(dtype=float)})
    hold = st.data_editor(st.session_state.holdings, num_rows="dynamic", width="stretch",
                          column_config={"Ticker": st.column_config.SelectboxColumn(options=names)})
    st.download_button("Download holdings (CSV)", hold.to_csv(index=False), "holdings.csv")
    sig = dv.current_signals(idx, names, arr, pre, params, hold)

    held = sig[sig.Held]
    cand = sig[~sig.Held & sig.BOH]
    st.markdown(f"**Per-buy amount:** Rs {params.capital / params.max_stocks / params.max_lots:,.0f}  |  "
                f"**Stocks held:** {len(held)}/{params.max_stocks}")

    st.markdown("#### 1. Place / update GTT buy orders")
    todo = sig[sig.Action.str.startswith(("CREATE", "UPDATE"))]
    show = ["Rank", "Ticker", "Action", "CMP", "GTT_buy", "GTT_vs_CMP", "Qty", "Amount", "Drop52w",
            "NextWeek_GTT", "TriggeredThisWeek"]
    if todo.empty:
        st.info("No buy GTTs to place right now.")
    else:
        st.dataframe(todo[show], width="stretch", hide_index=True)

    st.markdown("#### 2. Sell signals on holdings")
    if held.empty:
        st.info("Enter your holdings above to see sell targets and averaging status.")
    else:
        cols = ["Ticker", "Lots", "AvgPrice", "CMP", "PnLPct", "FIFO_Target", "LIFO_NextTarget", "SellSignal",
                "BelowYestLow", "YestLow", "LastBuy", "IgnoreLo", "IgnoreHi", "GTT_buy", "Action"]
        st.dataframe(held[cols], width="stretch", hide_index=True)
        if params.trail:
            st.caption("Trailing mode: once the target is hit, hold and sell when the day's low goes below yesterday's low "
                       "(see BelowYestLow).")

    with st.expander("Full scanner (all Nifty 50 stocks)"):
        st.dataframe(sig, width="stretch", hide_index=True)


with tab1:
    res = {m: run_one(token, data_key, m, params) for m in ("FIFO", "LIFO")}
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
            r, _, _ = run_one(token, data_key, m, replace(params, max_lots=ml, max_stocks=ms))
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

