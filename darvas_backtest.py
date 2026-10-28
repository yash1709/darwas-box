"""Backtest of the Darvas Box 2.0 strategy: FIFO vs LIFO partial-exit ordering.

Mirrors darvas_box_2_fifo_lifo.pine: signals on the close, fills at next open,
stops filled intrabar at min(open, stop).

Usage:
    py darvas_backtest.py                       # default tickers, 2010 -> today
    py darvas_backtest.py AAPL NVDA --start 2015-01-01
    py darvas_backtest.py --csv data.csv        # CSV with Date,Open,High,Low,Close,Volume
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "AVGO", "COST"]


@dataclass
class Params:
    near_high_pct: float = 10.0
    high_len: int = 252
    fast_len: int = 20
    slow_len: int = 50
    confirm_bars: int = 3
    use_vol: bool = True
    vol_len: int = 20
    risk_pct: float = 1.0
    max_lots: int = 4
    trail_all: bool = True
    partial_on: bool = True
    commission: float = 0.0005  # fraction per side
    capital: float = 100_000.0


def load(ticker, start, end):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def backtest(df, mode, p=Params()):
    """Run one instrument. mode is 'FIFO' or 'LIFO'. Returns (equity Series, trades DataFrame)."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    n = len(df)
    sma_f = df["Close"].rolling(p.fast_len).mean().to_numpy()
    sma_s = df["Close"].rolling(p.slow_len).mean().to_numpy()
    hh = df["High"].rolling(p.high_len, min_periods=1).max().to_numpy()
    vol_ma = df["Volume"].rolling(p.vol_len).mean().to_numpy()

    cash = p.capital
    lots, trades = [], []          # lot: dict(qty, px, stop, i)
    pend_entry = None              # (risk_qty, stop)
    pend_close = []                # lots to close at next open
    pend_all = False
    cand_top = cand_bot = None
    cnt = 0
    box_top = box_bot = None
    box_live = False
    last_entry = None
    equity = np.full(n, np.nan)

    def close_lot(lot, px, i, why):
        nonlocal cash
        cash += lot["qty"] * px * (1 - p.commission)
        trades.append(dict(entry_i=lot["i"], exit_i=i, qty=lot["qty"], entry=lot["px"], exit=px,
                           pnl=lot["qty"] * (px - lot["px"]) - p.commission * lot["qty"] * (px + lot["px"]),
                           bars=i - lot["i"], why=why))
        lots.remove(lot)

    for i in range(n):
        # 1) fill orders queued on the previous close, at this open
        if pend_all:
            for lot in list(lots):
                close_lot(lot, o[i], i, "trend")
        else:
            for lot in pend_close:
                if lot in lots:
                    close_lot(lot, o[i], i, f"{mode} peel")
        pend_all, pend_close = False, []
        if pend_entry is not None:
            risk_qty, stop = pend_entry
            qty = min(risk_qty, cash / (o[i] * (1 + p.commission)))
            if qty > 0 and o[i] > stop:
                cash -= qty * o[i] * (1 + p.commission)
                lots.append(dict(qty=qty, px=o[i], stop=stop, i=i))
            pend_entry = None

        # 2) intrabar stops
        for lot in list(lots):
            if l[i] <= lot["stop"]:
                close_lot(lot, min(o[i], lot["stop"]), i, "stop")

        equity[i] = cash + sum(lt["qty"] * c[i] for lt in lots)
        if not lots:
            last_entry = None

        # 3) Darvas box on this bar
        if cand_top is None or h[i] > cand_top:
            cand_top, cand_bot, cnt = h[i], None, 0
        else:
            cnt += 1
            cand_bot = l[i] if cand_bot is None else min(cand_bot, l[i])
            if cnt == p.confirm_bars:
                box_top, box_bot, box_live = cand_top, cand_bot, True
        if box_live and c[i] < box_bot:
            box_live = False
        if np.isnan(sma_s[i]) or np.isnan(vol_ma[i]):
            continue

        trend_ok = c[i] >= hh[i] * (1 - p.near_high_pct / 100) and c[i] > sma_f[i] and c[i] > sma_s[i]
        vol_ok = (not p.use_vol) or v[i] > vol_ma[i]
        n_lots = len(lots) + (1 if pend_entry else 0)

        # 4) signals at the close -> queued for next open
        if box_live and c[i] > box_top and trend_ok and vol_ok and n_lots < p.max_lots \
                and (last_entry is None or box_top > last_entry) and box_bot < c[i]:
            qty = equity[i] * p.risk_pct / 100 / (c[i] - box_bot)
            pend_entry = (qty, box_bot)
            last_entry = c[i]
            if p.trail_all:
                for lt in lots:
                    lt["stop"] = max(lt["stop"], box_bot)
            box_live = False

        if lots:
            if c[i] < sma_s[i]:
                pend_all = True
            elif p.partial_on and len(lots) > 1 and c[i] < sma_f[i]:
                pend_close = [lots[0] if mode == "FIFO" else lots[-1]]

    # liquidate at the last close so open P&L is counted as a trade
    for lot in list(lots):
        close_lot(lot, c[-1], n - 1, "end")
    return pd.Series(equity, index=df.index), pd.DataFrame(trades)


def metrics(eq, trades):
    eq = eq.dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    if len(trades):
        w, ls = trades.loc[trades.pnl > 0, "pnl"], trades.loc[trades.pnl <= 0, "pnl"]
        pf = w.sum() / -ls.sum() if ls.sum() < 0 else np.inf
        stats = dict(Lots=len(trades), WinRate=(trades.pnl > 0).mean(), ProfitFactor=pf,
                     AvgPnL=trades.pnl.mean(), AvgBars=trades.bars.mean())
    else:
        stats = dict(Lots=0, WinRate=np.nan, ProfitFactor=np.nan, AvgPnL=np.nan, AvgBars=np.nan)
    return dict(TotalRet=ret, CAGR=(1 + ret) ** (1 / yrs) - 1, MaxDD=dd, Sharpe=sharpe, **stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--csv", help="single CSV instead of downloading")
    ap.add_argument("--plot", default="darvas_fifo_vs_lifo.png")
    a = ap.parse_args()

    data = {}
    if a.csv:
        data["CSV"] = pd.read_csv(a.csv, index_col=0, parse_dates=True)
    else:
        for t in a.tickers:
            try:
                d = load(t, a.start, a.end)
                if len(d) > 300:
                    data[t] = d
            except Exception as e:
                print(f"skip {t}: {e}")
    if not data:
        raise SystemExit("no data")

    rows, curves = [], {"FIFO": [], "LIFO": []}
    for t, df in data.items():
        for mode in ("FIFO", "LIFO"):
            eq, tr = backtest(df, mode)
            curves[mode].append(eq / eq.dropna().iloc[0])
            rows.append(dict(Ticker=t, Mode=mode, **metrics(eq, tr)))
    res = pd.DataFrame(rows)

    pd.set_option("display.width", 200, "display.float_format", "{:,.3f}".format)
    print(res.to_string(index=False))
    print("\n=== Average across tickers ===")
    print(res.groupby("Mode")[["TotalRet", "CAGR", "MaxDD", "Sharpe", "WinRate", "ProfitFactor", "AvgBars"]]
          .mean().to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for mode, cs in curves.items():
        plt.plot(pd.concat(cs, axis=1).ffill().mean(axis=1), label=mode)
    plt.title("Equal-weight avg equity (start = 1.0)")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(a.plot, dpi=120)
    print(f"\nsaved {a.plot}")


if __name__ == "__main__":
    main()
