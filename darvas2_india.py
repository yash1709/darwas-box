"""Replica of 'Advanced Darvas Box 2.0 - FIFO & LIFO' (Fire in India) on the Nifty 50.

Rules taken from the video transcript:
  * Weekly Darvas: GTT buy at (previous week's high + Rs 0.05); fills when price crosses it.
  * One buy per stock per week.
  * Ignorable range: skip the buy if the trigger is within +/-2.75% of the LAST buy price.
  * Sizing: capital / max_stocks / parts(max_lots) per buy; compounding of realised profit
    (less an optional 'incentive' share you keep aside).
  * Scanner: among stocks not held, rank by fall from 52w-high close; only "BOH eligible"
    (52w low is more recent than 52w high) -> biggest drop first. Held stocks keep averaging.
  * No new stock once max_stocks are held.
  * FIFO: sell the whole position when the AVERAGE price reaches +target.
    LIFO: sell each lot individually when it reaches +target on its own buy price.

Usage:
    py darvas2_india.py                    # single run with video defaults (both modes)
    py darvas2_india.py --sweep            # sweep max lots x max stocks (x targets)
"""
import argparse
import os
import pickle
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

NIFTY50 = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
           "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
           "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO",
           "INFY", "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
           "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
           "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "MAXHEALTH"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "nifty50.pkl")
LOOKBACK = 252


@dataclass(frozen=True)
class P:
    capital: float = 200_000.0
    max_stocks: int = 4
    max_lots: int = 6            # "parts" per stock
    target: float = 0.05
    ignore: float = 0.0275
    tick: float = 0.05           # 5 paise above last week's high
    incentive: float = 0.0       # share of profit withdrawn instead of compounded
    cost: float = 0.0005         # per side
    trail: bool = False          # trailing profit: after target, sell when low < yesterday's low


# ---------------------------------------------------------------- data
def load_panel(start, end=None, refresh=False):
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, "rb") as f:
            raw = pickle.load(f)
    else:
        import yfinance as yf
        raw = {}
        for t in NIFTY50:
            try:
                d = yf.download(t + ".NS", start=start, end=end, auto_adjust=False, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Open", "High", "Low", "Close"]].dropna()
                if len(d) > LOOKBACK + 100:
                    raw[t] = d
            except Exception as e:
                print(f"skip {t}: {e}")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(raw, f)
    idx = sorted(set().union(*[d.index for d in raw.values()]))
    idx = pd.DatetimeIndex(idx)
    names = list(raw)
    arr = {k: np.full((len(idx), len(names)), np.nan) for k in ["Open", "High", "Low", "Close"]}
    for j, t in enumerate(names):
        d = raw[t].reindex(idx)
        for k in arr:
            arr[k][:, j] = d[k].to_numpy()
    return idx, names, arr


def prepare(idx, arr):
    """Previous-week high per day, new-week flags, and scanner metrics (known at prior close)."""
    O, H, L, C = arr["Open"], arr["High"], arr["Low"], arr["Close"]
    T, N = C.shape
    wk = pd.Series(idx.isocalendar().year.to_numpy() * 100 + idx.isocalendar().week.to_numpy())
    wk_id = wk.ne(wk.shift()).cumsum().to_numpy()               # increments each new week
    new_week = np.r_[True, wk_id[1:] != wk_id[:-1]]
    wk_high = pd.DataFrame(H).groupby(wk_id).max()               # high of each week
    prev_high = np.full((T, N), np.nan)
    for t in range(T):
        w = wk_id[t] - 1
        if w in wk_high.index:
            prev_high[t] = wk_high.loc[w].to_numpy()

    dist = np.full((T, N), np.nan)                               # drop from 52w-high close
    boh = np.zeros((T, N), bool)
    for j in range(N):
        c = C[:, j]
        for t in range(LOOKBACK, T):
            w = c[t - LOOKBACK + 1:t + 1]
            if np.isnan(w).any():
                continue
            hi, lo = w.argmax(), w.argmin()
            dist[t, j] = 1 - w[-1] / w[hi]
            boh[t, j] = lo > hi
    return dict(new_week=new_week, prev_high=prev_high, dist=dist, boh=boh)


# ---------------------------------------------------------------- simulation
def xirr(flows):
    """flows: list of (datetime, amount). Bisection on annual rate."""
    if not flows or all(a >= 0 for _, a in flows) or all(a <= 0 for _, a in flows):
        return np.nan
    t0 = flows[0][0]
    yrs = np.array([(d - t0).days / 365.0 for d, _ in flows])
    amt = np.array([a for _, a in flows])
    f = lambda r: (amt / (1 + r) ** yrs).sum()
    lo, hi = -0.99, 10.0
    if f(lo) * f(hi) > 0:
        return np.nan
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def simulate(idx, names, arr, pre, mode, p=P()):
    O, H, L, C = arr["Open"], arr["High"], arr["Low"], arr["Close"]
    T, N = C.shape
    ph, dist, boh, new_week = pre["prev_high"], pre["dist"], pre["boh"], pre["new_week"]
    pos = {}                        # j -> list of lots [qty, price, day]
    last_buy = {}
    watch, bought = [], set()
    base = p.capital                # compounding capital
    realised = 0.0
    flows, trades = [], []
    equity = np.full(T, np.nan)
    deployed = np.zeros(T)
    start = LOOKBACK + 1

    def sell_lot(j, lot, px, t, why):
        nonlocal realised, base
        qty, bpx, d = lot[:3]
        net = qty * (px - bpx) - p.cost * qty * (px + bpx)
        realised += net
        base += net * (1 - p.incentive)
        flows.append((idx[t], qty * px * (1 - p.cost)))
        trades.append(dict(ticker=names[j], buy_day=idx[d], sell_day=idx[t], qty=qty, buy=bpx, sell=px,
                           pnl=net, days=(idx[t] - idx[d]).days, why=why))

    for t in range(start, T):
        if new_week[t]:
            bought = set()
            held = [j for j in pos if pos[j]]
            free = p.max_stocks - len(held)
            cand = [j for j in range(N) if j not in pos or not pos[j]]
            cand = [j for j in cand if boh[t - 1, j] and not np.isnan(dist[t - 1, j]) and not np.isnan(ph[t, j])]
            cand.sort(key=lambda j: -dist[t - 1, j])
            watch = held + cand[:max(free, 0)]

        # --- sells (lots bought on earlier days only)
        for j in list(pos):
            lots = pos[j]
            if not lots or np.isnan(H[t, j]):
                continue
            pl = L[t - 1, j]                      # yesterday's low (trailing-profit stop)
            if mode == "FIFO":
                qty = sum(l[0] for l in lots)
                avg = sum(l[0] * l[1] for l in lots) / qty
                tpx = avg * (1 + p.target)
                if p.trail and all(l[3] for l in lots):
                    if L[t, j] < pl:
                        px = min(O[t, j], pl)
                        for lot in lots:
                            sell_lot(j, lot, px, t, "FIFO trail")
                        pos[j] = []
                elif H[t, j] >= tpx:
                    if p.trail:
                        for lot in lots:
                            lot[3] = True
                    else:
                        px = max(O[t, j], tpx)
                        for lot in lots:
                            sell_lot(j, lot, px, t, "FIFO target")
                        pos[j] = []
            else:
                for lot in list(lots):
                    tpx = lot[1] * (1 + p.target)
                    if p.trail and lot[3]:
                        if L[t, j] < pl:
                            sell_lot(j, lot, min(O[t, j], pl), t, "LIFO trail")
                            lots.remove(lot)
                    elif H[t, j] >= tpx:
                        if p.trail:
                            lot[3] = True
                        else:
                            sell_lot(j, lot, max(O[t, j], tpx), t, "LIFO target")
                            lots.remove(lot)
            if not pos[j]:
                last_buy.pop(j, None)

        # --- buys (GTT crossing)
        for j in watch:
            if j in bought or np.isnan(H[t, j]) or np.isnan(ph[t, j]):
                continue
            lots = pos.get(j, [])
            nheld = sum(1 for k in pos if pos[k])
            if not lots and nheld >= p.max_stocks:
                continue
            if len(lots) >= p.max_lots:
                continue
            trig = ph[t, j] + p.tick
            if j in last_buy and last_buy[j] * (1 - p.ignore) <= trig <= last_buy[j] * (1 + p.ignore):
                continue
            if H[t, j] >= trig:
                px = max(O[t, j], trig)
                qty = int((base / p.max_stocks / p.max_lots) // px)
                if qty < 1:
                    continue
                for l in lots:
                    l[3] = False      # average changed -> re-arm
                pos.setdefault(j, []).append([qty, px, t, False])
                last_buy[j] = trig
                bought.add(j)
                flows.append((idx[t], -qty * px * (1 + p.cost)))

        mtm = sum(l[0] * (C[t, j] if not np.isnan(C[t, j]) else l[1]) for j in pos for l in pos[j])
        cost_basis = sum(l[0] * l[1] for j in pos for l in pos[j])
        equity[t] = p.capital + realised + (mtm - cost_basis)
        deployed[t] = cost_basis

    end_mtm = sum(l[0] * (C[-1, j] if not np.isnan(C[-1, j]) else l[1]) for j in pos for l in pos[j])
    open_cost = sum(l[0] * l[1] for j in pos for l in pos[j])
    flows_x = flows + ([(idx[-1], end_mtm)] if end_mtm else [])
    tr = pd.DataFrame(trades)
    eq = pd.Series(equity, index=idx).dropna()
    tr.attrs["equity"] = eq
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    gp = tr.pnl[tr.pnl > 0].sum() if len(tr) else 0
    gl = -tr.pnl[tr.pnl <= 0].sum() if len(tr) else 0
    return dict(Mode=mode, XIRR=xirr(sorted(flows_x, key=lambda x: x[0])),
                CAGR_cap=(eq.iloc[-1] / p.capital) ** (1 / yrs) - 1, MaxDD=dd,
                Trades=len(tr), WinRate=(tr.pnl > 0).mean() if len(tr) else np.nan,
                ProfitFactor=gp / gl if gl > 0 else np.nan, AvgDays=tr.days.mean() if len(tr) else np.nan,
                NetProfit=realised, MaxDeployed=deployed.max(), StuckCost=open_cost,
                StuckMTM=end_mtm - open_cost, EndEquity=eq.iloc[-1]), tr


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--trail", action="store_true", help="trailing-profit exit instead of fixed target")
    ap.add_argument("--compare-trail", action="store_true", help="run fixed vs trailing side by side")
    ap.add_argument("--lots", default="3,4,6,8,10")
    ap.add_argument("--stocks", default="2,4,6,8")
    ap.add_argument("--targets", default="5")
    a = ap.parse_args()

    idx, names, arr = load_panel(a.start, refresh=a.refresh)
    print(f"{len(names)} Nifty50 stocks, {idx[0].date()} -> {idx[-1].date()} ({len(idx)} days)")
    pre = prepare(idx, arr)
    pd.set_option("display.width", 220, "display.float_format", "{:,.3f}".format)

    base_p = replace(P(), trail=a.trail)
    if a.compare_trail:
        rows = [dict(Exit="trail" if tr else "fixed", **simulate(idx, names, arr, pre, m, replace(P(), trail=tr))[0])
                for tr in (False, True) for m in ("FIFO", "LIFO")]
        print(pd.DataFrame(rows).to_string(index=False))
        return
    if not a.sweep:
        rows = [simulate(idx, names, arr, pre, m, base_p)[0] for m in ("FIFO", "LIFO")]
        print(pd.DataFrame(rows).to_string(index=False))
        return

    lots = [int(x) for x in a.lots.split(",")]
    stocks = [int(x) for x in a.stocks.split(",")]
    targets = [float(x) / 100 for x in a.targets.split(",")]
    rows = []
    for ml, ms, tg, m in product(lots, stocks, targets, ("FIFO", "LIFO")):
        r, _ = simulate(idx, names, arr, pre, m, replace(base_p, max_lots=ml, max_stocks=ms, target=tg))
        rows.append(dict(MaxLots=ml, MaxStocks=ms, Target=tg, **r))
        print(f"lots={ml:>2} stocks={ms} tgt={tg:.0%} {m}: XIRR={r['XIRR']:.3f} DD={r['MaxDD']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv("darvas2_sweep.csv", index=False)
    print("\n", res.sort_values("XIRR", ascending=False).head(15).to_string(index=False))
    print("\n=== Mean by mode ===")
    print(res.groupby("Mode")[["XIRR", "CAGR_cap", "MaxDD", "WinRate", "ProfitFactor", "AvgDays"]].mean())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tg0 = targets[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, m in enumerate(("FIFO", "LIFO")):
        pv = res[(res.Mode == m) & (res.Target == tg0)].pivot(index="MaxStocks", columns="MaxLots", values="XIRR")
        im = ax[k].imshow(pv.values, cmap="RdYlGn", aspect="auto")
        ax[k].set_xticks(range(len(pv.columns)), pv.columns)
        ax[k].set_yticks(range(len(pv.index)), pv.index)
        ax[k].set_title(f"{m} XIRR (target {tg0:.0%})")
        ax[k].set_xlabel("max lots per stock")
        ax[k].set_ylabel("max stocks")
        for (i, j), v in np.ndenumerate(pv.values):
            ax[k].text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax[k])
    plt.tight_layout()
    plt.savefig("darvas2_sweep.png", dpi=120)
    print("saved darvas2_sweep.csv, darvas2_sweep.png")


if __name__ == "__main__":
    main()
