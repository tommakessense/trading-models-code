"""
trend_protocol.py - trend following on Xetra (Robust 1 and Robust 2) with protocol mode for paper trading
================================================================================================================

Two finished strategies, each as its own account with CAPITAL EUR (without leverage, at most 10 positions of 1/10 each):

    Robust 1   Purchase when a stock rises to a new 200-day high (closing price).
               Sale after exactly 20 trading days.
    Robust 2   Purchase when a stock rises to a new 50-day high AND on that day at least twice as many
               shares are traded as in the median of the last 50 days.
               Sale after 60 trading days - or earlier if the closing price falls 15 % below the highest closing price
               since the purchase (trailing stop).

For both: only stocks with a median turnover of at least 1 million EUR per day (last 250 days) and a price >= 1 EUR.
If there are more signals than free slots, the stocks with the strongest 6-month trend are bought first.
Basis: trend_cpo.py / trend_evaluation.py / trend_cpo_long.py (backtest 2006-2026, walk-forward).

Execution
---------
The signal arises with the closing price of a day. Buying and selling take place in the Xetra closing auction of the
next trading day (order by 17:30, e.g. limited or as an auction order). All backtests
are based on this. The run before the market opens shows the orders for today's closing auction. The stop lines of
Robust 2 apply to today's closing price: If the price is below them shortly before 17:30, sell in the closing
auction.

Costs: order fees of Scalable Capital (FREE, Xetra 1.99 EUR per order) for all stocks tradable there,
otherwise IBKR (0.05 %, min. 3 EUR) - set in trend_model.py (BROKER, SCALABLE_PLAN,
scalable_not_tradable.csv) - plus SLIPPAGE_BPS per order. Taxes are not taken into account.

Usage (in the folder trend_model)
---------------------------------
    ..\\.venv\\python.exe trend_protocol.py --protocol       protocol: update prices, book the day, orders for today
    ..\\.venv\\python.exe trend_protocol.py --protocol-new   start the protocol anew (the old one is archived)
    ..\\.venv\\python.exe trend_protocol.py --backtest   backtest of both strategies with the current costs
    protocol_daily.bat                                    daily run (Windows task, see task_setup.bat)

Protocol mode
-------------
On the first call, the start day, settings and stock list are frozen (protocol/start.json). Signals from the
closing price of the start day onwards are counted; the first purchase is therefore on the next trading day at the earliest. Each run
    - loads the new prices from Yahoo (data/prices_protocol.pkl; the first time from the screener cache),
    - calculates both accounts from the start day and appends the daily status to protocol/protocol.csv
      (existing rows are never changed; days entered later are marked as "added_later"),
    - writes each order with the model price to protocol/executions.csv. Anyone who executes the orders in a demo account
      enters the actual price there in "real_price" (-> measured slippage). If a purchase was
      not possible, enter "no" in "executable": The account is then additionally shown without this trade.
    - shows the orders for today's closing auction, the stop lines and the result since the start compared
      to the DAX and to the equal-weighted index of the same stocks.
The share quantities of the orders are calculated with the closing price of the previous day; booking uses the actual
closing price (1/10 of the account at this price, whole shares), so they can differ by one share.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import trend_cpo as tc
import trend_model as tm

# ======================================================================================= Settings
STRATEGIES = {                   # name: (entry from trend_cpo.ENTRY_RULES, exit from trend_cpo.EXIT_RULES)
    "Robust 1": ("High 200", "Time 20"),
    "Robust 2": ("High 50 + volume surge", "Time 60 + stop 15%"),
}
DESCRIPTION = {
    "Robust 1": "new 200-day high, hold for 20 trading days",
    "Robust 2": "new 50-day high with double volume, hold for 60 trading days, stop 15 % below the high since purchase",
}
CAPITAL = tm.CAPITAL             # a separate account for each strategy
MIN_DAYS_FOR_SHARPE = 60
BENCH = "EXS1.DE"                # iShares Core DAX
DOWNLOAD_CHUNK = 50

BASE_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = BASE_DIR / "protocol"
PRICE_FILE = tm.DATA_DIR / "prices_protocol.pkl"
PRICE_START = BASE_DIR / "data_start" / "prices_protocol.pkl.xz"   # initial stock of prices (e.g. for GitHub Actions)
WEEKDAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def log(msg: str) -> None:
    tm.log(msg)


def eur(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def num(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> None:
    old = tm.OUT_DIR
    tm.OUT_DIR = PROTOCOL_DIR
    tm.save_csv(df, name, index=index)
    tm.OUT_DIR = old


# ======================================================================================= Prices
def _read_screener_cache() -> dict:
    src = tm.SCREENER_CACHE / "prices_stocks.pkl"
    if not src.exists():
        raise SystemExit(f"Price cache of the screener missing: {src}")
    raw = pd.read_pickle(src)
    d = {"close": raw["close"].astype(float), "volume": raw["volume"].astype(float), "bench": pd.DataFrame()}
    etf = tm.SCREENER_CACHE / tm.ETF_FILE
    if etf.exists():
        e = pd.read_pickle(etf)["close"]
        if BENCH in e.columns:
            d["bench"] = e[[BENCH]].astype(float)
    return d


def _merge(old: pd.DataFrame, new: pd.DataFrame, rescale: bool) -> pd.DataFrame:
    """Append new prices to the old ones. Yahoo adjusts the whole history for dividends: The old prices are
    therefore converted with the ratio new/old of the overlapping days, so that no jumps arise."""
    if new is None or new.empty:
        return old
    cols = old.columns.union(new.columns)
    old = old.reindex(columns=cols)
    new = new.reindex(columns=cols)
    if rescale:
        ov = old.index.intersection(new.index)
        if len(ov):
            ratio = (new.loc[ov] / old.loc[ov]).replace([np.inf, -np.inf], np.nan)
            f = ratio.median()
            f = f.where((ratio.notna().sum() >= 3) & ((f - 1).abs() > 5e-4), 1.0).fillna(1.0)
            old = old * f
    out = old.reindex(old.index.union(new.index))
    out.update(new)
    return out.sort_index()


def update_prices(download: bool = True) -> bool:
    """Load prices and turnover from Yahoo since the last stored day."""
    tm.DATA_DIR.mkdir(exist_ok=True)
    if PRICE_FILE.exists():
        d = pd.read_pickle(PRICE_FILE)
    elif PRICE_START.exists():
        log(f"Price data from the initial data set {PRICE_START.name}")
        d = pd.read_pickle(PRICE_START)
    else:
        d = _read_screener_cache()
    if not download:
        if not PRICE_FILE.exists():
            pd.to_pickle(d, PRICE_FILE)
        return True
    tickers = list(d["close"].columns)
    since = (d["close"].index.max() - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    try:
        import yfinance as yf
        closes, vols = [], []
        log(f"Loading prices since {since} for {len(tickers)} stocks from Yahoo Finance ...")
        for i in range(0, len(tickers) + 1, DOWNLOAD_CHUNK):
            chunk = tickers[i:i + DOWNLOAD_CHUNK]
            if i + DOWNLOAD_CHUNK >= len(tickers):
                chunk = chunk + [BENCH]
            if not chunk:
                continue
            data = None
            for attempt in range(3):
                try:
                    data = yf.download(chunk, start=since, auto_adjust=True, progress=False, threads=True,
                                       group_by="column")
                    break
                except Exception as e:
                    log(f"  Download error ({type(e).__name__}), new attempt in {10 * (attempt + 1)}s")
                    time.sleep(10 * (attempt + 1))
            if data is None or data.empty:
                continue
            c = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].set_axis(chunk[:1], axis=1)
            v = data["Volume"] if isinstance(data.columns, pd.MultiIndex) else data[["Volume"]].set_axis(chunk[:1], axis=1)
            closes.append(c)
            vols.append(v)
        if not closes:
            raise RuntimeError("no prices received")
        close = pd.concat(closes, axis=1)
        vol = pd.concat(vols, axis=1)
        close.index = vol.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close.loc[:, ~close.columns.duplicated()].dropna(how="all")
        vol = vol.loc[:, ~vol.columns.duplicated()].reindex(close.index)
        bench_new = close[[BENCH]] if BENCH in close.columns else pd.DataFrame()
        close = close.drop(columns=[BENCH], errors="ignore")
        vol = vol.drop(columns=[BENCH], errors="ignore")
        if close.notna().sum().sum() == 0:
            raise RuntimeError("no prices received")
        d["close"] = _merge(d["close"], close.reindex(columns=tickers), True)[tickers]
        d["volume"] = _merge(d["volume"], vol.reindex(columns=tickers), False)[tickers]
        if not bench_new.empty:
            d["bench"] = _merge(d["bench"] if len(d.get("bench", [])) else bench_new, bench_new, True)
        pd.to_pickle(d, PRICE_FILE)
        log(f"Prices updated (up to {d['close'].dropna(how='all').index.max():%d.%m.%Y})")
        return True
    except Exception as e:
        if not PRICE_FILE.exists():
            pd.to_pickle(d, PRICE_FILE)
        log(f"Yahoo not reachable ({type(e).__name__}) - using stored prices up to "
            f"{d['close'].dropna(how='all').index.max():%d.%m.%Y}")
        return False


def load_market() -> dict:
    """Market as in the trend model (tradability, turnover), but from the protocol prices."""
    tm.STOCK_FILE = str(PRICE_FILE)                         # absolute path replaces the screener cache
    m = tc.load_data()
    d = pd.read_pickle(PRICE_FILE)
    b = d.get("bench")
    m["bench"] = (b[BENCH].reindex(m["close"].index).ffill()
                  if isinstance(b, pd.DataFrame) and BENCH in b.columns else None)
    return m


# ======================================================================================= Account
def build_trades(m: dict, entry: str, exit_name: str, s0: int) -> tuple[pd.DataFrame, np.ndarray]:
    """All trades from signals from the start day (index s0) onwards. Positions still open: x = very large."""
    cal = m["close"].index
    sigs = tc.entry_signals(m, [entry])
    ev = sigs[entry].copy()
    ev[:s0] = False
    meta = tc.all_trades(m, {entry: ev})
    k = tc.EX_NAMES.index(exit_name)
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    rows = np.arange(len(meta))
    tr = tc.to_trades(meta, rows, np.full(len(rows), k), mom[meta["t_signal"].to_numpy(), meta["j"].to_numpy()])
    opn = meta.attrs["OPEN"][:, k] if len(meta) else np.zeros(0, dtype=bool)
    tr.loc[opn, "x"] = len(cal) + 10 ** 6
    tr.loc[opn, "reason"] = "open"
    return tr, ev


def run_account(m: dict, tr: pd.DataFrame, s0: int, skip: set | None = None, strat: str = "",
                hold: int = 10 ** 6) -> dict:
    """Account from the start day: like trend_model.simulate (long, closing auction), but open positions stay open
    and each order is recorded with quantity, price and costs."""
    cal = m["close"].index
    cols = m["close"].columns
    P = m["close"].ffill().to_numpy()
    n = len(cal)
    slip = tm.SLIPPAGE_BPS / 1e4
    first = s0 + 1
    by_entry: dict[int, list] = {}
    for r in tr.sort_values(["e", "rank"], ascending=[True, False]).itertuples(index=False):
        if skip and (strat, cols[r.j], f"{cal[r.e]:%Y-%m-%d}") in skip:
            continue
        by_entry.setdefault(r.e, []).append(r)
    cash, pos = float(CAPITAL), {}
    equity = np.full(n, np.nan)
    invested = np.full(n, np.nan)
    npos = np.full(n, np.nan)
    fills, closed = [], []
    fees = slips = 0.0
    if s0 < n:
        equity[s0], invested[s0], npos[s0] = CAPITAL, 0.0, 0
    for t in range(first, n):
        mf = tm.monthly_fee(cal, t, first)
        cash -= mf
        fees += mf
        for j in [j for j, q in pos.items() if q["x"] == t]:
            q = pos.pop(j)
            q["reason"] = "time" if t - q["e"] >= hold else "Stop"
            px = P[t, j]
            gross = q["shares"] * px
            fee, sl = tm.commission(gross, cols[j]), gross * slip
            cash += gross - fee - sl
            fees += fee
            slips += sl
            pnl = gross - fee - sl - q["alloc"] - q["cost_in"]
            fills.append({"date": f"{cal[t]:%Y-%m-%d}", "strategy": strat, "stock": cols[j], "order": "sell",
                          "side": "sale", "reason": q["reason"], "shares": q["shares"], "model_price": round(px, 4),
                          "fee_eur": round(fee, 2), "slippage_eur": round(sl, 2)})
            closed.append({"strategy": strat, "stock": cols[j], "purchase": f"{cal[q['e']]:%Y-%m-%d}",
                           "sale": f"{cal[t]:%Y-%m-%d}", "days": t - q["e"], "reason": q["reason"],
                           "shares": q["shares"], "buy_price": round(q["pe"], 4), "sell_price": round(px, 4),
                           "result_eur": round(pnl, 2), "return_pct": round(pnl / q["alloc"] * 100, 2)})
        eq_now = cash + sum(q["shares"] * P[t, j] for j, q in pos.items())
        for r in by_entry.get(t, []):
            if len(pos) >= tm.MAX_POSITIONS:
                break
            j = r.j
            if j in pos or not np.isfinite(P[t, j]) or r.x <= t:
                continue
            target = min(eq_now / tm.MAX_POSITIONS, cash)
            shares = int(target // P[t, j])
            if shares <= 0:
                continue
            alloc = shares * P[t, j]
            fee, sl = tm.commission(alloc, cols[j]), alloc * slip
            if alloc + fee + sl > cash:
                continue
            cash -= alloc + fee + sl
            fees += fee
            slips += sl
            pos[j] = {"e": t, "x": int(r.x), "shares": shares, "pe": P[t, j], "alloc": alloc, "cost_in": fee + sl,
                      "reason": ""}
            fills.append({"date": f"{cal[t]:%Y-%m-%d}", "strategy": strat, "stock": cols[j], "order": "buy",
                          "side": "purchase", "reason": "Signal", "shares": shares, "model_price": round(P[t, j], 4),
                          "fee_eur": round(fee, 2), "slippage_eur": round(sl, 2)})
        held = sum(q["shares"] * P[t, j] for j, q in pos.items())
        equity[t] = cash + held
        invested[t] = held
        npos[t] = len(pos)
    eq = pd.Series(equity, index=cal).iloc[s0:]
    return {"equity": eq, "invested": pd.Series(invested, index=cal).iloc[s0:], "npos": pd.Series(npos, index=cal).iloc[s0:],
            "cash": cash, "positions": pos, "fills": fills, "closed": pd.DataFrame(closed), "fees": fees, "slippage": slips}


def kpis(eq: pd.Series) -> dict:
    e = eq.dropna()
    r = e.pct_change().dropna()
    out = {"balance": float(e.iloc[-1]), "return": float(e.iloc[-1] / e.iloc[0] - 1),
           "max_dd": float((1 - e / e.cummax()).max()), "days": len(r)}
    out["sharpe"] = float(r.mean() / r.std() * np.sqrt(252)) if len(r) >= MIN_DAYS_FOR_SHARPE and r.std() > 0 else np.nan
    return out


def next_session(last: pd.Timestamp, test: bool = False) -> pd.Timestamp:
    """Next Xetra trading day after the last price day (holidays not taken into account)."""
    if test:
        return last + pd.offsets.BDay(1)
    try:
        now = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None)
    except Exception:
        now = pd.Timestamp.now()
    day = last + pd.offsets.BDay(1)
    today = now.normalize()
    if today > day and today.weekday() < 5 and now.hour * 60 + now.minute < 17 * 60 + 30:
        day = today
    return day


# ======================================================================================= Protocol
def frozen_settings() -> dict:
    s = {"STRATEGIES": {k: list(v) for k, v in STRATEGIES.items()}}
    for name, (en, ex) in STRATEGIES.items():
        s[f"{name}_entry"] = list(map(str, tc.ENTRY_RULES[en]))
        s[f"{name}_exit"] = tc.EXIT_RULES[ex]
    s.update({"MAX_POSITIONS": tm.MAX_POSITIONS, "CAPITAL": CAPITAL, "MIN_TURNOVER_EUR": tm.MIN_TURNOVER_EUR,
              "MIN_PRICE": tm.MIN_PRICE, "MIN_HISTORY": tm.MIN_HISTORY, "MOMENTUM_DAYS": tm.MOMENTUM_DAYS,
              "VOLUME_SURGE": tc.VOLUME_SURGE, "MAX_HOLDING_DAYS": tc.MAX_HOLDING_DAYS,
              "EXECUTION": "Xetra closing auction on the trading day after the signal"})
    return s


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def parse_dates(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").fillna(pd.to_datetime(s, format="%d.%m.%Y", errors="coerce"))


def to_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def run_protocol(restart: bool, cutoff: str | None) -> None:
    now = time.strftime("%Y-%m-%d %H:%M")
    fresh = update_prices(download=not cutoff)
    if not fresh and not (PROTOCOL_DIR / "start.json").exists() and not restart:
        raise SystemExit("No protocol created because the prices are not up to date (Yahoo not reachable). "
                         "The next run tries again.")
    m = load_market()
    if cutoff:
        cut = m["close"].index <= pd.Timestamp(cutoff)
        for k in ("close", "elig", "turnover", "volume"):
            m[k] = m[k][cut]
        if m.get("bench") is not None:
            m["bench"] = m["bench"][cut]
    cal = m["close"].index
    cols = m["close"].columns
    names = m["names"]
    last = cal[-1]
    n = len(cal)

    # ---- start: freeze or compare settings
    if restart and PROTOCOL_DIR.exists():
        target = BASE_DIR / f"protocol_old_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(PROTOCOL_DIR), str(target))
        log(f"Previous protocol archived in {target.name}")
    sp = PROTOCOL_DIR / "start.json"
    if not sp.exists():
        PROTOCOL_DIR.mkdir(exist_ok=True)
        start = {"start_day": f"{last:%Y-%m-%d}", "created_on": now, "settings": frozen_settings(),
                 "stocks": list(cols),
                 "assumptions_for_information": {"costs": tm.broker_text(), "SLIPPAGE_BPS": tm.SLIPPAGE_BPS},
                 "note": "Start day = last price day at creation. Signals from its closing price on are counted; "
                            "purchases are made on the next trading day at the earliest."}
        sp.write_text(json.dumps(start, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Protocol created. Start day {last:%d.%m.%Y}, settings frozen in protocol/start.json")
    else:
        start = json.loads(sp.read_text(encoding="utf-8"))
        cur = json.loads(json.dumps(frozen_settings()))
        diff = [k for k in sorted(set(start["settings"]) | set(cur)) if start["settings"].get(k) != cur.get(k)]
        if diff:
            raise SystemExit("The settings differ from the frozen state of the protocol: " + ", ".join(diff)
                             + "\nReset the settings (see protocol/start.json) or run\n"
                               "    ..\\.venv\\python.exe trend_protocol.py --protocol-new\nto start anew "
                               "(the old protocol is archived).")
    start_day = pd.Timestamp(start["start_day"])
    if start_day not in cal:
        raise SystemExit(f"The start day {start_day:%d.%m.%Y} is missing in the prices.")
    s0 = int(cal.get_loc(start_day))
    n_days = n - 1 - s0

    # ---- purchases that could not be executed (executions.csv, column 'executable' = no)
    fills_path = PROTOCOL_DIR / "executions.csv"
    fcols = ["date", "strategy", "stock", "name", "order", "side", "reason", "shares", "model_price", "fee_eur",
             "slippage_eur", "real_price", "deviation_bp", "executable"]
    old_fills, skip = None, set()
    if fills_path.exists():
        old_fills = read_semicolon_csv(fills_path)
        for c in fcols:
            if c not in old_fills.columns:
                old_fills[c] = "yes" if c == "executable" else ""
        old_fills["date"] = parse_dates(old_fills["date"]).dt.strftime("%Y-%m-%d")
        no = old_fills["executable"].astype(str).str.strip().str.lower().str.startswith("n")
        buy = old_fills["order"] == "buy"
        skip = set(zip(old_fills.loc[no & buy, "strategy"], old_fills.loc[no & buy, "stock"],
                       old_fills.loc[no & buy, "date"]))

    # ---- calculate both accounts
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    accts, clean, signals_today, order_rows, pos_rows, log_rows = {}, {}, {}, [], [], []
    nxt = next_session(last, bool(cutoff))
    for name, (entry, exit_name) in STRATEGIES.items():
        tr, ev = build_trades(m, entry, exit_name, s0)
        rule = tc.EXIT_RULES[exit_name]
        hold = rule.get("days", tc.MAX_HOLDING_DAYS)
        acct = run_account(m, tr, s0, None, name, hold)
        accts[name] = acct
        if any(s[0] == name for s in skip):
            clean[name] = run_account(m, tr, s0, skip, name, hold)
        # orders for the next closing auction
        sells, stops = [], []
        for j, q in acct["positions"].items():
            held_days = n - q["e"]                          # trading days since purchase if sold today
            seg = P[q["e"]:n, j]
            line = None
            if rule.get("pct"):
                line = float(np.nanmax(seg)) * (1 - rule["pct"])
            planned = q["e"] + rule["days"] if rule.get("days") else q["e"] + tc.MAX_HOLDING_DAYS
            row = {"strategy": name, "stock": cols[j], "name": names.get(cols[j], ""), "shares": q["shares"],
                   "purchase": f"{cal[q['e']]:%Y-%m-%d}", "buy_price": round(q["pe"], 4), "kurs": round(P[n - 1, j], 4),
                   "value_eur": round(q["shares"] * P[n - 1, j], 2),
                   "result_pct": round((P[n - 1, j] / q["pe"] - 1) * 100, 2),
                   "held_days": n - 1 - q["e"], "sale_at_latest": f"{nxt + pd.offsets.BDay(max(planned - n, 0)):%Y-%m-%d}",
                   "stop_line": round(line, 4) if line else ""}
            pos_rows.append(row)
            if planned <= n:
                sells.append((j, q, f"holding period {rule.get('days', tc.MAX_HOLDING_DAYS)} days reached"))
            elif line:
                stops.append((j, q, line))
        # new signals at the last closing price
        cand = np.flatnonzero(ev[n - 1]) if n - 1 >= s0 else np.array([], dtype=int)
        held = set(acct["positions"])
        cand = [j for j in cand if j not in held]
        cand.sort(key=lambda j: -np.nan_to_num(mom[n - 1, j], nan=-9))
        free = tm.MAX_POSITIONS - len(acct["positions"]) + len(sells)
        eq_last = float(acct["equity"].iloc[-1])
        cash_est = acct["cash"] + sum(q["shares"] * P[n - 1, j] for j, q, _ in sells)
        buys, waiting = [], []
        for j in cand:
            if len(buys) < free:
                amount = min(eq_last / tm.MAX_POSITIONS, cash_est)
                sh = int(amount // P[n - 1, j])
                if sh > 0:
                    buys.append((j, sh))
                    cash_est -= sh * P[n - 1, j]
                    continue
            waiting.append(j)
        signals_today[name] = {"buys": buys, "sells": sells, "stops": stops, "waiting": waiting}
        for j, sh in buys:
            order_rows.append({"for_day": f"{nxt:%Y-%m-%d}", "strategy": name, "order": "buy", "stock": cols[j],
                               "name": names.get(cols[j], ""), "shares_approx": sh, "close_previous_day": round(P[n - 1, j], 4),
                               "note": "Xetra closing auction"})
        for j, q, why in sells:
            order_rows.append({"for_day": f"{nxt:%Y-%m-%d}", "strategy": name, "order": "sell", "stock": cols[j],
                               "name": names.get(cols[j], ""), "shares_approx": q["shares"],
                               "close_previous_day": round(P[n - 1, j], 4), "note": why})
        for j, q, line in stops:
            order_rows.append({"for_day": f"{nxt:%Y-%m-%d}", "strategy": name, "order": "sell if close <= line",
                               "stock": cols[j], "name": names.get(cols[j], ""), "shares_approx": q["shares"],
                               "close_previous_day": round(P[n - 1, j], 4), "note": f"Stop line {line:.4f}"})
        # daily status for protocol.csv
        eq, inv, npos = acct["equity"], acct["invested"], acct["npos"]
        for t in range(1, len(eq)):
            day = eq.index[t]
            log_rows.append({"logged_on": now, "price_date": f"{day:%Y-%m-%d}",
                             "added_later": "yes" if day < last else "no", "strategy": name,
                             "account_eur": round(float(eq.iloc[t]), 2), "invested_eur": round(float(inv.iloc[t]), 2),
                             "positions_n": int(npos.iloc[t]),
                             "daily_return": round(float(eq.iloc[t] / eq.iloc[t - 1] - 1), 6)})

    # ---- benchmarks since the start day
    pf = m["close"].ffill()
    ew_r = pf.pct_change().where(m["elig"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ew_since = ew_r.iloc[s0:].copy()
    ew_since.iloc[0] = 0.0                                   # start day = initial value
    bench_curves = {"Stocks equal-weighted": CAPITAL * (1 + ew_since).cumprod()}
    if m.get("bench") is not None:
        b = m["bench"].iloc[s0:].ffill()
        if b.notna().all() and len(b):
            bench_curves["DAX (EXS1.DE)"] = CAPITAL * b / b.iloc[0]

    # ---- protocol.csv: only append new rows
    log_path = PROTOCOL_DIR / "protocol.csv"
    new_log = pd.DataFrame(log_rows)
    n_new, late = 0, False
    if len(new_log):
        known = set()
        if log_path.exists():
            old = read_semicolon_csv(log_path)
            known = set(zip(parse_dates(old["price_date"]).dt.strftime("%Y-%m-%d"), old["strategy"]))
        add = new_log[[(d, s) not in known for d, s in zip(new_log["price_date"], new_log["strategy"])]]
        n_new, late = len(add), bool((add["added_later"] == "yes").any())
        if n_new:
            try:
                add.to_csv(log_path, sep=";", decimal=",", index=False, mode="a", header=not log_path.exists(),
                           encoding="utf-8-sig" if not log_path.exists() else "utf-8")
            except PermissionError:
                n_new = 0
                log("protocol.csv is locked (open in Excel?) - the rows will be added on the next run")

    # ---- executions.csv: add new orders, keep entered real prices
    fills = pd.DataFrame([f for a in accts.values() for f in a["fills"]])
    if len(fills):
        fills["name"] = fills["stock"].map(names).fillna("")
        fills["real_price"], fills["deviation_bp"], fills["executable"] = "", "", "yes"
        fills = fills[fcols]
    else:
        fills = pd.DataFrame(columns=fcols)
    if old_fills is not None:
        key_old = set(zip(old_fills["date"], old_fills["strategy"], old_fills["stock"], old_fills["order"]))
        fills = fills[[(d, s, a, o) not in key_old for d, s, a, o in
                       zip(fills["date"], fills["strategy"], fills["stock"], fills["order"])]]
        fills = pd.concat([old_fills[fcols], fills], ignore_index=True)
    for c in ("shares", "fee_eur", "slippage_eur"):     # old rows come from the file as text
        fills[c] = to_float(fills[c])
    fills["shares"] = fills["shares"].round().astype("Int64")
    real, model = to_float(fills["real_price"]), to_float(fills["model_price"])
    fills["deviation_bp"] = pd.Series(np.where(fills["side"] == "purchase", real / model - 1, 1 - real / model) * 1e4,
                                       index=fills.index).round(1)
    fills["real_price"], fills["model_price"] = real, model
    fills = fills.sort_values(["date", "strategy", "stock"]).reset_index(drop=True)
    PROTOCOL_DIR.mkdir(exist_ok=True)
    try:
        fills.to_csv(fills_path, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    except PermissionError:
        log("executions.csv is locked (open in Excel?) - new orders will be added on the next run")
    measured = fills["deviation_bp"].dropna()

    # ---- result files
    daily = pd.DataFrame({f"account_{k}": a["equity"].round(2) for k, a in accts.items()})
    for k, c in bench_curves.items():
        daily[f"comparison_{k}"] = c.reindex(daily.index).round(2)
    save_csv(daily.rename_axis("date"), "results.csv", index=True)
    save_csv(pd.DataFrame(pos_rows), "positions.csv")
    save_csv(pd.concat([a["closed"] for a in accts.values()], ignore_index=True), "trades.csv")
    save_csv(pd.DataFrame(order_rows), "orders_today.csv")

    report(accts, clean, signals_today, bench_curves, start, start_day, last, nxt, n_days, names, cols, P, n,
           measured, n_new, late)


def report(accts, clean, sig, bench_curves, start, start_day, last, nxt, n_days, names, cols, P, n, measured,
           n_new, late) -> None:
    print()
    print("=" * 100)
    print(f"TREND PROTOCOL (paper trading)  -  Robust 1 and Robust 2, each {eur(CAPITAL)} EUR, without leverage  -  "
          f"prices up to {last:%d.%m.%Y}")
    print("=" * 100)
    print(f"Start day {start_day:%d.%m.%Y} (created on {start['created_on']}), settings unchanged. "
          f"Costs: {tm.broker_text()}, slippage {tm.SLIPPAGE_BPS} bp")
    for k in STRATEGIES:
        print(f"    {k}: {DESCRIPTION[k]}")
    print(f"\nORDERS FOR {WEEKDAY[nxt.weekday()]} {nxt:%d.%m.%Y} - Xetra closing auction (order by 17:30):")
    any_order = False
    for k, s in sig.items():
        print(f"  {k}:")
        if not (s["buys"] or s["sells"]):
            print("      no purchases or sales")
        for j, q, why in s["sells"]:
            any_order = True
            print(f"    SELL      {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} {q['shares']:>6} shares   "
                  f"({why}; last close {num(P[n - 1, j])} EUR, since purchase {P[n - 1, j] / q['pe'] - 1:+.1%})")
        for j, sh in s["buys"]:
            any_order = True
            print(f"    BUY       {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} {sh:>6} shares approx. "
                  f"(approx. {eur(sh * P[n - 1, j])} EUR, last close {num(P[n - 1, j])} EUR)")
        if s["waiting"]:
            print(f"      further signals without a free slot: "
                  + ", ".join(cols[j] for j in s["waiting"][:8]) + (" ..." if len(s["waiting"]) > 8 else ""))
        if s["stops"]:
            print(f"      stop lines (sell if the price is below them shortly before 17:30):")
            for j, q, line in sorted(s["stops"], key=lambda x: P[n - 1, x[0]] / x[2]):
                print(f"        {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} line {num(line):>10} EUR   "
                      f"last close {num(P[n - 1, j]):>10} EUR  (distance {line / P[n - 1, j] - 1:+.1%})")
    if not any_order:
        print("    -> no orders today")

    print(f"\nACCOUNTS SINCE START ({n_days} trading days)")
    rows = []
    for k, a in accts.items():
        kp = kpis(a["equity"])
        cl = a["closed"]
        rows.append((k, kp, len(a["positions"]), len(cl), int((cl["result_eur"] > 0).sum()) if len(cl) else 0,
                     a["fees"], a["slippage"]))
    print(f"    {'':<36}{'Account EUR':>12}{'Return':>10}{'max. DD':>15}{'Sharpe':>8}{'Positions':>12}"
          f"{'Trades':>8}{'Winners':>10}{'Fees':>11}{'Slippage':>10}")
    for k, kp, npos, ntr, nwin, fees, sl in rows:
        sh = f"{kp['sharpe']:.2f}" if np.isfinite(kp["sharpe"]) else "-"
        print(f"    {k:<36}{eur(kp['balance']):>12}{kp['return']:>+10.2%}{kp['max_dd']:>15.2%}{sh:>8}{npos:>12}"
              f"{ntr:>8}{nwin:>10}{num(fees):>11}{num(sl):>10}")
    for k, c in bench_curves.items():
        if len(c.dropna()) > 1:
            kp = kpis(c)
            sh = f"{kp['sharpe']:.2f}" if np.isfinite(kp["sharpe"]) else "-"
            print(f"    {'Comparison: ' + k:<36}{eur(kp['balance']):>12}{kp['return']:>+10.2%}{kp['max_dd']:>15.2%}{sh:>8}")
    if n_days < MIN_DAYS_FOR_SHARPE:
        print(f"    Sharpe ratio only from {MIN_DAYS_FOR_SHARPE} trading days on (meaningless before).")
    for k, a in clean.items():
        kp = kpis(a["equity"])
        print(f"    {k} without the purchases marked as not executable: {eur(kp['balance'])} EUR ({kp['return']:+.2%})")

    for k, a in accts.items():
        if not a["positions"]:
            continue
        print(f"\nOPEN POSITIONS {k}:")
        for j, q in sorted(a["positions"].items(), key=lambda x: x[1]["e"]):
            print(f"    {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} {q['shares']:>6} shares  bought at "
                  f"{num(q['pe'])} EUR, now {num(P[n - 1, j])} EUR ({P[n - 1, j] / q['pe'] - 1:+.1%}), "
                  f"{n - 1 - q['e']} days")
    last_fills = [f for a in accts.values() for f in a["fills"] if f["date"] == f"{last:%Y-%m-%d}"]
    if last_fills:
        print(f"\nBOOKED ON {last:%d.%m.%Y} (model, closing auction):")
        for f in last_fills:
            print(f"    {f['strategy']}: {f['order']} {f['stock']} {f['shares']} shares at {num(f['model_price'])} EUR"
                  + (f" ({f['reason']})" if f["order"] == "sell" else ""))
    if len(measured):
        print(f"\nMeasured deviation from the model price: on average {measured.mean():+.1f} bp per order ({len(measured)} prices; "
              f"assumed {tm.SLIPPAGE_BPS} bp). Positive = worse than the model.")
    else:
        print("\nNo real prices entered yet (protocol/executions.csv, column 'real_price'; mark purchases that were not possible "
              "in 'executable' with 'no').")
    if n_new:
        print(f"Logged: {n_new} new rows in protocol.csv"
              + (" - including days added later" if late else ""))
    elif n_days:
        print("Logged: nothing new (this price day is already in the protocol).")
    else:
        print("Today is the start day: counting starts from the next trading day.")
    print(f"Files: {PROTOCOL_DIR}  (protocol.csv, executions.csv, orders_today.csv, positions.csv, trades.csv, "
          "results.csv)")


# ======================================================================================= Backtest
def run_backtest() -> None:
    m = load_market() if PRICE_FILE.exists() else tc.load_data()
    cal = m["close"].index
    s0 = int(np.searchsorted(cal, pd.Timestamp(tc.SIGNAL_START)))
    print("\n" + "=" * 100)
    print(f"BACKTEST from {cal[s0]:%d.%m.%Y} to {cal[-1]:%d.%m.%Y}  (costs: {tm.broker_text()}, "
          f"slippage {tm.SLIPPAGE_BPS} bp)")
    print("=" * 100)
    for name, (entry, exit_name) in STRATEGIES.items():
        tr, _ = build_trades(m, entry, exit_name, s0)
        a = run_account(m, tr, s0, None, name, tc.EXIT_RULES[exit_name].get("days", tc.MAX_HOLDING_DAYS))
        eq = a["equity"]
        kp = kpis(eq)
        yrs = len(eq) / 252
        y = eq.groupby(eq.index.year).last()
        yr = (y / y.shift(1).fillna(eq.iloc[0]) - 1) * 100
        print(f"\n{name} ({DESCRIPTION[name]}):")
        print(f"    return p. a. {((kp['balance'] / CAPITAL) ** (1 / yrs) - 1):+.1%}, Sharpe {kp['sharpe']:.2f}, "
              f"max. drawdown {kp['max_dd']:.1%}, trades {len(a['closed'])}, fees {eur(a['fees'])} EUR, "
              f"slippage {eur(a['slippage'])} EUR")
        print("    Years: " + "  ".join(f"{yy}: {v:+.1f} %" for yy, v in yr.items()))
    print("\nLonger backtest (from 2006) and chance test: trend_cpo_long.py --long, trend_evaluation.py")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):                 # never let special characters in stock names cause an abort
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Trend following Xetra: Robust 1 and Robust 2 with protocol mode")
    ap.add_argument("--protocol", action="store_true", help="protocol mode (paper trading)")
    ap.add_argument("--protocol-new", dest="protocol_new", action="store_true", help="start the protocol anew")
    ap.add_argument("--backtest", action="store_true", help="backtest of both strategies")
    ap.add_argument("--cutoff", default=None, help=argparse.SUPPRESS)   # for testing only
    args = ap.parse_args()
    if args.protocol or args.protocol_new:
        run_protocol(args.protocol_new, args.cutoff)
    elif args.backtest:
        run_backtest()
    else:
        ap.print_help()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(1)
