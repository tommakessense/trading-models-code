"""
trend_protokoll.py - trend following on Xetra (Robust 1 and Robust 2) with protocol mode for paper trading
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
Basis: trend_cpo.py / trend_bewertung.py / trend_cpo_lang.py (backtest 2006-2026, walk-forward).

Execution
---------
The signal arises with the closing price of a day. Buying and selling take place in the Xetra closing auction of the
next trading day (order by 17:30, e.g. limited or as an auction order). All backtests
are based on this. The run before the market opens shows the orders for today's closing auction. The stop lines of
Robust 2 apply to today's closing price: If the price is below them shortly before 17:30, sell in the closing
auction.

Costs: order fees of Scalable Capital (FREE, Xetra 1.99 EUR per order) for all stocks tradable there,
otherwise IBKR (0.05 %, min. 3 EUR) - set in trend_model.py (BROKER, SCALABLE_TARIF,
scalable_nicht_handelbar.csv) - plus SLIPPAGE_BPS per order. Taxes are not taken into account.

Usage (in the folder trend_model)
---------------------------------
    ..\\.venv\\python.exe trend_protokoll.py --protokoll       protocol: update prices, book the day, orders for today
    ..\\.venv\\python.exe trend_protokoll.py --protokoll-neu   start the protocol anew (the old one is archived)
    ..\\.venv\\python.exe trend_protokoll.py --rueckrechnung   backtest of both strategies with the current costs
    protokoll_taeglich.bat                                    daily run (Windows task, see aufgabe_einrichten.bat)

Protocol mode
-------------
On the first call, the start day, settings and stock list are frozen (protokoll/start.json). Signals from the
closing price of the start day onwards are counted; the first purchase is therefore on the next trading day at the earliest. Each run
    - loads the new prices from Yahoo (daten/kurse_protokoll.pkl; the first time from the screener cache),
    - calculates both accounts from the start day and appends the daily status to protokoll/protokoll.csv
      (existing rows are never changed; days entered later are marked as "nachgetragen"),
    - writes each order with the model price to protokoll/ausfuehrungen.csv. Anyone who executes the orders in a demo account
      enters the actual price there in "realer_kurs" (-> measured slippage). If a purchase was
      not possible, enter "nein" in "ausfuehrbar": The account is then additionally shown without this trade.
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
STRATEGIEN = {                   # name: (entry from trend_cpo.EINSTIEGE, exit from trend_cpo.AUSSTIEGE)
    "Robust 1": ("Hoch 200", "Zeit 20"),
    "Robust 2": ("Hoch 50 + Volumenschub", "Zeit 60 + Stopp 15%"),
}
BESCHREIBUNG = {
    "Robust 1": "neues 200-Tage-Hoch, 20 Handelstage halten",
    "Robust 2": "neues 50-Tage-Hoch mit doppeltem Volumen, 60 Handelstage halten, Stopp 15 % unter dem Hoch seit Kauf",
}
CAPITAL = tm.CAPITAL             # a separate account for each strategy
MIN_DAYS_FOR_SHARPE = 60
BENCH = "EXS1.DE"                # iShares Core DAX
DOWNLOAD_CHUNK = 50

BASE_DIR = Path(__file__).resolve().parent
PROT_DIR = BASE_DIR / "protokoll"
KURS_DATEI = tm.DATA_DIR / "kurse_protokoll.pkl"
KURS_START = BASE_DIR / "daten_start" / "kurse_protokoll.pkl.xz"   # initial stock of prices (e.g. for GitHub Actions)
WOCHENTAG = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def log(msg: str) -> None:
    tm.log(msg)


def eur(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def num(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> None:
    old = tm.OUT_DIR
    tm.OUT_DIR = PROT_DIR
    tm.save_csv(df, name, index=index)
    tm.OUT_DIR = old


# ======================================================================================= Prices
def _read_screener_cache() -> dict:
    src = tm.SCREENER_CACHE / "prices_aktien.pkl"
    if not src.exists():
        raise SystemExit(f"Kurs-Cache des Screeners fehlt: {src}")
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
    if KURS_DATEI.exists():
        d = pd.read_pickle(KURS_DATEI)
    elif KURS_START.exists():
        log(f"Kursdaten aus dem Startbestand {KURS_START.name}")
        d = pd.read_pickle(KURS_START)
    else:
        d = _read_screener_cache()
    if not download:
        if not KURS_DATEI.exists():
            pd.to_pickle(d, KURS_DATEI)
        return True
    tickers = list(d["close"].columns)
    since = (d["close"].index.max() - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    try:
        import yfinance as yf
        closes, vols = [], []
        log(f"Lade Kurse seit {since} fuer {len(tickers)} Aktien von Yahoo Finance ...")
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
                    log(f"  Download-Fehler ({type(e).__name__}), neuer Versuch in {10 * (attempt + 1)}s")
                    time.sleep(10 * (attempt + 1))
            if data is None or data.empty:
                continue
            c = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].set_axis(chunk[:1], axis=1)
            v = data["Volume"] if isinstance(data.columns, pd.MultiIndex) else data[["Volume"]].set_axis(chunk[:1], axis=1)
            closes.append(c)
            vols.append(v)
        if not closes:
            raise RuntimeError("keine Kurse erhalten")
        close = pd.concat(closes, axis=1)
        vol = pd.concat(vols, axis=1)
        close.index = vol.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close.loc[:, ~close.columns.duplicated()].dropna(how="all")
        vol = vol.loc[:, ~vol.columns.duplicated()].reindex(close.index)
        bench_new = close[[BENCH]] if BENCH in close.columns else pd.DataFrame()
        close = close.drop(columns=[BENCH], errors="ignore")
        vol = vol.drop(columns=[BENCH], errors="ignore")
        if close.notna().sum().sum() == 0:
            raise RuntimeError("keine Kurse erhalten")
        d["close"] = _merge(d["close"], close.reindex(columns=tickers), True)[tickers]
        d["volume"] = _merge(d["volume"], vol.reindex(columns=tickers), False)[tickers]
        if not bench_new.empty:
            d["bench"] = _merge(d["bench"] if len(d.get("bench", [])) else bench_new, bench_new, True)
        pd.to_pickle(d, KURS_DATEI)
        log(f"Kurse aktualisiert (bis {d['close'].dropna(how='all').index.max():%d.%m.%Y})")
        return True
    except Exception as e:
        if not KURS_DATEI.exists():
            pd.to_pickle(d, KURS_DATEI)
        log(f"Yahoo nicht erreichbar ({type(e).__name__}) - verwende gespeicherte Kurse bis "
            f"{d['close'].dropna(how='all').index.max():%d.%m.%Y}")
        return False


def load_market() -> dict:
    """Market as in the trend model (tradability, turnover), but from the protocol prices."""
    tm.STOCK_FILE = str(KURS_DATEI)                         # absolute path replaces the screener cache
    m = tc.load_data()
    d = pd.read_pickle(KURS_DATEI)
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
    tr.loc[opn, "grund"] = "offen"
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
    for r in tr.sort_values(["e", "rang"], ascending=[True, False]).itertuples(index=False):
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
            q["grund"] = "Zeit" if t - q["e"] >= hold else "Stopp"
            px = P[t, j]
            gross = q["shares"] * px
            fee, sl = tm.commission(gross, cols[j]), gross * slip
            cash += gross - fee - sl
            fees += fee
            slips += sl
            pnl = gross - fee - sl - q["alloc"] - q["cost_in"]
            fills.append({"datum": f"{cal[t]:%Y-%m-%d}", "strategie": strat, "aktie": cols[j], "order": "verkaufen",
                          "seite": "verkauf", "grund": q["grund"], "stueck": q["shares"], "modellkurs": round(px, 4),
                          "gebuehr_eur": round(fee, 2), "slippage_eur": round(sl, 2)})
            closed.append({"strategie": strat, "aktie": cols[j], "kauf": f"{cal[q['e']]:%Y-%m-%d}",
                           "verkauf": f"{cal[t]:%Y-%m-%d}", "tage": t - q["e"], "grund": q["grund"],
                           "stueck": q["shares"], "kaufkurs": round(q["pe"], 4), "verkaufskurs": round(px, 4),
                           "ergebnis_eur": round(pnl, 2), "rendite_pct": round(pnl / q["alloc"] * 100, 2)})
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
                      "grund": ""}
            fills.append({"datum": f"{cal[t]:%Y-%m-%d}", "strategie": strat, "aktie": cols[j], "order": "kaufen",
                          "seite": "kauf", "grund": "Signal", "stueck": shares, "modellkurs": round(P[t, j], 4),
                          "gebuehr_eur": round(fee, 2), "slippage_eur": round(sl, 2)})
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
    out = {"stand": float(e.iloc[-1]), "rendite": float(e.iloc[-1] / e.iloc[0] - 1),
           "max_dd": float((1 - e / e.cummax()).max()), "tage": len(r)}
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
    s = {"STRATEGIEN": {k: list(v) for k, v in STRATEGIEN.items()}}
    for name, (en, ex) in STRATEGIEN.items():
        s[f"{name}_einstieg"] = list(map(str, tc.EINSTIEGE[en]))
        s[f"{name}_ausstieg"] = tc.AUSSTIEGE[ex]
    s.update({"MAX_POSITIONS": tm.MAX_POSITIONS, "CAPITAL": CAPITAL, "MIN_TURNOVER_EUR": tm.MIN_TURNOVER_EUR,
              "MIN_PRICE": tm.MIN_PRICE, "MIN_HISTORY": tm.MIN_HISTORY, "MOMENTUM_DAYS": tm.MOMENTUM_DAYS,
              "VOLUMEN_SCHUB": tc.VOLUMEN_SCHUB, "MAX_HALTEDAUER": tc.MAX_HALTEDAUER,
              "AUSFUEHRUNG": "Xetra-Schlussauktion am Handelstag nach dem Signal"})
    return s


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def parse_dates(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").fillna(pd.to_datetime(s, format="%d.%m.%Y", errors="coerce"))


def to_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def run_protocol(restart: bool, stichtag: str | None) -> None:
    now = time.strftime("%Y-%m-%d %H:%M")
    fresh = update_prices(download=not stichtag)
    if not fresh and not (PROT_DIR / "start.json").exists() and not restart:
        raise SystemExit("Kein Protokoll angelegt, weil die Kurse nicht aktuell sind (Yahoo nicht erreichbar). "
                         "Der naechste Lauf versucht es erneut.")
    m = load_market()
    if stichtag:
        cut = m["close"].index <= pd.Timestamp(stichtag)
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
    if restart and PROT_DIR.exists():
        target = BASE_DIR / f"protokoll_alt_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(PROT_DIR), str(target))
        log(f"Bisheriges Protokoll archiviert in {target.name}")
    sp = PROT_DIR / "start.json"
    if not sp.exists():
        PROT_DIR.mkdir(exist_ok=True)
        start = {"starttag": f"{last:%Y-%m-%d}", "angelegt_am": now, "einstellungen": frozen_settings(),
                 "aktien": list(cols),
                 "annahmen_zur_information": {"kosten": tm.broker_text(), "SLIPPAGE_BPS": tm.SLIPPAGE_BPS},
                 "hinweis": "Starttag = letzter Kurstag beim Anlegen. Gewertet werden Signale ab seinem Schlusskurs; "
                            "gekauft wird fruehestens am naechsten Handelstag."}
        sp.write_text(json.dumps(start, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Protokoll angelegt. Starttag {last:%d.%m.%Y}, Einstellungen eingefroren in protokoll/start.json")
    else:
        start = json.loads(sp.read_text(encoding="utf-8"))
        cur = json.loads(json.dumps(frozen_settings()))
        diff = [k for k in sorted(set(start["einstellungen"]) | set(cur)) if start["einstellungen"].get(k) != cur.get(k)]
        if diff:
            raise SystemExit("Die Einstellungen weichen vom eingefrorenen Stand des Protokolls ab: " + ", ".join(diff)
                             + "\nEinstellungen zuruecksetzen (siehe protokoll/start.json) oder mit\n"
                               "    ..\\.venv\\python.exe trend_protokoll.py --protokoll-neu\nneu beginnen "
                               "(das alte Protokoll wird archiviert).")
    start_day = pd.Timestamp(start["starttag"])
    if start_day not in cal:
        raise SystemExit(f"Der Starttag {start_day:%d.%m.%Y} fehlt in den Kursen.")
    s0 = int(cal.get_loc(start_day))
    n_days = n - 1 - s0

    # ---- purchases that could not be executed (ausfuehrungen.csv, column 'ausfuehrbar' = nein)
    fills_path = PROT_DIR / "ausfuehrungen.csv"
    fcols = ["datum", "strategie", "aktie", "name", "order", "seite", "grund", "stueck", "modellkurs", "gebuehr_eur",
             "slippage_eur", "realer_kurs", "abweichung_bp", "ausfuehrbar"]
    old_fills, skip = None, set()
    if fills_path.exists():
        old_fills = read_semicolon_csv(fills_path)
        for c in fcols:
            if c not in old_fills.columns:
                old_fills[c] = "ja" if c == "ausfuehrbar" else ""
        old_fills["datum"] = parse_dates(old_fills["datum"]).dt.strftime("%Y-%m-%d")
        no = old_fills["ausfuehrbar"].astype(str).str.strip().str.lower().str.startswith("n")
        buy = old_fills["order"] == "kaufen"
        skip = set(zip(old_fills.loc[no & buy, "strategie"], old_fills.loc[no & buy, "aktie"],
                       old_fills.loc[no & buy, "datum"]))

    # ---- calculate both accounts
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    accts, clean, signals_today, order_rows, pos_rows, log_rows = {}, {}, {}, [], [], []
    nxt = next_session(last, bool(stichtag))
    for name, (entry, exit_name) in STRATEGIEN.items():
        tr, ev = build_trades(m, entry, exit_name, s0)
        rule = tc.AUSSTIEGE[exit_name]
        hold = rule.get("tage", tc.MAX_HALTEDAUER)
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
            planned = q["e"] + rule["tage"] if rule.get("tage") else q["e"] + tc.MAX_HALTEDAUER
            row = {"strategie": name, "aktie": cols[j], "name": names.get(cols[j], ""), "stueck": q["shares"],
                   "kauf": f"{cal[q['e']]:%Y-%m-%d}", "kaufkurs": round(q["pe"], 4), "kurs": round(P[n - 1, j], 4),
                   "wert_eur": round(q["shares"] * P[n - 1, j], 2),
                   "ergebnis_pct": round((P[n - 1, j] / q["pe"] - 1) * 100, 2),
                   "gehalten_tage": n - 1 - q["e"], "verkauf_spaetestens": f"{nxt + pd.offsets.BDay(max(planned - n, 0)):%Y-%m-%d}",
                   "stopp_linie": round(line, 4) if line else ""}
            pos_rows.append(row)
            if planned <= n:
                sells.append((j, q, f"Haltedauer {rule.get('tage', tc.MAX_HALTEDAUER)} Tage erreicht"))
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
            order_rows.append({"fuer_tag": f"{nxt:%Y-%m-%d}", "strategie": name, "order": "kaufen", "aktie": cols[j],
                               "name": names.get(cols[j], ""), "stueck_ca": sh, "schluss_vortag": round(P[n - 1, j], 4),
                               "hinweis": "Xetra-Schlussauktion"})
        for j, q, why in sells:
            order_rows.append({"fuer_tag": f"{nxt:%Y-%m-%d}", "strategie": name, "order": "verkaufen", "aktie": cols[j],
                               "name": names.get(cols[j], ""), "stueck_ca": q["shares"],
                               "schluss_vortag": round(P[n - 1, j], 4), "hinweis": why})
        for j, q, line in stops:
            order_rows.append({"fuer_tag": f"{nxt:%Y-%m-%d}", "strategie": name, "order": "verkaufen, falls Schluss <= Linie",
                               "aktie": cols[j], "name": names.get(cols[j], ""), "stueck_ca": q["shares"],
                               "schluss_vortag": round(P[n - 1, j], 4), "hinweis": f"Stopp-Linie {line:.4f}"})
        # daily status for protokoll.csv
        eq, inv, npos = acct["equity"], acct["invested"], acct["npos"]
        for t in range(1, len(eq)):
            day = eq.index[t]
            log_rows.append({"protokolliert_am": now, "kursdatum": f"{day:%Y-%m-%d}",
                             "nachgetragen": "ja" if day < last else "nein", "strategie": name,
                             "konto_eur": round(float(eq.iloc[t]), 2), "investiert_eur": round(float(inv.iloc[t]), 2),
                             "positionen": int(npos.iloc[t]),
                             "tagesrendite": round(float(eq.iloc[t] / eq.iloc[t - 1] - 1), 6)})

    # ---- benchmarks since the start day
    pf = m["close"].ffill()
    ew_r = pf.pct_change().where(m["elig"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ew_seit = ew_r.iloc[s0:].copy()
    ew_seit.iloc[0] = 0.0                                   # start day = initial value
    bench_curves = {"Aktien gleichgewichtet": CAPITAL * (1 + ew_seit).cumprod()}
    if m.get("bench") is not None:
        b = m["bench"].iloc[s0:].ffill()
        if b.notna().all() and len(b):
            bench_curves["DAX (EXS1.DE)"] = CAPITAL * b / b.iloc[0]

    # ---- protokoll.csv: only append new rows
    log_path = PROT_DIR / "protokoll.csv"
    new_log = pd.DataFrame(log_rows)
    n_new, late = 0, False
    if len(new_log):
        known = set()
        if log_path.exists():
            old = read_semicolon_csv(log_path)
            known = set(zip(parse_dates(old["kursdatum"]).dt.strftime("%Y-%m-%d"), old["strategie"]))
        add = new_log[[(d, s) not in known for d, s in zip(new_log["kursdatum"], new_log["strategie"])]]
        n_new, late = len(add), bool((add["nachgetragen"] == "ja").any())
        if n_new:
            try:
                add.to_csv(log_path, sep=";", decimal=",", index=False, mode="a", header=not log_path.exists(),
                           encoding="utf-8-sig" if not log_path.exists() else "utf-8")
            except PermissionError:
                n_new = 0
                log("protokoll.csv ist gesperrt (in Excel offen?) - die Zeilen werden beim naechsten Lauf nachgetragen")

    # ---- ausfuehrungen.csv: add new orders, keep entered real prices
    fills = pd.DataFrame([f for a in accts.values() for f in a["fills"]])
    if len(fills):
        fills["name"] = fills["aktie"].map(names).fillna("")
        fills["realer_kurs"], fills["abweichung_bp"], fills["ausfuehrbar"] = "", "", "ja"
        fills = fills[fcols]
    else:
        fills = pd.DataFrame(columns=fcols)
    if old_fills is not None:
        key_old = set(zip(old_fills["datum"], old_fills["strategie"], old_fills["aktie"], old_fills["order"]))
        fills = fills[[(d, s, a, o) not in key_old for d, s, a, o in
                       zip(fills["datum"], fills["strategie"], fills["aktie"], fills["order"])]]
        fills = pd.concat([old_fills[fcols], fills], ignore_index=True)
    for c in ("stueck", "gebuehr_eur", "slippage_eur"):     # old rows come from the file as text
        fills[c] = to_float(fills[c])
    fills["stueck"] = fills["stueck"].round().astype("Int64")
    real, model = to_float(fills["realer_kurs"]), to_float(fills["modellkurs"])
    fills["abweichung_bp"] = pd.Series(np.where(fills["seite"] == "kauf", real / model - 1, 1 - real / model) * 1e4,
                                       index=fills.index).round(1)
    fills["realer_kurs"], fills["modellkurs"] = real, model
    fills = fills.sort_values(["datum", "strategie", "aktie"]).reset_index(drop=True)
    PROT_DIR.mkdir(exist_ok=True)
    try:
        fills.to_csv(fills_path, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    except PermissionError:
        log("ausfuehrungen.csv ist gesperrt (in Excel offen?) - neue Orders werden beim naechsten Lauf ergaenzt")
    measured = fills["abweichung_bp"].dropna()

    # ---- result files
    daily = pd.DataFrame({f"konto_{k}": a["equity"].round(2) for k, a in accts.items()})
    for k, c in bench_curves.items():
        daily[f"vergleich_{k}"] = c.reindex(daily.index).round(2)
    save_csv(daily.rename_axis("datum"), "ergebnisse.csv", index=True)
    save_csv(pd.DataFrame(pos_rows), "positionen.csv")
    save_csv(pd.concat([a["closed"] for a in accts.values()], ignore_index=True), "trades.csv")
    save_csv(pd.DataFrame(order_rows), "orders_heute.csv")

    report(accts, clean, signals_today, bench_curves, start, start_day, last, nxt, n_days, names, cols, P, n,
           measured, n_new, late)


def report(accts, clean, sig, bench_curves, start, start_day, last, nxt, n_days, names, cols, P, n, measured,
           n_new, late) -> None:
    print()
    print("=" * 100)
    print(f"TREND-PROTOKOLL (Paper Trading)  -  Robust 1 und Robust 2, je {eur(CAPITAL)} EUR, ohne Hebel  -  "
          f"Kurse bis {last:%d.%m.%Y}")
    print("=" * 100)
    print(f"Starttag {start_day:%d.%m.%Y} (angelegt am {start['angelegt_am']}), Einstellungen unveraendert. "
          f"Kosten: {tm.broker_text()}, Slippage {tm.SLIPPAGE_BPS} bp")
    for k in STRATEGIEN:
        print(f"    {k}: {BESCHREIBUNG[k]}")
    print(f"\nORDERS FUER {WOCHENTAG[nxt.weekday()]} {nxt:%d.%m.%Y} - Xetra-Schlussauktion (Order bis 17:30 Uhr):")
    any_order = False
    for k, s in sig.items():
        print(f"  {k}:")
        if not (s["buys"] or s["sells"]):
            print("      keine Kaeufe oder Verkaeufe")
        for j, q, why in s["sells"]:
            any_order = True
            print(f"    VERKAUFEN {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} {q['shares']:>6} Stueck   "
                  f"({why}; Schluss zuletzt {num(P[n - 1, j])} EUR, seit Kauf {P[n - 1, j] / q['pe'] - 1:+.1%})")
        for j, sh in s["buys"]:
            any_order = True
            print(f"    KAUFEN    {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} {sh:>6} Stueck ca. "
                  f"(ca. {eur(sh * P[n - 1, j])} EUR, Schluss zuletzt {num(P[n - 1, j])} EUR)")
        if s["waiting"]:
            print(f"      weitere Signale ohne freien Platz: "
                  + ", ".join(cols[j] for j in s["waiting"][:8]) + (" ..." if len(s["waiting"]) > 8 else ""))
        if s["stops"]:
            print(f"      Stopp-Linien (verkaufen, falls der Kurs kurz vor 17:30 Uhr darunter liegt):")
            for j, q, line in sorted(s["stops"], key=lambda x: P[n - 1, x[0]] / x[2]):
                print(f"        {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} Linie {num(line):>10} EUR   "
                      f"Schluss zuletzt {num(P[n - 1, j]):>10} EUR  (Abstand {line / P[n - 1, j] - 1:+.1%})")
    if not any_order:
        print("    -> heute keine Orders")

    print(f"\nKONTEN SEIT START ({n_days} Handelstage)")
    rows = []
    for k, a in accts.items():
        kp = kpis(a["equity"])
        cl = a["closed"]
        rows.append((k, kp, len(a["positions"]), len(cl), int((cl["ergebnis_eur"] > 0).sum()) if len(cl) else 0,
                     a["fees"], a["slippage"]))
    print(f"    {'':<36}{'Konto EUR':>12}{'Rendite':>10}{'gr. Rueckgang':>15}{'Sharpe':>8}{'Positionen':>12}"
          f"{'Trades':>8}{'Gewinner':>10}{'Gebuehren':>11}{'Slippage':>10}")
    for k, kp, npos, ntr, nwin, fees, sl in rows:
        sh = f"{kp['sharpe']:.2f}" if np.isfinite(kp["sharpe"]) else "-"
        print(f"    {k:<36}{eur(kp['stand']):>12}{kp['rendite']:>+10.2%}{kp['max_dd']:>15.2%}{sh:>8}{npos:>12}"
              f"{ntr:>8}{nwin:>10}{num(fees):>11}{num(sl):>10}")
    for k, c in bench_curves.items():
        if len(c.dropna()) > 1:
            kp = kpis(c)
            sh = f"{kp['sharpe']:.2f}" if np.isfinite(kp["sharpe"]) else "-"
            print(f"    {'Vergleich: ' + k:<36}{eur(kp['stand']):>12}{kp['rendite']:>+10.2%}{kp['max_dd']:>15.2%}{sh:>8}")
    if n_days < MIN_DAYS_FOR_SHARPE:
        print(f"    Sharpe Ratio erst ab {MIN_DAYS_FOR_SHARPE} Handelstagen (vorher ohne Aussagekraft).")
    for k, a in clean.items():
        kp = kpis(a["equity"])
        print(f"    {k} ohne die als nicht ausfuehrbar markierten Kaeufe: {eur(kp['stand'])} EUR ({kp['rendite']:+.2%})")

    for k, a in accts.items():
        if not a["positions"]:
            continue
        print(f"\nOFFENE POSITIONEN {k}:")
        for j, q in sorted(a["positions"].items(), key=lambda x: x[1]["e"]):
            print(f"    {cols[j]:<9} {str(names.get(cols[j], ''))[:30]:<31} {q['shares']:>6} Stueck  gekauft "
                  f"{num(q['pe'])} EUR, jetzt {num(P[n - 1, j])} EUR ({P[n - 1, j] / q['pe'] - 1:+.1%}), "
                  f"{n - 1 - q['e']} Tage")
    last_fills = [f for a in accts.values() for f in a["fills"] if f["datum"] == f"{last:%Y-%m-%d}"]
    if last_fills:
        print(f"\nGEBUCHT AM {last:%d.%m.%Y} (Modell, Schlussauktion):")
        for f in last_fills:
            print(f"    {f['strategie']}: {f['order']} {f['aktie']} {f['stueck']} Stueck zu {num(f['modellkurs'])} EUR"
                  + (f" ({f['grund']})" if f["order"] == "verkaufen" else ""))
    if len(measured):
        print(f"\nGemessene Abweichung vom Modellkurs: im Mittel {measured.mean():+.1f} bp je Order ({len(measured)} Kurse; "
              f"angenommen {tm.SLIPPAGE_BPS} bp). Positiv = schlechter als das Modell.")
    else:
        print("\nNoch keine realen Kurse eingetragen (protokoll/ausfuehrungen.csv, Spalte 'realer_kurs'; nicht moegliche "
              "Kaeufe in 'ausfuehrbar' mit 'nein').")
    if n_new:
        print(f"Protokolliert: {n_new} neue Zeilen in protokoll.csv"
              + (" - darunter nachgetragene Tage" if late else ""))
    elif n_days:
        print("Protokolliert: nichts Neues (dieser Kurstag steht bereits im Protokoll).")
    else:
        print("Heute ist der Starttag: Gewertet wird ab dem naechsten Handelstag.")
    print(f"Dateien: {PROT_DIR}  (protokoll.csv, ausfuehrungen.csv, orders_heute.csv, positionen.csv, trades.csv, "
          "ergebnisse.csv)")


# ======================================================================================= Backtest
def run_backtest() -> None:
    m = load_market() if KURS_DATEI.exists() else tc.load_data()
    cal = m["close"].index
    s0 = int(np.searchsorted(cal, pd.Timestamp(tc.SIGNAL_START)))
    print("\n" + "=" * 100)
    print(f"RUECKRECHNUNG ab {cal[s0]:%d.%m.%Y} bis {cal[-1]:%d.%m.%Y}  (Kosten: {tm.broker_text()}, "
          f"Slippage {tm.SLIPPAGE_BPS} bp)")
    print("=" * 100)
    for name, (entry, exit_name) in STRATEGIEN.items():
        tr, _ = build_trades(m, entry, exit_name, s0)
        a = run_account(m, tr, s0, None, name, tc.AUSSTIEGE[exit_name].get("tage", tc.MAX_HALTEDAUER))
        eq = a["equity"]
        kp = kpis(eq)
        yrs = len(eq) / 252
        y = eq.groupby(eq.index.year).last()
        yr = (y / y.shift(1).fillna(eq.iloc[0]) - 1) * 100
        print(f"\n{name} ({BESCHREIBUNG[name]}):")
        print(f"    Rendite p. a. {((kp['stand'] / CAPITAL) ** (1 / yrs) - 1):+.1%}, Sharpe {kp['sharpe']:.2f}, "
              f"groesster Rueckgang {kp['max_dd']:.1%}, Trades {len(a['closed'])}, Gebuehren {eur(a['fees'])} EUR, "
              f"Slippage {eur(a['slippage'])} EUR")
        print("    Jahre: " + "  ".join(f"{yy}: {v:+.1f} %" for yy, v in yr.items()))
    print("\nLaengere Rueckrechnung (ab 2006) und Zufallspruefung: trend_cpo_lang.py --lang, trend_bewertung.py")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):                 # never let special characters in stock names cause an abort
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Trend following Xetra: Robust 1 and Robust 2 with protocol mode")
    ap.add_argument("--protokoll", action="store_true", help="protocol mode (paper trading)")
    ap.add_argument("--protokoll-neu", dest="protokoll_neu", action="store_true", help="start the protocol anew")
    ap.add_argument("--rueckrechnung", action="store_true", help="backtest of both strategies")
    ap.add_argument("--stichtag", default=None, help=argparse.SUPPRESS)   # for testing only
    args = ap.parse_args()
    if args.protokoll or args.protokoll_neu:
        run_protocol(args.protokoll_neu, args.stichtag)
    elif args.rueckrechnung:
        run_backtest()
    else:
        ap.print_help()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
