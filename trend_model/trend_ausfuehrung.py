"""
Execution time for the trend model: closing auction or opening?
================================================================================================================

So far: signal at the closing price of day t (known only after the close of trading), purchase at the closing price of day t+1
(Xetra closing auction). This price is really tradable, so no stale price is used - but
between signal and purchase there is a night and a whole trading day.

Compared executions (each for purchase AND sale):
  A  close on the next day        as before: purchase/sale in the closing auction of the next day;
                                  stops: closing price below the stop line -> sale in the same closing auction
  B  opening on the next day      purchase in the opening auction of t+1 (9:00); time exit also at the
                                  opening; stop detected at the closing price -> sale at the next opening
  B+ shortly after the opening    like B, but continuous trading in the first minutes on EIX (at Scalable the
                                  cheapest venue; wider spreads: higher slippage, SLIP_NACH_EROEFFNUNG)
  C  opening + stop order         like B, but the trailing stop is placed in the market as a real stop order:
                                  if the intraday low falls below the line, sale at the line (with a
                                  downward price gap at the - worse - opening price)
  D  close on the signal day      upper bound: calculate the signal shortly before 17:30 with the almost final price and still
                                  buy in the same closing auction (no delay)

Additionally: How much do prices move after a signal overnight (close t -> opening t+1) and
during the day (opening t+1 -> close t+1)? This decides whether buying earlier helps.

Data: opening, high and low prices are missing in the price cache of the screener. On the first run the script
loads them with yfinance (like the screener, adjusted for dividends and splits) into pair_screener/cache/ohlc_aktien.pkl
and adjusts them to the closing prices of the cache.

Usage (in the folder trend_model):  ..\\.venv\\python.exe trend_ausfuehrung.py           (first run: download)
                                    ..\\.venv\\python.exe trend_ausfuehrung.py --neu     reload prices
Results in ergebnisse_cpo/: ausfuehrung_vergleich.csv, ausfuehrung_nacht_tag.csv, ausfuehrung.png
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

import trend_cpo as tc
import trend_model as tm

# ======================================================================================= Settings
OHLC_FILE = tm.SCREENER_CACHE / "ohlc_aktien.pkl"
DOWNLOAD_START = "2016-06-01"
DOWNLOAD_CHUNK = 40

STRATEGIEN = {                   # name: (entry from trend_cpo.EINSTIEGE, exit rule)
    "Robust 1: Hoch 200 | Zeit 20": ("Hoch 200", {"typ": "zeit", "tage": 20}),
    "Robust 2: Hoch 50 + Volumenschub | Zeit 60 + Stopp 15%": ("Hoch 50 + Volumenschub",
                                                              {"typ": "kombi", "tage": 60, "pct": 0.15}),
    "Hoch 50 + Volumenschub | Zeit 120": ("Hoch 50 + Volumenschub", {"typ": "zeit", "tage": 120}),
    "Hoch 200 | Nachzug 15%": ("Hoch 200", {"typ": "nachzug", "pct": 0.15}),
}
SLIP_SCHLUSS = 10                # basis points per purchase/sale in the closing auction (as in the trend model)
SLIP_EROEFFNUNG = 10             # opening auction
SLIP_NACH_EROEFFNUNG = 20        # continuous trading shortly after 9:00 (spreads wider in the first minutes)
MODI = {                         # name: (price for purchase, stop type, slippage, trading venue at Scalable)
    "A Schluss am Folgetag (bisher)": ("schluss_folgetag", "schluss", SLIP_SCHLUSS, "Xetra"),
    "B Eroeffnung am Folgetag": ("eroeffnung", "naechste_eroeffnung", SLIP_EROEFFNUNG, "Xetra"),
    "B+ kurz nach Eroeffnung (EIX)": ("eroeffnung", "naechste_eroeffnung", SLIP_NACH_EROEFFNUNG, "EIX"),
    "C Eroeffnung + Stopp-Order im Markt": ("eroeffnung", "stopp_order", SLIP_EROEFFNUNG, "Xetra"),
    "D Schluss am Signaltag (Obergrenze)": ("schluss_signaltag", "schluss", SLIP_SCHLUSS, "Xetra"),
}
START_DATEN = pd.date_range("2017-10-01", "2023-10-01", freq="6MS")


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Prices
def download_ohlc(tickers: list[str]) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance fehlt:  ..\\.venv\\python.exe -m pip install yfinance")
    parts = {k: [] for k in ("open", "high", "low", "close")}
    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i:i + DOWNLOAD_CHUNK]
        for attempt in range(4):
            try:
                data = yf.download(chunk, start=DOWNLOAD_START, auto_adjust=True, progress=False, threads=True,
                                   group_by="column")
                break
            except Exception as e:                          # rate limit or similar
                wait = 15 * (attempt + 1)
                log(f"  Download-Fehler ({type(e).__name__}), neuer Versuch in {wait}s")
                time.sleep(wait)
        else:
            continue
        if data is None or data.empty:
            continue
        for k, col in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
            d = data[col] if isinstance(data.columns, pd.MultiIndex) else data[[col]].set_axis(chunk[:1], axis=1)
            parts[k].append(d)
        log(f"  Eroeffnungskurse: {min(i + DOWNLOAD_CHUNK, len(tickers))}/{len(tickers)} Aktien")
        time.sleep(1)
    out = {k: pd.concat(v, axis=1) for k, v in parts.items() if v}
    for k in out:
        out[k].index = pd.to_datetime(out[k].index).tz_localize(None).normalize()
    return out


def load_ohlc(m: dict, refresh: bool) -> dict:
    tick = list(m["close"].columns)
    d = None
    if OHLC_FILE.exists() and not refresh:
        d = pd.read_pickle(OHLC_FILE)
        missing = [t for t in tick if t not in d["close"].columns]
        stale = d["close"].index.max() < m["close"].index.max() - pd.Timedelta(days=5)
        if missing or stale:
            log(f"Eroeffnungskurse veraltet oder unvollstaendig ({len(missing)} Aktien fehlen) - neu laden")
            d = None
    if d is None:
        log(f"Lade Eroeffnungs-, Hoch- und Tiefkurse fuer {len(tick)} Aktien (einmalig, einige Minuten) ...")
        d = download_ohlc(tick)
        if not d:
            raise SystemExit("Download fehlgeschlagen - Internetverbindung / yfinance pruefen.")
        pd.to_pickle(d, OHLC_FILE)
    # adjust to the calendar and the adjustment of the closing prices in the cache
    c = m["close"]
    dl_close = d["close"].reindex(index=c.index, columns=c.columns)
    factor = c / dl_close
    bad = ((factor - 1).abs() > 0.02).sum().sum() / max(factor.notna().sum().sum(), 1)
    if bad > 0.05:
        log(f"Hinweis: {bad:.0%} der Tage weichen > 2 % von den Cache-Kursen ab (unterschiedliche Bereinigung) - "
            "werden umgerechnet")
    out = {"close": c}
    for k in ("open", "high", "low"):
        x = d[k].reindex(index=c.index, columns=c.columns) * factor
        out[k] = x.where(x > 0)
    out["open"] = out["open"].where(out["open"].notna(), c)          # if the opening is missing: closing price
    out["low"] = out["low"].where(out["low"].notna(), np.minimum(out["open"], c))
    cover = d["open"].reindex(index=c.index, columns=c.columns).notna().where(c.notna()).mean().mean()
    log(f"Eroeffnungskurse vorhanden fuer {cover:.0%} der Kurstage")
    return out


# ======================================================================================= Trades
def trades_for(m: dict, px: dict, entry: str, rule: dict, mode: str, start: pd.Timestamp) -> pd.DataFrame:
    """All possible trades of an entry with purchase/sale price depending on the type of execution."""
    buy, stop_kind, _, _ = MODI[mode]
    cal = m["close"].index
    n = len(cal)
    C = m["close"].ffill().to_numpy()
    O = px["open"].to_numpy()
    L = px["low"].to_numpy()
    mom = C / np.vstack([np.full((126, C.shape[1]), np.nan), C[:-126]]) - 1
    ev = tc.entry_signals(m, [entry])[entry].copy()
    ev[cal < start] = False
    has = m["close"].notna().to_numpy()
    last = n - 1 - has[::-1].argmax(axis=0)
    rows = []
    for t, j in zip(*np.nonzero(ev)):
        e = t if buy == "schluss_signaltag" else t + 1
        if e > min(n - 1, last[j]):
            continue
        if buy == "eroeffnung":
            pe = O[e, j] if np.isfinite(O[e, j]) else C[e, j]
        else:
            pe = C[e, j]
        if not np.isfinite(pe) or pe <= 0:
            continue
        hold = rule.get("tage")
        cap = e + (hold if hold else tc.MAX_HALTEDAUER)
        x, pout, why = None, None, ""
        pct = rule.get("pct")
        if pct:                                             # trailing stop
            hi = pe
            first_d = e if stop_kind == "stopp_order" else e + 1
            for d in range(first_d, min(cap, n - 1) + 1):
                if stop_kind == "stopp_order":
                    lvl = hi * (1 - pct)                    # line from the highest close up to the previous day
                    o = O[d, j] if d > e and np.isfinite(O[d, j]) else np.inf
                    if d > e and o <= lvl:
                        x, pout, why = d, o, "Stopp (Kurslucke)"
                        break
                    if np.isfinite(L[d, j]) and L[d, j] <= lvl:
                        x, pout, why = d, lvl, "Stopp"
                        break
                    hi = max(hi, C[d, j])
                else:
                    hi = max(hi, C[d, j])
                    if C[d, j] <= hi * (1 - pct):
                        if stop_kind == "schluss":
                            x, pout, why = d, C[d, j], "Stopp"
                        else:                               # sell only on the next morning
                            x = d + 1
                            pout = (O[x, j] if x <= n - 1 and np.isfinite(O[x, j]) else C[min(x, n - 1), j])
                            why = "Stopp"
                        break
        if x is None:
            x = cap
            why = "Zeit" if hold else "Hoechstdauer"
            if x <= n - 1:
                pout = O[x, j] if (buy == "eroeffnung" and np.isfinite(O[x, j])) else C[x, j]
        if x > n - 1 or x > last[j]:                       # still open: valued at the last closing price
            x = int(min(n - 1, last[j]))
            pout, why = C[x, j], "offen"
        rows.append((j, t, e, x, pe, pout, why, mom[t, j]))
    return pd.DataFrame(rows, columns=["j", "t_signal", "e", "x", "kurs_ein", "kurs_aus", "grund", "rang"])


def simulate(m: dict, trades: pd.DataFrame, start: pd.Timestamp, slip_bps: float, at_open: bool,
             venue: str = "Xetra") -> pd.Series:
    """Portfolio like trend_model.simulate (long), but with its own purchase and sale price per trade.
    Portfolio valued daily at the closing price. Position size: purchase in the closing auction as in the trend model
    1/10 of the account at today's close; purchase at the opening 1/10 of the account from the previous evening (more is
    not known at 9:00)."""
    cal = m["close"].index
    P = m["close"].ffill().to_numpy()
    n = len(cal)
    slip = slip_bps / 1e4
    first = int(np.searchsorted(cal, start))
    by_entry: dict[int, list] = {}
    for r in trades.sort_values(["e", "rang"], ascending=[True, False]).itertuples(index=False):
        by_entry.setdefault(r.e, []).append(r)
    cash, pos = tm.CAPITAL, {}
    equity = np.full(n, np.nan)
    eq_prev = tm.CAPITAL
    cols = m["close"].columns
    fee = lambda v, j: tm.commission(v, cols[j], venue=venue)
    for t in range(first, n):
        cash -= tm.monthly_fee(cal, t, first)
        for j in [j for j, q in pos.items() if q["x"] == t]:
            q = pos.pop(j)
            value = q["shares"] * q["px_out"]
            cash += value - fee(value, j) - value * slip
        held_now = cash + sum(q["shares"] * P[t, jj] for jj, q in pos.items())
        for r in by_entry.get(t, []):
            if len(pos) >= tm.MAX_POSITIONS:
                break
            j = r.j
            if j in pos or r.x < t:
                continue
            eq_ref = eq_prev if at_open else held_now
            target = min(eq_ref / tm.MAX_POSITIONS, cash)
            shares = int(target // r.kurs_ein)
            if shares <= 0:
                continue
            alloc = shares * r.kurs_ein
            cost = fee(alloc, j) + alloc * slip
            if alloc + cost > cash:
                continue
            cash -= alloc + cost
            if r.x == t:                                    # stopped out on the day of purchase (stop order in the market)
                value = shares * r.kurs_aus
                cash += value - fee(value, j) - value * slip
                continue
            pos[j] = {"x": int(r.x), "shares": shares, "px_out": r.kurs_aus}
        equity[t] = cash + sum(q["shares"] * P[t, j] for j, q in pos.items())
        eq_prev = equity[t]
    return pd.Series(equity, index=cal).iloc[first:]


def stats(eq: pd.Series, a: pd.Timestamp | None = None) -> dict:
    s = eq.dropna()
    if a is not None:
        s = s[s.index >= a]
    r = s.pct_change().dropna()
    yrs = len(r) / 252
    return {"rendite_pa_pct": ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100,
            "sharpe": r.mean() / r.std() * np.sqrt(252), "max_drawdown_pct": (1 - s / s.cummax()).max() * 100}


# ======================================================================================= Night / day
def overnight_intraday(m: dict, px: dict) -> pd.DataFrame:
    """Average price move after signals: night (close -> opening) and day (opening -> close)
    on the signal day, on the purchase day (t+1) and on the following days; in addition all stocks for comparison."""
    cal = m["close"].index
    C = m["close"].to_numpy()
    O = px["open"].to_numpy()
    start = pd.Timestamp(tc.SIGNAL_START)
    night = O[1:] / C[:-1] - 1                              # night before day k: close k-1 -> opening k
    day = C / O - 1                                         # day k: opening k -> close k
    elig = m["elig"].to_numpy()
    rows = []
    real_open = px["open"].ne(m["close"]).to_numpy()        # only days with a real opening
    for entry in sorted({e for e, _ in STRATEGIEN.values()}):
        ev = tc.entry_signals(m, [entry])[entry].copy()
        ev[cal < start] = False
        ts, js = np.nonzero(ev)
        for k in range(0, 6):                               # k = 0: signal day itself (day only), 1 = purchase day ...
            tt = ts + k
            ok = tt < len(cal)
            tk, jk = tt[ok], js[ok]
            n_ = night[np.clip(tk - 1, 0, len(night) - 1), jk]
            d_ = day[tk, jk]
            msk = real_open[tk, jk] & np.isfinite(n_) & np.isfinite(d_)
            rows.append({"einstieg": entry, "tag_nach_signal": k, "signale": int(msk.sum()),
                         "nacht_bp": np.mean(n_[msk]) * 1e4 if k > 0 else np.nan,
                         "tag_bp": np.mean(d_[msk]) * 1e4,
                         "nacht_positiv_pct": (n_[msk] > 0).mean() * 100 if k > 0 else np.nan})
    # comparison: all tradable stocks on all days
    msk = real_open[1:] & elig[1:] & np.isfinite(night) & np.isfinite(day[1:])
    msk[cal[1:] < start] = False
    rows.append({"einstieg": "alle Aktien (Vergleich)", "tag_nach_signal": np.nan, "signale": int(msk.sum()),
                 "nacht_bp": night[msk].mean() * 1e4, "tag_bp": day[1:][msk].mean() * 1e4,
                 "nacht_positiv_pct": (night[msk] > 0).mean() * 100})
    return pd.DataFrame(rows)


# ======================================================================================= Procedure
def main() -> None:
    ap = argparse.ArgumentParser(description="Execution time for the trend model")
    ap.add_argument("--neu", action="store_true", help="reload opening prices")
    args = ap.parse_args()
    tm.OUT_DIR = tc.OUT_DIR
    m = tc.load_data()
    px = load_ohlc(m, args.neu)
    start = pd.Timestamp(tc.SIGNAL_START)
    test = pd.Timestamp("2023-01-01")

    log("Nacht- und Tagesbewegung nach Signalen ...")
    oi = overnight_intraday(m, px)
    tm.save_csv(oi, "ausfuehrung_nacht_tag.csv")

    log(f"Portfolio-Rechnung: {len(STRATEGIEN)} Strategien x {len(MODI)} Ausfuehrungen x "
        f"{len(START_DATEN) + 1} Startzeitpunkte ...")
    rows, curves = [], {}
    for sname, (entry, rule) in STRATEGIEN.items():
        for mode, (_, _, slip, venue) in MODI.items():
            tr = trades_for(m, px, entry, rule, mode, start)
            at_open = MODI[mode][0] == "eroeffnung"
            eq = simulate(m, tr, start, slip, at_open, venue)
            curves[(sname, mode)] = eq
            r = {"strategie": sname, "ausfuehrung": mode, "trades_moeglich": len(tr)}
            r.update({f"ab2017_{k}": v for k, v in stats(eq).items()})
            r.update({f"test_{k}": v for k, v in stats(eq, test).items()})
            res = []
            for s in START_DATEN:                           # robustness against the start date
                res.append(stats(simulate(m, tr[tr["e"] >= np.searchsorted(m["close"].index, s)], s, slip, at_open,
                                          venue)))
            d = pd.DataFrame(res)
            r.update({"starts_rendite_pa_mittel": d["rendite_pa_pct"].mean(), "starts_sharpe_mittel": d["sharpe"].mean(),
                      "starts_maxdd_mittel": d["max_drawdown_pct"].mean()})
            if rule.get("pct"):
                r["anteil_stopp_pct"] = tr["grund"].str.startswith("Stopp").mean() * 100
                r["anteil_kursluecke_pct"] = (tr["grund"] == "Stopp (Kurslucke)").mean() * 100
            rows.append(r)
        log(f"  {sname} fertig")
    res = pd.DataFrame(rows)
    tm.save_csv(res, "ausfuehrung_vergleich.csv")
    chart(curves)
    report(oi, res)


def chart(curves: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(dict.fromkeys(k[0] for k in curves))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), squeeze=False)
    colors = dict(zip(MODI, ["#1f77b4", "#d62728", "#ff9896", "#2ca02c", "#7f7f7f"]))
    for ax, nm in zip(axes.ravel(), names):
        for mode in MODI:
            eq = curves[(nm, mode)].dropna()
            ax.plot(eq.index, eq / 1000, label=mode, color=colors[mode], lw=1.4,
                    ls="--" if mode.startswith("D") else "-")
        ax.set_yscale("log")
        ax.set_title(nm, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Ausfuehrungszeitpunkt: Konto in Tsd. EUR (log), ab Sept. 2017")
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / "ausfuehrung.png", dpi=110)
    plt.close(fig)


def report(oi: pd.DataFrame, res: pd.DataFrame) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 120)
    print("1) KURSBEWEGUNG NACH EINEM SIGNAL (Basispunkte, Mittel; Tag 1 = Kauftag)")
    print("   nacht = Schluss Vortag -> Eroeffnung, tag = Eroeffnung -> Schluss")
    print("=" * 120)
    print(oi.round(1).to_string(index=False))
    print("\nLesart: Ist 'tag' am Kauftag (1) positiv, ist Kauf zur Eroeffnung guenstiger als in der Schlussauktion;")
    print("ist 'nacht' an Tag 1 deutlich positiv, kostet schon die Nacht nach dem Signal Rendite (dann hilft nur D).")
    print("\n" + "=" * 120)
    print(f"2) PORTFOLIO JE AUSFUEHRUNG  (Kosten: {tm.broker_text()})")
    print("=" * 120)
    cols = [c for c in ["strategie", "ausfuehrung", "ab2017_rendite_pa_pct", "ab2017_sharpe", "ab2017_max_drawdown_pct",
                        "test_rendite_pa_pct", "test_sharpe", "starts_rendite_pa_mittel", "starts_sharpe_mittel",
                        "starts_maxdd_mittel", "anteil_stopp_pct", "anteil_kursluecke_pct"] if c in res.columns]
    for s, g in res.groupby("strategie", sort=False):
        print(f"\n{s}")
        print(g[cols[1:]].round(2).to_string(index=False))
    print(f"\nDateien: {tc.OUT_DIR} (ausfuehrung_*.csv, ausfuehrung.png)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
