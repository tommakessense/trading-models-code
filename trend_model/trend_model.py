"""
Trend-following model for Xetra stocks and ETFs with media echo (sentiment analysis)
================================================================================

Entry     : classic trend signals - new 50- or 200-day high (long) or low (short) and the
            crossing of the 50-day above the 200-day average (golden cross / death cross). For stocks the signal is
            additionally filtered with the media echo (GDELT: number of articles and their tone).
Exit      : five variants, all compared in the backtest
            Zeit kurz (aggressiv)       after 20 trading days
            Zeit lang (konservativ)     after 120 trading days
            Trend                       closing price below the 20-day low (short: above the 20-day high)
            Preis aggressiv             stop loss 5 %, profit taking +10 %, trailing stop 8 %
            Preis konservativ           stop loss 15 %, no fixed profit taking, trailing stop 25 %
Universe  : Xetra stocks and ETFs from the cache of the screener (ticker <symbol>.DE), only sufficiently
            liquid securities (median turnover on Xetra in the previous year), ETFs without leveraged/short and money market ETFs.
Testing   : portfolio with 50,000 EUR and at most 10 positions, IBKR fees and slippage.
            Learning period 2018-2022 (selection of the best variant), test period from 2023.

Media echo (GDELT DOC 2.0, free, without registration): per stock and day the number of matching articles and their
average tone. From this (always only with data up to the previous day):
    stimmung_z  = tone of the last 7 days compared with its own annual mean (z-value)
    echo        = articles of the last 7 days relative to the average of the last 90 days
Sentiment variants: ohne | Stimmung positiv | Echo + Stimmung | Veto bei schlechter Stimmung

Usage (in the folder trend_model):
    ..\\.venv\\python.exe trend_model.py               backtest of all variants (loads GDELT data)
    ..\\.venv\\python.exe trend_model.py --nur-kurse   backtest without media echo
    ..\\.venv\\python.exe trend_model.py --heute       today's signals with media echo and headlines
    ..\\.venv\\python.exe trend_model.py --refresh     reload media echo data completely
    ..\\.venv\\python.exe trend_model.py --nur-medienecho  load only the media echo (can be aborted, continues later)
    ..\\.venv\\python.exe trend_model.py --max-aktien 100  media echo only for the 100 most liquid stocks
    ..\\.venv\\python.exe trend_model.py --mit-pleiten additionally calculate with bankruptcy cases (Wirecard etc.)
                                                     and compare with the result without them
    ..\\.venv\\python.exe trend_model.py --nur-cache   media echo only from the cache (no GDELT download)
    ..\\.venv\\python.exe trend_model.py --nur-cache --placebo   additionally placebo test of the sentiment
                                                     filters (200 random filters; --placebo 50 for a quick test)

Testing of the sentiment filters (runs with every backtest with media echo; result pruefung_trades.csv):
    1) Trade comparison: return of the trades the filter lets through against those it blocks - only signals
       with media data, p-value via random permutation within the same month.
    2) Placebo filters (with --placebo, pruefung_placebo.csv): random filters that discard the same number of signals,
       in the portfolio - is the Sharpe improvement of the real filter greater than that from mere thinning out?

Costs: order fees of Scalable Capital (BROKER, SCALABLE_TARIF; Xetra 1.99 EUR per order in the FREE plan),
for short sales and for stocks in scalable_nicht_handelbar.csv those of Interactive Brokers (0.05 %, min.
3 EUR); plus SLIPPAGE_BPS per order.

Bankruptcy cases: The price cache contains only stocks that are tradable today (survivorship bias). With --mit-pleiten
known bankruptcies from PLEITEN are additionally traded. The script tries to find their prices at Yahoo (also on
regional exchanges); if they are missing there, you can put them yourself as a CSV in daten/pleiten/, e.g.
daten/pleiten/WDI.csv with the columns Datum and Schluss (optionally Volumen/Stuecke/Umsatz), for instance as an export
of the historical prices from wallstreet-online.de or ariva.de. From the last price on, the stock is considered
delisted: open positions are closed at the last price x DELISTING_ERLOES.

The first run loads the GDELT data since 2017 for each stock (2 queries per stock; GDELT allows at most
one query per 5 seconds and slows down more under load). Each stock is saved immediately; an abort
loses nothing. After that, data is only added every GDELT_MAX_AGE_DAYS days.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================================================= Settings
BASE_DIR = Path(__file__).resolve().parent
SCREENER_CACHE = BASE_DIR.parent / "pair_screener" / "cache"
STOCK_FILE, ETF_FILE = "prices_aktien.pkl", "prices.pkl"
INFO_FILE, XETRA_FILE = "info_aktien.csv", "xetra_allTradableInstruments.csv"
DATA_DIR = BASE_DIR / "daten"
GDELT_DIR = DATA_DIR / "gdelt"
OUT_DIR = BASE_DIR / "ergebnisse"
QUERY_FILE = BASE_DIR / "suchbegriffe.csv"     # is created on the first run and may be edited

# --- Periods
START = "2018-01-01"             # first entries (media echo needs one year of lead time from 2017)
TRAIN_END = "2022-12-31"         # learning period up to here, test period afterwards

# --- Universe
MIN_TURNOVER_EUR = 1e6           # median daily turnover on Xetra over the last 250 days (up to the previous day)
MIN_PRICE = 1.0
MIN_HISTORY = 250                # trading days with a price before the first signal
ETF_MIN_VOL = 0.05               # exclude ETFs with hardly any volatility (money market, short-term bonds)
ETF_MAX_CORR = 0.97              # almost identical ETFs (e.g. several S&P 500 ETFs): only the most liquid one
ETF_EXCLUDE = r"LEV|SHORT|\bX2\b|\b2X|\b3X|INVERSE|DAILY|\bDLY|OVERNIGHT|MONEY MARK|\bCASH\b|0-1Y|SWAP RATE"
BENCHMARK = {"aktien": "EXS1.DE", "etf": "EUNL.DE"}   # iShares Core DAX or Core MSCI World

# --- Signals
BREAKOUT_SHORT, BREAKOUT_LONG = 50, 200
SMA_FAST, SMA_SLOW = 50, 200
ALLOW_SHORT = True               # short only for stocks (ETFs on Xetra can mostly not be borrowed)
MOMENTUM_DAYS = 126              # ranking if there are more signals than free slots: strongest trend first

# --- Exit
EXITS = {
    "Zeit kurz (aggressiv)": {"typ": "zeit", "tage": 20},
    "Zeit lang (konservativ)": {"typ": "zeit", "tage": 120},
    "Trend (20-Tage-Tief)": {"typ": "trend", "tage": 20},
    "Preis aggressiv": {"typ": "preis", "stop": 0.05, "ziel": 0.10, "nachzug": 0.08},
    "Preis konservativ": {"typ": "preis", "stop": 0.15, "ziel": None, "nachzug": 0.25},
}
MAX_HOLD_DAYS = 250              # upper limit for all variants
ORDERS_IN_MARKET = True          # price variants: stop and profit target are placed in the market as orders
                                 #   profit target: limit order, executed at the target price as soon as a closing price
                                 #   is above it (the price must have reached it during the day);
                                 #   stops: on the same day at the closing price (cautious, the price is already below)
                                 # False: all exits only at the closing price of the following day

# --- Media echo
GDELT_START = "20170101000000"
GDELT_PAUSE = 6.0                # minimum interval between two queries (GDELT: at most one per 5 s);
                                 #   after a rejection (too many queries) it is extended automatically
GDELT_PAUSE_MAX = 120.0
GDELT_TIMEOUT = (15, 150)        # seconds: connection setup, waiting for the response
GDELT_CHUNK_YEARS = 3            # if the query over all years takes too long: load in 3-year pieces
GDELT_MIN_ARTICLES_TONE = 30     # fewer articles in total: do not query the tone at all (saves time)
GDELT_MAX_AGE_DAYS = 30          # backtest: add media echo only after 30 days (for --heute: 12 hours)
GDELT_CONTEXT = "(Aktie OR Aktien OR Börse OR shares OR stock)"   # only articles related to the stock market
MAX_SENTIMENT_STOCKS = 200       # media echo for the N most liquid stocks
MIN_ARTICLES_7D = 3              # fewer articles in 7 days: no statement on sentiment
TONE_Z_STRONG = 0.5              # "Echo + Stimmung": sentiment clearly better than usual
BUZZ_MIN = 1.5                   # "Echo + Stimmung": at least 1.5 times as many articles as usual
TONE_Z_VETO = -1.0               # "Veto": no purchase if the sentiment is clearly worse than usual
SENTIMENTS = ["ohne", "Stimmung positiv", "Echo + Stimmung", "Veto bei schlechter Stimmung"]

# testing of the sentiment filters (trade comparison always, placebo filters with --placebo)
PRUEF_TRADE_LAEUFE = 1000        # random permutations for the p-value of the trade comparison
PRUEF_PLACEBO_LAEUFE = 200       # random filters per sentiment variant (portfolio calculation, takes time)
PRUEF_SCHICHT = "M"              # randomness only within the same month ("M") or year ("Y"): "M" tests whether
                                 # the filter finds the better ones among simultaneous signals (stricter, because
                                 # signals in the same month are strongly related); "Y" lets market timing count as well
PRUEF_MIN_TRADES = 5             # at least this many trades per group (let through / blocked)
PRUEF_SEED = 7

# --- Account and costs
CAPITAL = 50_000.0
MAX_POSITIONS = 10
SLIPPAGE_BPS = 10                # half spread + price move until execution (independent of the broker)
BORROW_FEE = 0.01                # borrowing fee p. a. for short sales

# Broker: Scalable Capital for all stocks tradable there, otherwise Interactive Brokers.
# Scalable does not offer short sales of stocks - short trades always go through IBKR.
BROKER = "scalable"              # "scalable" or "ibkr" (then everything at IBKR fees)
SCALABLE_TARIF = "FREE"          # "FREE" (0 EUR per month) or "PRIME+" (4.99 EUR per month)
SCALABLE_HANDELSPLATZ = "Xetra"  # purchase/sale in the closing auction only exists on Xetra (gettex and EIX have
                                 #   no auction, but prices of a market maker)
SCALABLE_GEBUEHR = {             # EUR per order, as of October 2026 (justETF, Broker-Test.at, press August 2026) -
    "FREE": {"Xetra": 1.99, "gettex": 1.99, "EIX": 0.99},     # please check in the list of prices and services
    "PRIME+": {"Xetra": 1.99, "gettex": 1.99, "EIX": 0.00},   # PRIME+: EIX free of charge from 250 EUR, below that 0.99 EUR
}
SCALABLE_EIX_KLEINORDER = (250.0, 0.99)   # PRIME+ on EIX: below 250 EUR order volume 0.99 EUR
SCALABLE_MONAT = {"FREE": 0.0, "PRIME+": 4.99}
SCALABLE_NICHT_HANDELBAR = BASE_DIR / "scalable_nicht_handelbar.csv"   # column "ticker": stocks NOT tradable
                                 #   there (calculated at IBKR fees); if the file is missing: all tradable
IBKR_PCT = 0.0005                # IBKR Fixed Europe: 0.05 % of the order volume ...
IBKR_MIN = 3.0                   # ... at least 3 EUR
COMMISSION_PCT, COMMISSION_MIN = IBKR_PCT, IBKR_MIN   # old names (for older scripts)

# --- Bankruptcy cases (only with --mit-pleiten). ISIN for searching price data - please check when downloading.
PLEITEN_DIR = DATA_DIR / "pleiten"
PLEITEN = {
    "WDI": ("Wirecard AG", "DE0007472060", "Insolvenzantrag 06/2020 nach Bilanzbetrug, Delisting Frankfurt 11/2021"),
    "SNH": ("Steinhoff International", "NL0011375019", "Bilanzskandal 12/2017, Aktionaere 2023 leer ausgegangen"),
    "GWI1": ("Gerry Weber International", "DE0003304101", "Insolvenz 01/2019, Altaktien wertlos"),
    "SEN": ("Senvion", "LU1377527517", "Insolvenz 04/2019"),
    "LEO": ("Leoni AG", "DE0005408884", "Kapitalschnitt auf null (StaRUG) 2023, Aktionaere leer ausgegangen"),
    "GGS": ("Gigaset AG", "DE0005156004", "Insolvenzantrag 09/2023"),
    "VAR1": ("Varta AG", "DE000A0TGJ55", "Kapitalschnitt auf null (StaRUG), rechtskraeftig 01/2025"),
}
PLEITEN_YAHOO = [".DE", ".F", ".SG", ".DU", ".BE", ".MU", ".HM"]   # Xetra and regional exchanges
PLEITEN_ANNAHME_LIQUIDE = True   # file without turnover data: treat as sufficiently liquid
DELISTING_ERLOES = 1.0           # 1.0 = sale at the last price, 0.0 = total loss of open positions

# --- today's signals (--heute)
LIVE = {"richtung": "long", "einstieg": "50-Tage-Hoch", "stimmung": "Stimmung positiv",
        "ausstieg": "Preis konservativ"}
LIVE_HEADLINES = 3               # most positive and most negative headlines per candidate (7 days)
LIVE_MAX_CANDIDATES = 15

ENTRIES = {"long": [f"{BREAKOUT_SHORT}-Tage-Hoch", f"{BREAKOUT_LONG}-Tage-Hoch", "Golden Cross"],
           "short": [f"{BREAKOUT_SHORT}-Tage-Tief", f"{BREAKOUT_LONG}-Tage-Tief", "Death Cross"]}


# ======================================================================================= Helpers
def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    try:
        df.to_csv(path, sep=";", decimal=",", index=index, encoding="utf-8-sig")
    except PermissionError:
        alt = OUT_DIR / f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}"
        df.to_csv(alt, sep=";", decimal=",", index=index, encoding="utf-8-sig")
        log(f"{name} ist gesperrt (in Excel offen?) - gespeichert als {alt.name}")


_NOT_SCALABLE: set | None = None


def not_at_scalable() -> set:
    global _NOT_SCALABLE
    if _NOT_SCALABLE is None:
        _NOT_SCALABLE = set()
        if SCALABLE_NICHT_HANDELBAR.exists():
            df = pd.read_csv(SCALABLE_NICHT_HANDELBAR, sep=None, engine="python", encoding="utf-8-sig")
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            _NOT_SCALABLE = {str(t).strip().upper() for t in df[col].dropna()}
            _NOT_SCALABLE |= {t + ".DE" for t in _NOT_SCALABLE if not t.endswith(".DE")}
    return _NOT_SCALABLE


def commission(value: float, ticker: str | None = None, short: bool = False, venue: str | None = None) -> float:
    """Order fee in EUR: Scalable (flat per order) if the stock is tradable there and it is not a
    short sale; otherwise IBKR (0.05 %, at least 3 EUR)."""
    if value <= 0:
        return 0.0
    if BROKER == "scalable" and not short and (ticker is None or ticker.upper() not in not_at_scalable()):
        v = venue or SCALABLE_HANDELSPLATZ
        fee = SCALABLE_GEBUEHR[SCALABLE_TARIF][v]
        if v == "EIX" and SCALABLE_TARIF == "PRIME+" and value < SCALABLE_EIX_KLEINORDER[0]:
            fee = SCALABLE_EIX_KLEINORDER[1]
        return fee
    return max(IBKR_MIN, IBKR_PCT * value)


def monthly_fee(cal: pd.DatetimeIndex, t: int, first: int) -> float:
    """Account fee (Scalable PRIME+: 4.99 EUR) on the first trading day of each month."""
    if BROKER != "scalable" or not SCALABLE_MONAT.get(SCALABLE_TARIF):
        return 0.0
    return SCALABLE_MONAT[SCALABLE_TARIF] if t == first or cal[t].month != cal[t - 1].month else 0.0


def cost_pct(short: bool = False) -> float:
    """Order fee in percent of a typical position (account / MAX_POSITIONS) - for estimates per trade."""
    v = CAPITAL / MAX_POSITIONS
    return commission(v, short=short) / v


def broker_text() -> str:
    if BROKER != "scalable":
        return "IBKR Fixed (0,05 %, mind. 3 EUR)"
    fee = SCALABLE_GEBUEHR[SCALABLE_TARIF][SCALABLE_HANDELSPLATZ]
    return (f"Scalable {SCALABLE_TARIF}, {SCALABLE_HANDELSPLATZ} {fee:.2f} EUR je Order"
            + (f" + {SCALABLE_MONAT[SCALABLE_TARIF]:.2f} EUR/Monat" if SCALABLE_MONAT[SCALABLE_TARIF] else "")
            + "; Short und nicht handelbare Aktien: IBKR")


# ======================================================================================= Data
def xetra_names() -> pd.DataFrame:
    path = SCREENER_CACHE / XETRA_FILE
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.splitlines()
    h = next(i for i, line in enumerate(lines) if line.startswith("Product Status"))
    rows = list(csv.reader(lines[h:], delimiter=";"))
    hdr = [c.strip() for c in rows[0]]
    df = pd.DataFrame([(r + [""] * len(hdr))[:len(hdr)] for r in rows[1:] if r], columns=hdr)
    df["ticker"] = df["Mnemonic"].str.strip() + ".DE"
    df["name"] = df["Instrument"].str.strip()
    df["gruppe"] = df["Product Assignment Group Description"].str.upper()
    return df.drop_duplicates("ticker").set_index("ticker")[["name", "gruppe"]]


def drop_running_day(close: pd.DataFrame) -> pd.DataFrame:
    """Do not use the current Xetra trading day (Yahoo provides a preliminary price during trading)."""
    try:
        now = pd.Timestamp.now(tz="Europe/Berlin")
        today = now.normalize().tz_localize(None)
        if len(close) and close.index[-1] >= today and now.hour * 60 + now.minute < 17 * 60 + 45:
            log(f"Handelstag {today:%d.%m.%Y} laeuft noch - er wird erst nach 17:45 Uhr verwendet")
            return close[close.index < today]
    except Exception:
        pass
    return close


def load_market(kind: str, extra: bool = False) -> dict:
    """Prices, turnover, names and tradability for 'aktien' or 'etf' (extra: with bankruptcy cases)."""
    f = SCREENER_CACHE / (STOCK_FILE if kind == "aktien" else ETF_FILE)
    if not f.exists():
        raise SystemExit(f"Kurs-Cache fehlt: {f}\nBitte den Screener im Xetra-Modus laufen lassen "
                         f"(MARKET = \"xetra\", MODE = \"{'aktien' if kind == 'aktien' else 'etf'}\").")
    raw = pd.read_pickle(f)
    close = raw["close"].sort_index().astype(float)
    cnt = close.notna().sum(axis=1)
    close = close[cnt >= 0.5 * cnt.rolling(21, center=True, min_periods=1).median()]   # holidays/gaps
    close = drop_running_day(close)
    volume = raw["volume"].reindex(index=close.index, columns=close.columns).astype(float)
    xn = xetra_names()
    names = xn["name"].reindex(close.columns)
    if kind == "aktien":
        info_path = SCREENER_CACHE / INFO_FILE
        if info_path.exists():
            info = pd.read_csv(info_path, keep_default_na=False, na_values=[""]).set_index("ticker")
            names = info["long_name"].reindex(close.columns).fillna(names)
    else:
        bad = names.fillna("").str.upper().str.contains(ETF_EXCLUDE, regex=True)
        keep = names.index[~bad & names.notna()]
        close, volume, names = close[keep], volume[keep], names[keep]
    added = []
    if extra and kind == "aktien":
        for key, (df, src) in load_pleiten(close.index).items():
            t = f"{key}.DE"
            s = df["close"].reindex(close.index)
            if s.notna().sum() < 50:
                log(f"Pleitefall {key}: Kursdaten passen nicht zum Xetra-Kalender - ausgelassen")
                continue
            close = close.assign(**{t: s})
            volume = volume.assign(**{t: df["volume"].reindex(close.index) if "volume" in df else np.nan})
            names[t] = f"{PLEITEN[key][0]} (Pleitefall)"
            added.append(t)
    turnover = (close * volume).rolling(250, min_periods=120).median().shift(1)
    for t in added:
        if PLEITEN_ANNAHME_LIQUIDE and volume[t].notna().sum() < 0.5 * close[t].notna().sum():
            turnover[t] = MIN_TURNOVER_EUR                  # turnover unknown: treat as tradable
    history = close.notna().cumsum() >= MIN_HISTORY
    elig = (turnover >= MIN_TURNOVER_EUR) & history & (close.shift(1) >= MIN_PRICE)
    if kind == "etf":
        vol = np.log(close).diff().rolling(250, min_periods=120).std().shift(1) * np.sqrt(252)
        elig &= vol >= ETF_MIN_VOL
        ever = elig.any()
        cand = ever[ever].index
        med = (close[cand] * volume[cand]).median().sort_values(ascending=False)
        r = np.log(close[cand]).diff().iloc[-750:]
        corr = r.corr(min_periods=120)
        kept = []
        for t in med.index:                                 # most liquid first, skip duplicates
            if all(not (corr.at[t, k] >= ETF_MAX_CORR) for k in kept):
                kept.append(t)
        drop = [t for t in cand if t not in kept]
        elig[drop] = False
        log(f"ETFs: {len(cand)} liquide, davon {len(drop)} nahezu gleiche ausgelassen -> {len(kept)}")
    ever = elig.any()
    tick = ever[ever].index
    log(f"{'Aktien' if kind == 'aktien' else 'ETFs'}: {len(tick)} Titel mindestens zeitweise handelbar "
        f"(Median-Umsatz >= {MIN_TURNOVER_EUR / 1e6:g} Mio. EUR), Kurse {close.index[0]:%d.%m.%Y} bis "
        f"{close.index[-1]:%d.%m.%Y}")
    bench = None
    if BENCHMARK[kind] in raw["close"].columns:
        bench = raw["close"][BENCHMARK[kind]].reindex(close.index).astype(float)
    elif kind == "aktien" and (SCREENER_CACHE / ETF_FILE).exists():
        etf_close = pd.read_pickle(SCREENER_CACHE / ETF_FILE)["close"]
        if BENCHMARK[kind] in etf_close.columns:
            bench = etf_close[BENCHMARK[kind]].reindex(close.index).astype(float)
    return {"kind": kind, "close": close[tick], "elig": elig[tick], "names": names.reindex(tick),
            "turnover": turnover[tick], "bench": bench, "pleiten": [t for t in added if t in tick]}


def parse_price_file(path: Path) -> pd.DataFrame:
    """Read a price file (own export): date, closing price, optionally number of shares or turnover in EUR."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = [line for line in text.splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if re.search(r"datum|date", line, re.I))
    sep = ";" if lines[start].count(";") >= lines[start].count(",") else ","
    rows = list(csv.reader(lines[start:], delimiter=sep))
    hdr = [h.strip().lower() for h in rows[0]]
    df = pd.DataFrame([(r + [""] * len(hdr))[:len(hdr)] for r in rows[1:]], columns=hdr)

    def pick(*keys):
        for k in keys:
            for c in hdr:
                if k in c:
                    return c
        return None

    def clean(col):
        return df[col].astype(str).str.strip().str.replace("\u00a0", "").str.replace(" ", "").str.replace("'", "")

    def is_german(col):                                     # detect from the closing price: 12,34 or 1.234,56
        s = clean(col)
        return s.str.fullmatch(r"-?[\d.]*\d,\d+").mean() > s.str.fullmatch(r"-?[\d,]*\d\.\d+").mean()

    def num(col):
        s = clean(col)
        if german:
            s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            s = s.str.replace(",", "", regex=False)
        return pd.to_numeric(s.str.replace(r"[^0-9.\-eE]", "", regex=True), errors="coerce")

    dcol = pick("datum", "date")
    ccol = pick("schlusskurs", "schluss", "close", "letzter", "kurs")
    if dcol is None or ccol is None:
        raise ValueError(f"Spalten Datum/Schluss nicht gefunden: {hdr}")
    german = is_german(ccol)
    ds = df[dcol].str.strip()
    if ds.str.match(r"\d{4}-\d{2}-\d{2}").mean() > 0.5:      # 2020-01-31
        dates = pd.to_datetime(ds.str[:10], format="%Y-%m-%d", errors="coerce")
    elif ds.str.match(r"\d{1,2}/\d{1,2}/\d{4}").mean() > 0.5:  # 01/31/2020 (US)
        dates = pd.to_datetime(ds, format="%m/%d/%Y", errors="coerce")
    else:                                                   # 31.01.2020
        dates = pd.to_datetime(ds, dayfirst=True, errors="coerce")
    out = pd.DataFrame({"close": num(ccol).to_numpy()}, index=dates)
    scol, ecol = pick("stück", "stueck", "stuck", "shares"), pick("umsatz")
    vcol = pick("volumen", "volume")
    if scol:
        out["volume"] = num(scol).to_numpy()
    elif ecol:
        out["volume"] = (num(ecol) / num(ccol)).to_numpy()   # turnover in EUR -> shares
    elif vcol:
        out["volume"] = num(vcol).to_numpy()
    out = out[out.index.notna() & (out["close"] > 0)].sort_index()
    return out[~out.index.duplicated(keep="last")]


def load_pleiten(cal: pd.DatetimeIndex) -> dict[str, tuple[pd.DataFrame, str]]:
    """Prices of the bankruptcy cases: own file in daten/pleiten, otherwise Yahoo (Xetra or regional exchange)."""
    PLEITEN_DIR.mkdir(parents=True, exist_ok=True)
    out, missing = {}, []
    for key, (name, isin, event) in PLEITEN.items():
        own = [p for p in [PLEITEN_DIR / f"{key}.csv", PLEITEN_DIR / f"{key}.DE.csv", PLEITEN_DIR / f"{isin}.csv"]
               if p.exists()]
        if own:
            try:
                out[key] = (parse_price_file(own[0]), own[0].name)
                continue
            except Exception as e:
                log(f"{own[0].name} nicht lesbar ({e}) - versuche Yahoo")
        cache = PLEITEN_DIR / f"{key}_yahoo.csv"
        if cache.exists():
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if len(df):
                out[key] = (df, "Yahoo (zwischengespeichert)")
                continue
        best, sym = None, ""
        try:
            import yfinance as yf
            for suf in PLEITEN_YAHOO:
                try:
                    d = yf.download(key + suf, start=cal[0] - pd.Timedelta(days=10), auto_adjust=True,
                                    progress=False)
                except Exception:
                    continue
                if d is None or d.empty:
                    continue
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d.rename(columns={"Close": "close", "Volume": "volume"})[["close", "volume"]].dropna(subset=["close"])
                if best is None or len(d) > len(best):
                    best, sym = d, key + suf
        except ImportError:
            pass
        if best is not None and len(best) >= 100:
            best.index = pd.to_datetime(best.index).tz_localize(None) if best.index.tz is not None else best.index
            best.to_csv(cache)
            out[key] = (best, f"Yahoo {sym}")
        else:
            missing.append(f"{key} ({name}, ISIN {isin})")
    for key, (df, src) in out.items():
        log(f"Pleitefall {PLEITEN[key][0]}: {len(df)} Kurse {df.index.min():%d.%m.%Y} bis {df.index.max():%d.%m.%Y} "
            f"(Quelle: {src})")
    if missing:
        log("Keine Kurse gefunden fuer: " + "; ".join(missing))
        log(f"  -> historische Kurse als CSV (Spalten Datum, Schluss, optional Volumen) in {PLEITEN_DIR} ablegen, "
            "Dateiname = Kuerzel, z. B. WDI.csv")
    return out


# ======================================================================================= Signals
def entry_events(close: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Entry events at the closing price (True only on the first day of a breakout or at the crossing)."""
    out = {}
    for n in (BREAKOUT_SHORT, BREAKOUT_LONG):
        hi = close.rolling(n, min_periods=n).max().shift(1)
        lo = close.rolling(n, min_periods=n).min().shift(1)
        up, dn = close > hi, close < lo
        out[("long", f"{n}-Tage-Hoch")] = up & ~up.shift(1, fill_value=False)
        out[("short", f"{n}-Tage-Tief")] = dn & ~dn.shift(1, fill_value=False)
    fast = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    slow = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    above = fast > slow
    valid = fast.notna() & slow.notna() & fast.shift(1).notna() & slow.shift(1).notna()
    out[("long", "Golden Cross")] = above & ~above.shift(1, fill_value=False) & valid
    out[("short", "Death Cross")] = ~above & above.shift(1, fill_value=False) & valid
    return out


def exit_index(p: np.ndarray, lo20: np.ndarray, hi20: np.ndarray, e: int, d: int, rule: dict) -> tuple:
    """Exit for an entry at the closing price of day e (direction d = +1/-1): (day, reason, price).
    Price = NaN means the closing price of the exit day. A signal at the closing price of day t is executed at the closing price
    t+1; with ORDERS_IN_MARKET, stops and profit targets are already executed on day t."""
    n = len(p)
    cap = e + MAX_HOLD_DAYS
    if rule["typ"] == "zeit":
        x = e + rule["tage"]
        return (x, "Zeit", np.nan) if x <= n - 1 else (n - 1, "offen", np.nan)
    end = min(n - 1, cap)
    seg = p[e + 1:end + 1]
    if len(seg) == 0:
        return n - 1, "offen", np.nan
    t = None
    reason = ""
    if rule["typ"] == "trend":
        lvl = lo20[e + 1:end + 1] if d > 0 else hi20[e + 1:end + 1]
        hit = np.flatnonzero(seg < lvl) if d > 0 else np.flatnonzero(seg > lvl)
        if len(hit):
            t, reason = e + 1 + hit[0], "Trendbruch"
    else:
        pe = p[e]
        r = d * (seg / pe - 1)
        ext = np.maximum.accumulate(np.concatenate([[pe], seg]))[1:] if d > 0 else \
            np.minimum.accumulate(np.concatenate([[pe], seg]))[1:]
        draw = d * (seg / ext - 1)                           # distance from the best price since entry (<= 0)
        cands = []
        h = np.flatnonzero(r <= -rule["stop"])
        if len(h):
            cands.append((h[0], "Stop-Loss"))
        if rule.get("ziel"):
            h = np.flatnonzero(r >= rule["ziel"])
            if len(h):
                cands.append((h[0], "Gewinnziel"))
        if rule.get("nachzug"):
            h = np.flatnonzero(draw <= -rule["nachzug"])
            if len(h):
                cands.append((h[0], "Nachzugsstopp"))
        if cands:
            k, reason = min(cands)
            t = e + 1 + k
    if t is None:
        if cap <= n - 1:
            return cap, "Hoechstdauer", np.nan
        return n - 1, "offen", np.nan
    if rule["typ"] == "preis" and ORDERS_IN_MARKET:
        if reason == "Gewinnziel":
            return t, reason, p[e] * (1 + d * rule["ziel"])
        return t, reason, np.nan
    x = t + 1
    return (x, reason, np.nan) if x <= n - 1 else (n - 1, "offen", np.nan)


def build_trades(m: dict, events: pd.DataFrame, direction: str, exit_name: str, start: pd.Timestamp) -> pd.DataFrame:
    """All possible trades of an entry signal (portfolio limits only come in simulate)."""
    close = m["close"]
    cal = close.index
    pf = close.ffill()
    P = pf.to_numpy()
    lo20 = pf.rolling(EXITS[exit_name].get("tage", 20)).min().shift(1).to_numpy()
    hi20 = pf.rolling(EXITS[exit_name].get("tage", 20)).max().shift(1).to_numpy()
    mom = (pf / pf.shift(MOMENTUM_DAYS) - 1).to_numpy()
    has = close.notna().to_numpy()
    last = len(cal) - 1 - has[::-1].argmax(axis=0)          # last day with a price per stock
    delisted = last < len(cal) - 11                         # price series ends prematurely -> delisting
    ev = (events & m["elig"]).to_numpy().copy()
    ev[cal < start] = False
    d = 1 if direction == "long" else -1
    rule = EXITS[exit_name]
    rows = []
    ts, js = np.nonzero(ev)
    for t, j in zip(ts, js):
        e = t + 1                                           # execution at the next closing price
        if e > len(cal) - 1 or not np.isfinite(P[e, j]):
            continue
        if e > last[j]:
            continue
        x, why, px = exit_index(P[:, j], lo20[:, j], hi20[:, j], e, d, rule)
        if delisted[j] and (x > last[j] or why == "offen"):          # still open at delisting
            x, why, px = int(last[j]), "Delisting", P[last[j], j] * DELISTING_ERLOES
        px_out = P[x, j] if np.isnan(px) else px
        rows.append((j, t, e, x, px, d * (px_out / P[e, j] - 1), x - e, why, d * mom[t, j]))
    return pd.DataFrame(rows, columns=["j", "t_signal", "e", "x", "kurs_aus", "rendite", "tage", "grund", "rang"])


# ======================================================================================= Portfolio
def simulate(m: dict, trades: pd.DataFrame, direction: str, start: pd.Timestamp, with_log: bool = True) -> dict:
    cal = m["close"].index
    P = m["close"].ffill().to_numpy()
    n = len(cal)
    d = 1 if direction == "long" else -1
    slip = SLIPPAGE_BPS / 1e4
    first = int(np.searchsorted(cal, start))
    by_entry: dict[int, list] = {}
    for r in trades.sort_values(["e", "rang"], ascending=[True, False]).itertuples(index=False):
        by_entry.setdefault(r.e, []).append(r)
    cash = CAPITAL
    pos: dict[int, dict] = {}                               # column j -> position
    equity = np.full(n, np.nan)
    log_rows = []
    for t in range(first, n):
        if d > 0:
            cash -= monthly_fee(cal, t, first)              # e.g. Scalable PRIME+ 4.99 EUR/month
        # 1. exits (execution at today's closing price)
        for j in [j for j, q in pos.items() if q["x"] == t]:
            q = pos.pop(j)
            px = P[t, j] if np.isnan(q["px_out"]) else q["px_out"]
            value = q["alloc"] + d * q["shares"] * (px - q["pe"])
            gross = q["shares"] * px
            cost = commission(gross, m["close"].columns[j], d < 0) + gross * slip
            cash += value - cost
            if not with_log:
                continue
            pnl = value - cost - q["alloc"] - q["cost_in"] - q["borrow"]
            log_rows.append({"ticker": m["close"].columns[j], "einstieg": cal[q["e"]].date(), "ausstieg": cal[t].date(),
                             "tage": t - q["e"], "grund": q["grund"], "einstand_eur": round(q["alloc"], 2),
                             "ergebnis_eur": round(pnl, 2), "rendite_pct": pnl / q["alloc"] * 100})
        # 2. value of the account before new entries
        held = sum(q["alloc"] + d * q["shares"] * (P[t, j] - q["pe"]) for j, q in pos.items())
        eq_now = cash + held
        # 3. entries (signal yesterday, execution today)
        for r in by_entry.get(t, []):
            if len(pos) >= MAX_POSITIONS:
                break
            j = r.j
            if j in pos or not np.isfinite(P[t, j]) or r.x <= t:
                continue
            target = min(eq_now / MAX_POSITIONS, cash)
            shares = int(target // P[t, j])
            if shares <= 0:
                continue
            alloc = shares * P[t, j]
            cost = commission(alloc, m["close"].columns[j], d < 0) + alloc * slip
            if alloc + cost > cash:
                continue
            cash -= alloc + cost
            pos[j] = {"e": t, "x": int(r.x), "pe": P[t, j], "shares": shares, "alloc": alloc, "cost_in": cost,
                      "grund": r.grund, "borrow": 0.0, "px_out": r.kurs_aus}
        # 4. borrowing fee for short sales
        if d < 0 and pos:
            for j, q in pos.items():
                fee = q["shares"] * P[t, j] * BORROW_FEE / 252
                q["borrow"] += fee
                cash -= fee
        held = sum(q["alloc"] + d * q["shares"] * (P[t, j] - q["pe"]) for j, q in pos.items())
        equity[t] = cash + held
    eq = pd.Series(equity, index=cal).iloc[first:]
    return {"equity": eq, "trades": pd.DataFrame(log_rows)}


def metrics(eq: pd.Series, tl: pd.DataFrame, a: str, b: str | None) -> dict:
    s = eq[(eq.index >= pd.Timestamp(a)) & ((eq.index <= pd.Timestamp(b)) if b else True)]
    if len(s) < 20:
        return {}
    r = s.pct_change().dropna()
    years = len(s) / 252
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if s.iloc[0] > 0 and s.iloc[-1] > 0 else -1.0
    dd = (1 - s / s.cummax()).max()
    out = {"rendite_pa_pct": cagr * 100, "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
           "max_drawdown_pct": dd * 100}
    if not tl.empty:
        sel = tl[(pd.to_datetime(tl["ausstieg"]) >= pd.Timestamp(a))
                 & ((pd.to_datetime(tl["ausstieg"]) <= pd.Timestamp(b)) if b else True)]
        out.update({"trades": len(sel), "trefferquote": (sel["ergebnis_eur"] > 0).mean() if len(sel) else np.nan,
                    "je_trade_pct": sel["rendite_pct"].mean() if len(sel) else np.nan,
                    "haltedauer": sel["tage"].mean() if len(sel) else np.nan})
    return out


# ======================================================================================= Media echo (GDELT)
_pace = {"last": 0.0, "pause": GDELT_PAUSE}
TIMEOUT = "timeout"


def gdelt_get(params: dict):
    """One query to the GDELT DOC 2.0 API (JSON). Returns: dict, None (error) or TIMEOUT.

    GDELT rejects queries that come faster than one per 5 seconds - even if at the same time a
    second program (or a second run of this script) queries from the same computer. After a rejection
    the interval is doubled and after successful queries slowly shortened again."""
    import requests
    for attempt in range(7):
        wait = _pace["pause"] - (time.time() - _pace["last"])
        if wait > 0:
            time.sleep(wait)
        _pace["last"] = time.time()
        try:
            r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params={**params, "format": "json"},
                             timeout=GDELT_TIMEOUT, headers={"User-Agent": "trend_model (privates Forschungsskript)"})
        except requests.exceptions.ReadTimeout:
            return TIMEOUT                                  # caller splits the period
        except requests.exceptions.RequestException as e:
            log(f"  GDELT nicht erreichbar ({type(e).__name__}) - neuer Versuch in {20 * (attempt + 1)} s")
            time.sleep(20 * (attempt + 1))
            continue
        txt = r.text.strip()
        if r.status_code == 429 or "limit requests" in txt.lower():
            _pace["pause"] = min(GDELT_PAUSE_MAX, max(15.0, _pace["pause"] * 2))
            log(f"  GDELT bremst (zu viele Abfragen) - Abstand jetzt {_pace['pause']:.0f} s"
                + (" (laeuft das Skript evtl. zweimal?)" if attempt == 2 else ""))
            time.sleep(_pace["pause"])
            continue
        if r.status_code >= 500:
            time.sleep(20 * (attempt + 1))
            continue
        _pace["pause"] = max(GDELT_PAUSE, _pace["pause"] * 0.8)
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            log(f"  GDELT-Antwort: {txt[:150]}")
            return None
    return None


def fetch_timeline(query: str, mode: str, key: str, start: pd.Timestamp, end: pd.Timestamp,
                   first: bool = True) -> pd.DataFrame | None:
    """Time series for [start, end]. If the response takes too long, the period is split
    (first into GDELT_CHUNK_YEARS-year pieces, then halved, down to 3 months)."""
    js = gdelt_get({"query": query, "mode": mode, "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                    "enddatetime": end.strftime("%Y%m%d%H%M%S")})
    if js is None:
        return None
    if js is not TIMEOUT:
        return parse_timeline(js, key)
    if (end - start).days <= 92:
        log(f"  GDELT antwortet nicht ({start:%m/%Y}-{end:%m/%Y})")
        return None
    if first and (end - start).days > 366 * GDELT_CHUNK_YEARS:
        cuts = list(pd.date_range(start, end, freq=f"{GDELT_CHUNK_YEARS}YS"))
        bounds = sorted({start, *[c for c in cuts if start < c < end], end})
    else:
        bounds = [start, start + (end - start) / 2, end]
    log(f"  Zeitueberschreitung - lade {start:%Y}-{end:%Y} in {len(bounds) - 1} Teilen")
    parts = []
    for a, b in zip(bounds, bounds[1:]):
        part = fetch_timeline(query, mode, key, a, b - pd.Timedelta(seconds=1) if b != end else b, first=False)
        if part is None:
            return None
        parts.append(part)
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def parse_timeline(js: dict | None, key: str) -> pd.DataFrame:
    rows = []
    for series in (js or {}).get("timeline", []) or []:
        for item in series.get("data", []) or []:
            rows.append((item.get("date"), item.get("value"), item.get("norm")))
        break
    if not rows:
        return pd.DataFrame(columns=[key])
    df = pd.DataFrame(rows, columns=["date", key, "norm"])
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df[key] = pd.to_numeric(df[key], errors="coerce")
    return df.groupby("date")[[key]].mean() if key == "tone" else df.groupby("date")[[key]].sum()


def query_table(m: dict) -> pd.DataFrame:
    """Search terms per stock. The file suchbegriffe.csv can be edited (column 'abfrage')."""
    overrides = {"MUV2.DE": "Munich Re", "BMW.DE": "BMW", "MRK.DE": "Merck KGaA", "VOW3.DE": "Volkswagen",
                 "MBG.DE": "Mercedes-Benz", "DTE.DE": "Deutsche Telekom", "BAYN.DE": "Bayer AG",
                 "CON.DE": "Continental AG", "HEN3.DE": "Henkel", "P911.DE": "Porsche AG", "PAH3.DE": "Porsche SE",
                 "SIE.DE": "Siemens AG", "DB1.DE": "Deutsche Börse", "HNR1.DE": "Hannover Rück", "DHL.DE": "DHL Group",
                 "1COV.DE": "Covestro", "ADS.DE": "Adidas", "BEI.DE": "Beiersdorf", "SAP.DE": "SAP SE",
                 "IFX.DE": "Infineon", "SHL.DE": "Siemens Healthineers", "MTX.DE": "MTU Aero", "AIR.DE": "Airbus"}
    legal = r"(\bGmbH\s*)?&\s*Co\.?(\s*KGaA)?|Kommanditgesellschaft auf Aktien|\b(Aktiengesellschaft|AG|SE|KGaA|" \
            r"N\.V\.?|NV|S\.A\.?|SA|S\.p\.A\.?|plc|PLC|Inc\.?|Ltd\.?|Corp\.?|Corporation|GmbH|Holdings?|" \
            r"in München|Inh\.?|O\.N\.?)(?=\s|$|,)"
    rows = []
    for t, name in m["names"].items():
        clean = overrides.get(t)
        if not clean:
            clean = re.sub(legal, " ", str(name or ""), flags=re.I)
            clean = re.sub(r"[,\.]+$", "", re.sub(r"\s+", " ", clean)).strip(" ,.-&")
        rows.append({"ticker": t, "name": name, "suchbegriff": clean,
                     "abfrage": f'"{clean}" {GDELT_CONTEXT}' if len(clean) >= 3 else ""})
    new = pd.DataFrame(rows)
    if QUERY_FILE.exists():
        old = pd.read_csv(QUERY_FILE, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        new = pd.concat([old, new[~new["ticker"].isin(old["ticker"])]], ignore_index=True)
    try:
        new.to_csv(QUERY_FILE, sep=";", index=False, encoding="utf-8-sig")
    except PermissionError:
        pass
    return new.set_index("ticker")


def load_gdelt(tickers: list[str], queries: pd.DataFrame, refresh: bool,
               max_age_days: float = GDELT_MAX_AGE_DAYS, offline: bool = False) -> dict:
    """Daily number of articles and tone per stock (cache per stock in daten/gdelt).

    Each finished stock is saved immediately: An abort (Ctrl+C, timeout, computer off)
    loses nothing, the next run continues at the same point."""
    GDELT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "gdelt_status.csv"
    status = pd.read_csv(status_path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("ticker") \
        if status_path.exists() else pd.DataFrame(columns=["ergebnis", "artikel", "sekunden", "stand", "abfrage"])
    if "abfrage" not in status:
        status["abfrage"] = ""
    status.index.name = "ticker"

    def query_changed(t: str) -> bool:                      # search term in suchbegriffe.csv changed?
        q = queries.at[t, "abfrage"] if t in queries.index else ""
        return t in status.index and status.at[t, "abfrage"] not in ("", q)

    out, todo, changed = {}, [], set()
    for t in tickers:
        path = GDELT_DIR / f"{t}.csv"
        if query_changed(t):
            changed.add(t)
            todo.append(t)
        elif path.exists() and not refresh and time.time() - path.stat().st_mtime < max_age_days * 86400:
            out[t] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            todo.append(t)
    if changed:
        log(f"Suchbegriff geaendert - Medienecho wird neu geladen fuer: {', '.join(sorted(changed))}")
    if not todo:
        return out
    if offline:                                             # --nur-cache: download nothing
        for t in todo:
            path = GDELT_DIR / f"{t}.csv"
            if path.exists():
                out[t] = pd.read_csv(path, index_col=0, parse_dates=True)
        log(f"Nur Cache: Medienecho fuer {len(out)} Aktien vorhanden, nichts nachgeladen")
        return out
    lock = GDELT_DIR / "_laeuft.txt"
    if lock.exists() and time.time() - lock.stat().st_mtime < 3 * 3600:
        log("Medienecho wird offenbar gerade von einem anderen Lauf geladen (daten/gdelt/_laeuft.txt) - "
            "zwei Laeufe gleichzeitig bremsen sich bei GDELT gegenseitig aus. Es wird nur der Cache verwendet. "
            "(Laeuft sicher nichts mehr: Datei loeschen.)")
        for t in todo:
            path = GDELT_DIR / f"{t}.csv"
            if path.exists():
                out[t] = pd.read_csv(path, index_col=0, parse_dates=True)
        return out
    log(f"Lade Medienecho von GDELT fuer {len(todo)} Aktien (Abbruch jederzeit moeglich, Fortsetzung beim "
        "naechsten Lauf) ...")
    failed, t_start, done = [], time.time(), 0
    try:
        for n, t in enumerate(todo, 1):
            lock.write_text(f"{time.strftime('%d.%m.%Y %H:%M')} {t}", encoding="utf-8")
            t0 = time.time()
            q = queries.at[t, "abfrage"] if t in queries.index else ""
            path = GDELT_DIR / f"{t}.csv"
            old = pd.read_csv(path, index_col=0, parse_dates=True) \
                if path.exists() and not refresh and t not in changed else None
            if not q:
                failed.append(t)
                status.loc[t] = ["kein Suchbegriff", "", "", time.strftime("%Y-%m-%d"), ""]
                continue
            start = pd.Timestamp(GDELT_START[:8]) if old is None or old.empty else old.index.max() - pd.Timedelta(days=10)
            end = pd.Timestamp.utcnow().tz_localize(None).floor("min")
            vol = fetch_timeline(q, "timelinevolraw", "artikel", start, end)
            tone = None
            if vol is not None:
                total = float(vol["artikel"].sum()) + (float(old["artikel"].sum()) if old is not None else 0)
                if total >= GDELT_MIN_ARTICLES_TONE:
                    tone = fetch_timeline(q, "timelinetone", "tone", start, end)
                else:
                    tone = pd.DataFrame(columns=["tone"])   # too few articles: no statement on sentiment possible
            if vol is None or tone is None:
                failed.append(t)
                status.loc[t] = ["Fehler (naechster Lauf versucht es erneut)", "", f"{time.time() - t0:.0f}",
                                 time.strftime("%Y-%m-%d"), status.at[t, "abfrage"] if t in status.index else ""]
                if old is not None:
                    out[t] = old
                continue
            df = vol.join(tone, how="outer")
            if old is not None and not old.empty:
                df = pd.concat([old[old.index < df.index.min()] if len(df) else old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path)
            out[t] = df
            done += 1
            status.loc[t] = ["ok", f"{df['artikel'].sum():.0f}", f"{time.time() - t0:.0f}", time.strftime("%Y-%m-%d"), q]
            per = (time.time() - t_start) / n
            log(f"  {n}/{len(todo)} {t}: {df['artikel'].sum():.0f} Artikel ({time.time() - t0:.0f} s), "
                f"Rest ca. {per * (len(todo) - n) / 60:.0f} min")
    except KeyboardInterrupt:
        log(f"Abgebrochen nach {done} Aktien - der naechste Lauf setzt fort.")
        raise
    finally:
        status[["ergebnis", "artikel", "sekunden", "stand", "abfrage"]].reset_index().to_csv(
            status_path, sep=";", index=False, encoding="utf-8-sig")
        try:
            lock.unlink()
        except OSError:
            pass
    if failed:
        log(f"Kein Medienecho fuer {len(failed)} Aktien: {', '.join(failed[:12])}{' ...' if len(failed) > 12 else ''}"
            " (siehe daten/gdelt_status.csv; Suchbegriffe in suchbegriffe.csv)")
    return out


def sentiment_tickers(m: dict, start: pd.Timestamp) -> list[str]:
    """The most liquid MAX_SENTIMENT_STOCKS stocks (median turnover from START) - loaded in this order."""
    top = m["turnover"].loc[m["turnover"].index >= start].median().sort_values(ascending=False)
    return top.index[:MAX_SENTIMENT_STOCKS].tolist()


def check_recent(gd: dict, cal: pd.DatetimeIndex) -> None:
    """Warns if GDELT provides hardly any articles for the last days (delay at GDELT)."""
    last_days = []
    for df in gd.values():
        if df is not None and "artikel" in df and (df["artikel"] > 0).any():
            last_days.append(df.index[df["artikel"] > 0].max())
    if last_days:
        newest = max(last_days)
        if newest < cal[-1] - pd.Timedelta(days=3):
            log(f"HINWEIS: GDELT liefert seit {newest:%d.%m.%Y} keine Artikel mehr (Verzoegerung oder Stoerung bei "
                "GDELT) - die Stimmung der letzten Tage fehlt.")


def sentiment_frames(gd: dict, cal: pd.DatetimeIndex, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """stimmung_z and echo per trading day (only articles up to the previous day: GDELT counts by UTC days,
    the daily bucket also contains articles after the Xetra close)."""
    tz, bz = {}, {}
    for t in tickers:
        df = gd.get(t)
        if df is None or df.empty or "artikel" not in df:
            continue
        days = pd.date_range(df.index.min(), max(df.index.max(), cal[-1]), freq="D")
        df = df.reindex(days)
        if "tone" not in df or df["tone"].notna().sum() == 0:
            continue                                        # too few articles - no statement on sentiment
        c = df["artikel"].fillna(0)
        tone = df["tone"].where(c > 0)
        c, tone = c.shift(1), tone.shift(1)                 # previous days only
        c7 = c.rolling(7, min_periods=1).sum()
        ct = c.where(tone.notna(), 0)                       # only days for which a tone is available
        t7 = (ct * tone.fillna(0)).rolling(7, min_periods=1).sum() / ct.rolling(7, min_periods=1).sum()
        t7 = t7.where(c7 >= MIN_ARTICLES_7D)
        mu = t7.rolling(365, min_periods=120).mean()
        sd = t7.rolling(365, min_periods=120).std()
        c90 = c.rolling(90, min_periods=60).sum()
        tz[t] = ((t7 - mu) / sd).reindex(cal)
        bz[t] = (c7 / (c90 / 90 * 7)).reindex(cal)
    return pd.DataFrame(tz, index=cal).reindex(columns=tickers), pd.DataFrame(bz, index=cal).reindex(columns=tickers)


def sentiment_mask(variant: str, direction: str, tone_z: pd.DataFrame, buzz: pd.DataFrame) -> pd.DataFrame:
    s = 1 if direction == "long" else -1
    z = tone_z * s
    if variant == "ohne":
        return pd.DataFrame(True, index=tone_z.index, columns=tone_z.columns)
    if variant == "Stimmung positiv":
        return z > 0
    if variant == "Echo + Stimmung":
        return (buzz >= BUZZ_MIN) & (z > TONE_Z_STRONG)
    if variant == "Veto bei schlechter Stimmung":
        return ~(z < TONE_Z_VETO)
    raise ValueError(variant)


# ======================================================================================= Backtest
def backtest(args, extra: bool = False, kinds: tuple = ("aktien", "etf")) -> tuple[pd.DataFrame, dict, list]:
    t0 = time.time()
    start = pd.Timestamp(START)
    results, curves, all_trades, coverage, pleiten = [], {}, {}, None, []
    check_ctx: dict = {}
    for kind in kinds:
        m = load_market(kind, extra)
        pleiten += m.get("pleiten", [])
        cal = m["close"].index
        events = entry_events(m["close"])
        sentiments = ["ohne"]
        tone_z = buzz = None
        if kind == "aktien" and not args.nur_kurse:
            tickers = sentiment_tickers(m, start)
            gd = load_gdelt(tickers, query_table(m), args.refresh, offline=args.nur_cache)
            check_recent(gd, cal)
            tone_z, buzz = sentiment_frames(gd, cal, list(m["close"].columns))
            avail = tone_z.loc[cal >= start].notna()
            coverage = pd.DataFrame({
                "ticker": tickers, "name": m["names"].reindex(tickers).values,
                "erster_tag": [gd[t].index.min().date() if t in gd and len(gd[t]) else None for t in tickers],
                "artikel_gesamt": [int(gd[t]["artikel"].sum()) if t in gd and "artikel" in gd[t] else 0 for t in tickers],
                "anteil_tage_mit_stimmung": avail.reindex(columns=tickers).mean().values})
            save_csv(coverage, "medienecho_abdeckung.csv")
            share = avail.reindex(columns=tickers).mean().mean()
            first_days = [d for d in coverage["erster_tag"] if d is not None]
            log(f"Medienecho: {sum(t in gd for t in tickers)} Aktien, im Mittel an {share:.0%} der Handelstage ab "
                f"{start:%Y} ein Stimmungswert" + (f", fruehester Tag {min(first_days)}" if first_days else ""))
            if share < 0.2:
                log("WARNUNG: zu wenig Medienecho-Daten fuer eine aussagekraeftige Rueckrechnung der Stimmungs-"
                    "varianten (GDELT liefert evtl. nur die letzten Monate) - Ergebnisse dieser Varianten mit Vorsicht")
            sentiments = SENTIMENTS
            check_ctx.update(m=m, tone_z=tone_z, buzz=buzz)
        directions = ["long", "short"] if kind == "aktien" and ALLOW_SHORT else ["long"]
        for direction in directions:
            for entry in ENTRIES[direction]:
                ev = events[(direction, entry)]
                for exit_name in EXITS:
                    base = build_trades(m, ev, direction, exit_name, start)
                    if len(sentiments) > 1:
                        check_ctx.setdefault("bases", {})[(direction, entry, exit_name)] = base
                    for sv in sentiments:
                        if sv == "ohne":
                            tr = base
                        else:
                            mask = sentiment_mask(sv, direction, tone_z, buzz).to_numpy()
                            tr = base[mask[base["t_signal"].to_numpy(), base["j"].to_numpy()]]
                        sim = simulate(m, tr, direction, start)
                        key = (kind, direction, entry, sv, exit_name)
                        row = {"markt": kind, "richtung": direction, "einstieg": entry, "stimmung": sv,
                               "ausstieg": exit_name, "signale": len(tr),
                               "signale_je_trade_pct": (tr["rendite"].mean() * 100
                                                        - 2 * (SLIPPAGE_BPS / 100 + cost_pct(direction == "short") * 100))
                               if len(tr) else np.nan}
                        for label, a, b in [("lern", START, TRAIN_END), ("test", pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1), None),
                                            ("gesamt", START, None)]:
                            for k, v in metrics(sim["equity"], sim["trades"], str(a), b).items():
                                row[f"{label}_{k}"] = v
                        results.append(row)
                        curves[key] = sim["equity"]
                        all_trades[key] = sim["trades"].assign(name=lambda d: d["ticker"].map(m["names"]))
            log(f"{kind} {direction}: {len(ENTRIES[direction]) * len(EXITS) * len(sentiments)} Varianten gerechnet")
        if m["bench"] is not None:
            b = m["bench"][m["bench"].index >= start].ffill().dropna()
            beq = CAPITAL * b / b.iloc[0]
            row = {"markt": kind, "richtung": "long", "einstieg": f"Kaufen und Halten {BENCHMARK[kind]}",
                   "stimmung": "-", "ausstieg": "-"}
            for label, a, bb in [("lern", START, TRAIN_END), ("test", pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1), None),
                                 ("gesamt", START, None)]:
                for k, v in metrics(beq, pd.DataFrame(), str(a), bb).items():
                    row[f"{label}_{k}"] = v
            results.append(row)
            curves[(kind, "bench")] = beq
    res = pd.DataFrame(results)
    save_csv(res, "varianten.csv")
    effect = sentiment_effect(res)
    if not effect.empty:
        save_csv(effect, "stimmung_effekt.csv")
    best = pick_best(res)
    save_csv(best, "beste_varianten.csv")
    for r in best.itertuples(index=False):
        key = (r.markt, r.richtung, r.einstieg, r.stimmung, r.ausstieg)
        if key in all_trades:
            save_csv(all_trades[key], f"trades_{r.markt}_{r.richtung}.csv")
    chart(res, best, curves)
    report(res, best, effect, coverage, time.time() - t0)
    if check_ctx.get("bases"):
        c = check_ctx
        tc = check_trades(c["m"], c["bases"], c["tone_z"], c["buzz"])
        pc = check_placebo(c["m"], c["bases"], c["tone_z"], c["buzz"], args.placebo) if args.placebo else None
        report_checks(tc, pc)
    return res, all_trades, pleiten


def compare_pleiten(res0: pd.DataFrame, res1: pd.DataFrame, trades1: dict, pleiten: list[str]) -> None:
    """Comparison without / with bankruptcy cases and all trades in bankrupt stocks."""
    keys = ["markt", "richtung", "einstieg", "stimmung", "ausstieg"]
    vals = [c for c in ["lern_rendite_pa_pct", "lern_sharpe", "lern_max_drawdown_pct", "test_rendite_pa_pct",
                        "test_sharpe", "test_max_drawdown_pct"] if c in res0.columns]
    m = res0[keys + vals].merge(res1[keys + vals], on=keys, suffixes=("_ohne", "_mit"))
    for v in vals:
        m[f"{v}_diff"] = m[f"{v}_mit"] - m[f"{v}_ohne"]
    rows = []
    for key, tl in trades1.items():
        if tl is None or tl.empty:
            continue
        sel = tl[tl["ticker"].isin(pleiten)]
        for r in sel.itertuples(index=False):
            rows.append({**dict(zip(keys, key)), "ticker": r.ticker, "name": getattr(r, "name", ""), "einstieg_tag": r.einstieg,
                         "ausstieg_tag": r.ausstieg, "grund": r.grund, "rendite_pct": r.rendite_pct, "ergebnis_eur": r.ergebnis_eur})
    pt = pd.DataFrame(rows)
    save_csv(m, "vergleich_ohne_mit_pleiten.csv")
    save_csv(pt, "pleite_trades.csv")
    a = m[(m["markt"] == "aktien") & (m["ausstieg"] != "-")]
    print("\n" + "=" * 110)
    print("EINFLUSS DER PLEITEFAELLE (Aktien; mit minus ohne)")
    print("=" * 110)
    if not pleiten:
        print("Keine Kursdaten fuer Pleitefaelle vorhanden - kein Vergleich moeglich.")
        return
    print("Aufgenommen: " + ", ".join(pleiten))
    cols = [c for c in ["lern_rendite_pa_pct_diff", "lern_sharpe_diff", "lern_max_drawdown_pct_diff",
                        "test_rendite_pa_pct_diff", "test_sharpe_diff"] if c in a.columns]
    print("\nMittlere Veraenderung je Richtung und Ausstieg:")
    print(a.groupby(["richtung", "ausstieg"])[cols].mean().round(2).to_string())
    if not pt.empty:
        print("\nTrades in Pleite-Aktien (alle Varianten):")
        s = pt.groupby(["ticker", "richtung"]).agg(trades=("rendite_pct", "size"), mittel_pct=("rendite_pct", "mean"),
                                                   schlechtester_pct=("rendite_pct", "min"), bester_pct=("rendite_pct", "max"))
        print(s.round(1).to_string())
        worst = pt.sort_values("rendite_pct").head(8)
        print("\nSchlechteste Einzeltrades:")
        print(worst[["ticker", "richtung", "einstieg", "ausstieg", "einstieg_tag", "ausstieg_tag", "grund", "rendite_pct"]]
              .round(1).to_string(index=False))
    else:
        print("\nIn keiner Variante wurde eine Pleite-Aktie gehandelt.")
    print(f"\nDateien: {OUT_DIR / 'vergleich_ohne_mit_pleiten.csv'}, {OUT_DIR / 'pleite_trades.csv'}")


def sentiment_effect(res: pd.DataFrame) -> pd.DataFrame:
    """Sharpe ratio of each sentiment variant minus 'ohne' with the same entry and exit."""
    r = res[(res["markt"] == "aktien") & res["stimmung"].isin(SENTIMENTS)]
    if r["stimmung"].nunique() < 2:
        return pd.DataFrame()
    base = r[r["stimmung"] == "ohne"].set_index(["richtung", "einstieg", "ausstieg"])
    rows = []
    for sv in SENTIMENTS[1:]:
        s = r[r["stimmung"] == sv].set_index(["richtung", "einstieg", "ausstieg"])
        for direction in s.index.get_level_values(0).unique():
            idx = [i for i in s.index if i[0] == direction]
            for per in ["lern", "test"]:
                diff = (s.loc[idx, f"{per}_sharpe"] - base.loc[idx, f"{per}_sharpe"]).dropna()
                rows.append({"stimmung": sv, "richtung": direction, "zeitraum": per, "varianten": len(diff),
                             "sharpe_diff_mittel": diff.mean(), "besser_als_ohne": f"{int((diff > 0).sum())} von {len(diff)}"})
    return pd.DataFrame(rows)


# ======================================================================================= Testing of the sentiment filters
def sentiment_available(variant: str, tone_z: pd.DataFrame, buzz: pd.DataFrame) -> pd.DataFrame:
    """Where the filter is based on real media data. Without data, the absence alone decides: 'Stimmung positiv'
    and 'Echo + Stimmung' then block, 'Veto' lets through - this is a universe effect, not a sentiment effect."""
    a = tone_z.notna()
    if variant == "Echo + Stimmung":
        a &= buzz.notna()
    return a


def _strata(cal: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(cal.year * 100 + (cal.month if PRUEF_SCHICHT == "M" else 0))


def _signals(bases: dict, direction: str, entry: str, A: np.ndarray) -> np.ndarray:
    """All signals (day * columns + stock) of an entry with media data - shared by all exits,
    so that a random filter discards the same signals in all exit variants (like the real filter)."""
    J = A.shape[1]
    codes = [bases[(direction, entry, x)]["t_signal"].to_numpy() * J + bases[(direction, entry, x)]["j"].to_numpy()
             for x in EXITS if (direction, entry, x) in bases]
    u = np.unique(np.concatenate(codes)).astype(np.int64) if codes else np.array([], dtype=np.int64)
    return u[A[u // J, u % J]] if len(u) else u


def _permuted(labels: np.ndarray, strata: np.ndarray, reps: int, rng) -> np.ndarray:
    """Row 0: real filter decisions; rows 1..reps: the same decisions randomly distributed over the signals
    of the same month (the same number let through, but without any relation to the sentiment)."""
    n = len(labels)
    out = np.empty((reps + 1, n), dtype=bool)
    out[0] = labels
    src = labels[np.argsort(strata, kind="stable")]
    for r in range(1, reps + 1):
        out[r, np.lexsort((rng.random(n), strata))] = src
    return out


def _lookup(b: pd.DataFrame, sig: np.ndarray, J: int) -> tuple[np.ndarray, np.ndarray]:
    """Position of each trade in sig (only trades with media data: ok=True)."""
    code = b["t_signal"].to_numpy().astype(np.int64) * J + b["j"].to_numpy()
    if not len(sig):
        return np.zeros(len(b), dtype=int), np.zeros(len(b), dtype=bool)
    pos = np.minimum(np.searchsorted(sig, code), len(sig) - 1)
    return pos, sig[pos] == code


def check_trades(m: dict, bases: dict, tone_z: pd.DataFrame, buzz: pd.DataFrame,
                 reps: int = PRUEF_TRADE_LAEUFE) -> pd.DataFrame:
    """Test 1: Are the trades that the filter lets through better than those it blocks?

    Only signals with media data, all possible trades (without portfolio limit), return after costs.
    p-value: share of the random permutations (within the same month) that produce a lead at least as
    large. Small p-value (< 0.05) = lead can hardly be explained by chance."""
    cal = m["close"].index
    J = len(m["close"].columns)
    strata = _strata(cal)
    lern_end = pd.Timestamp(TRAIN_END)
    cost = 2 * (SLIPPAGE_BPS / 1e4 + cost_pct())
    rng = np.random.default_rng(PRUEF_SEED)
    rows, detail = [], []
    directions = [d for d in ("long", "short") if any(k[0] == d for k in bases)]
    for sv in SENTIMENTS[1:]:
        A = sentiment_available(sv, tone_z, buzz).to_numpy()
        for direction in directions:
            M = sentiment_mask(sv, direction, tone_z, buzz).to_numpy()
            per_var = {"lern": [], "test": []}
            for entry in ENTRIES[direction]:
                sig = _signals(bases, direction, entry, A)
                if len(sig) < 2 * PRUEF_MIN_TRADES:
                    continue
                L = _permuted(M[sig // J, sig % J], strata[sig // J], reps, rng)
                for exit_name in EXITS:
                    b = bases[(direction, entry, exit_name)]
                    pos, ok = _lookup(b, sig, J)
                    r = b["rendite"].to_numpy()[ok] - cost
                    pos = pos[ok]
                    is_lern = cal[b["t_signal"].to_numpy()[ok]] <= lern_end
                    for per, sel in (("lern", is_lern), ("test", ~is_lern)):
                        Lp = L[:, pos[sel]].astype(float)
                        rr = r[sel]
                        n_pass = Lp.sum(axis=1)
                        n_block = Lp.shape[1] - n_pass
                        if n_pass[0] < PRUEF_MIN_TRADES or n_block[0] < PRUEF_MIN_TRADES:
                            continue
                        mp = Lp @ rr / np.maximum(n_pass, 1)
                        mb = (1 - Lp) @ rr / np.maximum(n_block, 1)
                        per_var[per].append((mp - mb, n_pass[0], n_block[0], mp[0], mb[0]))
                        detail.append({"stimmung": sv, "richtung": direction, "einstieg": entry, "ausstieg": exit_name,
                                       "zeitraum": per, "durchgelassen": int(n_pass[0]), "geblockt": int(n_block[0]),
                                       "rendite_durchgelassen_pct": mp[0] * 100, "rendite_geblockt_pct": mb[0] * 100,
                                       "differenz_pct": (mp[0] - mb[0]) * 100,
                                       "p_wert": (1 + (mp[1:] - mb[1:] >= mp[0] - mb[0]).sum()) / (1 + reps)})
            for per, lst in per_var.items():
                if not lst:
                    continue
                D = np.mean(np.vstack([x[0] for x in lst]), axis=0)      # mean over the variants
                real, null = D[0], D[1:]
                rows.append({"stimmung": sv, "richtung": direction, "zeitraum": per, "varianten": len(lst),
                             "durchgelassen": np.mean([x[1] for x in lst]), "geblockt": np.mean([x[2] for x in lst]),
                             "rendite_durchgelassen_pct": np.mean([x[3] for x in lst]) * 100,
                             "rendite_geblockt_pct": np.mean([x[4] for x in lst]) * 100,
                             "differenz_pct": real * 100,
                             "zufall_5pct": np.percentile(null, 5) * 100, "zufall_95pct": np.percentile(null, 95) * 100,
                             "p_wert": (1 + (null >= real).sum()) / (1 + reps)})
    out = pd.DataFrame(rows)
    save_csv(out, "pruefung_trades.csv")
    save_csv(pd.DataFrame(detail), "pruefung_trades_detail.csv")
    return out


def _sharpe(eq: pd.Series, a: pd.Timestamp, b: pd.Timestamp | None) -> float:
    s = eq[(eq.index >= a) & ((eq.index <= b) if b is not None else True)]
    r = s.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if len(r) > 20 and r.std() > 0 else np.nan


def check_placebo(m: dict, bases: dict, tone_z: pd.DataFrame, buzz: pd.DataFrame, reps: int) -> pd.DataFrame:
    """Test 2: placebo filters in the portfolio.

    The random filter discards exactly as many signals with media data as the real one (per month), treats stocks
    without media data exactly like the real one and applies equally to all exit variants. As in
    stimmung_effekt.csv, the mean Sharpe difference relative to 'ohne' over all 15 entry/exit
    variants is measured. Additionally 'nur Medien-Aktien': let all signals with media data through, without sentiment -
    shows how much of the effect comes solely from the restriction to the stocks with media data."""
    cal = m["close"].index
    J = len(m["close"].columns)
    strata = _strata(cal)
    start = pd.Timestamp(START)
    periods = {"lern": (start, pd.Timestamp(TRAIN_END)), "test": (pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1), None)}
    rng = np.random.default_rng(PRUEF_SEED + 1)
    directions = [d for d in ("long", "short") if any(k[0] == d for k in bases)]
    jobs = [(sv, d) for d in directions for sv in SENTIMENTS[1:]]
    n_var = sum(1 for d in directions for _ in ENTRIES[d] for _ in EXITS)
    total = (reps + 2) * n_var * len(SENTIMENTS[1:]) + n_var
    log(f"Placebo-Pruefung: {reps} Zufallsfilter je Stimmungsvariante, {total} Portfolio-Rechnungen ...")
    t0, done = time.time(), 0

    def sharpes(tr: pd.DataFrame, direction: str) -> tuple[float, float]:
        nonlocal done
        eq = simulate(m, tr, direction, start, with_log=False)["equity"]
        done += 1
        if done % 500 == 0:
            el = time.time() - t0
            log(f"  {done}/{total} Rechnungen, Rest ca. {el / done * (total - done) / 60:.0f} min")
        return tuple(_sharpe(eq, a, b) for a, b in periods.values())

    ohne = {k: sharpes(b, k[0]) for k, b in bases.items()}
    rows = []
    for sv, direction in jobs:
        M = sentiment_mask(sv, direction, tone_z, buzz).to_numpy()
        A = sentiment_available(sv, tone_z, buzz).to_numpy()
        prep = []
        for entry in ENTRIES[direction]:
            sig = _signals(bases, direction, entry, A)
            L = _permuted(M[sig // J, sig % J], strata[sig // J], reps, rng)
            for exit_name in EXITS:
                b = bases[(direction, entry, exit_name)]
                t, j = b["t_signal"].to_numpy(), b["j"].to_numpy()
                pos, ok = _lookup(b, sig, J)
                prep.append((b, M[t, j], ok, pos, L, ohne[(direction, entry, exit_name)]))
        diffs = np.full((reps + 2, len(prep), 2), np.nan)    # row 0 real, 1 media stocks only, 2.. placebo
        for v, (b, lab0, ok, pos, L, base) in enumerate(prep):
            for k in range(reps + 2):
                if k == 0:
                    lab = lab0
                elif k == 1:
                    lab = lab0 | ok                          # all signals with data through, without data as in the real one
                else:
                    lab = lab0.copy()
                    lab[ok] = L[k - 1, pos[ok]]
                diffs[k, v] = np.subtract(sharpes(b[lab], direction), base)
        mean = np.nanmean(diffs, axis=1)                     # (reps+2, 2): mean over the 15 variants
        for p, per in enumerate(periods):
            real, ref, null = mean[0, p], mean[1, p], mean[2:, p]
            null = null[np.isfinite(null)]
            rows.append({"stimmung": sv, "richtung": direction, "zeitraum": per, "varianten": len(prep),
                         "sharpe_diff_echt": real, "sharpe_diff_nur_medienaktien": ref,
                         "placebo_mittel": null.mean(), "placebo_5pct": np.percentile(null, 5),
                         "placebo_95pct": np.percentile(null, 95), "laeufe": len(null),
                         "p_wert": (1 + (null >= real).sum()) / (1 + len(null))})
    out = pd.DataFrame(rows)
    save_csv(out, "pruefung_placebo.csv")
    log(f"Placebo-Pruefung fertig ({(time.time() - t0) / 60:.1f} min)")
    return out


def report_checks(tc: pd.DataFrame, pc: pd.DataFrame | None) -> None:
    print("\n" + "=" * 110)
    print("PRUEFUNG DER STIMMUNGSFILTER")
    print("=" * 110)
    lbl = "Monats" if PRUEF_SCHICHT == "M" else "Jahres"
    if tc is not None and not tc.empty:
        print("\n1) Trade-Vergleich: Rendite je Trade nach Kosten (%), durchgelassene minus geblockte Signale,")
        print(f"   nur Aktien mit Mediendaten, Mittel ueber die Varianten. p-Wert aus {PRUEF_TRADE_LAEUFE} "
              f"Zufallsvertauschungen innerhalb des {lbl}.")
        print(tc.round(3).to_string(index=False))
    if pc is not None and not pc.empty:
        print("\n2) Placebo-Filter: mittlere Sharpe-Differenz gegenueber 'ohne' - echter Filter, nur Beschraenkung")
        print("   auf Aktien mit Mediendaten, und Zufallsfilter mit gleich vielen geblockten Signalen.")
        print(pc.round(3).to_string(index=False))
    print("\nLesart: p-Wert < 0,05 in Lern- UND Testzeitraum = Hinweis auf echte Information im Filter.")
    print("Bei 6 Filter/Richtungs-Kombinationen ist ein einzelnes p < 0,05 auch durch Zufall zu erwarten.")
    if pc is not None and not pc.empty:
        print("Liegt 'echt' nahe bei 'nur_medienaktien', kommt die Wirkung aus der Aktienauswahl, nicht aus der Stimmung.")
    print(f"Dateien: {OUT_DIR / 'pruefung_trades.csv'}"
          + (f", {OUT_DIR / 'pruefung_placebo.csv'}" if pc is not None else "")
          + f", {OUT_DIR / 'pruefung_trades_detail.csv'}")


def pick_best(res: pd.DataFrame) -> pd.DataFrame:
    """Best variant per market and direction by the Sharpe ratio in the learning period (not in the test!)."""
    r = res[~res["einstieg"].str.startswith("Kaufen")]
    out = []
    for (kind, direction), g in r.groupby(["markt", "richtung"]):
        g = g.dropna(subset=["lern_sharpe"])
        g = g[g.get("lern_trades", pd.Series(0, index=g.index)).fillna(0) >= 20]
        if not g.empty:
            out.append(g.sort_values("lern_sharpe", ascending=False).iloc[0])
    return pd.DataFrame(out)


def chart(res: pd.DataFrame, best: pd.DataFrame, curves: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    kinds = sorted(best["markt"].unique()) if not best.empty else []
    if not kinds:
        return
    fig, axes = plt.subplots(len(kinds), 1, figsize=(12, 5 * len(kinds)), squeeze=False)
    for ax, kind in zip(axes[:, 0], kinds):
        for r in best[best["markt"] == kind].itertuples(index=False):
            key = (r.markt, r.richtung, r.einstieg, r.stimmung, r.ausstieg)
            ax.plot(curves[key].index, curves[key].values / 1000,
                    label=f"{r.richtung}: {r.einstieg}, {r.stimmung}, {r.ausstieg}")
            base = (r.markt, r.richtung, r.einstieg, "ohne", r.ausstieg)
            if r.stimmung != "ohne" and base in curves:
                ax.plot(curves[base].index, curves[base].values / 1000, ls=":", label="  gleiche Variante ohne Medienecho")
        if (kind, "bench") in curves:
            ax.plot(curves[(kind, "bench")].index, curves[(kind, "bench")].values / 1000, color="grey",
                    label=f"Kaufen und Halten {BENCHMARK[kind]}")
        ax.axvline(pd.Timestamp(TRAIN_END), color="black", lw=0.8, ls="--")
        ax.text(pd.Timestamp(TRAIN_END), ax.get_ylim()[1], "  Testzeitraum ->", va="top", fontsize=9)
        ax.set_title(f"{'Aktien' if kind == 'aktien' else 'ETFs'}: beste Variante des Lernzeitraums (Konto in Tsd. EUR)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "trend_model.png", dpi=110)
    plt.close(fig)


def report(res, best, effect, coverage, secs) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = ["markt", "richtung", "einstieg", "stimmung", "ausstieg", "lern_rendite_pa_pct", "lern_sharpe",
            "test_rendite_pa_pct", "test_sharpe", "test_max_drawdown_pct", "test_trades", "test_trefferquote"]
    cols = [c for c in cols if c in res.columns]
    print("\n" + "=" * 110)
    print(f"TRENDFOLGE XETRA  Lernzeitraum {START[:4]}-{TRAIN_END[:4]}, Test ab {int(TRAIN_END[:4]) + 1}, "
          f"Konto {CAPITAL:,.0f} EUR, max. {MAX_POSITIONS} Positionen".replace(",", "."))
    print(f"Kosten: {broker_text()}, Slippage {SLIPPAGE_BPS} bp je Order")
    print("=" * 110)
    print("\nBeste Variante je Markt und Richtung (ausgewaehlt nur nach dem Lernzeitraum):")
    print(best[cols].round(2).to_string(index=False))
    bench = res[res["einstieg"].str.startswith("Kaufen")]
    if not bench.empty:
        print("\nVergleich Kaufen und Halten:")
        print(bench[[c for c in cols if c in bench.columns]].round(2).to_string(index=False))
    r = res[~res["einstieg"].str.startswith("Kaufen")]
    print("\nMittlere Sharpe Ratio je Ausstiegsvariante (ueber alle Einstiege/Stimmungen, long):")
    print(r[r["richtung"] == "long"].groupby(["markt", "ausstieg"])[["lern_sharpe", "test_sharpe"]].mean()
          .round(2).to_string())
    print("\nMittlere Sharpe Ratio je Einstieg (long):")
    print(r[r["richtung"] == "long"].groupby(["markt", "einstieg"])[["lern_sharpe", "test_sharpe"]].mean()
          .round(2).to_string())
    if not effect.empty:
        print("\nWirkung des Medienechos (Sharpe Ratio gegenueber 'ohne', gleiche Einstiege und Ausstiege):")
        print(effect.round(3).to_string(index=False))
    if coverage is not None:
        print(f"\nMedienecho-Abdeckung: siehe medienecho_abdeckung.csv; Suchbegriffe in {QUERY_FILE.name} anpassbar.")
    print(f"\nAlle {len(r)} Varianten: varianten.csv   Dateien: {OUT_DIR}   (Laufzeit {secs / 60:.1f} min)")
    print("Hinweis: Bei so vielen Varianten ist die beste im Lernzeitraum oft Zufall - entscheidend ist der Test.")


# ======================================================================================= Today
def today(args) -> None:
    m = load_market("aktien")
    cal = m["close"].index
    t = len(cal) - 1
    direction = LIVE["richtung"]
    entry = LIVE["einstieg"]
    ev = entry_events(m["close"])[(direction, entry)]
    sig = ev.iloc[t] & m["elig"].iloc[t]
    cands = sig[sig].index.tolist()
    log(f"Stand {cal[t]:%d.%m.%Y}: {len(cands)} Aktien mit Signal '{entry}' ({direction})")
    pf = m["close"].ffill()
    mom = (pf.iloc[t] / pf.iloc[t - MOMENTUM_DAYS] - 1).reindex(cands).sort_values(ascending=direction == "short")
    cands = mom.index[:LIVE_MAX_CANDIDATES].tolist()
    rows = []
    if cands and not args.nur_kurse:
        queries = query_table(m)
        gd = load_gdelt(cands, queries, args.refresh, max_age_days=0.5)
        check_recent(gd, cal)
        tone_z, buzz = sentiment_frames(gd, cal, cands)
    for tk in cands:
        px = pf[tk].iloc[t]
        rule = EXITS[LIVE["ausstieg"]]
        row = {"ticker": tk, "name": m["names"].get(tk), "schluss": px, "momentum_6m_pct": mom[tk] * 100}
        if rule["typ"] == "preis":
            s = 1 if direction == "long" else -1
            row["stop_loss"] = px * (1 - s * rule["stop"])
            row["ziel"] = px * (1 + s * rule["ziel"]) if rule.get("ziel") else None
            row["nachzug_pct"] = rule.get("nachzug", 0) * 100
        if not args.nur_kurse:
            z = tone_z[tk].iloc[t] if tk in tone_z else np.nan
            b = buzz[tk].iloc[t] if tk in buzz else np.nan
            ok = bool(sentiment_mask(LIVE["stimmung"], direction, tone_z[[tk]], buzz[[tk]]).iloc[t, 0])
            verdict = "kaufen" if ok else ("nicht kaufen (zu wenig Artikel fuer eine Stimmungsaussage)"
                                           if np.isnan(z) else "nicht kaufen (Medienecho)")
            row.update({"stimmung_z": z, "echo": b, "entscheidung": verdict})
            q = queries.at[tk, "abfrage"] if tk in queries.index else ""
            for sort, label in [("tonedesc", "positiv"), ("toneasc", "negativ")]:
                js = gdelt_get({"query": q, "mode": "artlist", "maxrecords": LIVE_HEADLINES, "timespan": "7d",
                                "sort": sort}) if q else None
                arts = (js or {}).get("articles", []) or []
                row[f"ueberschriften_{label}"] = " | ".join(f"{a.get('title', '').strip()} ({a.get('domain', '')})"
                                                            for a in arts)
        else:
            row["entscheidung"] = "kaufen (ohne Medienecho)"
        rows.append(row)
    out = pd.DataFrame(rows)
    save_csv(out, "signale_heute.csv")
    print(f"\nSIGNALE {cal[t]:%d.%m.%Y}  Variante: {direction}, {entry}, {LIVE['stimmung']}, {LIVE['ausstieg']}")
    print("Ausfuehrung: zum Schlusskurs des naechsten Handelstages (wie in der Rueckrechnung).\n")
    for r in rows:
        print(f"{r['ticker']:9s} {str(r['name'])[:34]:34s} Schluss {r['schluss']:9.2f}  -> {r['entscheidung']}")
        if "stimmung_z" in r:
            print(f"          Stimmung z = {r['stimmung_z']:.2f}   Echo = {r['echo']:.2f}")
            for label in ["positiv", "negativ"]:
                if r.get(f"ueberschriften_{label}"):
                    print(f"          {label}: {r[f'ueberschriften_{label}'][:240]}")
        if r.get("stop_loss"):
            print(f"          Stop-Loss {r['stop_loss']:.2f}" + (f", Ziel {r['ziel']:.2f}" if r.get("ziel") else "")
                  + f", nachgezogener Stopp {r['nachzug_pct']:.0f} % unter dem Hoechstkurs")
    if not rows:
        print("Keine Signale.")
    print(f"\nDatei: {OUT_DIR / 'signale_heute.csv'}")


def main() -> None:
    global MAX_SENTIMENT_STOCKS, OUT_DIR
    ap = argparse.ArgumentParser(description="Trend-following model Xetra with media echo")
    ap.add_argument("--heute", action="store_true", help="today's signals with media echo and headlines")
    ap.add_argument("--nur-kurse", action="store_true", help="without media echo (no GDELT queries)")
    ap.add_argument("--refresh", action="store_true", help="reload media echo completely")
    ap.add_argument("--mit-pleiten", action="store_true", help="additionally calculate with bankruptcy cases and compare")
    ap.add_argument("--nur-medienecho", action="store_true",
                    help="load or add only the media echo (can be aborted, continues next time)")
    ap.add_argument("--max-aktien", type=int, default=None, help=f"media echo only for the N most liquid stocks "
                                                                   f"(default {MAX_SENTIMENT_STOCKS})")
    ap.add_argument("--nur-cache", action="store_true",
                    help="use media echo only from the existing cache, do not load anything from GDELT")
    ap.add_argument("--placebo", type=int, nargs="?", const=PRUEF_PLACEBO_LAEUFE, default=0, metavar="N",
                    help=f"additionally placebo test with N random filters (default {PRUEF_PLACEBO_LAEUFE}; takes time)")
    args = ap.parse_args()
    if args.max_aktien:
        MAX_SENTIMENT_STOCKS = args.max_aktien
    if args.nur_medienecho:
        m = load_market("aktien", extra=True)
        gd = load_gdelt(sentiment_tickers(m, pd.Timestamp(START)), query_table(m), args.refresh)
        check_recent(gd, m["close"].index)
        log(f"Medienecho im Cache: {len(gd)} Aktien ({GDELT_DIR})")
    elif args.heute:
        today(args)
    elif args.mit_pleiten:
        log("Rechnung 1 von 2: ohne Pleitefaelle (nur Aktien)")
        res0, _, _ = backtest(args, extra=False, kinds=("aktien",))
        OUT_DIR = BASE_DIR / "ergebnisse_mit_pleiten"
        log("Rechnung 2 von 2: mit Pleitefaellen")
        res1, trades1, pleiten = backtest(args, extra=True, kinds=("aktien",))
        compare_pleiten(res0, res1, trades1, pleiten)
    else:
        backtest(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C). Bereits geladenes Medienecho bleibt gespeichert.")
        sys.exit(1)
