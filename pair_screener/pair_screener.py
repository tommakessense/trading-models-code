"""
pair_screener.py - finds pairs with (potential) cointegration among Xetra stocks (default),
                   among Xetra ETFs/ETCs (--etf) or among US stocks / US ETFs (--us, --us --etf)

The screener is limited to finding pairs and outputs a ranking.
Trading rules, backtest, Sharpe ratio and other metrics are deliberately not yet included.

Usage
-----
    python pair_screener.py                       # stocks: all pairs within the same industry
    python pair_screener.py --pairs               # stocks: only the pairs from pairs_stocks.csv
    python pair_screener.py --two-stage           # two-stage (can also be combined with --pairs)
    python pair_screener.py --mean-reversion      # additional evaluation (can also be combined with --pairs / --etf)
    python pair_screener.py --etf [...]           # the same for ETFs/ETCs (pair list: pairs.csv)
    python pair_screener.py --us [...]            # the same for stocks listed in the USA
    python pair_screener.py --us --etf [...]      # the same for ETFs listed in the USA
    python pair_screener.py --pairs meine_liste.csv
    python pair_screener.py --demo                # trial run with synthetic data, without internet
    python pair_screener.py --refresh             # ignore the cache, reload everything

Procedure in stock mode
-----------------------
1. Universe:    all stocks tradable on Xetra (list of Deutsche Boerse, instrument type CS),
                without special lines such as shares tendered for sale. Can be restricted with STOCK_SEGMENTS to e.g.
                DAX / MDAX / SDAX.
2. Prices:      daily data (adjusted close) from Yahoo Finance, ticker "<Xetra symbol>.DE".
3. Master data: market capitalization, sector and industry from Yahoo (output_stocks/stock_details.csv).
                Size filter: market capitalization > MIN_MARKET_CAP_USD.
4. Period:      the last FORMATION_YEARS (3) years up to the last price date.
5. Cleaning:    liquidity filter (median turnover on Xetra >= STOCK_MIN_MEDIAN_TURNOVER_EUR, hardly any days
                without a price change). The data error detection is off for stocks (STOCK_DETECT_PRICE_ERRORS).
6. Pairing:     only pairs within the same industry (STOCK_GROUP_BY = "industry"; "sector" = coarser,
                "" = all combinations), correlation of daily returns >= STOCK_MIN_RET_CORR, spread width
                >= MIN_SPREAD_STD. There is no 'same index' filter for stocks: common and
                preferred shares of the same company are exactly what is wanted here.
7. Test:        Engle-Granger in both directions, Benjamini-Hochberg correction over all tests.

Procedure in ETF mode (--etf)
-----------------------------
The universe is all Xetra ETFs and ETCs without leveraged / inverse products; the size filter is the
fund volume (MIN_AUM_USD, ETCs excepted). Instead of by industry, pre-filtering is done via the
correlation (MIN_RET_CORR). In addition, pairs with (almost) the same index are removed: same
reference index (Yahoo / index_map.csv), tracking error relative to volatility below MIN_REL_TE or
identical fund volume (= share classes of the same fund) -> same_index_excluded.csv. The pairs with
almost the same index but a wide spread are tested separately -> ranking_near_same.csv.

US market (--us, can be combined with all other switches)
---------------------------------------------------------
Universe:    all stocks or ETFs listed on Nasdaq, NYSE, NYSE American, NYSE Arca and Cboe
             (symbol lists from nasdaqtrader.com). Without warrants, subscription rights, preferred shares,
             units, SPACs, ETNs and test symbols; for ETFs without leveraged / inverse products.
Preselection: Because these are around 6,000 stocks or 4,000 ETFs, first 3 months of prices are loaded
             and only securities with enough turnover are kept (US_PRESCREEN_FACTOR x minimum turnover). Only for
             these is the long history loaded (cache/preselection_us_*.csv).
Thresholds:  median daily turnover >= US_STOCK_MIN_MEDIAN_TURNOVER_USD (stocks) or
             US_ETF_MIN_MEDIAN_TURNOVER_USD (ETFs); market capitalization / fund volume as above.
             Mean reversion calculates with US_MR_COST_BPS per purchase / sale.
Otherwise the same: industry groups for stocks, 'same index' filter for ETFs, tests, outputs.
Pair lists:  pairs_us_stocks.csv / pairs_us_etf.csv, tickers in Yahoo notation (BRK-B).
Note:        Retail investors in the EU can usually not buy US ETFs (missing key
             information document, PRIIPs); they are mostly tradable only via options / CFDs
             or as a professional client. US stocks are not affected by this.

Two-stage mode (--two-stage)
----------------------------
Stage 1 searches in the earlier period (STAGE1_YEARS years before the current period) and selects the
best candidates (at most STAGE1_TOP_N, p <= STAGE1_MAX_P). Stage 2 tests only these candidates
in the last FORMATION_YEARS years. Because the data of stage 2 were not used for the selection and
only a few tests run, the hurdle is lower and the result more reliable.

Additional evaluation mean reversion (--mean-reversion)
-------------------------------------------------------
Instead of strict cointegration, tests the weaker question: Does the spread of a pair return to its
moving average of the last months, and is anything left of this after trading costs?
Spread = log(A) - log(B) (same amount long and short). Entry at |z| > MR_ENTRY_Z relative to the
MR_LOOKBACK-day mean, exit at the mean or after MR_MAX_HOLD days. Costs MR_COST_BPS per purchase /
sale, borrowing fee MR_BORROW_FEE. Rolling: For each test year, the MR_TOP_N pairs with the best
result of the MR_FORMATION_YEARS previous years are selected and traded in the following year (new data).
Output: mr_years.csv (per test year: all pairs / selection), mr_selection.csv (selected pairs per year),
mr_pairs.csv (metrics per pair), mr_portfolio_returns.csv (daily returns).
Development of the best pairs: The MR_SHOW_TOP pairs with the highest Sharpe ratio over the whole period
are shown year by year (return and Sharpe per test year, last 6 months, Sharpe 'earlier' against
'recent' = last MR_TREND_YEARS years) -> mr_top_pairs.csv, mr_top_pairs_history.csv (cumulative
return per day) and mr_top_pairs.png. Caution: These pairs are selected in hindsight. That they look
good is therefore no proof; what is meaningful is only whether they have held up recently or are weakening.
Price breaks: daily moves above the factor MR_MAX_DAILY_MOVE are considered a break in the data (e.g. new
shares after an insolvency); the history before is not used -> mr_price_breaks.csv.
Cross-check (mr_top_pairs_cross_check.csv): the best pairs of only the earlier years and their result in
the last MR_TREND_YEARS years, which are new data for this selection.
Limits: only securities still listed today, today's market capitalization / industry, closing prices instead of
bid and ask prices. The result is therefore rather too favorable than too strict.

Results (CSV with ";" and decimal comma for Excel)
--------------------------------------------------
    Folder: ./output_stocks (Xetra stocks), ./output (Xetra ETFs), ./output_us_stocks, ./output_us_etf
    ranking.csv / ranking_pair_list.csv   all tested pairs, sorted by p-value
    cointegrated_pairs.csv                only the pairs significant after correction
    ranking_stage1.csv, ranking_two_stage.csv            two-stage run
    stock_details.csv / etf_details.csv  master data per instrument
    data_errors.csv, narrow_spread_excluded.csv, universe_used.csv, missing_mktcap.csv / missing_aum.csv
    ETFs only: same_index_excluded.csv, ranking_near_same*.csv

Own files next to the script (optional)
---------------------------------------
    pairs_stocks.csv / pairs.csv   ticker_a;ticker_b;category;rationale   (pair list for --pairs)
    pairs_us_stocks.csv / pairs_us_etf.csv   the same for --us
    universe_stocks.csv / universe.csv   ticker[;name]      (own universe instead of the Xetra list)
    groups_stocks.csv             ticker;peer_group            (assign industry manually, stocks only)
    aum_overrides.csv              ticker;aum_usd           (correct fund volume, ETFs only)
    index_map.csv                  ticker;index             (assign reference index, ETFs only)

    pip install yfinance statsmodels joblib pandas numpy requests
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

# ================================ Configuration ================================
HISTORY_YEARS = 10          # prices are loaded this far back (cache for later tests);
                            # only the last FORMATION_YEARS are used for the search
FORMATION_YEARS = 3         # period in which the cointegration is tested

# ---- Stock mode (default). The ETF/ETC mode runs with --etf and uses the settings below.
MIN_MARKET_CAP_USD = 500e6  # minimum market capitalization in USD
STOCK_MIN_MEDIAN_TURNOVER_EUR = 250_000   # median daily turnover on Xetra (liquidity, short selling)
STOCK_MIN_RET_CORR = 0.30   # pre-filter correlation of daily returns (stocks correlate less strongly than ETFs)
STOCK_GROUP_BY = "industry" # only test pairs within the same group: "industry" (industry, fine),
                            # "sector" (sector, coarse) or "" (all combinations)
STOCK_SEGMENTS = ()         # empty = all Xetra stocks; e.g. ("DAX", "MDAX", "SDAX", "DEUTSCHLAND")
STOCK_DETECT_PRICE_ERRORS = False  # data error detection (price jump relative to similar securities that
                            # jumps back). Off for stocks: jumps after quarterly figures are real there.

# ---- ETF/ETC mode (--etf)
INCLUDE_ETC = True          # include ETCs (gold, silver, oil, industrial metals ...) in the universe
MIN_AUM_USD = 100e6         # minimum fund volume in USD
AUM_FILTER_ETC = False      # apply the AUM filter to ETCs as well? (Yahoo rarely knows their volume)
MAX_STALE_SHARE = 0.05      # max. share of days without trading / without a price change on Xetra
MIN_MEDIAN_TURNOVER_EUR = 100_000   # median daily turnover on Xetra; less frequently traded ETFs have
                            # stale closing prices -> artificial 'mean reversion' in the spread
KEEP_UNKNOWN_TURNOVER = True  # keep ETFs for which Yahoo provides no turnover
KEEP_MISSING_AUM = False    # keep ETFs without AUM data? False = exclude conservatively
MIN_COVERAGE = 0.95         # minimum share of trading days with a price in the formation window

MIN_RET_CORR = 0.70         # pre-filter: correlation of the daily returns in the formation
MAX_RET_CORR = 0.995        # above this = practically the same index
MIN_REL_TE = 0.20           # tracking error / volatility (monthly returns). Below this, A and B track
                            # (almost) the same index -> the pair is excluded. Relative, so that
                            # low-volatility bond ETFs are not wrongly considered 'the same'.
SAME_AUM_IS_SAME_FUND = True  # exactly the same fund volume at Yahoo = share classes of the same fund
MIN_SPREAD_STD = 0.01       # minimum width of the spread: std of log(A) - beta*log(B) (0.01 = approx. 1 %).
                            # Narrower pairs are not tested: almost identical exposures,
                            # whose 'mean reversion' is mainly price noise. 0 = filter off.
                            # Applies only to 'all combinations', not to the own pair list (--pairs).
NEAR_SAME_MIN_SPREAD = 0.01 # Evaluation 'almost identical indices': pairs that the tracking error filter sorts out as
                            # 'same index', but whose spread is at least this wide
                            # (e.g. two gold miner or semiconductor indices). Separate ranking.
JUMP_THRESHOLD = 0.08       # data error: daily return deviates by more than 8 % from similar ETFs
JUMP_REVERT_DAYS = 30       # ... and jumps back in the opposite direction within 30 trading days
PEER_MIN_CORR = 0.70        # 'similar ETFs' = the 5 most strongly correlated with correlation >= 0.7
MAX_PAIRS_TO_TEST = 55_000  # upper limit of tested pairs (highest correlation first)
COINT_MAXLAG = 5            # max. lags in the ADF part of the Engle-Granger test

FDR_ALPHA = 0.05            # Benjamini-Hochberg level (multiple testing correction)
RANK_TOP_FOR_COUNT = 100    # frequency per ETF is counted among the best N pairs of the ranking

# Two-stage mode (--two-stage): stage 1 selects candidates in the earlier period,
# stage 2 tests only these candidates in the later period (= last FORMATION_YEARS years).
STAGE1_YEARS = 3            # length of the earlier period (directly before the period of stage 2)
STAGE1_TOP_N = 20           # at most this many candidates go into stage 2
STAGE1_MAX_P = 0.05         # candidate only if the uncorrected p-value in stage 1 is below this
STAGE1_MAX_PER_ETF = 3      # an ETF may appear in at most this many candidate pairs

# ---- US market (--us): stocks listed in the USA or, with --etf, ETFs, prices and turnover in USD
US_STOCK_MIN_MEDIAN_TURNOVER_USD = 5_000_000   # median daily turnover per stock in USD
US_ETF_MIN_MEDIAN_TURNOVER_USD = 1_000_000     # median daily turnover per ETF in USD
US_PRESCREEN_FACTOR = 0.5   # preselection: load the long history only for securities whose turnover in the last
                            # 3 months corresponds to at least this share of the turnover threshold
US_MR_COST_BPS = 5          # costs per purchase or sale in basis points (US securities have tighter spreads)

# ---- Additional evaluation --mean-reversion: return to the moving average, with costs, on new data
MR_LOOKBACK = 60            # trading days for the moving average and standard deviation of the spread
MR_ENTRY_Z = 2.0            # entry when the spread is more than 2 std away from the moving average
MR_EXIT_Z = 0.0             # exit when it reaches the mean again (0) or comes within EXIT_Z of it
MR_MAX_HOLD = 40            # close out after this many trading days at the latest (time stop)
MR_COST_BPS = 10            # costs per purchase or sale in basis points of the traded amount
MR_BORROW_FEE = 0.01        # borrowing fee p. a. on the short side
MR_FORMATION_YEARS = 3      # selection of the pairs based on their results in these previous years ...
MR_TEST_YEARS = 1           # ... tested in the respective following year (rolling)
MR_TOP_N = 20               # this many pairs are selected per test year
MR_MAX_PER_STOCK = 2        # a security may appear in at most this many selected pairs
MR_MIN_TRADES = 5           # minimum number of closed trades in the previous years
MR_MAX_PAIRS = 5_000        # upper limit of pairs per year (ETF mode: highest correlation first)
MR_MAX_DAILY_MOVE = 3.0     # daily move above factor 3 (more than +200 % or -67 %) = break in the price data
                            # (e.g. new shares after insolvency): the history before is not used
MR_SHOW_TOP = 10            # show this many of the best pairs (by Sharpe over the whole period) year by year
MR_TREND_YEARS = 2          # 'recent' = the last this many test years, compared with the years before
MR_TREND_MIN_DIFF = 0.5     # from this Sharpe difference on, the development counts as increasing or decreasing

N_JOBS = -1                 # CPU cores for the cointegration test (-1 = all)
# ===============================================================================

MODE = "stocks"             # is set in main(): "stocks" (default) or "etf" (--etf)
MARKET = "xetra"            # is set in main(): "xetra" (default) or "us" (--us)

US_LISTS = {"nasdaqlisted.txt": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "otherlisted.txt": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"}
# US stocks: no warrants, subscription rights, preferred and debt securities, units, empty shell companies
US_STOCK_EXCLUDE = re.compile(r"Warrant|\bRights?\b|Preferred|\bNotes?\b|Debenture|\bUnits?\b(?!.*Limited Partner)"
                              r"|Acquisition Corp|%|Depositary Shares?.*(?:Preferred|Interest in)|\bETN\b",
                              re.IGNORECASE)
US_LEVERAGE_PATTERN = re.compile(r"ULTRAPRO|ULTRASHORT|\bULTRA\s+(?!SHORT)|LEVERAG|\b\d(?:\.\d+)?X\b|-\dX\b"
                                 r"|INVERSE|\bBEAR\b|\bBULL\b|PROSHARES\s+SHORT|DAILY.*\bSHORT\b",
                                 re.IGNORECASE)


def cache_file(kind: str) -> Path:
    """Cache file per market and mode (Xetra keeps the previous names)."""
    us = MARKET == "us"
    names = {"prices": {("xetra", "stocks"): "prices_stocks.pkl", ("xetra", "etf"): "prices.pkl",
                        ("us", "stocks"): "prices_us_stocks.pkl", ("us", "etf"): "prices_us_etf.pkl"}[(MARKET, MODE)],
             "stock_info": "info_us_stocks.csv" if us else "info_stocks.csv",
             "aum": "aum_us.csv" if us else "aum.csv",
             "prescreen": f"preselection_us_{MODE}.csv"}
    return CACHE_DIR / names[kind]


def min_turnover() -> float:
    if MARKET == "us":
        return US_STOCK_MIN_MEDIAN_TURNOVER_USD if MODE == "stocks" else US_ETF_MIN_MEDIAN_TURNOVER_USD
    return STOCK_MIN_MEDIAN_TURNOVER_EUR if MODE == "stocks" else MIN_MEDIAN_TURNOVER_EUR

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUT_DIR = BASE_DIR / "output"

XETRA_PAGE = "https://www.cashmarket.deutsche-boerse.com/cash-en/trading/Tradable-Instruments-Xetra"
XETRA_CSV_FALLBACK = ("https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/"
                      "684b31b077a5de5d5777352984c7a7df/data/t7-xetr-allTradableInstruments.csv")
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) pair_screener"}

# leveraged / inverse products. "SHORT" only if not "Short Duration/Maturity/Term" or similar.
# stocks: shares tendered for sale, subscription rights and the like are not separate companies
STOCK_EXCLUDE_PATTERN = re.compile(r"VERK\.|Z\.\s*VERK|BEZUGSR|\bBZR\b|JUNGE|UMTAUSCH", re.IGNORECASE)

LEVERAGE_PATTERN = re.compile(
    r"(\bLEV\b|LEVERAG|\b[2-5]X\b|-[1-5]X\b|INVERSE|\bBEAR\b|"
    r"\bSHORT\b(?!\s*[-.]?\s*(DUR|MAT|TERM|TREA|DAT|CORP|GOV|BOND|BD)))",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """CSV for German Excel: semicolon, decimal comma, UTF-8 with BOM."""
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    if MODE == "stocks":                                    # internally the columns are called etf_a / etf_b
        df = df.rename(columns={"etf_a": "stock_a", "etf_b": "stock_b"})
    if MARKET == "us":                                      # turnover there is in thousands of USD
        df = df.rename(columns={"turnover_a_teur": "turnover_a_tusd", "turnover_b_teur": "turnover_b_tusd"})
    kw = dict(sep=";", decimal=",", index=index, encoding="utf-8-sig", float_format="%.6g")
    try:
        df.to_csv(path, **kw)
    except PermissionError:                          # e.g. the file is currently open in Excel
        path = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        df.to_csv(path, **kw)
        log(f"  {name} is locked (open in Excel?) - saved as {path.name}")
    return path


def read_user_csv(path: Path, **kw) -> pd.DataFrame:
    """Read own CSV files: separator ; or , automatically, UTF-8 or Windows encoding (Excel)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", **kw)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_results(name: str) -> pd.DataFrame:
    """Helper function for the notebook: load_results('current_candidates.csv')."""
    return pd.read_csv(OUT_DIR / name, sep=";", decimal=",", encoding="utf-8-sig")


# ------------------------------------------------------------------ Universe
def download_xetra_list(path: Path) -> None:
    import requests

    try:
        html = requests.get(XETRA_PAGE, headers=HTTP_HEADERS, timeout=30).text
        m = re.search(r'href="([^"]*t7-xetr-allTradableInstruments\.csv)"', html)
        url = urljoin(XETRA_PAGE, m.group(1)) if m else XETRA_CSV_FALLBACK
    except requests.RequestException:
        url = XETRA_CSV_FALLBACK
    log(f"Loading Xetra instrument list: {url}")
    r = requests.get(url, headers=HTTP_HEADERS, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)


def parse_xetra_list(path: Path, mode: str = "etf") -> pd.DataFrame:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.splitlines()
    hdr_idx = next(i for i, line in enumerate(lines) if line.startswith("Product Status"))
    rows = list(csv.reader(lines[hdr_idx:], delimiter=";"))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if r]
    df = pd.DataFrame(body, columns=header)

    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> str:
        if name.lower() not in cols:
            raise KeyError(f"Column '{name}' missing in the Xetra list. Available: {list(df.columns)}")
        return cols[name.lower()]

    grp = df[col("Product Assignment Group Description")].str.upper()
    if mode == "stocks":
        st = df[df[col("Instrument Type")].str.strip().str.upper() == "CS"].copy()
        if "instrument status" in cols:
            st = st[st[col("Instrument Status")].str.strip().str.lower() == "active"]
        st["sec_type"] = "Stock"
        st["segment"] = grp.loc[st.index].str.strip()
        st = st.rename(columns={col("Mnemonic"): "mnemonic", col("ISIN"): "isin", col("Instrument"): "name"})
        st = st[["mnemonic", "isin", "name", "sec_type", "segment"]].apply(lambda x: x.str.strip())
        st = st[st["mnemonic"] != ""].drop_duplicates("isin").drop_duplicates("mnemonic")
        st["ticker"] = st["mnemonic"] + ".DE"
        return st.reset_index(drop=True)
    is_etf = grp.str.contains("EXCHANGE TRADED FUNDS", na=False)
    is_etc = grp.str.contains("EXCHANGE TRADED COMMODIT", na=False)
    etf = df[is_etf | (is_etc & INCLUDE_ETC)].copy()
    etf["sec_type"] = np.where(is_etc.loc[etf.index], "ETC", "ETF")
    if "instrument status" in cols:
        etf = etf[etf[col("Instrument Status")].str.strip().str.lower() == "active"]
    etf = etf.rename(columns={col("Mnemonic"): "mnemonic", col("ISIN"): "isin", col("Instrument"): "name"})
    etf = etf[["mnemonic", "isin", "name", "sec_type"]].apply(lambda s: s.str.strip())
    etf = etf[etf["mnemonic"] != ""].drop_duplicates("isin").drop_duplicates("mnemonic")
    etf["ticker"] = etf["mnemonic"] + ".DE"
    return etf.reset_index(drop=True)


def load_us_universe(refresh: bool) -> pd.DataFrame:
    """All stocks or ETFs listed on Nasdaq, NYSE and the other US exchanges (lists from Nasdaq Trader)."""
    import requests

    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for fname, url in US_LISTS.items():
        path = CACHE_DIR / f"us_{fname}"
        if refresh or not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
            log(f"Loading US securities list: {url}")
            r = requests.get(url, headers=HTTP_HEADERS, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip() and not ln.startswith("File Creation Time")]
        rows = [ln.split("|") for ln in lines]
        df = pd.DataFrame([(r + [""] * len(rows[0]))[:len(rows[0])] for r in rows[1:]],
                          columns=[c.strip() for c in rows[0]])
        sym = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        d = pd.DataFrame({"symbol": df[sym].str.strip(), "name": df["Security Name"].str.strip(),
                          "etf": df["ETF"].str.strip().str.upper() == "Y",
                          "test": df["Test Issue"].str.strip().str.upper() == "Y",
                          "segment": df["Exchange"].str.strip() if "Exchange" in df.columns else "NASDAQ"})
        if "Financial Status" in df.columns:                # Nasdaq: only regularly listed securities
            d = d[df["Financial Status"].str.strip().isin(["N", ""]).to_numpy()]
        frames.append(d)
    u = pd.concat(frames, ignore_index=True)
    u = u[~u["test"] & u["symbol"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]     # no $, +, = special lines
    u["ticker"] = u["symbol"].str.replace(".", "-", regex=False)               # Yahoo: BRK.B -> BRK-B
    u = u.drop_duplicates("ticker")
    n_all = len(u)
    if MODE == "stocks":
        u = u[~u["etf"]]
        odd = u["name"].str.contains(US_STOCK_EXCLUDE)
        u = u[~odd].assign(sec_type="Stock")
        log(f"USA: {n_all} listed securities, of which {len(u)} stocks (without ETFs, warrants, preferred securities, "
            f"shell companies)")
    else:
        u = u[u["etf"]].assign(sec_type="ETF")
        lev = u["name"].str.contains(US_LEVERAGE_PATTERN)
        OUT_DIR.mkdir(exist_ok=True)
        save_csv(u[lev][["ticker", "name"]], "excluded_leveraged_inverse.csv")
        u = u[~lev]
        log(f"USA: {n_all} listed securities, of which {len(u) + int(lev.sum())} ETFs, {int(lev.sum())} leveraged/inverse "
            f"excluded, {len(u)} remain")
    u["isin"] = ""
    return u[["ticker", "name", "isin", "sec_type", "segment"]].reset_index(drop=True)


def us_prescreen(tickers: list[str], refresh: bool) -> list[str]:
    """Preselection by turnover of the last 3 months, so that 10 years are not loaded for more than 10,000 securities."""
    cache = cache_file("prescreen")
    known = pd.DataFrame(columns=["ticker", "turnover_usd"])
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 30 * 86400:
        known = pd.read_csv(cache, keep_default_na=False, na_values=[""])   # ticker "NA" is not a missing value
    have = set(known["ticker"])
    todo = [t for t in tickers if t not in have]
    if todo:
        log(f"Preselection: loading 3 months of prices for {len(todo)} US securities ...")
        rows = []
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            c, v = _download_chunk(chunk, None, period="3mo")
            med = (c * v.reindex_like(c)).median() if not c.empty else pd.Series(dtype=float)
            rows += [{"ticker": t, "turnover_usd": float(med.get(t, np.nan))} for t in chunk]
            log(f"  {min(i + 200, len(todo))}/{len(todo)}")
            time.sleep(1)
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)
    k = known.drop_duplicates("ticker", keep="last").set_index("ticker")["turnover_usd"].reindex(tickers)
    keep = k[k >= US_PRESCREEN_FACTOR * min_turnover()].index.tolist()
    log(f"Preselection: {len(keep)} of {len(tickers)} securities with median turnover >= "
        f"{US_PRESCREEN_FACTOR * min_turnover() / 1e6:.1f} million USD in the last 3 months")
    return keep


def load_universe(refresh: bool) -> pd.DataFrame:
    stocks = MODE == "stocks"
    us = MARKET == "us"
    override = BASE_DIR / (("universe_us_stocks.csv" if stocks else "universe_us_etf.csv") if us
                           else ("universe_stocks.csv" if stocks else "universe.csv"))
    if override.exists():
        u = read_user_csv(override, dtype=str)
        for c in ("name", "isin", "segment"):
            if c not in u.columns:
                u[c] = ""
        if "sec_type" not in u.columns:
            u["sec_type"] = "Stock" if stocks else "ETF"
        u["ticker"] = u["ticker"].str.strip().str.upper()
        if not us:
            u["ticker"] = np.where(u["ticker"].str.contains(r"\."), u["ticker"], u["ticker"] + ".DE")
        log(f"Own universe from {override.name}: {len(u)} instruments")
        return u[["ticker", "name", "isin", "sec_type", "segment"]]
    if us:
        return load_us_universe(refresh)

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "xetra_allTradableInstruments.csv"
    if refresh or not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
        download_xetra_list(path)

    if stocks:
        st = parse_xetra_list(path, "stocks")
        n_all = len(st)
        odd = st["name"].str.contains(STOCK_EXCLUDE_PATTERN)
        st = st[~odd]
        if STOCK_SEGMENTS:
            st = st[st["segment"].isin([x.upper() for x in STOCK_SEGMENTS])]
        log(f"Xetra: {n_all} stocks, {int(odd.sum())} special lines (tendered shares etc.) excluded, "
            f"{len(st)} remain" + (f" (segments: {', '.join(STOCK_SEGMENTS)})" if STOCK_SEGMENTS else ""))
        return st[["ticker", "name", "isin", "sec_type", "segment"]].reset_index(drop=True)

    etf = parse_xetra_list(path, "etf")
    etf["segment"] = ""
    lev = etf["name"].str.contains(LEVERAGE_PATTERN)
    save_csv(etf[lev], "excluded_leveraged_inverse.csv")
    etf = etf[~lev]
    log(f"Xetra: {len(etf) + lev.sum()} ETFs/ETCs, {lev.sum()} leveraged/inverse excluded, "
        f"{len(etf)} remain (of which {int((etf['sec_type'] == 'ETC').sum())} ETCs)")
    return etf[["ticker", "name", "isin", "sec_type", "segment"]].reset_index(drop=True)


# ------------------------------------------------------------------ Prices
def _download_chunk(chunk: list[str], start: str | None, period: str | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    span = {"period": period} if period else {"start": start}
    for attempt in range(4):
        try:
            data = yf.download(chunk, auto_adjust=True, progress=False,
                               threads=True, group_by="column", **span)
            break
        except Exception as e:  # rate limit or similar
            wait = 15 * (attempt + 1)
            log(f"  Download error ({type(e).__name__}), new attempt in {wait}s")
            time.sleep(wait)
    else:
        return pd.DataFrame(), pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"], data["Volume"]
    return data[["Close"]].set_axis(chunk[:1], axis=1), data[["Volume"]].set_axis(chunk[:1], axis=1)


def load_prices(tickers: list[str], refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache = cache_file("prices")
    start = (pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS, days=10)).strftime("%Y-%m-%d")

    old = None
    to_load = tickers
    if cache.exists() and not refresh:
        c = pd.read_pickle(cache)
        age_days = (pd.Timestamp.today() - c["fetched"]).days
        if age_days <= 3:
            todo = [t for t in tickers if t not in c["tried"]]
            if not todo:
                log(f"Prices from cache (as of {c['fetched']:%d.%m.%Y})")
                return c["close"], c["volume"]
            old, to_load = c, todo                     # only load new instruments
            log(f"Prices from cache (as of {c['fetched']:%d.%m.%Y}), {len(todo)} new instruments")
        else:
            log(f"Price cache is {age_days} days old, reloading")

    log(f"Loading prices for {len(to_load)} instruments from {start} from Yahoo Finance ...")
    closes, vols = [], []
    chunk_size = 100
    for i in range(0, len(to_load), chunk_size):
        chunk = to_load[i:i + chunk_size]
        c, v = _download_chunk(chunk, start)
        closes.append(c)
        vols.append(v)
        log(f"  {min(i + chunk_size, len(to_load))}/{len(to_load)}")
        time.sleep(1)
    if old is not None:
        closes.insert(0, old["close"])
        vols.insert(0, old["volume"])
    closes = [c for c in closes if not c.empty]
    vols = [v for v in vols if not v.empty]
    for frame in closes + vols:
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    close = pd.concat(closes, axis=1)
    volume = pd.concat(vols, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    close = close.loc[:, close.notna().sum() > 0]
    volume = volume.reindex(columns=close.columns)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = close.index
    tried = set(tickers) | (old["tried"] if old is not None else set())
    fetched = old["fetched"] if old is not None else pd.Timestamp.today()
    pd.to_pickle({"close": close, "volume": volume, "tried": tried, "fetched": fetched}, cache)
    log(f"Prices available for {close.columns.isin(tickers).sum()} of {len(tickers)} instruments")
    return close, volume


# ------------------------------------------------------------------ Fund volume (AUM)
FUND_CCY_CANDIDATES = ["EUR", "USD", "GBP", "CHF", "JPY", "SEK", "NOK", "DKK", "PLN",
                       "AUD", "CAD", "HKD", "CNY", "INR", "KRW"]
AUM_CACHE_COLUMNS = ["ticker", "total_assets", "currency", "nav", "price", "benchmark"]

# reference index from Yahoo's fund description ("... aims to track the performance of the X Index")
BENCHMARK_RE = re.compile(
    r"(?:track|replicat|reflect|mirror|return of|performance of)\w*\s+(?:the\s+)?"
    r"(?:(?:total\s+)?(?:return|performance)\s+of\s+)?(?:the\s+)?([^.;:()]{3,90}?\bIndex\b)",
    re.IGNORECASE)
INDEX_PROVIDERS = ("MSCI", "FTSE", "SP", "STOXX", "NASDAQ", "DAX", "SOLACTIVE", "BLOOMBERG", "ICE",
                   "IBOXX", "MARKIT", "RUSSELL", "DOWJONES", "DJ", "JPMORGAN", "JPM", "NIKKEI", "TOPIX",
                   "HANGSENG", "CAC", "IBEX", "FTSEMIB", "MORNINGSTAR", "INDXX", "SMI", "AEX", "WIG",
                   "BBG", "BARCLAYS", "CITI", "EUROSTOXX", "MDAX", "SDAX", "TECDAX", "NYSE", "CRSP")
_BENCH_DROP = re.compile(r"\b(THE|INDEX|NET|TOTAL|RETURN|NOTIONAL|GROSS|PRICE|TR|NTR|NR|GR|USD|EUR|"
                         r"GBP|CHF|JPY|IN|DOLLAR|EURO|DAILY|ACCUMULATING|DISTRIBUTING)\b")


def normalize_benchmark(text) -> str | None:
    """'Nasdaq-100 Notional Net Total Return Index' -> 'NASDAQ100'. None if no known provider."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.upper().replace("&", "").replace("S P", "SP")
    t = re.sub(r"[®™*]", "", t)
    t = _BENCH_DROP.sub(" ", re.sub(r"[^A-Z0-9 ]", " ", t))
    key = re.sub(r"\s+", "", t)
    if len(key) < 3 or not any(key.startswith(p) or p in key for p in INDEX_PROVIDERS):
        return None
    return key


def extract_benchmark(summary) -> str | None:
    if not isinstance(summary, str):
        return None
    m = BENCHMARK_RE.search(summary)
    return m.group(1).strip() if m else None


def _fetch_info(ticker: str) -> dict:
    import yfinance as yf

    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
            price = (info.get("previousClose") or info.get("regularMarketPreviousClose")
                     or info.get("regularMarketPrice"))
            return {"ticker": ticker, "total_assets": info.get("totalAssets"),
                    "currency": info.get("currency"), "nav": info.get("navPrice"), "price": price,
                    "benchmark": extract_benchmark(info.get("longBusinessSummary"))}
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {"ticker": ticker, "total_assets": None, "currency": None, "nav": None, "price": None,
            "benchmark": None}


def _usd_per_unit(currencies: list[str]) -> dict[str, float]:
    """How many USD one unit of the respective currency is worth (current)."""
    import yfinance as yf

    rates = {"USD": 1.0}
    todo = [c for c in dict.fromkeys(currencies) if c and c != "USD"]
    if not todo:
        return rates
    try:
        px = yf.download([f"{c}USD=X" for c in todo], period="10d", progress=False,
                         auto_adjust=True)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(f"{todo[0]}USD=X")
        for c in todo:
            col = f"{c}USD=X"
            if col in px.columns and px[col].notna().any():
                rates[c] = float(px[col].dropna().iloc[-1])
    except Exception as e:
        log(f"  Exchange rates could not be loaded ({type(e).__name__})")
    missing = [c for c in todo if c not in rates]
    if missing:
        log(f"  No exchange rate for: {', '.join(missing)}")
    return rates


def infer_fund_currency(nav: float, price: float, trade_ccy: str,
                        usd_per: dict[str, float]) -> tuple[str, str]:
    """Currency in which Yahoo states the NAV and fund volume, from the ratio NAV / exchange price.

    Example: price 5.10 EUR, NAV 870 -> ratio ~170 ~ EUR/JPY -> fund currency JPY.
    The exchange price is always in the trading currency (for .DE: EUR).
    """
    if trade_ccy not in usd_per:
        return trade_ccy, "uncertain: no exchange rate"
    if not (pd.notna(nav) and pd.notna(price) and nav > 0 and price > 0):
        return trade_ccy, "assumed trading currency (no NAV)"
    ratio = nav / price
    best, err = trade_ccy, np.inf
    for c, u in usd_per.items():
        e = abs(np.log(ratio / (usd_per[trade_ccy] / u)))
        if e < err:
            best, err = c, e
    if err < np.log(1.3):
        return best, "derived from NAV/price"
    return trade_ccy, "uncertain: NAV matches no currency"


def load_aum_usd(tickers: list[str], refresh: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Fund volume in USD per ticker (NaN = unknown) and details on the currency determination."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = cache_file("aum")
    known = pd.DataFrame(columns=AUM_CACHE_COLUMNS)
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 30 * 86400:
        cached = pd.read_csv(cache, keep_default_na=False, na_values=[""])   # ticker "NA" is not a missing value
        if set(AUM_CACHE_COLUMNS) <= set(cached.columns):     # old cache without NAV -> reload
            known = cached
    todo = [t for t in tickers if t not in set(known["ticker"])]
    if todo:
        log(f"Loading fund volume for {len(todo)} ETFs (Yahoo 'totalAssets' + NAV) ...")
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_fetch_info, t) for t in todo]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % 100 == 0:
                    log(f"  {k}/{len(todo)}")
                    pd.concat([known, pd.DataFrame(rows)]).to_csv(cache, index=False)
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)

    known = known.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(tickers)
    usd_per = _usd_per_unit(FUND_CCY_CANDIDATES + known["currency"].dropna().tolist())

    ccy, quality = [], []
    for _, r in known.iterrows():
        c, q = infer_fund_currency(pd.to_numeric(r["nav"], errors="coerce"),
                                   pd.to_numeric(r["price"], errors="coerce"),
                                   r["currency"] if isinstance(r["currency"], str)
                                   else ("USD" if MARKET == "us" else "EUR"), usd_per)
        ccy.append(c)
        quality.append(q)
    meta = pd.DataFrame({"total_assets": pd.to_numeric(known["total_assets"], errors="coerce"),
                         "fund_ccy": ccy, "aum_source": quality}, index=known.index)
    meta["aum_usd"] = meta["total_assets"] * meta["fund_ccy"].map(usd_per).astype(float)
    meta.loc[meta["total_assets"].isna(), "aum_source"] = "missing"

    overrides = BASE_DIR / "aum_overrides.csv"
    if overrides.exists():
        o = read_user_csv(overrides)
        o = o.set_index("ticker")["aum_usd"].astype(float)
        o = o[o.index.isin(meta.index)]
        meta.loc[o.index, "aum_usd"] = o
        meta.loc[o.index, "aum_source"] = "manual (aum_overrides.csv)"
        log(f"{len(o)} AUM values taken from aum_overrides.csv")

    # reference index: index_map.csv (manual) > Yahoo fund description
    meta["benchmark"] = known["benchmark"]
    meta["index_key"] = known["benchmark"].map(normalize_benchmark)
    meta["index_source"] = np.where(meta["index_key"].notna(), "Yahoo description", "unknown")
    index_map = BASE_DIR / "index_map.csv"
    if index_map.exists():
        im = read_user_csv(index_map, dtype=str)
        im = im.set_index("ticker")["index"].dropna()
        im = im[im.index.isin(meta.index)]
        meta.loc[im.index, "benchmark"] = im
        meta.loc[im.index, "index_key"] = im.map(lambda v: re.sub(r"[^A-Z0-9]", "", v.upper()))
        meta.loc[im.index, "index_source"] = "manual (index_map.csv)"
        log(f"{len(im)} indices taken from index_map.csv")
    log(f"Reference index known for {meta['index_key'].notna().sum()} of {len(meta)} ETFs")

    counts = meta["aum_source"].value_counts()
    log("Fund volume currency: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    changed = meta[(meta["fund_ccy"] != known["currency"]) & meta["total_assets"].notna()]
    if len(changed):
        log(f"  {len(changed)} ETFs do not report their volume in the trading currency "
            f"({', '.join(sorted(changed['fund_ccy'].unique()))})")
    return meta["aum_usd"], meta


# ------------------------------------------------------------------ Stocks: market capitalization, industry
STOCK_INFO_COLUMNS = ["ticker", "market_cap", "currency", "sector", "industry", "long_name"]


def _fetch_stock_info(ticker: str) -> dict:
    import yfinance as yf

    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
            return {"ticker": ticker, "market_cap": info.get("marketCap"), "currency": info.get("currency"),
                    "sector": info.get("sector"), "industry": info.get("industry"),
                    "long_name": info.get("longName") or info.get("shortName")}
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {"ticker": ticker, "market_cap": None, "currency": None, "sector": None, "industry": None,
            "long_name": None}


def load_stock_info(tickers: list[str], refresh: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Market capitalization in USD per ticker (NaN = unknown) as well as sector / industry according to Yahoo."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = cache_file("stock_info")
    known = pd.DataFrame(columns=STOCK_INFO_COLUMNS)
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 30 * 86400:
        cached = pd.read_csv(cache, keep_default_na=False, na_values=[""])   # ticker "NA" is not a missing value
        if set(STOCK_INFO_COLUMNS) <= set(cached.columns):
            known = cached
    todo = [t for t in tickers if t not in set(known["ticker"])]
    if todo:
        log(f"Loading market capitalization and industry for {len(todo)} stocks from Yahoo Finance ...")
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_fetch_stock_info, t) for t in todo]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % 100 == 0:
                    log(f"  {k}/{len(todo)}")
                    pd.concat([known, pd.DataFrame(rows)]).to_csv(cache, index=False)
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)

    known = known.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(tickers)
    usd_per = _usd_per_unit(["EUR"] + known["currency"].dropna().tolist())
    meta = pd.DataFrame({"market_cap": pd.to_numeric(known["market_cap"], errors="coerce"),
                         "currency": known["currency"], "sector": known["sector"],
                         "industry": known["industry"], "long_name": known["long_name"]}, index=known.index)
    meta["aum_usd"] = meta["market_cap"] * meta["currency"].map(usd_per).astype(float)

    # group for the pair comparison: industry or sector according to Yahoo, can be overwritten manually
    col = {"industry": "industry", "sector": "sector"}.get(STOCK_GROUP_BY)
    meta["peer_group"] = meta[col] if col else ""
    manual = BASE_DIR / ("groups_us_stocks.csv" if MARKET == "us" else "groups_stocks.csv")
    if manual.exists():
        g = read_user_csv(manual, dtype=str)
        g["ticker"] = g["ticker"].str.strip().str.upper()
        if MARKET != "us":
            g["ticker"] = np.where(g["ticker"].str.contains(r"\."), g["ticker"], g["ticker"] + ".DE")
        g = g.set_index("ticker")["peer_group"].dropna()
        g = g[g.index.isin(meta.index)]
        meta.loc[g.index, "peer_group"] = g
        log(f"{len(g)} groups from {manual.name} taken over")
    log(f"Market capitalization known for {int(meta['aum_usd'].notna().sum())} of {len(meta)} stocks, "
        f"Industry known for {int(meta['industry'].notna().sum())}")
    return meta["aum_usd"], meta


# ------------------------------------------------------------------ Statistics
def _test_chunk(pairs: np.ndarray, X: np.ndarray) -> list[tuple]:
    """Engle-Granger in both directions for a list of column pairs (i, j) from X."""
    out = []
    for i, j in pairs:
        x, y = X[:, i], X[:, j]
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        if len(x) < 200:
            continue
        try:
            p_xy = coint(x, y, maxlag=COINT_MAXLAG)[1]
            p_yx = coint(y, x, maxlag=COINT_MAXLAG)[1]
        except Exception:
            continue
        beta, alpha = np.polyfit(y, x, 1)          # log(A) = alpha + beta * log(B)
        spread_std = float(np.std(x - beta * y - alpha, ddof=1))
        out.append((int(i), int(j), max(p_xy, p_yx), p_xy, p_yx, beta, alpha, spread_std))
    return out


def liquidity_stats(px: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Per ETF in the window: share of 'stale' days (no trading / price unchanged) and median turnover in EUR."""
    listed = px.ffill().notna()
    ret = px.pct_change(fill_method=None)
    v = vol.reindex_like(px)
    n_listed = listed.sum().replace(0, np.nan)
    has_vol = (v > 0).sum() >= 0.05 * n_listed          # otherwise Yahoo provides no turnover
    traded = px.notna() & (ret.abs() > 0) & ((v > 0) | ~has_vol)
    stale = 1 - traded.where(listed).sum() / n_listed
    turnover = (px * v.fillna(0)).where(listed).median()
    turnover[~has_vol] = np.nan
    return pd.DataFrame({"stale_share": stale, "turnover_eur": turnover})


def monthly_te_stats(lp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Tracking error p. a. (beta = 1) on monthly returns and relative to volatility (TE / sqrt(vol_a*vol_b))."""
    mret = lp.iloc[::21].diff().iloc[1:]
    cov = mret.cov(min_periods=12).to_numpy()
    var = np.diag(cov)
    te = np.sqrt(np.clip((var[:, None] + var[None, :] - 2 * cov) * 12, 0, None))   # NaN if there are too few months
    vol = np.sqrt(np.clip(var, 0, None) * 12)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = te / np.sqrt(np.outer(vol, vol))
    return te, rel


def detect_price_errors(lp: pd.DataFrame) -> pd.DataFrame:
    """Price blocks that jump relative to similar ETFs and later jump back (Yahoo data errors).

    Comparison per ETF with the median of the 5 most strongly correlated ETFs (correlation >= PEER_MIN_CORR).
    ETFs without such comparison ETFs are not checked.
    """
    r = lp.diff()
    R = r.to_numpy()
    corr = np.corrcoef(np.nan_to_num(R), rowvar=False)
    np.fill_diagonal(corr, -np.inf)
    rows = []
    for k, t in enumerate(lp.columns):
        peers = np.argsort(-corr[k])[:5]
        peers = peers[corr[k, peers] >= PEER_MIN_CORR]
        if len(peers) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # days on which all peers are NaN
            resid = R[:, k] - np.nanmedian(R[:, peers], axis=1)
        big = np.where(np.abs(np.nan_to_num(resid)) > JUMP_THRESHOLD)[0]
        for d in big:
            back = big[(big > d) & (big <= d + JUMP_REVERT_DAYS)]
            back = back[np.sign(resid[back]) != np.sign(resid[d])]
            if len(back):
                rows.append({"ticker": t, "jump_on": lp.index[d].date(), "deviation": resid[d],
                             "back_on": lp.index[back[0]].date(), "deviation_back": resid[back[0]]})
                break
    return pd.DataFrame(rows, columns=["ticker", "jump_on", "deviation", "back_on", "deviation_back"])


def prepare_window(px: pd.DataFrame, vol: pd.DataFrame, names: pd.Series, day_label: str = "",
                   quiet: bool = False):
    """Check coverage, liquidity and data errors. Returns log prices of the clean ETFs and the status per ETF."""
    liq = liquidity_stats(px, vol)
    cov = px.notna().mean()
    status = pd.Series("ok", index=px.columns, dtype=object)
    status[cov < MIN_COVERAGE] = "too little price history"
    min_to = min_turnover()
    liquid = (liq["stale_share"] <= MAX_STALE_SHARE) & (
        (liq["turnover_eur"] >= min_to)
        | (liq["turnover_eur"].isna() & KEEP_UNKNOWN_TURNOVER))
    status[(status == "ok") & ~liquid.reindex(status.index, fill_value=False)] = "illiquid" if MARKET == "us" else "illiquid on Xetra"
    ok = status.index[status == "ok"]
    lp = np.log(px[ok]).ffill(limit=5)
    check_errors = len(ok) > 1 and (MODE != "stocks" or STOCK_DETECT_PRICE_ERRORS)
    errors = detect_price_errors(lp) if check_errors else pd.DataFrame(
        columns=["ticker", "jump_on", "deviation", "back_on", "deviation_back"])
    status[status.index.isin(errors["ticker"])] = "data error (price jump)"
    if len(errors):
        errors.insert(1, "name", errors["ticker"].map(names))
    clean = status.index[status == "ok"]
    if quiet:                                               # second pass in the same period
        return lp[clean], liq, status
    save_csv(errors, f"data_errors{day_label}.csv")
    log(f"    {int((cov >= MIN_COVERAGE).sum())} instruments with enough history, {len(ok)} liquid "
        f"(<= {MAX_STALE_SHARE:.0%} days without price change, median turnover >= "
        f"{min_to / 1e3:.0f} thous. {'USD' if MARKET == 'us' else 'EUR'}), {len(errors)} with data errors "
        f"-> {len(clean)} clean")
    return lp[clean], liq, status


def _bh(p: pd.Series) -> np.ndarray:
    return multipletests(p, method="fdr_bh")[1] if len(p) else np.array([])


def screen(px: pd.DataFrame, vol: pd.DataFrame, aum: pd.Series, names: pd.Series,
           index_key: pd.Series | None = None, pair_list: pd.DataFrame | None = None,
           day_label: str = "", spread_filter: bool = True, near_same: bool = False,
           groups: pd.Series | None = None) -> pd.DataFrame:
    """Test pairs and return them as a ranking (by p-value).

    Without pair_list: all combinations (with correlation pre-filter).
    With pair_list (columns ticker_a, ticker_b, ...): only these pairs, without correlation pre-filter;
    pairs that cannot be tested appear with a reason in the column 'status'.
    spread_filter: do not test pairs with a spread width below MIN_SPREAD_STD. For the own
    pair list the filter is off (narrow spreads, e.g. for bonds, are chosen deliberately there).
    near_same: evaluation 'almost identical indices'. Without pair_list, exactly those pairs are tested that
    count as 'same index' only because of the tracking error and whose spread is at least
    NEAR_SAME_MIN_SPREAD wide. With pair_list, the tracking error is not a reason for exclusion.
    groups: group per ticker (stocks: industry / sector). Without pair_list, only pairs within
    the same group are tested. In stock mode there is no 'same index / same fund' filter
    (common and preferred shares of the same company are exactly what is wanted there).
    """
    lp, liq, status = prepare_window(px, vol, names, day_label, quiet=near_same)
    cols = lp.columns
    pos = {t: k for k, t in enumerate(cols)}
    rets = lp.diff().fillna(0.0).to_numpy()
    corr = np.corrcoef(rets, rowvar=False) if len(cols) > 1 else np.ones((len(cols), len(cols)))
    te, rel = monthly_te_stats(lp)
    key = index_key.reindex(cols).to_numpy(dtype=object) if index_key is not None else None
    aum_v = aum.reindex(cols).to_numpy(dtype=float)
    # spread width in advance: std of the residual of the regression log(A) on log(B) = std(A) * sqrt(1 - rho^2)
    lvl_corr = lp.corr().to_numpy() if len(cols) > 1 else np.ones((len(cols), len(cols)))
    lvl_std = lp.std().to_numpy()

    TE_REASON = "same index (tracking error)"

    def spread_width(i, j) -> np.ndarray:
        return lvl_std[i] * np.sqrt(np.clip(1 - lvl_corr[i, j] ** 2, 0, None))

    def same_reason(i, j) -> np.ndarray:
        """Reason why a pair counts as 'same index / same fund' ('' = different)."""
        reason = np.full(len(i), "", dtype=object)
        if MODE != "etf":
            return reason
        reason[rel[i, j] < MIN_REL_TE] = TE_REASON                 # NaN comparison = False
        if key is not None:
            ki, kj = key[i], key[j]
            same_key = np.array([isinstance(a, str) and a != "" and a == b for a, b in zip(ki, kj)], dtype=bool)
            reason[same_key] = "same index according to Yahoo / index_map.csv"
        if SAME_AUM_IS_SAME_FUND:
            a, b = aum_v[i], aum_v[j]
            with np.errstate(invalid="ignore"):
                same_fund = (a > 0) & (np.abs(a - b) <= 1e-6 * np.maximum(a, b))
            reason[same_fund] = "same fund (identical fund volume)"
        return reason

    if pair_list is None and near_same:
        iu, ju = np.triu_indices(len(cols), k=1)
        c = corr[iu, ju]
        in_corr = (c >= MIN_RET_CORR) & (c <= MAX_RET_CORR)
        cand = in_corr & (same_reason(iu, ju) == TE_REASON)
        keep = cand & (spread_width(iu, ju) >= NEAR_SAME_MIN_SPREAD)
        pairs = np.column_stack([iu[keep], ju[keep]])
        log(f"    Almost identical indices: {cand.sum()} pairs with rel. tracking error < {MIN_REL_TE}, "
            f"of which {len(pairs)} with spread >= {NEAR_SAME_MIN_SPREAD:.1%} are tested")
        info = None
    elif pair_list is None:
        iu, ju = np.triu_indices(len(cols), k=1)
        c = corr[iu, ju]
        min_corr = STOCK_MIN_RET_CORR if MODE == "stocks" else MIN_RET_CORR
        in_corr = (c >= min_corr) & (c <= MAX_RET_CORR)
        n_all = len(iu)
        if groups is not None:                              # only pairs within the same group
            g = groups.reindex(cols).to_numpy(dtype=object)
            same_group = np.array([isinstance(x, str) and x != "" and x == y for x, y in zip(g[iu], g[ju])],
                                  dtype=bool)
            log(f"    {n_all} pairs, {int(same_group.sum())} within the same group "
                f"({pd.Series(g).replace('', np.nan).nunique()} groups)")
            in_corr &= same_group
        reason = same_reason(iu, ju)
        si = in_corr & (reason != "")
        width = spread_width(iu, ju)
        narrow = in_corr & ~si & (width < MIN_SPREAD_STD) & spread_filter

        def excluded(mask, extra):
            d = pd.DataFrame({"etf_a": cols[iu[mask]], "etf_b": cols[ju[mask]],
                              "name_a": names.reindex(cols[iu[mask]]).to_numpy(),
                              "name_b": names.reindex(cols[ju[mask]]).to_numpy(), **extra,
                              "ret_corr_daily": c[mask], "te_monthly": te[iu[mask], ju[mask]],
                              "rel_te": rel[iu[mask], ju[mask]], "spread_std": width[mask]})
            return d

        if MODE == "etf":
            save_csv(excluded(si, {"reason": reason[si]}).sort_values("rel_te"), f"same_index_excluded{day_label}.csv")
        save_csv(excluded(narrow, {}).sort_values("spread_std"), f"narrow_spread_excluded{day_label}.csv")
        keep = in_corr & ~si & ~narrow
        order = np.argsort(-c[keep])[:MAX_PAIRS_TO_TEST]
        pairs = np.column_stack([iu[keep][order], ju[keep][order]])
        if MODE == "etf":
            log(f"    {len(iu)} pairs, {in_corr.sum()} after correlation filter, {si.sum()} of which same "
                f"index/fund, {narrow.sum()} with spread < {MIN_SPREAD_STD:.1%}, {len(pairs)} are tested")
        else:
            log(f"    {in_corr.sum()} pairs after correlation filter (>= {min_corr}), {narrow.sum()} with "
                f"spread < {MIN_SPREAD_STD:.1%}, {len(pairs)} are tested")
        info = None
    else:
        info = pair_list.copy()
        st = []
        for a, b in zip(info["ticker_a"], info["ticker_b"]):
            sa, sb = status.get(a, "no prices / not in the universe"), status.get(b, "no prices / not in the universe")
            if sa != "ok" or sb != "ok":
                st.append("; ".join(f"{t}: {s}" for t, s in ((a, sa), (b, sb)) if s != "ok"))
            elif (why := same_reason(np.array([pos[a]]), np.array([pos[b]]))[0]) != "" and not (
                    near_same and why == TE_REASON):
                st.append(f"same index (rel. TE {rel[pos[a], pos[b]]:.2f})"
                          if why.startswith("same index (tracking") else why)
            elif spread_filter and (wd := spread_width(np.array([pos[a]]), np.array([pos[b]]))[0]) < MIN_SPREAD_STD:
                st.append(f"spread too narrow ({wd:.2%} < {MIN_SPREAD_STD:.1%})")
            else:
                st.append("tested")
        info["status"] = st
        ok = info["status"] == "tested"
        pairs = np.array([[pos[a], pos[b]] for a, b in zip(info.loc[ok, "ticker_a"], info.loc[ok, "ticker_b"])],
                         dtype=int).reshape(-1, 2)
        log(f"    Pair list: {len(info)} pairs, {len(pairs)} testable, {len(info) - len(pairs)} not testable "
            f"(reason in column 'status')")

    res = []
    if len(pairs):
        X = lp.to_numpy(dtype=float)
        n_chunks = max(1, min(len(pairs) // 200, 256))
        chunks = np.array_split(pairs, n_chunks)
        try:
            out = Parallel(n_jobs=N_JOBS)(delayed(_test_chunk)(ch, X) for ch in chunks)
        except Exception as e:                              # e.g. problems with process start under Windows
            log(f"    Parallel calculation failed ({type(e).__name__}: {e}); continuing sequentially")
            out = [_test_chunk(ch, X) for ch in chunks]
        res = [r for chunk in out for r in chunk]
    df = pd.DataFrame(res, columns=["i", "j", "p_value", "p_ab", "p_ba", "beta", "alpha", "spread_std"])
    df["etf_a"] = cols[df["i"]] if len(df) else []
    df["etf_b"] = cols[df["j"]] if len(df) else []
    df["ret_corr"] = corr[df["i"], df["j"]] if len(df) else []
    df["te_monthly"] = te[df["i"], df["j"]] if len(df) else []
    df["rel_te"] = rel[df["i"], df["j"]] if len(df) else []
    df["p_adj"] = _bh(df["p_value"])
    df["significant"] = (df["p_adj"] <= FDR_ALPHA) & (df["beta"] > 0)
    df = df.sort_values("p_value").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    if info is not None:                                    # keep the order of the list, add information
        df = info.merge(df.drop(columns=["i", "j", "rank"]), left_on=["ticker_a", "ticker_b"],
                        right_on=["etf_a", "etf_b"], how="left").drop(columns=["etf_a", "etf_b"])
        df = df.rename(columns={"ticker_a": "etf_a", "ticker_b": "etf_b"})
        df = df.sort_values("p_value", na_position="last").reset_index(drop=True)
        df.insert(0, "rank", np.where(df["p_value"].notna(), np.arange(1, len(df) + 1), np.nan))

    top = df.dropna(subset=["p_value"]).head(RANK_TOP_FOR_COUNT)
    freq = pd.concat([top["etf_a"], top["etf_b"]]).value_counts()
    df["name_a"] = df["etf_a"].map(names)
    df["name_b"] = df["etf_b"].map(names)
    df["aum_a_musd"] = df["etf_a"].map(aum) / 1e6
    df["aum_b_musd"] = df["etf_b"].map(aum) / 1e6
    df["turnover_a_teur"] = df["etf_a"].map(liq["turnover_eur"]) / 1e3
    df["turnover_b_teur"] = df["etf_b"].map(liq["turnover_eur"]) / 1e3
    df["frequency_a"] = df["etf_a"].map(freq).fillna(0).astype(int)
    df["frequency_b"] = df["etf_b"].map(freq).fillna(0).astype(int)
    n_sig = int(df["significant"].fillna(False).astype(bool).sum())
    log(f"    {df['p_value'].notna().sum()} pairs tested, {n_sig} significant (BH-p <= {FDR_ALPHA}, beta > 0)")

    front = ["rank", "etf_a", "etf_b", "name_a", "name_b"]
    if groups is not None:
        df["peer_group_a"] = df["etf_a"].map(groups)
        df["peer_group_b"] = df["etf_b"].map(groups)
        front += ["peer_group_a", "peer_group_b"]
    if info is not None:
        front += [c for c in info.columns if c not in ("ticker_a", "ticker_b")]
    cols_out = front + ["p_value", "p_adj", "significant", "p_ab", "p_ba", "beta", "alpha", "spread_std",
                        "ret_corr", "te_monthly", "rel_te", "aum_a_musd", "aum_b_musd", "turnover_a_teur", "turnover_b_teur",
                        "frequency_a", "frequency_b"]
    out = df[[c for c in cols_out if c in df.columns]]
    if MODE == "stocks":                                    # for stocks the size is the market capitalization
        out = out.rename(columns={"aum_a_musd": "mktcap_a_musd", "aum_b_musd": "mktcap_b_musd"})
    return out


# ------------------------------------------------------------------ Procedure
def clean_prices(close: pd.DataFrame) -> pd.DataFrame:
    close = close.sort_index()
    close = close[close.index.dayofweek < 5]
    close = close.where(close > 0)
    # remove holidays / data gaps: days on which fewer than half of the ETFs
    # that were already listed at that time have a price. (Do not measure against today's
    # universe - otherwise all early years drop out in which many ETFs did not yet exist.)
    listed = close.ffill().notna().sum(axis=1)
    quoted = close.notna().sum(axis=1)
    close = close[(listed > 0) & (quoted >= 0.5 * listed)]
    return close


def load_pair_list(path: Path) -> pd.DataFrame:
    """Own pair list: columns ticker_a;ticker_b (+ any others, e.g. category;rationale).

    Lines starting with # are comments. The separator (; , or tab) is detected from the header line.
    If the last column itself contains separators (e.g. a semicolon in the justification text),
    they remain part of the text and do not shift any columns.
    """
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise SystemExit(f"{path.name} is empty")
    sep = max(";,\t", key=lines[0].count)
    header = [h.strip().strip('"').lower() for h in lines[0].split(sep)]
    if not {"ticker_a", "ticker_b"} <= set(header):
        raise SystemExit(f"{path.name} needs the columns ticker_a and ticker_b (found: {header})")
    rows = []
    for ln in lines[1:]:
        parts = [p.strip().strip('"') for p in ln.split(sep, len(header) - 1)]
        rows.append((parts + [""] * len(header))[:len(header)])
    pl = pd.DataFrame(rows, columns=header)
    if MARKET == "us":                                      # Yahoo notation: BRK.B -> BRK-B
        valid = r"[A-Z]{1,5}(-[A-Z])?"
        for c in ("ticker_a", "ticker_b"):
            pl[c] = pl[c].str.upper().str.replace(".", "-", regex=False)
    else:
        valid = r"[A-Z0-9]{1,6}\.[A-Z]{1,3}"
        for c in ("ticker_a", "ticker_b"):
            pl[c] = pl[c].str.upper()
            pl[c] = np.where(pl[c].str.contains(r"\."), pl[c], pl[c] + ".DE")   # "EXV1" -> "EXV1.DE"
    bad = pl[~pl["ticker_a"].str.fullmatch(valid) | ~pl["ticker_b"].str.fullmatch(valid)]
    if len(bad):
        log(f"  {len(bad)} row(s) of the pair list have no valid ticker symbol and are skipped: "
            + "; ".join(f"{a} / {b}" for a, b in zip(bad["ticker_a"], bad["ticker_b"]))[:300])
        pl = pl.drop(bad.index)
    return pl.drop_duplicates(["ticker_a", "ticker_b"]).reset_index(drop=True)


def select_candidates(stage1: pd.DataFrame) -> pd.DataFrame:
    """Candidates for stage 2: best pairs from stage 1 (by p-value), beta > 0, limited per ETF."""
    ok = stage1[stage1["p_value"].notna() & (stage1["p_value"] <= STAGE1_MAX_P) & (stage1["beta"] > 0)]
    ok = ok.sort_values("p_value")
    used: dict[str, int] = {}
    rows = []
    for idx, r in ok.iterrows():
        if used.get(r["etf_a"], 0) >= STAGE1_MAX_PER_ETF or used.get(r["etf_b"], 0) >= STAGE1_MAX_PER_ETF:
            continue
        rows.append(idx)
        used[r["etf_a"]] = used.get(r["etf_a"], 0) + 1
        used[r["etf_b"]] = used.get(r["etf_b"], 0) + 1
        if len(rows) == STAGE1_TOP_N:
            break
    cand = ok.loc[rows]
    extra = [c for c in ("category", "rationale") if c in cand.columns]
    out = pd.DataFrame({"ticker_a": cand["etf_a"].to_numpy(), "ticker_b": cand["etf_b"].to_numpy()})
    for c in extra:
        out[c] = cand[c].to_numpy()
    out["rank_stage1"] = cand["rank"].to_numpy()
    out["p_stage1"] = cand["p_value"].to_numpy()
    out["beta_stage1"] = cand["beta"].to_numpy()
    return out


def run_two_stage(close, volume, aum, names, index_key, pair_list, mark_small, mode, f0, last,
                  near_same: bool = False, groups: pd.Series | None = None) -> None:
    """Stage 1: select candidates in the earlier period. Stage 2: test only these in the later period.

    The periods do not overlap. In stage 2, the correction for many tests refers only to
    the few candidates; the data of stage 2 were not used for the selection.
    """
    s0 = f0 - pd.DateOffset(years=STAGE1_YEARS)
    w1 = (close.index > s0) & (close.index <= f0)
    w2 = close.index > f0
    if w1.sum() < 252 * STAGE1_YEARS * 0.9:
        raise SystemExit(f"Too little price history for stage 1 ({s0:%d.%m.%Y} - {f0:%d.%m.%Y}): "
                         f"only {int(w1.sum())} trading days. Increase HISTORY_YEARS and reload with --refresh.")

    sfx = "_near_same" if near_same else ""
    if near_same:
        mode = "almost identical indices"
    log(f"STAGE 1 - selection ({mode}): {s0:%d.%m.%Y} - {f0:%d.%m.%Y}")
    spread_filter = pair_list is None and not near_same    # own pair list / almost identical: no spread filter
    stage1 = mark_small(screen(close[w1], volume[w1], aum, names, index_key, pair_list, day_label="_stage1",
                               spread_filter=spread_filter, near_same=near_same, groups=groups))
    save_csv(stage1, f"ranking{sfx}_stage1.csv")
    cand = select_candidates(stage1)
    n1 = int(stage1["p_value"].notna().sum())
    log(f"    {len(cand)} candidates for stage 2 (p <= {STAGE1_MAX_P} in stage 1, beta > 0, "
        f"max. {STAGE1_MAX_PER_ETF} pairs per ETF, max. {STAGE1_TOP_N})")

    print_header = lambda: (print(), print("=" * 78), print(f"SUMMARY (two-stage, {mode})"),
                            print("=" * 78))
    if len(cand) == 0:
        save_csv(cand, f"ranking{sfx}_two_stage.csv")
        print_header()
        print(f"Stage 1 ({s0:%d.%m.%Y} - {f0:%d.%m.%Y}): {n1} pairs tested, no candidate with "
              f"p <= {STAGE1_MAX_P}. Stage 2 is skipped.")
        print(f"\nResults: {OUT_DIR}")
        return

    log(f"STAGE 2 - confirmation: {f0:%d.%m.%Y} - {last:%d.%m.%Y}")
    stage2 = screen(close[w2], volume[w2], aum, names, index_key, cand, day_label="_stage2",
                    spread_filter=spread_filter, near_same=near_same, groups=groups)
    stage2 = stage2.rename(columns={"p_value": "p_stage2", "p_adj": "p_adj_stage2", "beta": "beta_stage2",
                                    "rank": "rank_stage2", "status": "status_stage2"})
    front = ["rank_stage2", "etf_a", "etf_b", "name_a", "name_b"]
    front += [c for c in ("category", "rationale") if c in stage2.columns]
    front += ["rank_stage1", "p_stage1", "p_stage2", "p_adj_stage2", "significant", "status_stage2",
              "beta_stage1", "beta_stage2"]
    stage2 = stage2[front + [c for c in stage2.columns if c not in front]]
    save_csv(stage2, f"ranking{sfx}_two_stage.csv")

    tested = stage2[stage2["p_stage2"].notna()]
    n_sig = int(tested["significant"].astype(bool).sum())
    print_header()
    print(f"Stage 1 ({s0:%d.%m.%Y} - {f0:%d.%m.%Y}): {n1} pairs tested -> {len(cand)} candidates")
    print(f"Stage 2 ({f0:%d.%m.%Y} - {last:%d.%m.%Y}): {len(tested)} candidates tested, "
          f"{n_sig} confirmed (BH-p <= {FDR_ALPHA} over {len(tested)} tests, beta > 0)")
    if len(tested):
        need = FDR_ALPHA / len(tested)
        print(f"  required p-value for the best pair in stage 2: {need:.2g} "
              f"(without preselection, with {n1} tests, it would be approx. {FDR_ALPHA / max(n1, 1):.1g})")
    view = stage2[["etf_a", "etf_b", "name_a", "name_b", "rank_stage1", "p_stage1", "p_stage2",
                   "p_adj_stage2", "significant"]]
    print()
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    not_tested = stage2[stage2["p_stage2"].isna()]
    if len(not_tested):
        print(f"\nNot testable in stage 2: {len(not_tested)} candidates (reason in column 'status_stage2')")
    print(f"\nResults: {OUT_DIR}")


def run_screen(universe: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
               aum: pd.Series, aum_meta: pd.DataFrame | None = None,
               pair_list: pd.DataFrame | None = None, two_stage: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    names = universe.set_index("ticker")["name"]
    sec_type = universe.set_index("ticker")["sec_type"] if "sec_type" in universe else pd.Series(dtype=object)
    index_key = aum_meta["index_key"] if aum_meta is not None and "index_key" in aum_meta else None
    stocks = MODE == "stocks"
    groups = None
    if stocks and aum_meta is not None and "peer_group" in aum_meta and STOCK_GROUP_BY:
        groups = aum_meta["peer_group"].where(aum_meta["peer_group"].notna(), "")
    size_name = "Market capitalization" if stocks else "Fund volume"
    min_size = MIN_MARKET_CAP_USD if stocks else MIN_AUM_USD
    close = clean_prices(close)

    # --- size filter: market capitalization (stocks) or fund volume (ETFs; ETCs optionally excepted)
    min_rows = int(252 * FORMATION_YEARS * MIN_COVERAGE)
    has_hist = close.columns[close.notna().sum() >= min_rows]
    aum = aum.reindex(close.columns)
    missing = [t for t in has_hist if pd.isna(aum.get(t))]
    save_csv(universe[universe["ticker"].isin(missing)], "missing_mktcap.csv" if stocks else "missing_aum.csv")
    exempt = sec_type.reindex(close.columns).eq("ETC") & (not AUM_FILTER_ETC)
    aum_ok = (aum >= min_size) | exempt
    if KEEP_MISSING_AUM or pair_list is not None:          # own pairs: do not exclude missing AUM
        aum_ok = aum_ok | aum.isna()
    tradable = [t for t in has_hist if aum_ok.get(t, False)]
    log(f"Universe: {close.shape[1]} instruments with prices, {len(has_hist)} with >= {FORMATION_YEARS} J. "
        f"history, {len(missing)} of which without data: {size_name}, {len(tradable)} after size filter "
        f"({size_name} > {min_size / 1e6:.0f} million USD"
        f"{'; ETCs excepted' if not stocks and MARKET != 'us' and not AUM_FILTER_ETC else ''})")

    uni_out = pd.DataFrame({"ticker": tradable, "name": names.reindex(tradable).to_numpy(),
                            "sec_type": sec_type.reindex(tradable).to_numpy(),
                            ("mktcap_musd" if stocks else "aum_musd"): (aum.reindex(tradable) / 1e6).to_numpy()})
    if stocks and aum_meta is not None:
        for c in ("sector", "industry", "peer_group", "long_name"):
            if c in aum_meta:
                uni_out[c] = aum_meta[c].reindex(tradable).to_numpy()
        check = aum_meta.join(names.rename("name")).sort_values("aum_usd", ascending=False)
        check["mktcap_musd"] = check.pop("aum_usd") / 1e6
        save_csv(check.reset_index(names="ticker"), "stock_details.csv")
    elif aum_meta is not None:
        uni_out["fund_ccy"] = aum_meta["fund_ccy"].reindex(tradable).to_numpy()
        uni_out["aum_source"] = aum_meta["aum_source"].reindex(tradable).to_numpy()
        check = aum_meta.join(names.rename("name")).sort_values("aum_usd", ascending=False)
        check["aum_musd"] = check.pop("aum_usd") / 1e6
        save_csv(check.reset_index(names="ticker"), "etf_details.csv")
    save_csv(uni_out, "universe_used.csv")

    if pair_list is not None:                              # mark pairs with a fund that is too small
        small = set(has_hist) - set(tradable)
        pair_list = pair_list.copy()
        pair_list["_small"] = [", ".join(t for t in (a, b) if t in small)
                               for a, b in zip(pair_list["ticker_a"], pair_list["ticker_b"])]
    close = close[tradable]
    volume = volume.reindex(index=close.index, columns=close.columns)

    # --- search: last FORMATION_YEARS years up to the last price date
    last = close.index[-1]
    f0 = last - pd.DateOffset(years=FORMATION_YEARS)
    mode = "pair list" if pair_list is not None else (
        f"all pairs per {'industry' if STOCK_GROUP_BY == 'industry' else 'sector'}"
        if groups is not None else "all combinations")
    mode = f"{'US-' if MARKET == 'us' else ''}{'stocks' if stocks else 'ETFs'}, {mode}"
    log(f"Searching for cointegrated pairs ({mode}): {f0:%d.%m.%Y} - {last:%d.%m.%Y}")
    in_form = close.index > f0
    pl = pair_list.drop(columns="_small") if pair_list is not None else None

    def mark_small(rk: pd.DataFrame) -> pd.DataFrame:
        if pair_list is None:
            return rk
        small_flag = dict(zip(zip(pair_list["ticker_a"], pair_list["ticker_b"]), pair_list["_small"]))
        for k, r in rk.iterrows():
            kl = small_flag.get((r["etf_a"], r["etf_b"]), "")
            if kl and r["status"] != "tested":
                rk.at[k, "status"] = f"{size_name} < {min_size / 1e6:.0f} million USD: {kl}"
        return rk

    if two_stage:
        run_two_stage(close, volume, aum, names, index_key, pl, mark_small, mode, f0, last, groups=groups)
        if pair_list is None and not stocks:               # additionally: almost identical indices, two-stage
            run_two_stage(close, volume, aum, names, index_key, None, mark_small, mode, f0, last,
                          near_same=True)
        return

    ranking = screen(close[in_form], volume[in_form], aum, names, index_key, pl,
                     spread_filter=pl is None, groups=groups)

    if pair_list is not None:
        ranking = mark_small(ranking)
        save_csv(ranking, "ranking_pair_list.csv")
    else:
        save_csv(ranking, "ranking.csv")
        save_csv(ranking[ranking["significant"]], "cointegrated_pairs.csv")

    # --- summary
    tested = ranking[ranking["p_value"].notna()]
    print()
    print("=" * 78)
    print(f"SUMMARY ({mode})")
    print("=" * 78)
    print(f"Period: {f0:%d.%m.%Y} - {last:%d.%m.%Y}")
    print(f"Tested pairs: {len(tested)} | significant after correction (BH-p <= {FDR_ALPHA}): "
          f"{int(tested['significant'].astype(bool).sum())}")
    if len(tested):
        for thr in (0.001, 0.01, 0.05):
            print(f"  p < {thr:<5}: {int((tested['p_value'] < thr).sum()):6d}   "
                  f"(expected by chance alone: {thr * len(tested):8.1f})")
        view = tested[["rank", "etf_a", "etf_b", "name_a", "name_b", "p_value", "p_adj", "beta"]].head(15)
        print()
        print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if pair_list is not None and (ranking["status"] != "tested").any():
        print(f"\nNot testable: {int((ranking['status'] != 'tested').sum())} pairs "
              f"(reason in ranking_pair_list.csv, column 'status')")

    # --- additional evaluation (ETFs only): almost identical indices with a wide spread (separate ranking)
    if pair_list is None and not stocks:
        near = screen(close[in_form], volume[in_form], aum, names, index_key, None, near_same=True)
        save_csv(near, "ranking_near_same.csv")
        print()
        print("-" * 78)
        print(f"ALMOST IDENTICAL INDICES with spread >= {NEAR_SAME_MIN_SPREAD:.1%} (rel. tracking error < {MIN_REL_TE})")
        print("-" * 78)
        print(f"Tested pairs: {len(near)} | significant after correction (BH-p <= {FDR_ALPHA}): "
              f"{int(near['significant'].astype(bool).sum()) if len(near) else 0}")
        if len(near):
            for thr in (0.001, 0.01, 0.05):
                print(f"  p < {thr:<5}: {int((near['p_value'] < thr).sum()):6d}   "
                      f"(expected by chance alone: {thr * len(near):8.1f})")
            view = near[["rank", "etf_a", "etf_b", "name_a", "name_b", "p_value", "p_adj", "beta",
                         "spread_std", "rel_te"]].head(15)
            print()
            print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nResults: {OUT_DIR}")


# ------------------------------------------------------------------ Additional evaluation: mean reversion
def mr_pair_series(la: np.ndarray, lb: np.ndarray) -> dict:
    """Trading rule for a pair over the whole history. la, lb: log prices (NaN allowed).

    Spread = log(A) - log(B), i.e. the same euro amount long and short. z measures the distance from the
    moving average of the last MR_LOOKBACK days in standard deviations. The position of day t
    is set with the closing price of t and only earns the return of t+1 (no look-ahead).
    Return per day relative to the capital employed (half each long and short).
    """
    s = pd.Series(la - lb)
    m = s.rolling(MR_LOOKBACK, min_periods=MR_LOOKBACK).mean()
    sd = s.rolling(MR_LOOKBACK, min_periods=MR_LOOKBACK).std()
    z = ((s - m) / sd).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    cur, held = 0, 0
    for t in range(n):
        zt = z[t]
        if zt != zt:                                        # NaN: no price -> close out
            cur, held = 0, 0
        elif cur == 0:
            if zt > MR_ENTRY_Z:
                cur, held = -1, 0
            elif zt < -MR_ENTRY_Z:
                cur, held = 1, 0
        else:
            held += 1
            if (cur == 1 and zt >= -MR_EXIT_Z) or (cur == -1 and zt <= MR_EXIT_Z) or held >= MR_MAX_HOLD:
                cur, held = 0, 0
        pos[t] = cur
    ra = np.concatenate([[0.0], np.expm1(np.diff(la))])
    rb = np.concatenate([[0.0], np.expm1(np.diff(lb))])
    spread_ret = 0.5 * (np.nan_to_num(ra) - np.nan_to_num(rb))
    prev = np.concatenate([[0.0], pos[:-1]])
    gross = prev * spread_ret
    costs = np.abs(np.diff(np.concatenate([[0.0], pos]))) * MR_COST_BPS / 1e4
    borrow = np.abs(prev) * 0.5 * MR_BORROW_FEE / 252
    return {"pos": pos, "gross": gross, "net": gross - costs - borrow}


def mr_stats(net: np.ndarray, pos: np.ndarray, gross: np.ndarray | None = None) -> dict:
    """Metrics of a period: Sharpe (annualized), return p. a., trades, hit rate, holding period."""
    out = {"sharpe": np.nan, "return_pa": np.nan, "trades": 0, "hit_rate": np.nan, "holding_days": np.nan}
    if len(net) < 20:
        return out
    sd = net.std()
    out["sharpe"] = float(np.sqrt(252) * net.mean() / sd) if sd > 0 else np.nan
    out["return_pa"] = float(net.mean() * 252)
    if gross is not None:
        gs = gross.std()
        out["sharpe_before_costs"] = float(np.sqrt(252) * gross.mean() / gs) if gs > 0 else np.nan
    # individual trades: blocks with position != 0; the return accrues on the following day in each case
    prev = np.concatenate([[0.0], pos[:-1]])
    start = np.where((pos != 0) & (prev != pos))[0]
    pnl, hold = [], []
    for st in start:
        end = st
        while end + 1 < len(pos) and pos[end + 1] == pos[st]:
            end += 1
        pnl.append(net[st:min(end + 2, len(net))].sum())
        hold.append(end - st + 1)
    if pnl:
        out["trades"] = len(pnl)
        out["hit_rate"] = float(np.mean(np.array(pnl) > 0))
        out["holding_days"] = float(np.mean(hold))
    return out


def mr_candidates(px: pd.DataFrame, vol: pd.DataFrame, names: pd.Series, groups: pd.Series | None,
                  pair_list: pd.DataFrame | None) -> list[tuple[str, str]]:
    """Admissible pairs of a selection period: liquid securities, same group or enough correlation."""
    lp, _, _ = prepare_window(px, vol, names, quiet=True)
    cols = list(lp.columns)
    ok = set(cols)
    if pair_list is not None:
        return [(a, b) for a, b in zip(pair_list["ticker_a"], pair_list["ticker_b"]) if a in ok and b in ok]
    if len(cols) < 2:
        return []
    corr = np.corrcoef(lp.diff().fillna(0.0).to_numpy(), rowvar=False)
    iu, ju = np.triu_indices(len(cols), k=1)
    c = corr[iu, ju]
    min_corr = STOCK_MIN_RET_CORR if MODE == "stocks" else MIN_RET_CORR
    keep = (c >= min_corr) & (c <= MAX_RET_CORR)
    if groups is not None:
        g = groups.reindex(cols).to_numpy(dtype=object)
        keep &= np.array([isinstance(x, str) and x != "" and x == y for x, y in zip(g[iu], g[ju])], dtype=bool)
    if MODE == "etf":                                       # (almost) identical indices: only price noise
        _, rel = monthly_te_stats(lp)
        keep &= ~(rel[iu, ju] < MIN_REL_TE)
    order = np.argsort(-c[keep])[:MR_MAX_PAIRS]
    return [(cols[i], cols[j]) for i, j in zip(iu[keep][order], ju[keep][order])]


def mr_top_development(pairs: pd.DataFrame, series: dict, idx: pd.DatetimeIndex, windows: list,
                       r_all: pd.Series) -> None:
    """The best pairs (by Sharpe over the whole period) year by year: return, Sharpe, trend.

    The pairs are selected in hindsight. For comparison, the average of these
    pairs and the average of all admissible pairs are shown below.
    """
    top = pairs[pairs["sharpe_after_costs"].notna()].head(MR_SHOW_TOP)
    if top.empty:
        return
    labels = [f"{t0:%Y}/{t1:%y}" for _, t0, t1 in windows]
    masks = [np.asarray((idx > t0) & (idx <= t1)) for _, t0, t1 in windows]
    in_all = np.asarray(idx > windows[0][1])
    n_recent = min(MR_TREND_YEARS, len(windows) - 1)
    recent = np.asarray(idx > windows[-n_recent][1]) if n_recent else np.zeros(len(idx), dtype=bool)
    early = in_all & ~recent
    half = np.asarray(idx > idx[-1] - pd.DateOffset(months=6))

    def sharpe(x: np.ndarray) -> float:
        sd = x.std() if len(x) > 20 else 0.0
        return float(np.sqrt(252) * x.mean() / sd) if sd > 0 else np.nan

    def describe(net: np.ndarray) -> dict:
        row = {f"return_{lab}": float(net[m].sum()) for lab, m in zip(labels, masks)}
        row.update({f"sharpe_{lab}": sharpe(net[m]) for lab, m in zip(labels, masks)})
        row["return_last_6_months"] = float(net[half].sum())
        if n_recent:
            a, b = sharpe(net[early]), sharpe(net[recent])
            row["sharpe_earlier"], row["sharpe_recent"] = a, b
            d = b - a
            row["development"] = ("" if d != d else "increasing" if d > MR_TREND_MIN_DIFF
                                  else "decreasing" if d < -MR_TREND_MIN_DIFF else "stable")
        return row

    nets, rows = {}, []
    for rank, r in enumerate(top.itertuples(index=False), 1):
        net = series[(r.etf_a, r.etf_b)]["net"]
        nets[f"{r.etf_a}/{r.etf_b}"] = net
        rows.append({"rank": rank, "etf_a": r.etf_a, "etf_b": r.etf_b, "name_a": r.name_a, "name_b": r.name_b,
                     "peer_group": r.peer_group, "sharpe_total": r.sharpe_after_costs, **describe(net)})
    top_net = np.mean(list(nets.values()), axis=0)
    all_net = r_all.reindex(idx).fillna(0.0).to_numpy()
    for label, net in ((f"Top {len(top)} combined", top_net), ("all pairs (average)", all_net)):
        rows.append({"rank": "", "etf_a": label, "etf_b": "", "name_a": "", "name_b": "", "peer_group": "",
                     "sharpe_total": sharpe(net[in_all]), **describe(net)})
    out = pd.DataFrame(rows)
    save_csv(out, "mr_top_pairs.csv")
    curves = pd.DataFrame({k: np.cumsum(np.where(in_all, v, 0.0)) for k, v in nets.items()}, index=idx)
    curves[f"Top {len(top)} combined"] = np.cumsum(np.where(in_all, top_net, 0.0))
    curves["all pairs"] = np.cumsum(np.where(in_all, all_net, 0.0))
    curves = curves[in_all]
    save_csv(curves.rename_axis("date"), "mr_top_pairs_history.csv", index=True)

    name = [f"{r['etf_a']}/{r['etf_b']}" if r["etf_b"] else r["etf_a"] for r in rows]
    width = max(len(x) for x in name) + 2
    print()
    print("-" * 78)
    print(f"DEVELOPMENT OF THE {len(top)} BEST PAIRS (by Sharpe over the whole period, selected in hindsight)")
    print("-" * 78)
    for title, key, fmt in (("Return per test year after costs (in %)", "return_", lambda v: f"{v * 100:+.1f}"),
                            ("Sharpe ratio per test year after costs", "sharpe_", lambda v: f"{v:.2f}")):
        cols = [f"{key}{lab}" for lab in labels]
        extra = (["return_last_6_months"] if key == "return_"
                 else ["sharpe_earlier", "sharpe_recent", "development"] if n_recent else [])
        head = labels + (["6 mon."] if key == "return_" else ["earlier", "recent", "Development"] if n_recent else [])
        print(f"\n{title}")
        print(" " * width + "".join(f"{h:>9}" for h in head[:len(cols) + 2])
              + (f"  {head[-1]}" if len(head) > len(cols) + 2 else ""))
        for i, r in enumerate(rows):
            if i == len(top):
                print(" " * width + "-" * (9 * (len(cols) + 2)))
            vals = [r[c] for c in cols + extra]
            cells = "".join(f"{(fmt(v) if v == v else '-'):>9}" for v in vals if not isinstance(v, str))
            print(f"{name[i]:<{width}}" + cells + "".join(f"  {v}" for v in vals if isinstance(v, str)))
    if n_recent:
        cnt = out["development"].iloc[:len(top)].value_counts()
        print(f"\n'recent' = the last {n_recent} test years (from {windows[-n_recent][1]:%d.%m.%Y}), 'earlier' = the years "
              f"before. Of the {len(top)} pairs: "
              + ", ".join(f"{cnt.get(k, 0)} {k}" for k in ("increasing", "stable", "decreasing")) + ".")
    print("Note: These pairs were selected BECAUSE they performed best over the whole period. Among thousands of\n"
          "pairs, the best ones look good even when only chance is at play. A pair that has recently been\n"
          "weakening is a warning sign; a pair that holds up is not yet proof.")
    mr_top_chart(curves, out, n_recent, windows)

    # cross-check without hindsight: the best pairs of the earlier years - how did they do afterwards?
    if n_recent and early.sum() > 250:
        cand = sorted(((sharpe(v["net"][early]), k) for k, v in series.items()
                       if np.count_nonzero(v["pos"][early]) >= 20), key=lambda x: -(x[0] if x[0] == x[0] else -9))
        cand = [(sh, k) for sh, k in cand if sh == sh][:MR_SHOW_TOP]
        if cand:
            yrs = recent.sum() / 252
            chk = pd.DataFrame([{"etf_a": k[0], "etf_b": k[1], "sharpe_earlier": sh,
                                 "sharpe_recent": sharpe(series[k]["net"][recent]),
                                 "return_pa_earlier": float(series[k]["net"][early].mean() * 252),
                                 "return_pa_recent": float(series[k]["net"][recent].sum() / yrs)} for sh, k in cand])
            both = np.mean([series[k]["net"] for _, k in cand], axis=0)
            save_csv(chk, "mr_top_pairs_cross_check.csv")
            print()
            print("-" * 78)
            print(f"CROSS-CHECK: the {len(chk)} best pairs up to {windows[-n_recent][1]:%d.%m.%Y} - and how they did afterwards")
            print("-" * 78)
            print(f"{'':<{width}}{'Sharpe':>9}{'Sharpe':>9}{'Return':>10}{'Return':>10}")
            print(f"{'':<{width}}{'earlier':>9}{'recent':>9}{'earlier':>10}{'recent':>10}   (return p. a. after costs)")
            for r in chk.itertuples(index=False):
                z = f"{r.sharpe_recent:.2f}" if r.sharpe_recent == r.sharpe_recent else "-"
                print(f"{r.etf_a + '/' + r.etf_b:<{width}}{r.sharpe_earlier:>9.2f}{z:>9}"
                      f"{r.return_pa_earlier * 100:>+9.1f}%{r.return_pa_recent * 100:>+9.1f}%")
            print(" " * width + "-" * 38)
            print(f"{'combined':<{width}}{sharpe(both[early]):>9.2f}{sharpe(both[recent]):>9.2f}"
                  f"{both[early].mean() * 252 * 100:>+9.1f}%{both[recent].sum() / yrs * 100:>+9.1f}%")
            print(f"{'all pairs (average)':<{width}}{sharpe(all_net[early]):>9.2f}{sharpe(all_net[recent]):>9.2f}"
                  f"{all_net[early].mean() * 252 * 100:>+9.1f}%{all_net[recent].sum() / yrs * 100:>+9.1f}%")
            print("These pairs were selected using only the earlier years; 'recent' is new data for them.\n"
                  "If little is left there, the earlier result was mainly chance.")


def mr_top_chart(curves: pd.DataFrame, table: pd.DataFrame, n_recent: int, windows: list) -> None:
    """Cumulative return of the best pairs as small individual charts -> mr_top_pairs.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        log("Chart mr_top_pairs.png skipped (matplotlib missing: pip install matplotlib)")
        return
    pair_cols = [c for c in curves.columns if c != "all pairs"]
    n = len(pair_cols)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    surface, ink, muted, blue, gray = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#a3a29c"
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow + 0.9), sharex=True, sharey=True,
                             squeeze=False, facecolor=surface)
    for k, ax in enumerate(axes.ravel()):
        ax.set_facecolor(surface)
        if k >= n:
            ax.set_visible(False)
            continue
        col = pair_cols[k]
        if n_recent:
            ax.axvspan(windows[-n_recent][1], curves.index[-1], color="#f0efec", lw=0)
        ax.axhline(0, color=muted, lw=0.8)
        ax.plot(curves.index, curves["all pairs"], color=gray, lw=1.2, label="all pairs (average)")
        ax.plot(curves.index, curves[col], color=blue, lw=2.0,
                label="Average of these pairs" if col.startswith("Top ") else "Pair")
        row = table.iloc[k]
        sub = ""
        if n_recent and row["sharpe_earlier"] == row["sharpe_earlier"] and row["sharpe_recent"] == row["sharpe_recent"]:
            sub = f"Sharpe {row['sharpe_earlier']:.1f} -> {row['sharpe_recent']:.1f}  ({row['development']})"
        ax.set_title((col if col.startswith("Top ") else f"{k + 1}. {col.replace('/', ' / ')}") + ("\n" + sub if sub else ""),
                     fontsize=10, color=ink, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f} %"))
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    if n_recent:
        from matplotlib.patches import Patch
        handles.append(Patch(color="#f0efec"))
        labels_.append(f"last {n_recent} test years ('recent')")
    fig.legend(handles, labels_, loc="upper right", ncol=len(handles), frameon=False, fontsize=9, labelcolor=muted)
    fig.suptitle("Cumulative return after costs: the best pairs over the whole period (selected in hindsight)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / "mr_top_pairs.png"
    try:
        fig.savefig(path, dpi=130, facecolor=surface)
    except PermissionError:
        path = OUT_DIR / f"mr_top_pairs_{time.strftime('%H%M%S')}.png"
        fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    log(f"Chart saved: {path.name}")


def run_mean_reversion(universe: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, aum: pd.Series,
                       aum_meta: pd.DataFrame | None = None, pair_list: pd.DataFrame | None = None) -> None:
    """Rolling test: select pairs based on the previous years, trade them in the following year (after costs)."""
    OUT_DIR.mkdir(exist_ok=True)
    stocks = MODE == "stocks"
    names = universe.set_index("ticker")["name"]
    sec_type = universe.set_index("ticker")["sec_type"] if "sec_type" in universe else pd.Series(dtype=object)
    groups = None
    if stocks and aum_meta is not None and "peer_group" in aum_meta and STOCK_GROUP_BY:
        groups = aum_meta["peer_group"].where(aum_meta["peer_group"].notna(), "")
    close = clean_prices(close)
    aum = aum.reindex(close.columns)
    min_size = MIN_MARKET_CAP_USD if stocks else MIN_AUM_USD
    size_ok = (aum >= min_size) | (sec_type.reindex(close.columns).eq("ETC") & (not AUM_FILTER_ETC))
    if KEEP_MISSING_AUM or pair_list is not None:
        size_ok = size_ok | aum.isna()
    close = close.loc[:, size_ok.reindex(close.columns, fill_value=False)]
    # breaks in the price data: Yahoo concatenates e.g. old and new shares after an insolvency. Such a
    # jump would be a huge fictitious profit in the test, therefore only the history from the last break counts.
    step = np.log(close.ffill()).diff().abs() > np.log(MR_MAX_DAILY_MOVE)
    breaks = []
    close = close.copy()
    for t in step.columns[step.any()]:
        d = step.index[step[t]][-1]
        breaks.append({"ticker": t, "name": names.get(t, ""), "last_break": d.date(),
                       "number_of_breaks": int(step[t].sum())})
        close.loc[close.index < d, t] = np.nan
    if breaks:
        save_csv(pd.DataFrame(breaks), "mr_price_breaks.csv")
        log(f"{len(breaks)} securities with a break in the price data (daily move above factor {MR_MAX_DAILY_MOVE:g}): "
            f"history before not used -> mr_price_breaks.csv")
    volume = volume.reindex(index=close.index, columns=close.columns)
    logp = np.log(close)
    idx = close.index
    first, last = idx[0], idx[-1]

    # test years backwards from the last date; before each, MR_FORMATION_YEARS selection period
    windows = []
    t1 = last
    while True:
        t0 = t1 - pd.DateOffset(years=MR_TEST_YEARS)
        f0 = t0 - pd.DateOffset(years=MR_FORMATION_YEARS)
        if f0 < first:
            break
        windows.append((f0, t0, t1))
        t1 = t0
    windows = windows[::-1]
    if not windows:
        raise SystemExit("Too little price history for the mean-reversion evaluation.")
    log(f"Mean-reversion evaluation: {len(windows)} test years, {windows[0][1]:%d.%m.%Y} - {last:%d.%m.%Y} "
        f"(selection in each case from the {MR_FORMATION_YEARS} years before)")

    series: dict[tuple[str, str], dict] = {}

    def pair_series(a: str, b: str) -> dict:
        if (a, b) not in series:
            series[(a, b)] = mr_pair_series(logp[a].to_numpy(), logp[b].to_numpy())
        return series[(a, b)]

    year_rows, pick_rows = [], []
    port_all, port_top, port_all_gross, port_top_gross = [], [], [], []
    for f0, t0, t1 in windows:
        in_form = (idx > f0) & (idx <= t0)
        in_test = (idx > t0) & (idx <= t1)
        cands = mr_candidates(close[in_form], volume[in_form], names, groups, pair_list)
        form = []
        for a, b in cands:
            ps_ = pair_series(a, b)
            st = mr_stats(ps_["net"][in_form], ps_["pos"][in_form])
            form.append((a, b, st["sharpe"], st["trades"]))
        form = pd.DataFrame(form, columns=["a", "b", "sharpe_prior_years", "trades_prior_years"])
        good = form[(form["trades_prior_years"] >= MR_MIN_TRADES) & (form["sharpe_prior_years"] > 0)]
        good = good.sort_values("sharpe_prior_years", ascending=False)
        used: dict[str, int] = {}
        top = []
        for _, r in good.iterrows():
            if used.get(r.a, 0) >= MR_MAX_PER_STOCK or used.get(r.b, 0) >= MR_MAX_PER_STOCK:
                continue
            top.append((r.a, r.b, r.sharpe_prior_years))
            used[r.a] = used.get(r.a, 0) + 1
            used[r.b] = used.get(r.b, 0) + 1
            if len(top) == MR_TOP_N:
                break
        test_idx = idx[in_test]
        all_ret = (np.mean([pair_series(a, b)["net"][in_test] for a, b in cands], axis=0)
                   if cands else np.zeros(in_test.sum()))
        top_ret = (np.mean([pair_series(a, b)["net"][in_test] for a, b, _ in top], axis=0)
                   if top else np.zeros(in_test.sum()))
        port_all.append(pd.Series(all_ret, index=test_idx))
        port_top.append(pd.Series(top_ret, index=test_idx))
        port_all_gross.append(pd.Series(np.mean([pair_series(a, b)["gross"][in_test] for a, b in cands], axis=0)
                                        if cands else np.zeros(in_test.sum()), index=test_idx))
        port_top_gross.append(pd.Series(np.mean([pair_series(a, b)["gross"][in_test] for a, b, _ in top], axis=0)
                                        if top else np.zeros(in_test.sum()), index=test_idx))
        for a, b, shf in top:
            ps_ = pair_series(a, b)
            st = mr_stats(ps_["net"][in_test], ps_["pos"][in_test], ps_["gross"][in_test])
            pick_rows.append({"test_year_from": t0.date(), "etf_a": a, "etf_b": b, "name_a": names.get(a, ""),
                              "name_b": names.get(b, ""), "sharpe_prior_years": shf,
                              "sharpe_test_year": st["sharpe"], "return_test_year": float(ps_["net"][in_test].sum()),
                              "trades_test_year": st["trades"], "hit_rate_test_year": st["hit_rate"]})
        sa, st_ = mr_stats(all_ret, np.zeros(len(all_ret))), mr_stats(top_ret, np.zeros(len(top_ret)))
        year_rows.append({"test_year_from": t0.date(), "test_year_to": t1.date(), "pairs_admissible": len(cands),
                          "pairs_selected": len(top),
                          "all_return": float(all_ret.sum()), "all_sharpe": sa["sharpe"],
                          "selection_return": float(top_ret.sum()), "selection_sharpe": st_["sharpe"]})
        log(f"  Test year from {t0:%d.%m.%Y}: {len(cands)} pairs admissible, {len(top)} selected | "
            f"all pairs {all_ret.sum():+.1%}, selection {top_ret.sum():+.1%} (after costs)")

    if not series:
        log("No pair was admissible in any test year (prices, history, size or turnover missing) - "
            "no evaluation possible.")
        return
    years = pd.DataFrame(year_rows)
    picks = pd.DataFrame(pick_rows)
    r_all, r_top = pd.concat(port_all), pd.concat(port_top)
    g_all, g_top = pd.concat(port_all_gross), pd.concat(port_top_gross)
    save_csv(years, "mr_years.csv")
    save_csv(picks, "mr_selection.csv")
    save_csv(pd.DataFrame({"all_pairs": r_all, "selection": r_top}).rename_axis("date"),
             "mr_portfolio_returns.csv", index=True)

    # metrics per pair over the whole test period (all pairs that were ever admissible)
    in_all = idx > windows[0][1]
    n_years = max(in_all.sum() / 252, 1e-9)
    rows = []
    for (a, b), ps_ in series.items():
        st = mr_stats(ps_["net"][in_all], ps_["pos"][in_all], ps_["gross"][in_all])
        yearly = [ps_["net"][(idx > t0) & (idx <= t1)].sum() for _, t0, t1 in windows]
        rows.append({"etf_a": a, "etf_b": b, "name_a": names.get(a, ""), "name_b": names.get(b, ""),
                     "peer_group": groups.get(a, "") if groups is not None else "",
                     "sharpe_after_costs": st["sharpe"], "sharpe_before_costs": st.get("sharpe_before_costs"),
                     "return_pa": st["return_pa"], "trades_per_year": st["trades"] / n_years,
                     "hit_rate": st["hit_rate"], "holding_days_mean": st["holding_days"],
                     "share_positive_years": float(np.mean(np.array(yearly) > 0)),
                     "times_selected": int(((picks["etf_a"] == a) & (picks["etf_b"] == b)).sum())
                     if len(picks) else 0})
    pairs = pd.DataFrame(rows).sort_values("sharpe_after_costs", ascending=False)
    save_csv(pairs, "mr_pairs.csv")

    def total(r: pd.Series) -> tuple[float, float, float, float]:
        sd = r.std()
        sh = np.sqrt(252) * r.mean() / sd if sd > 0 else np.nan
        eq = (1 + r).cumprod()
        return sh, r.mean() * 252, (eq / eq.cummax() - 1).min(), sh * np.sqrt(len(r) / 252)

    sh_a, ret_a, dd_a, t_a = total(r_all)
    sh_t, ret_t, dd_t, t_t = total(r_top)
    print()
    print("=" * 78)
    print(f"MEAN-REVERSION EVALUATION ({'US-' if MARKET == 'us' else ''}{'stocks' if stocks else 'ETFs'}"
          f"{', pair list' if pair_list is not None else ''})")
    print("=" * 78)
    print(f"Rule: spread = log(A) - log(B); entry at |z| > {MR_ENTRY_Z} relative to the {MR_LOOKBACK}-day mean, "
          f"exit at the mean\n       or after {MR_MAX_HOLD} days. Costs {MR_COST_BPS} basis points per purchase/sale, "
          f"borrowing fee {MR_BORROW_FEE:.0%} p. a.")
    print(f"Selection: per test year the {MR_TOP_N} pairs with the best result of the {MR_FORMATION_YEARS} previous years.\n")
    view = years.copy()
    for c in ("all_return", "selection_return"):
        view[c] = (view[c] * 100).map(lambda v: f"{v:+.1f} %")
    print(view.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nTotal over {len(windows)} test years (after costs, on new data):")
    print(f"  all admissible pairs  : Sharpe {sh_a:5.2f} | return p. a. {ret_a:+.1%} | max. loss {dd_a:.1%} | "
          f"t-value {t_a:.1f} | before costs {g_all.mean() * 252:+.1%} p. a.")
    print(f"  selection by prior yrs: Sharpe {sh_t:5.2f} | return p. a. {ret_t:+.1%} | max. loss {dd_t:.1%} | "
          f"t-value {t_t:.1f} | before costs {g_top.mean() * 252:+.1%} p. a.")
    print("  (t-value above about 2 = hardly explainable by chance; returns relative to the capital employed)")
    if len(pairs):
        pos_share = float((pairs["sharpe_after_costs"] > 0).mean())
        gross_share = float((pairs["sharpe_before_costs"] > 0).mean())
        print(f"\nIndividual pairs ({len(pairs)}): {gross_share:.0%} positive before costs, {pos_share:.0%} after costs; "
              f"mean hit rate {pairs['hit_rate'].mean():.0%}, "
              f"holding period {pairs['holding_days_mean'].mean():.0f} days")
    mr_top_development(pairs, series, idx, windows, r_all)
    print(f"\nResults: {OUT_DIR}  (mr_years.csv, mr_selection.csv, mr_pairs.csv, mr_portfolio_returns.csv,\n"
          f"            mr_top_pairs.csv, mr_top_pairs_history.csv, mr_top_pairs.png, mr_top_pairs_cross_check.csv)")


# ------------------------------------------------------------------ Demo data
def make_demo_data(seed: int = 7):
    """Synthetic 'ETFs' for testing the pipeline:
    CA/CB  6 real cointegrated pairs (should be found)
    RW     30 merely correlated securities (should NOT be found)
    SA/SB  4 cointegrated pairs with a fund volume that is too small (AUM filter)
    NQ     3 ETFs on the same index with slight price noise (tracking error filter)
    ST     1 rarely traded twin of CB0 with stale prices (liquidity filter)
    JX/JY  same reference index according to metadata, but noisy (index filter)
    DE0    copy of CA1 with a faulty price block (data error detection)
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * HISTORY_YEARS)
    n = len(idx)
    market = np.cumsum(rng.normal(0.0003, 0.010, n))
    series, names, aum, keys = {}, {}, {}, {}

    def add(t, logp, name, a, key=None):
        a = a * (1 + 0.0007 * len(series))                  # unique fund volume per demo ETF
        series[t], names[t], aum[t], keys[t] = np.exp(logp), name, a, key

    def ou(half_life, sigma):
        x = np.zeros(n)
        theta = np.log(2) / half_life
        for t in range(1, n):
            x[t] = x[t - 1] * (1 - theta) + rng.normal(0, sigma)
        return x

    for g in range(6):                                   # cointegrated pairs
        base = market + np.cumsum(rng.normal(0, 0.007, n))
        beta = rng.uniform(0.7, 1.4)
        add(f"CB{g}.DE", base + rng.normal(0, 0.0005, n), f"DEMO COINT B{g}", 5e8)
        add(f"CA{g}.DE", 0.3 + beta * base + ou(rng.uniform(8, 30), 0.004), f"DEMO COINT A{g}", 5e8)
    for g in range(30):                                   # only correlated, not cointegrated
        add(f"RW{g}.DE", market + np.cumsum(rng.normal(0, 0.006, n)), f"DEMO RANDOM {g}", 3e8)
    for g in range(4):                                    # cointegrated, but too small (AUM)
        base = market + np.cumsum(rng.normal(0, 0.007, n))
        add(f"SB{g}.DE", base, f"DEMO SMALL B{g}", 4e7)
        add(f"SA{g}.DE", base + ou(14, 0.004), f"DEMO SMALL A{g}", 4e7)
    nq = market + np.cumsum(rng.normal(0.0002, 0.008, n))  # same index, 3 providers
    for g in range(3):
        add(f"NQ{g}.DE", 0.5 * g + nq + ou(6, 0.0015), f"DEMO NASDAQ ETF {g}", 2e9)
    stale = np.log(series["CB0.DE"]) + ou(10, 0.003)       # rarely traded: price only on 40 % of the days
    stale[rng.random(n) > 0.4] = np.nan
    add("ST0.DE", pd.Series(stale).ffill().bfill().to_numpy(), "DEMO ILLIQUID", 3e8)
    bad = np.log(series["CA1.DE"]).copy()                   # data error: price block +13 % for 20 days
    bad[n - 120:n - 100] += 0.13
    add("DE0.DE", bad, "DEMO DATA ERROR", 5e8)
    jp = market + np.cumsum(rng.normal(0, 0.008, n))       # same index according to Yahoo, noisy
    add("JX0.DE", jp + ou(12, 0.006), "DEMO JAPAN A", 1e9, "MSCIJAPAN")
    add("JY0.DE", jp + ou(12, 0.006), "DEMO JAPAN B", 1e9, "MSCIJAPAN")

    close = pd.DataFrame(series, index=idx)
    close.iloc[: 252 * 4, close.columns.get_loc("RW29.DE")] = np.nan   # ETF launched later
    volume = (1.2 * max(min_turnover(), 250_000) / close).round()      # turnover just above the threshold
    unchanged = close["ST0.DE"].diff() == 0
    volume.loc[unchanged, "ST0.DE"] = 0
    volume["ST0.DE"] = volume["ST0.DE"] * 0.2
    universe = pd.DataFrame({"ticker": list(names), "name": list(names.values()), "isin": "", "sec_type": "ETF",
                             "segment": ""})
    aum_s = pd.Series(aum, dtype=float)
    meta = pd.DataFrame({"total_assets": aum_s, "fund_ccy": "EUR", "aum_source": "demo",
                         "aum_usd": aum_s, "benchmark": pd.Series(keys), "index_key": pd.Series(keys),
                         "index_source": "demo"})
    if MODE == "stocks":                                    # demo industries: pair partners in the same industry
        tick = list(names)

        def demo_group(t: str) -> str:
            digits = "".join(ch for ch in t.split(".")[0] if ch.isdigit()) or "0"
            if t[:2] in ("CA", "CB", "SA", "SB"):
                return f"Demo industry {int(digits) % 3 + 1}"
            return f"Demo industry {int(digits) % 5 + 1}"

        meta = pd.DataFrame({"market_cap": aum_s * 3, "currency": "EUR", "sector": "Demo sector",
                             "industry": [demo_group(t) for t in tick], "long_name": list(names.values()),
                             "aum_usd": aum_s * 3}, index=tick)
        meta["peer_group"] = meta["industry"] if STOCK_GROUP_BY else ""
        universe["sec_type"] = "Stock"
        aum_s = meta["aum_usd"]
    return universe, close, volume, aum_s, meta


# ------------------------------------------------------------------ main
def main() -> None:
    global OUT_DIR, MODE, MARKET, MR_COST_BPS
    ap = argparse.ArgumentParser(description="Screener for cointegrated pairs (stocks or ETFs, Xetra or USA)")
    ap.add_argument("--etf", action="store_true", help="ETF/ETC mode instead of stocks")
    ap.add_argument("--us", action="store_true", help="US market (Nasdaq, NYSE etc.) instead of Xetra")
    ap.add_argument("--demo", action="store_true", help="trial run with synthetic data")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and reload everything")
    ap.add_argument("--pairs", nargs="?", const="", default=None, metavar="FILE",
                    help="test only the pairs from FILE (default: pairs_stocks.csv, with --etf pairs.csv)")
    ap.add_argument("--two-stage", dest="two_stage", action="store_true",
                    help="select candidates in the earlier period, confirm them in the later one")
    ap.add_argument("--mean-reversion", "--mr", dest="mean_reversion", action="store_true",
                    help="additional evaluation: return to the moving average, with costs, tested on a rolling basis")
    args = ap.parse_args()

    MODE = "etf" if args.etf else "stocks"
    MARKET = "us"
    stocks, us = MODE == "stocks", MARKET == "us"
    OUT_DIR = BASE_DIR / ((f"output_us_{MODE}") if us else ("output_stocks" if stocks else "output"))
    if us:
        MR_COST_BPS = US_MR_COST_BPS
    log(f"Mode: {'US-' if us else 'Xetra-'}{'stocks' if stocks else ('ETFs' if us else 'ETFs/ETCs')}")

    t_start = time.time()
    pair_list = None
    if args.pairs is not None:
        fname = args.pairs or ((f"pairs_us_{MODE}.csv") if us
                               else ("pairs_stocks.csv" if stocks else "pairs.csv"))
        path = Path(fname) if Path(fname).is_absolute() else BASE_DIR / fname
        if not path.exists():
            raise SystemExit(f"Pair list {path} not found")
        pair_list = load_pair_list(path)
        log(f"Pair list {path.name}: {len(pair_list)} pairs")

    if args.demo:
        OUT_DIR = BASE_DIR / ((f"output_demo_us_{MODE}") if us
                              else ("output_demo_stocks" if stocks else "output_demo"))
        log(f"DEMO mode with synthetic data (results in {OUT_DIR.name})")
        universe, close, volume, aum, aum_meta = make_demo_data()
    else:
        universe = load_universe(args.refresh)
        tickers = universe["ticker"].tolist()
        if pair_list is not None:                          # tickers of the list that are not in the universe
            extra = sorted((set(pair_list["ticker_a"]) | set(pair_list["ticker_b"])) - set(tickers))
            if extra:
                log(f"{len(extra)} tickers from the pair list are not in the {'US' if us else 'Xetra'} list: "
                    f"{', '.join(extra)} (are tried anyway)")
                universe = pd.concat([universe, pd.DataFrame({"ticker": extra, "name": "", "isin": "",
                                                              "sec_type": "Stock" if stocks else "ETF",
                                                              "segment": ""})], ignore_index=True)
                tickers += extra
        if us and pair_list is not None:                   # pair list: load only their securities (fast)
            tickers = sorted(set(pair_list["ticker_a"]) | set(pair_list["ticker_b"]))
        elif us:                                           # long history only for sufficiently liquid securities
            tickers = us_prescreen(tickers, args.refresh)
        close, volume = load_prices(tickers, args.refresh)
        min_rows = int(252 * FORMATION_YEARS * MIN_COVERAGE)
        candidates = close.columns[close.notna().sum() >= min_rows].tolist()
        if stocks:
            aum, aum_meta = load_stock_info(candidates, args.refresh)
        else:
            aum, aum_meta = load_aum_usd(candidates, args.refresh)

    if args.mean_reversion:
        run_mean_reversion(universe, close, volume, aum, aum_meta, pair_list)
    else:
        run_screen(universe, close, volume, aum, aum_meta, pair_list, args.two_stage)
    log(f"Finished after {(time.time() - t_start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
