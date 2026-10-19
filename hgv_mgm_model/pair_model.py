"""
pair_model.py - mean-reversion model for a basket of stock pairs, with protocol mode for paper trading

The model trades the spread of each pair according to its z-score. All pairs use THE SAME settings;
the capital is distributed evenly across the pairs. The evaluation takes place in separate periods:

    Training:   the TRAIN_YEARS years before the test phase   (the settings are chosen here)
    Test:       the last TEST_YEARS years up to today         (control; not really new for the pair selection)
    Protocol:   from the start day of the protocol            (paper trading - the only really new data)

Usage
-----
    python pair_model.py                  # backtest: training and test, basket and individual pairs
    python pair_model.py --grid           # additionally: Sharpe of the basket in TRAINING for several entry/exit values
    python pair_model.py --protocol      # protocol mode: record the daily status, orders for tomorrow, result since start
    python pair_model.py --protocol-new  # start the protocol anew (the old one is moved to an archive folder)
    python pair_model.py --refresh        # reload prices from Yahoo (otherwise at most every 12 hours)

Trading rule (per pair A / B)
-----------------------------
    Spread  = log(price A) - HEDGE_RATIO * log(price B)
    z-score = (spread - mean) / standard deviation
    z < -ENTRY_Z  ->  spread is unusually low:   buy A, short-sell B  ("long spread")
    z > +ENTRY_Z  ->  spread is unusually high:  short-sell A, buy B  ("short spread")
    Exit as soon as one of the exit rules applies (EXIT_Z, MAX_HOLD_DAYS, STOP_Z - see below).

The signal arises with the closing price of a day. Depending on EXECUTION, trading takes place at this closing price,
at the opening of the next trading day or at the next closing price; the position only earns the
price move after its execution (no look into the future). With "next_open", each day is split into the
move overnight (close -> opening) and during the day (opening -> close).
Returns of a pair refer to the capital allocated to the pair (long and short side together),
returns of the basket to the total capital. The Sharpe ratio is annualized (square root of 252), without deducting
a risk-free rate, because the purchase and the short sale finance each other.

Simulated account and three profiles
------------------------------------
In addition to the returns of the signals, the script calculates accounts in USD (CAPITAL): Only whole shares are traded.
A new pair gets  target leverage x equity / number of pairs, half each on the long and the
short side, calculated with the equity and closing prices of the signal day; the share quantities remain unchanged until the
exit. Per order, the broker's fee is charged (Interactive Brokers: per share, with a minimum and
maximum amount) and additionally slippage (COST_BPS). On top of that come the borrowing fee on short sales and credit interest on
the part of the long side that exceeds the equity (FINANCING_BASIS = "credit") or on the whole long side.

Backtest and protocol calculate three profiles, each as its own account:
    No leverage       leverage 1.
    Private           highest leverage of a normal margin account for US stocks (RETAIL_LEVERAGE = 2). Margin rules:
                     A new position is rejected if the equity afterwards covers less than MARGIN_INITIAL of all
                     open positions; if the coverage falls below MARGIN_MAINTENANCE, the broker closes everything
                     compulsorily (margin call).
    Institutional   trading with the capital of a trading firm, for which these rules do not apply. Target leverage =
                     KELLY_FRACTION x Kelly leverage from the training, at most INSTITUTIONAL_MAX_LEVERAGE. The own
                     capital bears all losses. The firm closes the account at a total loss of 10 %
                     (INSTITUTIONAL_STOP) or a loss of 5 % in one day (INSTITUTIONAL_DAILY_LOSS) - at the latest when
                     the capital is used up. After that, the model no longer trades in this profile.

Kelly formula and stress test
-----------------------------
Optimal leverage f = mean return / variance of the return, calculated ONLY from the training phase
of the basket (in the protocol: from the TRAIN_YEARS years before the start day, recorded in leverage.json). Because the
pairs offset each other, the basket fluctuates little, and the formula yields very high leverage; it assumes
normally distributed returns and known metrics. The stress test of the backtest runs the account through with a
series of leverage values without any limit and shows from which leverage a total loss would have occurred. The
cost scale shows the accounts at different levels of slippage.

Protocol mode (--protocol)
---------------------------
On the first call, the start day and the settings are frozen (protocol/start.json). Each pair starts
without a position and only enters after its z-score has once been in the normal range after the start day
(|z| < ENTRY_Z) - so there is no entry in the middle of an ongoing move. Each further call
    - checks whether the settings still correspond to the frozen state (otherwise abort),
    - appends the daily status of each pair to protocol/protocol.csv (existing rows are never changed;
      days that are entered only later are marked as "added_later"),
    - shows the orders for the next trading day,
    - writes the result since the start day to a separate file (protocol/results.csv).
protocol/executions.csv contains every order of the model with the model price. Anyone who executes the orders in a
demo account enters the actual price there in the column "real_price"; from this the script calculates
the measured deviation (slippage) in basis points. If a trade was not possible in the demo account (e.g. stock
not borrowable), enter "no" there in the column "executable" for the entry orders: The protocol then additionally shows the
account without these trades. Best called daily, e.g. in the morning.
Leverage: orders, share quantities and the main result apply to the account without leverage. Additionally, the
protocol calculates the profiles "Private" and "Institutional" as separate accounts - exactly, with larger share quantities,
fees, credit interest and the rules of the profile, not by multiplying the return.
A US trading day that is still running is not used (preliminary prices), even if the script runs in the afternoon.
Note: Yahoo adjusts prices retroactively for dividends. The model price of an order is recorded when it is first
entered; if a day is entered only after an ex-dividend date, it can therefore be slightly
below the price traded at the time. This is another reason why the daily call is worthwhile.

Files
-----
    output/     metrics.csv (account per phase), pairs.csv (per pair), account_trades.csv (trades in the account, in USD),
                trades.csv (signals, in %), daily_values.csv, cost_scale.csv, leverage_stress_test.csv,
                margin_calls.csv (if any occurred), model.png
    protocol/  start.json, leverage.json, protocol.csv, executions.csv, results.csv, results_pairs.csv,
                results_leverage.csv, trades.csv, account_trades.csv
    (CSV with ";" and decimal comma for Excel)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================================================
# SETTINGS
# ======================================================================================================
# The 15 best pairs from pair_screener/output_us_aktien/mr_pairs.csv (ranking of 04.10.2026). At later points in time, a separate test for potentially cointegrated pairs must accordingly be carried out with the pair_screener. 
# The list is entered as fixed on purpose: It should not change during the paper trading.
PAIRS = [
    ("HGV", "MGM"),     # Resorts & Casinos
    ("ALGM", "STM"),    # Semiconductors
    ("COF", "OMF"),     # Credit Services
    ("FHB", "FRME"),    # Banks - Regional
    ("MTDR", "OVV"),    # Oil & Gas E&P
    ("AX", "PNC"),      # Banks - Regional
    ("MCO", "SPGI"),    # Financial Data & Stock Exchanges
    ("CRL", "RVTY"),    # Diagnostics & Research
    ("BZH", "MHO"),     # Residential Construction
    ("NWN", "SR"),      # Utilities - Regulated Gas
    ("FRO", "INSW"),    # Oil & Gas Midstream (Tanker)
    ("BYD", "HGV"),     # Resorts & Casinos
    ("CRBG", "EQH"),    # Asset Management
    ("FBP", "OFG"),     # Banks - Regional
    ("EWBC", "PNFP"),   # Banks - Regional
]

# ---- Periods
TEST_YEARS = 1.0            # test phase: this many years up to the last price date
TRAIN_YEARS = 3.0           # training phase: this many years directly before

# ---- z-score
Z_MODE = "rolling"          # "rolling"  = mean and std from the last LOOKBACK trading days (moving)
                            # "training" = mean and std fixed from the training phase (as in Chan's examples)
LOOKBACK = 60               # window length in trading days (only for Z_MODE = "rolling")

# ---- Entry
ENTRY_Z = 1.25              # entry when the z-score is above +ENTRY_Z or below -ENTRY_Z

# ---- Exit (each rule can be set individually; whichever occurs first applies)
EXIT_Z = 0.25               # exit when the z-score has returned to within EXIT_Z of the mean:
                            #    0.0 = at the mean,  0.5 = already earlier (at +/-0.5),
                            #   -0.5 = only after overshooting to the other side
MAX_HOLD_DAYS = 40          # time stop: close after this many trading days at the latest (None = off)
STOP_Z = None               # loss stop: close if the z-score runs away further than STOP_Z
                            #   (e.g. 4.0; None = off). After that no new entry until |z| < ENTRY_Z again.

# ---- Hedge ratio
HEDGE_RATIO = 1.0           # 1.0 = same amount in both stocks (dollar-neutral)
                            # "training" = estimated per pair by regressing log(A) on log(B) in the training phase

# ---- Simulated account
CAPITAL = 50_000            # equity in USD at the beginning of each phase (training, test, protocol).
                            #   Only whole shares are traded; if the amount per stock is not enough for one
                            #   share, the trade is dropped.

# ---- Fees: Interactive Brokers, IBKR Pro, US stocks, "Fixed" plan (as of October 2026)
COMMISSION_PER_SHARE = 0.005    # USD per share
COMMISSION_MIN = 1.00           # minimum fee per order in USD
COMMISSION_MAX_PCT = 0.01       # at most 1 % of the order value
                            #   "Tiered" plan (up to 300,000 shares/month): 0.0035 per share, at least 0.35 per order,
                            #   plus exchange and clearing fees, which are not modeled here.

# ---- Three account profiles (backtest and protocol calculate each as its own account with CAPITAL)
#   1. No leverage:     leverage 1, no margin issues.
#   2. Private:         the highest leverage of a normal margin account for US stocks (Reg T).
#   3. Institutional:   trading with the capital of a trading firm (proprietary trading firm), for which Reg T does not apply.
#                       Target leverage = KELLY_FRACTION x Kelly leverage from the training, at most INSTITUTIONAL_MAX_LEVERAGE.
RETAIL_LEVERAGE = 2.0       # target leverage in the profile "Private"
MARGIN_INITIAL = 0.50       # Private: A new position is only opened if the equity afterwards covers at least
                            #   this share of all open positions (long plus short side).
                            #   0.50 = Reg T -> at most leverage 2. None = do not check.
MARGIN_MAINTENANCE = 0.30   # Private: If the equity falls below this share of the open positions, the
                            #   broker closes everything compulsorily (margin call). Usual: 25 % for purchases, 30 % for short sales.
KELLY_FRACTION = 0.25       # Institutional: share of the Kelly leverage (1.0 = full Kelly, 0.5 = half, 0.25 = quarter)
INSTITUTIONAL_MAX_LEVERAGE = 6   # Institutional: highest leverage that the trading firm provides. New positions are
                            #   only opened up to this multiple of the own capital. The own capital bears
                            #   all losses; once it is used up, the firm closes the account (TOTALVERLUST).
INSTITUTIONAL_STOP = 0.90   # Institutional, total loss limit: The firm closes the account if the capital falls below
                            #   this share of the starting capital. 0.90 = at most 10 % loss - the value most frequently
                            #   cited by trading firms (6 to 12 % is usual). None = off.
INSTITUTIONAL_DAILY_LOSS = 0.05     # Institutional, daily loss limit: The account is closed if the loss
                            #   of one day reaches this share of the account balance of the previous evening. 0.05 = 5 %
                            #   (usual: 4 to 6 %). None = off. The total loss limit applies fixed from the
                            #   starting capital and is not trailed.
                            #   The contract of the respective trading firm is decisive - check there.
STRESS_LEVERAGE = [1, 2, 4, 8, 16, 32, 64, 128]   # leverage for the stress test "would a total loss have occurred?"
COST_LADDER_BPS = [0, 5, 10, 20, 30]                # slippage levels for the cost scale

# ---- Costs and execution
COST_BPS = 5                # slippage per purchase or sale in basis points of the order value (distance between
                            #   model price and actual price). The broker's fees are added in the account.
BORROW_FEE = 0.01           # borrowing fee p. a. on the short side
FINANCING_RATE = 0.0538     # interest p. a. for the leverage (securities loan). 5.38 % = Interactive Brokers (IBKR Pro),
                            #   USD loan up to 100,000, as of October 2026: reference rate 3.88 % + 1.5 % markup.
                            #   Many German brokers and CFD providers are higher - enter the rate of your own broker.
FINANCING_BASIS = "credit"  # what the interest is charged on while positions are open:
                            #   "credit" = only on the part of the long side that exceeds the equity
                            #              (securities loan / margin account)
                            #   "full"   = on the whole long side (as usual with CFDs)
SHORT_PROCEEDS_RATE = 0.0   # interest p. a. that the broker credits on the proceeds of the short sales
                            #   (private clients usually get nothing -> 0)
EXECUTION = "next_open"     # when trading takes place after a signal (closing price of the day):
                            #   "close"      = at the closing price of the same day (slightly flattering: the closing price
                            #                  is only fixed when trading is over)
                            #   "next_open"  = at the opening price of the next trading day (order overnight)
                            #   "next_close" = at the closing price of the next trading day (strictest variant)

# ---- Grid for --grid (evaluated ONLY in the training phase, for the whole basket)
GRID_ENTRY_Z = [1.0, 1.25, 1.5, 2.0, 2.5]
GRID_EXIT_Z = [-0.5, 0.0, 0.25, 0.5, 1.0]
# ======================================================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"
PROTOCOL_DIR = BASE_DIR / "protocol"
HISTORY_YEARS = 12          # this much history is loaded from Yahoo (also enough for longer phases)
MIN_DAYS_FOR_SHARPE = 60    # in the protocol, a Sharpe ratio is only shown from this many trading days on


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def pair_name(a: str, b: str) -> str:
    return f"{a}/{b}"


def tickers() -> list[str]:
    return list(dict.fromkeys(t for pair in PAIRS for t in pair))


def save_csv(df: pd.DataFrame, name: str, index: bool = False, folder: Path | None = None) -> bool:
    """Write a CSV for Excel. Returns False if the file was locked (then under a different name)."""
    folder = folder or OUT_DIR
    folder.mkdir(exist_ok=True)
    path = folder / name
    try:
        df.to_csv(path, sep=";", decimal=",", index=index, encoding="utf-8-sig")
        return True
    except PermissionError:                                 # the file is open in Excel, for example
        alt = folder / f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}"
        df.to_csv(alt, sep=";", decimal=",", index=index, encoding="utf-8-sig")
        log(f"{name} is locked (open in Excel?) - saved as {alt.name}")
        return False


# ------------------------------------------------------------------------------------------ Prices
def drop_unfinished_day(px: pd.DataFrame) -> pd.DataFrame:
    """Remove the current US trading day: During trading, Yahoo provides a preliminary 'closing price'
    for today. A signal from it would be wrong and would be written into the protocol permanently."""
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        today, done = now.normalize().tz_localize(None), now.hour * 60 + now.minute >= 16 * 60 + 20
    except Exception:                                       # no time zone data: approximation via UTC
        now = pd.Timestamp.now(tz="UTC")
        today, done = now.normalize().tz_localize(None), now.hour * 60 + now.minute >= 21 * 60 + 20
    if len(px) and px.index[-1] >= today and not done:
        log(f"The US trading day {px.index[-1]:%d.%m.%Y} is still running - it is only taken into account after the close of trading "
            f"(approx. 22:20 German time)")
        px = px[px.index < today]
    return px


def load_prices(refresh: bool) -> pd.DataFrame:
    """Daily prices of all stocks, adjusted for dividends and splits.

    Columns: ticker = closing price, "<Ticker>_open" = opening price (for EXECUTION = "next_open").
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "prices.csv"
    tk = tickers()
    open_cols = [f"{t}_open" for t in tk]
    need_open = EXECUTION == "next_open"

    def read_cache() -> pd.DataFrame:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    def usable(px: pd.DataFrame) -> bool:                   # all stocks available (and opening prices, if needed)?
        return set(tk) <= set(px.columns) and (not need_open or set(open_cols) <= set(px.columns))

    fresh = cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600
    if fresh and not refresh:
        px = read_cache()
        if usable(px):
            log(f"Prices from {cache.name} (up to {px.index[-1]:%d.%m.%Y})")
            return px
        log("Stored prices are incomplete (new stocks or missing opening prices) - reloading")
    try:
        import yfinance as yf

        start = (pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
        log(f"Loading prices for {len(tk)} stocks from Yahoo Finance ...")
        data = yf.download(tk, start=start, auto_adjust=True, progress=False)
        px = data["Close"].reindex(columns=tk).copy()
        px[open_cols] = data["Open"].reindex(columns=tk).to_numpy()
        px = px.dropna(how="all")
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = drop_unfinished_day(px)
        missing = [t for t in tk if px[t].notna().sum() == 0]
        if len(missing) == len(tk):
            raise RuntimeError("no prices received")
        if missing:
            log(f"No prices for: {', '.join(missing)} - pairs with these stocks are skipped")
        px.rename_axis("date").to_csv(cache)
        log(f"Prices loaded (up to {px.index[-1]:%d.%m.%Y})")
        return px
    except Exception as e:
        if cache.exists() and usable(read_cache()):
            px = read_cache()
            log(f"Yahoo not reachable ({type(e).__name__}) - using stored prices up to {px.index[-1]:%d.%m.%Y}")
            return px
        raise SystemExit(f"Prices could not be loaded ({type(e).__name__}: {e}). "
                         f"Is yfinance installed (pip install yfinance) and is there an internet connection?")


# ------------------------------------------------------------------------------------------ Model
def build_signals(px: pd.DataFrame, a: str, b: str, train: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Spread and z-score of a pair for the whole history. train = mask of the training days (on px.index).

    The result has the same index as px; on days without a price for one of the two stocks there is NaN.
    """
    cols = [a, b] + ([f"{a}_open", f"{b}_open"] if f"{a}_open" in px.columns and f"{b}_open" in px.columns else [])
    sub = px[cols].dropna()
    tr = pd.Series(train, index=px.index).reindex(sub.index).to_numpy()
    la, lb = np.log(sub[a]), np.log(sub[b])
    if HEDGE_RATIO == "training":                           # regression only with training data
        if tr.sum() < 60:
            raise SystemExit(f'{pair_name(a, b)}: too little training data for HEDGE_RATIO = "training"')
        beta = float(np.polyfit(lb[tr], la[tr], 1)[0])
    else:
        beta = float(HEDGE_RATIO)
    spread = la - beta * lb
    if Z_MODE == "training":                                # fixed values from the training phase
        z = (spread - spread[tr].mean()) / spread[tr].std()
    elif Z_MODE == "rolling":                               # moving, uses only past days
        roll = spread.rolling(LOOKBACK, min_periods=LOOKBACK)
        z = (spread - roll.mean()) / roll.std()
    else:
        raise SystemExit('Z_MODE must be "rolling" or "training"')
    w_a, w_b = 1 / (1 + abs(beta)), beta / (1 + abs(beta))  # capital shares of the two sides
    out = pd.DataFrame({"price_a": sub[a], "price_b": sub[b], "spread": spread, "z": z,
                        "spread_return": w_a * sub[a].pct_change() - w_b * sub[b].pct_change()})
    if len(cols) == 4:                                      # decomposition for the execution at the opening
        oa, ob = sub[f"{a}_open"], sub[f"{b}_open"]
        out["open_a"], out["open_b"] = oa, ob
        out["return_night"] = w_a * (oa / sub[a].shift(1) - 1) - w_b * (ob / sub[b].shift(1) - 1)
        out["return_day"] = w_a * (sub[a] / oa - 1) - w_b * (sub[b] / ob - 1)
    return out.reindex(px.index), beta


def simulate(sig: pd.DataFrame, beta: float, entry_z: float, exit_z: float,
             start_blocked: bool = False, close_at_end: bool = True) -> pd.DataFrame:
    """Trades a pair from the beginning (without a position) to the end of a period. Returns daily values.

    start_blocked: only enter after |z| has once been below entry_z (protocol: no entry into an
                   ongoing move). close_at_end: apply closing costs for a position that is open at the end.
    """
    z = sig["z"].to_numpy()
    n = len(z)
    signal = np.zeros(n)                                    # desired position after the closing price of the day
    reason = np.array([""] * n, dtype=object)               # reason for an exit on this day
    cur, held, blocked = 0, 0, bool(start_blocked)
    for t in range(n):
        zt = z[t]
        if zt != zt:                                        # no z-score (too little history): no position
            if cur != 0:
                reason[t] = "no data"
            cur, held = 0, 0
        elif cur == 0:
            if blocked and abs(zt) < entry_z:               # after a stop / protocol start only again in the normal range
                blocked = False
            if not blocked:
                if zt > entry_z:
                    cur, held = -1, 0                       # spread too high -> short spread
                elif zt < -entry_z:
                    cur, held = 1, 0                        # spread too low -> long spread
        else:
            held += 1
            if STOP_Z is not None and ((cur == 1 and zt <= -STOP_Z) or (cur == -1 and zt >= STOP_Z)):
                reason[t], cur, held, blocked = "Stop", 0, 0, True
            elif (cur == 1 and zt >= -exit_z) or (cur == -1 and zt <= exit_z):
                reason[t], cur, held = "z-score", 0, 0
            elif MAX_HOLD_DAYS is not None and held >= MAX_HOLD_DAYS:
                reason[t], cur, held = "time", 0, 0
        signal[t] = cur
    wanted = signal.copy()                                  # signal after the closing price, before the execution
    if EXECUTION in ("next_open", "next_close"):            # execution only on the next trading day
        signal = np.concatenate([[0.0], signal[:-1]])
        reason = np.concatenate([np.array([""], dtype=object), reason[:-1]])
    # signal = position at the end of the day; prev = position that was held overnight
    prev = np.concatenate([[0.0], signal[:-1]])
    full_day = np.nan_to_num(sig["spread_return"].to_numpy())
    if EXECUTION == "next_open":
        # trading at the opening: overnight the old position still earns, during the day already the new one
        night, day = np.nan_to_num(sig["return_night"].to_numpy()), np.nan_to_num(sig["return_day"].to_numpy())
        gross = np.where(prev == signal, prev * full_day, prev * night + signal * day)
    else:
        gross = prev * full_day                             # trading at the closing price: return from the following day
    cost = np.abs(signal - prev) * COST_BPS / 1e4
    short_share = np.where(prev > 0, abs(beta), 1.0) / (1 + abs(beta))      # long spread: B is sold short
    borrow = np.abs(prev) * short_share * BORROW_FEE / 252
    if n and signal[-1] != 0:                               # position open at the end
        reason[-1] = "open"
        if close_at_end:
            cost[-1] += abs(signal[-1]) * COST_BPS / 1e4
    long_share = np.abs(prev) * np.where(prev > 0, 1.0, abs(beta)) / (1 + abs(beta))    # long side: A or B
    return pd.DataFrame({"signal": wanted, "position": signal, "gross": gross, "costs": cost + borrow,
                         "net": gross - cost - borrow, "exit_reason": reason,
                         "long_side": long_share, "short_side": np.abs(prev) * short_share}, index=sig.index)


def direction(pos: float, a: str, b: str) -> str:
    return f"long {a} / short {b}" if pos > 0 else f"short {a} / long {b}" if pos < 0 else "no position"


def list_trades(res: pd.DataFrame, sig: pd.DataFrame, phase: str, a: str, b: str) -> pd.DataFrame:
    """Individual trades of a pair in a period: blocks with position != 0."""
    pos, net = res["position"].to_numpy(), res["net"].to_numpy()
    idx, z = res.index, sig["z"].to_numpy()
    prev = np.concatenate([[0.0], pos[:-1]])
    lag = 0 if EXECUTION == "close" else 1                  # days between signal and execution
    rows = []
    for i in np.where((pos != 0) & (prev != pos))[0]:
        e = i
        while e + 1 < len(pos) and pos[e + 1] == pos[i]:
            e += 1
        is_open = e == len(pos) - 1                         # still running at the end of the period
        x = min(e + 1, len(pos) - 1)                        # day on which the position is closed
        rows.append({"phase": phase, "pair": pair_name(a, b), "entry": idx[i].date(),
                     "exit": None if is_open else idx[x].date(), "direction": direction(pos[i], a, b),
                     "z_entry": z[max(i - lag, 0)], "z_exit": z[max(x - lag, 0)],   # z-score of the signal
                     "trading_days": e - i + 1,
                     "reason": "open" if is_open else (res["exit_reason"].iloc[x] or "z-score"),
                     "result": float(net[i:x + 1].sum())})
    return pd.DataFrame(rows, columns=["phase", "pair", "entry", "exit", "direction", "z_entry",
                                       "z_exit", "trading_days", "reason", "result"])


def combine(results: list[pd.DataFrame]) -> pd.DataFrame:
    """Basket: same capital per pair. Returns and capital shares are the average over all pairs."""
    out = pd.DataFrame(index=results[0].index)
    for col in ("gross", "costs", "net", "long_side", "short_side"):
        out[col] = np.mean([r[col].to_numpy() for r in results], axis=0)
    out["position"] = np.sum([np.abs(r["position"].to_numpy()) for r in results], axis=0)    # open pairs
    return out


def stats(res: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Metrics of a period from the daily returns after costs."""
    r = res["net"].to_numpy()
    years = max(len(r) / 252, 1e-9)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float(np.sqrt(252) * r.mean() / sd) if sd > 0 else float("nan")
    g = res["gross"].to_numpy()
    gsd = g.std(ddof=1) if len(g) > 1 else 0.0
    eq = np.cumprod(1 + r)
    closed = trades[trades["reason"] != "open"] if len(trades) else trades
    return {"date_from": res.index[0].date(), "date_to": res.index[-1].date(), "trading_days": len(r),
            "sharpe": sharpe,
            # rule of thumb for the uncertainty of an estimated Sharpe ratio (standard error)
            "sharpe_uncertainty": float(np.sqrt((1 + 0.5 * sharpe ** 2) / years)) if sharpe == sharpe else float("nan"),
            "sharpe_before_costs": float(np.sqrt(252) * g.mean() / gsd) if gsd > 0 else float("nan"),
            "return_pa": float(r.mean() * 252), "total_return": float(eq[-1] - 1),
            "growth_pa": float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0,   # with compound interest
            "worst_day": float(r.min()),
            "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "trades": len(trades), "winners": int((trades["result"] > 0).sum()),
            "mean_per_trade": float(trades["result"].mean()) if len(trades) else float("nan"),
            "worst_trade": float(trades["result"].min()) if len(trades) else float("nan"),
            "mean_holding_days": float(closed["trading_days"].mean()) if len(closed) else float("nan"),
            "share_days_invested": float((res["position"] != 0).mean())}


def kelly_leverage(res: pd.DataFrame) -> dict:
    """Kelly leverage of a period: f = m / s^2 with m = mean daily return, s^2 = variance of the daily return.

    m is the return per unit of leverage AFTER the financing costs that a further unit of leverage costs
    (interest on the additional long side). In Chan, this corresponds to deducting the financing rate.
    The maximum expected growth rate at full Kelly leverage is then approximately m^2 / (2 s^2) (continuous).
    """
    marginal = (res["long_side"].to_numpy() * FINANCING_RATE
                - res["short_side"].to_numpy() * SHORT_PROCEEDS_RATE) / 252
    r = res["net"].to_numpy()
    x = r - marginal
    m, var = x.mean(), x.var(ddof=1)
    f = float(m / var) if var > 0 else float("nan")
    return {"return_pa": float(r.mean() * 252), "financing_per_leverage_pa": float(marginal.mean() * 252),
            "return_after_financing_pa": float(m * 252), "volatility_pa": float(np.sqrt(var * 252)),
            "kelly_leverage": f, "growth_at_kelly": float(m * 252) ** 2 / (2 * var * 252) if var > 0 else float("nan")}


def sharpe_of(r: np.ndarray) -> float:
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(np.sqrt(252) * r.mean() / sd) if sd > 0 else float("nan")


def commission(shares: float, value: float) -> float:
    """Broker's fee for an order."""
    if shares <= 0:
        return 0.0
    return float(min(max(COMMISSION_MIN, COMMISSION_PER_SHARE * shares), COMMISSION_MAX_PCT * value))


def simulate_account(pairs: list[tuple[str, str]], sigs: dict, betas: dict, positions: dict, capital: float,
                     lev: float, phase: str, slippage_bps: float | None = None, margin_rules: bool = True,
                     skip: set | None = None, close_at_end: bool = True, margin_initial: float | None = -1.0,
                     margin_maintenance: float | None = -1.0, stop_level: float | None = None,
                     daily_loss_limit: float | None = None) -> dict:
    """Account in USD: whole shares, fees per order, borrowing fee, credit interest, margin call, total loss.

    sigs[name]:      daily values of the pair in the period (price_a, price_b, if applicable open_a, open_b)
    positions[name]: position at the end of the day (+1 / 0 / -1) from simulate() - the signals do not depend on the account
    lev:             target leverage. A new pair gets  lev * equity / number of pairs  (long plus
                     short side), calculated with the equity and closing prices of the signal day.
    Share quantities remain unchanged until the exit (no daily readjustment).
    skip:            {(pair, entry day)} that are not opened (marked as not executable in the protocol)
    margin_initial / margin_maintenance: coverage rules of this account (-1 = use the settings MARGIN_INITIAL /
                     MARGIN_MAINTENANCE, None = no check); margin_rules = False switches both off.
    stop_level:      The account is closed if the equity falls below this share of the starting capital.
    daily_loss_limit: The account is closed if the loss of one day reaches this share of the account balance of the
                     previous evening. After a closure, the model no longer trades.
    """
    m_init = MARGIN_INITIAL if margin_initial == -1.0 else margin_initial
    m_maint = MARGIN_MAINTENANCE if margin_maintenance == -1.0 else margin_maintenance
    if not margin_rules:
        m_init = m_maint = None
    floor = 0.0 if stop_level is None else float(stop_level) * capital
    slip = (COST_BPS if slippage_bps is None else slippage_bps) / 1e4
    skip = skip or set()
    names = [pair_name(a, b) for a, b in pairs]
    idx = sigs[names[0]].index
    n, n_pairs = len(idx), len(pairs)
    at_open = EXECUTION == "next_open"
    w = {nm: (1 / (1 + abs(betas[nm])), abs(betas[nm]) / (1 + abs(betas[nm]))) for nm in names}
    ca = {nm: sigs[nm]["price_a"].to_numpy() for nm in names}
    cb = {nm: sigs[nm]["price_b"].to_numpy() for nm in names}
    oa = {nm: sigs[nm]["open_a"].to_numpy() for nm in names} if at_open else ca
    ob = {nm: sigs[nm]["open_b"].to_numpy() for nm in names} if at_open else cb
    pos = {nm: np.asarray(positions[nm], dtype=float) for nm in names}

    equity = np.full(n, float(capital))
    gross_ratio = np.zeros(n)
    hold = {nm: [0.0, 0.0] for nm in names}                 # shares A, shares B (negative = sold short)
    live: dict[str, dict] = {}                              # ongoing trades
    orders, trades, margin_calls = [], [], []
    cost = {"fees": 0.0, "slippage": 0.0, "borrowing": 0.0, "interest": 0.0}
    count = {"orders": 0, "too_small": 0, "margin_rejected": 0, "not_executable": 0}
    ruin_day = None
    close_reason = ""
    E = float(capital)
    day_start = float(capital)                              # equity at the closing price of the previous day

    def num(x: float) -> float:
        return 0.0 if x != x else float(x)

    def gross(pa: dict, pb: dict, t: int) -> float:
        return sum(abs(hold[nm][0]) * num(pa[nm][t]) + abs(hold[nm][1]) * num(pb[nm][t]) for nm in names)

    def order(t: int, nm: str, k: int, shares: float, price: float, what: str) -> float:
        """Book an order. shares > 0 = purchase, < 0 = sale. Returns the costs."""
        value = abs(shares) * price
        fee, sl = commission(abs(shares), value), slip * value
        cost["fees"] += fee
        cost["slippage"] += sl
        count["orders"] += 1
        orders.append({"date": idx[t], "pair": nm, "stock": pairs[names.index(nm)][k], "order": what,
                       "side": "purchase" if shares > 0 else "sale", "shares": int(abs(shares)),
                       "model_price": price, "fee": fee})
        return fee + sl

    def close_pair(t: int, nm: str, pa: float, pb: float, why: str) -> float:
        """Close the position of a pair. Returns the costs."""
        sa, sb = hold[nm]
        c = 0.0
        if sa:
            c += order(t, nm, 0, -sa, pa, "sell (close position)" if sa > 0
                       else "buy back (close short sale)")
        if sb:
            c += order(t, nm, 1, -sb, pb, "sell (close position)" if sb > 0
                       else "buy back (close short sale)")
        hold[nm] = [0.0, 0.0]
        tr = live.pop(nm, None)
        if tr is not None:
            tr["result_usd"] -= c
            tr.update({"exit": idx[t].date(), "reason": why})
            trades.append(tr)
        return c

    def check(t: int, pa: dict, pb: dict) -> None:
        """Check for total loss and margin call (pa, pb = current prices)."""
        nonlocal E, ruin_day, close_reason
        g = gross(pa, pb, t)
        why = ""
        if E <= 0:
            why = "Total loss"
        elif E <= floor and floor > 0:
            why = "Total loss limit"
        elif daily_loss_limit is not None and day_start - E >= daily_loss_limit * day_start:
            why = "Daily loss limit"
        if why:                                             # account is closed: model drops out
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    if E > 0:                               # positions are still closed regularly
                        E -= close_pair(t, nm, num(pa[nm][t]), num(pb[nm][t]), why)
                    else:
                        hold[nm] = [0.0, 0.0]
                        tr = live.pop(nm, None)
                        if tr is not None:
                            tr.update({"exit": idx[t].date(), "reason": why})
                            trades.append(tr)
            E, ruin_day, close_reason = max(E, 0.0), idx[t], why
        elif m_maint is not None and g > 0 and E < m_maint * g:
            before = E
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    E -= close_pair(t, nm, num(pa[nm][t]), num(pb[nm][t]), "Margin Call")
            margin_calls.append({"phase": phase, "date": idx[t].date(), "equity_usd": before,
                                 "open_positions": g, "coverage": before / g})

    for t in range(n):
        if ruin_day is not None:
            equity[t] = E
            continue
        E_signal = E                                        # equity at the closing price of the previous day
        day_start = E
        # ---- borrowing fee and credit interest for the positions held overnight (values at the previous day's close)
        if t > 0:
            long_mv = short_mv = 0.0
            for nm in names:
                va, vb = hold[nm][0] * num(ca[nm][t - 1]), hold[nm][1] * num(cb[nm][t - 1])
                long_mv += max(va, 0.0) + max(vb, 0.0)
                short_here = max(-va, 0.0) + max(-vb, 0.0)
                short_mv += short_here
                if nm in live and short_here:
                    live[nm]["result_usd"] -= short_here * BORROW_FEE / 252
            debt = long_mv if FINANCING_BASIS == "full" else max(long_mv - E, 0.0)
            borrow = short_mv * BORROW_FEE / 252
            interest = debt * FINANCING_RATE / 252 - short_mv * SHORT_PROCEEDS_RATE / 252
            cost["borrowing"] += borrow
            cost["interest"] += interest
            E -= borrow + interest
        # ---- price move until the execution
        if at_open and t > 0:                               # overnight: previous day's close -> opening
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    pnl = hold[nm][0] * num(oa[nm][t] - ca[nm][t - 1]) + hold[nm][1] * num(ob[nm][t] - cb[nm][t - 1])
                    E += pnl
                    live[nm]["result_usd"] += pnl
            check(t, oa, ob)
        elif not at_open and t > 0:                         # whole day: close -> close
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    pnl = hold[nm][0] * num(ca[nm][t] - ca[nm][t - 1]) + hold[nm][1] * num(cb[nm][t] - cb[nm][t - 1])
                    E += pnl
                    live[nm]["result_usd"] += pnl
            check(t, ca, cb)
        # ---- orders: first close, then open
        if ruin_day is None:
            xa, xb = (oa, ob) if at_open else (ca, cb)      # execution prices
            changed = [nm for nm in names if pos[nm][t] != (pos[nm][t - 1] if t > 0 else 0.0)]
            for nm in changed:
                if hold[nm][0] or hold[nm][1]:
                    E -= close_pair(t, nm, num(xa[nm][t]), num(xb[nm][t]), "Signal")
            for nm in changed:
                p = pos[nm][t]
                if p == 0:
                    continue
                if (nm, idx[t].date()) in skip:
                    count["not_executable"] += 1
                    continue
                # size: with equity and closing prices of the signal day (known before the opening)
                if EXECUTION == "close":
                    e_size, sa_price, sb_price = E, num(ca[nm][t]), num(cb[nm][t])
                else:
                    e_size = E_signal
                    sa_price = num(ca[nm][t - 1]) if t > 0 else 0.0
                    sb_price = num(cb[nm][t - 1]) if t > 0 else 0.0
                pa_x, pb_x = num(xa[nm][t]), num(xb[nm][t])
                if min(sa_price, sb_price, pa_x, pb_x) <= 0:
                    continue
                budget = lev * e_size / n_pairs
                na, nb = round(w[nm][0] * budget / sa_price), round(w[nm][1] * budget / sb_price)
                if na < 1 or nb < 1:
                    count["too_small"] += 1
                    continue
                if m_init is not None:
                    after = gross(xa, xb, t) + na * pa_x + nb * pb_x
                    if E < m_init * after:
                        count["margin_rejected"] += 1
                        continue
                hold[nm] = [p * na, -p * nb]
                a_, b_ = pairs[names.index(nm)]
                c = (order(t, nm, 0, p * na, pa_x, "buy" if p > 0 else "sell short")
                     + order(t, nm, 1, -p * nb, pb_x, "sell short" if p > 0 else "buy"))
                E -= c
                live[nm] = {"phase": phase, "pair": nm, "entry": idx[t].date(), "exit": None,
                            "direction": direction(p, a_, b_), "shares_a": int(na), "shares_b": int(nb),
                            "value_a_usd": na * pa_x, "value_b_usd": nb * pb_x, "reason": "open", "result_usd": -c}
            if at_open:                                     # during the day: opening -> close
                for nm in names:
                    if hold[nm][0] or hold[nm][1]:
                        pnl = hold[nm][0] * num(ca[nm][t] - oa[nm][t]) + hold[nm][1] * num(cb[nm][t] - ob[nm][t])
                        E += pnl
                        live[nm]["result_usd"] += pnl
            check(t, ca, cb)
        equity[t] = max(E, 0.0)
        gross_ratio[t] = gross(ca, cb, t) / E if E > 0 else 0.0

    # ---- positions open at the end: valued at the last closing price; closing costs only in the backtest
    holdings = {nm: (int(hold[nm][0]), int(hold[nm][1])) for nm in names}
    if ruin_day is None:
        for nm in list(live):
            tr = live[nm]
            if close_at_end:
                sa, sb = hold[nm]
                c = sum(commission(abs(q), abs(q) * num(pr[nm][-1])) + slip * abs(q) * num(pr[nm][-1])
                        for q, pr in ((sa, ca), (sb, cb)))
                E -= c
                tr["result_usd"] -= c
                cost["fees"] += sum(commission(abs(q), abs(q) * num(pr[nm][-1])) for q, pr in ((sa, ca), (sb, cb)))
                cost["slippage"] += sum(slip * abs(q) * num(pr[nm][-1]) for q, pr in ((sa, ca), (sb, cb)))
            trades.append(tr)
        equity[-1] = max(E, 0.0)
    eq = pd.Series(equity, index=idx)
    r = eq.pct_change().fillna(eq.iloc[0] / capital - 1).replace([np.inf, -np.inf], 0.0).to_numpy()
    years = max(n / 252, 1e-9)
    tr_df = pd.DataFrame(trades, columns=["phase", "pair", "entry", "exit", "direction", "shares_a", "shares_b",
                                          "value_a_usd", "value_b_usd", "reason", "result_usd"])
    for c in ("value_a_usd", "value_b_usd", "result_usd"):
        tr_df[c] = tr_df[c].astype(float).round(2)
    opened = len(tr_df)
    imbalance = (float(((tr_df["value_a_usd"] - tr_df["value_b_usd"]).abs()
                        / ((tr_df["value_a_usd"] + tr_df["value_b_usd"]) / 2)).mean()) if opened else float("nan"))
    order_value = sum(o["shares"] * o["model_price"] for o in orders)
    return {"equity": eq, "returns": r, "trades": tr_df, "orders": orders, "margin_calls": margin_calls,
            "holdings": holdings, "ruin_day": ruin_day, "cost": cost, "count": count, "order_value": order_value,
            "gross_ratio": pd.Series(gross_ratio, index=idx),
            "kpi": {"date_from": idx[0].date(), "date_to": idx[-1].date(), "trading_days": n,
                    "start_capital": float(capital), "end_capital": float(eq.iloc[-1]),
                    "total_return": float(eq.iloc[-1] / capital - 1),
                    "growth_pa": float((eq.iloc[-1] / capital) ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0,
                    "sharpe": sharpe_of(r), "max_drawdown": float((eq / eq.cummax() - 1).min()),
                    "lowest_balance": float(eq.min()), "worst_day": float(r.min()),
                    "trades": opened, "winners": int((tr_df["result_usd"] > 0).sum()),
                    "orders": count["orders"], "too_small": count["too_small"],
                    "margin_rejected": count["margin_rejected"], "not_executable": count["not_executable"],
                    "margin_calls": len(margin_calls),
                    "total_loss": "" if ruin_day is None else f"{ruin_day:%d.%m.%Y}",
                    "close_reason": close_reason,
                    "fees_usd": cost["fees"], "slippage_usd": cost["slippage"],
                    "borrow_fee_usd": cost["borrowing"], "credit_interest_usd": cost["interest"],
                    "costs_per_order_bp": float((cost["fees"] + cost["slippage"]) / order_value * 1e4)
                    if order_value else float("nan"),
                    "leverage_mean": float(np.mean(gross_ratio)), "leverage_max": float(np.max(gross_ratio)),
                    "imbalance_mean": imbalance}}


def usable_pairs(px: pd.DataFrame) -> list[tuple[str, str]]:
    """Pairs for which both stocks have prices."""
    ok = []
    for a, b in PAIRS:
        if a in px.columns and b in px.columns and px[[a, b]].dropna().shape[0] > LOOKBACK + 20:
            ok.append((a, b))
        else:
            log(f"Pair {pair_name(a, b)} skipped: no or too few prices")
    if not ok:
        raise SystemExit("No prices are available for any pair.")
    return ok


def check_settings() -> None:
    if EXECUTION not in ("close", "next_open", "next_close"):
        raise SystemExit('EXECUTION must be "close", "next_open" or "next_close"')
    if EXIT_Z >= ENTRY_Z:
        raise SystemExit("EXIT_Z must be smaller than ENTRY_Z (otherwise the position would be exited immediately).")
    if STOP_Z is not None and STOP_Z <= ENTRY_Z:
        raise SystemExit("STOP_Z must be greater than ENTRY_Z.")
    if len(set(PAIRS)) != len(PAIRS):
        raise SystemExit("PAIRS contains a pair twice.")


# ------------------------------------------------------------------------------------------ Chart
def make_chart(curves: dict[str, pd.Series], labels: dict[str, str], test_start: pd.Timestamp) -> None:
    """Cumulative return without leverage: basket and each pair as a small individual chart -> output/model.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        log("Chart skipped (matplotlib missing: pip install matplotlib)")
        return
    surface, ink, muted, blue, band = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#f0efec"
    names = list(curves)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow + 0.8), sharex=True, squeeze=False,
                             facecolor=surface)
    for k, ax in enumerate(axes.ravel()):
        ax.set_facecolor(surface)
        if k >= len(names):
            ax.set_visible(False)
            continue
        s = curves[names[k]]
        ax.axvspan(test_start, s.index[-1], color=band, lw=0)
        ax.axhline(0, color=muted, lw=0.8)
        ax.plot(s.index, s, color=blue, lw=2.0 if k == 0 else 1.6)
        ax.set_title(f"{names[k]}\n{labels[names[k]]}", fontsize=9.5, color=ink, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f} %"))
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    fig.legend([Patch(color=band)], ["Test phase"], loc="upper right", frameon=False, fontsize=9, labelcolor=muted)
    fig.suptitle("Cumulative return after costs  (first chart: account, starts anew in each phase; "
                 "pairs without leverage; each chart has its own scale)", x=0.01, ha="left", fontsize=11.5, color=ink)
    import matplotlib.dates as mdates
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "model.png"
    try:
        fig.savefig(path, dpi=130, facecolor=surface)
    except PermissionError:
        path = OUT_DIR / f"model_{time.strftime('%H%M%S')}.png"
        fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    log(f"Chart saved: {path.name}")


# ------------------------------------------------------------------------------------------ Backtest
def profiles(kelly_f: float) -> list[dict]:
    """The three account profiles: without leverage, private (Reg T), institutional (quarter Kelly or similar, without Reg T)."""
    out = [{"name": "No leverage", "slug": "no_leverage", "lev": 1.0, "init": None, "maint": None, "stop": None,
            "daily": None, "note": "Leverage 1"},
           {"name": "Private", "slug": "private", "lev": float(RETAIL_LEVERAGE), "init": MARGIN_INITIAL,
            "maint": MARGIN_MAINTENANCE, "stop": None, "daily": None,
            "note": f"Target leverage {RETAIL_LEVERAGE:g}, margin rules of a normal account"}]
    if kelly_f == kelly_f and kelly_f > 0:
        q = KELLY_FRACTION * kelly_f
        cap = INSTITUTIONAL_MAX_LEVERAGE
        lev = min(q, cap) if cap is not None else q
        out.append({"name": "Institutional", "slug": "institutional", "lev": float(lev),
                    "init": (1.0 / cap) if cap else None, "maint": None, "stop": INSTITUTIONAL_STOP,
                    "daily": INSTITUTIONAL_DAILY_LOSS,
                    "note": f"{KELLY_FRACTION:g} x Kelly leverage {kelly_f:.1f} = {q:.1f}"
                            + (f", limited to {cap:g}" if cap is not None and q > cap else "")})
    return out


def usd(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def run_backtest(px: pd.DataFrame, show_grid: bool) -> None:
    idx = px.index
    last = idx[-1]
    test_start = last - pd.DateOffset(months=round(12 * TEST_YEARS))
    train_start = test_start - pd.DateOffset(months=round(12 * TRAIN_YEARS))
    train = np.asarray((idx > train_start) & (idx <= test_start))
    test = np.asarray(idx > test_start)
    phases = {"Training": train, "Test": test}
    pairs = usable_pairs(px)

    # ---- signals and returns per pair (without leverage, without account): shows the quality of the signals
    sigs, betas, raw, trades = {}, {}, {}, []
    pair_rows = []
    for a, b in pairs:
        name = pair_name(a, b)
        sigs[name], betas[name] = build_signals(px, a, b, train)
        row = {"pair": name}
        for ph, mask in phases.items():
            res = simulate(sigs[name][mask], betas[name], ENTRY_Z, EXIT_Z)
            tr = list_trades(res, sigs[name][mask], ph, a, b)
            raw[(name, ph)] = res
            trades.append(tr)
            k = stats(res, tr)
            p = ph.lower()
            row.update({f"sharpe_{p}": k["sharpe"], f"return_pa_{p}": k["return_pa"], f"trades_{p}": k["trades"],
                        f"winners_{p}": k["winners"], f"max_drawdown_{p}": k["max_drawdown"]})
        first = sigs[name]["z"].first_valid_index()
        row["z_from"] = first.date() if first is not None else None
        row["hedge_ratio"] = betas[name]
        row["z_current"] = float(sigs[name]["z"].iloc[-1])
        row["position"] = direction(raw[(name, "Test")]["position"].iloc[-1], a, b)
        want, pos = raw[(name, "Test")]["signal"].iloc[-1], raw[(name, "Test")]["position"].iloc[-1]
        row["open_signal"] = "" if want == pos else ("close position" if want == 0 else direction(want, a, b))
        pair_rows.append(row)
    all_trades = pd.concat(trades, ignore_index=True)
    port = {ph: combine([raw[(pair_name(a, b), ph)] for a, b in pairs]) for ph in phases}

    # ---- Kelly leverage of the basket from the training -> target leverage of the profile "Institutional"
    k_train, k_test = kelly_leverage(port["Training"]), kelly_leverage(port["Test"])
    f = k_train["kelly_leverage"]
    profs = profiles(f)

    # ---- accounts: whole shares, fees, rules of the respective profile; each phase starts with CAPITAL
    def account(ph: str, leverage: float, **kw) -> dict:
        m = phases[ph]
        return simulate_account(pairs, {n: sigs[n][m] for n in sigs}, betas,
                                {n: raw[(n, ph)]["position"].to_numpy() for n in sigs}, CAPITAL, leverage, ph, **kw)

    def run_profile(pf: dict, ph: str, **kw) -> dict:
        return account(ph, pf["lev"], margin_initial=pf["init"], margin_maintenance=pf["maint"],
                       stop_level=pf["stop"], daily_loss_limit=pf["daily"], **kw)

    acct = {(pf["slug"], ph): run_profile(pf, ph) for pf in profs for ph in phases}
    base = profs[0]["slug"]                                 # account without leverage
    for r in pair_rows:                                     # result per pair in the account without leverage
        for ph in phases:
            t = acct[(base, ph)]["trades"]
            r[f"account_usd_{ph.lower()}"] = float(t.loc[t["pair"] == r["pair"], "result_usd"].sum())
    pair_tab = pd.DataFrame(pair_rows)

    # ---- Files
    daily = pd.concat([pd.DataFrame({"phase": ph, "open_pairs": port[ph]["position"].astype(int),
                                     "basket_return_no_leverage": port[ph]["net"],
                                     **{f"account_usd_{pf['slug']}": acct[(pf["slug"], ph)]["equity"] for pf in profs},
                                     **{f"leverage_{pf['slug']}": acct[(pf["slug"], ph)]["gross_ratio"] for pf in profs}})
                       for ph in phases])
    for a, b in pairs:                                      # daily return of each pair without leverage
        name = pair_name(a, b)
        daily[f"net_{name}"] = pd.concat([raw[(name, ph)]["net"] for ph in phases])
    save_csv(daily.rename_axis("date"), "daily_values.csv", index=True)
    save_csv(all_trades, "trades.csv")
    save_csv(pd.concat([acct[(pf["slug"], ph)]["trades"].assign(profile=pf["name"]) for pf in profs for ph in phases],
                       ignore_index=True), "account_trades.csv")
    save_csv(pair_tab, "pairs.csv")
    save_csv(pd.DataFrame([{"profile": pf["name"], "target_leverage": pf["lev"], "phase": ph, **acct[(pf["slug"], ph)]["kpi"]}
                           for pf in profs for ph in phases]), "metrics.csv")
    calls = [dict(c, profile=pf["name"]) for pf in profs for ph in phases for c in acct[(pf["slug"], ph)]["margin_calls"]]
    if calls:
        save_csv(pd.DataFrame(calls), "margin_calls.csv")

    # ---- Output
    exits = [f"z-score back at {EXIT_Z:g}"]
    if MAX_HOLD_DAYS is not None:
        exits.append(f"at the latest after {MAX_HOLD_DAYS} trading days")
    if STOP_Z is not None:
        exits.append(f"stop at |z| >= {STOP_Z:g}")
    print()
    print("=" * 110)
    print(f"MEAN-REVERSION MODEL  -  basket of {len(pairs)} pairs, same settings for all")
    print("=" * 110)
    print(f"Spread   : log(A) - hedge ratio * log(B), hedge ratio "
          + ("estimated per pair in training" if HEDGE_RATIO == "training" else f"{float(HEDGE_RATIO):g}"))
    print(f"z-score  : " + (f"moving over {LOOKBACK} trading days" if Z_MODE == "rolling"
                            else "mean and std fixed from the training phase"))
    print(f"Entry    : |z| > {ENTRY_Z:g}")
    print(f"Exit     : " + "; ".join(exits))
    print("Trading  : " + {"close": "at the closing price of the signal day",
                           "next_open": "at the opening price of the next trading day",
                           "next_close": "at the closing price of the next trading day"}[EXECUTION])
    print(f"Account  : {usd(CAPITAL)} USD per phase and profile, whole shares, per pair target leverage x equity / {len(pairs)}")
    print(f"Fees     : {COMMISSION_PER_SHARE:g} USD per share, at least {COMMISSION_MIN:.2f} USD and at most "
          f"{COMMISSION_MAX_PCT:.0%} of the order value per order; slippage {COST_BPS:g} basis points")
    print(f"Financing: borrowing fee {BORROW_FEE:.1%} p. a.; credit interest {FINANCING_RATE:.2%} p. a. on "
          + ("the part of the long side above the equity" if FINANCING_BASIS == "credit" else "the whole long side"))
    print("Profiles :")
    for pf in profs:
        rules = []
        if pf["init"] is not None:
            rules.append(f"new position only up to leverage {1 / pf['init']:.0f}")
        if pf["maint"] is not None:
            rules.append(f"forced closing at coverage below {pf['maint']:.0%}")
        if pf["stop"] is not None:
            rules.append(f"account is closed at {1 - pf['stop']:.0%} total loss")
        if pf["daily"] is not None:
            rules.append(f"or at {pf['daily']:.0%} loss in one day")
        if pf["slug"] == "institutional" and pf["stop"] is None and pf["daily"] is None:
            rules.append("account is closed when the own capital is used up")
        print(f"    {pf['name']:<15} target leverage {pf['lev']:>5.1f}   ({pf['note']}" + ("; " + "; ".join(rules) if rules else "") + ")")
    if len(profs) < 3:
        print("    Institutional    not applicable: Kelly leverage in training not positive")
    frozen = read_start()
    if frozen is not None:
        diff = settings_diff(frozen["settings"], frozen_settings())
        print(f"Protocol: running since {frozen['start_day']}"
              + ("" if not diff else "  -  CAUTION: current settings differ from the protocol ("
                 + ", ".join(diff) + ")"))
    print()
    print("ACCOUNTS  -  in USD, after fees, slippage, borrowing fee and credit interest")
    rows = [("End capital", lambda k: usd(k["end_capital"])),
            ("Growth p. a.", lambda k: f"{k['growth_pa']:+.1%}"),
            ("Sharpe Ratio", lambda k: f"{k['sharpe']:.2f}"),
            ("Max. drawdown", lambda k: f"{k['max_drawdown']:.1%}"),
            ("Worst day", lambda k: f"{k['worst_day']:+.1%}"),
            ("Lowest balance", lambda k: usd(k["lowest_balance"])),
            ("ACCOUNT CLOSED on", lambda k: k["total_loss"] or "no"),
            ("  Reason", lambda k: k["close_reason"][:15] or "-"),
            ("Margin Calls", lambda k: f"{k['margin_calls']}"),
            ("Leverage reached (mean / max.)", lambda k: f"{k['leverage_mean']:.1f} / {k['leverage_max']:.1f}"),
            ("Trades (of which winners)", lambda k: f"{k['trades']} ({k['winners']})"),
            ("  rejected: leverage limit", lambda k: f"{k['margin_rejected']}"),
            ("  dropped: amount < 1 share", lambda k: f"{k['too_small']}"),
            ("Fees", lambda k: f"-{usd(k['fees_usd'])}"),
            ("Slippage", lambda k: f"-{usd(k['slippage_usd'])}"),
            ("Borrowing fee", lambda k: f"-{usd(k['borrow_fee_usd'])}"),
            ("Credit interest", lambda k: f"-{usd(k['credit_interest_usd'])}"),
            ("Costs per order", lambda k: f"{k['costs_per_order_bp']:.1f} bp")]
    w = 16
    for ph in phases:
        k0 = acct[(base, ph)]["kpi"]
        print(f"\n{ph.upper()}  {k0['date_from']:%d.%m.%Y} - {k0['date_to']:%d.%m.%Y}, starting capital {usd(CAPITAL)} USD")
        print(f"{'':<34}" + "".join(f"{pf['name']:>{w}}" for pf in profs))
        print(f"{'Target leverage':<34}" + "".join(f"{pf['lev']:>{w}.1f}" for pf in profs))
        for label, fmt in rows:
            print(f"{label:<34}" + "".join(f"{fmt(acct[(pf['slug'], ph)]['kpi']):>{w}}" for pf in profs))
    for c in calls:
        print(f"Margin call on {c['date']:%d.%m.%Y} ({c['profile']}, {c['phase']}): equity {usd(c['equity_usd'])} USD "
              f"covered only {c['coverage']:.0%} of the open positions - everything was closed compulsorily.")

    print()
    print("INDIVIDUAL PAIRS  -  signals without leverage, return relative to the capital of the pair; on the right the result in the account without leverage")
    print(f"{'Pair':<12}{'Sharpe':>8}{'Sharpe':>8}{'Return':>9}{'Return':>9}{'Trades':>8}{'Trades':>7}{'Account USD':>11}"
          f"{'Account USD':>11}{'z today':>9}  Position / open signal")
    print(f"{'':<12}{'Train.':>8}{'Test':>8}{'Train.':>9}{'Test':>9}{'Train.':>8}{'Test':>7}{'Train.':>11}{'Test':>11}")
    for r in pair_rows:
        state = r["position"] + (f"  ->  {r['open_signal']}" if r["open_signal"] else "")
        print(f"{r['pair']:<12}{r['sharpe_training']:>8.2f}{r['sharpe_test']:>8.2f}{r['return_pa_training']:>+9.1%}"
              f"{r['return_pa_test']:>+9.1%}{r['trades_training']:>8}{r['trades_test']:>7}"
              f"{usd(r['account_usd_training']):>11}{usd(r['account_usd_test']):>11}{r['z_current']:>+9.2f}  {state}")
    late = [r["pair"] for r in pair_rows if r["z_from"] is not None and pd.Timestamp(r["z_from"]) > idx[train][0]]
    if late:
        print(f"Shorter history (z-score only available after the start of training): {', '.join(late)}")
    multi = [t for t in tickers() if sum(t in p for p in pairs) > 1]
    if multi:
        print(f"Contained in several pairs: {', '.join(multi)} - these pairs are not independent of each other.")
    print(f"\nAs of {last:%d.%m.%Y}. 'Open signal' = order for "
          + {"close": "the next closing price", "next_open": "the next opening",
             "next_close": "the next closing price"}[EXECUTION] + ".")
    print("\nNote: The pairs were selected because they performed best from 2019 to 2026 - the test phase was\n"
          "known at the time. TEST is therefore too favorable and no proof. Anyone who also changes the settings until\n"
          "TEST looks good no longer has a test. Only the protocol provides really new data\n"
          "(--protocol).")

    # ---- cost scale: how much slippage can the model tolerate?
    print()
    print("-" * 110)
    print("COST SCALE  -  growth p. a. per profile at different slippage per order (fees as above)")
    print("-" * 110)
    print(f"{'Slippage':>10} |" + " |".join("".join(f"{pf['name']:>{w}}" for pf in profs) for _ in phases))
    print(f"{'':>10} |" + " |".join(f"{ph.upper():^{w * len(profs)}}" for ph in phases))
    ladder_rows = []
    for bp in COST_LADDER_BPS:
        ks = {(pf["slug"], ph): (acct[(pf["slug"], ph)]["kpi"] if bp == COST_BPS
                                 else run_profile(pf, ph, slippage_bps=bp)["kpi"]) for pf in profs for ph in phases}
        print(f"{format(bp, 'g') + ' bp':>10} |" + " |".join(
            "".join(f"{ks[(pf['slug'], ph)]['growth_pa']:>+{w}.1%}" for pf in profs) for ph in phases)
            + ("   <- assumption" if bp == COST_BPS else ""))
        ladder_rows.append({"slippage_bp": bp, **{f"{ph.lower()}_{pf['slug']}_{k}": ks[(pf["slug"], ph)][k]
                                                  for ph in phases for pf in profs
                                                  for k in ("end_capital", "growth_pa", "sharpe")}})
    for ph in phases:                                       # slippage at which nothing of the profit would be left
        a0 = run_profile(profs[0], ph, slippage_bps=0)
        gain, vol = a0["kpi"]["end_capital"] - CAPITAL, a0["order_value"]
        if vol > 0 and gain > 0:
            print(f"{ph} (no leverage): The profit would be used up at about {gain / vol * 1e4:.0f} basis points of slippage per "
                  f"order (approximation: profit without slippage / order value).")
    save_csv(pd.DataFrame(ladder_rows), "cost_scale.csv")

    # ---- Kelly formula and stress test: would a total loss have occurred with more leverage?
    print()
    print("-" * 110)
    print("KELLY FORMULA AND LEVERAGE STRESS TEST")
    print("-" * 110)
    print(f"Kelly formula (training, basket without leverage): return {k_train['return_pa']:+.1%} p. a., minus financing per "
          f"unit of leverage {k_train['financing_per_leverage_pa']:.1%},\nvolatility {k_train['volatility_pa']:.1%} p. a.  ->  "
          f"Kelly leverage {f:.1f}" + (f"; profile Institutional: {profs[-1]['note']} -> target leverage {profs[-1]['lev']:.1f}"
                                     if len(profs) == 3 else "  (not positive: according to Kelly, do not trade)"))
    print("\nStress test: the same account with different target leverages, without leverage limit and without forced closing.\n"
          "Total loss = equity used up, the model drops out.")
    print(f"\n{'Leverage':>7} |{'End capital':>13}{'max. DD':>14}{'lowest bal.':>16}{'Total loss':>15} |"
          f"{'End capital':>13}{'max. DD':>14}{'lowest bal.':>16}{'Total loss':>15}")
    print(f"{'':>7} |{'TRAINING':^58} |{'TEST':^58}")
    stress_rows = []
    marks = {round(pf["lev"], 6): pf["name"] for pf in profs}
    levels = sorted(set([float(x) for x in STRESS_LEVERAGE] + [float(pf["lev"]) for pf in profs]))
    for lv in levels:
        ks = {ph: account(ph, lv, margin_rules=False)["kpi"] for ph in phases}
        print(f"{lv:>7.1f} |" + " |".join(
            f"{usd(ks[ph]['end_capital']):>13}{ks[ph]['max_drawdown']:>14.0%}{usd(ks[ph]['lowest_balance']):>16}"
            f"{(ks[ph]['total_loss'] or 'no'):>15}" for ph in phases)
            + (f"   <- {marks[round(lv, 6)]}" if round(lv, 6) in marks else ""))
        stress_rows.append({"leverage": lv, **{f"{ph.lower()}_{k}": ks[ph][k] for ph in phases
                                            for k in ("end_capital", "max_drawdown", "lowest_balance", "total_loss")}})
    save_csv(pd.DataFrame(stress_rows), "leverage_stress_test.csv")
    print("\nHow to read: No total loss in these four years does not mean that the leverage is safe. The pairs are selected in\n"
          "hindsight, and the worst day of the future is not in this data.")

    if show_grid:
        print()
        print("-" * 100)
        print("GRID - training phase only, whole basket, without leverage and without account: Sharpe ratio (in brackets: trades)")
        print("-" * 100)
        head = "EXIT_Z ->"
        print(f"{head:>13}" + "".join(f"{x:>15g}" for x in GRID_EXIT_Z))
        grid_rows = []
        for en in GRID_ENTRY_Z:
            cells = []
            for ex in GRID_EXIT_Z:
                if ex >= en:
                    cells.append(f"{'-':>15}")
                    continue
                rs = [simulate(sigs[pair_name(a, b)][train], betas[pair_name(a, b)], en, ex) for a, b in pairs]
                n_tr = sum(len(list_trades(r, sigs[pair_name(a, b)][train], "Training", a, b))
                           for r, (a, b) in zip(rs, pairs))
                pt = combine(rs)
                cells.append(f"{sharpe_of(pt['net'].to_numpy()):>8.2f} ({n_tr:>4})")
                grid_rows.append({"entry_z": en, "exit_z": ex, "sharpe_training": sharpe_of(pt["net"].to_numpy()),
                                  "return_pa_training": float(pt["net"].mean() * 252), "trades_training": n_tr})
            print(f"{'ENTRY_Z ' + format(en, 'g'):>13}" + "".join(cells))
        save_csv(pd.DataFrame(grid_rows), "grid_training.csv")
        print("The grid applies to all pairs together. Neighboring cells should look similar; a single\n"
              "peak value between weak neighbors is mostly chance. While the protocol is running, do not change the\n"
              "settings any more.")

    # ---- chart: account without leverage and pairs
    eq_all = pd.concat([acct[(base, ph)]["equity"] / CAPITAL - 1 for ph in phases])
    first = f"ACCOUNT without leverage ({usd(CAPITAL)} USD per phase)"
    curves = {first: eq_all}
    labels = {first: f"Sharpe Training {acct[(base, 'Training')]['kpi']['sharpe']:.2f}, "
                     f"Test {acct[(base, 'Test')]['kpi']['sharpe']:.2f}"}
    for r in pair_rows:
        sr = pd.concat([raw[(r["pair"], ph)]["net"] for ph in phases])
        curves[r["pair"]] = np.cumprod(1 + sr) - 1
        labels[r["pair"]] = f"Sharpe Training {r['sharpe_training']:.2f}, Test {r['sharpe_test']:.2f}"
    make_chart(curves, labels, test_start)
    print(f"\nResults: {OUT_DIR}  (metrics.csv, pairs.csv, account_trades.csv, trades.csv, daily_values.csv,\n"
          f"            cost_scale.csv, leverage_stress_test.csv, model.png)")


# ------------------------------------------------------------------------------------------ Protocol
def frozen_settings() -> dict:
    """Everything that determines the signals. These values must not change during the protocol."""
    s = {"PAIRS": [list(p) for p in PAIRS], "Z_MODE": Z_MODE, "LOOKBACK": LOOKBACK, "ENTRY_Z": ENTRY_Z,
         "EXIT_Z": EXIT_Z, "MAX_HOLD_DAYS": MAX_HOLD_DAYS, "STOP_Z": STOP_Z, "HEDGE_RATIO": HEDGE_RATIO,
         "EXECUTION": EXECUTION}
    if Z_MODE == "training" or HEDGE_RATIO == "training":   # only then do the signals depend on the training period
        s["TRAIN_YEARS"] = TRAIN_YEARS
    return s


def settings_diff(old: dict, new: dict) -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k):
            if k == "PAIRS":
                out.append("PAIRS (pair list)")
            else:
                out.append(f"{k}: {old.get(k)} -> {new.get(k)}")
    return out


def read_start() -> dict | None:
    path = PROTOCOL_DIR / "start.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dates(col: pd.Series) -> pd.Series:
    """Read the date column, even if Excel has rewritten it as 05.10.2026 when saving."""
    s = col.astype(str).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    de = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(de)


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def to_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def order_text(want: float, pos: float, a: str, b: str) -> str:
    """Order that is pending after the closing price of a day for the next execution."""
    if want == pos:
        return "hold" if pos != 0 else ""
    if want == 0:
        return "close position"
    return f"{a} buy / {b} sell short" if want > 0 else f"{a} sell short / {b} buy"


def run_protocol(px: pd.DataFrame, restart: bool) -> None:
    idx = px.index
    last = idx[-1]
    now = time.strftime("%Y-%m-%d %H:%M")

    # ---- start: freeze the settings or compare them with the frozen state
    if restart and PROTOCOL_DIR.exists():
        target = BASE_DIR / f"protocol_old_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(PROTOCOL_DIR), str(target))
        log(f"Previous protocol archived in {target.name}")
    start = read_start()
    if start is None:
        PROTOCOL_DIR.mkdir(exist_ok=True)
        start = {"start_day": f"{last:%Y-%m-%d}", "created_on": now, "settings": frozen_settings(),
                 "assumptions_for_information": {"COST_BPS": COST_BPS, "BORROW_FEE": BORROW_FEE},
                 "note": "Start day = last price day at creation. Only signals after it are counted."}
        (PROTOCOL_DIR / "start.json").write_text(json.dumps(start, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Protocol created. Start day {last:%d.%m.%Y}, settings frozen in protocol/start.json")
    else:
        diff = settings_diff(start["settings"], frozen_settings())
        if diff:
            raise SystemExit(
                "The settings in the script differ from the frozen state of the protocol:\n    "
                + "\n    ".join(diff)
                + "\nEither reset the settings (see protocol/start.json) or run\n"
                  "    python pair_model.py --protocol-new\nto start a new protocol. The old one is archived, "
                  "the previous results then no longer count.")
    start_day = pd.Timestamp(start["start_day"])
    train = np.asarray((idx > start_day - pd.DateOffset(months=round(12 * TRAIN_YEARS))) & (idx <= start_day))
    period = np.asarray(idx > start_day)
    pairs = usable_pairs(px)
    n_days = int(period.sum())
    lag = 0 if EXECUTION == "close" else 1
    exec_label = {"close": "closing price", "next_open": "opening", "next_close": "closing price"}[EXECUTION]
    next_label = {"close": "at the next closing price", "next_open": "at the next opening",
                  "next_close": "at the next closing price"}[EXECUTION]

    # ---- trade each pair from the start day (without leverage)
    sigs, betas, results, trades, log_rows, order_rows, pair_rows = {}, {}, {}, [], [], [], []
    for a, b in pairs:
        name = pair_name(a, b)
        sig, beta = build_signals(px, a, b, train)
        sigs[name], betas[name] = sig, beta
        z_start = sig["z"].loc[:start_day].dropna()
        z_now = float(sig["z"].iloc[-1])
        if n_days == 0:
            waiting = bool(len(z_start)) and abs(z_start.iloc[-1]) >= ENTRY_Z
            pair_rows.append({"pair": name, "z_current": z_now, "position": "no position",
                              "order": "waiting for return to the normal range" if waiting else "",
                              "trades": 0, "open": 0, "return_since_start": 0.0})
            continue
        s = sig[period]
        res = simulate(s, beta, ENTRY_Z, EXIT_Z, start_blocked=True, close_at_end=False)
        tr = list_trades(res, s, "Protocol", a, b)
        results[name] = res
        trades.append(tr)
        pos, want = res["position"].to_numpy(), res["signal"].to_numpy()
        ever_inside = np.maximum.accumulate((s["z"].abs() < ENTRY_Z).fillna(False).to_numpy())
        for t in range(len(s)):
            day = s.index[t]
            order = order_text(want[t], pos[t], a, b)
            if not order and want[t] == 0 and not ever_inside[t] and abs(s["z"].iloc[t]) >= ENTRY_Z:
                order = "waiting for return to the normal range"
            log_rows.append({"logged_on": now, "price_date": f"{day:%Y-%m-%d}",
                             "added_later": "yes" if day < last else "no", "pair": name,
                             "price_a": round(float(s["price_a"].iloc[t]), 4), "price_b": round(float(s["price_b"].iloc[t]), 4),
                             "z": round(float(s["z"].iloc[t]), 3),
                             "position": direction(pos[t], a, b), "order_after_close": order,
                             "daily_return": round(float(res["net"].iloc[t]) + 0.0, 6)})
        order_now = order_text(want[-1], pos[-1], a, b)
        if not order_now and not ever_inside[-1] and abs(z_now) >= ENTRY_Z:
            order_now = "waiting for return to the normal range"
        closed = tr[tr["reason"] != "open"]
        pair_rows.append({"pair": name, "z_current": z_now, "position": direction(pos[-1], a, b), "order": order_now,
                          "want": float(want[-1]), "pos": float(pos[-1]),
                          "trades": len(closed), "winners": int((closed["result"] > 0).sum()),
                          "open": int(pos[-1] != 0), "return_since_start": float(np.prod(1 + res["net"]) - 1),
                          "sharpe": sharpe_of(res["net"].to_numpy()) if n_days >= MIN_DAYS_FOR_SHARPE else float("nan"),
                          "worst_trade": float(tr["result"].min()) if len(tr) else float("nan")})

    # ---- account since the start day: CAPITAL, without leverage, whole shares, fees of the broker
    fills_path = PROTOCOL_DIR / "executions.csv"
    cols = ["date", "pair", "stock", "order", "side", "shares", "execution", "model_price", "real_price",
            "deviation_bp", "executable"]
    old_fills = None
    skip: set = set()
    if fills_path.exists():
        old_fills = read_semicolon_csv(fills_path)
        old_fills["date"] = parse_dates(old_fills["date"]).dt.strftime("%Y-%m-%d")
        for c in cols:                                      # older files do not yet have all columns
            if c not in old_fills.columns:
                old_fills[c] = "yes" if c == "executable" else ""
        no = old_fills["executable"].astype(str).str.strip().str.lower().str.startswith("n")
        entry = old_fills["order"].isin(["buy", "sell short"])
        skip = {(p_, pd.Timestamp(d).date()) for p_, d in zip(old_fills.loc[no & entry, "pair"],
                                                              old_fills.loc[no & entry, "date"])}
    acct = acct_clean = None
    equity_now = float(CAPITAL)
    if n_days:
        def account(lev: float = 1.0, margin_rules: bool = False, **kw) -> dict:
            return simulate_account(pairs, {n: sigs[n][period] for n in sigs}, betas,
                                    {n: results[n]["position"].to_numpy() for n in sigs}, CAPITAL, lev, "Protocol",
                                    margin_rules=margin_rules, close_at_end=False, **kw)
        acct = account()
        equity_now = float(acct["equity"].iloc[-1])
        if skip:
            acct_clean = account(skip=skip)
        for o in acct["orders"]:
            order_rows.append({"date": f"{o['date']:%Y-%m-%d}", "pair": o["pair"], "stock": o["stock"],
                               "order": o["order"], "side": o["side"], "shares": o["shares"],
                               "execution": exec_label, "model_price": round(float(o["model_price"]), 4),
                               "real_price": "", "deviation_bp": "", "executable": "yes"})
    # orders for the next execution with share quantities (equity and closing prices of today)
    for r, (a, b) in zip(pair_rows, pairs):
        want_, pos_ = r.get("want", 0.0), r.get("pos", 0.0)
        if want_ == pos_:
            continue
        name = r["pair"]
        if want_ != 0:
            w_a, w_b = 1 / (1 + abs(betas[name])), abs(betas[name]) / (1 + abs(betas[name]))
            budget = equity_now / len(pairs)
            na = round(w_a * budget / float(sigs[name]["price_a"].iloc[-1]))
            nb = round(w_b * budget / float(sigs[name]["price_b"].iloc[-1]))
            if na < 1 or nb < 1:
                r["order"] = "dropped (amount not enough for one share)"
            elif want_ > 0:
                r["order"] = f"{a} {na} shares buy / {b} {nb} shares sell short"
            else:
                r["order"] = f"{a} {na} shares sell short / {b} {nb} shares buy"
        elif acct is not None:
            ha, hb = acct["holdings"][name]
            if ha == 0 and hb == 0:
                r["order"] = ""                             # not opened in the account at all
            else:
                r["order"] = (f"close: {a} {abs(ha)} shares {'sell' if ha > 0 else 'buy back'} / "
                              f"{b} {abs(hb)} shares {'sell' if hb > 0 else 'buy back'}")

    # ---- leverage variants: fixed leverage values and the Kelly leverage from the years BEFORE the start day.
    # The Kelly leverage is calculated once and recorded in leverage.json so that it does not change later.
    lev_path = PROTOCOL_DIR / "leverage.json"
    if lev_path.exists():
        kelly_info = json.loads(lev_path.read_text(encoding="utf-8"))
    else:
        kk = kelly_leverage(combine([simulate(sigs[n][train], betas[n], ENTRY_Z, EXIT_Z) for n in sigs]))
        t_idx = idx[train]
        kelly_info = {"kelly_leverage": kk["kelly_leverage"], "calculated_on": now,
                      "period": f"{t_idx[0]:%Y-%m-%d} to {t_idx[-1]:%Y-%m-%d}",
                      "return_pa_no_leverage": kk["return_pa"], "volatility_pa": kk["volatility_pa"],
                      "financing_per_leverage_pa": kk["financing_per_leverage_pa"],
                      "note": "Kelly leverage of the basket from the years before the start day; is not recalculated."}
        lev_path.write_text(json.dumps(kelly_info, indent=2, ensure_ascii=False), encoding="utf-8")
    kelly_f = float(kelly_info["kelly_leverage"])
    profs = profiles(kelly_f)
    lev_runs = []
    if n_days:
        for pf in profs:
            lev_runs.append((pf, account(lev=pf["lev"], margin_rules=True, margin_initial=pf["init"],
                                         margin_maintenance=pf["maint"], stop_level=pf["stop"],
                                         daily_loss_limit=pf["daily"])))

    # ---- protocol.csv: only append new rows, never change existing ones
    log_path = PROTOCOL_DIR / "protocol.csv"
    new_log = pd.DataFrame(log_rows)
    n_new, added_late = 0, False
    if len(new_log):
        known: set[tuple[str, str]] = set()
        if log_path.exists():
            old = read_semicolon_csv(log_path)
            known = set(zip(parse_dates(old["price_date"]).dt.strftime("%Y-%m-%d"), old["pair"]))
        add = new_log[[(d, p) not in known for d, p in zip(new_log["price_date"], new_log["pair"])]]
        n_new = len(add)
        added_late = bool((add["added_later"] == "yes").any())
        if n_new:
            try:
                add.to_csv(log_path, sep=";", decimal=",", index=False, encoding="utf-8-sig" if not log_path.exists()
                           else "utf-8", mode="a", header=not log_path.exists())
            except PermissionError:
                n_new = 0
                log("protocol.csv is locked (open in Excel?) - today's rows will be "
                    "added on the next run")

    # ---- executions.csv: add new orders, keep entered real prices, calculate the deviation
    fills = pd.DataFrame(order_rows, columns=cols)
    if old_fills is not None:
        old = old_fills
        old["model_price"] = to_float(old["model_price"])
        key_old = set(zip(old["date"], old["pair"], old["stock"]))
        fills = fills[[(d, p, t) not in key_old for d, p, t in zip(fills["date"], fills["pair"], fills["stock"])]]
        fills = pd.concat([old[cols], fills], ignore_index=True)
    real, model = to_float(fills["real_price"]), to_float(fills["model_price"])
    # positive = worse than the model: bought more expensively or sold more cheaply
    dev = np.where(fills["side"] == "purchase", real / model - 1, 1 - real / model) * 1e4
    fills["deviation_bp"] = pd.Series(dev, index=fills.index).round(1)
    fills["real_price"] = real
    fills["model_price"] = model
    fills = fills.sort_values(["date", "pair", "stock"]).reset_index(drop=True)
    if len(fills) or not fills_path.exists():
        try:
            fills.to_csv(fills_path, sep=";", decimal=",", index=False, encoding="utf-8-sig")
        except PermissionError:
            log("executions.csv is locked (open in Excel?) - new orders will be added on the next run")
    measured = fills["deviation_bp"].dropna()

    # ---- result files
    pair_tab = pd.DataFrame(pair_rows).drop(columns=["want", "pos"], errors="ignore")
    save_csv(pair_tab, "results_pairs.csv", folder=PROTOCOL_DIR)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    if n_days:
        port = combine([results[pair_name(a, b)] for a, b in pairs])
        daily = pd.DataFrame({"open_pairs": port["position"].astype(int), "account_usd": acct["equity"].round(2),
                              "account_return_since_start": acct["equity"] / CAPITAL - 1,
                              "basket_return": port["net"],
                              "basket_cumulative": np.cumprod(1 + port["net"]) - 1})
        for pf, run in lev_runs:                            # account balance per profile
            daily[f"account_usd_{pf['slug']}"] = run["equity"].round(2)
        for a, b in pairs:
            name = pair_name(a, b)
            daily[f"cumulative_{name}"] = np.cumprod(1 + results[name]["net"]) - 1
        save_csv(daily.rename_axis("date"), "results.csv", index=True, folder=PROTOCOL_DIR)
        save_csv(pd.DataFrame([{"profile": pf["name"], "target_leverage": pf["lev"], "derivation": pf["note"],
                                **{k: run["kpi"][k] for k in
                                   ("end_capital", "total_return", "growth_pa", "sharpe", "max_drawdown",
                                    "worst_day", "lowest_balance", "leverage_mean", "leverage_max", "trades",
                                    "winners", "margin_rejected", "too_small", "margin_calls", "total_loss",
                                    "close_reason",
                                    "fees_usd", "slippage_usd", "borrow_fee_usd", "credit_interest_usd")}}
                               for pf, run in lev_runs]), "results_leverage.csv", folder=PROTOCOL_DIR)
        save_csv(all_trades, "trades.csv", folder=PROTOCOL_DIR)
        save_csv(acct["trades"], "account_trades.csv", folder=PROTOCOL_DIR)

    # ---- Output
    print()
    print("=" * 96)
    print(f"PROTOCOL (paper trading)  -  {len(pairs)} pairs, without leverage, account {usd(CAPITAL)} USD  -  prices up to {last:%d.%m.%Y}")
    print("=" * 96)
    print(f"Start day {start_day:%d.%m.%Y}, created on {start['created_on']}; settings unchanged "
          f"(entry |z| > {ENTRY_Z:g}, exit at {EXIT_Z:g}, trading: {exec_label})")
    orders = [r for r in pair_rows if r["order"] and r["order"] not in ("hold",)
              and not r["order"].startswith("waiting") and not r["order"].startswith("dropped")]
    print(f"\nORDERS {next_label.upper()}:")
    if orders:
        for r in orders:
            print(f"    {r['pair']:<12} {r['order']:<62} (z-score {r['z_current']:+.2f})")
    else:
        print("    none")
    print(f"\n{'Pair':<12}{'z today':>9}  {'Position':<26}{'Trades':>7}{'Winners':>9}{'since start':>12}  Remark")
    for r in pair_rows:
        note = r["order"] if r["order"].startswith(("waiting", "dropped")) else ""
        print(f"{r['pair']:<12}{r['z_current']:>+9.2f}  {r['position']:<26}{r['trades']:>7}{r.get('winners', 0):>9}"
              f"{r['return_since_start']:>+12.1%}  {note}")
    print()
    if n_days:
        r = port["net"].to_numpy()
        eq = np.cumprod(1 + r)
        closed = all_trades[all_trades["reason"] != "open"] if len(all_trades) else all_trades
        print(f"BASKET SINCE START ({n_days} trading days): return {eq[-1] - 1:+.2%}, max. drawdown "
              f"{(eq / np.maximum.accumulate(eq) - 1).min():.2%}, closed trades {len(closed)}"
              + (f" (of which {int((closed['result'] > 0).sum())} winners)" if len(closed) else "")
              + f", open pairs {int(port['position'].iloc[-1])}")
        if n_days >= MIN_DAYS_FOR_SHARPE:
            sh = sharpe_of(r)
            print(f"Sharpe ratio since start {sh:.2f}  (uncertainty +/- {np.sqrt((1 + 0.5 * sh ** 2) / (n_days / 252)):.2f})")
        else:
            print(f"Sharpe ratio: only shown from {MIN_DAYS_FOR_SHARPE} trading days on (meaningless before).")
        k = acct["kpi"]
        print(f"\nACCOUNT SINCE START ({usd(CAPITAL)} USD, whole shares, broker fees, {COST_BPS:g} bp slippage):")
        print(f"    Balance {usd(k['end_capital'])} USD ({k['total_return']:+.2%}), max. drawdown {k['max_drawdown']:.2%}, "
              f"{k['orders']} orders, fees {usd(k['fees_usd'])} USD, slippage {usd(k['slippage_usd'])} USD, "
              f"borrowing fee {usd(k['borrow_fee_usd'])} USD")
        if k["too_small"]:
            print(f"    {k['too_small']} trade(s) dropped because the amount was not enough for a whole share.")
        if k["orders"]:
            parts = []
            for bp in COST_LADDER_BPS:
                kk = k if bp == COST_BPS else account(slippage_bps=bp)["kpi"]
                parts.append(f"{bp:g} bp: {kk['total_return']:+.2%}")
            print("    Cost scale (slippage per order -> return since start):  " + "   ".join(parts))
            a0 = account(slippage_bps=0)
            gain = a0["kpi"]["end_capital"] - CAPITAL
            if gain > 0 and a0["order_value"] > 0:
                print(f"    The profit would be used up at about {gain / a0['order_value'] * 1e4:.0f} basis points of slippage per order.")
        if acct_clean is not None:
            kc = acct_clean["kpi"]
            print(f"    Without the {kc['not_executable']} trades marked as not executable: balance {usd(kc['end_capital'])} USD "
                  f"({kc['total_return']:+.2%})")
        # ---- The three profiles
        print(f"\nTHREE PROFILES SINCE START  -  each its own account with {usd(CAPITAL)} USD, same signals, different share quantities")
        w = 16
        print(f"{'':<34}" + "".join(f"{pf['name']:>{w}}" for pf, _ in lev_runs))
        print(f"{'Target leverage':<34}" + "".join(f"{pf['lev']:>{w}.1f}" for pf, _ in lev_runs))
        for label, fmt in (("Balance USD", lambda k: usd(k["end_capital"])),
                           ("Return since start", lambda k: f"{k['total_return']:+.2%}"),
                           ("Max. drawdown", lambda k: f"{k['max_drawdown']:.1%}"),
                           ("Worst day", lambda k: f"{k['worst_day']:+.1%}"),
                           ("Leverage reached (mean / max.)", lambda k: f"{k['leverage_mean']:.1f} / {k['leverage_max']:.1f}"),
                           ("Trades", lambda k: f"{k['trades']}"),
                           ("  rejected: leverage limit", lambda k: f"{k['margin_rejected']}"),
                           ("Margin Calls", lambda k: f"{k['margin_calls']}"),
                           ("ACCOUNT CLOSED on", lambda k: k["total_loss"] or "no"),
                           ("  Reason", lambda k: k["close_reason"][:15] or "-"),
                           ("Credit interest USD", lambda k: f"-{usd(k['credit_interest_usd'])}")):
            print(f"{label:<34}" + "".join(f"{fmt(run['kpi']):>{w}}" for _, run in lev_runs))
        if n_days >= MIN_DAYS_FOR_SHARPE:
            print(f"{'Sharpe Ratio':<34}" + "".join(f"{run['kpi']['sharpe']:>{w}.2f}" for _, run in lev_runs))
        for pf in profs:
            print(f"    {pf['name']}: {pf['note']}")
        print(f"    Kelly leverage {kelly_f:.1f} from {kelly_info['period']} (before the start day, recorded in leverage.json).\n"
              "    The orders above and executions.csv apply to the account without leverage.")
    else:
        print("Today is the start day: There is no protocol day yet. Counting starts from the next trading day.")
        print("Profiles from the first protocol day: " + ", ".join(f"{pf['name']} (target leverage {pf['lev']:.1f})" for pf in profs)
              + f"  -  Kelly leverage {kelly_f:.1f} from {kelly_info['period']}")
    if len(measured):
        print(f"Measured deviation from the model price: on average {measured.mean():+.1f} basis points per order "
              f"({len(measured)} entered prices; assumed in the model: {COST_BPS:g}). Positive = worse than the model.")
    else:
        print("No real prices entered yet (protocol/executions.csv, column 'real_price'; mark trades that were not possible\n"
              "in the demo account there in the column 'executable' with 'no').")
    if n_new:
        print(f"\nLogged: {n_new} new rows in protocol.csv"
              + (" - including days added later (the script did not run on every trading day)" if added_late else ""))
    elif n_days:
        print("\nLogged: nothing new (this price day is already in the protocol).")
    print(f"Files: {PROTOCOL_DIR}  (protocol.csv, executions.csv, results.csv, results_pairs.csv,\n"
          f"         results_leverage.csv, trades.csv, account_trades.csv)")


# ------------------------------------------------------------------------------------------ Procedure
def main() -> None:
    ap = argparse.ArgumentParser(description="Mean-reversion model for a basket of stock pairs")
    ap.add_argument("--grid", action="store_true",
                    help="show the Sharpe ratio of the basket in training for several entry and exit values")
    ap.add_argument("--protocol", action="store_true",
                    help="protocol mode (paper trading): record the daily status, orders and result since start")
    ap.add_argument("--protocol-new", dest="protocol_new", action="store_true",
                    help="start the protocol anew; the previous one is archived")
    ap.add_argument("--refresh", action="store_true", help="reload prices from Yahoo")
    ap.add_argument("--cutoff", default=None, help=argparse.SUPPRESS)     # for testing only: prices up to this day
    args = ap.parse_args()
    check_settings()

    px = load_prices(args.refresh)
    px = px[px[tickers()].notna().any(axis=1)]
    if args.cutoff:
        px = px[px.index <= pd.Timestamp(args.cutoff)]
    if args.protocol or args.protocol_new:
        run_protocol(px, args.protocol_new)
    else:
        run_backtest(px, args.grid)


if __name__ == "__main__":
    main()
