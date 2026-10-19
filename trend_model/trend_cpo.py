"""
CPO (Conditional Parameter Optimization, following Ernest Chan) for the trend-following model - Xetra stocks, long
================================================================================================================

Idea: Instead of choosing once and for all "200-day high, hold for 20 days", a model decides for each signal,
based on the situation on the signal day (stock and overall market), which exit can be expected to give the best
return per holding day - and whether the signal is worth it at all.

Entries   : 15 variants from 8 families (breakout to an N-day high, price above a moving average, crossing
            of two averages, time-series momentum, cross-sectional momentum 12-1, Bollinger breakout,
            breakout with a volume surge), see ENTRY_RULES.
Exits     : 12 variants - fixed holding period 5 to 120 days, trend break (N-day low), trailing stop.
Model     : one gradient boosting model for all combinations. Target: return per holding day after costs.
            Features: return/volatility/distance to the high/turnover of the stock, situation of the overall market
            (trend, volatility, share of stocks above the 200-day average, number of breakouts, dispersion),
            type of entry and exit, plus a purely random feature (placebo) as a control.
Testing   : walk-forward - for each test year from FIRST_TEST_YEAR onwards, training uses only signals whose trades
            were closed before the start of the year. Then a portfolio as in the trend model (50,000 EUR,
            max. 10 positions, IBKR fees, slippage).
Comparison: fixed variant of the trend model, always 120 days, best fixed combination chosen anew every year,
            random choice (same number of signals or random exit), buy and hold.

Usage (in the folder trend_model):
    ..\\.venv\\python.exe trend_cpo.py            walk-forward backtest (a few minutes)
    ..\\.venv\\python.exe trend_cpo.py --today    today's signals with the exit chosen by the model

Results in results_cpo/: comparison.csv, years.csv, grid_sharpe.csv/.png (all fixed combinations),
chosen_exits.csv, features.csv (importance), hit_quality.csv, trades_*.csv.

Note: The price cache contains only stocks that are still tradable today (survivorship bias) - this flatters all
variants equally; for comparing the variants with each other it is less important.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import trend_model as tm

warnings.filterwarnings("ignore")

# ======================================================================================= Settings
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results_cpo"

SIGNAL_START = "2017-09-01"      # first tradable stocks (250 days of price history in advance)
FIRST_TEST_YEAR = 2020           # walk-forward: 2020, 2021, ... (training always only with the years before)

ENTRY_RULES = {                    # name: (family, parameters)
    "High 20": ("high", 20),
    "High 50": ("high", 50),
    "High 100": ("high", 100),
    "High 200": ("high", 200),
    "High 250 (52 weeks)": ("high", 250),
    "Above MA 50": ("ma", 50),
    "Above MA 100": ("ma", 100),
    "Above MA 200": ("ma", 200),
    "Cross 20/50": ("cross", (20, 50)),
    "Cross 50/200": ("cross", (50, 200)),
    "Momentum 6M positive": ("tsmom", 126),
    "Momentum 12M positive": ("tsmom", 252),
    "Top-10% Momentum 12-1": ("xsmom", 252),
    "Bollinger 20/2": ("bollinger", 20),
    "High 50 + volume surge": ("volume_surge", 50),
}
EXIT_RULES = {
    "Time 5": {"exit_kind": "time", "days": 5},
    "Time 10": {"exit_kind": "time", "days": 10},
    "Time 20": {"exit_kind": "time", "days": 20},
    "Time 40": {"exit_kind": "time", "days": 40},
    "Time 60": {"exit_kind": "time", "days": 60},
    "Time 120": {"exit_kind": "time", "days": 120},
    "Trend 10": {"exit_kind": "trend", "days": 10},           # close below the 10-day low
    "Trend 20": {"exit_kind": "trend", "days": 20},
    "Trend 50": {"exit_kind": "trend", "days": 50},
    "Trailing 10%": {"exit_kind": "trailing", "pct": 0.10},     # 10 % below the highest close since entry
    "Trailing 15%": {"exit_kind": "trailing", "pct": 0.15},
    "Trailing 25%": {"exit_kind": "trailing", "pct": 0.25},
    # combination: fixed holding period, out earlier if the price falls X % below the highest close since entry
    "Time 40 + stop 10%": {"exit_kind": "combo", "days": 40, "pct": 0.10},
    "Time 40 + stop 15%": {"exit_kind": "combo", "days": 40, "pct": 0.15},
    "Time 60 + stop 10%": {"exit_kind": "combo", "days": 60, "pct": 0.10},
    "Time 60 + stop 15%": {"exit_kind": "combo", "days": 60, "pct": 0.15},
    "Time 120 + stop 10%": {"exit_kind": "combo", "days": 120, "pct": 0.10},
    "Time 120 + stop 15%": {"exit_kind": "combo", "days": 120, "pct": 0.15},
}
MAX_HOLDING_DAYS = 250
BASIS = ("High 200", "Time 20")  # best variant of the trend model in the learning period (200-day high, short time)
LONG_MODE = ("High 200", "Time 120")
VOLUME_SURGE = 2.0              # "High 50 + volume surge": shares at least 2x the median of the last 50 days
MIN_SCORE_BP = 0.0               # CPO entry: only signals with an expected return per day > 0 (after costs)
TARGETS = {                        # what the model predicts (both are calculated and compared)
    "per day": "absolute",         # return per holding day after costs
    "relative": "relative",        # the same minus the mean of all signals of the same month: separates the selection among
}                                # simultaneous signals from the general market trend
RANDOM_RUNS = 20
SEED = 11
MAX_TRAIN_ROWS = 600_000       # more training rows: random selection (run time)
MODEL = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
              l2_regularization=1.0, random_state=SEED)
COSTS = 2 * (tm.SLIPPAGE_BPS / 1e4 + tm.cost_pct())   # purchase + sale, as in the trend model (broker there)

EX_NAMES = list(EXIT_RULES)
EN_NAMES = list(ENTRY_RULES)


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Data and signals
def load_data() -> dict:
    m = tm.load_market("stocks")
    raw = pd.read_pickle(tm.SCREENER_CACHE / tm.STOCK_FILE)
    m["volume"] = raw["volume"].reindex(index=m["close"].index, columns=m["close"].columns).astype(float)
    return m


def _first(cond: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """First day on which the condition holds (on the previous day it did not hold, but the previous day could already be calculated)."""
    return cond & ~cond.shift(1, fill_value=False) & valid.shift(1, fill_value=False)


def entry_signals(m: dict, names: list[str] | None = None) -> dict[str, np.ndarray]:
    c = m["close"]
    idx = c.index
    out = {}
    for name, (fam, n) in ENTRY_RULES.items():
        if names is not None and name not in names:
            continue
        if fam in ("high", "volume_surge"):
            hi = c.rolling(n, min_periods=n).max().shift(1)
            cond = c > hi
            if fam == "volume_surge":
                v = m["volume"]
                cond &= v > VOLUME_SURGE * v.rolling(50, min_periods=30).median().shift(1)
            ev = _first(cond, hi.notna())
        elif fam == "ma":
            s = c.rolling(n, min_periods=n).mean()
            ev = _first(c > s, s.notna())
        elif fam == "cross":
            f, sl = n
            a, b = c.rolling(f, min_periods=f).mean(), c.rolling(sl, min_periods=sl).mean()
            ev = _first(a > b, a.notna() & b.notna())
        elif fam == "tsmom":
            r = c / c.shift(n) - 1
            ev = _first(r > 0, r.notna())
        elif fam == "bollinger":
            s, sd = c.rolling(n, min_periods=n).mean(), c.rolling(n, min_periods=n).std()
            ev = _first(c > s + 2 * sd, sd.notna())
        elif fam == "xsmom":                              # month end: new in the best 10 % (12-1 months)
            score = (c.shift(21) / c.shift(n) - 1).where(m["elig"])
            top = score.rank(axis=1, pct=True) >= 0.9
            per = idx.to_period("M")
            me = pd.Series(idx, index=idx).groupby(per).transform("max").to_numpy() == idx.to_numpy()
            me[-1] = False                                  # current month not yet over
            t_me = top[me]
            ev = (t_me & ~t_me.shift(1, fill_value=False)).reindex(idx, fill_value=False)
        else:
            raise ValueError(fam)
        ok = (ev & m["elig"]).to_numpy().copy()
        ok[idx < pd.Timestamp(SIGNAL_START)] = False
        out[name] = ok
    return out


def exits_for(p: np.ndarray, lows: dict, e: int) -> tuple[np.ndarray, np.ndarray]:
    """Exit day per exit variant for an entry at the close of day e; open = price series ends before."""
    n = len(p)
    x = np.empty(len(EX_NAMES), dtype=np.int64)
    opn = np.zeros(len(EX_NAMES), dtype=bool)
    end = min(n - 1, e + MAX_HOLDING_DAYS)
    seg = p[e + 1:end + 1]
    run_max = np.maximum.accumulate(np.concatenate([[p[e]], seg]))[1:] if len(seg) else seg
    draw = seg / run_max - 1 if len(seg) else seg
    for k, name in enumerate(EX_NAMES):
        r = EXIT_RULES[name]
        if r["exit_kind"] == "time":
            t = e + r["days"]
        elif r["exit_kind"] == "combo":
            hit = np.flatnonzero(draw[:r["days"]] <= -r["pct"])
            t = e + 1 + hit[0] if len(hit) else e + r["days"]
        else:
            if r["exit_kind"] == "trend":
                hit = np.flatnonzero(seg < lows[r["days"]][e + 1:end + 1])
                t = e + 1 + hit[0] + 1 if len(hit) else None    # signal at the close, sale at the next close
            else:
                hit = np.flatnonzero(draw <= -r["pct"])
                t = e + 1 + hit[0] if len(hit) else None        # stop is placed in the market: sale on the same day
            if t is None:
                t = e + MAX_HOLDING_DAYS
        if t > n - 1:
            t, opn[k] = n - 1, True
        x[k] = t
    return x, opn


def all_trades(m: dict, sigs: dict) -> pd.DataFrame:
    """Each signal with the result of all 12 exits (without portfolio limits)."""
    c = m["close"]
    pf = c.ffill()
    P = pf.to_numpy()
    lows = {n: pf.rolling(n).min().shift(1).to_numpy()
            for n in {r["days"] for r in EXIT_RULES.values() if r["exit_kind"] == "trend"}}
    has = c.notna().to_numpy()
    last = len(c) - 1 - has[::-1].argmax(axis=0)
    rows_meta, X, O = [], [], []
    for v, name in enumerate(EN_NAMES):
        if name not in sigs:
            continue
        ts, js = np.nonzero(sigs[name])
        for t, j in zip(ts, js):
            e = t + 1
            if e > len(c) - 1 or e > last[j] or not np.isfinite(P[e, j]):
                continue
            x, opn = exits_for(P[:, j], {k: a[:, j] for k, a in lows.items()}, e)
            rows_meta.append((v, t, j, e))
            X.append(x)
            O.append(opn)
    meta = pd.DataFrame(rows_meta, columns=["entry", "t_signal", "j", "e"]).astype(np.int64)
    if not len(meta):                                       # no signals (e.g. on the first protocol day)
        k = len(EX_NAMES)
        meta.attrs.update(X=np.zeros((0, k), dtype=np.int64), OPEN=np.zeros((0, k), dtype=bool),
                          R=np.zeros((0, k)), D=np.ones((0, k)))
        return meta
    X, O = np.array(X), np.array(O)
    pe = P[meta["e"].to_numpy(), meta["j"].to_numpy()]
    px = P[X, meta["j"].to_numpy()[:, None]]
    meta.attrs.update(X=X, OPEN=O, R=px / pe[:, None] - 1, D=np.maximum(X - meta["e"].to_numpy()[:, None], 1))
    return meta


# ======================================================================================= Features
def feature_panels(m: dict) -> tuple[dict, pd.DataFrame]:
    """Features as panels (day x stock) and market features (day) - calculate once, query as often as needed."""
    c, elig = m["close"], m["elig"]
    pf = c.ffill()
    lr = np.log(pf).diff()
    ann = np.sqrt(252)
    F = {}
    for n in (5, 20, 60, 126, 252):
        F[f"return_{n}"] = pf / pf.shift(n) - 1
    v20, v60 = lr.rolling(20).std() * ann, lr.rolling(60).std() * ann
    F["volatility_20"], F["volatility_60"], F["volatility_ratio"] = v20, v60, v20 / v60
    F["distance_high_252"] = pf / pf.rolling(252, min_periods=120).max() - 1
    sma50, sma200 = pf.rolling(50).mean(), pf.rolling(200).mean()
    F["above_ma_200"] = pf / sma200 - 1
    F["ma_50_to_200"] = sma50 / sma200 - 1
    F["breakout_strength_20"] = pf / pf.rolling(20).max().shift(1) - 1
    F["turnover_log"] = np.log10(m["turnover"])
    vol = m["volume"].where(m["volume"] > 0)
    F["volume_surge"] = np.log(vol.rolling(5, min_periods=3).mean() / vol.rolling(60, min_periods=30).median().shift(5))
    F["rank_momentum_126"] = (pf / pf.shift(126) - 1).where(elig).rank(axis=1, pct=True)
    # overall market: equal-weighted index of the tradable stocks
    mret = lr.where(elig.shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ix = np.exp(mret.cumsum())
    M = pd.DataFrame(index=c.index)
    for n in (20, 60, 252):
        M[f"market_return_{n}"] = ix / ix.shift(n) - 1
    M["market_volatility_20"] = mret.rolling(20).std() * ann
    M["market_volatility_60"] = mret.rolling(60).std() * ann
    M["market_distance_high"] = ix / ix.rolling(252, min_periods=60).max() - 1
    M["market_breadth_200"] = (pf > sma200).astype(float).where(elig & sma200.notna()).mean(axis=1)
    M["market_breakouts_50"] = (pf > pf.rolling(50).max().shift(1)).astype(float).where(elig).mean(axis=1)
    M["market_dispersion_20"] = (pf / pf.shift(20) - 1).where(elig).std(axis=1)
    return {k: df.to_numpy() for k, df in F.items()}, M


def sample_features(panels: tuple[dict, pd.DataFrame], t: np.ndarray, j: np.ndarray) -> pd.DataFrame:
    F, M = panels
    out = pd.DataFrame({k: a[t, j] for k, a in F.items()})
    for k in M.columns:
        out[k] = M[k].to_numpy()[t]
    return out


def features(m: dict, meta: pd.DataFrame, panels: tuple | None = None) -> pd.DataFrame:
    panels = panels or feature_panels(m)
    out = sample_features(panels, meta["t_signal"].to_numpy(), meta["j"].to_numpy())
    out["entry"] = meta["entry"].to_numpy()
    out["entry_days"] = [ENTRY_RULES[EN_NAMES[v]][1][1] if isinstance(ENTRY_RULES[EN_NAMES[v]][1], tuple)
                            else ENTRY_RULES[EN_NAMES[v]][1] for v in out["entry"]]
    out["placebo"] = np.random.default_rng(SEED).normal(size=len(out))
    return out.replace([np.inf, -np.inf], np.nan)


def stack(feat: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """One row per exit for each signal (features of the exit appended)."""
    k = len(EX_NAMES)
    f = feat.iloc[np.repeat(rows, k)].reset_index(drop=True)
    exit_kind = {"time": 0, "trend": 1, "trailing": 2, "combo": 3}
    f["exit_type"] = np.tile([exit_kind[EXIT_RULES[x]["exit_kind"]] for x in EX_NAMES], len(rows))
    f["exit_days"] = np.tile([EXIT_RULES[x].get("days", 0) for x in EX_NAMES], len(rows))
    f["exit_stop"] = np.tile([EXIT_RULES[x].get("pct", 0) * 100 for x in EX_NAMES], len(rows))
    return f


def target(meta: pd.DataFrame) -> np.ndarray:
    """Return per holding day after costs in basis points (signals x exits)."""
    return (meta.attrs["R"] - COSTS) / meta.attrs["D"] * 1e4


def model_stamp() -> dict:
    """Identifier of the data and settings that stored models match."""
    import sklearn
    return {"prices": (tm.SCREENER_CACHE / tm.STOCK_FILE).stat().st_mtime, "test_year": FIRST_TEST_YEAR,
            "entries": EN_NAMES, "exits": EX_NAMES, "targets": TARGETS, "model": MODEL,
            "sklearn": sklearn.__version__}


def save_models(models: dict) -> None:
    import pickle
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "models.pkl", "wb") as f:
        pickle.dump({"stamp": model_stamp(), "models": models}, f)


def load_models() -> dict | None:
    import pickle
    path = OUT_DIR / "models.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    return obj["models"] if obj.get("stamp") == model_stamp() else None


def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(categorical_features=["entry", "exit_type"], **MODEL)


# ======================================================================================= Walk-forward
def _demean(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return v - pd.Series(v).groupby(groups).transform("mean").to_numpy()


def walk_forward(m: dict, meta: pd.DataFrame, feat: pd.DataFrame, target_kind: str = "absolute",
                 models_out: dict | None = None) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Prediction (signals x exits) for all signals from FIRST_TEST_YEAR onwards, trained per year only on prior years.
    target 'relative': training target minus the mean of all trades of the same signal month (without market timing)."""
    cal = m["close"].index
    y_all = target(meta)
    K = len(EX_NAMES)
    sd = cal[meta["t_signal"].to_numpy()]
    month = np.asarray(sd.year * 12 + sd.month)
    X, O = meta.attrs["X"], meta.attrs["OPEN"]
    sig_year = cal[meta["t_signal"].to_numpy()].year
    pred = np.full(y_all.shape, np.nan)
    rng = np.random.default_rng(SEED)
    imp_rows, ic_rows = [], []
    years = range(FIRST_TEST_YEAR, cal[-1].year + 1)
    for Y in years:
        y0 = pd.Timestamp(f"{Y}-01-01")
        done = (cal.values[X] < np.datetime64(y0)) & ~O    # trade closed before the start of the year
        tr_sig = np.flatnonzero(done.any(axis=1))
        te_sig = np.flatnonzero(sig_year == Y)
        if len(te_sig) == 0 or len(tr_sig) < 1000:
            continue
        Xtr = stack(feat, tr_sig)
        ytr = y_all[tr_sig].ravel()
        keep = done[tr_sig].ravel() & np.isfinite(ytr)
        idx = np.flatnonzero(keep)
        if len(idx) > MAX_TRAIN_ROWS:
            idx = np.sort(rng.choice(idx, MAX_TRAIN_ROWS, replace=False))
        lo, hi = np.percentile(ytr[idx], [1, 99])
        yfit = np.clip(ytr[idx], lo, hi)
        if target_kind == "relative":
            yfit = _demean(yfit, np.repeat(month[tr_sig], K)[idx])
        model = make_model()
        t0 = time.time()
        model.fit(Xtr.iloc[idx], yfit)
        if models_out is not None:
            models_out[Y] = model
        Xte = stack(feat, te_sig)
        p = model.predict(Xte).reshape(len(te_sig), len(EX_NAMES))
        pred[te_sig] = p
        # hit quality: rank correlation of prediction / later result (closed trades only)
        yte = y_all[te_sig]
        okte = ~O[te_sig]
        ic_all = pd.Series(p[okte]).corr(pd.Series(yte[okte]), method="spearman")
        best = p.argmax(axis=1)
        okb = okte[np.arange(len(te_sig)), best]
        ic_sig = pd.Series(p.max(axis=1)[okb]).corr(pd.Series(yte[np.arange(len(te_sig)), best][okb]),
                                                    method="spearman")
        g = np.repeat(month[te_sig], K).reshape(len(te_sig), K)[okte]
        ic_mon = pd.Series(_demean(p[okte], g)).corr(pd.Series(_demean(np.clip(yte[okte], lo, hi), g)),
                                                     method="spearman")
        ic_rows.append({"target": target_kind, "year": Y, "training_signals": len(tr_sig), "test_signals": len(te_sig),
                        "rank_correlation_all": ic_all, "rank_correlation_in_month": ic_mon,
                        "rank_correlation_chosen": ic_sig})
        # importance of the features in the test year (permutation, sample)
        try:
            from sklearn.inspection import permutation_importance
            rows = np.flatnonzero(okte.ravel())
            if len(rows) > 2000:
                rows = rng.choice(rows, min(len(rows), 40_000), replace=False)
                ysc = np.clip(yte.ravel()[rows], lo, hi)
                if target_kind == "relative":
                    ysc = _demean(ysc, np.repeat(month[te_sig], K)[rows])
                pi = permutation_importance(model, Xte.iloc[rows], ysc,
                                            n_repeats=3, random_state=SEED, scoring="neg_mean_squared_error")
                for k, col in enumerate(Xte.columns):
                    imp_rows.append({"target": target_kind, "year": Y, "feature": col, "importance": pi.importances_mean[k]})
        except Exception as ex:                             # importance is only additional information
            log(f"  importance {Y} not calculated: {ex}")
        log(f"  [{target_kind}] test year {Y}: trained with {len(idx):,} rows ({time.time() - t0:.0f} s), "
            f"rank correlation {ic_all:+.3f}, in month {ic_mon:+.3f}".replace(",", "."))
    return pred, pd.DataFrame(ic_rows), pd.DataFrame(imp_rows)


# ======================================================================================= Strategies
def to_trades(meta: pd.DataFrame, rows: np.ndarray, exits: np.ndarray, rank: np.ndarray) -> pd.DataFrame:
    """Format of trend_model.simulate: j, t_signal, e, x, exit_price, return, days, reason, rank."""
    X, R = meta.attrs["X"], meta.attrs["R"]
    return pd.DataFrame({"j": meta["j"].to_numpy()[rows], "t_signal": meta["t_signal"].to_numpy()[rows],
                         "e": meta["e"].to_numpy()[rows], "x": X[rows, exits], "exit_price": np.nan,
                         "return": R[rows, exits], "days": X[rows, exits] - meta["e"].to_numpy()[rows],
                         "reason": [EX_NAMES[k] for k in exits], "rank": rank})


def one_per_day(meta: pd.DataFrame, rows: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Several entries on the same day for the same stock: only the one with the highest rating."""
    d = pd.DataFrame({"r": rows, "s": score, "t": meta["t_signal"].to_numpy()[rows], "j": meta["j"].to_numpy()[rows]})
    d = d.sort_values("s", ascending=False).drop_duplicates(["t", "j"])
    return d["r"].to_numpy(), d["s"].to_numpy()


def summarize(name: str, sim: dict, a: pd.Timestamp) -> dict:
    out = {"strategy": name}
    out.update(tm.metrics(sim["equity"], sim["trades"], str(a.date()), None))
    return out


def yearly(eq: pd.Series) -> pd.Series:
    e = eq.dropna()
    ye = e.groupby(e.index.year).last()
    first = e.iloc[0]
    return (ye / ye.shift(1).fillna(first) - 1) * 100


def run_backtest() -> None:
    t_start = time.time()
    m = load_data()
    cal = m["close"].index
    P = m["close"].ffill().to_numpy()
    mom = (P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1)
    log("Signals of all entries ...")
    sigs = entry_signals(m)
    meta = all_trades(m, sigs)
    counts = meta.groupby("entry").size().rename(index=dict(enumerate(EN_NAMES)))
    log(f"{len(meta):,} signals, per signal {len(EX_NAMES)} exits calculated".replace(",", "."))
    feat = features(m, meta)
    test_start = pd.Timestamp(f"{FIRST_TEST_YEAR}-01-01")
    sig_t = meta["t_signal"].to_numpy()
    in_test = cal[sig_t] >= test_start
    en = meta["entry"].to_numpy()

    def sim(tr: pd.DataFrame, start: pd.Timestamp = test_start) -> dict:
        return tm.simulate(m, tr, "long", start)

    # ---- 1. Grid of all fixed combinations (over the whole period, causal) -> selection per previous years
    log(f"Grid: {len(EN_NAMES) * len(EX_NAMES)} fixed combinations in the portfolio ...")
    grid_eq, grid_rows = {}, []
    sig_start = pd.Timestamp(SIGNAL_START)
    for v, en_name in enumerate(EN_NAMES):
        rows = np.flatnonzero(en == v)
        for k, ex_name in enumerate(EX_NAMES):
            tr = to_trades(meta, rows, np.full(len(rows), k), mom[sig_t[rows], meta["j"].to_numpy()[rows]])
            eq = sim(tr, sig_start)["equity"]
            grid_eq[(en_name, ex_name)] = eq
            r = {"entry": en_name, "exit": ex_name, "signals": len(rows)}
            for lbl, a, b in (("to_2022", sig_start, pd.Timestamp("2022-12-31")),
                              ("from_2023", pd.Timestamp("2023-01-01"), None),
                              ("total", sig_start, None)):
                r[f"sharpe_{lbl}"] = tm._sharpe(eq, a, b)
            grid_rows.append(r)
    grid = pd.DataFrame(grid_rows)
    tm.save_csv(grid, "grid_sharpe.csv")
    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({f"{a} | {b}": eq for (a, b), eq in grid_eq.items()}).to_pickle(OUT_DIR / "grid_curves.pkl")
    grid_chart(grid)

    # walk-forward: per test year the fixed combination with the best Sharpe ratio of all previous years
    wf_rows, wf_choice = [], {}
    for Y in range(FIRST_TEST_YEAR, cal[-1].year + 1):
        best = max(grid_eq, key=lambda k: np.nan_to_num(tm._sharpe(grid_eq[k], sig_start, pd.Timestamp(f"{Y - 1}-12-31")),
                                                        nan=-9))
        wf_choice[Y] = best
        v, k = EN_NAMES.index(best[0]), EX_NAMES.index(best[1])
        rows = np.flatnonzero((en == v) & (cal[sig_t].year == Y))
        wf_rows.append(to_trades(meta, rows, np.full(len(rows), k), mom[sig_t[rows], meta["j"].to_numpy()[rows]]))
    wf_trades = pd.concat(wf_rows, ignore_index=True)

    # ---- 2. CPO walk-forward (both targets)
    preds, ics, imps, models = {}, [], [], {}
    for label, target_kind in TARGETS.items():
        log(f"CPO walk-forward, target '{label}' ...")
        models[label] = {}
        preds[label], ic_, imp_ = walk_forward(m, meta, feat, target_kind, models_out=models[label])
        ics.append(ic_)
        imps.append(imp_)
    save_models(models)                                     # for cpo_stress_test.py
    ic, imp = pd.concat(ics, ignore_index=True), pd.concat(imps, ignore_index=True)

    strategies: dict[str, pd.DataFrame] = {}
    vb, kb = EN_NAMES.index(BASIS[0]), EX_NAMES.index(BASIS[1])
    base_rows = np.flatnonzero((en == vb) & in_test)
    base_rank = mom[sig_t[base_rows], meta["j"].to_numpy()[base_rows]]
    strategies[f"Fixed: {BASIS[0]} + {BASIS[1]} (trend model)"] = to_trades(meta, base_rows, np.full(len(base_rows), kb), base_rank)
    kl = EX_NAMES.index(LONG_MODE[1])
    strategies[f"Fixed: {LONG_MODE[0]} + {LONG_MODE[1]}"] = to_trades(meta, base_rows, np.full(len(base_rows), kl), base_rank)
    strategies["Walk-forward: best fixed combination of the previous years"] = wf_trades
    cpo_rows_by = {}
    for label, pred in preds.items():
        ok_pred = np.isfinite(pred).all(axis=1)
        rows = base_rows[ok_pred[base_rows]]
        strategies[f"CPO[{label}] exit (entry {BASIS[0]})"] = to_trades(
            meta, rows, pred[rows].argmax(axis=1), mom[sig_t[rows], meta["j"].to_numpy()[rows]])
        allr = np.flatnonzero(in_test & ok_pred)
        r1, s1 = one_per_day(meta, allr, pred[allr, kb])
        sel = s1 > MIN_SCORE_BP
        strategies[f"CPO[{label}] entry (exit {BASIS[1]})"] = to_trades(meta, r1[sel], np.full(sel.sum(), kb), s1[sel])
        r2, s2 = one_per_day(meta, allr, pred[allr].max(axis=1))
        sel2 = s2 > MIN_SCORE_BP
        cpo_rows_by[label] = r2[sel2]
        strategies[f"CPO[{label}] entry + exit"] = to_trades(meta, r2[sel2], pred[r2[sel2]].argmax(axis=1), s2[sel2])

    # ---- 3. Chance for comparison
    rng = np.random.default_rng(SEED)
    rand_specs = {
        f"Chance: random exit (entry {BASIS[0]})":
            lambda: to_trades(meta, base_rows, rng.integers(0, len(EX_NAMES), len(base_rows)), base_rank),
    }
    for label, cr in cpo_rows_by.items():
        rand_specs[f"Chance: same number of signals as CPO[{label}], random exit"] = \
            (lambda cr=cr: random_like(meta, cal, np.flatnonzero(in_test), cr, rng))

    log("Portfolio calculation of the strategies ...")
    rows_out, curves, year_tab, trades_out = [], {}, {}, {}
    for name, tr in strategies.items():
        s = sim(tr)
        rows_out.append(summarize(name, s, test_start))
        curves[name] = s["equity"]
        year_tab[name] = yearly(s["equity"])
        trades_out[name] = s["trades"]
    for name, fn in rand_specs.items():
        res = [summarize(name, sim(fn()), test_start) for _ in range(RANDOM_RUNS)]
        d = pd.DataFrame(res)
        r = d.drop(columns="strategy").mean().to_dict()
        r["strategy"] = f"{name} (mean of {RANDOM_RUNS})"
        r["sharpe_range"] = f"{d['sharpe'].min():.2f} to {d['sharpe'].max():.2f}"
        rows_out.append(r)
    if m["bench"] is not None:
        b = m["bench"][m["bench"].index >= test_start].ffill().dropna()
        beq = tm.CAPITAL * b / b.iloc[0]
        r = {"strategy": f"Buy and hold {tm.BENCHMARK['stocks']}"}
        r.update(tm.metrics(beq, pd.DataFrame(), str(test_start.date()), None))
        rows_out.append(r)
        curves[r["strategy"]] = beq
        year_tab[r["strategy"]] = yearly(beq)
    comp = pd.DataFrame(rows_out)
    comp = comp[["strategy"] + [c for c in comp.columns if c != "strategy"]]
    tm.save_csv(comp, "comparison.csv")
    years = pd.DataFrame(year_tab).T.round(1)
    tm.save_csv(years, "years.csv", index=True)
    tm.save_csv(ic, "hit_quality.csv")
    if not imp.empty:
        tm.save_csv(imp.pivot_table(index="feature", columns=["target", "year"], values="importance")
                    .assign(mean=lambda d: d.mean(axis=1)).sort_values("mean", ascending=False), "features.csv",
                    index=True)
    for name, tl in trades_out.items():
        if name.startswith("CPO") and tl is not None and not tl.empty:
            fn = "trades_" + name.split("(")[0].strip().lower()
            for a, b in ((" ", "_"), ("+", "and"), ("[", "_"), ("]", ""), ("__", "_"), ("__", "_")):
                fn = fn.replace(a, b)
            tm.save_csv(tl.assign(name=tl["ticker"].map(m["names"])), fn + ".csv")
    # Which exits and entries does CPO choose per year?
    chosen, ent = {}, {}
    for label in preds:
        for kind in ("entry + exit", f"exit (entry {BASIS[0]})"):
            ch = strategies[f"CPO[{label}] {kind}"]
            chosen[f"CPO[{label}] {kind}"] = pd.crosstab(cal[ch["t_signal"]].year, ch["reason"], normalize="index") \
                .reindex(columns=EX_NAMES, fill_value=0)
        ch = strategies[f"CPO[{label}] entry + exit"]
        e_ = pd.crosstab(cal[ch["t_signal"]].year, en[cpo_rows_by[label]], normalize="index")
        e_.columns = [EN_NAMES[c] for c in e_.columns]
        ent[f"CPO[{label}] entry + exit"] = e_
    tm.save_csv(pd.concat(chosen).mul(100).round(1), "chosen_exits.csv", index=True)
    tm.save_csv(pd.concat(ent).mul(100).round(1), "chosen_entries.csv", index=True)
    chart(curves)
    report(comp, years, grid, wf_choice, ic, imp, chosen, ent, counts, time.time() - t_start)


def random_like(meta: pd.DataFrame, cal: pd.DatetimeIndex, pool: np.ndarray, chosen: np.ndarray, rng) -> pd.DataFrame:
    """Same number of signals per year as CPO, randomly from all entries, random exit and rank."""
    yr_pool = cal[meta["t_signal"].to_numpy()[pool]].year
    yr_ch = pd.Series(cal[meta["t_signal"].to_numpy()[chosen]].year).value_counts()
    rows = []
    for y, n in yr_ch.items():
        cand = pool[yr_pool == y]
        rows.append(rng.choice(cand, min(n, len(cand)), replace=False))
    rows = np.concatenate(rows) if rows else np.array([], dtype=int)
    return to_trades(meta, rows, rng.integers(0, len(EX_NAMES), len(rows)), rng.random(len(rows)))


# ======================================================================================= Output
def grid_chart(grid: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5))
    for ax, col, title in ((axes[0], "sharpe_to_2022", "Sharpe 2017-2022"), (axes[1], "sharpe_from_2023", "Sharpe from 2023")):
        pv = grid.pivot(index="entry", columns="exit", values=col).reindex(index=EN_NAMES, columns=EX_NAMES)
        im = ax.imshow(pv.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(EX_NAMES)), EX_NAMES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(EN_NAMES)), EN_NAMES if ax is axes[0] else [""] * len(EN_NAMES), fontsize=8)
        for (i, k), val in np.ndenumerate(pv.to_numpy()):
            if np.isfinite(val):
                ax.text(k, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.7)
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "grid_sharpe.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def chart(curves: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in curves.items():
        kw = {"color": "grey"} if name.startswith("Buy and hold") else ({"lw": 2.2} if name.startswith("CPO") else {"ls": "--"})
        ax.plot(eq.index, eq.values / 1000, label=name, **kw)
    ax.set_yscale("log")
    ax.set_title(f"Walk-forward from {FIRST_TEST_YEAR}: account in thousand EUR (log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "trend_cpo.png", dpi=110)
    plt.close(fig)


def report(comp, years, grid, wf_choice, ic, imp, chosen, ent, counts, secs) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    print("\n" + "=" * 120)
    print(f"CPO TREND FOLLOWING XETRA (long)  walk-forward from {FIRST_TEST_YEAR}, account {tm.CAPITAL:,.0f} EUR, "
          f"max. {tm.MAX_POSITIONS} positions".replace(",", "."))
    print(f"Costs: {tm.broker_text()}, slippage {tm.SLIPPAGE_BPS} bp per order")
    print("=" * 120)
    print("\nSignals per entry (2017-today):")
    print(counts.to_string())
    cols = [c for c in ["strategy", "return_pa_pct", "sharpe", "sharpe_range", "max_drawdown_pct", "trades",
                        "hit_rate", "per_trade_pct", "holding_days"] if c in comp.columns]
    print(f"\nComparison in the test period from {FIRST_TEST_YEAR} (everything out of sample):")
    print(comp[cols].round(2).to_string(index=False))
    print("\nReturn per calendar year (%):")
    print(years.to_string())
    print("\nWalk-forward: best fixed combination of the previous years in each case:")
    for y, (a, b) in wf_choice.items():
        print(f"  {y}: {a} + {b}")
    piv = grid.pivot(index="entry", columns="exit", values="sharpe_to_2022").reindex(index=EN_NAMES, columns=EX_NAMES)
    piv2 = grid.pivot(index="entry", columns="exit", values="sharpe_from_2023").reindex(index=EN_NAMES, columns=EX_NAMES)
    print("\nGrid of fixed combinations - Sharpe 2017-2022:")
    print(piv.round(2).to_string())
    print("\nGrid of fixed combinations - Sharpe from 2023:")
    print(piv2.round(2).to_string())
    rc = pd.Series(piv.to_numpy().ravel()).corr(pd.Series(piv2.to_numpy().ravel()), method="spearman")
    print(f"Rank correlation of the combinations between both periods: {rc:+.2f} "
          "(close to 0 = the ranking is chance, close to 1 = stable)")
    if not ic.empty:
        print("\nHit quality of the CPO prediction in the respective test year (rank correlation prediction / result;")
        print("'in_month' = only differences between simultaneous signals, 'chosen' = among the chosen ones):")
        print(ic.round(3).to_string(index=False))
    if not imp.empty:
        for z, g in imp.groupby("target"):
            mi = g.groupby("feature")["importance"].mean().sort_values(ascending=False)
            pl = mi.get("placebo", np.nan)
            print(f"\nMost important features, target '{z}' (permutation, mean of the test years; 'placebo' = random feature):")
            print(mi.head(10).round(2).to_string())
            print(f"  placebo: {pl:.3g} (rank {list(mi.index).index('placebo') + 1} of {len(mi)})")
    for name, tab in chosen.items():
        print(f"\nChosen exits per year (% of signals), {name}:")
        print(tab.mul(100).round(0).astype(int).to_string())
    for name, tab in ent.items():
        print(f"\nChosen entries per year (%), {name}:")
        print(tab.mul(100).round(0).astype(int).to_string())
    print(f"\nFiles: {OUT_DIR}   (run time {secs / 60:.1f} min)")
    print("How to read: CPO is only worthwhile if it beats the fixed variant, the best fixed combination chosen anew every year")
    print("AND the random variants, the rank correlation is > 0 in most years and 'placebo' is far")
    print("behind.")


# ======================================================================================= Today
def today() -> None:
    m = load_data()
    cal = m["close"].index
    sigs = entry_signals(m)
    meta = all_trades(m, sigs)
    feat = features(m, meta)
    y = target(meta)
    X, O = meta.attrs["X"], meta.attrs["OPEN"]
    done = ~O
    rows = np.flatnonzero(done.any(axis=1))
    Xtr = stack(feat, rows)
    ytr = y[rows].ravel()
    keep = np.flatnonzero(done[rows].ravel() & np.isfinite(ytr))
    rng = np.random.default_rng(SEED)
    if len(keep) > MAX_TRAIN_ROWS:
        keep = np.sort(rng.choice(keep, MAX_TRAIN_ROWS, replace=False))
    lo, hi = np.percentile(ytr[keep], [1, 99])
    model = make_model().fit(Xtr.iloc[keep], np.clip(ytr[keep], lo, hi))
    # signals on the last trading day (entry at the next closing price) - all_trades discards them because the
    # entry day is still missing; therefore taken directly from the signals here
    t_last = len(cal) - 1
    cand = [(v, j) for v, name in enumerate(EN_NAMES) for j in np.flatnonzero(sigs[name][t_last])]
    print("\n" + "=" * 100)
    print(f"CPO SIGNALS of {cal[-1]:%d.%m.%Y} (purchase at the next closing price)")
    print("=" * 100)
    if not cand:
        print("No entry signal today.")
        return
    mt = pd.DataFrame({"entry": [v for v, _ in cand], "t_signal": t_last, "j": [j for _, j in cand], "e": t_last})
    ft = features(m, mt)
    p = model.predict(stack(ft, np.arange(len(mt)))).reshape(len(mt), len(EX_NAMES))
    out = pd.DataFrame({"ticker": m["close"].columns[mt["j"]], "name": m["names"].reindex(m["close"].columns[mt["j"]]).values,
                        "entry": [EN_NAMES[v] for v in mt["entry"]],
                        "best_exit": [EX_NAMES[k] for k in p.argmax(axis=1)],
                        "expected_bp_per_day": p.max(axis=1)})
    out = out.sort_values("expected_bp_per_day", ascending=False).drop_duplicates("ticker")
    out["buy"] = np.where(out["expected_bp_per_day"] > MIN_SCORE_BP, "yes", "no")
    print(out.round(1).head(25).to_string(index=False))
    tm.save_csv(out, "today.csv")
    print(f"\nModel trained with all closed trades up to {cal[-1]:%d.%m.%Y}. At most "
          f"{tm.MAX_POSITIONS} positions; with more candidates, those with the highest expectation.")


def main() -> None:
    ap = argparse.ArgumentParser(description="CPO for the trend-following model (Xetra stocks, long)")
    ap.add_argument("--today", action="store_true", help="today's signals with the exit chosen by the model")
    args = ap.parse_args()
    tm.OUT_DIR = OUT_DIR
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("scikit-learn missing:  ..\\.venv\\python.exe -m pip install scikit-learn")
    if args.today:
        today()
    else:
        run_backtest()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(1)
