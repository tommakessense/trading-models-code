"""
Stress test: CPO exit against fixed exits in simulated crashes, bear markets and choppy markets
================================================================================================================

Question: Does the exit chosen by CPO limit the losses better than fixed rules when the market behaves
differently than in the training data?

Procedure
1. As in trend_cpo.py, the CPO models are trained year by year ONLY on real data of the previous years
   (the result is cached in results_cpo/models.pkl).
2. Real significance: Is the lead of the CPO exit in the real data greater than chance?
   (Block bootstrap of the daily returns, paired against each fixed variant.)
3. Stress scenarios: An artificial shock is built into the real prices from several start dates in the test period
   - as a market move that hits each stock with its beta, plus random noise:
     Crash with recovery (V)       -35 % in 20 days, then +40 % in 60 days
     Bear market (L)              -45 % over 120 days, no recovery
     Sideways with false breakouts  +-12 % waves every 40 days over 160 days (poison for trend following)
     Flash-Crash                  -15 % in 3 days, recovery in 7 days
     Large dispersion              market unchanged, but individual stocks diverge strongly
   All signals, features and results are recalculated on the modified prices; the models
   do not know the scenarios (as in real life).
4. All strategies with the same entry (High 200) are compared: fixed holding periods, trend break,
   trailing stop and the two CPO variants - drawdown, return in the shock window, Sharpe ratio.

Usage (in the folder trend_model):  ..\\.venv\\python.exe cpo_stress_test.py      (after trend_cpo.py: approx. 2 minutes)
                                    ..\\.venv\\python.exe cpo_stress_test.py --entry "High 50 + volume surge"
                                    ..\\.venv\\python.exe cpo_stress_test.py --retrain   retrain the models
Results in results_cpo/: stress_single.csv, stress_summary.csv, significance.csv, stress.png
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
ENTRY = tc.BASIS[0]           # can be changed with --entry
SUFFIX = ""
FIXED_EXITS = ["Time 20", "Time 40", "Time 120", "Trend 50", "Trend 20", "Trailing 10%", "Trailing 15%", "Trailing 25%",
         "Time 40 + stop 10%", "Time 40 + stop 15%", "Time 60 + stop 10%", "Time 60 + stop 15%",
         "Time 120 + stop 10%", "Time 120 + stop 15%"]
START_DATES = ["2020-09-01", "2021-06-01", "2022-03-01", "2023-03-01", "2024-06-03", "2025-06-02"]
AFTERMATH_DAYS = 60                    # shock window + 60 trading days are evaluated separately
BOOTSTRAP = 2000
BLOCK = 20                       # block length (trading days) for the bootstrap
SEED = 5

SCENARIOS = {
    # name: list of phases (days, total market move as a factor, additional daily volatility of the market)
    "Crash with recovery (V)": {"phases": [(20, 0.65, 0.025), (60, 1.40, 0.015)]},
    "Bear market (L)": {"phases": [(120, 0.55, 0.020)]},
    "Sideways with false breakouts": {"wave": (160, 0.12, 40, 0.015)},   # days, amplitude (log), period, noise
    "Flash-Crash": {"phases": [(3, 0.85, 0.03), (7, 1 / 0.85, 0.015)]},
    "Large dispersion": {"dispersion": (120, 2.0, 0.35)},   # days, factor on individual moves, trend dispersion p. a.
}


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Models
def train_models(m: dict, refit: bool) -> dict:
    models = None if refit else tc.load_models()
    if models is not None:
        log("CPO models loaded from trend_cpo.py (results_cpo/models.pkl)")
        return models
    log("Training CPO models (all entries, real data only, per test year only previous years) ...")
    sigs = tc.entry_signals(m)
    meta = tc.all_trades(m, sigs)
    feat = tc.features(m, meta)
    models = {}
    for label, target in tc.TARGETS.items():
        models[label] = {}
        tc.walk_forward(m, meta, feat, target, models_out=models[label])
    tc.save_models(models)
    return models


# ======================================================================================= Scenarios
def market_parts(m: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    pf = m["close"].ffill()
    lr = np.log(pf).diff()
    f = lr.where(m["elig"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    beta = lr.apply(lambda s: s.cov(f) / f.var() if s.notna().sum() > 120 else np.nan).clip(0.3, 2.0).fillna(1.0)
    resid = lr - np.outer(f.to_numpy(), beta.to_numpy())
    return lr, f, beta, resid


def shock_path(spec: dict, rng) -> tuple[np.ndarray, float, float]:
    """Daily additional log return of the market; factor on individual moves; trend dispersion."""
    if "phases" in spec:
        parts = []
        for days, factor, noise in spec["phases"]:
            parts.append(np.full(days, np.log(factor) / days) + rng.normal(0, noise, days))
        return np.concatenate(parts), 1.0, 0.0
    if "wave" in spec:
        days, amp, period, noise = spec["wave"]
        level = amp * np.sin(2 * np.pi * np.arange(days + 1) / period)
        return np.diff(level) + rng.normal(0, noise, days), 1.0, 0.0
    days, mult, spread = spec["dispersion"]
    return np.zeros(days), mult, spread


def apply_scenario(m: dict, parts, spec: dict, start: str, rng) -> tuple[dict, tuple[int, int]]:
    lr, f, beta, resid = parts
    c = m["close"]
    cal = c.index
    t0 = int(np.searchsorted(cal, pd.Timestamp(start)))
    s, mult, spread = shock_path(spec, rng)
    t1 = min(t0 + len(s), len(cal))
    s = s[:t1 - t0]
    add = np.zeros(lr.shape)
    add[t0:t1] = np.outer(s, beta.to_numpy())
    if mult != 1.0:
        add[t0:t1] += (mult - 1) * np.nan_to_num(resid.to_numpy()[t0:t1])
    if spread > 0:                                      # each stock gets its own trend in the window
        drift = rng.normal(0, spread, lr.shape[1]) / 252   # dispersion of the annual trends (e.g. 35 % p. a.)
        add[t0:t1] += drift
    new_lr = np.nan_to_num(lr.to_numpy()) + add
    pf = c.ffill().to_numpy()
    first = np.argmax(~np.isnan(pf), axis=0)
    logp = np.log(pf[first, np.arange(pf.shape[1])]) + np.cumsum(np.where(np.arange(len(cal))[:, None] > first, new_lr, 0), axis=0)
    close = pd.DataFrame(np.exp(logp), index=cal, columns=c.columns).where(c.notna())
    m2 = dict(m)
    m2["close"] = close
    if m["bench"] is not None:
        b = m["bench"].ffill()
        blr = np.log(b).diff().fillna(0).to_numpy().copy()
        blr[t0:t1] += s
        m2["bench"] = pd.Series(np.exp(np.log(b.iloc[0]) + np.cumsum(blr)), index=cal).where(m["bench"].notna())
    return m2, (t0, t1)


# ======================================================================================= Strategies
def run_strategies(m: dict, models: dict) -> dict[str, pd.Series]:
    cal = m["close"].index
    test_start = pd.Timestamp(f"{tc.FIRST_TEST_YEAR}-01-01")
    sigs = tc.entry_signals(m, [ENTRY])
    meta = tc.all_trades(m, sigs)
    sig_t = meta["t_signal"].to_numpy()
    rows = np.flatnonzero(cal[sig_t] >= test_start)
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    rank = mom[sig_t[rows], meta["j"].to_numpy()[rows]]
    out = {}
    for ex in FIXED_EXITS:
        k = tc.EX_NAMES.index(ex)
        tr = tc.to_trades(meta, rows, np.full(len(rows), k), rank)
        out[f"Fixed {ex}"] = tm.simulate(m, tr, "long", test_start, with_log=False)["equity"]
    feat = tc.features(m, meta)
    years = cal[sig_t[rows]].year
    for label, mods in models.items():
        choice = np.full(len(rows), tc.EX_NAMES.index(tc.BASIS[1]))
        for Y, model in mods.items():
            sel = np.flatnonzero(years == Y)
            if len(sel):
                p = model.predict(tc.stack(feat, rows[sel])).reshape(len(sel), len(tc.EX_NAMES))
                choice[sel] = p.argmax(axis=1)
        tr = tc.to_trades(meta, rows, choice, rank)
        out[f"CPO[{label}]"] = tm.simulate(m, tr, "long", test_start, with_log=False)["equity"]
    if m["bench"] is not None:
        b = m["bench"][m["bench"].index >= test_start].ffill().dropna()
        out["Buy and hold DAX"] = tm.CAPITAL * b / b.iloc[0]
    return out


def max_dd(eq: pd.Series) -> float:
    e = eq.dropna()
    return float((1 - e / e.cummax()).max() * 100) if len(e) else np.nan


def window_stats(eq: pd.Series, cal: pd.DatetimeIndex, t0: int, t1: int) -> tuple[float, float]:
    a, b = cal[t0 - 1], cal[min(t1 + AFTERMATH_DAYS, len(cal) - 1)]
    w = eq[(eq.index >= a) & (eq.index <= b)].dropna()
    if len(w) < 2:
        return np.nan, np.nan
    return float((w.iloc[-1] / w.iloc[0] - 1) * 100), max_dd(w)


def sharpe(eq: pd.Series) -> float:
    r = eq.dropna().pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan


# ======================================================================================= Significance
def significance(curves: dict) -> pd.DataFrame:
    """Paired block bootstrap: Sharpe(CPO) - Sharpe(fixed variant) with randomly reassembled
    time segments. Share of draws <= 0 = probability that the lead is chance."""
    rng = np.random.default_rng(SEED)
    rets = pd.DataFrame({k: v.pct_change() for k, v in curves.items()}).dropna()
    n = len(rets)
    starts = rng.integers(0, n - BLOCK, size=(BOOTSTRAP, n // BLOCK + 1))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(BOOTSTRAP, -1)[:, :n]
    R = rets.to_numpy()
    sh = lambda x: x.mean(axis=-1) / x.std(axis=-1) * np.sqrt(252)
    rows = []
    for c in [k for k in curves if k.startswith("CPO")]:
        ci = rets.columns.get_loc(c)
        for f in [k for k in curves if k.startswith("Fixed")]:
            fi = rets.columns.get_loc(f)
            a, b = R[idx, ci], R[idx, fi]
            d = sh(a) - sh(b)
            real = sh(R[:, ci]) - sh(R[:, fi])
            yearly = (rets[[c, f]] + 1).groupby(rets.index.year).prod()
            rows.append({"cpo": c, "fixed": f, "sharpe_diff": real, "diff_5pct": np.percentile(d, 5),
                         "diff_95pct": np.percentile(d, 95), "p_chance": float((d <= 0).mean()),
                         "years_cpo_better": f"{int((yearly[c] > yearly[f]).sum())} of {len(yearly)}",
                         "maxdd_cpo": max_dd(curves[c]), "maxdd_fixed": max_dd(curves[f])})
    return pd.DataFrame(rows)


# ======================================================================================= Procedure
def main() -> None:
    ap = argparse.ArgumentParser(description="Stress test CPO exit against fixed exits")
    ap.add_argument("--retrain", action="store_true", help="retrain the CPO models")
    ap.add_argument("--entry", default=None, help=f"entry instead of '{tc.BASIS[0]}', e.g. \"High 50 + volume surge\"")
    args = ap.parse_args()
    global ENTRY, SUFFIX
    if args.entry:
        if args.entry not in tc.ENTRY_RULES:
            raise SystemExit(f"Unknown entry. Possible: {', '.join(tc.ENTRY_RULES)}")
        ENTRY = args.entry
        SUFFIX = "_" + "".join(ch if ch.isalnum() else "_" for ch in args.entry.lower()).strip("_")
    tm.OUT_DIR = tc.OUT_DIR
    t_start = time.time()
    m = tc.load_data()
    models = train_models(m, args.retrain)
    cal = m["close"].index

    log("Real prices ...")
    real = run_strategies(m, models)
    sig = significance({k: v for k, v in real.items() if not k.startswith("Buy and hold")})
    tm.save_csv(sig, f"significance{SUFFIX}.csv")

    tc_parts = market_parts(m)
    rows = [{"scenario": "Real prices", "start": "-", "strategy": k, "sharpe_total": sharpe(v),
             "maxdd_total_pct": max_dd(v), "return_window_pct": np.nan, "maxdd_window_pct": np.nan}
            for k, v in real.items()]
    rng = np.random.default_rng(SEED)
    total = len(SCENARIOS) * len(START_DATES)
    n = 0
    for sname, spec in SCENARIOS.items():
        for start in START_DATES:
            n += 1
            t0 = time.time()
            m2, (a, b) = apply_scenario(m, tc_parts, spec, start, rng)
            curves = run_strategies(m2, models)
            for k, eq in curves.items():
                wr, wdd = window_stats(eq, cal, a, b)
                rows.append({"scenario": sname, "start": start, "strategy": k, "sharpe_total": sharpe(eq),
                             "maxdd_total_pct": max_dd(eq), "return_window_pct": wr, "maxdd_window_pct": wdd})
            log(f"  {n}/{total} {sname} from {start} ({time.time() - t0:.0f} s)")
    res = pd.DataFrame(rows)
    tm.save_csv(res, f"stress_single{SUFFIX}.csv")
    summ = summarize(res)
    tm.save_csv(summ, f"stress_summary{SUFFIX}.csv")
    chart(res)
    report(sig, res, summ, time.time() - t_start)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    s = res[res["scenario"] != "Real prices"]
    g = s.groupby(["scenario", "strategy"])[["maxdd_window_pct", "return_window_pct", "maxdd_total_pct",
                                              "sharpe_total"]].mean().reset_index()
    # How often does CPO have a smaller drawdown in the shock window than the respective best fixed variant?
    wins = []
    for (sc, st), d in s.groupby(["scenario", "start"]):
        fixed = d[d["strategy"].str.startswith("Fixed")]
        best_fixed = fixed["maxdd_window_pct"].min()
        for c in d[d["strategy"].str.startswith("CPO")].itertuples():
            wins.append({"scenario": sc, "strategy": c.strategy,
                         "smaller_dd_than_best_fixed": c.maxdd_window_pct < best_fixed,
                         "rank_dd_among_all": int((d["maxdd_window_pct"] < c.maxdd_window_pct).sum()) + 1})
    w = pd.DataFrame(wins).groupby(["scenario", "strategy"]).agg(
        dd_better_than_best_fixed=("smaller_dd_than_best_fixed", "sum"),
        mean_dd_rank=("rank_dd_among_all", "mean")).reset_index()
    return g.merge(w, on=["scenario", "strategy"], how="left")


def chart(res: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s = res[res["scenario"] != "Real prices"]
    piv = s.groupby(["scenario", "strategy"])["maxdd_window_pct"].mean().unstack()
    order = [c for c in piv.columns if c.startswith("Fixed")] + [c for c in piv.columns if c.startswith("CPO")] + \
            [c for c in piv.columns if c.startswith("Buy and hold")]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(17, 6.5))
    x = np.arange(len(piv.index))
    w = 0.8 / len(order)
    fixed = [c for c in order if c.startswith("Fixed")]
    blues = plt.cm.Blues(np.linspace(0.35, 0.9, len(fixed)))
    colors = {c: blues[i] for i, c in enumerate(fixed)}
    colors.update({c: col for c, col in zip([c for c in order if c.startswith("CPO")], ["#d62728", "#ff7f0e"])})
    for i, col in enumerate(order):
        ax.bar(x + i * w, piv[col], w, label=col, color=colors.get(col, "#9e9e9e"),
               edgecolor="black" if col.startswith("CPO") else "none", lw=0.6)
    ax.set_xticks(x + 0.4 - w / 2, piv.index, fontsize=9)
    ax.set_ylabel("max. drawdown in the shock window + 60 days (%)")
    ax.set_title(f"Stress test (entry {ENTRY}): mean drawdown per scenario (6 start dates) - smaller is better")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / f"stress{SUFFIX}.png", dpi=110)
    plt.close(fig)


def report(sig: pd.DataFrame, res: pd.DataFrame, summ: pd.DataFrame, secs: float) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 110)
    print(f"1) REAL PRICES: lead of the CPO exit (entry {ENTRY}, test from {tc.FIRST_TEST_YEAR})")
    print("=" * 110)
    print("p_chance = share of the bootstrap draws in which CPO is NOT better (< 0.05 = reliable)")
    print(sig.round(3).to_string(index=False))
    print("\n" + "=" * 110)
    print("2) STRESS TEST: mean over 6 start dates per scenario")
    print("=" * 110)
    for sc, d in summ.groupby("scenario", sort=False):
        print(f"\n{sc}")
        print(d.drop(columns="scenario").sort_values("maxdd_window_pct").round(2).to_string(index=False))
    s = res[res["scenario"] != "Real prices"]
    print("\nOver all scenarios and start dates (mean):")
    print(s.groupby("strategy")[["maxdd_window_pct", "return_window_pct", "maxdd_total_pct", "sharpe_total"]]
          .mean().sort_values("maxdd_window_pct").round(2).to_string())
    print(f"\nFiles: {tc.OUT_DIR}  (run time {secs / 60:.1f} min)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(1)
