"""
Evaluation of the trend-following variants: winner, plausibility, sources of profit, probability of chance
================================================================================================================

Prerequisite: trend_cpo.py has been run (results_cpo/grid_curves.pkl = equity curve of all fixed
combinations of entry and exit).

1. Winner    : best Sharpe ratio and highest return over the whole period; in addition the variant that one
               would have chosen at the end of 2022 with only the data up to then, and how it did afterwards (out-of-sample).
2. Market    : DAX and equal-weighted index of all tradable stocks per year, market breadth, number of signals.
3. Profits   : hit rate, win/loss ratio, share of the best trades and stocks in the profit,
               result without the best stocks, market beta and alpha.
4. Chance    : Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR, takes into account the number of
               tested variants), bootstrap reality check (White) for the best of all variants,
               each in absolute terms and relative to the equal-weighted stock index.

Usage (in the folder trend_model):  ..\\.venv\\python.exe trend_evaluation.py
Results in results_cpo/: evaluation_*.csv, evaluation.png
"""
from __future__ import annotations

import sys
from math import e as EULER_E

import numpy as np
import pandas as pd
from scipy.stats import norm

import trend_cpo as tc
import trend_model as tm

LEARN = (pd.Timestamp(tc.SIGNAL_START), pd.Timestamp("2022-12-31"))
TEST = (pd.Timestamp("2023-01-01"), None)
BOOTSTRAP = 2000
BLOCK = 20                       # mean block length (trading days) in the stationary bootstrap
SEED = 3
START_DATES = pd.date_range("2017-10-01", "2023-10-01", freq="6MS")   # start dates for the robustness
GAMMA = 0.5772156649             # Euler-Mascheroni constant (expected value of the maximum)


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Metrics
def cut(s: pd.Series, per: tuple) -> pd.Series:
    a, b = per
    s = s.dropna()
    return s[(s.index >= a) & ((s.index <= b) if b is not None else True)]


def stats(eq: pd.Series, per: tuple, bench_r: pd.Series | None = None) -> dict:
    s = cut(eq, per)
    r = s.pct_change().dropna()
    if len(r) < 60:
        return {}
    yrs = len(r) / 252
    out = {"return_pa_pct": ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100,
           "sharpe": r.mean() / r.std() * np.sqrt(252),
           "volatility_pct": r.std() * np.sqrt(252) * 100,
           "max_drawdown_pct": (1 - s / s.cummax()).max() * 100}
    if bench_r is not None:
        b = bench_r.reindex(r.index).fillna(0)
        beta = np.cov(r, b)[0, 1] / b.var()
        out["beta"] = beta
        out["alpha_pa_pct"] = (r.mean() - beta * b.mean()) * 252 * 100
        x = r - b
        out["info_ratio"] = x.mean() / x.std() * np.sqrt(252)
    return out


def psr(r: np.ndarray, sr_star: float = 0.0) -> tuple[float, float, float]:
    """Probabilistic Sharpe Ratio (Bailey/Lopez de Prado 2012), everything in daily units.
    Returns (daily Sharpe, standard error, probability that the true Sharpe > sr_star)."""
    r = r[np.isfinite(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = pd.Series(r).skew()
    g4 = pd.Series(r).kurt() + 3                            # not excess kurtosis
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1))
    return sr, se, float(norm.cdf((sr - sr_star) / se))


def effective_n(R: np.ndarray) -> float:
    """Effective number of independent variants (participation ratio of the eigenvalues of the correlation matrix)."""
    c = np.corrcoef(R, rowvar=False)
    c = np.nan_to_num(c)
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def expected_max_sr(var_sr: float, n: float) -> float:
    """Expected highest Sharpe ratio (daily) of n independent variants WITHOUT a real edge."""
    n = max(n, 2)
    return np.sqrt(var_sr) * ((1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * EULER_E)))


def reality_check(R: np.ndarray, rng) -> tuple[float, np.ndarray]:
    """White's Reality Check: How often does the BEST of all variants reach by chance (all means set to
    zero, days redrawn in blocks) a Sharpe ratio as high as the best real one?"""
    n, k = R.shape
    sd = R.std(axis=0, ddof=1)
    real = (R.mean(axis=0) / sd).max()
    Z = R - R.mean(axis=0)                                  # null hypothesis: no edge
    mx = np.empty(BOOTSTRAP)
    ar = np.arange(n)
    for b0 in range(0, BOOTSTRAP, 100):                    # stationary bootstrap (Politis/Romano), 100 each
        B = min(100, BOOTSTRAP - b0)
        new = rng.random((B, n)) < 1 / BLOCK                # a new block starts with probability 1/BLOCK
        new[:, 0] = True
        blk = np.cumsum(new, axis=1) - 1
        last = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
        starts = rng.integers(0, n, size=(B, n))
        idx = (np.take_along_axis(starts, blk, axis=1) + (ar - last)) % n
        for i in range(B):
            Zb = Z[idx[i]]
            mx[b0 + i] = (Zb.mean(axis=0) / Zb.std(axis=0, ddof=1)).max()
    return float((mx >= real).mean()), mx


# ======================================================================================= Trades of the winner
def winner_trades(m: dict, entry: str, exit_name: str, exclude: list[str] | None = None) -> dict:
    cal = m["close"].index
    sigs = tc.entry_signals(m, [entry])
    meta = tc.all_trades(m, sigs)
    rows = np.arange(len(meta))
    if exclude:
        cols = m["close"].columns
        rows = rows[~np.isin(cols[meta["j"].to_numpy()], exclude)]
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    rank = mom[meta["t_signal"].to_numpy()[rows], meta["j"].to_numpy()[rows]]
    k = tc.EX_NAMES.index(exit_name)
    tr = tc.to_trades(meta, rows, np.full(len(rows), k), rank)
    sim = tm.simulate(m, tr, "long", pd.Timestamp(tc.SIGNAL_START))
    tl = sim["trades"]
    if not tl.empty:
        tl["name"] = tl["ticker"].map(m["names"])
        # reason: stop before the end of the holding period or regular
        tl["kind"] = np.where(tl["days"] < tc.EXIT_RULES[exit_name].get("days", 10 ** 6), "stop/trend break", "time expiry")
    return {"equity": sim["equity"], "trades": tl, "signals": len(rows)}


def trade_profile(tl: pd.DataFrame) -> dict:
    g = tl["result_eur"]
    win, loss = g[g > 0], g[g <= 0]
    top = g.sort_values(ascending=False)
    n10 = max(1, int(len(g) * 0.1))
    return {"trades": len(g), "hit_rate_pct": len(win) / len(g) * 100,
            "mean_gain_pct": tl.loc[g > 0, "return_pct"].mean(),
            "mean_loss_pct": tl.loc[g <= 0, "return_pct"].mean(),
            "gain_loss_ratio": tl.loc[g > 0, "return_pct"].mean() / -tl.loc[g <= 0, "return_pct"].mean(),
            "profit_factor": win.sum() / -loss.sum(),
            "profit_total_eur": g.sum(),
            "share_top10pct_trades_of_profit_pct": top.iloc[:n10].sum() / g.sum() * 100,
            "share_top10_trades_of_profit_pct": top.iloc[:10].sum() / g.sum() * 100,
            "holding_days_mean": tl["days"].mean()}


def start_robustness(m: dict) -> pd.DataFrame:
    """Run each combination from several start dates (each up to today): With only 10 slots and long
    holding periods, the result depends heavily on which stocks happen to land in the portfolio at the start."""
    cal = m["close"].index
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    rows = []
    for en in tc.EN_NAMES:
        meta = tc.all_trades(m, tc.entry_signals(m, [en]))
        st = meta["t_signal"].to_numpy()
        for k, ex in enumerate(tc.EX_NAMES):
            res = []
            for s in START_DATES:
                sel = np.flatnonzero(cal[st] >= s)
                tr = tc.to_trades(meta, sel, np.full(len(sel), k), mom[st[sel], meta["j"].to_numpy()[sel]])
                eq = tm.simulate(m, tr, "long", s, with_log=False)["equity"].dropna()
                r = eq.pct_change().dropna()
                yrs = len(r) / 252
                res.append((((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100,
                            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                            (1 - eq / eq.cummax()).max() * 100))
            a = np.array(res)
            rows.append({"entry": en, "exit": ex, "return_pa_mean": a[:, 0].mean(),
                         "sharpe_mean": np.nanmean(a[:, 1]), "sharpe_min": np.nanmin(a[:, 1]),
                         "sharpe_max": np.nanmax(a[:, 1]), "maxdd_mean": a[:, 2].mean(), "maxdd_max": a[:, 2].max()})
        log(f"  Start dates: {en} done")
    return pd.DataFrame(rows)


# ======================================================================================= Procedure
def main() -> None:
    tm.OUT_DIR = tc.OUT_DIR
    path = tc.OUT_DIR / "grid_curves.pkl"
    if not path.exists():
        raise SystemExit("Please run trend_cpo.py first (grid_curves.pkl missing).")
    curves = pd.read_pickle(path)
    m = tc.load_data()
    cal = m["close"].index
    start = pd.Timestamp(tc.SIGNAL_START)
    curves = curves[curves.index >= start]

    # Benchmarks: DAX ETF and equal-weighted index of all tradable stocks (same
    # survivorship bias as the strategies - the fair comparison)
    pf = m["close"].ffill()
    ew_r = pf.pct_change().where(m["elig"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ew = tm.CAPITAL * (1 + ew_r[ew_r.index >= start]).cumprod()
    bench = {"Equal-weighted stock index (universe)": ew}
    if m["bench"] is not None:
        b = m["bench"][m["bench"].index >= start].ffill().dropna()
        bench[f"DAX ({tm.BENCHMARK['stocks']})"] = tm.CAPITAL * b / b.iloc[0]
    ew_ret = ew.pct_change()

    # ---------------- 1. Metrics of all combinations
    rows = []
    for col in curves.columns:
        en, ex = col.split(" | ")
        r = {"entry": en, "exit": ex}
        for lbl, per in (("total", (start, None)), ("learn", LEARN), ("test", TEST)):
            for k, v in stats(curves[col], per, ew_ret).items():
                r[f"{lbl}_{k}"] = v
        rows.append(r)
    for name, eq in bench.items():
        r = {"entry": name, "exit": "-"}
        for lbl, per in (("total", (start, None)), ("learn", LEARN), ("test", TEST)):
            for k, v in stats(eq, per, ew_ret).items():
                r[f"{lbl}_{k}"] = v
        rows.append(r)
    allm = pd.DataFrame(rows)
    tm.save_csv(allm, "evaluation_all_variants.csv")
    strat = allm[allm["exit"] != "-"]

    log(f"Robustness: all combinations from {len(START_DATES)} start dates ...")
    robs = start_robustness(m)
    tm.save_csv(robs, "evaluation_start_dates.csv")
    best_sh = strat.sort_values("total_sharpe", ascending=False).iloc[0]
    best_ret = strat.sort_values("total_return_pa_pct", ascending=False).iloc[0]
    best_learn = strat.sort_values("learn_sharpe", ascending=False).iloc[0]
    rank_test = int((strat["test_sharpe"] > best_learn["test_sharpe"]).sum()) + 1
    key = lambda r: f"{r['entry']} | {r['exit']}"

    # ---------------- 2. Market conditions per year
    sma200 = pf.rolling(200).mean()
    breadth = (pf > sma200).astype(float).where(m["elig"] & sma200.notna()).mean(axis=1)
    hi200 = (pf > pf.rolling(200).max().shift(1)).astype(float).where(m["elig"]).sum(axis=1)
    yr = lambda eq: cut(eq, (start, None)).groupby(lambda d: d.year).last()
    def yearly(eq):
        y = yr(eq)
        return (y / y.shift(1).fillna(cut(eq, (start, None)).iloc[0]) - 1) * 100
    regime = pd.DataFrame({name: yearly(eq) for name, eq in bench.items()})
    regime["Market breadth (share above 200-day MA, %)"] = breadth.groupby(breadth.index.year).mean() * 100
    regime["new 200-day highs per day"] = hi200.groupby(hi200.index.year).mean()
    regime[f"Best Sharpe: {key(best_sh)}"] = yearly(curves[key(best_sh)])
    for i, (_, r) in enumerate(robs.sort_values("sharpe_mean", ascending=False).head(2).iterrows()):
        regime[f"Robust {i + 1}: {key(r)}"] = yearly(curves[key(r)])
    if key(best_ret) != key(best_sh):
        regime[f"Highest return: {key(best_ret)}"] = yearly(curves[key(best_ret)])
    regime = regime[regime.index >= start.year]
    tm.save_csv(regime.round(1), "evaluation_market.csv", index=True)

    # ---------------- 3. Sources of profit of the winner (best Sharpe) and of the highest return
    profiles, tops, robust = {}, {}, []
    rob_top = robs.sort_values("sharpe_mean", ascending=False).head(2)
    cands = [("Best Sharpe", best_sh), ("Highest return", best_ret)] + \
            [(f"Robust {i + 1}", r) for i, (_, r) in enumerate(rob_top.iterrows())]
    for label, row in cands:
        k = key(row)
        if k in profiles:
            continue
        w = winner_trades(m, row["entry"], row["exit"])
        tl = w["trades"]
        profiles[k] = trade_profile(tl)
        profiles[k].update({f"share_{a}_pct": v for a, v in (tl["kind"].value_counts(normalize=True) * 100).items()})
        for a, g in tl.groupby("kind"):
            profiles[k][f"return_{a}_pct"] = g["return_pct"].mean()
        by_tick = tl.groupby(["ticker", "name"])["result_eur"].agg(["sum", "size"]).sort_values("sum", ascending=False)
        by_tick["share_of_profit_pct"] = by_tick["sum"] / tl["result_eur"].sum() * 100
        tops[k] = by_tick.head(10)
        tm.save_csv(tl, "evaluation_trades_" + "".join(ch if ch.isalnum() else "_" for ch in k.lower()) + ".csv")
        for n_ex in (0, 3, 5, 10):
            ex = list(by_tick.index.get_level_values(0)[:n_ex])
            eq = w["equity"] if n_ex == 0 else winner_trades(m, row["entry"], row["exit"], ex)["equity"]
            s_all, s_test = stats(eq, (start, None)), stats(eq, TEST)
            robust.append({"variant": k, "without_best_stocks": n_ex, "return_pa_pct": s_all["return_pa_pct"],
                           "sharpe": s_all["sharpe"], "test_return_pa_pct": s_test["return_pa_pct"],
                           "test_sharpe": s_test["sharpe"], "max_drawdown_pct": s_all["max_drawdown_pct"]})
    prof = pd.DataFrame(profiles).T
    tm.save_csv(prof, "evaluation_profit_sources.csv", index=True)
    rob = pd.DataFrame(robust)
    tm.save_csv(rob, "evaluation_without_best_stocks.csv")

    # ---------------- 4. Probability of chance
    R_abs = curves.pct_change().iloc[1:].fillna(0)
    R_exc = R_abs.sub(ew_ret.reindex(R_abs.index).fillna(0), axis=0)
    n_eff = {"absolute": effective_n(R_abs.to_numpy()), "over index": effective_n(R_exc.to_numpy())}
    rng = np.random.default_rng(SEED)
    stat_rows = []
    rc = {}
    for lbl, R in (("absolute", R_abs), ("over index", R_exc)):
        srs = R.mean() / R.std()
        var_sr = srs.var()
        sr0 = expected_max_sr(var_sr, len(R.columns))
        sr0_eff = expected_max_sr(var_sr, n_eff[lbl])
        log(f"Reality Check ({lbl}, {BOOTSTRAP} draws) ...")
        p_rc, mx = reality_check(R.to_numpy(), rng)
        rc[lbl] = (p_rc, mx, srs.max())
        for name, row in [("Best Sharpe (total)", best_sh), ("Highest return (total)", best_ret)] + \
                [(f"Robust {i + 1} (mean start dates)", r) for i, (_, r) in enumerate(rob_top.iterrows())]:
            r = R[key(row)].to_numpy()
            sr, se, p0 = psr(r, 0.0)
            _, _, d_all = psr(r, sr0)
            _, _, d_eff = psr(r, sr0_eff)
            stat_rows.append({"variant": f"{name}: {key(row)}", "benchmark": lbl, "period": "total",
                              "sharpe_pa": sr * np.sqrt(252), "standard_error_pa": se * np.sqrt(252),
                              "t_value": sr / se, "psr_true_sharpe_above_0": p0,
                              "variants": len(R.columns), "effectively_independent": n_eff[lbl],
                              "expected_max_sharpe_chance_pa": sr0 * np.sqrt(252),
                              "expected_max_sharpe_chance_eff_pa": sr0_eff * np.sqrt(252),
                              "dsr_all": d_all, "dsr_effective": d_eff, "reality_check_p": p_rc})
        # out-of-sample: chosen at the end of 2022, test period only (no selection in the test -> no deflation needed)
        rt = cut(R[key(best_learn)], TEST).to_numpy()
        sr, se, p0 = psr(rt, 0.0)
        stat_rows.append({"variant": f"Chosen at end of 2022: {key(best_learn)}", "benchmark": lbl, "period": "test",
                          "sharpe_pa": sr * np.sqrt(252), "standard_error_pa": se * np.sqrt(252), "t_value": sr / se,
                          "psr_true_sharpe_above_0": p0, "variants": 1})
    st = pd.DataFrame(stat_rows)
    tm.save_csv(st, "evaluation_chance.csv")

    chart(curves, bench, best_sh, best_ret, best_learn, key, rc, rob_top)
    report(allm, strat, best_sh, best_ret, best_learn, rank_test, regime, prof, tops, rob, st, key, robs)


def chart(curves, bench, best_sh, best_ret, best_learn, key, rc, rob_top) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    for col in curves.columns:
        ax.plot(curves.index, curves[col] / 1000, color="#cccccc", lw=0.4, zorder=1)
    shown = []
    rr = [r for _, r in rob_top.iterrows()]
    for lbl, row, c in [("Best Sharpe from 2017", best_sh, "#d62728"), ("Highest return", best_ret, "#ff7f0e"),
                        ("Chosen at end of 2022", best_learn, "#2ca02c")] + \
                       [(f"Robust {i + 1} (start dates)", r, c) for i, (r, c) in enumerate(zip(rr, ["#9467bd", "#8c564b"]))]:
        k = key(row)
        if k in shown:
            continue
        shown.append(k)
        ax.plot(curves.index, curves[k] / 1000, color=c, lw=2, label=f"{lbl}: {k}", zorder=3)
    for (name, eq), c in zip(bench.items(), ["#1f77b4", "#555555"]):
        ax.plot(eq.index, eq / 1000, color=c, lw=1.6, ls="--", label=name, zorder=2)
    ax.axvline(TEST[0], color="black", lw=0.8, ls=":")
    ax.text(TEST[0], ax.get_ylim()[1] if ax.get_yscale() == "linear" else 1, "  Test from 2023", va="top", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("All 270 fixed combinations (grey) - account in thousand EUR (log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    p, mx, real = rc["over index"]
    ax2.hist(mx * np.sqrt(252), bins=50, color="#9ecae1")
    ax2.axvline(real * np.sqrt(252), color="#d62728", lw=2)
    ax2.set_title(f"Reality check: best information ratio of all variants\nby chance (blue) vs. real (red), p = {p:.2f}")
    ax2.set_xlabel("highest information ratio relative to the stock index (p. a.)")
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / "evaluation.png", dpi=110)
    plt.close(fig)


def report(allm, strat, best_sh, best_ret, best_learn, rank_test, regime, prof, tops, rob, st, key, robs) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    cols = ["entry", "exit", "total_return_pa_pct", "total_sharpe", "total_max_drawdown_pct",
            "total_beta", "total_alpha_pa_pct", "learn_sharpe", "test_sharpe", "learn_return_pa_pct",
            "test_return_pa_pct"]
    print("\n" + "=" * 120)
    print("1) WINNER (270 fixed combinations, Sept. 2017 to today)")
    print("=" * 120)
    print("\nTop 10 by Sharpe ratio (total):")
    print(strat.sort_values("total_sharpe", ascending=False)[cols].head(10).round(2).to_string(index=False))
    print("\nTop 10 by return p. a. (total):")
    print(strat.sort_values("total_return_pa_pct", ascending=False)[cols].head(10).round(2).to_string(index=False))
    print("\nBenchmarks:")
    print(allm[allm["exit"] == "-"][cols].round(2).to_string(index=False))
    print(f"\nChosen at end of 2022 (with learning data only): {key(best_learn)} - learning Sharpe {best_learn['learn_sharpe']:.2f}, "
          f"test Sharpe {best_learn['test_sharpe']:.2f} (rank {rank_test} of {len(strat)} in the test)")
    rc = strat[["learn_sharpe", "test_sharpe"]].corr(method="spearman").iloc[0, 1]
    print(f"Rank correlation of learning vs. test Sharpe over all combinations: {rc:+.2f}")
    print(f"\nRobustness over {len(START_DATES)} start dates (semi-annually 2017-2023, each up to today):")
    rc_ = ["entry", "exit", "return_pa_mean", "sharpe_mean", "sharpe_min", "sharpe_max", "maxdd_mean",
           "maxdd_max"]
    print("Top 10 by mean Sharpe ratio:")
    print(robs.sort_values("sharpe_mean", ascending=False)[rc_].head(10).round(2).to_string(index=False))
    print("Top 10 by mean return:")
    print(robs.sort_values("return_pa_mean", ascending=False)[rc_].head(10).round(2).to_string(index=False))
    for lbl, row in (("Best Sharpe from 2017", best_sh),):
        rr = robs[(robs["entry"] == row["entry"]) & (robs["exit"] == row["exit"])]
        print(f"{lbl} ({key(row)}) over all start dates:")
        print(rr[rc_].round(2).to_string(index=False))
    print("\n" + "=" * 120)
    print("2) MARKET CONDITIONS PER YEAR (return in %)")
    print("=" * 120)
    print(regime.round(1).to_string())
    print("\n" + "=" * 120)
    print("3) HOW THE PROFIT ARISES")
    print("=" * 120)
    print(prof.T.round(2).to_string())
    for k, t in tops.items():
        print(f"\nStocks with the largest profit contribution - {k}:")
        print(t.round(1).to_string())
    print("\nResult without the stocks with the largest profit contribution:")
    print(rob.round(2).to_string(index=False))
    print("\n" + "=" * 120)
    print("4) PROBABILITY OF CHANCE")
    print("=" * 120)
    print(st.round(3).to_string(index=False))
    print(f"\nFiles: {tc.OUT_DIR} (evaluation_*.csv, evaluation.png)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(1)
