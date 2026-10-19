"""
Position size in the trend model: fixed 1/10 or depending on signal quality (CPO and simple rules)?
================================================================================================================

Same stock selection as before (signal, ranking by 6-month momentum), only the size per purchase changes:
    Weight w between WEIGHT_MIN and WEIGHT_MAX of the portfolio value (default 1/10)
    Rule for purchases above 1/10: The extra amount comes only from free cash - after the purchase,
    RESERVE (1/10) of the portfolio value must still be left as cash, so that the next normal signal can be bought.
    If the cash is not sufficient, less is bought (at least the default weight, as far as available).

Sizing rules (each signal is rated only with data up to the signal day):
    fixed 1/10            as before
    CPO[per day]          prediction of the CPO model (trend_cpo.py, walk-forward, trained per year only on prior years)
    CPO[relative]         ditto, target relative to simultaneous signals
    Momentum strength     stronger 6-month trend -> larger position
    Volatility           calmer stock -> larger position (same risk per position, "inverse volatility")
    Chance               same distribution of weights as the tier rules, but random (control)
Tiers: rating in the top fifth of the last 250 trading days -> WEIGHT_MAX, in the bottom fifth -> WEIGHT_MIN,
otherwise 1/10.

Usage (in the folder trend_model, after trend_cpo.py):  ..\\.venv\\python.exe trend_position_size.py
Results in results_cpo/: position_size.csv, position_size.png
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import trend_cpo as tc
import trend_model as tm

STRATEGIES = {
    "Robust 1: High 200 | Time 20": ("High 200", "Time 20"),
    "Robust 2: High 50 + volume surge | Time 60 + stop 15%": ("High 50 + volume surge", "Time 60 + stop 15%"),
}
WEIGHT_STANDARD = 0.10
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.20
RESERVE = 0.10                   # after a purchase above the default weight, this much cash must be left
UPPER, LOWER = 0.8, 0.2           # fifths for the tiers
HISTORY_DAYS = 250                   # comparison with the ratings of the signals of the last 250 trading days
MAX_POSITIONS_CAP = 20              # upper limit of the number (with small positions, more than 10 fit in)
MIN_PURCHASE = 0.05                  # no purchase below 5 % of the portfolio value (otherwise mini positions arise from leftover cash)
RANDOM_RUNS = 20
START_DATES = pd.date_range(f"{tc.FIRST_TEST_YEAR}-01-01", "2023-10-01", freq="6MS")
SEED = 21


def log(msg: str) -> None:
    tm.log(msg)


def tiers(score: np.ndarray, t_sig: np.ndarray) -> np.ndarray:
    """Tiers from the rank of the rating among the signals of the last HISTORY_DAYS trading days (without the future)."""
    w = np.full(len(score), WEIGHT_STANDARD)
    order = np.argsort(t_sig, kind="stable")
    hist_t, hist_s = [], []
    for i in order:
        t, s = t_sig[i], score[i]
        while hist_t and hist_t[0] < t - HISTORY_DAYS:
            hist_t.pop(0)
            hist_s.pop(0)
        if np.isfinite(s) and len(hist_s) >= 30:
            p = (np.array(hist_s) < s).mean()
            w[i] = WEIGHT_MAX if p >= UPPER else (WEIGHT_MIN if p < LOWER else WEIGHT_STANDARD)
        if np.isfinite(s):
            hist_t.append(t)
            hist_s.append(s)
    return w


def simulate(m: dict, tr: pd.DataFrame, start: pd.Timestamp) -> dict:
    """Like trend_model.simulate (long, closing auction), but with a weight per trade and the reserve rule."""
    cal = m["close"].index
    cols = m["close"].columns
    P = m["close"].ffill().to_numpy()
    n = len(cal)
    slip = tm.SLIPPAGE_BPS / 1e4
    first = int(np.searchsorted(cal, start))
    by_entry: dict[int, list] = {}
    for r in tr.sort_values(["e", "rank"], ascending=[True, False]).itertuples(index=False):
        by_entry.setdefault(r.e, []).append(r)
    cash, pos = tm.CAPITAL, {}
    equity = np.full(n, np.nan)
    invested = np.full(n, np.nan)
    npos = np.full(n, np.nan)
    sizes = []
    for t in range(first, n):
        cash -= tm.monthly_fee(cal, t, first)
        for j in [j for j, q in pos.items() if q["x"] == t]:
            q = pos.pop(j)
            px = P[t, j] if np.isnan(q["px_out"]) else q["px_out"]
            gross = q["shares"] * px
            cash += gross - tm.commission(gross, cols[j]) - gross * slip
        V = cash + sum(q["shares"] * P[t, j] for j, q in pos.items())
        for r in by_entry.get(t, []):
            if len(pos) >= MAX_POSITIONS_CAP:
                break
            j = r.j
            if j in pos or not np.isfinite(P[t, j]) or r.x <= t:
                continue
            w = min(r.w, WEIGHT_MAX)
            if w > WEIGHT_STANDARD:
                amount = min(w * V, cash - RESERVE * V)        # extra amount only from free cash
                amount = max(amount, min(WEIGHT_STANDARD * V, cash))
            else:
                amount = min(w * V, cash)
            if amount < MIN_PURCHASE * V:
                continue
            shares = int(amount // P[t, j])
            if shares <= 0:
                continue
            alloc = shares * P[t, j]
            cost = tm.commission(alloc, cols[j]) + alloc * slip
            if alloc + cost > cash:
                continue
            cash -= alloc + cost
            pos[j] = {"x": int(r.x), "shares": shares, "px_out": r.exit_price}
            sizes.append((r.w, alloc / V))
        held = sum(q["shares"] * P[t, j] for j, q in pos.items())
        equity[t] = cash + held
        invested[t] = held / equity[t]
        npos[t] = len(pos)
    s = slice(first, n)
    return {"equity": pd.Series(equity, index=cal).iloc[s], "invested_share": np.nanmean(invested[s]),
            "positions_n": np.nanmean(npos[s]), "size_mean": np.mean([b for _, b in sizes]) if sizes else np.nan,
            "large_executed": (np.mean([b >= 0.15 for a, b in sizes if a > WEIGHT_STANDARD])
                                if any(a > WEIGHT_STANDARD for a, _ in sizes) else np.nan)}


def stats(eq: pd.Series) -> dict:
    s = eq.dropna()
    r = s.pct_change().dropna()
    y = len(r) / 252
    return {"return_pa_pct": ((s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1) * 100,
            "sharpe": r.mean() / r.std() * np.sqrt(252), "max_drawdown_pct": (1 - s / s.cummax()).max() * 100}


def run_rule(m, meta, rows, k, rank, w, start_list) -> dict:
    out = []
    extra = {}
    for i, s in enumerate(start_list):
        sel = rows[m["close"].index[meta["t_signal"].to_numpy()[rows]] >= s]
        idx = np.searchsorted(rows, sel)
        tr = tc.to_trades(meta, sel, np.full(len(sel), k), rank[idx]).assign(w=w[idx])
        res = simulate(m, tr, s)
        out.append(stats(res["equity"]))
        if i == 0:
            extra = {"invested_pct": res["invested_share"] * 100, "positions_mean": res["positions_n"],
                     "purchase_size_mean_pct": res["size_mean"] * 100,
                     "large_purchases_executed_pct": res["large_executed"] * 100}
            first_eq = res["equity"]
    d = pd.DataFrame(out)
    r = {f"from{tc.FIRST_TEST_YEAR}_{c}": d.iloc[0][c] for c in d.columns}
    r.update({f"starts_{c}_mean": d[c].mean() for c in d.columns})
    r.update(extra)
    return r, first_eq


def main() -> None:
    tm.OUT_DIR = tc.OUT_DIR
    m = tc.load_data()
    cal = m["close"].index
    models = tc.load_models()
    if models is None:
        import cpo_stress_test as cs
        models = cs.train_models(m, False)
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    vol = np.log(m["close"].ffill()).diff().rolling(60, min_periods=40).std().to_numpy() * np.sqrt(252)
    test_start = pd.Timestamp(f"{tc.FIRST_TEST_YEAR}-01-01")
    starts = [test_start] + [s for s in START_DATES if s > test_start]
    rng = np.random.default_rng(SEED)
    rows_out, curves, ic_rows = [], {}, []
    for sname, (entry, exit_name) in STRATEGIES.items():
        log(f"{sname} ...")
        meta = tc.all_trades(m, tc.entry_signals(m, [entry]))
        st = meta["t_signal"].to_numpy()
        jj = meta["j"].to_numpy()
        rows = np.flatnonzero(cal[st] >= test_start)
        k = tc.EX_NAMES.index(exit_name)
        feat = tc.features(m, meta)
        rank = mom[st[rows], jj[rows]]
        years = cal[st[rows]].year
        scores = {}
        for label, mods in models.items():
            sc = np.full(len(rows), np.nan)
            for Y, model in mods.items():
                sel = np.flatnonzero(years == Y)
                if len(sel):
                    p = model.predict(tc.stack(feat, rows[sel])).reshape(len(sel), len(tc.EX_NAMES))
                    sc[sel] = p[:, k]
            scores[f"CPO[{label}]"] = sc
        scores["Momentum strength"] = rank
        # Volatility: target = same risk per position as a stock with average volatility
        v = vol[st[rows], jj[rows]]
        med = np.nanmedian(vol, axis=1)[st[rows]]
        w_vol = np.clip(WEIGHT_STANDARD * med / v, WEIGHT_MIN, WEIGHT_MAX)
        w_vol = np.where(np.isfinite(w_vol), w_vol, WEIGHT_STANDARD)
        # How well does the rating predict the later return per trade? (closed trades)
        y = (meta.attrs["R"][rows, k] - tc.COSTS)
        done = ~meta.attrs["OPEN"][rows, k]
        for nm, sc in scores.items():
            ok = done & np.isfinite(sc)
            ic_rows.append({"strategy": sname, "rating": nm,
                            "rank_correlation_with_trade_return": pd.Series(sc[ok]).corr(pd.Series(y[ok]), method="spearman"),
                            "return_top_fifth_pct": y[ok][sc[ok] >= np.nanquantile(sc[ok], UPPER)].mean() * 100,
                            "return_bottom_fifth_pct": y[ok][sc[ok] < np.nanquantile(sc[ok], LOWER)].mean() * 100,
                            "return_all_pct": y[ok].mean() * 100})
        # Reference: the previous trend model (fixed 1/10, at most 10 positions, leftover cash is also invested in small amounts)
        res0 = []
        for i, s0 in enumerate(starts):
            sel = rows[cal[st[rows]] >= s0]
            tr0 = tc.to_trades(meta, sel, np.full(len(sel), k), mom[st[sel], jj[sel]])
            eq0 = tm.simulate(m, tr0, "long", s0, with_log=False)["equity"]
            res0.append(stats(eq0))
            if i == 0:
                curves[(sname, "Trend model so far")] = eq0
        d0 = pd.DataFrame(res0)
        r0 = {f"from{tc.FIRST_TEST_YEAR}_{c}": d0.iloc[0][c] for c in d0.columns}
        r0.update({f"starts_{c}_mean": d0[c].mean() for c in d0.columns})
        r0.update({"strategy": sname, "size_rule": "Trend model so far (1/10, max. 10 positions)"})
        rows_out.append(r0)
        rules = {"fixed 1/10": np.full(len(rows), WEIGHT_STANDARD)}
        for nm, sc in scores.items():
            rules[nm] = tiers(sc, st[rows])
        rules["Volatility (inverse volatility)"] = w_vol
        for nm, w in rules.items():
            r, eq = run_rule(m, meta, rows, k, rank, w, starts)
            r.update({"strategy": sname, "size_rule": nm,
                      "share_large_pct": (w > WEIGHT_STANDARD).mean() * 100,
                      "share_small_pct": (w < WEIGHT_STANDARD).mean() * 100})
            rows_out.append(r)
            curves[(sname, nm)] = eq
        # random with the same distribution of tiers as CPO[per day]
        w_ref = rules[[n for n in rules if n.startswith("CPO")][0]]
        rand = []
        for _ in range(RANDOM_RUNS):
            r, _eq = run_rule(m, meta, rows, k, rank, rng.permutation(w_ref), starts[:1])
            rand.append(r)
        d = pd.DataFrame(rand)
        r = d.mean(numeric_only=True).to_dict()
        r.update({"strategy": sname, "size_rule": f"Chance (mean of {RANDOM_RUNS})",
                  "chance_sharpe_range": f"{d[f'from{tc.FIRST_TEST_YEAR}_sharpe'].min():.2f} to "
                                          f"{d[f'ab{tc.FIRST_TEST_YEAR}_sharpe'].max():.2f}"})
        rows_out.append(r)
    res = pd.DataFrame(rows_out)
    res = res[["strategy", "size_rule"] + [c for c in res.columns if c not in ("strategy", "size_rule")]]
    ic = pd.DataFrame(ic_rows)
    tm.save_csv(res, "position_size.csv")
    tm.save_csv(ic, "position_size_prediction.csv")
    chart(curves)
    report(res, ic)


def chart(curves: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(dict.fromkeys(k[0] for k in curves))
    fig, axes = plt.subplots(1, len(names), figsize=(8 * len(names), 5.5), squeeze=False)
    for ax, nm in zip(axes[0], names):
        for (s, rule), eq in curves.items():
            if s != nm:
                continue
            kw = {"color": "black", "lw": 2} if rule.startswith("Trend model") else {"lw": 1.3}
            ax.plot(eq.index, eq / 1000, label=rule, **kw)
        ax.set_yscale("log")
        ax.set_title(nm, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Position size: account in thousand EUR (log), from {tc.FIRST_TEST_YEAR}")
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / "position_size.png", dpi=110)
    plt.close(fig)


def report(res: pd.DataFrame, ic: pd.DataFrame) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    y = tc.FIRST_TEST_YEAR
    cols = [c for c in ["size_rule", f"ab{y}_return_pa_pct", f"ab{y}_sharpe", f"ab{y}_max_drawdown_pct",
                        "starts_return_pa_pct_mean", "starts_sharpe_mean", "starts_max_drawdown_pct_mean",
                        "invested_pct", "positions_mean", "purchase_size_mean_pct", "large_purchases_executed_pct",
                        "share_large_pct",
                        "share_small_pct", "chance_sharpe_range"] if c in res.columns]
    print("\n" + "=" * 120)
    print(f"POSITION SIZE  ({WEIGHT_MIN:.0%} to {WEIGHT_MAX:.0%} per purchase, reserve {RESERVE:.0%}; costs: "
          f"{tm.broker_text()})")
    print("=" * 120)
    for s, g in res.groupby("strategy", sort=False):
        print(f"\n{s}")
        print(g[cols].round(2).to_string(index=False))
    print("\nDoes the rating predict the return per trade? (test years, closed trades, after costs)")
    print(ic.round(3).to_string(index=False))
    print(f"\nFiles: {tc.OUT_DIR} (position_size*.csv, position_size.png)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(1)
