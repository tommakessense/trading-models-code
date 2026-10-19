"""
Stress test: CPO exit against fixed exits in simulated crashes, bear markets and choppy markets
================================================================================================================

Question: Does the exit chosen by CPO limit the losses better than fixed rules when the market behaves
differently than in the training data?

Procedure
1. As in trend_cpo.py, the CPO models are trained year by year ONLY on real data of the previous years
   (the result is cached in ergebnisse_cpo/modelle.pkl).
2. Real significance: Is the lead of the CPO exit in the real data greater than chance?
   (Block bootstrap of the daily returns, paired against each fixed variant.)
3. Stress scenarios: An artificial shock is built into the real prices from several start dates in the test period
   - as a market move that hits each stock with its beta, plus random noise:
     Crash mit Erholung (V)       -35 % in 20 days, then +40 % in 60 days
     Baerenmarkt (L)              -45 % over 120 days, no recovery
     Seitwaerts mit Fehlausbruechen  +-12 % waves every 40 days over 160 days (poison for trend following)
     Flash-Crash                  -15 % in 3 days, recovery in 7 days
     Grosse Streuung              market unchanged, but individual stocks diverge strongly
   All signals, features and results are recalculated on the modified prices; the models
   do not know the scenarios (as in real life).
4. All strategies with the same entry (Hoch 200) are compared: fixed holding periods, trend break,
   trailing stop and the two CPO variants - drawdown, return in the shock window, Sharpe ratio.

Usage (in the folder trend_model):  ..\\.venv\\python.exe cpo_stresstest.py      (after trend_cpo.py: approx. 2 minutes)
                                    ..\\.venv\\python.exe cpo_stresstest.py --einstieg "Hoch 50 + Volumenschub"
                                    ..\\.venv\\python.exe cpo_stresstest.py --neu   retrain the models
Results in ergebnisse_cpo/: stress_einzeln.csv, stress_zusammenfassung.csv, signifikanz.csv, stress.png
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
EINSTIEG = tc.BASIS[0]           # can be changed with --einstieg
SUFFIX = ""
FESTE = ["Zeit 20", "Zeit 40", "Zeit 120", "Trend 50", "Trend 20", "Nachzug 10%", "Nachzug 15%", "Nachzug 25%",
         "Zeit 40 + Stopp 10%", "Zeit 40 + Stopp 15%", "Zeit 60 + Stopp 10%", "Zeit 60 + Stopp 15%",
         "Zeit 120 + Stopp 10%", "Zeit 120 + Stopp 15%"]
START_DATEN = ["2020-09-01", "2021-06-01", "2022-03-01", "2023-03-01", "2024-06-03", "2025-06-02"]
NACHLAUF = 60                    # shock window + 60 trading days are evaluated separately
BOOTSTRAP = 2000
BLOCK = 20                       # block length (trading days) for the bootstrap
SEED = 5

SZENARIEN = {
    # name: list of phases (days, total market move as a factor, additional daily volatility of the market)
    "Crash mit Erholung (V)": {"phasen": [(20, 0.65, 0.025), (60, 1.40, 0.015)]},
    "Baerenmarkt (L)": {"phasen": [(120, 0.55, 0.020)]},
    "Seitwaerts mit Fehlausbruechen": {"welle": (160, 0.12, 40, 0.015)},   # days, amplitude (log), period, noise
    "Flash-Crash": {"phasen": [(3, 0.85, 0.03), (7, 1 / 0.85, 0.015)]},
    "Grosse Streuung": {"streuung": (120, 2.0, 0.35)},   # days, factor on individual moves, trend dispersion p. a.
}


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Models
def train_models(m: dict, refit: bool) -> dict:
    models = None if refit else tc.load_models()
    if models is not None:
        log("CPO-Modelle aus trend_cpo.py geladen (ergebnisse_cpo/modelle.pkl)")
        return models
    log("Trainiere CPO-Modelle (alle Einstiege, nur echte Daten, je Testjahr nur Vorjahre) ...")
    sigs = tc.entry_signals(m)
    meta = tc.all_trades(m, sigs)
    feat = tc.features(m, meta)
    models = {}
    for label, ziel in tc.ZIELE.items():
        models[label] = {}
        tc.walk_forward(m, meta, feat, ziel, models_out=models[label])
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
    if "phasen" in spec:
        parts = []
        for days, factor, noise in spec["phasen"]:
            parts.append(np.full(days, np.log(factor) / days) + rng.normal(0, noise, days))
        return np.concatenate(parts), 1.0, 0.0
    if "welle" in spec:
        days, amp, period, noise = spec["welle"]
        level = amp * np.sin(2 * np.pi * np.arange(days + 1) / period)
        return np.diff(level) + rng.normal(0, noise, days), 1.0, 0.0
    days, mult, spread = spec["streuung"]
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
    test_start = pd.Timestamp(f"{tc.ERSTES_TESTJAHR}-01-01")
    sigs = tc.entry_signals(m, [EINSTIEG])
    meta = tc.all_trades(m, sigs)
    sig_t = meta["t_signal"].to_numpy()
    rows = np.flatnonzero(cal[sig_t] >= test_start)
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    rank = mom[sig_t[rows], meta["j"].to_numpy()[rows]]
    out = {}
    for ex in FESTE:
        k = tc.EX_NAMES.index(ex)
        tr = tc.to_trades(meta, rows, np.full(len(rows), k), rank)
        out[f"Fest {ex}"] = tm.simulate(m, tr, "long", test_start, with_log=False)["equity"]
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
        out["Kaufen und Halten DAX"] = tm.CAPITAL * b / b.iloc[0]
    return out


def max_dd(eq: pd.Series) -> float:
    e = eq.dropna()
    return float((1 - e / e.cummax()).max() * 100) if len(e) else np.nan


def window_stats(eq: pd.Series, cal: pd.DatetimeIndex, t0: int, t1: int) -> tuple[float, float]:
    a, b = cal[t0 - 1], cal[min(t1 + NACHLAUF, len(cal) - 1)]
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
        for f in [k for k in curves if k.startswith("Fest")]:
            fi = rets.columns.get_loc(f)
            a, b = R[idx, ci], R[idx, fi]
            d = sh(a) - sh(b)
            real = sh(R[:, ci]) - sh(R[:, fi])
            yearly = (rets[[c, f]] + 1).groupby(rets.index.year).prod()
            rows.append({"cpo": c, "fest": f, "sharpe_diff": real, "diff_5pct": np.percentile(d, 5),
                         "diff_95pct": np.percentile(d, 95), "p_zufall": float((d <= 0).mean()),
                         "jahre_cpo_besser": f"{int((yearly[c] > yearly[f]).sum())} von {len(yearly)}",
                         "maxdd_cpo": max_dd(curves[c]), "maxdd_fest": max_dd(curves[f])})
    return pd.DataFrame(rows)


# ======================================================================================= Procedure
def main() -> None:
    ap = argparse.ArgumentParser(description="Stress test CPO exit against fixed exits")
    ap.add_argument("--neu", action="store_true", help="retrain the CPO models")
    ap.add_argument("--einstieg", default=None, help=f"entry instead of '{tc.BASIS[0]}', e.g. \"Hoch 50 + Volumenschub\"")
    args = ap.parse_args()
    global EINSTIEG, SUFFIX
    if args.einstieg:
        if args.einstieg not in tc.EINSTIEGE:
            raise SystemExit(f"Unbekannter Einstieg. Moeglich: {', '.join(tc.EINSTIEGE)}")
        EINSTIEG = args.einstieg
        SUFFIX = "_" + "".join(ch if ch.isalnum() else "_" for ch in args.einstieg.lower()).strip("_")
    tm.OUT_DIR = tc.OUT_DIR
    t_start = time.time()
    m = tc.load_data()
    models = train_models(m, args.neu)
    cal = m["close"].index

    log("Echte Kurse ...")
    real = run_strategies(m, models)
    sig = significance({k: v for k, v in real.items() if not k.startswith("Kaufen")})
    tm.save_csv(sig, f"signifikanz{SUFFIX}.csv")

    tc_parts = market_parts(m)
    rows = [{"szenario": "Echte Kurse", "start": "-", "strategie": k, "sharpe_gesamt": sharpe(v),
             "maxdd_gesamt_pct": max_dd(v), "rendite_fenster_pct": np.nan, "maxdd_fenster_pct": np.nan}
            for k, v in real.items()]
    rng = np.random.default_rng(SEED)
    total = len(SZENARIEN) * len(START_DATEN)
    n = 0
    for sname, spec in SZENARIEN.items():
        for start in START_DATEN:
            n += 1
            t0 = time.time()
            m2, (a, b) = apply_scenario(m, tc_parts, spec, start, rng)
            curves = run_strategies(m2, models)
            for k, eq in curves.items():
                wr, wdd = window_stats(eq, cal, a, b)
                rows.append({"szenario": sname, "start": start, "strategie": k, "sharpe_gesamt": sharpe(eq),
                             "maxdd_gesamt_pct": max_dd(eq), "rendite_fenster_pct": wr, "maxdd_fenster_pct": wdd})
            log(f"  {n}/{total} {sname} ab {start} ({time.time() - t0:.0f} s)")
    res = pd.DataFrame(rows)
    tm.save_csv(res, f"stress_einzeln{SUFFIX}.csv")
    summ = summarize(res)
    tm.save_csv(summ, f"stress_zusammenfassung{SUFFIX}.csv")
    chart(res)
    report(sig, res, summ, time.time() - t_start)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    s = res[res["szenario"] != "Echte Kurse"]
    g = s.groupby(["szenario", "strategie"])[["maxdd_fenster_pct", "rendite_fenster_pct", "maxdd_gesamt_pct",
                                              "sharpe_gesamt"]].mean().reset_index()
    # How often does CPO have a smaller drawdown in the shock window than the respective best fixed variant?
    wins = []
    for (sc, st), d in s.groupby(["szenario", "start"]):
        fixed = d[d["strategie"].str.startswith("Fest")]
        best_fixed = fixed["maxdd_fenster_pct"].min()
        for c in d[d["strategie"].str.startswith("CPO")].itertuples():
            wins.append({"szenario": sc, "strategie": c.strategie,
                         "kleinerer_dd_als_beste_feste": c.maxdd_fenster_pct < best_fixed,
                         "rang_dd_unter_allen": int((d["maxdd_fenster_pct"] < c.maxdd_fenster_pct).sum()) + 1})
    w = pd.DataFrame(wins).groupby(["szenario", "strategie"]).agg(
        dd_besser_als_beste_feste=("kleinerer_dd_als_beste_feste", "sum"),
        mittlerer_dd_rang=("rang_dd_unter_allen", "mean")).reset_index()
    return g.merge(w, on=["szenario", "strategie"], how="left")


def chart(res: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s = res[res["szenario"] != "Echte Kurse"]
    piv = s.groupby(["szenario", "strategie"])["maxdd_fenster_pct"].mean().unstack()
    order = [c for c in piv.columns if c.startswith("Fest")] + [c for c in piv.columns if c.startswith("CPO")] + \
            [c for c in piv.columns if c.startswith("Kaufen")]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(17, 6.5))
    x = np.arange(len(piv.index))
    w = 0.8 / len(order)
    fixed = [c for c in order if c.startswith("Fest")]
    blues = plt.cm.Blues(np.linspace(0.35, 0.9, len(fixed)))
    colors = {c: blues[i] for i, c in enumerate(fixed)}
    colors.update({c: col for c, col in zip([c for c in order if c.startswith("CPO")], ["#d62728", "#ff7f0e"])})
    for i, col in enumerate(order):
        ax.bar(x + i * w, piv[col], w, label=col, color=colors.get(col, "#9e9e9e"),
               edgecolor="black" if col.startswith("CPO") else "none", lw=0.6)
    ax.set_xticks(x + 0.4 - w / 2, piv.index, fontsize=9)
    ax.set_ylabel("max. Drawdown im Schockfenster + 60 Tage (%)")
    ax.set_title(f"Stresstest (Einstieg {EINSTIEG}): mittlerer Drawdown je Szenario (6 Startdaten) - kleiner ist besser")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / f"stress{SUFFIX}.png", dpi=110)
    plt.close(fig)


def report(sig: pd.DataFrame, res: pd.DataFrame, summ: pd.DataFrame, secs: float) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 110)
    print(f"1) ECHTE KURSE: Vorsprung des CPO-Ausstiegs (Einstieg {EINSTIEG}, Test ab {tc.ERSTES_TESTJAHR})")
    print("=" * 110)
    print("p_zufall = Anteil der Bootstrap-Ziehungen, in denen CPO NICHT besser ist (< 0,05 = belastbar)")
    print(sig.round(3).to_string(index=False))
    print("\n" + "=" * 110)
    print("2) STRESSTEST: Mittel ueber 6 Startdaten je Szenario")
    print("=" * 110)
    for sc, d in summ.groupby("szenario", sort=False):
        print(f"\n{sc}")
        print(d.drop(columns="szenario").sort_values("maxdd_fenster_pct").round(2).to_string(index=False))
    s = res[res["szenario"] != "Echte Kurse"]
    print("\nUeber alle Szenarien und Startdaten (Mittel):")
    print(s.groupby("strategie")[["maxdd_fenster_pct", "rendite_fenster_pct", "maxdd_gesamt_pct", "sharpe_gesamt"]]
          .mean().sort_values("maxdd_fenster_pct").round(2).to_string())
    print(f"\nDateien: {tc.OUT_DIR}  (Laufzeit {secs / 60:.1f} min)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
