"""
Bewertung der Trendfolge-Varianten: Sieger, Plausibilitaet, Gewinnquellen, Zufallswahrscheinlichkeit
================================================================================================================

Voraussetzung: trend_cpo.py wurde gerechnet (ergebnisse_cpo/raster_kurven.pkl = Kontoverlauf aller festen
Kombinationen aus Einstieg und Ausstieg).

1. Sieger    : beste Sharpe Ratio und hoechste Rendite ueber den ganzen Zeitraum; dazu die Variante, die man
               Ende 2022 nur mit den Daten bis dahin gewaehlt haette, und wie sie danach lief (out-of-sample).
2. Marktlage : DAX und gleichgewichteter Index aller handelbaren Aktien je Jahr, Marktbreite, Zahl der Signale.
3. Gewinne   : Trefferquote, Gewinn-/Verlust-Verhaeltnis, Anteil der besten Trades und Aktien am Gewinn,
               Ergebnis ohne die besten Aktien, Markt-Beta und Alpha.
4. Zufall    : Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR, beruecksichtigt die Zahl der
               getesteten Varianten), Bootstrap-Reality-Check (White) fuer die beste von allen Varianten,
               jeweils absolut und gegenueber dem gleichgewichteten Aktienindex.

Aufruf (im Ordner trend_modell):  ..\\.venv\\python.exe trend_bewertung.py
Ergebnisse in ergebnisse_cpo/: bewertung_*.csv, bewertung.png
"""
from __future__ import annotations

import sys
from math import e as EULER_E

import numpy as np
import pandas as pd
from scipy.stats import norm

import trend_cpo as tc
import trend_modell as tm

LERN = (pd.Timestamp(tc.SIGNAL_START), pd.Timestamp("2022-12-31"))
TEST = (pd.Timestamp("2023-01-01"), None)
BOOTSTRAP = 2000
BLOCK = 20                       # mittlere Blocklaenge (Handelstage) im stationaeren Bootstrap
SEED = 3
START_DATEN = pd.date_range("2017-10-01", "2023-10-01", freq="6MS")   # Startzeitpunkte fuer die Robustheit
GAMMA = 0.5772156649             # Euler-Mascheroni-Konstante (Erwartungswert des Maximums)


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Kennzahlen
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
    out = {"rendite_pa_pct": ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100,
           "sharpe": r.mean() / r.std() * np.sqrt(252),
           "schwankung_pct": r.std() * np.sqrt(252) * 100,
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
    """Probabilistic Sharpe Ratio (Bailey/Lopez de Prado 2012), alles in Tageseinheiten.
    Gibt (Sharpe taeglich, Standardfehler, Wahrscheinlichkeit wahre Sharpe > sr_star) zurueck."""
    r = r[np.isfinite(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = pd.Series(r).skew()
    g4 = pd.Series(r).kurt() + 3                            # nicht ueberschuessige Kurtosis
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1))
    return sr, se, float(norm.cdf((sr - sr_star) / se))


def effective_n(R: np.ndarray) -> float:
    """Effektive Zahl unabhaengiger Varianten (Participation Ratio der Eigenwerte der Korrelationsmatrix)."""
    c = np.corrcoef(R, rowvar=False)
    c = np.nan_to_num(c)
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def expected_max_sr(var_sr: float, n: float) -> float:
    """Erwartete hoechste Sharpe Ratio (taeglich) von n unabhaengigen Varianten OHNE echten Vorteil."""
    n = max(n, 2)
    return np.sqrt(var_sr) * ((1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * EULER_E)))


def reality_check(R: np.ndarray, rng) -> tuple[float, np.ndarray]:
    """White's Reality Check: Wie oft erreicht die BESTE von allen Varianten im Zufall (alle Mittelwerte auf
    null gesetzt, Tage blockweise neu gezogen) eine so hohe Sharpe Ratio wie die beste echte?"""
    n, k = R.shape
    sd = R.std(axis=0, ddof=1)
    real = (R.mean(axis=0) / sd).max()
    Z = R - R.mean(axis=0)                                  # Nullhypothese: kein Vorteil
    mx = np.empty(BOOTSTRAP)
    ar = np.arange(n)
    for b0 in range(0, BOOTSTRAP, 100):                    # stationaerer Bootstrap (Politis/Romano), je 100
        B = min(100, BOOTSTRAP - b0)
        new = rng.random((B, n)) < 1 / BLOCK                # neuer Block beginnt mit Wahrscheinlichkeit 1/BLOCK
        new[:, 0] = True
        blk = np.cumsum(new, axis=1) - 1
        last = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
        starts = rng.integers(0, n, size=(B, n))
        idx = (np.take_along_axis(starts, blk, axis=1) + (ar - last)) % n
        for i in range(B):
            Zb = Z[idx[i]]
            mx[b0 + i] = (Zb.mean(axis=0) / Zb.std(axis=0, ddof=1)).max()
    return float((mx >= real).mean()), mx


# ======================================================================================= Trades des Siegers
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
        # Grund: Stopp vor Ablauf der Haltedauer oder regulaer
        tl["art"] = np.where(tl["tage"] < tc.AUSSTIEGE[exit_name].get("tage", 10 ** 6), "Stopp/Trendbruch", "Zeitablauf")
    return {"equity": sim["equity"], "trades": tl, "signale": len(rows)}


def trade_profile(tl: pd.DataFrame) -> dict:
    g = tl["ergebnis_eur"]
    win, loss = g[g > 0], g[g <= 0]
    top = g.sort_values(ascending=False)
    n10 = max(1, int(len(g) * 0.1))
    return {"trades": len(g), "trefferquote_pct": len(win) / len(g) * 100,
            "mittlerer_gewinn_pct": tl.loc[g > 0, "rendite_pct"].mean(),
            "mittlerer_verlust_pct": tl.loc[g <= 0, "rendite_pct"].mean(),
            "gewinn_verlust_verh": tl.loc[g > 0, "rendite_pct"].mean() / -tl.loc[g <= 0, "rendite_pct"].mean(),
            "profit_faktor": win.sum() / -loss.sum(),
            "gewinn_gesamt_eur": g.sum(),
            "anteil_top10pct_trades_am_gewinn_pct": top.iloc[:n10].sum() / g.sum() * 100,
            "anteil_top10_trades_am_gewinn_pct": top.iloc[:10].sum() / g.sum() * 100,
            "haltedauer_tage": tl["tage"].mean()}


def start_robustness(m: dict) -> pd.DataFrame:
    """Jede Kombination ab mehreren Startzeitpunkten rechnen (jeweils bis heute): Mit nur 10 Plaetzen und langen
    Haltedauern haengt das Ergebnis stark davon ab, welche Aktien zufaellig zu Beginn im Depot landen."""
    cal = m["close"].index
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    rows = []
    for en in tc.EN_NAMES:
        meta = tc.all_trades(m, tc.entry_signals(m, [en]))
        st = meta["t_signal"].to_numpy()
        for k, ex in enumerate(tc.EX_NAMES):
            res = []
            for s in START_DATEN:
                sel = np.flatnonzero(cal[st] >= s)
                tr = tc.to_trades(meta, sel, np.full(len(sel), k), mom[st[sel], meta["j"].to_numpy()[sel]])
                eq = tm.simulate(m, tr, "long", s, with_log=False)["equity"].dropna()
                r = eq.pct_change().dropna()
                yrs = len(r) / 252
                res.append((((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100,
                            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                            (1 - eq / eq.cummax()).max() * 100))
            a = np.array(res)
            rows.append({"einstieg": en, "ausstieg": ex, "rendite_pa_mittel": a[:, 0].mean(),
                         "sharpe_mittel": np.nanmean(a[:, 1]), "sharpe_min": np.nanmin(a[:, 1]),
                         "sharpe_max": np.nanmax(a[:, 1]), "maxdd_mittel": a[:, 2].mean(), "maxdd_max": a[:, 2].max()})
        log(f"  Startzeitpunkte: {en} fertig")
    return pd.DataFrame(rows)


# ======================================================================================= Ablauf
def main() -> None:
    tm.OUT_DIR = tc.OUT_DIR
    path = tc.OUT_DIR / "raster_kurven.pkl"
    if not path.exists():
        raise SystemExit("Bitte zuerst trend_cpo.py rechnen (raster_kurven.pkl fehlt).")
    curves = pd.read_pickle(path)
    m = tc.load_data()
    cal = m["close"].index
    start = pd.Timestamp(tc.SIGNAL_START)
    curves = curves[curves.index >= start]

    # Vergleichsmassstaebe: DAX-ETF und gleichgewichteter Index aller handelbaren Aktien (gleicher
    # Survivorship Bias wie die Strategien - der faire Vergleich)
    pf = m["close"].ffill()
    ew_r = pf.pct_change().where(m["elig"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ew = tm.CAPITAL * (1 + ew_r[ew_r.index >= start]).cumprod()
    bench = {"Gleichgewichteter Aktienindex (Universum)": ew}
    if m["bench"] is not None:
        b = m["bench"][m["bench"].index >= start].ffill().dropna()
        bench[f"DAX ({tm.BENCHMARK['aktien']})"] = tm.CAPITAL * b / b.iloc[0]
    ew_ret = ew.pct_change()

    # ---------------- 1. Kennzahlen aller Kombinationen
    rows = []
    for col in curves.columns:
        en, ex = col.split(" | ")
        r = {"einstieg": en, "ausstieg": ex}
        for lbl, per in (("gesamt", (start, None)), ("lern", LERN), ("test", TEST)):
            for k, v in stats(curves[col], per, ew_ret).items():
                r[f"{lbl}_{k}"] = v
        rows.append(r)
    for name, eq in bench.items():
        r = {"einstieg": name, "ausstieg": "-"}
        for lbl, per in (("gesamt", (start, None)), ("lern", LERN), ("test", TEST)):
            for k, v in stats(eq, per, ew_ret).items():
                r[f"{lbl}_{k}"] = v
        rows.append(r)
    allm = pd.DataFrame(rows)
    tm.save_csv(allm, "bewertung_alle_varianten.csv")
    strat = allm[allm["ausstieg"] != "-"]

    log(f"Robustheit: alle Kombinationen ab {len(START_DATEN)} Startzeitpunkten ...")
    robs = start_robustness(m)
    tm.save_csv(robs, "bewertung_startzeitpunkte.csv")
    best_sh = strat.sort_values("gesamt_sharpe", ascending=False).iloc[0]
    best_ret = strat.sort_values("gesamt_rendite_pa_pct", ascending=False).iloc[0]
    best_lern = strat.sort_values("lern_sharpe", ascending=False).iloc[0]
    rank_test = int((strat["test_sharpe"] > best_lern["test_sharpe"]).sum()) + 1
    key = lambda r: f"{r['einstieg']} | {r['ausstieg']}"

    # ---------------- 2. Marktlage je Jahr
    sma200 = pf.rolling(200).mean()
    breadth = (pf > sma200).astype(float).where(m["elig"] & sma200.notna()).mean(axis=1)
    hi200 = (pf > pf.rolling(200).max().shift(1)).astype(float).where(m["elig"]).sum(axis=1)
    yr = lambda eq: cut(eq, (start, None)).groupby(lambda d: d.year).last()
    def yearly(eq):
        y = yr(eq)
        return (y / y.shift(1).fillna(cut(eq, (start, None)).iloc[0]) - 1) * 100
    regime = pd.DataFrame({name: yearly(eq) for name, eq in bench.items()})
    regime["Marktbreite (Anteil ueber 200-Tage-Linie, %)"] = breadth.groupby(breadth.index.year).mean() * 100
    regime["neue 200-Tage-Hochs je Tag"] = hi200.groupby(hi200.index.year).mean()
    regime[f"Beste Sharpe: {key(best_sh)}"] = yearly(curves[key(best_sh)])
    for i, (_, r) in enumerate(robs.sort_values("sharpe_mittel", ascending=False).head(2).iterrows()):
        regime[f"Robust {i + 1}: {key(r)}"] = yearly(curves[key(r)])
    if key(best_ret) != key(best_sh):
        regime[f"Hoechste Rendite: {key(best_ret)}"] = yearly(curves[key(best_ret)])
    regime = regime[regime.index >= start.year]
    tm.save_csv(regime.round(1), "bewertung_marktlage.csv", index=True)

    # ---------------- 3. Gewinnquellen des Siegers (beste Sharpe) und der hoechsten Rendite
    profiles, tops, robust = {}, {}, []
    rob_top = robs.sort_values("sharpe_mittel", ascending=False).head(2)
    cands = [("Beste Sharpe", best_sh), ("Hoechste Rendite", best_ret)] + \
            [(f"Robust {i + 1}", r) for i, (_, r) in enumerate(rob_top.iterrows())]
    for label, row in cands:
        k = key(row)
        if k in profiles:
            continue
        w = winner_trades(m, row["einstieg"], row["ausstieg"])
        tl = w["trades"]
        profiles[k] = trade_profile(tl)
        profiles[k].update({f"anteil_{a}_pct": v for a, v in (tl["art"].value_counts(normalize=True) * 100).items()})
        for a, g in tl.groupby("art"):
            profiles[k][f"rendite_{a}_pct"] = g["rendite_pct"].mean()
        by_tick = tl.groupby(["ticker", "name"])["ergebnis_eur"].agg(["sum", "size"]).sort_values("sum", ascending=False)
        by_tick["anteil_am_gewinn_pct"] = by_tick["sum"] / tl["ergebnis_eur"].sum() * 100
        tops[k] = by_tick.head(10)
        tm.save_csv(tl, "bewertung_trades_" + "".join(ch if ch.isalnum() else "_" for ch in k.lower()) + ".csv")
        for n_ex in (0, 3, 5, 10):
            ex = list(by_tick.index.get_level_values(0)[:n_ex])
            eq = w["equity"] if n_ex == 0 else winner_trades(m, row["einstieg"], row["ausstieg"], ex)["equity"]
            s_all, s_test = stats(eq, (start, None)), stats(eq, TEST)
            robust.append({"variante": k, "ohne_beste_aktien": n_ex, "rendite_pa_pct": s_all["rendite_pa_pct"],
                           "sharpe": s_all["sharpe"], "test_rendite_pa_pct": s_test["rendite_pa_pct"],
                           "test_sharpe": s_test["sharpe"], "max_drawdown_pct": s_all["max_drawdown_pct"]})
    prof = pd.DataFrame(profiles).T
    tm.save_csv(prof, "bewertung_gewinnquellen.csv", index=True)
    rob = pd.DataFrame(robust)
    tm.save_csv(rob, "bewertung_ohne_beste_aktien.csv")

    # ---------------- 4. Zufallswahrscheinlichkeit
    R_abs = curves.pct_change().iloc[1:].fillna(0)
    R_exc = R_abs.sub(ew_ret.reindex(R_abs.index).fillna(0), axis=0)
    n_eff = {"absolut": effective_n(R_abs.to_numpy()), "ueber Index": effective_n(R_exc.to_numpy())}
    rng = np.random.default_rng(SEED)
    stat_rows = []
    rc = {}
    for lbl, R in (("absolut", R_abs), ("ueber Index", R_exc)):
        srs = R.mean() / R.std()
        var_sr = srs.var()
        sr0 = expected_max_sr(var_sr, len(R.columns))
        sr0_eff = expected_max_sr(var_sr, n_eff[lbl])
        log(f"Reality Check ({lbl}, {BOOTSTRAP} Ziehungen) ...")
        p_rc, mx = reality_check(R.to_numpy(), rng)
        rc[lbl] = (p_rc, mx, srs.max())
        for name, row in [("Beste Sharpe (gesamt)", best_sh), ("Hoechste Rendite (gesamt)", best_ret)] + \
                [(f"Robust {i + 1} (Mittel Startzeitpunkte)", r) for i, (_, r) in enumerate(rob_top.iterrows())]:
            r = R[key(row)].to_numpy()
            sr, se, p0 = psr(r, 0.0)
            _, _, d_all = psr(r, sr0)
            _, _, d_eff = psr(r, sr0_eff)
            stat_rows.append({"variante": f"{name}: {key(row)}", "massstab": lbl, "zeitraum": "gesamt",
                              "sharpe_pa": sr * np.sqrt(252), "standardfehler_pa": se * np.sqrt(252),
                              "t_wert": sr / se, "psr_wahre_sharpe_ueber_0": p0,
                              "varianten": len(R.columns), "effektiv_unabhaengig": n_eff[lbl],
                              "erwartete_max_sharpe_zufall_pa": sr0 * np.sqrt(252),
                              "erwartete_max_sharpe_zufall_eff_pa": sr0_eff * np.sqrt(252),
                              "dsr_alle": d_all, "dsr_effektiv": d_eff, "reality_check_p": p_rc})
        # out-of-sample: Ende 2022 gewaehlt, nur Testzeitraum (keine Auswahl im Test -> keine Deflation noetig)
        rt = cut(R[key(best_lern)], TEST).to_numpy()
        sr, se, p0 = psr(rt, 0.0)
        stat_rows.append({"variante": f"Ende 2022 gewaehlt: {key(best_lern)}", "massstab": lbl, "zeitraum": "test",
                          "sharpe_pa": sr * np.sqrt(252), "standardfehler_pa": se * np.sqrt(252), "t_wert": sr / se,
                          "psr_wahre_sharpe_ueber_0": p0, "varianten": 1})
    st = pd.DataFrame(stat_rows)
    tm.save_csv(st, "bewertung_zufall.csv")

    chart(curves, bench, best_sh, best_ret, best_lern, key, rc, rob_top)
    report(allm, strat, best_sh, best_ret, best_lern, rank_test, regime, prof, tops, rob, st, key, robs)


def chart(curves, bench, best_sh, best_ret, best_lern, key, rc, rob_top) -> None:
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
    for lbl, row, c in [("Beste Sharpe ab 2017", best_sh, "#d62728"), ("Hoechste Rendite", best_ret, "#ff7f0e"),
                        ("Ende 2022 gewaehlt", best_lern, "#2ca02c")] + \
                       [(f"Robust {i + 1} (Startzeitpunkte)", r, c) for i, (r, c) in enumerate(zip(rr, ["#9467bd", "#8c564b"]))]:
        k = key(row)
        if k in shown:
            continue
        shown.append(k)
        ax.plot(curves.index, curves[k] / 1000, color=c, lw=2, label=f"{lbl}: {k}", zorder=3)
    for (name, eq), c in zip(bench.items(), ["#1f77b4", "#555555"]):
        ax.plot(eq.index, eq / 1000, color=c, lw=1.6, ls="--", label=name, zorder=2)
    ax.axvline(TEST[0], color="black", lw=0.8, ls=":")
    ax.text(TEST[0], ax.get_ylim()[1] if ax.get_yscale() == "linear" else 1, "  Test ab 2023", va="top", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Alle 270 festen Kombinationen (grau) - Konto in Tsd. EUR (log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    p, mx, real = rc["ueber Index"]
    ax2.hist(mx * np.sqrt(252), bins=50, color="#9ecae1")
    ax2.axvline(real * np.sqrt(252), color="#d62728", lw=2)
    ax2.set_title(f"Reality Check: beste Information Ratio aller Varianten\nim Zufall (blau) vs. echt (rot), p = {p:.2f}")
    ax2.set_xlabel("hoechste Information Ratio gegenueber dem Aktienindex (p. a.)")
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / "bewertung.png", dpi=110)
    plt.close(fig)


def report(allm, strat, best_sh, best_ret, best_lern, rank_test, regime, prof, tops, rob, st, key, robs) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    cols = ["einstieg", "ausstieg", "gesamt_rendite_pa_pct", "gesamt_sharpe", "gesamt_max_drawdown_pct",
            "gesamt_beta", "gesamt_alpha_pa_pct", "lern_sharpe", "test_sharpe", "lern_rendite_pa_pct",
            "test_rendite_pa_pct"]
    print("\n" + "=" * 120)
    print("1) SIEGER (270 feste Kombinationen, Sept. 2017 bis heute)")
    print("=" * 120)
    print("\nTop 10 nach Sharpe Ratio (gesamt):")
    print(strat.sort_values("gesamt_sharpe", ascending=False)[cols].head(10).round(2).to_string(index=False))
    print("\nTop 10 nach Rendite p. a. (gesamt):")
    print(strat.sort_values("gesamt_rendite_pa_pct", ascending=False)[cols].head(10).round(2).to_string(index=False))
    print("\nVergleichsmassstaebe:")
    print(allm[allm["ausstieg"] == "-"][cols].round(2).to_string(index=False))
    print(f"\nEnde 2022 (nur mit Lerndaten) gewaehlt: {key(best_lern)} - Lern-Sharpe {best_lern['lern_sharpe']:.2f}, "
          f"Test-Sharpe {best_lern['test_sharpe']:.2f} (Rang {rank_test} von {len(strat)} im Test)")
    rc = strat[["lern_sharpe", "test_sharpe"]].corr(method="spearman").iloc[0, 1]
    print(f"Rangkorrelation Lern- gegen Test-Sharpe ueber alle Kombinationen: {rc:+.2f}")
    print(f"\nRobustheit ueber {len(START_DATEN)} Startzeitpunkte (halbjaehrlich 2017-2023, jeweils bis heute):")
    rc_ = ["einstieg", "ausstieg", "rendite_pa_mittel", "sharpe_mittel", "sharpe_min", "sharpe_max", "maxdd_mittel",
           "maxdd_max"]
    print("Top 10 nach mittlerer Sharpe Ratio:")
    print(robs.sort_values("sharpe_mittel", ascending=False)[rc_].head(10).round(2).to_string(index=False))
    print("Top 10 nach mittlerer Rendite:")
    print(robs.sort_values("rendite_pa_mittel", ascending=False)[rc_].head(10).round(2).to_string(index=False))
    for lbl, row in (("Beste Sharpe ab 2017", best_sh),):
        rr = robs[(robs["einstieg"] == row["einstieg"]) & (robs["ausstieg"] == row["ausstieg"])]
        print(f"{lbl} ({key(row)}) ueber alle Startzeitpunkte:")
        print(rr[rc_].round(2).to_string(index=False))
    print("\n" + "=" * 120)
    print("2) MARKTLAGE JE JAHR (Rendite in %)")
    print("=" * 120)
    print(regime.round(1).to_string())
    print("\n" + "=" * 120)
    print("3) WIE DER GEWINN ENTSTEHT")
    print("=" * 120)
    print(prof.T.round(2).to_string())
    for k, t in tops.items():
        print(f"\nAktien mit dem groessten Gewinnbeitrag - {k}:")
        print(t.round(1).to_string())
    print("\nErgebnis ohne die Aktien mit dem groessten Gewinnbeitrag:")
    print(rob.round(2).to_string(index=False))
    print("\n" + "=" * 120)
    print("4) WAHRSCHEINLICHKEIT FUER ZUFALL")
    print("=" * 120)
    print(st.round(3).to_string(index=False))
    print(f"\nDateien: {tc.OUT_DIR} (bewertung_*.csv, bewertung.png)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
