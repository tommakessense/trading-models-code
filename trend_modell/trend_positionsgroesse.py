"""
Positionsgroesse im Trendmodell: feste 1/10 oder je nach Signalqualitaet (CPO und einfache Regeln)?
================================================================================================================

Gleiche Aktienauswahl wie bisher (Signal, Rangfolge nach 6-Monats-Momentum), nur die Groesse je Kauf aendert sich:
    Gewicht w zwischen GEWICHT_MIN und GEWICHT_MAX des Depotwerts (Standard 1/10)
    Regel fuer Kaeufe ueber 1/10: Der Mehrbetrag kommt nur aus freiem Geld - nach dem Kauf muessen noch
    RESERVE (1/10) des Depotwerts als Bargeld uebrig sein, damit das naechste normale Signal gekauft werden kann.
    Reicht das Geld nicht, wird weniger (mindestens das Standardgewicht, soweit vorhanden) gekauft.

Groessenregeln (Bewertung jedes Signals nur mit Daten bis zum Signaltag):
    fest 1/10            wie bisher
    CPO[je Tag]          Vorhersage des CPO-Modells (trend_cpo.py, Walk-forward, je Jahr nur mit Vorjahren trainiert)
    CPO[relativ]         dito, Ziel relativ zu gleichzeitigen Signalen
    Momentum-Staerke     staerkerer 6-Monats-Trend -> groessere Position
    Schwankung           ruhigere Aktie -> groessere Position (gleiches Risiko je Position, "inverse Volatilitaet")
    Zufall               gleiche Verteilung der Gewichte wie die Stufenregeln, aber zufaellig (Kontrolle)
Stufen: Bewertung im obersten Fuenftel der letzten 250 Handelstage -> GEWICHT_MAX, im untersten -> GEWICHT_MIN,
sonst 1/10.

Aufruf (im Ordner trend_modell, nach trend_cpo.py):  ..\\.venv\\python.exe trend_positionsgroesse.py
Ergebnisse in ergebnisse_cpo/: positionsgroesse.csv, positionsgroesse.png
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import trend_cpo as tc
import trend_modell as tm

STRATEGIEN = {
    "Robust 1: Hoch 200 | Zeit 20": ("Hoch 200", "Zeit 20"),
    "Robust 2: Hoch 50 + Volumenschub | Zeit 60 + Stopp 15%": ("Hoch 50 + Volumenschub", "Zeit 60 + Stopp 15%"),
}
GEWICHT_STANDARD = 0.10
GEWICHT_MIN = 0.05
GEWICHT_MAX = 0.20
RESERVE = 0.10                   # nach einem Kauf ueber Standardgewicht muss so viel Bargeld uebrig sein
OBEN, UNTEN = 0.8, 0.2           # Fuenftel fuer die Stufen
HISTORIE = 250                   # Vergleich mit den Bewertungen der Signale der letzten 250 Handelstage
MAX_POSITIONEN = 20              # Obergrenze der Anzahl (bei kleinen Positionen passen mehr als 10 hinein)
MIN_KAUF = 0.05                  # kein Kauf unter 5 % des Depotwerts (sonst entstehen Mini-Positionen aus Restgeld)
ZUFALL_LAEUFE = 20
START_DATEN = pd.date_range(f"{tc.ERSTES_TESTJAHR}-01-01", "2023-10-01", freq="6MS")
SEED = 21


def log(msg: str) -> None:
    tm.log(msg)


def tiers(score: np.ndarray, t_sig: np.ndarray) -> np.ndarray:
    """Stufen aus dem Rang der Bewertung unter den Signalen der letzten HISTORIE Handelstage (ohne Zukunft)."""
    w = np.full(len(score), GEWICHT_STANDARD)
    order = np.argsort(t_sig, kind="stable")
    hist_t, hist_s = [], []
    for i in order:
        t, s = t_sig[i], score[i]
        while hist_t and hist_t[0] < t - HISTORIE:
            hist_t.pop(0)
            hist_s.pop(0)
        if np.isfinite(s) and len(hist_s) >= 30:
            p = (np.array(hist_s) < s).mean()
            w[i] = GEWICHT_MAX if p >= OBEN else (GEWICHT_MIN if p < UNTEN else GEWICHT_STANDARD)
        if np.isfinite(s):
            hist_t.append(t)
            hist_s.append(s)
    return w


def simulate(m: dict, tr: pd.DataFrame, start: pd.Timestamp) -> dict:
    """Wie trend_modell.simulate (long, Schlussauktion), aber mit Gewicht je Trade und Reserve-Regel."""
    cal = m["close"].index
    cols = m["close"].columns
    P = m["close"].ffill().to_numpy()
    n = len(cal)
    slip = tm.SLIPPAGE_BPS / 1e4
    first = int(np.searchsorted(cal, start))
    by_entry: dict[int, list] = {}
    for r in tr.sort_values(["e", "rang"], ascending=[True, False]).itertuples(index=False):
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
            if len(pos) >= MAX_POSITIONEN:
                break
            j = r.j
            if j in pos or not np.isfinite(P[t, j]) or r.x <= t:
                continue
            w = min(r.w, GEWICHT_MAX)
            if w > GEWICHT_STANDARD:
                amount = min(w * V, cash - RESERVE * V)        # Mehrbetrag nur aus freiem Geld
                amount = max(amount, min(GEWICHT_STANDARD * V, cash))
            else:
                amount = min(w * V, cash)
            if amount < MIN_KAUF * V:
                continue
            shares = int(amount // P[t, j])
            if shares <= 0:
                continue
            alloc = shares * P[t, j]
            cost = tm.commission(alloc, cols[j]) + alloc * slip
            if alloc + cost > cash:
                continue
            cash -= alloc + cost
            pos[j] = {"x": int(r.x), "shares": shares, "px_out": r.kurs_aus}
            sizes.append((r.w, alloc / V))
        held = sum(q["shares"] * P[t, j] for j, q in pos.items())
        equity[t] = cash + held
        invested[t] = held / equity[t]
        npos[t] = len(pos)
    s = slice(first, n)
    return {"equity": pd.Series(equity, index=cal).iloc[s], "investiert": np.nanmean(invested[s]),
            "positionen": np.nanmean(npos[s]), "groesse_mittel": np.mean([b for _, b in sizes]) if sizes else np.nan,
            "gross_umgesetzt": (np.mean([b >= 0.15 for a, b in sizes if a > GEWICHT_STANDARD])
                                if any(a > GEWICHT_STANDARD for a, _ in sizes) else np.nan)}


def stats(eq: pd.Series) -> dict:
    s = eq.dropna()
    r = s.pct_change().dropna()
    y = len(r) / 252
    return {"rendite_pa_pct": ((s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1) * 100,
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
            extra = {"investiert_pct": res["investiert"] * 100, "positionen_mittel": res["positionen"],
                     "kaufgroesse_mittel_pct": res["groesse_mittel"] * 100,
                     "grosse_kaeufe_umgesetzt_pct": res["gross_umgesetzt"] * 100}
            first_eq = res["equity"]
    d = pd.DataFrame(out)
    r = {f"ab{tc.ERSTES_TESTJAHR}_{c}": d.iloc[0][c] for c in d.columns}
    r.update({f"starts_{c}_mittel": d[c].mean() for c in d.columns})
    r.update(extra)
    return r, first_eq


def main() -> None:
    tm.OUT_DIR = tc.OUT_DIR
    m = tc.load_data()
    cal = m["close"].index
    models = tc.load_models()
    if models is None:
        import cpo_stresstest as cs
        models = cs.train_models(m, False)
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    vol = np.log(m["close"].ffill()).diff().rolling(60, min_periods=40).std().to_numpy() * np.sqrt(252)
    test_start = pd.Timestamp(f"{tc.ERSTES_TESTJAHR}-01-01")
    starts = [test_start] + [s for s in START_DATEN if s > test_start]
    rng = np.random.default_rng(SEED)
    rows_out, curves, ic_rows = [], {}, []
    for sname, (entry, exit_name) in STRATEGIEN.items():
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
        scores["Momentum-Staerke"] = rank
        # Schwankung: Ziel = gleiches Risiko je Position wie eine durchschnittlich schwankende Aktie
        v = vol[st[rows], jj[rows]]
        med = np.nanmedian(vol, axis=1)[st[rows]]
        w_vol = np.clip(GEWICHT_STANDARD * med / v, GEWICHT_MIN, GEWICHT_MAX)
        w_vol = np.where(np.isfinite(w_vol), w_vol, GEWICHT_STANDARD)
        # Wie gut sagt die Bewertung die spaetere Rendite je Trade voraus? (abgeschlossene Trades)
        y = (meta.attrs["R"][rows, k] - tc.KOSTEN)
        done = ~meta.attrs["OPEN"][rows, k]
        for nm, sc in scores.items():
            ok = done & np.isfinite(sc)
            ic_rows.append({"strategie": sname, "bewertung": nm,
                            "rangkorrelation_mit_trade_rendite": pd.Series(sc[ok]).corr(pd.Series(y[ok]), method="spearman"),
                            "rendite_oberstes_fuenftel_pct": y[ok][sc[ok] >= np.nanquantile(sc[ok], OBEN)].mean() * 100,
                            "rendite_unterstes_fuenftel_pct": y[ok][sc[ok] < np.nanquantile(sc[ok], UNTEN)].mean() * 100,
                            "rendite_alle_pct": y[ok].mean() * 100})
        # Bezug: das bisherige Trendmodell (feste 1/10, hoechstens 10 Positionen, Restgeld wird auch klein investiert)
        res0 = []
        for i, s0 in enumerate(starts):
            sel = rows[cal[st[rows]] >= s0]
            tr0 = tc.to_trades(meta, sel, np.full(len(sel), k), mom[st[sel], jj[sel]])
            eq0 = tm.simulate(m, tr0, "long", s0, with_log=False)["equity"]
            res0.append(stats(eq0))
            if i == 0:
                curves[(sname, "Trendmodell bisher")] = eq0
        d0 = pd.DataFrame(res0)
        r0 = {f"ab{tc.ERSTES_TESTJAHR}_{c}": d0.iloc[0][c] for c in d0.columns}
        r0.update({f"starts_{c}_mittel": d0[c].mean() for c in d0.columns})
        r0.update({"strategie": sname, "groessenregel": "Trendmodell bisher (1/10, max. 10 Positionen)"})
        rows_out.append(r0)
        rules = {"fest 1/10": np.full(len(rows), GEWICHT_STANDARD)}
        for nm, sc in scores.items():
            rules[nm] = tiers(sc, st[rows])
        rules["Schwankung (inverse Volatilitaet)"] = w_vol
        for nm, w in rules.items():
            r, eq = run_rule(m, meta, rows, k, rank, w, starts)
            r.update({"strategie": sname, "groessenregel": nm,
                      "anteil_gross_pct": (w > GEWICHT_STANDARD).mean() * 100,
                      "anteil_klein_pct": (w < GEWICHT_STANDARD).mean() * 100})
            rows_out.append(r)
            curves[(sname, nm)] = eq
        # Zufall mit gleicher Verteilung der Stufen wie CPO[je Tag]
        w_ref = rules[[n for n in rules if n.startswith("CPO")][0]]
        rand = []
        for _ in range(ZUFALL_LAEUFE):
            r, _eq = run_rule(m, meta, rows, k, rank, rng.permutation(w_ref), starts[:1])
            rand.append(r)
        d = pd.DataFrame(rand)
        r = d.mean(numeric_only=True).to_dict()
        r.update({"strategie": sname, "groessenregel": f"Zufall (Mittel aus {ZUFALL_LAEUFE})",
                  "zufall_sharpe_spanne": f"{d[f'ab{tc.ERSTES_TESTJAHR}_sharpe'].min():.2f} bis "
                                          f"{d[f'ab{tc.ERSTES_TESTJAHR}_sharpe'].max():.2f}"})
        rows_out.append(r)
    res = pd.DataFrame(rows_out)
    res = res[["strategie", "groessenregel"] + [c for c in res.columns if c not in ("strategie", "groessenregel")]]
    ic = pd.DataFrame(ic_rows)
    tm.save_csv(res, "positionsgroesse.csv")
    tm.save_csv(ic, "positionsgroesse_vorhersage.csv")
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
            kw = {"color": "black", "lw": 2} if rule.startswith("Trendmodell") else {"lw": 1.3}
            ax.plot(eq.index, eq / 1000, label=rule, **kw)
        ax.set_yscale("log")
        ax.set_title(nm, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Positionsgroesse: Konto in Tsd. EUR (log), ab {tc.ERSTES_TESTJAHR}")
    fig.tight_layout()
    fig.savefig(tc.OUT_DIR / "positionsgroesse.png", dpi=110)
    plt.close(fig)


def report(res: pd.DataFrame, ic: pd.DataFrame) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    y = tc.ERSTES_TESTJAHR
    cols = [c for c in ["groessenregel", f"ab{y}_rendite_pa_pct", f"ab{y}_sharpe", f"ab{y}_max_drawdown_pct",
                        "starts_rendite_pa_pct_mittel", "starts_sharpe_mittel", "starts_max_drawdown_pct_mittel",
                        "investiert_pct", "positionen_mittel", "kaufgroesse_mittel_pct", "grosse_kaeufe_umgesetzt_pct",
                        "anteil_gross_pct",
                        "anteil_klein_pct", "zufall_sharpe_spanne"] if c in res.columns]
    print("\n" + "=" * 120)
    print(f"POSITIONSGROESSE  ({GEWICHT_MIN:.0%} bis {GEWICHT_MAX:.0%} je Kauf, Reserve {RESERVE:.0%}; Kosten: "
          f"{tm.broker_text()})")
    print("=" * 120)
    for s, g in res.groupby("strategie", sort=False):
        print(f"\n{s}")
        print(g[cols].round(2).to_string(index=False))
    print("\nSagt die Bewertung die Rendite je Trade voraus? (Testjahre, abgeschlossene Trades, nach Kosten)")
    print(ic.round(3).to_string(index=False))
    print(f"\nDateien: {tc.OUT_DIR} (positionsgroesse*.csv, positionsgroesse.png)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
