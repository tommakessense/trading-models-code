"""
CPO (Conditional Parameter Optimization, nach Ernest Chan) fuer das Trendfolge-Modell - Xetra-Aktien, long
================================================================================================================

Idee: Statt einmal fest "200-Tage-Hoch, 20 Tage halten" zu waehlen, entscheidet ein Modell bei jedem Signal
anhand der Lage am Signaltag (Aktie und Gesamtmarkt), welcher Ausstieg die beste Rendite je Haltetag
erwarten laesst - und ob sich das Signal ueberhaupt lohnt.

Einstiege : 15 Varianten aus 8 Familien (Ausbruch auf N-Tage-Hoch, Kurs ueber gleitender Linie, Kreuzen
            zweier Linien, Zeitreihen-Momentum, Querschnitts-Momentum 12-1, Bollinger-Ausbruch,
            Ausbruch mit Volumenschub), siehe EINSTIEGE.
Ausstiege : 12 Varianten - feste Haltedauer 5 bis 120 Tage, Trendbruch (N-Tage-Tief), nachgezogener Stopp.
Modell    : ein Gradient-Boosting-Modell fuer alle Kombinationen. Ziel: Rendite je Haltetag nach Kosten.
            Merkmale: Rendite/Schwankung/Abstand zum Hoch/Umsatz der Aktie, Lage des Gesamtmarkts
            (Trend, Schwankung, Anteil der Aktien ueber der 200-Tage-Linie, Zahl der Ausbrueche, Streuung),
            Art des Einstiegs und des Ausstiegs, dazu ein reines Zufallsmerkmal (Placebo) als Kontrolle.
Pruefung  : Walk-forward - fuer jedes Testjahr ab ERSTES_TESTJAHR wird nur mit Signalen trainiert, deren Trades
            vor Jahresbeginn abgeschlossen waren. Danach Portfolio wie im Trendmodell (50.000 EUR,
            max. 10 Positionen, IBKR-Gebuehren, Slippage).
Vergleich : feste Variante des Trendmodells, immer 120 Tage, jaehrlich neu gewaehlte beste feste Kombination,
            Zufallswahl (gleich viele Signale bzw. zufaelliger Ausstieg), Kaufen und Halten.

Aufruf (im Ordner trend_modell):
    ..\\.venv\\python.exe trend_cpo.py            Walk-forward-Rueckrechnung (einige Minuten)
    ..\\.venv\\python.exe trend_cpo.py --heute    heutige Signale mit vom Modell gewaehltem Ausstieg

Ergebnisse in ergebnisse_cpo/: vergleich.csv, jahre.csv, raster_sharpe.csv/.png (alle festen Kombinationen),
gewaehlte_ausstiege.csv, merkmale.csv (Wichtigkeit), trefferguete.csv, trades_*.csv.

Hinweis: Der Kurs-Cache enthaelt nur heute noch handelbare Aktien (Survivorship Bias) - das schoent alle
Varianten gleichermassen; fuer den Vergleich der Varianten untereinander ist es weniger wichtig.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import trend_modell as tm

warnings.filterwarnings("ignore")

# ======================================================================================= Einstellungen
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "ergebnisse_cpo"

SIGNAL_START = "2017-09-01"      # erste handelbare Aktien (250 Tage Kursvorlauf)
ERSTES_TESTJAHR = 2020           # Walk-forward: 2020, 2021, ... (Training immer nur mit den Jahren davor)

EINSTIEGE = {                    # Name: (Familie, Parameter)
    "Hoch 20": ("hoch", 20),
    "Hoch 50": ("hoch", 50),
    "Hoch 100": ("hoch", 100),
    "Hoch 200": ("hoch", 200),
    "Hoch 250 (52 Wochen)": ("hoch", 250),
    "Ueber Linie 50": ("linie", 50),
    "Ueber Linie 100": ("linie", 100),
    "Ueber Linie 200": ("linie", 200),
    "Kreuz 20/50": ("kreuz", (20, 50)),
    "Kreuz 50/200": ("kreuz", (50, 200)),
    "Momentum 6M positiv": ("zeitmom", 126),
    "Momentum 12M positiv": ("zeitmom", 252),
    "Top-10% Momentum 12-1": ("quermom", 252),
    "Bollinger 20/2": ("bollinger", 20),
    "Hoch 50 + Volumenschub": ("volumen", 50),
}
AUSSTIEGE = {
    "Zeit 5": {"typ": "zeit", "tage": 5},
    "Zeit 10": {"typ": "zeit", "tage": 10},
    "Zeit 20": {"typ": "zeit", "tage": 20},
    "Zeit 40": {"typ": "zeit", "tage": 40},
    "Zeit 60": {"typ": "zeit", "tage": 60},
    "Zeit 120": {"typ": "zeit", "tage": 120},
    "Trend 10": {"typ": "trend", "tage": 10},           # Schluss unter dem 10-Tage-Tief
    "Trend 20": {"typ": "trend", "tage": 20},
    "Trend 50": {"typ": "trend", "tage": 50},
    "Nachzug 10%": {"typ": "nachzug", "pct": 0.10},     # 10 % unter dem hoechsten Schluss seit Einstieg
    "Nachzug 15%": {"typ": "nachzug", "pct": 0.15},
    "Nachzug 25%": {"typ": "nachzug", "pct": 0.25},
    # Kombination: feste Haltedauer, vorher raus, wenn der Kurs X % unter den hoechsten Schluss seit Einstieg faellt
    "Zeit 40 + Stopp 10%": {"typ": "kombi", "tage": 40, "pct": 0.10},
    "Zeit 40 + Stopp 15%": {"typ": "kombi", "tage": 40, "pct": 0.15},
    "Zeit 60 + Stopp 10%": {"typ": "kombi", "tage": 60, "pct": 0.10},
    "Zeit 60 + Stopp 15%": {"typ": "kombi", "tage": 60, "pct": 0.15},
    "Zeit 120 + Stopp 10%": {"typ": "kombi", "tage": 120, "pct": 0.10},
    "Zeit 120 + Stopp 15%": {"typ": "kombi", "tage": 120, "pct": 0.15},
}
MAX_HALTEDAUER = 250
BASIS = ("Hoch 200", "Zeit 20")  # beste Variante des Trendmodells im Lernzeitraum (200-Tage-Hoch, Zeit kurz)
LANG = ("Hoch 200", "Zeit 120")
VOLUMEN_SCHUB = 2.0              # "Hoch 50 + Volumenschub": Stuecke mind. 2x Median der letzten 50 Tage
MIN_SCORE_BP = 0.0               # CPO-Einstieg: nur Signale mit erwarteter Rendite je Tag > 0 (nach Kosten)
ZIELE = {                        # was das Modell vorhersagt (beide werden gerechnet und verglichen)
    "je Tag": "absolut",         # Rendite je Haltetag nach Kosten
    "relativ": "relativ",        # dasselbe minus Mittel aller Signale desselben Monats: trennt die Auswahl unter
}                                # gleichzeitigen Signalen vom allgemeinen Marktverlauf
ZUFALL_LAEUFE = 20
SEED = 11
MAX_TRAIN_ZEILEN = 600_000       # mehr Trainingszeilen: zufaellige Auswahl (Laufzeit)
MODELL = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
              l2_regularization=1.0, random_state=SEED)
KOSTEN = 2 * (tm.SLIPPAGE_BPS / 1e4 + tm.cost_pct())   # Kauf + Verkauf, wie im Trendmodell (Broker dort)

EX_NAMES = list(AUSSTIEGE)
EN_NAMES = list(EINSTIEGE)


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Daten und Signale
def load_data() -> dict:
    m = tm.load_market("aktien")
    raw = pd.read_pickle(tm.SCREENER_CACHE / tm.STOCK_FILE)
    m["volume"] = raw["volume"].reindex(index=m["close"].index, columns=m["close"].columns).astype(float)
    return m


def _first(cond: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Erster Tag, an dem die Bedingung gilt (am Vortag galt sie nicht, Vortag aber schon berechenbar)."""
    return cond & ~cond.shift(1, fill_value=False) & valid.shift(1, fill_value=False)


def entry_signals(m: dict, names: list[str] | None = None) -> dict[str, np.ndarray]:
    c = m["close"]
    idx = c.index
    out = {}
    for name, (fam, n) in EINSTIEGE.items():
        if names is not None and name not in names:
            continue
        if fam in ("hoch", "volumen"):
            hi = c.rolling(n, min_periods=n).max().shift(1)
            cond = c > hi
            if fam == "volumen":
                v = m["volume"]
                cond &= v > VOLUMEN_SCHUB * v.rolling(50, min_periods=30).median().shift(1)
            ev = _first(cond, hi.notna())
        elif fam == "linie":
            s = c.rolling(n, min_periods=n).mean()
            ev = _first(c > s, s.notna())
        elif fam == "kreuz":
            f, sl = n
            a, b = c.rolling(f, min_periods=f).mean(), c.rolling(sl, min_periods=sl).mean()
            ev = _first(a > b, a.notna() & b.notna())
        elif fam == "zeitmom":
            r = c / c.shift(n) - 1
            ev = _first(r > 0, r.notna())
        elif fam == "bollinger":
            s, sd = c.rolling(n, min_periods=n).mean(), c.rolling(n, min_periods=n).std()
            ev = _first(c > s + 2 * sd, sd.notna())
        elif fam == "quermom":                              # Monatsende: neu in den besten 10 % (12-1 Monate)
            score = (c.shift(21) / c.shift(n) - 1).where(m["elig"])
            top = score.rank(axis=1, pct=True) >= 0.9
            per = idx.to_period("M")
            me = pd.Series(idx, index=idx).groupby(per).transform("max").to_numpy() == idx.to_numpy()
            me[-1] = False                                  # laufender Monat noch nicht zu Ende
            t_me = top[me]
            ev = (t_me & ~t_me.shift(1, fill_value=False)).reindex(idx, fill_value=False)
        else:
            raise ValueError(fam)
        ok = (ev & m["elig"]).to_numpy().copy()
        ok[idx < pd.Timestamp(SIGNAL_START)] = False
        out[name] = ok
    return out


def exits_for(p: np.ndarray, lows: dict, e: int) -> tuple[np.ndarray, np.ndarray]:
    """Ausstiegstag je Ausstiegsvariante fuer einen Einstieg am Schluss von Tag e; offen = Kursreihe endet vorher."""
    n = len(p)
    x = np.empty(len(EX_NAMES), dtype=np.int64)
    opn = np.zeros(len(EX_NAMES), dtype=bool)
    end = min(n - 1, e + MAX_HALTEDAUER)
    seg = p[e + 1:end + 1]
    run_max = np.maximum.accumulate(np.concatenate([[p[e]], seg]))[1:] if len(seg) else seg
    draw = seg / run_max - 1 if len(seg) else seg
    for k, name in enumerate(EX_NAMES):
        r = AUSSTIEGE[name]
        if r["typ"] == "zeit":
            t = e + r["tage"]
        elif r["typ"] == "kombi":
            hit = np.flatnonzero(draw[:r["tage"]] <= -r["pct"])
            t = e + 1 + hit[0] if len(hit) else e + r["tage"]
        else:
            if r["typ"] == "trend":
                hit = np.flatnonzero(seg < lows[r["tage"]][e + 1:end + 1])
                t = e + 1 + hit[0] + 1 if len(hit) else None    # Signal am Schluss, Verkauf am naechsten Schluss
            else:
                hit = np.flatnonzero(draw <= -r["pct"])
                t = e + 1 + hit[0] if len(hit) else None        # Stopp liegt im Markt: Verkauf am selben Tag
            if t is None:
                t = e + MAX_HALTEDAUER
        if t > n - 1:
            t, opn[k] = n - 1, True
        x[k] = t
    return x, opn


def all_trades(m: dict, sigs: dict) -> pd.DataFrame:
    """Jedes Signal mit dem Ergebnis aller 12 Ausstiege (ohne Portfolio-Grenzen)."""
    c = m["close"]
    pf = c.ffill()
    P = pf.to_numpy()
    lows = {n: pf.rolling(n).min().shift(1).to_numpy()
            for n in {r["tage"] for r in AUSSTIEGE.values() if r["typ"] == "trend"}}
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
    meta = pd.DataFrame(rows_meta, columns=["einstieg", "t_signal", "j", "e"]).astype(np.int64)
    if not len(meta):                                       # keine Signale (z. B. am ersten Protokolltag)
        k = len(EX_NAMES)
        meta.attrs.update(X=np.zeros((0, k), dtype=np.int64), OPEN=np.zeros((0, k), dtype=bool),
                          R=np.zeros((0, k)), D=np.ones((0, k)))
        return meta
    X, O = np.array(X), np.array(O)
    pe = P[meta["e"].to_numpy(), meta["j"].to_numpy()]
    px = P[X, meta["j"].to_numpy()[:, None]]
    meta.attrs.update(X=X, OPEN=O, R=px / pe[:, None] - 1, D=np.maximum(X - meta["e"].to_numpy()[:, None], 1))
    return meta


# ======================================================================================= Merkmale
def feature_panels(m: dict) -> tuple[dict, pd.DataFrame]:
    """Merkmale als Tafeln (Tag x Aktie) und Marktmerkmale (Tag) - einmal rechnen, beliebig oft abfragen."""
    c, elig = m["close"], m["elig"]
    pf = c.ffill()
    lr = np.log(pf).diff()
    ann = np.sqrt(252)
    F = {}
    for n in (5, 20, 60, 126, 252):
        F[f"rendite_{n}"] = pf / pf.shift(n) - 1
    v20, v60 = lr.rolling(20).std() * ann, lr.rolling(60).std() * ann
    F["schwankung_20"], F["schwankung_60"], F["schwankung_verh"] = v20, v60, v20 / v60
    F["abstand_hoch_252"] = pf / pf.rolling(252, min_periods=120).max() - 1
    sma50, sma200 = pf.rolling(50).mean(), pf.rolling(200).mean()
    F["ueber_linie_200"] = pf / sma200 - 1
    F["linie_50_zu_200"] = sma50 / sma200 - 1
    F["ausbruch_staerke_20"] = pf / pf.rolling(20).max().shift(1) - 1
    F["umsatz_log"] = np.log10(m["turnover"])
    vol = m["volume"].where(m["volume"] > 0)
    F["volumen_schub"] = np.log(vol.rolling(5, min_periods=3).mean() / vol.rolling(60, min_periods=30).median().shift(5))
    F["rang_momentum_126"] = (pf / pf.shift(126) - 1).where(elig).rank(axis=1, pct=True)
    # Gesamtmarkt: gleichgewichteter Index der handelbaren Aktien
    mret = lr.where(elig.shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ix = np.exp(mret.cumsum())
    M = pd.DataFrame(index=c.index)
    for n in (20, 60, 252):
        M[f"markt_rendite_{n}"] = ix / ix.shift(n) - 1
    M["markt_schwankung_20"] = mret.rolling(20).std() * ann
    M["markt_schwankung_60"] = mret.rolling(60).std() * ann
    M["markt_abstand_hoch"] = ix / ix.rolling(252, min_periods=60).max() - 1
    M["markt_breite_200"] = (pf > sma200).astype(float).where(elig & sma200.notna()).mean(axis=1)
    M["markt_ausbrueche_50"] = (pf > pf.rolling(50).max().shift(1)).astype(float).where(elig).mean(axis=1)
    M["markt_streuung_20"] = (pf / pf.shift(20) - 1).where(elig).std(axis=1)
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
    out["einstieg"] = meta["einstieg"].to_numpy()
    out["einstieg_tage"] = [EINSTIEGE[EN_NAMES[v]][1][1] if isinstance(EINSTIEGE[EN_NAMES[v]][1], tuple)
                            else EINSTIEGE[EN_NAMES[v]][1] for v in out["einstieg"]]
    out["placebo"] = np.random.default_rng(SEED).normal(size=len(out))
    return out.replace([np.inf, -np.inf], np.nan)


def stack(feat: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Je Signal eine Zeile pro Ausstieg (Merkmale des Ausstiegs angehaengt)."""
    k = len(EX_NAMES)
    f = feat.iloc[np.repeat(rows, k)].reset_index(drop=True)
    typ = {"zeit": 0, "trend": 1, "nachzug": 2, "kombi": 3}
    f["ausstieg_typ"] = np.tile([typ[AUSSTIEGE[x]["typ"]] for x in EX_NAMES], len(rows))
    f["ausstieg_tage"] = np.tile([AUSSTIEGE[x].get("tage", 0) for x in EX_NAMES], len(rows))
    f["ausstieg_stopp"] = np.tile([AUSSTIEGE[x].get("pct", 0) * 100 for x in EX_NAMES], len(rows))
    return f


def target(meta: pd.DataFrame) -> np.ndarray:
    """Rendite je Haltetag nach Kosten in Basispunkten (Signale x Ausstiege)."""
    return (meta.attrs["R"] - KOSTEN) / meta.attrs["D"] * 1e4


def model_stamp() -> dict:
    """Kennung der Daten und Einstellungen, zu denen gespeicherte Modelle passen."""
    import sklearn
    return {"kurse": (tm.SCREENER_CACHE / tm.STOCK_FILE).stat().st_mtime, "testjahr": ERSTES_TESTJAHR,
            "einstiege": EN_NAMES, "ausstiege": EX_NAMES, "ziele": ZIELE, "modell": MODELL,
            "sklearn": sklearn.__version__}


def save_models(models: dict) -> None:
    import pickle
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "modelle.pkl", "wb") as f:
        pickle.dump({"stamp": model_stamp(), "modelle": models}, f)


def load_models() -> dict | None:
    import pickle
    path = OUT_DIR / "modelle.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    return obj["modelle"] if obj.get("stamp") == model_stamp() else None


def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(categorical_features=["einstieg", "ausstieg_typ"], **MODELL)


# ======================================================================================= Walk-forward
def _demean(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return v - pd.Series(v).groupby(groups).transform("mean").to_numpy()


def walk_forward(m: dict, meta: pd.DataFrame, feat: pd.DataFrame, ziel: str = "absolut",
                 models_out: dict | None = None) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Vorhersage (Signale x Ausstiege) fuer alle Signale ab ERSTES_TESTJAHR, je Jahr nur mit Vorjahren trainiert.
    ziel 'relativ': Trainingsziel minus Mittel aller Trades desselben Signalmonats (ohne Markttiming)."""
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
    years = range(ERSTES_TESTJAHR, cal[-1].year + 1)
    for Y in years:
        y0 = pd.Timestamp(f"{Y}-01-01")
        done = (cal.values[X] < np.datetime64(y0)) & ~O    # Trade vor Jahresbeginn abgeschlossen
        tr_sig = np.flatnonzero(done.any(axis=1))
        te_sig = np.flatnonzero(sig_year == Y)
        if len(te_sig) == 0 or len(tr_sig) < 1000:
            continue
        Xtr = stack(feat, tr_sig)
        ytr = y_all[tr_sig].ravel()
        keep = done[tr_sig].ravel() & np.isfinite(ytr)
        idx = np.flatnonzero(keep)
        if len(idx) > MAX_TRAIN_ZEILEN:
            idx = np.sort(rng.choice(idx, MAX_TRAIN_ZEILEN, replace=False))
        lo, hi = np.percentile(ytr[idx], [1, 99])
        yfit = np.clip(ytr[idx], lo, hi)
        if ziel == "relativ":
            yfit = _demean(yfit, np.repeat(month[tr_sig], K)[idx])
        model = make_model()
        t0 = time.time()
        model.fit(Xtr.iloc[idx], yfit)
        if models_out is not None:
            models_out[Y] = model
        Xte = stack(feat, te_sig)
        p = model.predict(Xte).reshape(len(te_sig), len(EX_NAMES))
        pred[te_sig] = p
        # Trefferguete: Rangkorrelation Vorhersage / spaeteres Ergebnis (nur abgeschlossene Trades)
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
        ic_rows.append({"ziel": ziel, "jahr": Y, "trainings_signale": len(tr_sig), "test_signale": len(te_sig),
                        "rangkorrelation_alle": ic_all, "rangkorrelation_im_monat": ic_mon,
                        "rangkorrelation_gewaehlt": ic_sig})
        # Wichtigkeit der Merkmale im Testjahr (Permutation, Stichprobe)
        try:
            from sklearn.inspection import permutation_importance
            rows = np.flatnonzero(okte.ravel())
            if len(rows) > 2000:
                rows = rng.choice(rows, min(len(rows), 40_000), replace=False)
                ysc = np.clip(yte.ravel()[rows], lo, hi)
                if ziel == "relativ":
                    ysc = _demean(ysc, np.repeat(month[te_sig], K)[rows])
                pi = permutation_importance(model, Xte.iloc[rows], ysc,
                                            n_repeats=3, random_state=SEED, scoring="neg_mean_squared_error")
                for k, col in enumerate(Xte.columns):
                    imp_rows.append({"ziel": ziel, "jahr": Y, "merkmal": col, "wichtigkeit": pi.importances_mean[k]})
        except Exception as ex:                             # Wichtigkeit ist nur Zusatzinfo
            log(f"  Wichtigkeit {Y} nicht berechnet: {ex}")
        log(f"  [{ziel}] Testjahr {Y}: trainiert mit {len(idx):,} Zeilen ({time.time() - t0:.0f} s), "
            f"Rangkorrelation {ic_all:+.3f}, im Monat {ic_mon:+.3f}".replace(",", "."))
    return pred, pd.DataFrame(ic_rows), pd.DataFrame(imp_rows)


# ======================================================================================= Strategien
def to_trades(meta: pd.DataFrame, rows: np.ndarray, exits: np.ndarray, rang: np.ndarray) -> pd.DataFrame:
    """Format von trend_modell.simulate: j, t_signal, e, x, kurs_aus, rendite, tage, grund, rang."""
    X, R = meta.attrs["X"], meta.attrs["R"]
    return pd.DataFrame({"j": meta["j"].to_numpy()[rows], "t_signal": meta["t_signal"].to_numpy()[rows],
                         "e": meta["e"].to_numpy()[rows], "x": X[rows, exits], "kurs_aus": np.nan,
                         "rendite": R[rows, exits], "tage": X[rows, exits] - meta["e"].to_numpy()[rows],
                         "grund": [EX_NAMES[k] for k in exits], "rang": rang})


def one_per_day(meta: pd.DataFrame, rows: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mehrere Einstiege am selben Tag fuer dieselbe Aktie: nur der mit der hoechsten Bewertung."""
    d = pd.DataFrame({"r": rows, "s": score, "t": meta["t_signal"].to_numpy()[rows], "j": meta["j"].to_numpy()[rows]})
    d = d.sort_values("s", ascending=False).drop_duplicates(["t", "j"])
    return d["r"].to_numpy(), d["s"].to_numpy()


def summarize(name: str, sim: dict, a: pd.Timestamp) -> dict:
    out = {"strategie": name}
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
    log("Signale aller Einstiege ...")
    sigs = entry_signals(m)
    meta = all_trades(m, sigs)
    counts = meta.groupby("einstieg").size().rename(index=dict(enumerate(EN_NAMES)))
    log(f"{len(meta):,} Signale, je {len(EX_NAMES)} Ausstiege durchgerechnet".replace(",", "."))
    feat = features(m, meta)
    test_start = pd.Timestamp(f"{ERSTES_TESTJAHR}-01-01")
    sig_t = meta["t_signal"].to_numpy()
    in_test = cal[sig_t] >= test_start
    en = meta["einstieg"].to_numpy()

    def sim(tr: pd.DataFrame, start: pd.Timestamp = test_start) -> dict:
        return tm.simulate(m, tr, "long", start)

    # ---- 1. Raster aller festen Kombinationen (ueber den ganzen Zeitraum, causal) -> Auswahl je Vorjahre
    log(f"Raster: {len(EN_NAMES) * len(EX_NAMES)} feste Kombinationen im Portfolio ...")
    grid_eq, grid_rows = {}, []
    sig_start = pd.Timestamp(SIGNAL_START)
    for v, en_name in enumerate(EN_NAMES):
        rows = np.flatnonzero(en == v)
        for k, ex_name in enumerate(EX_NAMES):
            tr = to_trades(meta, rows, np.full(len(rows), k), mom[sig_t[rows], meta["j"].to_numpy()[rows]])
            eq = sim(tr, sig_start)["equity"]
            grid_eq[(en_name, ex_name)] = eq
            r = {"einstieg": en_name, "ausstieg": ex_name, "signale": len(rows)}
            for lbl, a, b in (("bis_2022", sig_start, pd.Timestamp("2022-12-31")),
                              ("ab_2023", pd.Timestamp("2023-01-01"), None),
                              ("gesamt", sig_start, None)):
                r[f"sharpe_{lbl}"] = tm._sharpe(eq, a, b)
            grid_rows.append(r)
    grid = pd.DataFrame(grid_rows)
    tm.save_csv(grid, "raster_sharpe.csv")
    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({f"{a} | {b}": eq for (a, b), eq in grid_eq.items()}).to_pickle(OUT_DIR / "raster_kurven.pkl")
    raster_chart(grid)

    # Walk-forward: je Testjahr die feste Kombination mit der besten Sharpe Ratio aller Vorjahre
    wf_rows, wf_choice = [], {}
    for Y in range(ERSTES_TESTJAHR, cal[-1].year + 1):
        best = max(grid_eq, key=lambda k: np.nan_to_num(tm._sharpe(grid_eq[k], sig_start, pd.Timestamp(f"{Y - 1}-12-31")),
                                                        nan=-9))
        wf_choice[Y] = best
        v, k = EN_NAMES.index(best[0]), EX_NAMES.index(best[1])
        rows = np.flatnonzero((en == v) & (cal[sig_t].year == Y))
        wf_rows.append(to_trades(meta, rows, np.full(len(rows), k), mom[sig_t[rows], meta["j"].to_numpy()[rows]]))
    wf_trades = pd.concat(wf_rows, ignore_index=True)

    # ---- 2. CPO walk-forward (beide Ziele)
    preds, ics, imps, models = {}, [], [], {}
    for label, ziel in ZIELE.items():
        log(f"CPO walk-forward, Ziel '{label}' ...")
        models[label] = {}
        preds[label], ic_, imp_ = walk_forward(m, meta, feat, ziel, models_out=models[label])
        ics.append(ic_)
        imps.append(imp_)
    save_models(models)                                     # fuer cpo_stresstest.py
    ic, imp = pd.concat(ics, ignore_index=True), pd.concat(imps, ignore_index=True)

    strategies: dict[str, pd.DataFrame] = {}
    vb, kb = EN_NAMES.index(BASIS[0]), EX_NAMES.index(BASIS[1])
    base_rows = np.flatnonzero((en == vb) & in_test)
    base_rank = mom[sig_t[base_rows], meta["j"].to_numpy()[base_rows]]
    strategies[f"Fest: {BASIS[0]} + {BASIS[1]} (Trendmodell)"] = to_trades(meta, base_rows, np.full(len(base_rows), kb), base_rank)
    kl = EX_NAMES.index(LANG[1])
    strategies[f"Fest: {LANG[0]} + {LANG[1]}"] = to_trades(meta, base_rows, np.full(len(base_rows), kl), base_rank)
    strategies["Walk-forward: beste feste Kombination der Vorjahre"] = wf_trades
    cpo_rows_by = {}
    for label, pred in preds.items():
        ok_pred = np.isfinite(pred).all(axis=1)
        rows = base_rows[ok_pred[base_rows]]
        strategies[f"CPO[{label}] Ausstieg (Einstieg {BASIS[0]})"] = to_trades(
            meta, rows, pred[rows].argmax(axis=1), mom[sig_t[rows], meta["j"].to_numpy()[rows]])
        allr = np.flatnonzero(in_test & ok_pred)
        r1, s1 = one_per_day(meta, allr, pred[allr, kb])
        sel = s1 > MIN_SCORE_BP
        strategies[f"CPO[{label}] Einstieg (Ausstieg {BASIS[1]})"] = to_trades(meta, r1[sel], np.full(sel.sum(), kb), s1[sel])
        r2, s2 = one_per_day(meta, allr, pred[allr].max(axis=1))
        sel2 = s2 > MIN_SCORE_BP
        cpo_rows_by[label] = r2[sel2]
        strategies[f"CPO[{label}] Einstieg + Ausstieg"] = to_trades(meta, r2[sel2], pred[r2[sel2]].argmax(axis=1), s2[sel2])

    # ---- 3. Zufall zum Vergleich
    rng = np.random.default_rng(SEED)
    rand_specs = {
        f"Zufall: Ausstieg zufaellig (Einstieg {BASIS[0]})":
            lambda: to_trades(meta, base_rows, rng.integers(0, len(EX_NAMES), len(base_rows)), base_rank),
    }
    for label, cr in cpo_rows_by.items():
        rand_specs[f"Zufall: gleich viele Signale wie CPO[{label}], zufaelliger Ausstieg"] = \
            (lambda cr=cr: random_like(meta, cal, np.flatnonzero(in_test), cr, rng))

    log("Portfolio-Rechnung der Strategien ...")
    rows_out, curves, year_tab, trades_out = [], {}, {}, {}
    for name, tr in strategies.items():
        s = sim(tr)
        rows_out.append(summarize(name, s, test_start))
        curves[name] = s["equity"]
        year_tab[name] = yearly(s["equity"])
        trades_out[name] = s["trades"]
    for name, fn in rand_specs.items():
        res = [summarize(name, sim(fn()), test_start) for _ in range(ZUFALL_LAEUFE)]
        d = pd.DataFrame(res)
        r = d.drop(columns="strategie").mean().to_dict()
        r["strategie"] = f"{name} (Mittel aus {ZUFALL_LAEUFE})"
        r["sharpe_spanne"] = f"{d['sharpe'].min():.2f} bis {d['sharpe'].max():.2f}"
        rows_out.append(r)
    if m["bench"] is not None:
        b = m["bench"][m["bench"].index >= test_start].ffill().dropna()
        beq = tm.CAPITAL * b / b.iloc[0]
        r = {"strategie": f"Kaufen und Halten {tm.BENCHMARK['aktien']}"}
        r.update(tm.metrics(beq, pd.DataFrame(), str(test_start.date()), None))
        rows_out.append(r)
        curves[r["strategie"]] = beq
        year_tab[r["strategie"]] = yearly(beq)
    comp = pd.DataFrame(rows_out)
    comp = comp[["strategie"] + [c for c in comp.columns if c != "strategie"]]
    tm.save_csv(comp, "vergleich.csv")
    years = pd.DataFrame(year_tab).T.round(1)
    tm.save_csv(years, "jahre.csv", index=True)
    tm.save_csv(ic, "trefferguete.csv")
    if not imp.empty:
        tm.save_csv(imp.pivot_table(index="merkmal", columns=["ziel", "jahr"], values="wichtigkeit")
                    .assign(mittel=lambda d: d.mean(axis=1)).sort_values("mittel", ascending=False), "merkmale.csv",
                    index=True)
    for name, tl in trades_out.items():
        if name.startswith("CPO") and tl is not None and not tl.empty:
            fn = "trades_" + name.split("(")[0].strip().lower()
            for a, b in ((" ", "_"), ("+", "und"), ("[", "_"), ("]", ""), ("__", "_"), ("__", "_")):
                fn = fn.replace(a, b)
            tm.save_csv(tl.assign(name=tl["ticker"].map(m["names"])), fn + ".csv")
    # Welche Ausstiege und Einstiege waehlt CPO je Jahr?
    chosen, ent = {}, {}
    for label in preds:
        for kind in ("Einstieg + Ausstieg", f"Ausstieg (Einstieg {BASIS[0]})"):
            ch = strategies[f"CPO[{label}] {kind}"]
            chosen[f"CPO[{label}] {kind}"] = pd.crosstab(cal[ch["t_signal"]].year, ch["grund"], normalize="index") \
                .reindex(columns=EX_NAMES, fill_value=0)
        ch = strategies[f"CPO[{label}] Einstieg + Ausstieg"]
        e_ = pd.crosstab(cal[ch["t_signal"]].year, en[cpo_rows_by[label]], normalize="index")
        e_.columns = [EN_NAMES[c] for c in e_.columns]
        ent[f"CPO[{label}] Einstieg + Ausstieg"] = e_
    tm.save_csv(pd.concat(chosen).mul(100).round(1), "gewaehlte_ausstiege.csv", index=True)
    tm.save_csv(pd.concat(ent).mul(100).round(1), "gewaehlte_einstiege.csv", index=True)
    chart(curves)
    report(comp, years, grid, wf_choice, ic, imp, chosen, ent, counts, time.time() - t_start)


def random_like(meta: pd.DataFrame, cal: pd.DatetimeIndex, pool: np.ndarray, chosen: np.ndarray, rng) -> pd.DataFrame:
    """Gleich viele Signale je Jahr wie CPO, zufaellig aus allen Einstiegen, zufaelliger Ausstieg und Rang."""
    yr_pool = cal[meta["t_signal"].to_numpy()[pool]].year
    yr_ch = pd.Series(cal[meta["t_signal"].to_numpy()[chosen]].year).value_counts()
    rows = []
    for y, n in yr_ch.items():
        cand = pool[yr_pool == y]
        rows.append(rng.choice(cand, min(n, len(cand)), replace=False))
    rows = np.concatenate(rows) if rows else np.array([], dtype=int)
    return to_trades(meta, rows, rng.integers(0, len(EX_NAMES), len(rows)), rng.random(len(rows)))


# ======================================================================================= Ausgabe
def raster_chart(grid: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5))
    for ax, col, title in ((axes[0], "sharpe_bis_2022", "Sharpe 2017-2022"), (axes[1], "sharpe_ab_2023", "Sharpe ab 2023")):
        pv = grid.pivot(index="einstieg", columns="ausstieg", values=col).reindex(index=EN_NAMES, columns=EX_NAMES)
        im = ax.imshow(pv.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(EX_NAMES)), EX_NAMES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(EN_NAMES)), EN_NAMES if ax is axes[0] else [""] * len(EN_NAMES), fontsize=8)
        for (i, k), val in np.ndenumerate(pv.to_numpy()):
            if np.isfinite(val):
                ax.text(k, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.7)
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "raster_sharpe.png", dpi=110, bbox_inches="tight")
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
        kw = {"color": "grey"} if name.startswith("Kaufen") else ({"lw": 2.2} if name.startswith("CPO") else {"ls": "--"})
        ax.plot(eq.index, eq.values / 1000, label=name, **kw)
    ax.set_yscale("log")
    ax.set_title(f"Walk-forward ab {ERSTES_TESTJAHR}: Konto in Tsd. EUR (log)")
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
    print(f"CPO TRENDFOLGE XETRA (long)  Walk-forward ab {ERSTES_TESTJAHR}, Konto {tm.CAPITAL:,.0f} EUR, "
          f"max. {tm.MAX_POSITIONS} Positionen".replace(",", "."))
    print(f"Kosten: {tm.broker_text()}, Slippage {tm.SLIPPAGE_BPS} bp je Order")
    print("=" * 120)
    print("\nSignale je Einstieg (2017-heute):")
    print(counts.to_string())
    cols = [c for c in ["strategie", "rendite_pa_pct", "sharpe", "sharpe_spanne", "max_drawdown_pct", "trades",
                        "trefferquote", "je_trade_pct", "haltedauer"] if c in comp.columns]
    print(f"\nVergleich im Testzeitraum ab {ERSTES_TESTJAHR} (alles ausserhalb der Stichprobe):")
    print(comp[cols].round(2).to_string(index=False))
    print("\nRendite je Kalenderjahr (%):")
    print(years.to_string())
    print("\nWalk-forward: jeweils beste feste Kombination der Vorjahre:")
    for y, (a, b) in wf_choice.items():
        print(f"  {y}: {a} + {b}")
    piv = grid.pivot(index="einstieg", columns="ausstieg", values="sharpe_bis_2022").reindex(index=EN_NAMES, columns=EX_NAMES)
    piv2 = grid.pivot(index="einstieg", columns="ausstieg", values="sharpe_ab_2023").reindex(index=EN_NAMES, columns=EX_NAMES)
    print("\nRaster fester Kombinationen - Sharpe 2017-2022:")
    print(piv.round(2).to_string())
    print("\nRaster fester Kombinationen - Sharpe ab 2023:")
    print(piv2.round(2).to_string())
    rc = pd.Series(piv.to_numpy().ravel()).corr(pd.Series(piv2.to_numpy().ravel()), method="spearman")
    print(f"Rangkorrelation der Kombinationen zwischen beiden Zeitraeumen: {rc:+.2f} "
          "(nahe 0 = die Rangfolge ist Zufall, nahe 1 = stabil)")
    if not ic.empty:
        print("\nTrefferguete der CPO-Vorhersage im jeweiligen Testjahr (Rangkorrelation Vorhersage / Ergebnis;")
        print("'im_monat' = nur Unterschiede zwischen gleichzeitigen Signalen, 'gewaehlt' = unter den gewaehlten):")
        print(ic.round(3).to_string(index=False))
    if not imp.empty:
        for z, g in imp.groupby("ziel"):
            mi = g.groupby("merkmal")["wichtigkeit"].mean().sort_values(ascending=False)
            pl = mi.get("placebo", np.nan)
            print(f"\nWichtigste Merkmale, Ziel '{z}' (Permutation, Mittel der Testjahre; 'placebo' = Zufallsmerkmal):")
            print(mi.head(10).round(2).to_string())
            print(f"  placebo: {pl:.3g} (Rang {list(mi.index).index('placebo') + 1} von {len(mi)})")
    for name, tab in chosen.items():
        print(f"\nGewaehlte Ausstiege je Jahr (% der Signale), {name}:")
        print(tab.mul(100).round(0).astype(int).to_string())
    for name, tab in ent.items():
        print(f"\nGewaehlte Einstiege je Jahr (%), {name}:")
        print(tab.mul(100).round(0).astype(int).to_string())
    print(f"\nDateien: {OUT_DIR}   (Laufzeit {secs / 60:.1f} min)")
    print("Lesart: CPO lohnt sich nur, wenn es die feste Variante, die jaehrlich neu gewaehlte beste feste Kombination")
    print("UND die Zufallsvarianten schlaegt, die Rangkorrelation in den meisten Jahren > 0 ist und 'placebo' weit")
    print("hinten liegt.")


# ======================================================================================= Heute
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
    if len(keep) > MAX_TRAIN_ZEILEN:
        keep = np.sort(rng.choice(keep, MAX_TRAIN_ZEILEN, replace=False))
    lo, hi = np.percentile(ytr[keep], [1, 99])
    model = make_model().fit(Xtr.iloc[keep], np.clip(ytr[keep], lo, hi))
    # Signale am letzten Handelstag (Einstieg zum naechsten Schlusskurs) - all_trades verwirft sie, weil der
    # Einstiegstag noch fehlt; deshalb hier direkt aus den Signalen
    t_last = len(cal) - 1
    cand = [(v, j) for v, name in enumerate(EN_NAMES) for j in np.flatnonzero(sigs[name][t_last])]
    print("\n" + "=" * 100)
    print(f"CPO-SIGNALE vom {cal[-1]:%d.%m.%Y} (Kauf zum naechsten Schlusskurs)")
    print("=" * 100)
    if not cand:
        print("Heute kein Einstiegssignal.")
        return
    mt = pd.DataFrame({"einstieg": [v for v, _ in cand], "t_signal": t_last, "j": [j for _, j in cand], "e": t_last})
    ft = features(m, mt)
    p = model.predict(stack(ft, np.arange(len(mt)))).reshape(len(mt), len(EX_NAMES))
    out = pd.DataFrame({"ticker": m["close"].columns[mt["j"]], "name": m["names"].reindex(m["close"].columns[mt["j"]]).values,
                        "einstieg": [EN_NAMES[v] for v in mt["einstieg"]],
                        "bester_ausstieg": [EX_NAMES[k] for k in p.argmax(axis=1)],
                        "erwartet_bp_je_tag": p.max(axis=1)})
    out = out.sort_values("erwartet_bp_je_tag", ascending=False).drop_duplicates("ticker")
    out["kaufen"] = np.where(out["erwartet_bp_je_tag"] > MIN_SCORE_BP, "ja", "nein")
    print(out.round(1).head(25).to_string(index=False))
    tm.save_csv(out, "heute.csv")
    print(f"\nModell trainiert mit allen abgeschlossenen Trades bis {cal[-1]:%d.%m.%Y}. Hoechstens "
          f"{tm.MAX_POSITIONS} Positionen; bei mehr Kandidaten die mit der hoechsten Erwartung.")


def main() -> None:
    ap = argparse.ArgumentParser(description="CPO fuer das Trendfolge-Modell (Xetra-Aktien, long)")
    ap.add_argument("--heute", action="store_true", help="heutige Signale mit vom Modell gewaehltem Ausstieg")
    args = ap.parse_args()
    tm.OUT_DIR = OUT_DIR
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("scikit-learn fehlt:  ..\\.venv\\python.exe -m pip install scikit-learn")
    if args.heute:
        today()
    else:
        run_backtest()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
