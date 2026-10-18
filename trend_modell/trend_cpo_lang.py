"""
CPO-Test mit langer Kurshistorie, vorsichtiger Auswahl und CPO als laufendem Ausstiegssignal
================================================================================================================

Drei Fragen in einem Test (fuer Robust 1 und Robust 2 aus der Bewertung):

1. Lange Historie: Kurse ab 2000 statt ab 2016 -> Training und Test ueber viele Marktphasen (Dotcom-Crash,
   Finanzkrise 2008, Euro-Krise 2011, 2015/16, 2018, Corona, 2022). Mit --download werden die Kurse aller
   Aktien des Screener-Caches ab 2000 per yfinance geladen (einmalig), --lang rechnet damit.
   Achtung: Es sind nur heute noch gehandelte Aktien (Survivorship Bias). Ueber 25 Jahre wiegt das schwerer
   als ueber 9 Jahre - deshalb ist der faire Vergleich der gleichgewichtete Index derselben Aktien.

2. Vorsichtige Auswahl beim Kauf: statt der besten von 18 Ausstiegsregeln nur 3 Kandidaten; mehrere Modelle
   (ENSEMBLE) liefern Mittelwert und Streuung der Vorhersage. Von der Standardregel wird nur abgewichen, wenn
   die vorsichtige Schaetzung (Mittel - LAMBDA x Streuung) um mindestens MIN_VORTEIL_BP besser ist.

3. CPO als laufendes Ausstiegssignal: Fuer jede offene Position sagt ein Modell jeden Tag die Rendite der
   naechsten HORIZONT Tage voraus (Merkmale der Aktie, des Markts und der Position: Tage gehalten, Gewinn seit
   Kauf, Abstand vom Hoch seit Kauf). Verkauft wird (am naechsten Schlusskurs), sobald selbst die optimistische
   Schaetzung (Mittel + LAMBDA_AUS x Streuung) unter SCHWELLE_BP liegt - also nur bei klarer Erwartung eines
   Kursrueckgangs. Zwei Formen: als Fruehwarnung zusaetzlich zur Standardregel (Verkauf beim frueheren der
   beiden Zeitpunkte) und als alleiniger Ausstieg (spaetestens nach MAX_HALTE Tagen).
   Kontrolle: zufaellige Fruehwarnung mit gleicher Haeufigkeit.

Bewaehrungsprobe (Varianten "mit Bewaehrung im Depot"): CPO darf in einem Jahr nur eingreifen, wenn ein Depot
mit seinen Entscheidungen in den Vorjahren - ausserhalb der Trainingsdaten - besser lief als das Depot mit der
Standardregel (t-Wert der taeglichen Differenz > BEWAEHRUNG_T). Die Einigkeit mehrerer Modelle allein ist kein
Vorsichtsmass: Sie lernen aus denselben Daten dieselben Fehler.

Walk-forward: Modelle fuer ein Signaljahr werden nur mit Trades/Tagen trainiert, die vor Jahresbeginn
abgeschlossen waren; jede Position behaelt das Modell ihres Kaufjahres.

Aufruf (im Ordner trend_modell):
    ..\\.venv\\python.exe trend_cpo_lang.py --download      Kurse ab 2000 laden (einmalig, ca. 10-20 Minuten)
    ..\\.venv\\python.exe trend_cpo_lang.py --lang          Test mit langer Historie (Testjahre ab 2006)
    ..\\.venv\\python.exe trend_cpo_lang.py                 derselbe Test mit dem bisherigen Cache (ab 2020)
Ergebnisse in ergebnisse_cpo_lang/ bzw. ergebnisse_cpo_kurz/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import trend_cpo as tc
import trend_modell as tm

# ======================================================================================= Einstellungen
LANG_FILE = "prices_aktien_lang.pkl"
DOWNLOAD_START = "2000-01-01"
DOWNLOAD_CHUNK = 40
LANG_SIGNAL_START = "2001-06-01"
LANG_ERSTES_TESTJAHR = 2006      # Training zuerst mit 2001-2005 (Dotcom-Crash und Erholung)
BENCH_TICKER = ["EXS1.DE", "^GDAXI"]

STRATEGIEN = {
    "Robust 1: Hoch 200 | Zeit 20": ("Hoch 200", "Zeit 20"),
    "Robust 2: Hoch 50 + Volumenschub | Zeit 60 + Stopp 15%": ("Hoch 50 + Volumenschub", "Zeit 60 + Stopp 15%"),
}
KANDIDATEN = ["Zeit 20", "Zeit 60 + Stopp 15%", "Zeit 120"]
ENSEMBLE = 3                     # Modelle je Testjahr (je auf 70 % der Trainingszeilen)
ENSEMBLE_ANTEIL = 0.7
LAMBDA = 1.0                     # vorsichtige Schaetzung = Mittel - LAMBDA x Streuung der Modelle
MIN_VORTEIL_BP = 1.0             # ... muss die Standardregel um so viele Basispunkte je Tag schlagen
BEWAEHRUNG_T = 1.0               # Bewaehrungsprobe: CPO darf in einem Jahr nur eingreifen, wenn sein Depot in den
BEWAEHRUNG_MIN = 30              #   Vorjahren besser lief (t-Wert > 1, mindestens 30 Trades); sonst Standardregel

HORIZONT = 20                    # Ausstiegssignal: vorhergesagte Rendite der naechsten 20 Handelstage
MIN_HALTE = 5                    # fruehestens nach 5 Tagen pruefen
MAX_HALTE = 120                  # spaetestens nach 120 Tagen verkaufen
SCHRITT_TRAINING = 5             # Trainingszeilen: jeder 5. Tag einer Position
LAMBDA_AUS = 1.0                 # verkaufen, wenn Mittel + LAMBDA_AUS x Streuung < SCHWELLE_BP
SCHWELLE_BP = 0.0                # Basispunkte je Tag

MAX_TRAIN = 250_000
MODELL = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0)
ZUFALL_LAEUFE = 20
SEED = 17


def log(msg: str) -> None:
    tm.log(msg)


# ======================================================================================= Daten
def download_long() -> None:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance fehlt:  ..\\.venv\\python.exe -m pip install yfinance")
    src = tm.SCREENER_CACHE / tm.STOCK_FILE
    if not src.exists():
        raise SystemExit(f"Kurs-Cache fehlt: {src}")
    tickers = list(pd.read_pickle(src)["close"].columns)
    closes, vols = [], []

    def get(chunk):
        for attempt in range(4):
            try:
                return yf.download(chunk, start=DOWNLOAD_START, auto_adjust=True, progress=False, threads=True,
                                   group_by="column")
            except Exception as e:
                wait = 15 * (attempt + 1)
                log(f"  Download-Fehler ({type(e).__name__}), neuer Versuch in {wait}s")
                time.sleep(wait)
        return None

    log(f"Lade Kurse ab {DOWNLOAD_START} fuer {len(tickers)} Aktien ...")
    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i:i + DOWNLOAD_CHUNK]
        data = get(chunk)
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            closes.append(data["Close"])
            vols.append(data["Volume"])
        else:
            closes.append(data[["Close"]].set_axis(chunk[:1], axis=1))
            vols.append(data[["Volume"]].set_axis(chunk[:1], axis=1))
        log(f"  {min(i + DOWNLOAD_CHUNK, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    close = pd.concat(closes, axis=1)
    volume = pd.concat(vols, axis=1)
    bench = get(BENCH_TICKER)
    bench = bench["Close"] if bench is not None and not bench.empty else pd.DataFrame()
    for df in (close, volume, bench):
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    first = close.apply(lambda s: s.first_valid_index())
    log(f"Kurse vorhanden: {close.notna().any().sum()} Aktien; schon 2001 gehandelt: "
        f"{(first < pd.Timestamp('2001-06-01')).sum()}, 2006: {(first < pd.Timestamp('2006-01-01')).sum()}, "
        f"2011: {(first < pd.Timestamp('2011-01-01')).sum()}")
    pd.to_pickle({"close": close, "volume": volume, "bench": bench}, tm.SCREENER_CACHE / LANG_FILE)
    log(f"Gespeichert: {tm.SCREENER_CACHE / LANG_FILE}")


def load(long: bool) -> dict:
    if long:
        f = tm.SCREENER_CACHE / LANG_FILE
        if not f.exists():
            raise SystemExit("Lange Kurshistorie fehlt - zuerst:  ..\\.venv\\python.exe trend_cpo_lang.py --download")
        tm.STOCK_FILE = LANG_FILE
        tc.SIGNAL_START = LANG_SIGNAL_START
        tc.ERSTES_TESTJAHR = LANG_ERSTES_TESTJAHR
    m = tc.load_data()
    if long:
        raw = pd.read_pickle(tm.SCREENER_CACHE / LANG_FILE)
        b = raw.get("bench", pd.DataFrame())
        m["bench"] = None
        for t in BENCH_TICKER:
            if t in getattr(b, "columns", []) and b[t].notna().sum() > 500:
                m["bench"] = b[t].reindex(m["close"].index).ffill()
                m["bench_name"] = t
                break
        el = m["elig"].groupby(m["elig"].index.year).any().sum(axis=1)
        vol_ok = m["volume"].notna().groupby(m["volume"].index.year).mean().mean(axis=1) * 100
        log("Handelbare Aktien je Jahr (Umsatz >= 1 Mio. EUR): " + ", ".join(f"{y}: {n}" for y, n in el.items()))
        log("Anteil der Kurstage mit Umsatzangabe je Jahr (%): "
            + ", ".join(f"{y}: {v:.0f}" for y, v in vol_ok.items()))
        early = el.loc[(el.index > el.index.min()) & (el.index <= LANG_ERSTES_TESTJAHR)]   # erstes Jahr: kein Vorlauf
        if len(early) and early.min() < 30:
            log("WARNUNG: In den ersten Jahren sind kaum Aktien handelbar (fehlende Umsaetze bei Yahoo?) - "
                "die fruehen Jahre sind dann wenig aussagekraeftig.")
    else:
        m["bench_name"] = tm.BENCHMARK["aktien"]
    return m


# ======================================================================================= Modelle
def fit_ensemble(X: pd.DataFrame, y: np.ndarray, rng) -> list:
    from sklearn.ensemble import HistGradientBoostingRegressor
    idx = np.flatnonzero(np.isfinite(y))
    if len(idx) > MAX_TRAIN:
        idx = rng.choice(idx, MAX_TRAIN, replace=False)
    lo, hi = np.percentile(y[idx], [1, 99])
    cat = [c for c in ("ausstieg_typ",) if c in X.columns]
    models = []
    for k in range(ENSEMBLE):
        sub = rng.choice(idx, int(len(idx) * ENSEMBLE_ANTEIL), replace=False)
        mdl = HistGradientBoostingRegressor(categorical_features=cat or None, random_state=SEED + k, **MODELL)
        mdl.fit(X.iloc[sub], np.clip(y[sub], lo, hi))
        models.append(mdl)
    return models


def predict(models: list, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = np.stack([mdl.predict(X) for mdl in models])
    return p.mean(axis=0), p.std(axis=0)


def stack_cand(feat: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    k = len(KANDIDATEN)
    f = feat.iloc[np.repeat(rows, k)].reset_index(drop=True)
    typ = {"zeit": 0, "trend": 1, "nachzug": 2, "kombi": 3}
    f["ausstieg_typ"] = np.tile([typ[tc.AUSSTIEGE[x]["typ"]] for x in KANDIDATEN], len(rows))
    f["ausstieg_tage"] = np.tile([tc.AUSSTIEGE[x].get("tage", 0) for x in KANDIDATEN], len(rows))
    f["ausstieg_stopp"] = np.tile([tc.AUSSTIEGE[x].get("pct", 0) * 100 for x in KANDIDATEN], len(rows))
    return f


# ======================================================================================= Teil 2: vorsichtige Wahl
def entry_choice(m: dict, meta: pd.DataFrame, feat: pd.DataFrame, default: str, rng) -> dict:
    cal = m["close"].index
    cand = [tc.EX_NAMES.index(x) for x in KANDIDATEN]
    d0 = KANDIDATEN.index(default)
    X, O = meta.attrs["X"][:, cand], meta.attrs["OPEN"][:, cand]
    y = (meta.attrs["R"][:, cand] - tc.KOSTEN) / meta.attrs["D"][:, cand] * 1e4
    sig_year = cal[meta["t_signal"].to_numpy()].year
    n = len(meta)
    mean = np.full((n, len(cand)), np.nan)
    std = np.full((n, len(cand)), np.nan)
    ic = []
    for Y in range(tc.ERSTES_TESTJAHR, cal[-1].year + 1):
        y0 = np.datetime64(pd.Timestamp(f"{Y}-01-01"))
        done = (cal.values[X] < y0) & ~O
        tr = np.flatnonzero(done.all(axis=1))
        te = np.flatnonzero(sig_year == Y)
        if len(te) == 0 or len(tr) < 300:
            continue
        models = fit_ensemble(stack_cand(feat, tr), y[tr].ravel(), rng)
        mu, sd = predict(models, stack_cand(feat, te))
        mean[te], std[te] = mu.reshape(len(te), -1), sd.reshape(len(te), -1)
        ok = ~O[te]
        ic.append({"jahr": Y, "trainings_signale": len(tr), "test_signale": len(te),
                   "rangkorrelation": pd.Series(mean[te][ok]).corr(pd.Series(y[te][ok]), method="spearman")})
    has = np.isfinite(mean).all(axis=1)
    greedy = np.where(has, np.nanargmax(np.nan_to_num(mean, nan=-1e9), axis=1), d0)
    cons = mean - LAMBDA * std
    best = np.nanargmax(np.nan_to_num(cons, nan=-1e9), axis=1)
    gain = cons[np.arange(n), best] - mean[:, d0]
    careful = np.where(has & (gain > MIN_VORTEIL_BP), best, d0)
    return {"greedy": np.array(cand)[greedy], "careful": np.array(cand)[careful], "has": has,
            "ic": pd.DataFrame(ic)}


# ======================================================================================= Teil 3: Tagesausstieg
def position_days(meta: pd.DataFrame, sig: np.ndarray, n: int, last: np.ndarray, step: int,
                  with_target: bool) -> tuple[np.ndarray, np.ndarray]:
    """(Signal-Index, Tag) fuer alle Pruef-Tage der Positionen."""
    E, J = meta["e"].to_numpy(), meta["j"].to_numpy()
    ii, dd = [], []
    for i in sig:
        end = min(E[i] + MAX_HALTE - 1, last[J[i]], n - 1)
        if with_target:
            end = min(end, last[J[i]] - 1 - HORIZONT, n - 2 - HORIZONT)
        ds = np.arange(E[i] + MIN_HALTE, end + 1, step)
        ii.append(np.full(len(ds), i))
        dd.append(ds)
    if not ii:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(ii), np.concatenate(dd)


def day_features(panels, meta: pd.DataFrame, C: np.ndarray, runmax_fn, ii: np.ndarray, dd: np.ndarray) -> pd.DataFrame:
    J, E = meta["j"].to_numpy()[ii], meta["e"].to_numpy()[ii]
    X = tc.sample_features(panels, dd, J)
    X["tage_gehalten"] = dd - E
    X["seit_kauf"] = C[dd, J] / C[E, J] - 1
    X["vom_hoch_seit_kauf"] = C[dd, J] / runmax_fn(ii, dd) - 1
    return X.replace([np.inf, -np.inf], np.nan)


def daily_exit(m: dict, meta: pd.DataFrame, panels, rng) -> dict:
    cal = m["close"].index
    n = len(cal)
    C = m["close"].ffill().to_numpy()
    has = m["close"].notna().to_numpy()
    last = n - 1 - has[::-1].argmax(axis=0)
    E, J = meta["e"].to_numpy(), meta["j"].to_numpy()
    # hoechster Schluss seit Kauf: je Signal einmal berechnen
    path_max = {}

    def runmax(ii, dd):
        out = np.empty(len(ii))
        for k, (i, d) in enumerate(zip(ii, dd)):
            pm = path_max.get(i)
            if pm is None:
                seg = C[E[i]:min(E[i] + MAX_HALTE + 1, n), J[i]]
                pm = path_max[i] = np.maximum.accumulate(seg)
            out[k] = pm[d - E[i]]
        return out

    sig_year = cal[meta["t_signal"].to_numpy()].year
    exit_x = np.full(len(meta), -1)
    exit_cpo = np.zeros(len(meta), dtype=bool)
    ic, checked_days = [], 0
    tr_i, tr_d = position_days(meta, np.arange(len(meta)), n, last, SCHRITT_TRAINING, True)
    tr_end = cal.values[np.minimum(tr_d + 1 + HORIZONT, n - 1)]
    y_all = (C[tr_d + 1 + HORIZONT, J[tr_i]] / C[tr_d + 1, J[tr_i]] - 1) / HORIZONT * 1e4
    X_all = day_features(panels, meta, C, runmax, tr_i, tr_d)
    for Y in range(tc.ERSTES_TESTJAHR, cal[-1].year + 1):
        te = np.flatnonzero(sig_year == Y)
        if len(te) == 0:
            continue
        sel = tr_end < np.datetime64(pd.Timestamp(f"{Y}-01-01"))
        if sel.sum() < 1000:
            continue
        models = fit_ensemble(X_all[sel].reset_index(drop=True), y_all[sel], rng)
        ii, dd = position_days(meta, te, n, last, 1, False)
        mu, sd = predict(models, day_features(panels, meta, C, runmax, ii, dd))
        sell = (mu + LAMBDA_AUS * sd) < SCHWELLE_BP
        checked_days += len(ii)
        # erster Verkaufstag je Signal
        df = pd.DataFrame({"i": ii, "d": dd, "sell": sell})
        first = df[df["sell"]].groupby("i")["d"].min()
        exit_x[first.index.to_numpy()] = first.to_numpy() + 1
        exit_cpo[first.index.to_numpy()] = True
        # Trefferguete: Vorhersage gegen spaetere Rendite (wo bekannt)
        ok = (dd + 1 + HORIZONT <= np.minimum(last[J[ii]], n - 1))
        if ok.sum() > 100:
            real = C[dd[ok] + 1 + HORIZONT, J[ii[ok]]] / C[dd[ok] + 1, J[ii[ok]]] - 1
            ic.append({"jahr": Y, "trainings_zeilen": int(sel.sum()), "pruef_tage": len(ii),
                       "verkaufssignale_pct": sell.mean() * 100,
                       "rangkorrelation": pd.Series(mu[ok]).corr(pd.Series(real), method="spearman"),
                       "rendite_nach_verkaufssignal_bp": real[sell[ok]].mean() * 1e4 if sell[ok].any() else np.nan,
                       "rendite_sonst_bp": real[~sell[ok]].mean() * 1e4})
        log(f"    Tagesausstieg {Y}: {len(te)} Positionen, {int(sel.sum()):,} Trainingszeilen".replace(",", "."))
    return {"x": exit_x, "cpo": exit_cpo, "ic": pd.DataFrame(ic), "last": last}


# ======================================================================================= Trades und Depot
def bewaehrung(m: dict, rows: np.ndarray, sig_year: np.ndarray, tr_std: pd.DataFrame, tr_alt: pd.DataFrame,
               test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bewaehrungsprobe im Depot: Fuer jedes Jahr Y werden Standard- und CPO-Trades der Testjahre VOR Y im Depot
    gerechnet (Kontostand nur bis Ende Y-1). CPO darf in Y nur eingreifen, wenn sein Depot taeglich im Mittel
    besser war als das Standard-Depot mit t-Wert > BEWAEHRUNG_T. Ergebnis: Trades mit Eingriff nur in solchen
    Jahren und eine Tabelle der Bewaehrung je Jahr."""
    cal = m["close"].index
    ys = sig_year[rows]
    out, rows_t = [], []
    for Y in range(tc.ERSTES_TESTJAHR, cal[-1].year + 1):
        y0 = pd.Timestamp(f"{Y}-01-01")
        past = ys < Y
        t = np.nan
        if past.sum() >= BEWAEHRUNG_MIN and y0 > test_start + pd.Timedelta(days=200):
            eqs = []
            for tr in (tr_std, tr_alt):
                eq = tm.simulate(m, tr[past], "long", test_start, with_log=False)["equity"]
                eqs.append(eq[eq.index < y0].pct_change().dropna())
            diff = (eqs[1] - eqs[0]).dropna()
            if len(diff) > 60 and diff.std() > 0:
                t = diff.mean() / diff.std() * np.sqrt(len(diff))
        ok = bool(np.isfinite(t) and t > BEWAEHRUNG_T)
        rows_t.append({"jahr": Y, "bewaehrung_t_wert": t, "cpo_darf_eingreifen": ok})
        cur = ys == Y
        out.append((tr_alt if ok else tr_std)[cur])
    return pd.concat(out).sort_index(), pd.DataFrame(rows_t)


def trades_early(m, meta, rows, x_cpo, tr_std: pd.DataFrame) -> pd.DataFrame:
    """Fruehwarnung: Verkauf beim frueheren Zeitpunkt - Standardregel oder CPO-Verkaufssignal."""
    C = m["close"].ffill().to_numpy()
    J, E = meta["j"].to_numpy()[rows], meta["e"].to_numpy()[rows]
    xs = tr_std["x"].to_numpy().copy()
    xc = x_cpo[rows]
    use = (xc > 0) & (xc < xs)
    xs[use] = xc[use]
    out = tr_std.copy()
    out["x"] = xs
    out["kurs_aus"] = np.where(use, np.nan, out["kurs_aus"])
    out["rendite"] = np.where(use, C[xs, J] / C[E, J] - 1, out["rendite"])
    out["tage"] = xs - E
    out["grund"] = np.where(use, "CPO", out["grund"])
    return out


def trades_fixed(meta, rows, k, rank) -> pd.DataFrame:
    return tc.to_trades(meta, rows, np.full(len(rows), k), rank)


def trades_daily(m, meta, rows, x_cpo, rank, stop: float | None, last) -> pd.DataFrame:
    """Ausstieg am Tag x_cpo (CPO), sonst nach MAX_HALTE; optional zusaetzlicher Nachzugsstopp."""
    n = len(m["close"])
    C = m["close"].ffill().to_numpy()
    E, J = meta["e"].to_numpy(), meta["j"].to_numpy()
    xs, why = [], []
    for i in rows:
        cap = min(E[i] + MAX_HALTE, n - 1, last[J[i]])
        x = x_cpo[i] if x_cpo[i] > 0 else cap
        w = "CPO" if x_cpo[i] > 0 else "Zeit"
        if stop:
            seg = C[E[i] + 1:cap + 1, J[i]]
            if len(seg):
                runm = np.maximum.accumulate(np.concatenate([[C[E[i], J[i]]], seg]))[1:]
                hit = np.flatnonzero(seg / runm - 1 <= -stop)
                if len(hit) and E[i] + 1 + hit[0] < x:
                    x, w = E[i] + 1 + hit[0], "Stopp"
        xs.append(min(x, cap))
        why.append(w)
    xs = np.array(xs, dtype=int)
    pe = C[E[rows], J[rows]]
    return pd.DataFrame({"j": J[rows], "t_signal": meta["t_signal"].to_numpy()[rows], "e": E[rows], "x": xs,
                         "kurs_aus": np.nan, "rendite": C[xs, J[rows]] / pe - 1, "tage": xs - E[rows],
                         "grund": why, "rang": rank})


def stats(eq: pd.Series) -> dict:
    s = eq.dropna()
    r = s.pct_change().dropna()
    y = len(r) / 252
    return {"rendite_pa_pct": ((s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1) * 100,
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "max_drawdown_pct": (1 - s / s.cummax()).max() * 100}


def evaluate(m, tr: pd.DataFrame, starts: list) -> tuple[dict, pd.Series]:
    res, eq0 = [], None
    for k, s in enumerate(starts):
        sim = tm.simulate(m, tr[tr["e"] >= np.searchsorted(m["close"].index, s)], "long", s, with_log=False)
        res.append(stats(sim["equity"]))
        if k == 0:
            eq0 = sim["equity"]
    d = pd.DataFrame(res)
    out = {f"ab_test_{c}": d.iloc[0][c] for c in d.columns}
    out.update({f"starts_{c}_mittel": d[c].mean() for c in d.columns})
    out.update({"trades": len(tr), "haltedauer": tr["tage"].mean()})
    return out, eq0


def yearly(eq: pd.Series) -> pd.Series:
    e = eq.dropna()
    ye = e.groupby(e.index.year).last()
    return (ye / ye.shift(1).fillna(e.iloc[0]) - 1) * 100


# ======================================================================================= Ablauf
def main() -> None:
    ap = argparse.ArgumentParser(description="CPO-Test: lange Historie, vorsichtige Wahl, Tagesausstieg")
    ap.add_argument("--download", action="store_true", help="Kurse ab 2000 laden")
    ap.add_argument("--lang", action="store_true", help="mit der langen Kurshistorie rechnen")
    args = ap.parse_args()
    if args.download:
        download_long()
        return
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("scikit-learn fehlt:  ..\\.venv\\python.exe -m pip install scikit-learn")
    out_dir = tm.BASE_DIR / ("ergebnisse_cpo_lang" if args.lang else "ergebnisse_cpo_kurz")
    tm.OUT_DIR = out_dir
    t0 = time.time()
    m = load(args.lang)
    cal = m["close"].index
    test_start = pd.Timestamp(f"{tc.ERSTES_TESTJAHR}-01-01")
    step = "2YS" if args.lang else "6MS"
    starts = [test_start] + [s for s in pd.date_range(test_start, cal[-1] - pd.Timedelta(days=3 * 365), freq=step)
                             if s > test_start]
    log(f"Kurse {cal[0]:%Y}-{cal[-1]:%Y}, Testjahre ab {tc.ERSTES_TESTJAHR}, {len(starts)} Startzeitpunkte")
    P = m["close"].ffill().to_numpy()
    mom = P / np.vstack([np.full((126, P.shape[1]), np.nan), P[:-126]]) - 1
    panels = tc.feature_panels(m)
    rng = np.random.default_rng(SEED)
    rows_out, curves, ics, yearly_tab = [], {}, [], {}

    # Vergleichsmassstab: gleichgewichteter Index derselben Aktien, DAX
    pf = m["close"].ffill()
    ew_r = pf.pct_change().where(m["elig"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    ew = tm.CAPITAL * (1 + ew_r[ew_r.index >= test_start]).cumprod()
    bench = {"Gleichgewichteter Index (dieselben Aktien)": ew}
    if m.get("bench") is not None and m["bench"][m["bench"].index >= test_start].notna().sum() > 250:
        b = m["bench"][m["bench"].index >= test_start].ffill().dropna()
        bench[f"DAX ({m.get('bench_name')})"] = tm.CAPITAL * b / b.iloc[0]
    for nm, eq in bench.items():
        r = {"strategie": "-", "variante": nm}
        r.update({f"ab_test_{k}": v for k, v in stats(eq).items()})
        rows_out.append(r)
        yearly_tab[nm] = yearly(eq)

    for sname, (entry, default) in STRATEGIEN.items():
        log(f"{sname}")
        meta = tc.all_trades(m, tc.entry_signals(m, [entry]))
        st, jj = meta["t_signal"].to_numpy(), meta["j"].to_numpy()
        rows = np.flatnonzero(cal[st] >= test_start)
        rank = mom[st[rows], jj[rows]]
        feat = tc.features(m, meta, panels)
        log(f"  {len(meta):,} Signale, davon {len(rows):,} im Testzeitraum".replace(",", "."))

        log("  Teil 2: vorsichtige Wahl beim Kauf ...")
        ch = entry_choice(m, meta, feat, default, rng)
        ics.append(ch["ic"].assign(strategie=sname, modell="Wahl beim Kauf"))
        log("  Teil 3: Tagesausstieg ...")
        dx = daily_exit(m, meta, panels, rng)
        ics.append(dx["ic"].assign(strategie=sname, modell="Tagesausstieg"))

        sig_year = cal[st].year
        k_std = tc.EX_NAMES.index(default)
        tr_std = trades_fixed(meta, rows, k_std, rank)
        variants = {}
        for ex in KANDIDATEN:
            variants[f"Fest: {ex}" + (" (Standard)" if ex == default else "")] = \
                tr_std if ex == default else trades_fixed(meta, rows, tc.EX_NAMES.index(ex), rank)
        variants["CPO-Wahl unvorsichtig (beste von 3)"] = tc.to_trades(meta, rows, ch["greedy"][rows], rank)
        tr_careful = tc.to_trades(meta, rows, ch["careful"][rows], rank)
        variants["CPO-Wahl vorsichtig (Modelle einig)"] = tr_careful
        variants["CPO-Wahl mit Bewaehrung im Depot"], bw1 = bewaehrung(m, rows, sig_year, tr_std, tr_careful, test_start)
        tr_early = trades_early(m, meta, rows, dx["x"], tr_std)
        variants["CPO-Fruehwarnung zusaetzlich zur Standardregel"] = tr_early
        variants["CPO-Fruehwarnung mit Bewaehrung im Depot"], bw2 = bewaehrung(m, rows, sig_year, tr_std, tr_early, test_start)
        variants[f"CPO-Tagesausstieg statt fester Regel (max. {MAX_HALTE} Tage)"] = \
            trades_daily(m, meta, rows, dx["x"], rank, None, dx["last"])
        bw = bw1.rename(columns={"bewaehrung_t_wert": "wahl_t", "cpo_darf_eingreifen": "wahl_darf"}).merge(
            bw2.rename(columns={"bewaehrung_t_wert": "fruehwarnung_t", "cpo_darf_eingreifen": "fruehwarnung_darf"}),
            on="jahr")
        ics.append(bw.assign(strategie=sname, modell="Bewaehrung im Depot"))
        k_std_arr = np.full(len(rows), k_std)
        for vn, tr in variants.items():
            r, eq = evaluate(m, tr, starts)
            r.update({"strategie": sname, "variante": vn})
            if vn.startswith("CPO-Wahl"):
                chosen = tr.sort_index()["grund"].to_numpy()
                r["abweichung_von_standard_pct"] = (chosen != default).mean() * 100
            if "Fruehwarnung" in vn or "Tagesausstieg" in vn:
                r["cpo_verkauf_pct"] = (tr["grund"] == "CPO").mean() * 100
            rows_out.append(r)
            curves[(sname, vn)] = eq
            yearly_tab[f"{sname[:8]} | {vn}"] = yearly(eq)
        # Kontrolle: zufaellige Fruehwarnung mit gleicher Haeufigkeit
        n_cpo = (tr_early["grund"] == "CPO").sum()
        days = np.clip(tr_std["tage"].to_numpy() - MIN_HALTE, 0, None).sum()
        p_day = n_cpo / max(days, 1)
        rr = []
        for _ in range(ZUFALL_LAEUFE):
            xr = np.full(len(meta), -1)
            g = rng.geometric(max(p_day, 1e-9), len(rows)) + MIN_HALTE - 1
            xr[rows] = meta["e"].to_numpy()[rows] + g + 1
            r, _ = evaluate(m, trades_early(m, meta, rows, xr, tr_std), starts[:1])
            rr.append(r)
        d = pd.DataFrame(rr)
        r = d.mean(numeric_only=True).to_dict()
        r.update({"strategie": sname, "variante": f"Zufalls-Fruehwarnung gleich haeufig (Mittel aus {ZUFALL_LAEUFE})",
                  "sharpe_spanne": f"{d['ab_test_sharpe'].min():.2f} bis {d['ab_test_sharpe'].max():.2f}"})
        rows_out.append(r)

    res = pd.DataFrame(rows_out)
    res = res[["strategie", "variante"] + [c for c in res.columns if c not in ("strategie", "variante")]]
    ic = pd.concat(ics, ignore_index=True)
    years = pd.DataFrame(yearly_tab).T.round(1)
    tm.save_csv(res, "vergleich.csv")
    tm.save_csv(ic, "trefferguete.csv")
    tm.save_csv(years, "jahre.csv", index=True)
    chart(curves, bench, out_dir)
    report(res, ic, years, args.lang, time.time() - t0)


def chart(curves: dict, bench: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(dict.fromkeys(k[0] for k in curves))
    fig, axes = plt.subplots(1, len(names), figsize=(8.5 * len(names), 6), squeeze=False)
    for ax, nm in zip(axes[0], names):
        for (s, v), eq in curves.items():
            if s != nm:
                continue
            kw = {"lw": 2.2} if v.startswith("CPO") else ({"color": "black", "lw": 1.8} if "Standard" in v else {"lw": 1, "ls": ":"})
            ax.plot(eq.index, eq / 1000, label=v, **kw)
        for (bn, eq), c in zip(bench.items(), ["#1f77b4", "#555555"]):
            ax.plot(eq.index, eq / 1000, color=c, ls="--", lw=1.2, label=bn)
        ax.set_yscale("log")
        ax.set_title(nm, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Walk-forward ab {tc.ERSTES_TESTJAHR}: Konto in Tsd. EUR (log)")
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "cpo_test.png", dpi=110)
    plt.close(fig)


def report(res, ic, years, long, secs) -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 120)
    print(f"CPO-TEST {'MIT LANGER HISTORIE' if long else 'MIT BISHERIGEM CACHE'}  (Testjahre ab {tc.ERSTES_TESTJAHR}; "
          f"Kosten: {tm.broker_text()})")
    print("=" * 120)
    cols = [c for c in ["variante", "ab_test_rendite_pa_pct", "ab_test_sharpe", "ab_test_max_drawdown_pct",
                        "starts_rendite_pa_pct_mittel", "starts_sharpe_mittel", "starts_max_drawdown_pct_mittel",
                        "trades", "haltedauer", "abweichung_von_standard_pct", "cpo_verkauf_pct", "sharpe_spanne"]
            if c in res.columns]
    for s, g in res.groupby("strategie", sort=False):
        print(f"\n{s if s != '-' else 'Vergleichsmassstaebe'}")
        print(g[cols].round(2).to_string(index=False))
    print("\nTrefferguete je Testjahr (Rangkorrelation Vorhersage / spaetere Rendite):")
    for (s, mo), g in ic.groupby(["strategie", "modell"], sort=False):
        print(f"\n{s} - {mo}")
        print(g.drop(columns=["strategie", "modell"]).round(3).to_string(index=False))
    print("\nRendite je Kalenderjahr (%):")
    print(years.to_string())
    print(f"\nDateien: {tm.OUT_DIR}  (Laufzeit {secs / 60:.1f} min)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(1)
