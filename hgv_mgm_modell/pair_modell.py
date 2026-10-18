"""
pair_modell.py - Mean-Reversion-Modell fuer einen Korb von Aktienpaaren, mit Protokollmodus fuer Paper Trading

Das Modell handelt den Spread jedes Paares nach seinem z-Score. Alle Paare verwenden DIESELBEN Einstellungen;
das Kapital ist gleichmaessig auf die Paare verteilt. Ausgewertet wird in getrennten Zeitraeumen:

    Training:   die TRAIN_YEARS Jahre vor der Testphase   (hier werden die Einstellungen gewaehlt)
    Test:       die letzten TEST_YEARS Jahre bis heute    (Kontrolle; fuer die Paarauswahl nicht wirklich neu)
    Protokoll:  ab dem Starttag des Protokolls            (Paper Trading - die einzigen wirklich neuen Daten)

Aufruf
------
    python pair_modell.py                  # Rueckrechnung: Training und Test, Korb und einzelne Paare
    python pair_modell.py --grid           # zusaetzlich: Sharpe des Korbs im TRAINING fuer mehrere Einstiegs-/Ausstiegswerte
    python pair_modell.py --protokoll      # Protokollmodus: Tagesstand festhalten, Orders fuer morgen, Ergebnis seit Start
    python pair_modell.py --protokoll-neu  # Protokoll neu beginnen (das alte wird in einen Archivordner verschoben)
    python pair_modell.py --refresh        # Kurse neu von Yahoo laden (sonst hoechstens alle 12 Stunden)

Handelsregel (je Paar A / B)
----------------------------
    Spread  = log(Kurs A) - HEDGE_RATIO * log(Kurs B)
    z-Score = (Spread - Mittelwert) / Standardabweichung
    z < -ENTRY_Z  ->  Spread ist ungewoehnlich niedrig:  A kaufen, B leerverkaufen  ("long Spread")
    z > +ENTRY_Z  ->  Spread ist ungewoehnlich hoch:     A leerverkaufen, B kaufen  ("short Spread")
    Ausstieg, sobald eine der Ausstiegsregeln greift (EXIT_Z, MAX_HOLD_DAYS, STOP_Z - siehe unten).

Das Signal entsteht mit dem Schlusskurs eines Tages. Gehandelt wird je nach EXECUTION zu diesem Schlusskurs,
zur Eroeffnung des naechsten Handelstags oder zum naechsten Schlusskurs; die Position verdient erst die
Kursbewegung nach ihrer Ausfuehrung (kein Blick in die Zukunft). Bei "next_open" wird jeder Tag in die
Bewegung ueber Nacht (Schluss -> Eroeffnung) und im Tagesverlauf (Eroeffnung -> Schluss) zerlegt.
Renditen eines Paares beziehen sich auf das dem Paar zugeteilte Kapital (Kauf- und Leerverkaufsseite zusammen),
Renditen des Korbs auf das Gesamtkapital. Die Sharpe Ratio ist annualisiert (Wurzel aus 252), ohne Abzug
eines risikofreien Zinses, weil sich Kauf und Leerverkauf gegenseitig finanzieren.

Simuliertes Konto und drei Profile
----------------------------------
Neben den Renditen der Signale rechnet das Skript Konten in USD (CAPITAL): Gehandelt werden nur ganze Stuecke.
Ein neues Paar bekommt  Zielhebel x Eigenkapital / Zahl der Paare, je zur Haelfte auf der Kauf- und der
Leerverkaufsseite, berechnet mit Eigenkapital und Schlusskursen des Signaltags; die Stueckzahlen bleiben bis zum
Ausstieg unveraendert. Je Order faellt die Gebuehr des Brokers an (Interactive Brokers: je Stueck, mit Mindest- und
Hoechstbetrag) und zusaetzlich Slippage (COST_BPS). Dazu kommen Leihgebuehr auf Leerverkaeufe und Kreditzinsen auf
den Teil der Kaufseite, der das Eigenkapital uebersteigt (FINANCING_BASIS = "kredit") bzw. auf die ganze Kaufseite.

Rueckrechnung und Protokoll rechnen drei Profile, jedes als eigenes Konto:
    Ohne Hebel       Hebel 1.
    Privat           hoechster Hebel eines normalen Margin-Kontos fuer US-Aktien (RETAIL_LEVERAGE = 2). Margin-Regeln:
                     Eine neue Position wird abgelehnt, wenn das Eigenkapital danach weniger als MARGIN_INITIAL aller
                     offenen Positionen deckt; sinkt die Deckung unter MARGIN_MAINTENANCE, schliesst der Broker alles
                     zwangsweise (Margin Call).
    Institutionell   Handel mit Kapital eines Handelshauses, fuer das diese Regeln nicht gelten. Zielhebel =
                     KELLY_FRACTION x Kelly-Hebel aus dem Training, hoechstens INSTITUTIONAL_MAX_LEVERAGE. Das eigene
                     Kapital traegt alle Verluste. Das Haus schliesst das Konto bei 10 % Gesamtverlust
                     (INSTITUTIONAL_STOP) oder 5 % Verlust an einem Tag (INSTITUTIONAL_DAILY_LOSS) - spaetestens, wenn
                     das Kapital aufgebraucht ist. Danach handelt das Modell in diesem Profil nicht weiter.

Kelly-Formel und Stresstest
---------------------------
Optimaler Hebel f = mittlere Rendite / Varianz der Rendite (Chan, Kapitel 6), berechnet NUR aus der Trainingsphase
des Korbs (im Protokoll: aus den TRAIN_YEARS Jahren vor dem Starttag, festgehalten in hebel.json). Weil sich die
Paare gegenseitig ausgleichen, schwankt der Korb wenig, und die Formel liefert sehr hohe Hebel; sie unterstellt
normalverteilte Renditen und bekannte Kennzahlen. Der Stresstest der Rueckrechnung rechnet das Konto mit einer
Reihe von Hebeln ohne jede Grenze durch und zeigt, ab welchem Hebel ein Totalverlust eingetreten waere. Die
Kostenstaffel zeigt die Konten bei unterschiedlich hoher Slippage.

Protokollmodus (--protokoll)
----------------------------
Beim ersten Aufruf werden Starttag und Einstellungen eingefroren (protokoll/start.json). Jedes Paar beginnt
ohne Position und steigt erst ein, nachdem sein z-Score nach dem Starttag einmal im Normalbereich
(|z| < ENTRY_Z) war - es wird also nicht mitten in eine laufende Bewegung eingestiegen. Jeder weitere Aufruf
    - prueft, ob die Einstellungen noch dem eingefrorenen Stand entsprechen (sonst Abbruch),
    - haengt den Tagesstand jedes Paares an protokoll/protokoll.csv an (bestehende Zeilen werden nie geaendert;
      Tage, die erst spaeter eingetragen werden, sind als "nachgetragen" markiert),
    - zeigt die Orders fuer den naechsten Handelstag,
    - schreibt das Ergebnis seit dem Starttag in eine eigene Datei (protokoll/ergebnisse.csv).
In protokoll/ausfuehrungen.csv steht jede Order des Modells mit dem Modellkurs. Wer die Orders in einem
Demokonto ausfuehrt, traegt dort in der Spalte "realer_kurs" den tatsaechlichen Kurs ein; das Skript berechnet
daraus die gemessene Abweichung (Slippage) in Basispunkten. War ein Trade im Demokonto nicht moeglich (z. B. Aktie
nicht leihbar), dort in der Spalte "ausfuehrbar" bei den Einstiegsorders "nein" eintragen: Das Protokoll zeigt das
Konto dann zusaetzlich ohne diese Trades. Am besten taeglich aufrufen, z. B. morgens.
Hebel: Orders, Stueckzahlen und das Hauptergebnis gelten fuer das Konto ohne Hebel. Zusaetzlich rechnet das
Protokoll die Profile "Privat" und "Institutionell" als eigene Konten - exakt, mit groesseren Stueckzahlen,
Gebuehren, Kreditzinsen und den Regeln des Profils, nicht durch Multiplikation der Rendite.
Ein noch laufender US-Handelstag wird nicht verwendet (vorlaeufige Kurse), auch wenn das Skript nachmittags laeuft.
Hinweis: Yahoo bereinigt Kurse rueckwirkend um Dividenden. Der Modellkurs einer Order wird beim ersten
Eintragen festgehalten; wird ein Tag erst nach einem Dividendentermin nachgetragen, kann er deshalb leicht
unter dem damals gehandelten Kurs liegen. Auch aus diesem Grund lohnt der taegliche Aufruf.

Dateien
-------
    output/     kennzahlen.csv (Konto je Phase), paare.csv (je Paar), konto_trades.csv (Trades im Konto, in USD),
                trades.csv (Signale, in %), tageswerte.csv, kostenstaffel.csv, hebel_stresstest.csv,
                margin_calls.csv (falls vorgekommen), modell.png
    protokoll/  start.json, hebel.json, protokoll.csv, ausfuehrungen.csv, ergebnisse.csv, ergebnisse_paare.csv,
                ergebnisse_hebel.csv, trades.csv, konto_trades.csv
    (CSV mit ";" und Dezimalkomma fuer Excel)
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
# EINSTELLUNGEN - hier anpassen
# ======================================================================================================
# Die 15 besten Paare aus pair_screener/output_us_aktien/mr_paare.csv (Rangliste vom 04.10.2026).
# Die Liste ist bewusst fest eingetragen: Sie soll sich waehrend des Paper Tradings nicht aendern.
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

# ---- Zeitraeume
TEST_YEARS = 1.0            # Testphase: so viele Jahre bis zum letzten Kursdatum
TRAIN_YEARS = 3.0           # Trainingsphase: so viele Jahre direkt davor

# ---- z-Score
Z_MODE = "rolling"          # "rolling"  = Mittelwert und Std aus den letzten LOOKBACK Handelstagen (gleitend)
                            # "training" = Mittelwert und Std fest aus der Trainingsphase (wie in Chans Beispielen)
LOOKBACK = 60               # Fensterlaenge in Handelstagen (nur fuer Z_MODE = "rolling")

# ---- Einstieg
ENTRY_Z = 1.25              # Einstieg, wenn der z-Score ueber +ENTRY_Z oder unter -ENTRY_Z liegt

# ---- Ausstieg (jede Regel einzeln einstellbar; es gilt, was zuerst eintritt)
EXIT_Z = 0.25               # Ausstieg, wenn der z-Score bis auf EXIT_Z an den Mittelwert zurueckgekehrt ist:
                            #    0.0 = am Mittelwert,  0.5 = schon frueher (bei +/-0.5),
                            #   -0.5 = erst nach Ueberschiessen auf die andere Seite
MAX_HOLD_DAYS = 40          # Zeit-Stopp: spaetestens nach so vielen Handelstagen schliessen (None = aus)
STOP_Z = None               # Verlust-Stopp: schliessen, wenn der z-Score weiter weglaeuft als STOP_Z
                            #   (z. B. 4.0; None = aus). Danach kein neuer Einstieg, bis |z| wieder < ENTRY_Z.

# ---- Hedge Ratio
HEDGE_RATIO = 1.0           # 1.0 = gleicher Betrag in beiden Aktien (dollar-neutral)
                            # "training" = je Paar per Regression log(A) auf log(B) in der Trainingsphase geschaetzt

# ---- Simuliertes Konto
CAPITAL = 50_000            # Eigenkapital in USD zu Beginn jeder Phase (Training, Test, Protokoll).
                            #   Gehandelt werden nur ganze Stuecke; reicht der Betrag je Aktie nicht fuer ein
                            #   Stueck, faellt der Trade aus.

# ---- Gebuehren: Interactive Brokers, IBKR Pro, US-Aktien, Tarif "Fixed" (Stand Oktober 2026)
COMMISSION_PER_SHARE = 0.005    # USD je Stueck
COMMISSION_MIN = 1.00           # Mindestgebuehr je Order in USD
COMMISSION_MAX_PCT = 0.01       # hoechstens 1 % des Orderwerts
                            #   Tarif "Tiered" (bis 300.000 Stueck/Monat): 0.0035 je Stueck, mindestens 0.35 je Order,
                            #   zuzueglich Boersen- und Clearinggebuehren, die hier nicht modelliert sind.

# ---- Drei Konto-Profile (Rueckrechnung und Protokoll rechnen jedes als eigenes Konto mit CAPITAL)
#   1. Ohne Hebel:      Hebel 1, keine Margin-Fragen.
#   2. Privat:          der hoechste Hebel eines normalen Margin-Kontos fuer US-Aktien (Reg T).
#   3. Institutionell:  Handel mit Kapital eines Handelshauses (Proprietary Trading Firm), fuer das Reg T nicht gilt.
#                       Zielhebel = KELLY_FRACTION x Kelly-Hebel aus dem Training, hoechstens INSTITUTIONAL_MAX_LEVERAGE.
RETAIL_LEVERAGE = 2.0       # Zielhebel im Profil "Privat"
MARGIN_INITIAL = 0.50       # Privat: Eine neue Position wird nur eroeffnet, wenn das Eigenkapital danach mindestens
                            #   diesen Anteil aller offenen Positionen (Kauf- plus Leerverkaufsseite) deckt.
                            #   0.50 = Reg T -> hoechstens Hebel 2. None = nicht pruefen.
MARGIN_MAINTENANCE = 0.30   # Privat: Faellt das Eigenkapital unter diesen Anteil der offenen Positionen, schliesst der
                            #   Broker alles zwangsweise (Margin Call). Ueblich: 25 % fuer Kaeufe, 30 % fuer Leerverkaeufe.
KELLY_FRACTION = 0.25       # Institutionell: Anteil des Kelly-Hebels (1.0 = voller Kelly, 0.5 = halber, 0.25 = Viertel)
INSTITUTIONAL_MAX_LEVERAGE = 6   # Institutionell: hoechster Hebel, den das Handelshaus stellt. Neue Positionen werden
                            #   nur bis zu diesem Vielfachen des eigenen Kapitals eroeffnet. Das eigene Kapital traegt
                            #   alle Verluste; ist es aufgebraucht, schliesst das Haus das Konto (TOTALVERLUST).
INSTITUTIONAL_STOP = 0.90   # Institutionell, Gesamtverlustgrenze: Das Haus schliesst das Konto, wenn das Kapital unter
                            #   diesen Anteil des Startkapitals faellt. 0.90 = hoechstens 10 % Verlust - der bei
                            #   Handelshaeusern am haeufigsten genannte Wert (ueblich sind 6 bis 12 %). None = aus.
INSTITUTIONAL_DAILY_LOSS = 0.05     # Institutionell, Tagesverlustgrenze: Das Konto wird geschlossen, wenn der Verlust
                            #   eines Tages diesen Anteil des Kontostands vom Vorabend erreicht. 0.05 = 5 %
                            #   (ueblich: 4 bis 6 %). None = aus. Die Gesamtverlustgrenze gilt fest ab dem
                            #   Startkapital und wird nicht nachgezogen.
                            #   Massgeblich ist der Vertrag des jeweiligen Handelshauses - dort nachsehen.
STRESS_LEVERAGE = [1, 2, 4, 8, 16, 32, 64, 128]   # Hebel fuer den Stresstest "waere ein Totalverlust eingetreten?"
COST_LADDER_BPS = [0, 5, 10, 20, 30]                # Slippage-Stufen fuer die Kostenstaffel

# ---- Kosten und Ausfuehrung
COST_BPS = 5                # Slippage je Kauf oder Verkauf in Basispunkten des Orderwerts (Abstand zwischen
                            #   Modellkurs und tatsaechlichem Kurs). Die Gebuehren des Brokers kommen im Konto dazu.
BORROW_FEE = 0.01           # Leihgebuehr p. a. auf die leerverkaufte Seite
FINANCING_RATE = 0.0538     # Zins p. a. fuer den Hebel (Wertpapierkredit). 5,38 % = Interactive Brokers (IBKR Pro),
                            #   USD-Kredit bis 100.000, Stand Oktober 2026: Referenzzins 3,88 % + 1,5 % Aufschlag.
                            #   Viele deutsche Broker und CFD-Anbieter liegen hoeher - Satz des eigenen Brokers eintragen.
FINANCING_BASIS = "kredit"  # worauf der Zins anfaellt, solange Positionen offen sind:
                            #   "kredit" = nur auf den Teil der Kaufseite, der das Eigenkapital uebersteigt
                            #              (Wertpapierkredit / Margin-Konto)
                            #   "voll"   = auf die gesamte Kaufseite (wie bei CFDs ueblich)
SHORT_PROCEEDS_RATE = 0.0   # Zins p. a., den der Broker auf den Erloes der Leerverkaeufe gutschreibt
                            #   (Privatkunden erhalten meist nichts -> 0)
EXECUTION = "next_open"     # wann nach einem Signal (Schlusskurs des Tages) gehandelt wird:
                            #   "close"      = zum Schlusskurs desselben Tages (leicht geschoent: der Schlusskurs
                            #                  steht erst fest, wenn der Handel vorbei ist)
                            #   "next_open"  = zum Eroeffnungskurs des naechsten Handelstags (Order ueber Nacht)
                            #   "next_close" = zum Schlusskurs des naechsten Handelstags (strengste Variante)

# ---- Raster fuer --grid (wird NUR in der Trainingsphase ausgewertet, fuer den ganzen Korb)
GRID_ENTRY_Z = [1.0, 1.25, 1.5, 2.0, 2.5]
GRID_EXIT_Z = [-0.5, 0.0, 0.25, 0.5, 1.0]
# ======================================================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "daten"
OUT_DIR = BASE_DIR / "output"
PROT_DIR = BASE_DIR / "protokoll"
HISTORY_YEARS = 12          # so viel Historie wird von Yahoo geladen (reicht auch fuer laengere Phasen)
MIN_DAYS_FOR_SHARPE = 60    # im Protokoll wird eine Sharpe Ratio erst ab so vielen Handelstagen gezeigt


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def pair_name(a: str, b: str) -> str:
    return f"{a}/{b}"


def tickers() -> list[str]:
    return list(dict.fromkeys(t for pair in PAIRS for t in pair))


def save_csv(df: pd.DataFrame, name: str, index: bool = False, folder: Path | None = None) -> bool:
    """CSV fuer Excel schreiben. Gibt False zurueck, wenn die Datei gesperrt war (dann unter anderem Namen)."""
    folder = folder or OUT_DIR
    folder.mkdir(exist_ok=True)
    path = folder / name
    try:
        df.to_csv(path, sep=";", decimal=",", index=index, encoding="utf-8-sig")
        return True
    except PermissionError:                                 # Datei ist z. B. in Excel geoeffnet
        alt = folder / f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}"
        df.to_csv(alt, sep=";", decimal=",", index=index, encoding="utf-8-sig")
        log(f"{name} ist gesperrt (in Excel offen?) - gespeichert als {alt.name}")
        return False


# ------------------------------------------------------------------------------------------ Kurse
def drop_unfinished_day(px: pd.DataFrame) -> pd.DataFrame:
    """Laufenden US-Handelstag entfernen: Waehrend des Handels liefert Yahoo fuer heute einen vorlaeufigen
    'Schlusskurs'. Ein Signal daraus waere falsch und wuerde im Protokoll festgeschrieben."""
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        today, done = now.normalize().tz_localize(None), now.hour * 60 + now.minute >= 16 * 60 + 20
    except Exception:                                       # keine Zeitzonendaten: Naeherung ueber Weltzeit
        now = pd.Timestamp.now(tz="UTC")
        today, done = now.normalize().tz_localize(None), now.hour * 60 + now.minute >= 21 * 60 + 20
    if len(px) and px.index[-1] >= today and not done:
        log(f"Der US-Handelstag {px.index[-1]:%d.%m.%Y} laeuft noch - er wird erst nach Handelsschluss "
            f"(ca. 22:20 Uhr deutscher Zeit) beruecksichtigt")
        px = px[px.index < today]
    return px


def load_prices(refresh: bool) -> pd.DataFrame:
    """Tageskurse aller Aktien, um Dividenden und Splits bereinigt.

    Spalten: Ticker = Schlusskurs, "<Ticker>_open" = Eroeffnungskurs (fuer EXECUTION = "next_open").
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "kurse.csv"
    tk = tickers()
    open_cols = [f"{t}_open" for t in tk]
    need_open = EXECUTION == "next_open"

    def read_cache() -> pd.DataFrame:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    def usable(px: pd.DataFrame) -> bool:                   # alle Aktien vorhanden (und Eroeffnungskurse, falls noetig)?
        return set(tk) <= set(px.columns) and (not need_open or set(open_cols) <= set(px.columns))

    fresh = cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600
    if fresh and not refresh:
        px = read_cache()
        if usable(px):
            log(f"Kurse aus {cache.name} (bis {px.index[-1]:%d.%m.%Y})")
            return px
        log("Gespeicherte Kurse sind unvollstaendig (neue Aktien oder fehlende Eroeffnungskurse) - lade neu")
    try:
        import yfinance as yf

        start = (pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
        log(f"Lade Kurse fuer {len(tk)} Aktien von Yahoo Finance ...")
        data = yf.download(tk, start=start, auto_adjust=True, progress=False)
        px = data["Close"].reindex(columns=tk).copy()
        px[open_cols] = data["Open"].reindex(columns=tk).to_numpy()
        px = px.dropna(how="all")
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = drop_unfinished_day(px)
        missing = [t for t in tk if px[t].notna().sum() == 0]
        if len(missing) == len(tk):
            raise RuntimeError("keine Kurse erhalten")
        if missing:
            log(f"Keine Kurse fuer: {', '.join(missing)} - Paare mit diesen Aktien werden ausgelassen")
        px.rename_axis("datum").to_csv(cache)
        log(f"Kurse geladen (bis {px.index[-1]:%d.%m.%Y})")
        return px
    except Exception as e:
        if cache.exists() and usable(read_cache()):
            px = read_cache()
            log(f"Yahoo nicht erreichbar ({type(e).__name__}) - verwende gespeicherte Kurse bis {px.index[-1]:%d.%m.%Y}")
            return px
        raise SystemExit(f"Kurse konnten nicht geladen werden ({type(e).__name__}: {e}). "
                         f"Ist yfinance installiert (pip install yfinance) und besteht eine Internetverbindung?")


# ------------------------------------------------------------------------------------------ Modell
def build_signals(px: pd.DataFrame, a: str, b: str, train: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Spread und z-Score eines Paares fuer die ganze Historie. train = Maske der Trainingstage (auf px.index).

    Das Ergebnis hat denselben Index wie px; an Tagen ohne Kurs einer der beiden Aktien steht NaN.
    """
    cols = [a, b] + ([f"{a}_open", f"{b}_open"] if f"{a}_open" in px.columns and f"{b}_open" in px.columns else [])
    sub = px[cols].dropna()
    tr = pd.Series(train, index=px.index).reindex(sub.index).to_numpy()
    la, lb = np.log(sub[a]), np.log(sub[b])
    if HEDGE_RATIO == "training":                           # Regression nur mit Trainingsdaten
        if tr.sum() < 60:
            raise SystemExit(f'{pair_name(a, b)}: zu wenig Trainingsdaten fuer HEDGE_RATIO = "training"')
        beta = float(np.polyfit(lb[tr], la[tr], 1)[0])
    else:
        beta = float(HEDGE_RATIO)
    spread = la - beta * lb
    if Z_MODE == "training":                                # feste Werte aus der Trainingsphase
        z = (spread - spread[tr].mean()) / spread[tr].std()
    elif Z_MODE == "rolling":                               # gleitend, nutzt nur vergangene Tage
        roll = spread.rolling(LOOKBACK, min_periods=LOOKBACK)
        z = (spread - roll.mean()) / roll.std()
    else:
        raise SystemExit('Z_MODE muss "rolling" oder "training" sein')
    w_a, w_b = 1 / (1 + abs(beta)), beta / (1 + abs(beta))  # Kapitalanteile der beiden Seiten
    out = pd.DataFrame({"kurs_a": sub[a], "kurs_b": sub[b], "spread": spread, "z": z,
                        "spread_rendite": w_a * sub[a].pct_change() - w_b * sub[b].pct_change()})
    if len(cols) == 4:                                      # Zerlegung fuer die Ausfuehrung zur Eroeffnung
        oa, ob = sub[f"{a}_open"], sub[f"{b}_open"]
        out["open_a"], out["open_b"] = oa, ob
        out["rendite_nacht"] = w_a * (oa / sub[a].shift(1) - 1) - w_b * (ob / sub[b].shift(1) - 1)
        out["rendite_tag"] = w_a * (sub[a] / oa - 1) - w_b * (sub[b] / ob - 1)
    return out.reindex(px.index), beta


def simulate(sig: pd.DataFrame, beta: float, entry_z: float, exit_z: float,
             start_blocked: bool = False, close_at_end: bool = True) -> pd.DataFrame:
    """Handelt ein Paar von Anfang (ohne Position) bis Ende eines Zeitraums. Gibt Tageswerte zurueck.

    start_blocked: erst einsteigen, nachdem |z| einmal unter entry_z lag (Protokoll: kein Einstieg in eine
                   laufende Bewegung). close_at_end: fuer eine am Ende offene Position Schlusskosten ansetzen.
    """
    z = sig["z"].to_numpy()
    n = len(z)
    signal = np.zeros(n)                                    # gewuenschte Position nach Schlusskurs des Tages
    reason = np.array([""] * n, dtype=object)               # Grund eines Ausstiegs an diesem Tag
    cur, held, blocked = 0, 0, bool(start_blocked)
    for t in range(n):
        zt = z[t]
        if zt != zt:                                        # kein z-Score (zu wenig Historie): keine Position
            if cur != 0:
                reason[t] = "keine Daten"
            cur, held = 0, 0
        elif cur == 0:
            if blocked and abs(zt) < entry_z:               # nach Stopp / Protokollstart erst wieder im Normalbereich
                blocked = False
            if not blocked:
                if zt > entry_z:
                    cur, held = -1, 0                       # Spread zu hoch -> short Spread
                elif zt < -entry_z:
                    cur, held = 1, 0                        # Spread zu niedrig -> long Spread
        else:
            held += 1
            if STOP_Z is not None and ((cur == 1 and zt <= -STOP_Z) or (cur == -1 and zt >= STOP_Z)):
                reason[t], cur, held, blocked = "Stopp", 0, 0, True
            elif (cur == 1 and zt >= -exit_z) or (cur == -1 and zt <= exit_z):
                reason[t], cur, held = "z-Score", 0, 0
            elif MAX_HOLD_DAYS is not None and held >= MAX_HOLD_DAYS:
                reason[t], cur, held = "Zeit", 0, 0
        signal[t] = cur
    wanted = signal.copy()                                  # Signal nach Schlusskurs, vor der Ausfuehrung
    if EXECUTION in ("next_open", "next_close"):            # Ausfuehrung erst am naechsten Handelstag
        signal = np.concatenate([[0.0], signal[:-1]])
        reason = np.concatenate([np.array([""], dtype=object), reason[:-1]])
    # signal = Position am Tagesende; prev = Position, die ueber Nacht gehalten wurde
    prev = np.concatenate([[0.0], signal[:-1]])
    full_day = np.nan_to_num(sig["spread_rendite"].to_numpy())
    if EXECUTION == "next_open":
        # Handel zur Eroeffnung: ueber Nacht verdient noch die alte Position, im Tagesverlauf schon die neue
        night, day = np.nan_to_num(sig["rendite_nacht"].to_numpy()), np.nan_to_num(sig["rendite_tag"].to_numpy())
        gross = np.where(prev == signal, prev * full_day, prev * night + signal * day)
    else:
        gross = prev * full_day                             # Handel zum Schlusskurs: Rendite ab dem Folgetag
    cost = np.abs(signal - prev) * COST_BPS / 1e4
    short_share = np.where(prev > 0, abs(beta), 1.0) / (1 + abs(beta))      # long Spread: B ist leerverkauft
    borrow = np.abs(prev) * short_share * BORROW_FEE / 252
    if n and signal[-1] != 0:                               # am Ende offene Position
        reason[-1] = "offen"
        if close_at_end:
            cost[-1] += abs(signal[-1]) * COST_BPS / 1e4
    long_share = np.abs(prev) * np.where(prev > 0, 1.0, abs(beta)) / (1 + abs(beta))    # Kaufseite: A bzw. B
    return pd.DataFrame({"signal": wanted, "position": signal, "brutto": gross, "kosten": cost + borrow,
                         "netto": gross - cost - borrow, "ausstieg_grund": reason,
                         "kaufseite": long_share, "leerseite": np.abs(prev) * short_share}, index=sig.index)


def direction(pos: float, a: str, b: str) -> str:
    return f"long {a} / short {b}" if pos > 0 else f"short {a} / long {b}" if pos < 0 else "keine Position"


def list_trades(res: pd.DataFrame, sig: pd.DataFrame, phase: str, a: str, b: str) -> pd.DataFrame:
    """Einzelne Trades eines Paares in einem Zeitraum: Bloecke mit Position != 0."""
    pos, net = res["position"].to_numpy(), res["netto"].to_numpy()
    idx, z = res.index, sig["z"].to_numpy()
    prev = np.concatenate([[0.0], pos[:-1]])
    lag = 0 if EXECUTION == "close" else 1                  # Tage zwischen Signal und Ausfuehrung
    rows = []
    for i in np.where((pos != 0) & (prev != pos))[0]:
        e = i
        while e + 1 < len(pos) and pos[e + 1] == pos[i]:
            e += 1
        is_open = e == len(pos) - 1                         # laeuft am Ende des Zeitraums noch
        x = min(e + 1, len(pos) - 1)                        # Tag, an dem geschlossen wird
        rows.append({"phase": phase, "paar": pair_name(a, b), "einstieg": idx[i].date(),
                     "ausstieg": None if is_open else idx[x].date(), "richtung": direction(pos[i], a, b),
                     "z_einstieg": z[max(i - lag, 0)], "z_ausstieg": z[max(x - lag, 0)],   # z-Score des Signals
                     "handelstage": e - i + 1,
                     "grund": "offen" if is_open else (res["ausstieg_grund"].iloc[x] or "z-Score"),
                     "ergebnis": float(net[i:x + 1].sum())})
    return pd.DataFrame(rows, columns=["phase", "paar", "einstieg", "ausstieg", "richtung", "z_einstieg",
                                       "z_ausstieg", "handelstage", "grund", "ergebnis"])


def combine(results: list[pd.DataFrame]) -> pd.DataFrame:
    """Korb: gleiches Kapital je Paar. Renditen und Kapitalanteile sind der Durchschnitt ueber alle Paare."""
    out = pd.DataFrame(index=results[0].index)
    for col in ("brutto", "kosten", "netto", "kaufseite", "leerseite"):
        out[col] = np.mean([r[col].to_numpy() for r in results], axis=0)
    out["position"] = np.sum([np.abs(r["position"].to_numpy()) for r in results], axis=0)    # offene Paare
    return out


def stats(res: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Kennzahlen eines Zeitraums aus den Tagesrenditen nach Kosten."""
    r = res["netto"].to_numpy()
    years = max(len(r) / 252, 1e-9)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float(np.sqrt(252) * r.mean() / sd) if sd > 0 else float("nan")
    g = res["brutto"].to_numpy()
    gsd = g.std(ddof=1) if len(g) > 1 else 0.0
    eq = np.cumprod(1 + r)
    closed = trades[trades["grund"] != "offen"] if len(trades) else trades
    return {"von": res.index[0].date(), "bis": res.index[-1].date(), "handelstage": len(r),
            "sharpe": sharpe,
            # Faustformel fuer die Unsicherheit einer geschaetzten Sharpe Ratio (Standardfehler)
            "sharpe_unsicherheit": float(np.sqrt((1 + 0.5 * sharpe ** 2) / years)) if sharpe == sharpe else float("nan"),
            "sharpe_vor_kosten": float(np.sqrt(252) * g.mean() / gsd) if gsd > 0 else float("nan"),
            "rendite_pa": float(r.mean() * 252), "gesamtrendite": float(eq[-1] - 1),
            "wachstum_pa": float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0,   # mit Zinseszins
            "schlechtester_tag": float(r.min()),
            "max_rueckgang": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "trades": len(trades), "gewinner": int((trades["ergebnis"] > 0).sum()),
            "mittel_je_trade": float(trades["ergebnis"].mean()) if len(trades) else float("nan"),
            "schlechtester_trade": float(trades["ergebnis"].min()) if len(trades) else float("nan"),
            "mittlere_haltedauer": float(closed["handelstage"].mean()) if len(closed) else float("nan"),
            "anteil_tage_investiert": float((res["position"] != 0).mean())}


def kelly_leverage(res: pd.DataFrame) -> dict:
    """Kelly-Hebel eines Zeitraums: f = m / s^2 mit m = mittlere Tagesrendite, s^2 = Varianz der Tagesrendite.

    m ist die Rendite je Hebel-Einheit NACH den Finanzierungskosten, die eine weitere Hebel-Einheit kostet
    (Zins auf die zusaetzliche Kaufseite). Das entspricht bei Chan dem Abzug des Finanzierungszinses.
    Die maximale erwartete Wachstumsrate bei vollem Kelly-Hebel ist dann etwa m^2 / (2 s^2) (stetig).
    """
    marginal = (res["kaufseite"].to_numpy() * FINANCING_RATE
                - res["leerseite"].to_numpy() * SHORT_PROCEEDS_RATE) / 252
    r = res["netto"].to_numpy()
    x = r - marginal
    m, var = x.mean(), x.var(ddof=1)
    f = float(m / var) if var > 0 else float("nan")
    return {"rendite_pa": float(r.mean() * 252), "finanzierung_je_hebel_pa": float(marginal.mean() * 252),
            "rendite_nach_finanzierung_pa": float(m * 252), "volatilitaet_pa": float(np.sqrt(var * 252)),
            "kelly_hebel": f, "wachstum_bei_kelly": float(m * 252) ** 2 / (2 * var * 252) if var > 0 else float("nan")}


def sharpe_of(r: np.ndarray) -> float:
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(np.sqrt(252) * r.mean() / sd) if sd > 0 else float("nan")


def commission(shares: float, value: float) -> float:
    """Gebuehr des Brokers fuer eine Order."""
    if shares <= 0:
        return 0.0
    return float(min(max(COMMISSION_MIN, COMMISSION_PER_SHARE * shares), COMMISSION_MAX_PCT * value))


def simulate_account(pairs: list[tuple[str, str]], sigs: dict, betas: dict, positions: dict, capital: float,
                     lev: float, phase: str, slippage_bps: float | None = None, margin_rules: bool = True,
                     skip: set | None = None, close_at_end: bool = True, margin_initial: float | None = -1.0,
                     margin_maintenance: float | None = -1.0, stop_level: float | None = None,
                     daily_loss_limit: float | None = None) -> dict:
    """Konto in USD: ganze Stuecke, Gebuehren je Order, Leihgebuehr, Kreditzinsen, Margin Call, Totalverlust.

    sigs[name]:      Tageswerte des Paares im Zeitraum (kurs_a, kurs_b, ggf. open_a, open_b)
    positions[name]: Position am Tagesende (+1 / 0 / -1) aus simulate() - die Signale haengen nicht vom Konto ab
    lev:             Zielhebel. Ein neues Paar bekommt  lev * Eigenkapital / Zahl der Paare  (Kauf- plus
                     Leerverkaufsseite), berechnet mit Eigenkapital und Schlusskursen des Signaltags.
    Stueckzahlen bleiben bis zum Ausstieg unveraendert (kein taegliches Nachjustieren).
    skip:            {(Paar, Einstiegstag)}, die nicht eroeffnet werden (im Protokoll als nicht ausfuehrbar markiert)
    margin_initial / margin_maintenance: Deckungsregeln dieses Kontos (-1 = die Einstellungen MARGIN_INITIAL /
                     MARGIN_MAINTENANCE verwenden, None = keine Pruefung); margin_rules = False schaltet beide ab.
    stop_level:      Das Konto wird geschlossen, wenn das Eigenkapital unter diesen Anteil des Startkapitals faellt.
    daily_loss_limit: Das Konto wird geschlossen, wenn der Verlust eines Tages diesen Anteil des Kontostands vom
                     Vorabend erreicht. Nach einer Schliessung handelt das Modell nicht weiter.
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
    ca = {nm: sigs[nm]["kurs_a"].to_numpy() for nm in names}
    cb = {nm: sigs[nm]["kurs_b"].to_numpy() for nm in names}
    oa = {nm: sigs[nm]["open_a"].to_numpy() for nm in names} if at_open else ca
    ob = {nm: sigs[nm]["open_b"].to_numpy() for nm in names} if at_open else cb
    pos = {nm: np.asarray(positions[nm], dtype=float) for nm in names}

    equity = np.full(n, float(capital))
    gross_ratio = np.zeros(n)
    hold = {nm: [0.0, 0.0] for nm in names}                 # Stueck A, Stueck B (negativ = leerverkauft)
    live: dict[str, dict] = {}                              # laufende Trades
    orders, trades, margin_calls = [], [], []
    cost = {"gebuehren": 0.0, "slippage": 0.0, "leihe": 0.0, "zinsen": 0.0}
    count = {"orders": 0, "zu_klein": 0, "margin_abgelehnt": 0, "nicht_ausfuehrbar": 0}
    ruin_day = None
    close_reason = ""
    E = float(capital)
    day_start = float(capital)                              # Eigenkapital zum Schlusskurs des Vortags

    def num(x: float) -> float:
        return 0.0 if x != x else float(x)

    def gross(pa: dict, pb: dict, t: int) -> float:
        return sum(abs(hold[nm][0]) * num(pa[nm][t]) + abs(hold[nm][1]) * num(pb[nm][t]) for nm in names)

    def order(t: int, nm: str, k: int, shares: float, price: float, what: str) -> float:
        """Eine Order buchen. shares > 0 = Kauf, < 0 = Verkauf. Gibt die Kosten zurueck."""
        value = abs(shares) * price
        fee, sl = commission(abs(shares), value), slip * value
        cost["gebuehren"] += fee
        cost["slippage"] += sl
        count["orders"] += 1
        orders.append({"datum": idx[t], "paar": nm, "aktie": pairs[names.index(nm)][k], "order": what,
                       "seite": "kauf" if shares > 0 else "verkauf", "stueck": int(abs(shares)),
                       "modellkurs": price, "gebuehr": fee})
        return fee + sl

    def close_pair(t: int, nm: str, pa: float, pb: float, why: str) -> float:
        """Position eines Paares schliessen. Gibt die Kosten zurueck."""
        sa, sb = hold[nm]
        c = 0.0
        if sa:
            c += order(t, nm, 0, -sa, pa, "verkaufen (Position schliessen)" if sa > 0
                       else "zurueckkaufen (Leerverkauf schliessen)")
        if sb:
            c += order(t, nm, 1, -sb, pb, "verkaufen (Position schliessen)" if sb > 0
                       else "zurueckkaufen (Leerverkauf schliessen)")
        hold[nm] = [0.0, 0.0]
        tr = live.pop(nm, None)
        if tr is not None:
            tr["ergebnis_usd"] -= c
            tr.update({"ausstieg": idx[t].date(), "grund": why})
            trades.append(tr)
        return c

    def check(t: int, pa: dict, pb: dict) -> None:
        """Totalverlust und Margin Call pruefen (pa, pb = aktuelle Kurse)."""
        nonlocal E, ruin_day, close_reason
        g = gross(pa, pb, t)
        why = ""
        if E <= 0:
            why = "Totalverlust"
        elif E <= floor and floor > 0:
            why = "Gesamtverlust"
        elif daily_loss_limit is not None and day_start - E >= daily_loss_limit * day_start:
            why = "Tagesverlust"
        if why:                                             # Konto wird geschlossen: Modell faellt aus
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    if E > 0:                               # Positionen werden noch regulaer geschlossen
                        E -= close_pair(t, nm, num(pa[nm][t]), num(pb[nm][t]), why)
                    else:
                        hold[nm] = [0.0, 0.0]
                        tr = live.pop(nm, None)
                        if tr is not None:
                            tr.update({"ausstieg": idx[t].date(), "grund": why})
                            trades.append(tr)
            E, ruin_day, close_reason = max(E, 0.0), idx[t], why
        elif m_maint is not None and g > 0 and E < m_maint * g:
            before = E
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    E -= close_pair(t, nm, num(pa[nm][t]), num(pb[nm][t]), "Margin Call")
            margin_calls.append({"phase": phase, "datum": idx[t].date(), "eigenkapital": before,
                                 "offene_positionen": g, "deckung": before / g})

    for t in range(n):
        if ruin_day is not None:
            equity[t] = E
            continue
        E_signal = E                                        # Eigenkapital zum Schlusskurs des Vortags
        day_start = E
        # ---- Leihgebuehr und Kreditzinsen fuer die ueber Nacht gehaltenen Positionen (Werte zum Vortagesschluss)
        if t > 0:
            long_mv = short_mv = 0.0
            for nm in names:
                va, vb = hold[nm][0] * num(ca[nm][t - 1]), hold[nm][1] * num(cb[nm][t - 1])
                long_mv += max(va, 0.0) + max(vb, 0.0)
                short_here = max(-va, 0.0) + max(-vb, 0.0)
                short_mv += short_here
                if nm in live and short_here:
                    live[nm]["ergebnis_usd"] -= short_here * BORROW_FEE / 252
            debt = long_mv if FINANCING_BASIS == "voll" else max(long_mv - E, 0.0)
            borrow = short_mv * BORROW_FEE / 252
            interest = debt * FINANCING_RATE / 252 - short_mv * SHORT_PROCEEDS_RATE / 252
            cost["leihe"] += borrow
            cost["zinsen"] += interest
            E -= borrow + interest
        # ---- Kursbewegung bis zur Ausfuehrung
        if at_open and t > 0:                               # ueber Nacht: Vortagesschluss -> Eroeffnung
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    pnl = hold[nm][0] * num(oa[nm][t] - ca[nm][t - 1]) + hold[nm][1] * num(ob[nm][t] - cb[nm][t - 1])
                    E += pnl
                    live[nm]["ergebnis_usd"] += pnl
            check(t, oa, ob)
        elif not at_open and t > 0:                         # ganzer Tag: Schluss -> Schluss
            for nm in names:
                if hold[nm][0] or hold[nm][1]:
                    pnl = hold[nm][0] * num(ca[nm][t] - ca[nm][t - 1]) + hold[nm][1] * num(cb[nm][t] - cb[nm][t - 1])
                    E += pnl
                    live[nm]["ergebnis_usd"] += pnl
            check(t, ca, cb)
        # ---- Orders: erst schliessen, dann eroeffnen
        if ruin_day is None:
            xa, xb = (oa, ob) if at_open else (ca, cb)      # Ausfuehrungskurse
            changed = [nm for nm in names if pos[nm][t] != (pos[nm][t - 1] if t > 0 else 0.0)]
            for nm in changed:
                if hold[nm][0] or hold[nm][1]:
                    E -= close_pair(t, nm, num(xa[nm][t]), num(xb[nm][t]), "Signal")
            for nm in changed:
                p = pos[nm][t]
                if p == 0:
                    continue
                if (nm, idx[t].date()) in skip:
                    count["nicht_ausfuehrbar"] += 1
                    continue
                # Groesse: mit Eigenkapital und Schlusskursen des Signaltags (vor der Eroeffnung bekannt)
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
                    count["zu_klein"] += 1
                    continue
                if m_init is not None:
                    after = gross(xa, xb, t) + na * pa_x + nb * pb_x
                    if E < m_init * after:
                        count["margin_abgelehnt"] += 1
                        continue
                hold[nm] = [p * na, -p * nb]
                a_, b_ = pairs[names.index(nm)]
                c = (order(t, nm, 0, p * na, pa_x, "kaufen" if p > 0 else "leerverkaufen")
                     + order(t, nm, 1, -p * nb, pb_x, "leerverkaufen" if p > 0 else "kaufen"))
                E -= c
                live[nm] = {"phase": phase, "paar": nm, "einstieg": idx[t].date(), "ausstieg": None,
                            "richtung": direction(p, a_, b_), "stueck_a": int(na), "stueck_b": int(nb),
                            "wert_a_usd": na * pa_x, "wert_b_usd": nb * pb_x, "grund": "offen", "ergebnis_usd": -c}
            if at_open:                                     # Tagesverlauf: Eroeffnung -> Schluss
                for nm in names:
                    if hold[nm][0] or hold[nm][1]:
                        pnl = hold[nm][0] * num(ca[nm][t] - oa[nm][t]) + hold[nm][1] * num(cb[nm][t] - ob[nm][t])
                        E += pnl
                        live[nm]["ergebnis_usd"] += pnl
            check(t, ca, cb)
        equity[t] = max(E, 0.0)
        gross_ratio[t] = gross(ca, cb, t) / E if E > 0 else 0.0

    # ---- am Ende offene Positionen: zum letzten Schlusskurs bewertet; Schlusskosten nur in der Rueckrechnung
    holdings = {nm: (int(hold[nm][0]), int(hold[nm][1])) for nm in names}
    if ruin_day is None:
        for nm in list(live):
            tr = live[nm]
            if close_at_end:
                sa, sb = hold[nm]
                c = sum(commission(abs(q), abs(q) * num(pr[nm][-1])) + slip * abs(q) * num(pr[nm][-1])
                        for q, pr in ((sa, ca), (sb, cb)))
                E -= c
                tr["ergebnis_usd"] -= c
                cost["gebuehren"] += sum(commission(abs(q), abs(q) * num(pr[nm][-1])) for q, pr in ((sa, ca), (sb, cb)))
                cost["slippage"] += sum(slip * abs(q) * num(pr[nm][-1]) for q, pr in ((sa, ca), (sb, cb)))
            trades.append(tr)
        equity[-1] = max(E, 0.0)
    eq = pd.Series(equity, index=idx)
    r = eq.pct_change().fillna(eq.iloc[0] / capital - 1).replace([np.inf, -np.inf], 0.0).to_numpy()
    years = max(n / 252, 1e-9)
    tr_df = pd.DataFrame(trades, columns=["phase", "paar", "einstieg", "ausstieg", "richtung", "stueck_a", "stueck_b",
                                          "wert_a_usd", "wert_b_usd", "grund", "ergebnis_usd"])
    for c in ("wert_a_usd", "wert_b_usd", "ergebnis_usd"):
        tr_df[c] = tr_df[c].astype(float).round(2)
    opened = len(tr_df)
    imbalance = (float(((tr_df["wert_a_usd"] - tr_df["wert_b_usd"]).abs()
                        / ((tr_df["wert_a_usd"] + tr_df["wert_b_usd"]) / 2)).mean()) if opened else float("nan"))
    order_value = sum(o["stueck"] * o["modellkurs"] for o in orders)
    return {"equity": eq, "returns": r, "trades": tr_df, "orders": orders, "margin_calls": margin_calls,
            "holdings": holdings, "ruin_day": ruin_day, "cost": cost, "count": count, "order_value": order_value,
            "gross_ratio": pd.Series(gross_ratio, index=idx),
            "kpi": {"von": idx[0].date(), "bis": idx[-1].date(), "handelstage": n,
                    "startkapital": float(capital), "endkapital": float(eq.iloc[-1]),
                    "gesamtrendite": float(eq.iloc[-1] / capital - 1),
                    "wachstum_pa": float((eq.iloc[-1] / capital) ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0,
                    "sharpe": sharpe_of(r), "max_rueckgang": float((eq / eq.cummax() - 1).min()),
                    "tiefster_stand": float(eq.min()), "schlechtester_tag": float(r.min()),
                    "trades": opened, "gewinner": int((tr_df["ergebnis_usd"] > 0).sum()),
                    "orders": count["orders"], "zu_klein": count["zu_klein"],
                    "margin_abgelehnt": count["margin_abgelehnt"], "nicht_ausfuehrbar": count["nicht_ausfuehrbar"],
                    "margin_calls": len(margin_calls),
                    "totalverlust": "" if ruin_day is None else f"{ruin_day:%d.%m.%Y}",
                    "schliessgrund": close_reason,
                    "gebuehren_usd": cost["gebuehren"], "slippage_usd": cost["slippage"],
                    "leihgebuehr_usd": cost["leihe"], "kreditzinsen_usd": cost["zinsen"],
                    "kosten_je_order_bp": float((cost["gebuehren"] + cost["slippage"]) / order_value * 1e4)
                    if order_value else float("nan"),
                    "hebel_mittel": float(np.mean(gross_ratio)), "hebel_max": float(np.max(gross_ratio)),
                    "ungleichgewicht_mittel": imbalance}}


def usable_pairs(px: pd.DataFrame) -> list[tuple[str, str]]:
    """Paare, fuer die beide Aktien Kurse haben."""
    ok = []
    for a, b in PAIRS:
        if a in px.columns and b in px.columns and px[[a, b]].dropna().shape[0] > LOOKBACK + 20:
            ok.append((a, b))
        else:
            log(f"Paar {pair_name(a, b)} ausgelassen: keine oder zu wenige Kurse")
    if not ok:
        raise SystemExit("Fuer kein Paar sind Kurse vorhanden.")
    return ok


def check_settings() -> None:
    if EXECUTION not in ("close", "next_open", "next_close"):
        raise SystemExit('EXECUTION muss "close", "next_open" oder "next_close" sein')
    if EXIT_Z >= ENTRY_Z:
        raise SystemExit("EXIT_Z muss kleiner als ENTRY_Z sein (sonst wuerde sofort wieder ausgestiegen).")
    if STOP_Z is not None and STOP_Z <= ENTRY_Z:
        raise SystemExit("STOP_Z muss groesser als ENTRY_Z sein.")
    if len(set(PAIRS)) != len(PAIRS):
        raise SystemExit("PAIRS enthaelt ein Paar doppelt.")


# ------------------------------------------------------------------------------------------ Diagramm
def make_chart(curves: dict[str, pd.Series], labels: dict[str, str], test_start: pd.Timestamp) -> None:
    """Aufgelaufene Rendite ohne Hebel: Korb und jedes Paar als kleines Einzeldiagramm -> output/modell.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        log("Diagramm uebersprungen (matplotlib fehlt: pip install matplotlib)")
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
    fig.legend([Patch(color=band)], ["Testphase"], loc="upper right", frameon=False, fontsize=9, labelcolor=muted)
    fig.suptitle("Aufgelaufene Rendite nach Kosten  (erstes Diagramm: Konto, beginnt in jeder Phase neu; "
                 "Paare ohne Hebel; jedes Diagramm hat seine eigene Skala)", x=0.01, ha="left", fontsize=11.5, color=ink)
    import matplotlib.dates as mdates
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "modell.png"
    try:
        fig.savefig(path, dpi=130, facecolor=surface)
    except PermissionError:
        path = OUT_DIR / f"modell_{time.strftime('%H%M%S')}.png"
        fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    log(f"Diagramm gespeichert: {path.name}")


# ------------------------------------------------------------------------------------------ Rueckrechnung
def profiles(kelly_f: float) -> list[dict]:
    """Die drei Konto-Profile: ohne Hebel, privat (Reg T), institutionell (Viertel-Kelly o. ae., ohne Reg T)."""
    out = [{"name": "Ohne Hebel", "slug": "ohne_hebel", "lev": 1.0, "init": None, "maint": None, "stop": None,
            "daily": None, "note": "Hebel 1"},
           {"name": "Privat", "slug": "privat", "lev": float(RETAIL_LEVERAGE), "init": MARGIN_INITIAL,
            "maint": MARGIN_MAINTENANCE, "stop": None, "daily": None,
            "note": f"Zielhebel {RETAIL_LEVERAGE:g}, Margin-Regeln eines normalen Kontos"}]
    if kelly_f == kelly_f and kelly_f > 0:
        q = KELLY_FRACTION * kelly_f
        cap = INSTITUTIONAL_MAX_LEVERAGE
        lev = min(q, cap) if cap is not None else q
        out.append({"name": "Institutionell", "slug": "institutionell", "lev": float(lev),
                    "init": (1.0 / cap) if cap else None, "maint": None, "stop": INSTITUTIONAL_STOP,
                    "daily": INSTITUTIONAL_DAILY_LOSS,
                    "note": f"{KELLY_FRACTION:g} x Kelly-Hebel {kelly_f:.1f} = {q:.1f}"
                            + (f", begrenzt auf {cap:g}" if cap is not None and q > cap else "")})
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

    # ---- Signale und Renditen je Paar (ohne Hebel, ohne Konto): zeigt die Guete der Signale
    sigs, betas, raw, trades = {}, {}, {}, []
    pair_rows = []
    for a, b in pairs:
        name = pair_name(a, b)
        sigs[name], betas[name] = build_signals(px, a, b, train)
        row = {"paar": name}
        for ph, mask in phases.items():
            res = simulate(sigs[name][mask], betas[name], ENTRY_Z, EXIT_Z)
            tr = list_trades(res, sigs[name][mask], ph, a, b)
            raw[(name, ph)] = res
            trades.append(tr)
            k = stats(res, tr)
            p = ph.lower()
            row.update({f"sharpe_{p}": k["sharpe"], f"rendite_pa_{p}": k["rendite_pa"], f"trades_{p}": k["trades"],
                        f"gewinner_{p}": k["gewinner"], f"max_rueckgang_{p}": k["max_rueckgang"]})
        first = sigs[name]["z"].first_valid_index()
        row["z_ab"] = first.date() if first is not None else None
        row["hedge_ratio"] = betas[name]
        row["z_aktuell"] = float(sigs[name]["z"].iloc[-1])
        row["position"] = direction(raw[(name, "Test")]["position"].iloc[-1], a, b)
        want, pos = raw[(name, "Test")]["signal"].iloc[-1], raw[(name, "Test")]["position"].iloc[-1]
        row["offenes_signal"] = "" if want == pos else ("schliessen" if want == 0 else direction(want, a, b))
        pair_rows.append(row)
    all_trades = pd.concat(trades, ignore_index=True)
    port = {ph: combine([raw[(pair_name(a, b), ph)] for a, b in pairs]) for ph in phases}

    # ---- Kelly-Hebel des Korbs aus dem Training -> Zielhebel des Profils "Institutionell"
    k_train, k_test = kelly_leverage(port["Training"]), kelly_leverage(port["Test"])
    f = k_train["kelly_hebel"]
    profs = profiles(f)

    # ---- Konten: ganze Stuecke, Gebuehren, Regeln des jeweiligen Profils; jede Phase beginnt mit CAPITAL
    def account(ph: str, leverage: float, **kw) -> dict:
        m = phases[ph]
        return simulate_account(pairs, {n: sigs[n][m] for n in sigs}, betas,
                                {n: raw[(n, ph)]["position"].to_numpy() for n in sigs}, CAPITAL, leverage, ph, **kw)

    def run_profile(pf: dict, ph: str, **kw) -> dict:
        return account(ph, pf["lev"], margin_initial=pf["init"], margin_maintenance=pf["maint"],
                       stop_level=pf["stop"], daily_loss_limit=pf["daily"], **kw)

    acct = {(pf["slug"], ph): run_profile(pf, ph) for pf in profs for ph in phases}
    base = profs[0]["slug"]                                 # Konto ohne Hebel
    for r in pair_rows:                                     # Ergebnis je Paar im Konto ohne Hebel
        for ph in phases:
            t = acct[(base, ph)]["trades"]
            r[f"konto_usd_{ph.lower()}"] = float(t.loc[t["paar"] == r["paar"], "ergebnis_usd"].sum())
    pair_tab = pd.DataFrame(pair_rows)

    # ---- Dateien
    daily = pd.concat([pd.DataFrame({"phase": ph, "offene_paare": port[ph]["position"].astype(int),
                                     "korb_rendite_ohne_hebel": port[ph]["netto"],
                                     **{f"konto_usd_{pf['slug']}": acct[(pf["slug"], ph)]["equity"] for pf in profs},
                                     **{f"hebel_{pf['slug']}": acct[(pf["slug"], ph)]["gross_ratio"] for pf in profs}})
                       for ph in phases])
    for a, b in pairs:                                      # Tagesrendite jedes Paares ohne Hebel
        name = pair_name(a, b)
        daily[f"netto_{name}"] = pd.concat([raw[(name, ph)]["netto"] for ph in phases])
    save_csv(daily.rename_axis("datum"), "tageswerte.csv", index=True)
    save_csv(all_trades, "trades.csv")
    save_csv(pd.concat([acct[(pf["slug"], ph)]["trades"].assign(profil=pf["name"]) for pf in profs for ph in phases],
                       ignore_index=True), "konto_trades.csv")
    save_csv(pair_tab, "paare.csv")
    save_csv(pd.DataFrame([{"profil": pf["name"], "zielhebel": pf["lev"], "phase": ph, **acct[(pf["slug"], ph)]["kpi"]}
                           for pf in profs for ph in phases]), "kennzahlen.csv")
    calls = [dict(c, profil=pf["name"]) for pf in profs for ph in phases for c in acct[(pf["slug"], ph)]["margin_calls"]]
    if calls:
        save_csv(pd.DataFrame(calls), "margin_calls.csv")

    # ---- Ausgabe
    exits = [f"z-Score zurueck bei {EXIT_Z:g}"]
    if MAX_HOLD_DAYS is not None:
        exits.append(f"spaetestens nach {MAX_HOLD_DAYS} Handelstagen")
    if STOP_Z is not None:
        exits.append(f"Stopp bei |z| >= {STOP_Z:g}")
    print()
    print("=" * 110)
    print(f"MEAN-REVERSION-MODELL  -  Korb aus {len(pairs)} Paaren, gleiche Einstellungen fuer alle")
    print("=" * 110)
    print(f"Spread   : log(A) - Hedge Ratio * log(B), Hedge Ratio "
          + ("je Paar im Training geschaetzt" if HEDGE_RATIO == "training" else f"{float(HEDGE_RATIO):g}"))
    print(f"z-Score  : " + (f"gleitend ueber {LOOKBACK} Handelstage" if Z_MODE == "rolling"
                            else "Mittelwert und Std fest aus der Trainingsphase"))
    print(f"Einstieg : |z| > {ENTRY_Z:g}")
    print(f"Ausstieg : " + "; ".join(exits))
    print("Handel   : " + {"close": "zum Schlusskurs des Signaltags",
                           "next_open": "zum Eroeffnungskurs des naechsten Handelstags",
                           "next_close": "zum Schlusskurs des naechsten Handelstags"}[EXECUTION])
    print(f"Konto    : {usd(CAPITAL)} USD je Phase und Profil, ganze Stuecke, je Paar Zielhebel x Eigenkapital / {len(pairs)}")
    print(f"Gebuehren: {COMMISSION_PER_SHARE:g} USD je Stueck, mindestens {COMMISSION_MIN:.2f} USD und hoechstens "
          f"{COMMISSION_MAX_PCT:.0%} des Orderwerts je Order; Slippage {COST_BPS:g} Basispunkte")
    print(f"Finanz.  : Leihgebuehr {BORROW_FEE:.1%} p. a.; Kreditzins {FINANCING_RATE:.2%} p. a. auf "
          + ("den Teil der Kaufseite ueber dem Eigenkapital" if FINANCING_BASIS == "kredit" else "die gesamte Kaufseite"))
    print("Profile  :")
    for pf in profs:
        rules = []
        if pf["init"] is not None:
            rules.append(f"neue Position nur bis Hebel {1 / pf['init']:.0f}")
        if pf["maint"] is not None:
            rules.append(f"Zwangsschliessung bei Deckung unter {pf['maint']:.0%}")
        if pf["stop"] is not None:
            rules.append(f"Konto wird geschlossen bei {1 - pf['stop']:.0%} Gesamtverlust")
        if pf["daily"] is not None:
            rules.append(f"oder bei {pf['daily']:.0%} Verlust an einem Tag")
        if pf["slug"] == "institutionell" and pf["stop"] is None and pf["daily"] is None:
            rules.append("Konto wird geschlossen, wenn das eigene Kapital aufgebraucht ist")
        print(f"    {pf['name']:<15} Zielhebel {pf['lev']:>5.1f}   ({pf['note']}" + ("; " + "; ".join(rules) if rules else "") + ")")
    if len(profs) < 3:
        print("    Institutionell   entfaellt: Kelly-Hebel im Training nicht positiv")
    frozen = read_start()
    if frozen is not None:
        diff = settings_diff(frozen["einstellungen"], frozen_settings())
        print(f"Protokoll: laeuft seit {frozen['starttag']}"
              + ("" if not diff else "  -  ACHTUNG: aktuelle Einstellungen weichen vom Protokoll ab ("
                 + ", ".join(diff) + ")"))
    print()
    print("KONTEN  -  in USD, nach Gebuehren, Slippage, Leihgebuehr und Kreditzinsen")
    rows = [("Endkapital", lambda k: usd(k["endkapital"])),
            ("Wachstum p. a.", lambda k: f"{k['wachstum_pa']:+.1%}"),
            ("Sharpe Ratio", lambda k: f"{k['sharpe']:.2f}"),
            ("Groesster Rueckgang", lambda k: f"{k['max_rueckgang']:.1%}"),
            ("Schlechtester Tag", lambda k: f"{k['schlechtester_tag']:+.1%}"),
            ("Tiefster Kontostand", lambda k: usd(k["tiefster_stand"])),
            ("KONTO GESCHLOSSEN am", lambda k: k["totalverlust"] or "nein"),
            ("  Grund", lambda k: k["schliessgrund"][:15] or "-"),
            ("Margin Calls", lambda k: f"{k['margin_calls']}"),
            ("Hebel erreicht (Mittel / Max.)", lambda k: f"{k['hebel_mittel']:.1f} / {k['hebel_max']:.1f}"),
            ("Trades (davon Gewinner)", lambda k: f"{k['trades']} ({k['gewinner']})"),
            ("  abgelehnt: Hebelgrenze", lambda k: f"{k['margin_abgelehnt']}"),
            ("  ausgefallen: Betrag < 1 Stueck", lambda k: f"{k['zu_klein']}"),
            ("Gebuehren", lambda k: f"-{usd(k['gebuehren_usd'])}"),
            ("Slippage", lambda k: f"-{usd(k['slippage_usd'])}"),
            ("Leihgebuehr", lambda k: f"-{usd(k['leihgebuehr_usd'])}"),
            ("Kreditzinsen", lambda k: f"-{usd(k['kreditzinsen_usd'])}"),
            ("Kosten je Order", lambda k: f"{k['kosten_je_order_bp']:.1f} bp")]
    w = 16
    for ph in phases:
        k0 = acct[(base, ph)]["kpi"]
        print(f"\n{ph.upper()}  {k0['von']:%d.%m.%Y} - {k0['bis']:%d.%m.%Y}, Startkapital {usd(CAPITAL)} USD")
        print(f"{'':<34}" + "".join(f"{pf['name']:>{w}}" for pf in profs))
        print(f"{'Zielhebel':<34}" + "".join(f"{pf['lev']:>{w}.1f}" for pf in profs))
        for label, fmt in rows:
            print(f"{label:<34}" + "".join(f"{fmt(acct[(pf['slug'], ph)]['kpi']):>{w}}" for pf in profs))
    for c in calls:
        print(f"Margin Call am {c['datum']:%d.%m.%Y} ({c['profil']}, {c['phase']}): Eigenkapital {usd(c['eigenkapital'])} USD "
              f"deckte nur noch {c['deckung']:.0%} der offenen Positionen - alles wurde zwangsweise geschlossen.")

    print()
    print("EINZELNE PAARE  -  Signale ohne Hebel, Rendite bezogen auf das Kapital des Paares; rechts das Ergebnis im Konto ohne Hebel")
    print(f"{'Paar':<12}{'Sharpe':>8}{'Sharpe':>8}{'Rendite':>9}{'Rendite':>9}{'Trades':>8}{'Trades':>7}{'Konto USD':>11}"
          f"{'Konto USD':>11}{'z heute':>9}  Position / offenes Signal")
    print(f"{'':<12}{'Train.':>8}{'Test':>8}{'Train.':>9}{'Test':>9}{'Train.':>8}{'Test':>7}{'Train.':>11}{'Test':>11}")
    for r in pair_rows:
        state = r["position"] + (f"  ->  {r['offenes_signal']}" if r["offenes_signal"] else "")
        print(f"{r['paar']:<12}{r['sharpe_training']:>8.2f}{r['sharpe_test']:>8.2f}{r['rendite_pa_training']:>+9.1%}"
              f"{r['rendite_pa_test']:>+9.1%}{r['trades_training']:>8}{r['trades_test']:>7}"
              f"{usd(r['konto_usd_training']):>11}{usd(r['konto_usd_test']):>11}{r['z_aktuell']:>+9.2f}  {state}")
    late = [r["paar"] for r in pair_rows if r["z_ab"] is not None and pd.Timestamp(r["z_ab"]) > idx[train][0]]
    if late:
        print(f"Kuerzere Historie (z-Score erst nach Beginn des Trainings verfuegbar): {', '.join(late)}")
    multi = [t for t in tickers() if sum(t in p for p in pairs) > 1]
    if multi:
        print(f"In mehreren Paaren enthalten: {', '.join(multi)} - diese Paare sind nicht unabhaengig voneinander.")
    print(f"\nStand {last:%d.%m.%Y}. 'Offenes Signal' = Order fuer "
          + {"close": "den naechsten Schlusskurs", "next_open": "die naechste Eroeffnung",
             "next_close": "den naechsten Schlusskurs"}[EXECUTION] + ".")
    print("\nHinweis: Die Paare wurden ausgewaehlt, weil sie von 2019 bis 2026 am besten liefen - die Testphase war\n"
          "dabei bekannt. TEST ist deshalb zu guenstig und kein Beleg. Wer ausserdem die Einstellungen so lange\n"
          "aendert, bis TEST gut aussieht, hat keinen Test mehr. Wirklich neue Daten liefert nur das Protokoll\n"
          "(--protokoll).")

    # ---- Kostenstaffel: wie viel Slippage vertraegt das Modell?
    print()
    print("-" * 110)
    print("KOSTENSTAFFEL  -  Wachstum p. a. je Profil bei unterschiedlicher Slippage je Order (Gebuehren wie oben)")
    print("-" * 110)
    print(f"{'Slippage':>10} |" + " |".join("".join(f"{pf['name']:>{w}}" for pf in profs) for _ in phases))
    print(f"{'':>10} |" + " |".join(f"{ph.upper():^{w * len(profs)}}" for ph in phases))
    ladder_rows = []
    for bp in COST_LADDER_BPS:
        ks = {(pf["slug"], ph): (acct[(pf["slug"], ph)]["kpi"] if bp == COST_BPS
                                 else run_profile(pf, ph, slippage_bps=bp)["kpi"]) for pf in profs for ph in phases}
        print(f"{format(bp, 'g') + ' bp':>10} |" + " |".join(
            "".join(f"{ks[(pf['slug'], ph)]['wachstum_pa']:>+{w}.1%}" for pf in profs) for ph in phases)
            + ("   <- Annahme" if bp == COST_BPS else ""))
        ladder_rows.append({"slippage_bp": bp, **{f"{ph.lower()}_{pf['slug']}_{k}": ks[(pf["slug"], ph)][k]
                                                  for ph in phases for pf in profs
                                                  for k in ("endkapital", "wachstum_pa", "sharpe")}})
    for ph in phases:                                       # Slippage, bei der vom Gewinn nichts uebrig bliebe
        a0 = run_profile(profs[0], ph, slippage_bps=0)
        gain, vol = a0["kpi"]["endkapital"] - CAPITAL, a0["order_value"]
        if vol > 0 and gain > 0:
            print(f"{ph} (ohne Hebel): Der Gewinn waere aufgezehrt bei etwa {gain / vol * 1e4:.0f} Basispunkten Slippage je "
                  f"Order (Naeherung: Gewinn ohne Slippage / Orderwert).")
    save_csv(pd.DataFrame(ladder_rows), "kostenstaffel.csv")

    # ---- Kelly-Formel und Stresstest: waere mit mehr Hebel ein Totalverlust eingetreten?
    print()
    print("-" * 110)
    print("KELLY-FORMEL UND HEBEL-STRESSTEST")
    print("-" * 110)
    print(f"Kelly-Formel (Training, Korb ohne Hebel): Rendite {k_train['rendite_pa']:+.1%} p. a., abzueglich Finanzierung je "
          f"Hebel-Einheit {k_train['finanzierung_je_hebel_pa']:.1%},\nVolatilitaet {k_train['volatilitaet_pa']:.1%} p. a.  ->  "
          f"Kelly-Hebel {f:.1f}" + (f"; Profil Institutionell: {profs[-1]['note']} -> Zielhebel {profs[-1]['lev']:.1f}"
                                     if len(profs) == 3 else "  (nicht positiv: nach Kelly nicht handeln)"))
    print("\nStresstest: dasselbe Konto mit verschiedenen Zielhebeln, ohne Hebelgrenze und ohne Zwangsschliessung.\n"
          "Totalverlust = Eigenkapital aufgebraucht, das Modell faellt aus.")
    print(f"\n{'Hebel':>7} |{'Endkapital':>13}{'max. Rueckg.':>14}{'tiefster Stand':>16}{'Totalverlust':>15} |"
          f"{'Endkapital':>13}{'max. Rueckg.':>14}{'tiefster Stand':>16}{'Totalverlust':>15}")
    print(f"{'':>7} |{'TRAINING':^58} |{'TEST':^58}")
    stress_rows = []
    marks = {round(pf["lev"], 6): pf["name"] for pf in profs}
    levels = sorted(set([float(x) for x in STRESS_LEVERAGE] + [float(pf["lev"]) for pf in profs]))
    for lv in levels:
        ks = {ph: account(ph, lv, margin_rules=False)["kpi"] for ph in phases}
        print(f"{lv:>7.1f} |" + " |".join(
            f"{usd(ks[ph]['endkapital']):>13}{ks[ph]['max_rueckgang']:>14.0%}{usd(ks[ph]['tiefster_stand']):>16}"
            f"{(ks[ph]['totalverlust'] or 'nein'):>15}" for ph in phases)
            + (f"   <- {marks[round(lv, 6)]}" if round(lv, 6) in marks else ""))
        stress_rows.append({"hebel": lv, **{f"{ph.lower()}_{k}": ks[ph][k] for ph in phases
                                            for k in ("endkapital", "max_rueckgang", "tiefster_stand", "totalverlust")}})
    save_csv(pd.DataFrame(stress_rows), "hebel_stresstest.csv")
    print("\nLesehilfe: Kein Totalverlust in diesen vier Jahren heisst nicht, dass der Hebel sicher ist. Die Paare sind im\n"
          "Rueckblick ausgewaehlt, und der schlimmste Tag der Zukunft steht nicht in diesen Daten.")

    if show_grid:
        print()
        print("-" * 100)
        print("RASTER - nur Trainingsphase, ganzer Korb, ohne Hebel und ohne Konto: Sharpe Ratio (in Klammern: Trades)")
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
                cells.append(f"{sharpe_of(pt['netto'].to_numpy()):>8.2f} ({n_tr:>4})")
                grid_rows.append({"entry_z": en, "exit_z": ex, "sharpe_training": sharpe_of(pt["netto"].to_numpy()),
                                  "rendite_pa_training": float(pt["netto"].mean() * 252), "trades_training": n_tr})
            print(f"{'ENTRY_Z ' + format(en, 'g'):>13}" + "".join(cells))
        save_csv(pd.DataFrame(grid_rows), "raster_training.csv")
        print("Das Raster gilt fuer alle Paare gemeinsam. Benachbarte Felder sollten aehnlich aussehen; ein einzelner\n"
              "Spitzenwert zwischen schwachen Nachbarn ist meist Zufall. Waehrend das Protokoll laeuft, die\n"
              "Einstellungen nicht mehr aendern.")

    # ---- Diagramm: Konto ohne Hebel und Paare
    eq_all = pd.concat([acct[(base, ph)]["equity"] / CAPITAL - 1 for ph in phases])
    first = f"KONTO ohne Hebel ({usd(CAPITAL)} USD je Phase)"
    curves = {first: eq_all}
    labels = {first: f"Sharpe Training {acct[(base, 'Training')]['kpi']['sharpe']:.2f}, "
                     f"Test {acct[(base, 'Test')]['kpi']['sharpe']:.2f}"}
    for r in pair_rows:
        sr = pd.concat([raw[(r["paar"], ph)]["netto"] for ph in phases])
        curves[r["paar"]] = np.cumprod(1 + sr) - 1
        labels[r["paar"]] = f"Sharpe Training {r['sharpe_training']:.2f}, Test {r['sharpe_test']:.2f}"
    make_chart(curves, labels, test_start)
    print(f"\nErgebnisse: {OUT_DIR}  (kennzahlen.csv, paare.csv, konto_trades.csv, trades.csv, tageswerte.csv,\n"
          f"            kostenstaffel.csv, hebel_stresstest.csv, modell.png)")


# ------------------------------------------------------------------------------------------ Protokoll
def frozen_settings() -> dict:
    """Alles, was die Signale bestimmt. Diese Werte duerfen sich waehrend des Protokolls nicht aendern."""
    s = {"PAIRS": [list(p) for p in PAIRS], "Z_MODE": Z_MODE, "LOOKBACK": LOOKBACK, "ENTRY_Z": ENTRY_Z,
         "EXIT_Z": EXIT_Z, "MAX_HOLD_DAYS": MAX_HOLD_DAYS, "STOP_Z": STOP_Z, "HEDGE_RATIO": HEDGE_RATIO,
         "EXECUTION": EXECUTION}
    if Z_MODE == "training" or HEDGE_RATIO == "training":   # nur dann haengen die Signale vom Trainingszeitraum ab
        s["TRAIN_YEARS"] = TRAIN_YEARS
    return s


def settings_diff(old: dict, new: dict) -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k):
            if k == "PAIRS":
                out.append("PAIRS (Paarliste)")
            else:
                out.append(f"{k}: {old.get(k)} -> {new.get(k)}")
    return out


def read_start() -> dict | None:
    path = PROT_DIR / "start.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dates(col: pd.Series) -> pd.Series:
    """Datumsspalte lesen, auch wenn Excel sie beim Speichern in 05.10.2026 umgeschrieben hat."""
    s = col.astype(str).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    de = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(de)


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def to_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def order_text(want: float, pos: float, a: str, b: str) -> str:
    """Order, die nach dem Schlusskurs eines Tages fuer die naechste Ausfuehrung ansteht."""
    if want == pos:
        return "halten" if pos != 0 else ""
    if want == 0:
        return "schliessen"
    return f"{a} kaufen / {b} leerverkaufen" if want > 0 else f"{a} leerverkaufen / {b} kaufen"


def run_protocol(px: pd.DataFrame, restart: bool) -> None:
    idx = px.index
    last = idx[-1]
    now = time.strftime("%Y-%m-%d %H:%M")

    # ---- Start: Einstellungen einfrieren oder mit dem eingefrorenen Stand vergleichen
    if restart and PROT_DIR.exists():
        target = BASE_DIR / f"protokoll_alt_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(PROT_DIR), str(target))
        log(f"Bisheriges Protokoll archiviert in {target.name}")
    start = read_start()
    if start is None:
        PROT_DIR.mkdir(exist_ok=True)
        start = {"starttag": f"{last:%Y-%m-%d}", "angelegt_am": now, "einstellungen": frozen_settings(),
                 "annahmen_zur_information": {"COST_BPS": COST_BPS, "BORROW_FEE": BORROW_FEE},
                 "hinweis": "Starttag = letzter Kurstag beim Anlegen. Gewertet werden nur Signale danach."}
        (PROT_DIR / "start.json").write_text(json.dumps(start, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Protokoll angelegt. Starttag {last:%d.%m.%Y}, Einstellungen eingefroren in protokoll/start.json")
    else:
        diff = settings_diff(start["einstellungen"], frozen_settings())
        if diff:
            raise SystemExit(
                "Die Einstellungen im Skript weichen vom eingefrorenen Stand des Protokolls ab:\n    "
                + "\n    ".join(diff)
                + "\nEntweder die Einstellungen zuruecksetzen (siehe protokoll/start.json) oder mit\n"
                  "    python pair_modell.py --protokoll-neu\nein neues Protokoll beginnen. Das alte wird archiviert, "
                  "die bisherigen Ergebnisse zaehlen dann nicht mehr.")
    start_day = pd.Timestamp(start["starttag"])
    train = np.asarray((idx > start_day - pd.DateOffset(months=round(12 * TRAIN_YEARS))) & (idx <= start_day))
    period = np.asarray(idx > start_day)
    pairs = usable_pairs(px)
    n_days = int(period.sum())
    lag = 0 if EXECUTION == "close" else 1
    exec_label = {"close": "Schlusskurs", "next_open": "Eroeffnung", "next_close": "Schlusskurs"}[EXECUTION]
    next_label = {"close": "zum naechsten Schlusskurs", "next_open": "zur naechsten Eroeffnung",
                  "next_close": "zum naechsten Schlusskurs"}[EXECUTION]

    # ---- jedes Paar ab dem Starttag handeln (ohne Hebel)
    sigs, betas, results, trades, log_rows, order_rows, pair_rows = {}, {}, {}, [], [], [], []
    for a, b in pairs:
        name = pair_name(a, b)
        sig, beta = build_signals(px, a, b, train)
        sigs[name], betas[name] = sig, beta
        z_start = sig["z"].loc[:start_day].dropna()
        z_now = float(sig["z"].iloc[-1])
        if n_days == 0:
            waiting = bool(len(z_start)) and abs(z_start.iloc[-1]) >= ENTRY_Z
            pair_rows.append({"paar": name, "z_aktuell": z_now, "position": "keine Position",
                              "order": "wartet auf Rueckkehr in den Normalbereich" if waiting else "",
                              "trades": 0, "offen": 0, "rendite_seit_start": 0.0})
            continue
        s = sig[period]
        res = simulate(s, beta, ENTRY_Z, EXIT_Z, start_blocked=True, close_at_end=False)
        tr = list_trades(res, s, "Protokoll", a, b)
        results[name] = res
        trades.append(tr)
        pos, want = res["position"].to_numpy(), res["signal"].to_numpy()
        ever_inside = np.maximum.accumulate((s["z"].abs() < ENTRY_Z).fillna(False).to_numpy())
        for t in range(len(s)):
            day = s.index[t]
            order = order_text(want[t], pos[t], a, b)
            if not order and want[t] == 0 and not ever_inside[t] and abs(s["z"].iloc[t]) >= ENTRY_Z:
                order = "wartet auf Rueckkehr in den Normalbereich"
            log_rows.append({"protokolliert_am": now, "kursdatum": f"{day:%Y-%m-%d}",
                             "nachgetragen": "ja" if day < last else "nein", "paar": name,
                             "kurs_a": round(float(s["kurs_a"].iloc[t]), 4), "kurs_b": round(float(s["kurs_b"].iloc[t]), 4),
                             "z": round(float(s["z"].iloc[t]), 3),
                             "position": direction(pos[t], a, b), "order_nach_schluss": order,
                             "tagesrendite": round(float(res["netto"].iloc[t]) + 0.0, 6)})
        order_now = order_text(want[-1], pos[-1], a, b)
        if not order_now and not ever_inside[-1] and abs(z_now) >= ENTRY_Z:
            order_now = "wartet auf Rueckkehr in den Normalbereich"
        closed = tr[tr["grund"] != "offen"]
        pair_rows.append({"paar": name, "z_aktuell": z_now, "position": direction(pos[-1], a, b), "order": order_now,
                          "want": float(want[-1]), "pos": float(pos[-1]),
                          "trades": len(closed), "gewinner": int((closed["ergebnis"] > 0).sum()),
                          "offen": int(pos[-1] != 0), "rendite_seit_start": float(np.prod(1 + res["netto"]) - 1),
                          "sharpe": sharpe_of(res["netto"].to_numpy()) if n_days >= MIN_DAYS_FOR_SHARPE else float("nan"),
                          "schlechtester_trade": float(tr["ergebnis"].min()) if len(tr) else float("nan")})

    # ---- Konto seit dem Starttag: CAPITAL, ohne Hebel, ganze Stuecke, Gebuehren des Brokers
    fills_path = PROT_DIR / "ausfuehrungen.csv"
    cols = ["datum", "paar", "aktie", "order", "seite", "stueck", "ausfuehrung", "modellkurs", "realer_kurs",
            "abweichung_bp", "ausfuehrbar"]
    old_fills = None
    skip: set = set()
    if fills_path.exists():
        old_fills = read_semicolon_csv(fills_path)
        old_fills["datum"] = parse_dates(old_fills["datum"]).dt.strftime("%Y-%m-%d")
        for c in cols:                                      # aeltere Dateien haben noch nicht alle Spalten
            if c not in old_fills.columns:
                old_fills[c] = "ja" if c == "ausfuehrbar" else ""
        no = old_fills["ausfuehrbar"].astype(str).str.strip().str.lower().str.startswith("n")
        entry = old_fills["order"].isin(["kaufen", "leerverkaufen"])
        skip = {(p_, pd.Timestamp(d).date()) for p_, d in zip(old_fills.loc[no & entry, "paar"],
                                                              old_fills.loc[no & entry, "datum"])}
    acct = acct_clean = None
    equity_now = float(CAPITAL)
    if n_days:
        def account(lev: float = 1.0, margin_rules: bool = False, **kw) -> dict:
            return simulate_account(pairs, {n: sigs[n][period] for n in sigs}, betas,
                                    {n: results[n]["position"].to_numpy() for n in sigs}, CAPITAL, lev, "Protokoll",
                                    margin_rules=margin_rules, close_at_end=False, **kw)
        acct = account()
        equity_now = float(acct["equity"].iloc[-1])
        if skip:
            acct_clean = account(skip=skip)
        for o in acct["orders"]:
            order_rows.append({"datum": f"{o['datum']:%Y-%m-%d}", "paar": o["paar"], "aktie": o["aktie"],
                               "order": o["order"], "seite": o["seite"], "stueck": o["stueck"],
                               "ausfuehrung": exec_label, "modellkurs": round(float(o["modellkurs"]), 4),
                               "realer_kurs": "", "abweichung_bp": "", "ausfuehrbar": "ja"})
    # Orders fuer die naechste Ausfuehrung mit Stueckzahlen (Eigenkapital und Schlusskurse von heute)
    for r, (a, b) in zip(pair_rows, pairs):
        want_, pos_ = r.get("want", 0.0), r.get("pos", 0.0)
        if want_ == pos_:
            continue
        name = r["paar"]
        if want_ != 0:
            w_a, w_b = 1 / (1 + abs(betas[name])), abs(betas[name]) / (1 + abs(betas[name]))
            budget = equity_now / len(pairs)
            na = round(w_a * budget / float(sigs[name]["kurs_a"].iloc[-1]))
            nb = round(w_b * budget / float(sigs[name]["kurs_b"].iloc[-1]))
            if na < 1 or nb < 1:
                r["order"] = "faellt aus (Betrag reicht nicht fuer ein Stueck)"
            elif want_ > 0:
                r["order"] = f"{a} {na} Stueck kaufen / {b} {nb} Stueck leerverkaufen"
            else:
                r["order"] = f"{a} {na} Stueck leerverkaufen / {b} {nb} Stueck kaufen"
        elif acct is not None:
            ha, hb = acct["holdings"][name]
            if ha == 0 and hb == 0:
                r["order"] = ""                             # im Konto gar nicht eroeffnet
            else:
                r["order"] = (f"schliessen: {a} {abs(ha)} Stueck {'verkaufen' if ha > 0 else 'zurueckkaufen'} / "
                              f"{b} {abs(hb)} Stueck {'verkaufen' if hb > 0 else 'zurueckkaufen'}")

    # ---- Hebel-Varianten: feste Hebel und der Kelly-Hebel aus den Jahren VOR dem Starttag.
    # Der Kelly-Hebel wird einmal berechnet und in hebel.json festgehalten, damit er sich spaeter nicht aendert.
    lev_path = PROT_DIR / "hebel.json"
    if lev_path.exists():
        kelly_info = json.loads(lev_path.read_text(encoding="utf-8"))
    else:
        kk = kelly_leverage(combine([simulate(sigs[n][train], betas[n], ENTRY_Z, EXIT_Z) for n in sigs]))
        t_idx = idx[train]
        kelly_info = {"kelly_hebel": kk["kelly_hebel"], "berechnet_am": now,
                      "zeitraum": f"{t_idx[0]:%Y-%m-%d} bis {t_idx[-1]:%Y-%m-%d}",
                      "rendite_pa_ohne_hebel": kk["rendite_pa"], "volatilitaet_pa": kk["volatilitaet_pa"],
                      "finanzierung_je_hebel_pa": kk["finanzierung_je_hebel_pa"],
                      "hinweis": "Kelly-Hebel des Korbs aus den Jahren vor dem Starttag; wird nicht neu berechnet."}
        lev_path.write_text(json.dumps(kelly_info, indent=2, ensure_ascii=False), encoding="utf-8")
    kelly_f = float(kelly_info["kelly_hebel"])
    profs = profiles(kelly_f)
    lev_runs = []
    if n_days:
        for pf in profs:
            lev_runs.append((pf, account(lev=pf["lev"], margin_rules=True, margin_initial=pf["init"],
                                         margin_maintenance=pf["maint"], stop_level=pf["stop"],
                                         daily_loss_limit=pf["daily"])))

    # ---- protokoll.csv: nur neue Zeilen anhaengen, bestehende nie aendern
    log_path = PROT_DIR / "protokoll.csv"
    new_log = pd.DataFrame(log_rows)
    n_new, added_late = 0, False
    if len(new_log):
        known: set[tuple[str, str]] = set()
        if log_path.exists():
            old = read_semicolon_csv(log_path)
            known = set(zip(parse_dates(old["kursdatum"]).dt.strftime("%Y-%m-%d"), old["paar"]))
        add = new_log[[(d, p) not in known for d, p in zip(new_log["kursdatum"], new_log["paar"])]]
        n_new = len(add)
        added_late = bool((add["nachgetragen"] == "ja").any())
        if n_new:
            try:
                add.to_csv(log_path, sep=";", decimal=",", index=False, encoding="utf-8-sig" if not log_path.exists()
                           else "utf-8", mode="a", header=not log_path.exists())
            except PermissionError:
                n_new = 0
                log("protokoll.csv ist gesperrt (in Excel offen?) - die heutigen Zeilen werden beim naechsten Lauf "
                    "nachgetragen")

    # ---- ausfuehrungen.csv: neue Orders ergaenzen, eingetragene reale Kurse behalten, Abweichung berechnen
    fills = pd.DataFrame(order_rows, columns=cols)
    if old_fills is not None:
        old = old_fills
        old["modellkurs"] = to_float(old["modellkurs"])
        key_old = set(zip(old["datum"], old["paar"], old["aktie"]))
        fills = fills[[(d, p, t) not in key_old for d, p, t in zip(fills["datum"], fills["paar"], fills["aktie"])]]
        fills = pd.concat([old[cols], fills], ignore_index=True)
    real, model = to_float(fills["realer_kurs"]), to_float(fills["modellkurs"])
    # positiv = schlechter als das Modell: teurer gekauft oder billiger verkauft
    dev = np.where(fills["seite"] == "kauf", real / model - 1, 1 - real / model) * 1e4
    fills["abweichung_bp"] = pd.Series(dev, index=fills.index).round(1)
    fills["realer_kurs"] = real
    fills["modellkurs"] = model
    fills = fills.sort_values(["datum", "paar", "aktie"]).reset_index(drop=True)
    if len(fills) or not fills_path.exists():
        try:
            fills.to_csv(fills_path, sep=";", decimal=",", index=False, encoding="utf-8-sig")
        except PermissionError:
            log("ausfuehrungen.csv ist gesperrt (in Excel offen?) - neue Orders werden beim naechsten Lauf ergaenzt")
    measured = fills["abweichung_bp"].dropna()

    # ---- Ergebnisdateien
    pair_tab = pd.DataFrame(pair_rows).drop(columns=["want", "pos"], errors="ignore")
    save_csv(pair_tab, "ergebnisse_paare.csv", folder=PROT_DIR)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    if n_days:
        port = combine([results[pair_name(a, b)] for a, b in pairs])
        daily = pd.DataFrame({"offene_paare": port["position"].astype(int), "konto_usd": acct["equity"].round(2),
                              "konto_rendite_seit_start": acct["equity"] / CAPITAL - 1,
                              "korb_rendite": port["netto"],
                              "korb_aufgelaufen": np.cumprod(1 + port["netto"]) - 1})
        for pf, run in lev_runs:                            # Kontostand je Profil
            daily[f"konto_usd_{pf['slug']}"] = run["equity"].round(2)
        for a, b in pairs:
            name = pair_name(a, b)
            daily[f"aufgelaufen_{name}"] = np.cumprod(1 + results[name]["netto"]) - 1
        save_csv(daily.rename_axis("datum"), "ergebnisse.csv", index=True, folder=PROT_DIR)
        save_csv(pd.DataFrame([{"profil": pf["name"], "zielhebel": pf["lev"], "herleitung": pf["note"],
                                **{k: run["kpi"][k] for k in
                                   ("endkapital", "gesamtrendite", "wachstum_pa", "sharpe", "max_rueckgang",
                                    "schlechtester_tag", "tiefster_stand", "hebel_mittel", "hebel_max", "trades",
                                    "gewinner", "margin_abgelehnt", "zu_klein", "margin_calls", "totalverlust",
                                    "schliessgrund",
                                    "gebuehren_usd", "slippage_usd", "leihgebuehr_usd", "kreditzinsen_usd")}}
                               for pf, run in lev_runs]), "ergebnisse_hebel.csv", folder=PROT_DIR)
        save_csv(all_trades, "trades.csv", folder=PROT_DIR)
        save_csv(acct["trades"], "konto_trades.csv", folder=PROT_DIR)

    # ---- Ausgabe
    print()
    print("=" * 96)
    print(f"PROTOKOLL (Paper Trading)  -  {len(pairs)} Paare, ohne Hebel, Konto {usd(CAPITAL)} USD  -  Kurse bis {last:%d.%m.%Y}")
    print("=" * 96)
    print(f"Starttag {start_day:%d.%m.%Y}, angelegt am {start['angelegt_am']}; Einstellungen unveraendert "
          f"(Einstieg |z| > {ENTRY_Z:g}, Ausstieg bei {EXIT_Z:g}, Handel: {exec_label})")
    orders = [r for r in pair_rows if r["order"] and r["order"] not in ("halten",)
              and not r["order"].startswith("wartet") and not r["order"].startswith("faellt aus")]
    print(f"\nORDERS {next_label.upper()}:")
    if orders:
        for r in orders:
            print(f"    {r['paar']:<12} {r['order']:<62} (z-Score {r['z_aktuell']:+.2f})")
    else:
        print("    keine")
    print(f"\n{'Paar':<12}{'z heute':>9}  {'Position':<26}{'Trades':>7}{'Gewinner':>9}{'seit Start':>12}  Anmerkung")
    for r in pair_rows:
        note = r["order"] if r["order"].startswith(("wartet", "faellt aus")) else ""
        print(f"{r['paar']:<12}{r['z_aktuell']:>+9.2f}  {r['position']:<26}{r['trades']:>7}{r.get('gewinner', 0):>9}"
              f"{r['rendite_seit_start']:>+12.1%}  {note}")
    print()
    if n_days:
        r = port["netto"].to_numpy()
        eq = np.cumprod(1 + r)
        closed = all_trades[all_trades["grund"] != "offen"] if len(all_trades) else all_trades
        print(f"KORB SEIT START ({n_days} Handelstage): Rendite {eq[-1] - 1:+.2%}, groesster Rueckgang "
              f"{(eq / np.maximum.accumulate(eq) - 1).min():.2%}, abgeschlossene Trades {len(closed)}"
              + (f" (davon {int((closed['ergebnis'] > 0).sum())} Gewinner)" if len(closed) else "")
              + f", offene Paare {int(port['position'].iloc[-1])}")
        if n_days >= MIN_DAYS_FOR_SHARPE:
            sh = sharpe_of(r)
            print(f"Sharpe Ratio seit Start {sh:.2f}  (Unsicherheit +/- {np.sqrt((1 + 0.5 * sh ** 2) / (n_days / 252)):.2f})")
        else:
            print(f"Sharpe Ratio: wird erst ab {MIN_DAYS_FOR_SHARPE} Handelstagen gezeigt (vorher ohne Aussagekraft).")
        k = acct["kpi"]
        print(f"\nKONTO SEIT START ({usd(CAPITAL)} USD, ganze Stuecke, Gebuehren des Brokers, {COST_BPS:g} bp Slippage):")
        print(f"    Stand {usd(k['endkapital'])} USD ({k['gesamtrendite']:+.2%}), groesster Rueckgang {k['max_rueckgang']:.2%}, "
              f"{k['orders']} Orders, Gebuehren {usd(k['gebuehren_usd'])} USD, Slippage {usd(k['slippage_usd'])} USD, "
              f"Leihgebuehr {usd(k['leihgebuehr_usd'])} USD")
        if k["zu_klein"]:
            print(f"    {k['zu_klein']} Trade(s) ausgefallen, weil der Betrag nicht fuer ein ganzes Stueck reichte.")
        if k["orders"]:
            parts = []
            for bp in COST_LADDER_BPS:
                kk = k if bp == COST_BPS else account(slippage_bps=bp)["kpi"]
                parts.append(f"{bp:g} bp: {kk['gesamtrendite']:+.2%}")
            print("    Kostenstaffel (Slippage je Order -> Rendite seit Start):  " + "   ".join(parts))
            a0 = account(slippage_bps=0)
            gain = a0["kpi"]["endkapital"] - CAPITAL
            if gain > 0 and a0["order_value"] > 0:
                print(f"    Der Gewinn waere aufgezehrt bei etwa {gain / a0['order_value'] * 1e4:.0f} Basispunkten Slippage je Order.")
        if acct_clean is not None:
            kc = acct_clean["kpi"]
            print(f"    Ohne die {kc['nicht_ausfuehrbar']} als nicht ausfuehrbar markierten Trades: Stand {usd(kc['endkapital'])} USD "
                  f"({kc['gesamtrendite']:+.2%})")
        # ---- Die drei Profile
        print(f"\nDREI PROFILE SEIT START  -  jeweils ein eigenes Konto mit {usd(CAPITAL)} USD, gleiche Signale, andere Stueckzahlen")
        w = 16
        print(f"{'':<34}" + "".join(f"{pf['name']:>{w}}" for pf, _ in lev_runs))
        print(f"{'Zielhebel':<34}" + "".join(f"{pf['lev']:>{w}.1f}" for pf, _ in lev_runs))
        for label, fmt in (("Kontostand USD", lambda k: usd(k["endkapital"])),
                           ("Rendite seit Start", lambda k: f"{k['gesamtrendite']:+.2%}"),
                           ("Groesster Rueckgang", lambda k: f"{k['max_rueckgang']:.1%}"),
                           ("Schlechtester Tag", lambda k: f"{k['schlechtester_tag']:+.1%}"),
                           ("Hebel erreicht (Mittel / Max.)", lambda k: f"{k['hebel_mittel']:.1f} / {k['hebel_max']:.1f}"),
                           ("Trades", lambda k: f"{k['trades']}"),
                           ("  abgelehnt: Hebelgrenze", lambda k: f"{k['margin_abgelehnt']}"),
                           ("Margin Calls", lambda k: f"{k['margin_calls']}"),
                           ("KONTO GESCHLOSSEN am", lambda k: k["totalverlust"] or "nein"),
                           ("  Grund", lambda k: k["schliessgrund"][:15] or "-"),
                           ("Kreditzinsen USD", lambda k: f"-{usd(k['kreditzinsen_usd'])}")):
            print(f"{label:<34}" + "".join(f"{fmt(run['kpi']):>{w}}" for _, run in lev_runs))
        if n_days >= MIN_DAYS_FOR_SHARPE:
            print(f"{'Sharpe Ratio':<34}" + "".join(f"{run['kpi']['sharpe']:>{w}.2f}" for _, run in lev_runs))
        for pf in profs:
            print(f"    {pf['name']}: {pf['note']}")
        print(f"    Kelly-Hebel {kelly_f:.1f} aus {kelly_info['zeitraum']} (vor dem Starttag, festgehalten in hebel.json).\n"
              "    Die Orders oben und ausfuehrungen.csv gelten fuer das Konto ohne Hebel.")
    else:
        print("Heute ist der Starttag: Es gibt noch keinen Protokolltag. Ab dem naechsten Handelstag wird gezaehlt.")
        print("Profile ab dem ersten Protokolltag: " + ", ".join(f"{pf['name']} (Zielhebel {pf['lev']:.1f})" for pf in profs)
              + f"  -  Kelly-Hebel {kelly_f:.1f} aus {kelly_info['zeitraum']}")
    if len(measured):
        print(f"Gemessene Abweichung vom Modellkurs: im Mittel {measured.mean():+.1f} Basispunkte je Order "
              f"({len(measured)} eingetragene Kurse; im Modell angenommen: {COST_BPS:g}). Positiv = schlechter als das Modell.")
    else:
        print("Noch keine realen Kurse eingetragen (protokoll/ausfuehrungen.csv, Spalte 'realer_kurs'; Trades, die im\n"
              "Demokonto nicht moeglich waren, dort in der Spalte 'ausfuehrbar' mit 'nein' kennzeichnen).")
    if n_new:
        print(f"\nProtokolliert: {n_new} neue Zeilen in protokoll.csv"
              + (" - darunter nachgetragene Tage (das Skript lief nicht an jedem Handelstag)" if added_late else ""))
    elif n_days:
        print("\nProtokolliert: nichts Neues (dieser Kurstag steht bereits im Protokoll).")
    print(f"Dateien: {PROT_DIR}  (protokoll.csv, ausfuehrungen.csv, ergebnisse.csv, ergebnisse_paare.csv,\n"
          f"         ergebnisse_hebel.csv, trades.csv, konto_trades.csv)")


# ------------------------------------------------------------------------------------------ Ablauf
def main() -> None:
    ap = argparse.ArgumentParser(description="Mean-Reversion-Modell fuer einen Korb von Aktienpaaren")
    ap.add_argument("--grid", action="store_true",
                    help="Sharpe Ratio des Korbs im Training fuer mehrere Einstiegs- und Ausstiegswerte zeigen")
    ap.add_argument("--protokoll", action="store_true",
                    help="Protokollmodus (Paper Trading): Tagesstand festhalten, Orders und Ergebnis seit Start")
    ap.add_argument("--protokoll-neu", dest="protokoll_neu", action="store_true",
                    help="Protokoll neu beginnen; das bisherige wird archiviert")
    ap.add_argument("--refresh", action="store_true", help="Kurse neu von Yahoo laden")
    ap.add_argument("--stichtag", default=None, help=argparse.SUPPRESS)     # nur zum Testen: Kurse bis zu diesem Tag
    args = ap.parse_args()
    check_settings()

    px = load_prices(args.refresh)
    px = px[px[tickers()].notna().any(axis=1)]
    if args.stichtag:
        px = px[px.index <= pd.Timestamp(args.stichtag)]
    if args.protokoll or args.protokoll_neu:
        run_protocol(px, args.protokoll_neu)
    else:
        run_backtest(px, args.grid)


if __name__ == "__main__":
    main()
