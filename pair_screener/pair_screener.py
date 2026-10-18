"""
pair_screener.py - findet Paare mit (potenzieller) Cointegration unter Xetra-Aktien (Standard),
                   unter Xetra-ETFs/ETCs (--etf) oder unter US-Aktien / US-ETFs (--us, --us --etf)

Der Screener beschraenkt sich auf das Auffinden von Paaren und gibt eine Rangliste aus.
Handelsregeln, Backtest, Sharpe Ratio und weitere Kennzahlen sind bewusst noch nicht enthalten.

Aufruf
------
    python pair_screener.py                       # Aktien: alle Paare innerhalb derselben Branche
    python pair_screener.py --pairs               # Aktien: nur die Paare aus pairs_aktien.csv
    python pair_screener.py --two-stage           # zweistufig (auch mit --pairs kombinierbar)
    python pair_screener.py --mean-reversion      # Zusatzauswertung (auch mit --pairs / --etf kombinierbar)
    python pair_screener.py --etf [...]           # dasselbe fuer ETFs/ETCs (Paarliste: pairs.csv)
    python pair_screener.py --us [...]            # dasselbe fuer in den USA gelistete Aktien
    python pair_screener.py --us --etf [...]      # dasselbe fuer in den USA gelistete ETFs
    python pair_screener.py --pairs meine_liste.csv
    python pair_screener.py --demo                # Probelauf mit synthetischen Daten, ohne Internet
    python pair_screener.py --refresh             # Cache ignorieren, alles neu laden

Ablauf im Aktien-Modus
----------------------
1. Universum:   alle auf Xetra handelbaren Aktien (Liste der Deutschen Boerse, Instrumententyp CS),
                ohne Sonderlinien wie zum Verkauf eingereichte Stuecke. Mit STOCK_SEGMENTS auf z. B.
                DAX / MDAX / SDAX einschraenkbar.
2. Kurse:       Tagesdaten (Adjusted Close) von Yahoo Finance, Ticker "<Xetra-Kuerzel>.DE".
3. Stammdaten:  Marktkapitalisierung, Sektor und Branche von Yahoo (output_aktien/aktien_details.csv).
                Groessenfilter: Marktkapitalisierung > MIN_MARKET_CAP_USD.
4. Zeitraum:    die letzten FORMATION_YEARS (3) Jahre bis zum letzten Kursdatum.
5. Bereinigung: Liquiditaetsfilter (Median-Umsatz auf Xetra >= STOCK_MIN_MEDIAN_TURNOVER_EUR, kaum Tage
                ohne Kursaenderung). Die Datenfehler-Erkennung ist bei Aktien aus (STOCK_DETECT_PRICE_ERRORS).
6. Paarbildung: nur Paare innerhalb derselben Branche (STOCK_GROUP_BY = "industry"; "sector" = groeber,
                "" = alle Kombinationen), Tagesrenditen-Korrelation >= STOCK_MIN_RET_CORR, Spread-Breite
                >= MIN_SPREAD_STD. Einen Filter 'gleicher Index' gibt es bei Aktien nicht: Stamm- und
                Vorzugsaktien desselben Unternehmens sind hier gerade erwuenscht.
7. Test:        Engle-Granger in beide Richtungen, Benjamini-Hochberg-Korrektur ueber alle Tests.

Ablauf im ETF-Modus (--etf)
---------------------------
Universum sind alle Xetra-ETFs und -ETCs ohne gehebelte / inverse Produkte; Groessenfilter ist das
Fondsvolumen (MIN_AUM_USD, ETCs ausgenommen). Statt nach Branchen wird ueber die Korrelation
vorgefiltert (MIN_RET_CORR). Zusaetzlich fliegen Paare mit (fast) gleichem Index raus: gleicher
Referenzindex (Yahoo / index_map.csv), Tracking Error relativ zur Volatilitaet unter MIN_REL_TE oder
identisches Fondsvolumen (= Anteilsklassen desselben Fonds) -> same_index_excluded.csv. Die Paare mit
fast gleichem Index, aber breitem Spread, werden separat getestet -> ranking_fast_gleich.csv.

US-Markt (--us, mit allen anderen Schaltern kombinierbar)
---------------------------------------------------------
Universum:   alle an Nasdaq, NYSE, NYSE American, NYSE Arca und Cboe gelisteten Aktien bzw. ETFs
             (Symbollisten von nasdaqtrader.com). Ohne Optionsscheine, Bezugsrechte, Vorzugsaktien,
             Units, SPACs, ETNs und Testsymbole; bei ETFs ohne gehebelte / inverse Produkte.
Vorauswahl:  Weil das rund 6.000 Aktien bzw. 4.000 ETFs sind, werden zuerst 3 Monate Kurse geladen
             und nur Titel mit genug Umsatz behalten (US_PRESCREEN_FACTOR x Mindestumsatz). Erst fuer
             diese wird die lange Historie geladen (cache/vorauswahl_us_*.csv).
Schwellen:   Median-Tagesumsatz >= US_STOCK_MIN_MEDIAN_TURNOVER_USD (Aktien) bzw.
             US_ETF_MIN_MEDIAN_TURNOVER_USD (ETFs); Marktkapitalisierung / Fondsvolumen wie oben.
             Mean Reversion rechnet mit US_MR_COST_BPS je Kauf / Verkauf.
Sonst gleich: Branchengruppen bei Aktien, Filter 'gleicher Index' bei ETFs, Tests, Ausgaben.
Paarlisten:  pairs_us_aktien.csv / pairs_us_etf.csv, Ticker in Yahoo-Schreibweise (BRK-B).
Hinweis:     US-ETFs koennen Privatanleger in der EU in der Regel nicht kaufen (fehlendes
             Basisinformationsblatt, PRIIPs); handelbar sind sie meist nur ueber Optionen / CFDs
             oder als professioneller Kunde. US-Aktien sind davon nicht betroffen.

Zweistufiger Modus (--two-stage)
--------------------------------
Stufe 1 sucht im frueheren Zeitraum (STAGE1_YEARS Jahre vor dem aktuellen Zeitraum) und waehlt die
besten Kandidaten aus (hoechstens STAGE1_TOP_N, p <= STAGE1_MAX_P). Stufe 2 testet nur diese Kandidaten
in den letzten FORMATION_YEARS Jahren. Weil die Daten der Stufe 2 nicht fuer die Auswahl verwendet
wurden und nur wenige Tests laufen, ist die Huerde niedriger und das Ergebnis belastbarer.

Zusatzauswertung Mean Reversion (--mean-reversion)
--------------------------------------------------
Prueft statt strenger Cointegration die schwaechere Frage: Kehrt der Spread eines Paares zu seinem
gleitenden Mittel der letzten Monate zurueck, und bleibt davon nach Handelskosten etwas uebrig?
Spread = log(A) - log(B) (gleicher Betrag long und short). Einstieg bei |z| > MR_ENTRY_Z gegenueber dem
MR_LOOKBACK-Tage-Mittel, Ausstieg am Mittel oder nach MR_MAX_HOLD Tagen. Kosten MR_COST_BPS je Kauf /
Verkauf, Leihgebuehr MR_BORROW_FEE. Rollierend: Je Testjahr werden die MR_TOP_N Paare mit dem besten
Ergebnis der MR_FORMATION_YEARS Vorjahre ausgewaehlt und im Folgejahr gehandelt (neue Daten).
Ausgabe: mr_jahre.csv (je Testjahr: alle Paare / Auswahl), mr_auswahl.csv (ausgewaehlte Paare je Jahr),
mr_paare.csv (Kennzahlen je Paar), mr_portfolio_renditen.csv (Tagesrenditen).
Entwicklung der besten Paare: Die MR_SHOW_TOP Paare mit der hoechsten Sharpe Ratio im Gesamtzeitraum
werden Jahr fuer Jahr gezeigt (Rendite und Sharpe je Testjahr, letzte 6 Monate, Sharpe 'frueher' gegen
'zuletzt' = letzte MR_TREND_YEARS Jahre) -> mr_top_paare.csv, mr_top_paare_verlauf.csv (aufgelaufene
Rendite je Tag) und mr_top_paare.png. Achtung: Diese Paare sind im Rueckblick ausgesucht. Dass sie gut
aussehen, ist deshalb kein Beleg; aussagekraeftig ist nur, ob sie sich zuletzt halten oder nachlassen.
Kursbrueche: Tagesbewegungen ueber Faktor MR_MAX_DAILY_MOVE gelten als Bruch in den Daten (z. B. neue
Aktien nach einer Insolvenz); die Historie davor wird nicht verwendet -> mr_kursbrueche.csv.
Gegenprobe (mr_top_paare_gegenprobe.csv): die besten Paare nur der frueheren Jahre und ihr Ergebnis in
den letzten MR_TREND_YEARS Jahren, die fuer diese Auswahl neue Daten sind.
Grenzen: nur heute noch gelistete Titel, heutige Marktkapitalisierung / Branche, Schlusskurse statt
Geld- und Briefkursen. Das Ergebnis ist deshalb eher zu guenstig als zu streng.

Ergebnisse (CSV mit ";" und Dezimalkomma fuer Excel)
----------------------------------------------------
    Ordner: ./output_aktien (Xetra-Aktien), ./output (Xetra-ETFs), ./output_us_aktien, ./output_us_etf
    ranking.csv / ranking_paarliste.csv   alle getesteten Paare, sortiert nach p-Wert
    cointegrated_pairs.csv                nur die nach Korrektur signifikanten Paare
    ranking_stufe1.csv, ranking_zweistufig.csv            zweistufiger Lauf
    aktien_details.csv / etf_details.csv  Stammdaten je Instrument
    data_errors.csv, narrow_spread_excluded.csv, universe_used.csv, missing_marktkap.csv / missing_aum.csv
    nur ETFs: same_index_excluded.csv, ranking_fast_gleich*.csv

Eigene Dateien neben dem Skript (optional)
------------------------------------------
    pairs_aktien.csv / pairs.csv   ticker_a;ticker_b;kategorie;begruendung   (Paarliste fuer --pairs)
    pairs_us_aktien.csv / pairs_us_etf.csv   dasselbe fuer --us
    universe_aktien.csv / universe.csv   ticker[;name]      (eigenes Universum statt Xetra-Liste)
    gruppen_aktien.csv             ticker;gruppe            (Branche manuell zuordnen, nur Aktien)
    aum_overrides.csv              ticker;aum_usd           (Fondsvolumen korrigieren, nur ETFs)
    index_map.csv                  ticker;index             (Referenzindex zuordnen, nur ETFs)

    pip install yfinance statsmodels joblib pandas numpy requests
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

# ================================ Konfiguration ================================
HISTORY_YEARS = 10          # so weit zurueck werden Kurse geladen (Cache fuer spaetere Tests);
                            # fuer die Suche werden nur die letzten FORMATION_YEARS genutzt
FORMATION_YEARS = 3         # Zeitraum, in dem die Cointegration getestet wird

# ---- Aktien-Modus (Standard). Der ETF/ETC-Modus laeuft mit --etf und nutzt die Einstellungen darunter.
MIN_MARKET_CAP_USD = 500e6  # Mindest-Marktkapitalisierung in USD
STOCK_MIN_MEDIAN_TURNOVER_EUR = 250_000   # Median-Tagesumsatz auf Xetra (Liquiditaet, Leerverkauf)
STOCK_MIN_RET_CORR = 0.30   # Vorfilter Korrelation der Tagesrenditen (Aktien korrelieren schwaecher als ETFs)
STOCK_GROUP_BY = "industry" # nur Paare innerhalb derselben Gruppe testen: "industry" (Branche, fein),
                            # "sector" (Sektor, grob) oder "" (alle Kombinationen)
STOCK_SEGMENTS = ()         # leer = alle Xetra-Aktien; z. B. ("DAX", "MDAX", "SDAX", "DEUTSCHLAND")
STOCK_DETECT_PRICE_ERRORS = False  # Datenfehler-Erkennung (Kurssprung gegenueber aehnlichen Titeln, der
                            # zurueckspringt). Bei Aktien aus: Spruenge nach Quartalszahlen sind dort echt.

# ---- ETF/ETC-Modus (--etf)
INCLUDE_ETC = True          # ETCs (Gold, Silber, Oel, Industriemetalle ...) ins Universum aufnehmen
MIN_AUM_USD = 100e6         # Mindest-Fondsvolumen in USD
AUM_FILTER_ETC = False      # AUM-Filter auch auf ETCs anwenden? (Yahoo kennt deren Volumen selten)
MAX_STALE_SHARE = 0.05      # max. Anteil Tage ohne Handel / ohne Kursaenderung auf Xetra
MIN_MEDIAN_TURNOVER_EUR = 100_000   # Median-Tagesumsatz auf Xetra; seltener gehandelte ETFs haben
                            # veraltete Schlusskurse -> kuenstliche 'Mean Reversion' im Spread
KEEP_UNKNOWN_TURNOVER = True  # ETFs behalten, fuer die Yahoo keine Umsaetze liefert
KEEP_MISSING_AUM = False    # ETFs ohne AUM-Angabe behalten? False = konservativ ausschliessen
MIN_COVERAGE = 0.95         # Mindestanteil Handelstage mit Kurs im Formationsfenster

MIN_RET_CORR = 0.70         # Vorfilter: Korrelation der Tagesrenditen in der Formation
MAX_RET_CORR = 0.995        # darueber = praktisch derselbe Index
MIN_REL_TE = 0.20           # Tracking Error / Volatilitaet (Monatsrenditen). Darunter bilden A und B
                            # (fast) denselben Index ab -> Paar wird ausgeschlossen. Relativ, damit
                            # schwankungsarme Anleihen-ETFs nicht faelschlich als 'gleich' gelten.
SAME_AUM_IS_SAME_FUND = True  # exakt gleiches Fondsvolumen bei Yahoo = Anteilsklassen desselben Fonds
MIN_SPREAD_STD = 0.01       # Mindestbreite des Spreads: Std von log(A) - beta*log(B) (0.01 = ca. 1 %).
                            # Schmalere Paare werden nicht getestet: fast identische Engagements,
                            # deren 'Mean Reversion' vor allem Kursrauschen ist. 0 = Filter aus.
                            # Gilt nur fuer 'alle Kombinationen', nicht fuer die eigene Paarliste (--pairs).
NEAR_SAME_MIN_SPREAD = 0.01 # Auswertung 'fast gleiche Indizes': Paare, die der Tracking-Error-Filter als
                            # 'gleicher Index' aussortiert, deren Spread aber mindestens so breit ist
                            # (z. B. zwei Goldminen- oder Halbleiter-Indizes). Eigene Rangliste.
JUMP_THRESHOLD = 0.08       # Datenfehler: Tagesrendite weicht um mehr als 8 % von aehnlichen ETFs ab
JUMP_REVERT_DAYS = 30       # ... und springt innerhalb von 30 Handelstagen in Gegenrichtung zurueck
PEER_MIN_CORR = 0.70        # 'aehnliche ETFs' = die 5 am staerksten korrelierten mit Korrelation >= 0,7
MAX_PAIRS_TO_TEST = 55_000  # Obergrenze getesteter Paare (hoechste Korrelation zuerst)
COINT_MAXLAG = 5            # max. Lags im ADF-Teil des Engle-Granger-Tests

FDR_ALPHA = 0.05            # Benjamini-Hochberg-Niveau (Multiple-Testing-Korrektur)
RANK_TOP_FOR_COUNT = 100    # Haeufigkeit je ETF wird unter den besten N Paaren der Rangliste gezaehlt

# Zweistufiger Modus (--two-stage): Stufe 1 waehlt Kandidaten im frueheren Zeitraum aus,
# Stufe 2 testet nur diese Kandidaten im spaeteren Zeitraum (= letzte FORMATION_YEARS Jahre).
STAGE1_YEARS = 3            # Laenge des frueheren Zeitraums (direkt vor dem Zeitraum der Stufe 2)
STAGE1_TOP_N = 20           # hoechstens so viele Kandidaten gehen in Stufe 2
STAGE1_MAX_P = 0.05         # Kandidat nur, wenn der unkorrigierte p-Wert in Stufe 1 darunter liegt
STAGE1_MAX_PER_ETF = 3      # ein ETF darf hoechstens in so vielen Kandidaten-Paaren vorkommen

# ---- US-Markt (--us): in den USA gelistete Aktien bzw. mit --etf ETFs, Kurse und Umsaetze in USD
US_STOCK_MIN_MEDIAN_TURNOVER_USD = 5_000_000   # Median-Tagesumsatz je Aktie in USD
US_ETF_MIN_MEDIAN_TURNOVER_USD = 1_000_000     # Median-Tagesumsatz je ETF in USD
US_PRESCREEN_FACTOR = 0.5   # Vorauswahl: lange Historie nur fuer Titel laden, deren Umsatz der letzten
                            # 3 Monate mindestens diesem Anteil der Umsatzschwelle entspricht
US_MR_COST_BPS = 5          # Kosten je Kauf oder Verkauf in Basispunkten (US-Titel sind enger gestellt)

# ---- Zusatzauswertung --mean-reversion: Rueckkehr zum gleitenden Mittel, mit Kosten, auf neuen Daten
MR_LOOKBACK = 60            # Handelstage fuer gleitenden Mittelwert und Standardabweichung des Spreads
MR_ENTRY_Z = 2.0            # Einstieg, wenn der Spread mehr als 2 Std vom gleitenden Mittel entfernt ist
MR_EXIT_Z = 0.0             # Ausstieg, wenn er das Mittel wieder erreicht (0) bzw. bis auf EXIT_Z herankommt
MR_MAX_HOLD = 40            # spaetestens nach so vielen Handelstagen glattstellen (Zeit-Stopp)
MR_COST_BPS = 10            # Kosten je Kauf oder Verkauf in Basispunkten des gehandelten Betrags
MR_BORROW_FEE = 0.01        # Leihgebuehr p. a. auf die leerverkaufte Seite
MR_FORMATION_YEARS = 3      # Auswahl der Paare anhand ihrer Ergebnisse in diesen Vorjahren ...
MR_TEST_YEARS = 1           # ... getestet im jeweils folgenden Jahr (rollierend)
MR_TOP_N = 20               # so viele Paare werden je Testjahr ausgewaehlt
MR_MAX_PER_STOCK = 2        # ein Titel darf hoechstens in so vielen ausgewaehlten Paaren vorkommen
MR_MIN_TRADES = 5           # Mindestzahl abgeschlossener Trades in den Vorjahren
MR_MAX_PAIRS = 5_000        # Obergrenze Paare je Jahr (ETF-Modus: hoechste Korrelation zuerst)
MR_MAX_DAILY_MOVE = 3.0     # Tagesbewegung ueber Faktor 3 (mehr als +200 % oder -67 %) = Bruch in den Kursdaten
                            # (z. B. neue Aktien nach Insolvenz): die Historie davor wird nicht verwendet
MR_SHOW_TOP = 10            # so viele der besten Paare (nach Sharpe im Gesamtzeitraum) Jahr fuer Jahr zeigen
MR_TREND_YEARS = 2          # 'zuletzt' = die letzten so vielen Testjahre, verglichen mit den Jahren davor
MR_TREND_MIN_DIFF = 0.5     # ab diesem Sharpe-Unterschied gilt die Entwicklung als zu- bzw. abnehmend

N_JOBS = -1                 # CPU-Kerne fuer den Cointegrationstest (-1 = alle)
# ===============================================================================

MODE = "aktien"             # wird in main() gesetzt: "aktien" (Standard) oder "etf" (--etf)
MARKET = "xetra"            # wird in main() gesetzt: "xetra" (Standard) oder "us" (--us)

US_LISTS = {"nasdaqlisted.txt": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "otherlisted.txt": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"}
# US-Aktien: keine Optionsscheine, Bezugsrechte, Vorzugs- und Schuldtitel, Units, leere Boersenmaentel
US_STOCK_EXCLUDE = re.compile(r"Warrant|\bRights?\b|Preferred|\bNotes?\b|Debenture|\bUnits?\b(?!.*Limited Partner)"
                              r"|Acquisition Corp|%|Depositary Shares?.*(?:Preferred|Interest in)|\bETN\b",
                              re.IGNORECASE)
US_LEVERAGE_PATTERN = re.compile(r"ULTRAPRO|ULTRASHORT|\bULTRA\s+(?!SHORT)|LEVERAG|\b\d(?:\.\d+)?X\b|-\dX\b"
                                 r"|INVERSE|\bBEAR\b|\bBULL\b|PROSHARES\s+SHORT|DAILY.*\bSHORT\b",
                                 re.IGNORECASE)


def cache_file(kind: str) -> Path:
    """Cache-Datei je Markt und Modus (Xetra behaelt die bisherigen Namen)."""
    us = MARKET == "us"
    names = {"prices": {("xetra", "aktien"): "prices_aktien.pkl", ("xetra", "etf"): "prices.pkl",
                        ("us", "aktien"): "prices_us_aktien.pkl", ("us", "etf"): "prices_us_etf.pkl"}[(MARKET, MODE)],
             "stock_info": "info_us_aktien.csv" if us else "info_aktien.csv",
             "aum": "aum_us.csv" if us else "aum.csv",
             "prescreen": f"vorauswahl_us_{MODE}.csv"}
    return CACHE_DIR / names[kind]


def min_turnover() -> float:
    if MARKET == "us":
        return US_STOCK_MIN_MEDIAN_TURNOVER_USD if MODE == "aktien" else US_ETF_MIN_MEDIAN_TURNOVER_USD
    return STOCK_MIN_MEDIAN_TURNOVER_EUR if MODE == "aktien" else MIN_MEDIAN_TURNOVER_EUR

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUT_DIR = BASE_DIR / "output"

XETRA_PAGE = "https://www.cashmarket.deutsche-boerse.com/cash-en/trading/Tradable-Instruments-Xetra"
XETRA_CSV_FALLBACK = ("https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/"
                      "684b31b077a5de5d5777352984c7a7df/data/t7-xetr-allTradableInstruments.csv")
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) pair_screener"}

# Gehebelte / inverse Produkte. "SHORT" nur, wenn nicht "Short Duration/Maturity/Term" o. ae.
# Aktien: zum Verkauf eingereichte Stuecke, Bezugsrechte u. ae. sind keine eigenen Unternehmen
STOCK_EXCLUDE_PATTERN = re.compile(r"VERK\.|Z\.\s*VERK|BEZUGSR|\bBZR\b|JUNGE|UMTAUSCH", re.IGNORECASE)

LEVERAGE_PATTERN = re.compile(
    r"(\bLEV\b|LEVERAG|\b[2-5]X\b|-[1-5]X\b|INVERSE|\bBEAR\b|"
    r"\bSHORT\b(?!\s*[-.]?\s*(DUR|MAT|TERM|TREA|DAT|CORP|GOV|BOND|BD)))",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """CSV fuer deutsches Excel: Semikolon, Dezimalkomma, UTF-8 mit BOM."""
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    if MODE == "aktien":                                    # intern heissen die Spalten etf_a / etf_b
        df = df.rename(columns={"etf_a": "aktie_a", "etf_b": "aktie_b"})
    if MARKET == "us":                                      # Umsaetze sind dort in Tsd. USD
        df = df.rename(columns={"umsatz_a_teur": "umsatz_a_tusd", "umsatz_b_teur": "umsatz_b_tusd"})
    kw = dict(sep=";", decimal=",", index=index, encoding="utf-8-sig", float_format="%.6g")
    try:
        df.to_csv(path, **kw)
    except PermissionError:                          # z. B. Datei gerade in Excel geoeffnet
        path = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        df.to_csv(path, **kw)
        log(f"  {name} ist gesperrt (in Excel geoeffnet?) - gespeichert als {path.name}")
    return path


def read_user_csv(path: Path, **kw) -> pd.DataFrame:
    """Eigene CSV-Dateien lesen: Trennzeichen ; oder , automatisch, UTF-8 oder Windows-Kodierung (Excel)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", **kw)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_results(name: str) -> pd.DataFrame:
    """Hilfsfunktion fuers Notebook: load_results('current_candidates.csv')."""
    return pd.read_csv(OUT_DIR / name, sep=";", decimal=",", encoding="utf-8-sig")


# ------------------------------------------------------------------ Universum
def download_xetra_list(path: Path) -> None:
    import requests

    try:
        html = requests.get(XETRA_PAGE, headers=HTTP_HEADERS, timeout=30).text
        m = re.search(r'href="([^"]*t7-xetr-allTradableInstruments\.csv)"', html)
        url = urljoin(XETRA_PAGE, m.group(1)) if m else XETRA_CSV_FALLBACK
    except requests.RequestException:
        url = XETRA_CSV_FALLBACK
    log(f"Lade Xetra-Instrumentenliste: {url}")
    r = requests.get(url, headers=HTTP_HEADERS, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)


def parse_xetra_list(path: Path, mode: str = "etf") -> pd.DataFrame:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.splitlines()
    hdr_idx = next(i for i, line in enumerate(lines) if line.startswith("Product Status"))
    rows = list(csv.reader(lines[hdr_idx:], delimiter=";"))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if r]
    df = pd.DataFrame(body, columns=header)

    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> str:
        if name.lower() not in cols:
            raise KeyError(f"Spalte '{name}' fehlt in der Xetra-Liste. Vorhanden: {list(df.columns)}")
        return cols[name.lower()]

    grp = df[col("Product Assignment Group Description")].str.upper()
    if mode == "aktien":
        st = df[df[col("Instrument Type")].str.strip().str.upper() == "CS"].copy()
        if "instrument status" in cols:
            st = st[st[col("Instrument Status")].str.strip().str.lower() == "active"]
        st["typ"] = "Aktie"
        st["segment"] = grp.loc[st.index].str.strip()
        st = st.rename(columns={col("Mnemonic"): "mnemonic", col("ISIN"): "isin", col("Instrument"): "name"})
        st = st[["mnemonic", "isin", "name", "typ", "segment"]].apply(lambda x: x.str.strip())
        st = st[st["mnemonic"] != ""].drop_duplicates("isin").drop_duplicates("mnemonic")
        st["ticker"] = st["mnemonic"] + ".DE"
        return st.reset_index(drop=True)
    is_etf = grp.str.contains("EXCHANGE TRADED FUNDS", na=False)
    is_etc = grp.str.contains("EXCHANGE TRADED COMMODIT", na=False)
    etf = df[is_etf | (is_etc & INCLUDE_ETC)].copy()
    etf["typ"] = np.where(is_etc.loc[etf.index], "ETC", "ETF")
    if "instrument status" in cols:
        etf = etf[etf[col("Instrument Status")].str.strip().str.lower() == "active"]
    etf = etf.rename(columns={col("Mnemonic"): "mnemonic", col("ISIN"): "isin", col("Instrument"): "name"})
    etf = etf[["mnemonic", "isin", "name", "typ"]].apply(lambda s: s.str.strip())
    etf = etf[etf["mnemonic"] != ""].drop_duplicates("isin").drop_duplicates("mnemonic")
    etf["ticker"] = etf["mnemonic"] + ".DE"
    return etf.reset_index(drop=True)


def load_us_universe(refresh: bool) -> pd.DataFrame:
    """Alle an Nasdaq, NYSE und den uebrigen US-Boersen gelisteten Aktien bzw. ETFs (Listen von Nasdaq Trader)."""
    import requests

    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for fname, url in US_LISTS.items():
        path = CACHE_DIR / f"us_{fname}"
        if refresh or not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
            log(f"Lade US-Wertpapierliste: {url}")
            r = requests.get(url, headers=HTTP_HEADERS, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip() and not ln.startswith("File Creation Time")]
        rows = [ln.split("|") for ln in lines]
        df = pd.DataFrame([(r + [""] * len(rows[0]))[:len(rows[0])] for r in rows[1:]],
                          columns=[c.strip() for c in rows[0]])
        sym = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        d = pd.DataFrame({"symbol": df[sym].str.strip(), "name": df["Security Name"].str.strip(),
                          "etf": df["ETF"].str.strip().str.upper() == "Y",
                          "test": df["Test Issue"].str.strip().str.upper() == "Y",
                          "segment": df["Exchange"].str.strip() if "Exchange" in df.columns else "NASDAQ"})
        if "Financial Status" in df.columns:                # Nasdaq: nur regulaer notierte Titel
            d = d[df["Financial Status"].str.strip().isin(["N", ""]).to_numpy()]
        frames.append(d)
    u = pd.concat(frames, ignore_index=True)
    u = u[~u["test"] & u["symbol"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]     # keine $-, +-, =-Sonderlinien
    u["ticker"] = u["symbol"].str.replace(".", "-", regex=False)               # Yahoo: BRK.B -> BRK-B
    u = u.drop_duplicates("ticker")
    n_all = len(u)
    if MODE == "aktien":
        u = u[~u["etf"]]
        odd = u["name"].str.contains(US_STOCK_EXCLUDE)
        u = u[~odd].assign(typ="Aktie")
        log(f"USA: {n_all} gelistete Titel, davon {len(u)} Aktien (ohne ETFs, Optionsscheine, Vorzugstitel, "
            f"Boersenmaentel)")
    else:
        u = u[u["etf"]].assign(typ="ETF")
        lev = u["name"].str.contains(US_LEVERAGE_PATTERN)
        OUT_DIR.mkdir(exist_ok=True)
        save_csv(u[lev][["ticker", "name"]], "excluded_leveraged_inverse.csv")
        u = u[~lev]
        log(f"USA: {n_all} gelistete Titel, davon {len(u) + int(lev.sum())} ETFs, {int(lev.sum())} gehebelt/invers "
            f"ausgeschlossen, {len(u)} verbleiben")
    u["isin"] = ""
    return u[["ticker", "name", "isin", "typ", "segment"]].reset_index(drop=True)


def us_prescreen(tickers: list[str], refresh: bool) -> list[str]:
    """Vorauswahl nach Umsatz der letzten 3 Monate, damit nicht fuer ueber 10.000 Titel 10 Jahre geladen werden."""
    cache = cache_file("prescreen")
    known = pd.DataFrame(columns=["ticker", "umsatz_usd"])
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 30 * 86400:
        known = pd.read_csv(cache, keep_default_na=False, na_values=[""])   # Ticker "NA" ist kein fehlender Wert
    have = set(known["ticker"])
    todo = [t for t in tickers if t not in have]
    if todo:
        log(f"Vorauswahl: lade 3 Monate Kurse fuer {len(todo)} US-Titel ...")
        rows = []
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            c, v = _download_chunk(chunk, None, period="3mo")
            med = (c * v.reindex_like(c)).median() if not c.empty else pd.Series(dtype=float)
            rows += [{"ticker": t, "umsatz_usd": float(med.get(t, np.nan))} for t in chunk]
            log(f"  {min(i + 200, len(todo))}/{len(todo)}")
            time.sleep(1)
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)
    k = known.drop_duplicates("ticker", keep="last").set_index("ticker")["umsatz_usd"].reindex(tickers)
    keep = k[k >= US_PRESCREEN_FACTOR * min_turnover()].index.tolist()
    log(f"Vorauswahl: {len(keep)} von {len(tickers)} Titeln mit Median-Umsatz >= "
        f"{US_PRESCREEN_FACTOR * min_turnover() / 1e6:.1f} Mio. USD in den letzten 3 Monaten")
    return keep


def load_universe(refresh: bool) -> pd.DataFrame:
    stocks = MODE == "aktien"
    us = MARKET == "us"
    override = BASE_DIR / (("universe_us_aktien.csv" if stocks else "universe_us_etf.csv") if us
                           else ("universe_aktien.csv" if stocks else "universe.csv"))
    if override.exists():
        u = read_user_csv(override, dtype=str)
        for c in ("name", "isin", "segment"):
            if c not in u.columns:
                u[c] = ""
        if "typ" not in u.columns:
            u["typ"] = "Aktie" if stocks else "ETF"
        u["ticker"] = u["ticker"].str.strip().str.upper()
        if not us:
            u["ticker"] = np.where(u["ticker"].str.contains(r"\."), u["ticker"], u["ticker"] + ".DE")
        log(f"Eigenes Universum aus {override.name}: {len(u)} Instrumente")
        return u[["ticker", "name", "isin", "typ", "segment"]]
    if us:
        return load_us_universe(refresh)

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "xetra_allTradableInstruments.csv"
    if refresh or not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
        download_xetra_list(path)

    if stocks:
        st = parse_xetra_list(path, "aktien")
        n_all = len(st)
        odd = st["name"].str.contains(STOCK_EXCLUDE_PATTERN)
        st = st[~odd]
        if STOCK_SEGMENTS:
            st = st[st["segment"].isin([x.upper() for x in STOCK_SEGMENTS])]
        log(f"Xetra: {n_all} Aktien, {int(odd.sum())} Sonderlinien (eingereichte Stuecke u. ae.) ausgeschlossen, "
            f"{len(st)} verbleiben" + (f" (Segmente: {', '.join(STOCK_SEGMENTS)})" if STOCK_SEGMENTS else ""))
        return st[["ticker", "name", "isin", "typ", "segment"]].reset_index(drop=True)

    etf = parse_xetra_list(path, "etf")
    etf["segment"] = ""
    lev = etf["name"].str.contains(LEVERAGE_PATTERN)
    save_csv(etf[lev], "excluded_leveraged_inverse.csv")
    etf = etf[~lev]
    log(f"Xetra: {len(etf) + lev.sum()} ETFs/ETCs, {lev.sum()} gehebelt/invers ausgeschlossen, "
        f"{len(etf)} verbleiben (davon {int((etf['typ'] == 'ETC').sum())} ETCs)")
    return etf[["ticker", "name", "isin", "typ", "segment"]].reset_index(drop=True)


# ------------------------------------------------------------------ Kurse
def _download_chunk(chunk: list[str], start: str | None, period: str | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    span = {"period": period} if period else {"start": start}
    for attempt in range(4):
        try:
            data = yf.download(chunk, auto_adjust=True, progress=False,
                               threads=True, group_by="column", **span)
            break
        except Exception as e:  # Rate-Limit o. ae.
            wait = 15 * (attempt + 1)
            log(f"  Download-Fehler ({type(e).__name__}), neuer Versuch in {wait}s")
            time.sleep(wait)
    else:
        return pd.DataFrame(), pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"], data["Volume"]
    return data[["Close"]].set_axis(chunk[:1], axis=1), data[["Volume"]].set_axis(chunk[:1], axis=1)


def load_prices(tickers: list[str], refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache = cache_file("prices")
    start = (pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS, days=10)).strftime("%Y-%m-%d")

    old = None
    to_load = tickers
    if cache.exists() and not refresh:
        c = pd.read_pickle(cache)
        age_days = (pd.Timestamp.today() - c["fetched"]).days
        if age_days <= 3:
            todo = [t for t in tickers if t not in c["tried"]]
            if not todo:
                log(f"Kurse aus Cache (Stand {c['fetched']:%d.%m.%Y})")
                return c["close"], c["volume"]
            old, to_load = c, todo                     # nur neue Instrumente nachladen
            log(f"Kurse aus Cache (Stand {c['fetched']:%d.%m.%Y}), {len(todo)} neue Instrumente")
        else:
            log(f"Kurs-Cache ist {age_days} Tage alt, lade neu")

    log(f"Lade Kurse fuer {len(to_load)} Instrumente ab {start} von Yahoo Finance ...")
    closes, vols = [], []
    chunk_size = 100
    for i in range(0, len(to_load), chunk_size):
        chunk = to_load[i:i + chunk_size]
        c, v = _download_chunk(chunk, start)
        closes.append(c)
        vols.append(v)
        log(f"  {min(i + chunk_size, len(to_load))}/{len(to_load)}")
        time.sleep(1)
    if old is not None:
        closes.insert(0, old["close"])
        vols.insert(0, old["volume"])
    closes = [c for c in closes if not c.empty]
    vols = [v for v in vols if not v.empty]
    for frame in closes + vols:
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    close = pd.concat(closes, axis=1)
    volume = pd.concat(vols, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    close = close.loc[:, close.notna().sum() > 0]
    volume = volume.reindex(columns=close.columns)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = close.index
    tried = set(tickers) | (old["tried"] if old is not None else set())
    fetched = old["fetched"] if old is not None else pd.Timestamp.today()
    pd.to_pickle({"close": close, "volume": volume, "tried": tried, "fetched": fetched}, cache)
    log(f"Kurse vorhanden fuer {close.columns.isin(tickers).sum()} von {len(tickers)} Instrumenten")
    return close, volume


# ------------------------------------------------------------------ Fondsvolumen (AUM)
FUND_CCY_CANDIDATES = ["EUR", "USD", "GBP", "CHF", "JPY", "SEK", "NOK", "DKK", "PLN",
                       "AUD", "CAD", "HKD", "CNY", "INR", "KRW"]
AUM_CACHE_COLUMNS = ["ticker", "total_assets", "currency", "nav", "price", "benchmark"]

# Referenzindex aus Yahoos Fondsbeschreibung ("... aims to track the performance of the X Index")
BENCHMARK_RE = re.compile(
    r"(?:track|replicat|reflect|mirror|return of|performance of)\w*\s+(?:the\s+)?"
    r"(?:(?:total\s+)?(?:return|performance)\s+of\s+)?(?:the\s+)?([^.;:()]{3,90}?\bIndex\b)",
    re.IGNORECASE)
INDEX_PROVIDERS = ("MSCI", "FTSE", "SP", "STOXX", "NASDAQ", "DAX", "SOLACTIVE", "BLOOMBERG", "ICE",
                   "IBOXX", "MARKIT", "RUSSELL", "DOWJONES", "DJ", "JPMORGAN", "JPM", "NIKKEI", "TOPIX",
                   "HANGSENG", "CAC", "IBEX", "FTSEMIB", "MORNINGSTAR", "INDXX", "SMI", "AEX", "WIG",
                   "BBG", "BARCLAYS", "CITI", "EUROSTOXX", "MDAX", "SDAX", "TECDAX", "NYSE", "CRSP")
_BENCH_DROP = re.compile(r"\b(THE|INDEX|NET|TOTAL|RETURN|NOTIONAL|GROSS|PRICE|TR|NTR|NR|GR|USD|EUR|"
                         r"GBP|CHF|JPY|IN|DOLLAR|EURO|DAILY|ACCUMULATING|DISTRIBUTING)\b")


def normalize_benchmark(text) -> str | None:
    """'Nasdaq-100 Notional Net Total Return Index' -> 'NASDAQ100'. None, wenn kein bekannter Anbieter."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.upper().replace("&", "").replace("S P", "SP")
    t = re.sub(r"[®™*]", "", t)
    t = _BENCH_DROP.sub(" ", re.sub(r"[^A-Z0-9 ]", " ", t))
    key = re.sub(r"\s+", "", t)
    if len(key) < 3 or not any(key.startswith(p) or p in key for p in INDEX_PROVIDERS):
        return None
    return key


def extract_benchmark(summary) -> str | None:
    if not isinstance(summary, str):
        return None
    m = BENCHMARK_RE.search(summary)
    return m.group(1).strip() if m else None


def _fetch_info(ticker: str) -> dict:
    import yfinance as yf

    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
            price = (info.get("previousClose") or info.get("regularMarketPreviousClose")
                     or info.get("regularMarketPrice"))
            return {"ticker": ticker, "total_assets": info.get("totalAssets"),
                    "currency": info.get("currency"), "nav": info.get("navPrice"), "price": price,
                    "benchmark": extract_benchmark(info.get("longBusinessSummary"))}
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {"ticker": ticker, "total_assets": None, "currency": None, "nav": None, "price": None,
            "benchmark": None}


def _usd_per_unit(currencies: list[str]) -> dict[str, float]:
    """Wie viele USD eine Einheit der jeweiligen Waehrung wert ist (aktuell)."""
    import yfinance as yf

    rates = {"USD": 1.0}
    todo = [c for c in dict.fromkeys(currencies) if c and c != "USD"]
    if not todo:
        return rates
    try:
        px = yf.download([f"{c}USD=X" for c in todo], period="10d", progress=False,
                         auto_adjust=True)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(f"{todo[0]}USD=X")
        for c in todo:
            col = f"{c}USD=X"
            if col in px.columns and px[col].notna().any():
                rates[c] = float(px[col].dropna().iloc[-1])
    except Exception as e:
        log(f"  Wechselkurse konnten nicht geladen werden ({type(e).__name__})")
    missing = [c for c in todo if c not in rates]
    if missing:
        log(f"  Kein Wechselkurs fuer: {', '.join(missing)}")
    return rates


def infer_fund_currency(nav: float, price: float, trade_ccy: str,
                        usd_per: dict[str, float]) -> tuple[str, str]:
    """Waehrung, in der Yahoo NAV und Fondsvolumen angibt, aus dem Verhaeltnis NAV / Boersenkurs.

    Beispiel: Kurs 5,10 EUR, NAV 870 -> Verhaeltnis ~170 ~ EUR/JPY -> Fondswaehrung JPY.
    Der Boersenkurs steht immer in der Handelswaehrung (bei .DE: EUR).
    """
    if trade_ccy not in usd_per:
        return trade_ccy, "unsicher: kein Wechselkurs"
    if not (pd.notna(nav) and pd.notna(price) and nav > 0 and price > 0):
        return trade_ccy, "Annahme Handelswaehrung (keine NAV)"
    ratio = nav / price
    best, err = trade_ccy, np.inf
    for c, u in usd_per.items():
        e = abs(np.log(ratio / (usd_per[trade_ccy] / u)))
        if e < err:
            best, err = c, e
    if err < np.log(1.3):
        return best, "aus NAV/Kurs abgeleitet"
    return trade_ccy, "unsicher: NAV passt zu keiner Waehrung"


def load_aum_usd(tickers: list[str], refresh: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Fondsvolumen in USD je Ticker (NaN = unbekannt) und Details zur Waehrungsbestimmung."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = cache_file("aum")
    known = pd.DataFrame(columns=AUM_CACHE_COLUMNS)
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 30 * 86400:
        cached = pd.read_csv(cache, keep_default_na=False, na_values=[""])   # Ticker "NA" ist kein fehlender Wert
        if set(AUM_CACHE_COLUMNS) <= set(cached.columns):     # alter Cache ohne NAV -> neu laden
            known = cached
    todo = [t for t in tickers if t not in set(known["ticker"])]
    if todo:
        log(f"Lade Fondsvolumen fuer {len(todo)} ETFs (Yahoo 'totalAssets' + NAV) ...")
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_fetch_info, t) for t in todo]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % 100 == 0:
                    log(f"  {k}/{len(todo)}")
                    pd.concat([known, pd.DataFrame(rows)]).to_csv(cache, index=False)
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)

    known = known.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(tickers)
    usd_per = _usd_per_unit(FUND_CCY_CANDIDATES + known["currency"].dropna().tolist())

    ccy, quality = [], []
    for _, r in known.iterrows():
        c, q = infer_fund_currency(pd.to_numeric(r["nav"], errors="coerce"),
                                   pd.to_numeric(r["price"], errors="coerce"),
                                   r["currency"] if isinstance(r["currency"], str)
                                   else ("USD" if MARKET == "us" else "EUR"), usd_per)
        ccy.append(c)
        quality.append(q)
    meta = pd.DataFrame({"total_assets": pd.to_numeric(known["total_assets"], errors="coerce"),
                         "fund_ccy": ccy, "aum_quelle": quality}, index=known.index)
    meta["aum_usd"] = meta["total_assets"] * meta["fund_ccy"].map(usd_per).astype(float)
    meta.loc[meta["total_assets"].isna(), "aum_quelle"] = "fehlt"

    overrides = BASE_DIR / "aum_overrides.csv"
    if overrides.exists():
        o = read_user_csv(overrides)
        o = o.set_index("ticker")["aum_usd"].astype(float)
        o = o[o.index.isin(meta.index)]
        meta.loc[o.index, "aum_usd"] = o
        meta.loc[o.index, "aum_quelle"] = "manuell (aum_overrides.csv)"
        log(f"{len(o)} AUM-Werte aus aum_overrides.csv uebernommen")

    # Referenzindex: index_map.csv (manuell) > Yahoo-Fondsbeschreibung
    meta["benchmark"] = known["benchmark"]
    meta["index_key"] = known["benchmark"].map(normalize_benchmark)
    meta["index_quelle"] = np.where(meta["index_key"].notna(), "Yahoo-Beschreibung", "unbekannt")
    index_map = BASE_DIR / "index_map.csv"
    if index_map.exists():
        im = read_user_csv(index_map, dtype=str)
        im = im.set_index("ticker")["index"].dropna()
        im = im[im.index.isin(meta.index)]
        meta.loc[im.index, "benchmark"] = im
        meta.loc[im.index, "index_key"] = im.map(lambda v: re.sub(r"[^A-Z0-9]", "", v.upper()))
        meta.loc[im.index, "index_quelle"] = "manuell (index_map.csv)"
        log(f"{len(im)} Indizes aus index_map.csv uebernommen")
    log(f"Referenzindex bekannt fuer {meta['index_key'].notna().sum()} von {len(meta)} ETFs")

    counts = meta["aum_quelle"].value_counts()
    log("Fondsvolumen-Waehrung: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    changed = meta[(meta["fund_ccy"] != known["currency"]) & meta["total_assets"].notna()]
    if len(changed):
        log(f"  {len(changed)} ETFs melden ihr Volumen nicht in der Handelswaehrung "
            f"({', '.join(sorted(changed['fund_ccy'].unique()))})")
    return meta["aum_usd"], meta


# ------------------------------------------------------------------ Aktien: Marktkapitalisierung, Branche
STOCK_INFO_COLUMNS = ["ticker", "market_cap", "currency", "sector", "industry", "long_name"]


def _fetch_stock_info(ticker: str) -> dict:
    import yfinance as yf

    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
            return {"ticker": ticker, "market_cap": info.get("marketCap"), "currency": info.get("currency"),
                    "sector": info.get("sector"), "industry": info.get("industry"),
                    "long_name": info.get("longName") or info.get("shortName")}
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {"ticker": ticker, "market_cap": None, "currency": None, "sector": None, "industry": None,
            "long_name": None}


def load_stock_info(tickers: list[str], refresh: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Marktkapitalisierung in USD je Ticker (NaN = unbekannt) sowie Sektor / Branche laut Yahoo."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = cache_file("stock_info")
    known = pd.DataFrame(columns=STOCK_INFO_COLUMNS)
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 30 * 86400:
        cached = pd.read_csv(cache, keep_default_na=False, na_values=[""])   # Ticker "NA" ist kein fehlender Wert
        if set(STOCK_INFO_COLUMNS) <= set(cached.columns):
            known = cached
    todo = [t for t in tickers if t not in set(known["ticker"])]
    if todo:
        log(f"Lade Marktkapitalisierung und Branche fuer {len(todo)} Aktien von Yahoo Finance ...")
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_fetch_stock_info, t) for t in todo]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % 100 == 0:
                    log(f"  {k}/{len(todo)}")
                    pd.concat([known, pd.DataFrame(rows)]).to_csv(cache, index=False)
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)

    known = known.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(tickers)
    usd_per = _usd_per_unit(["EUR"] + known["currency"].dropna().tolist())
    meta = pd.DataFrame({"market_cap": pd.to_numeric(known["market_cap"], errors="coerce"),
                         "currency": known["currency"], "sector": known["sector"],
                         "industry": known["industry"], "long_name": known["long_name"]}, index=known.index)
    meta["aum_usd"] = meta["market_cap"] * meta["currency"].map(usd_per).astype(float)

    # Gruppe fuer den Paarvergleich: Branche oder Sektor laut Yahoo, manuell ueberschreibbar
    col = {"industry": "industry", "sector": "sector"}.get(STOCK_GROUP_BY)
    meta["gruppe"] = meta[col] if col else ""
    manual = BASE_DIR / ("gruppen_us_aktien.csv" if MARKET == "us" else "gruppen_aktien.csv")
    if manual.exists():
        g = read_user_csv(manual, dtype=str)
        g["ticker"] = g["ticker"].str.strip().str.upper()
        if MARKET != "us":
            g["ticker"] = np.where(g["ticker"].str.contains(r"\."), g["ticker"], g["ticker"] + ".DE")
        g = g.set_index("ticker")["gruppe"].dropna()
        g = g[g.index.isin(meta.index)]
        meta.loc[g.index, "gruppe"] = g
        log(f"{len(g)} Gruppen aus {manual.name} uebernommen")
    log(f"Marktkapitalisierung bekannt fuer {int(meta['aum_usd'].notna().sum())} von {len(meta)} Aktien, "
        f"Branche bekannt fuer {int(meta['industry'].notna().sum())}")
    return meta["aum_usd"], meta


# ------------------------------------------------------------------ Statistik
def _test_chunk(pairs: np.ndarray, X: np.ndarray) -> list[tuple]:
    """Engle-Granger in beide Richtungen fuer eine Liste von Spaltenpaaren (i, j) aus X."""
    out = []
    for i, j in pairs:
        x, y = X[:, i], X[:, j]
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        if len(x) < 200:
            continue
        try:
            p_xy = coint(x, y, maxlag=COINT_MAXLAG)[1]
            p_yx = coint(y, x, maxlag=COINT_MAXLAG)[1]
        except Exception:
            continue
        beta, alpha = np.polyfit(y, x, 1)          # log(A) = alpha + beta * log(B)
        spread_std = float(np.std(x - beta * y - alpha, ddof=1))
        out.append((int(i), int(j), max(p_xy, p_yx), p_xy, p_yx, beta, alpha, spread_std))
    return out


def liquidity_stats(px: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Je ETF im Fenster: Anteil 'stale' Tage (kein Handel / Kurs unveraendert) und Median-Umsatz in EUR."""
    listed = px.ffill().notna()
    ret = px.pct_change(fill_method=None)
    v = vol.reindex_like(px)
    n_listed = listed.sum().replace(0, np.nan)
    has_vol = (v > 0).sum() >= 0.05 * n_listed          # sonst liefert Yahoo keine Umsaetze
    traded = px.notna() & (ret.abs() > 0) & ((v > 0) | ~has_vol)
    stale = 1 - traded.where(listed).sum() / n_listed
    turnover = (px * v.fillna(0)).where(listed).median()
    turnover[~has_vol] = np.nan
    return pd.DataFrame({"stale_share": stale, "turnover_eur": turnover})


def monthly_te_stats(lp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Tracking Error p. a. (beta = 1) auf Monatsrenditen und relativ zur Volatilitaet (TE / sqrt(vol_a*vol_b))."""
    mret = lp.iloc[::21].diff().iloc[1:]
    cov = mret.cov(min_periods=12).to_numpy()
    var = np.diag(cov)
    te = np.sqrt(np.clip((var[:, None] + var[None, :] - 2 * cov) * 12, 0, None))   # NaN bei zu wenig Monaten
    vol = np.sqrt(np.clip(var, 0, None) * 12)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = te / np.sqrt(np.outer(vol, vol))
    return te, rel


def detect_price_errors(lp: pd.DataFrame) -> pd.DataFrame:
    """Kursbloecke, die gegenueber aehnlichen ETFs springen und spaeter zurueckspringen (Yahoo-Datenfehler).

    Vergleich je ETF mit dem Median der 5 am staerksten korrelierten ETFs (Korrelation >= PEER_MIN_CORR).
    ETFs ohne solche Vergleichs-ETFs werden nicht geprueft.
    """
    r = lp.diff()
    R = r.to_numpy()
    corr = np.corrcoef(np.nan_to_num(R), rowvar=False)
    np.fill_diagonal(corr, -np.inf)
    rows = []
    for k, t in enumerate(lp.columns):
        peers = np.argsort(-corr[k])[:5]
        peers = peers[corr[k, peers] >= PEER_MIN_CORR]
        if len(peers) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # Tage, an denen alle Peers NaN sind
            resid = R[:, k] - np.nanmedian(R[:, peers], axis=1)
        big = np.where(np.abs(np.nan_to_num(resid)) > JUMP_THRESHOLD)[0]
        for d in big:
            back = big[(big > d) & (big <= d + JUMP_REVERT_DAYS)]
            back = back[np.sign(resid[back]) != np.sign(resid[d])]
            if len(back):
                rows.append({"ticker": t, "sprung_am": lp.index[d].date(), "abweichung": resid[d],
                             "zurueck_am": lp.index[back[0]].date(), "abweichung_zurueck": resid[back[0]]})
                break
    return pd.DataFrame(rows, columns=["ticker", "sprung_am", "abweichung", "zurueck_am", "abweichung_zurueck"])


def prepare_window(px: pd.DataFrame, vol: pd.DataFrame, names: pd.Series, tag: str = "",
                   quiet: bool = False):
    """Abdeckung, Liquiditaet und Datenfehler pruefen. Gibt Log-Kurse der sauberen ETFs und Status je ETF zurueck."""
    liq = liquidity_stats(px, vol)
    cov = px.notna().mean()
    status = pd.Series("ok", index=px.columns, dtype=object)
    status[cov < MIN_COVERAGE] = "zu wenig Kurshistorie"
    min_to = min_turnover()
    liquid = (liq["stale_share"] <= MAX_STALE_SHARE) & (
        (liq["turnover_eur"] >= min_to)
        | (liq["turnover_eur"].isna() & KEEP_UNKNOWN_TURNOVER))
    status[(status == "ok") & ~liquid.reindex(status.index, fill_value=False)] = "illiquide" if MARKET == "us" else "illiquide auf Xetra"
    ok = status.index[status == "ok"]
    lp = np.log(px[ok]).ffill(limit=5)
    check_errors = len(ok) > 1 and (MODE != "aktien" or STOCK_DETECT_PRICE_ERRORS)
    errors = detect_price_errors(lp) if check_errors else pd.DataFrame(
        columns=["ticker", "sprung_am", "abweichung", "zurueck_am", "abweichung_zurueck"])
    status[status.index.isin(errors["ticker"])] = "Datenfehler (Kurssprung)"
    if len(errors):
        errors.insert(1, "name", errors["ticker"].map(names))
    clean = status.index[status == "ok"]
    if quiet:                                               # zweiter Durchlauf im selben Zeitraum
        return lp[clean], liq, status
    save_csv(errors, f"data_errors{tag}.csv")
    log(f"    {int((cov >= MIN_COVERAGE).sum())} Instrumente mit genug Historie, {len(ok)} liquide "
        f"(<= {MAX_STALE_SHARE:.0%} Tage ohne Kursaenderung, Median-Umsatz >= "
        f"{min_to / 1e3:.0f} Tsd. {'USD' if MARKET == 'us' else 'EUR'}), {len(errors)} mit Datenfehlern "
        f"-> {len(clean)} sauber")
    return lp[clean], liq, status


def _bh(p: pd.Series) -> np.ndarray:
    return multipletests(p, method="fdr_bh")[1] if len(p) else np.array([])


def screen(px: pd.DataFrame, vol: pd.DataFrame, aum: pd.Series, names: pd.Series,
           index_key: pd.Series | None = None, pair_list: pd.DataFrame | None = None,
           tag: str = "", spread_filter: bool = True, near_same: bool = False,
           groups: pd.Series | None = None) -> pd.DataFrame:
    """Paare testen und als Rangliste (nach p-Wert) zurueckgeben.

    Ohne pair_list: alle Kombinationen (mit Korrelations-Vorfilter).
    Mit pair_list (Spalten ticker_a, ticker_b, ...): nur diese Paare, ohne Korrelations-Vorfilter;
    nicht testbare Paare erscheinen mit Begruendung in der Spalte 'status'.
    spread_filter: Paare mit Spread-Breite unter MIN_SPREAD_STD nicht testen. Fuer die eigene
    Paarliste ist der Filter aus (dort sind schmale Spreads, z. B. bei Anleihen, bewusst gewaehlt).
    near_same: Auswertung 'fast gleiche Indizes'. Ohne pair_list werden genau die Paare getestet, die
    nur wegen des Tracking Errors als 'gleicher Index' gelten und deren Spread mindestens
    NEAR_SAME_MIN_SPREAD breit ist. Mit pair_list gilt der Tracking Error nicht als Ausschlussgrund.
    groups: Gruppe je Ticker (Aktien: Branche / Sektor). Ohne pair_list werden nur Paare innerhalb
    derselben Gruppe getestet. Im Aktien-Modus gibt es keinen Filter 'gleicher Index / gleicher Fonds'
    (Stamm- und Vorzugsaktien desselben Unternehmens sind dort gerade erwuenscht).
    """
    lp, liq, status = prepare_window(px, vol, names, tag, quiet=near_same)
    cols = lp.columns
    pos = {t: k for k, t in enumerate(cols)}
    rets = lp.diff().fillna(0.0).to_numpy()
    corr = np.corrcoef(rets, rowvar=False) if len(cols) > 1 else np.ones((len(cols), len(cols)))
    te, rel = monthly_te_stats(lp)
    key = index_key.reindex(cols).to_numpy(dtype=object) if index_key is not None else None
    aum_v = aum.reindex(cols).to_numpy(dtype=float)
    # Spread-Breite vorab: Std des Residuums der Regression log(A) auf log(B) = std(A) * sqrt(1 - rho^2)
    lvl_corr = lp.corr().to_numpy() if len(cols) > 1 else np.ones((len(cols), len(cols)))
    lvl_std = lp.std().to_numpy()

    TE_REASON = "gleicher Index (Tracking Error)"

    def spread_width(i, j) -> np.ndarray:
        return lvl_std[i] * np.sqrt(np.clip(1 - lvl_corr[i, j] ** 2, 0, None))

    def same_reason(i, j) -> np.ndarray:
        """Grund, warum ein Paar als 'gleicher Index / gleicher Fonds' gilt ('' = unterschiedlich)."""
        reason = np.full(len(i), "", dtype=object)
        if MODE != "etf":
            return reason
        reason[rel[i, j] < MIN_REL_TE] = TE_REASON                 # NaN-Vergleich = False
        if key is not None:
            ki, kj = key[i], key[j]
            same_key = np.array([isinstance(a, str) and a != "" and a == b for a, b in zip(ki, kj)], dtype=bool)
            reason[same_key] = "gleicher Index laut Yahoo / index_map.csv"
        if SAME_AUM_IS_SAME_FUND:
            a, b = aum_v[i], aum_v[j]
            with np.errstate(invalid="ignore"):
                same_fund = (a > 0) & (np.abs(a - b) <= 1e-6 * np.maximum(a, b))
            reason[same_fund] = "gleicher Fonds (identisches Fondsvolumen)"
        return reason

    if pair_list is None and near_same:
        iu, ju = np.triu_indices(len(cols), k=1)
        c = corr[iu, ju]
        in_corr = (c >= MIN_RET_CORR) & (c <= MAX_RET_CORR)
        cand = in_corr & (same_reason(iu, ju) == TE_REASON)
        keep = cand & (spread_width(iu, ju) >= NEAR_SAME_MIN_SPREAD)
        pairs = np.column_stack([iu[keep], ju[keep]])
        log(f"    Fast gleiche Indizes: {cand.sum()} Paare mit rel. Tracking Error < {MIN_REL_TE}, "
            f"davon {len(pairs)} mit Spread >= {NEAR_SAME_MIN_SPREAD:.1%} werden getestet")
        info = None
    elif pair_list is None:
        iu, ju = np.triu_indices(len(cols), k=1)
        c = corr[iu, ju]
        min_corr = STOCK_MIN_RET_CORR if MODE == "aktien" else MIN_RET_CORR
        in_corr = (c >= min_corr) & (c <= MAX_RET_CORR)
        n_all = len(iu)
        if groups is not None:                              # nur Paare innerhalb derselben Gruppe
            g = groups.reindex(cols).to_numpy(dtype=object)
            same_group = np.array([isinstance(x, str) and x != "" and x == y for x, y in zip(g[iu], g[ju])],
                                  dtype=bool)
            log(f"    {n_all} Paare, {int(same_group.sum())} innerhalb derselben Gruppe "
                f"({pd.Series(g).replace('', np.nan).nunique()} Gruppen)")
            in_corr &= same_group
        reason = same_reason(iu, ju)
        si = in_corr & (reason != "")
        width = spread_width(iu, ju)
        narrow = in_corr & ~si & (width < MIN_SPREAD_STD) & spread_filter

        def excluded(mask, extra):
            d = pd.DataFrame({"etf_a": cols[iu[mask]], "etf_b": cols[ju[mask]],
                              "name_a": names.reindex(cols[iu[mask]]).to_numpy(),
                              "name_b": names.reindex(cols[ju[mask]]).to_numpy(), **extra,
                              "ret_corr_daily": c[mask], "te_monthly": te[iu[mask], ju[mask]],
                              "rel_te": rel[iu[mask], ju[mask]], "spread_std": width[mask]})
            return d

        if MODE == "etf":
            save_csv(excluded(si, {"grund": reason[si]}).sort_values("rel_te"), f"same_index_excluded{tag}.csv")
        save_csv(excluded(narrow, {}).sort_values("spread_std"), f"narrow_spread_excluded{tag}.csv")
        keep = in_corr & ~si & ~narrow
        order = np.argsort(-c[keep])[:MAX_PAIRS_TO_TEST]
        pairs = np.column_stack([iu[keep][order], ju[keep][order]])
        if MODE == "etf":
            log(f"    {len(iu)} Paare, {in_corr.sum()} nach Korrelationsfilter, {si.sum()} davon gleicher "
                f"Index/Fonds, {narrow.sum()} mit Spread < {MIN_SPREAD_STD:.1%}, {len(pairs)} werden getestet")
        else:
            log(f"    {in_corr.sum()} Paare nach Korrelationsfilter (>= {min_corr}), {narrow.sum()} mit "
                f"Spread < {MIN_SPREAD_STD:.1%}, {len(pairs)} werden getestet")
        info = None
    else:
        info = pair_list.copy()
        st = []
        for a, b in zip(info["ticker_a"], info["ticker_b"]):
            sa, sb = status.get(a, "keine Kurse / nicht im Universum"), status.get(b, "keine Kurse / nicht im Universum")
            if sa != "ok" or sb != "ok":
                st.append("; ".join(f"{t}: {s}" for t, s in ((a, sa), (b, sb)) if s != "ok"))
            elif (why := same_reason(np.array([pos[a]]), np.array([pos[b]]))[0]) != "" and not (
                    near_same and why == TE_REASON):
                st.append(f"gleicher Index (rel. TE {rel[pos[a], pos[b]]:.2f})"
                          if why.startswith("gleicher Index (Tracking") else why)
            elif spread_filter and (wd := spread_width(np.array([pos[a]]), np.array([pos[b]]))[0]) < MIN_SPREAD_STD:
                st.append(f"Spread zu schmal ({wd:.2%} < {MIN_SPREAD_STD:.1%})")
            else:
                st.append("getestet")
        info["status"] = st
        ok = info["status"] == "getestet"
        pairs = np.array([[pos[a], pos[b]] for a, b in zip(info.loc[ok, "ticker_a"], info.loc[ok, "ticker_b"])],
                         dtype=int).reshape(-1, 2)
        log(f"    Paarliste: {len(info)} Paare, {len(pairs)} testbar, {len(info) - len(pairs)} nicht testbar "
            f"(Grund in Spalte 'status')")

    res = []
    if len(pairs):
        X = lp.to_numpy(dtype=float)
        n_chunks = max(1, min(len(pairs) // 200, 256))
        chunks = np.array_split(pairs, n_chunks)
        try:
            out = Parallel(n_jobs=N_JOBS)(delayed(_test_chunk)(ch, X) for ch in chunks)
        except Exception as e:                              # z. B. Probleme mit Prozess-Start unter Windows
            log(f"    Parallele Berechnung fehlgeschlagen ({type(e).__name__}: {e}); rechne nacheinander weiter")
            out = [_test_chunk(ch, X) for ch in chunks]
        res = [r for chunk in out for r in chunk]
    df = pd.DataFrame(res, columns=["i", "j", "p_value", "p_ab", "p_ba", "beta", "alpha", "spread_std"])
    df["etf_a"] = cols[df["i"]] if len(df) else []
    df["etf_b"] = cols[df["j"]] if len(df) else []
    df["ret_corr"] = corr[df["i"], df["j"]] if len(df) else []
    df["te_monthly"] = te[df["i"], df["j"]] if len(df) else []
    df["rel_te"] = rel[df["i"], df["j"]] if len(df) else []
    df["p_adj"] = _bh(df["p_value"])
    df["signifikant"] = (df["p_adj"] <= FDR_ALPHA) & (df["beta"] > 0)
    df = df.sort_values("p_value").reset_index(drop=True)
    df.insert(0, "rang", np.arange(1, len(df) + 1))

    if info is not None:                                    # Reihenfolge der Liste beibehalten, Infos ergaenzen
        df = info.merge(df.drop(columns=["i", "j", "rang"]), left_on=["ticker_a", "ticker_b"],
                        right_on=["etf_a", "etf_b"], how="left").drop(columns=["etf_a", "etf_b"])
        df = df.rename(columns={"ticker_a": "etf_a", "ticker_b": "etf_b"})
        df = df.sort_values("p_value", na_position="last").reset_index(drop=True)
        df.insert(0, "rang", np.where(df["p_value"].notna(), np.arange(1, len(df) + 1), np.nan))

    top = df.dropna(subset=["p_value"]).head(RANK_TOP_FOR_COUNT)
    freq = pd.concat([top["etf_a"], top["etf_b"]]).value_counts()
    df["name_a"] = df["etf_a"].map(names)
    df["name_b"] = df["etf_b"].map(names)
    df["aum_a_musd"] = df["etf_a"].map(aum) / 1e6
    df["aum_b_musd"] = df["etf_b"].map(aum) / 1e6
    df["umsatz_a_teur"] = df["etf_a"].map(liq["turnover_eur"]) / 1e3
    df["umsatz_b_teur"] = df["etf_b"].map(liq["turnover_eur"]) / 1e3
    df["haeufigkeit_a"] = df["etf_a"].map(freq).fillna(0).astype(int)
    df["haeufigkeit_b"] = df["etf_b"].map(freq).fillna(0).astype(int)
    n_sig = int(df["signifikant"].fillna(False).astype(bool).sum())
    log(f"    {df['p_value'].notna().sum()} Paare getestet, {n_sig} signifikant (BH-p <= {FDR_ALPHA}, beta > 0)")

    front = ["rang", "etf_a", "etf_b", "name_a", "name_b"]
    if groups is not None:
        df["gruppe_a"] = df["etf_a"].map(groups)
        df["gruppe_b"] = df["etf_b"].map(groups)
        front += ["gruppe_a", "gruppe_b"]
    if info is not None:
        front += [c for c in info.columns if c not in ("ticker_a", "ticker_b")]
    cols_out = front + ["p_value", "p_adj", "signifikant", "p_ab", "p_ba", "beta", "alpha", "spread_std",
                        "ret_corr", "te_monthly", "rel_te", "aum_a_musd", "aum_b_musd", "umsatz_a_teur", "umsatz_b_teur",
                        "haeufigkeit_a", "haeufigkeit_b"]
    out = df[[c for c in cols_out if c in df.columns]]
    if MODE == "aktien":                                    # bei Aktien ist die Groesse die Marktkapitalisierung
        out = out.rename(columns={"aum_a_musd": "marktkap_a_musd", "aum_b_musd": "marktkap_b_musd"})
    return out


# ------------------------------------------------------------------ Ablauf
def clean_prices(close: pd.DataFrame) -> pd.DataFrame:
    close = close.sort_index()
    close = close[close.index.dayofweek < 5]
    close = close.where(close > 0)
    # Feiertage / Datenluecken entfernen: Tage, an denen weniger als die Haelfte der ETFs,
    # die zu diesem Zeitpunkt schon gelistet waren, einen Kurs hat. (Nicht am heutigen
    # Universum messen - sonst fallen alle fruehen Jahre weg, in denen es viele ETFs noch nicht gab.)
    listed = close.ffill().notna().sum(axis=1)
    quoted = close.notna().sum(axis=1)
    close = close[(listed > 0) & (quoted >= 0.5 * listed)]
    return close


def load_pair_list(path: Path) -> pd.DataFrame:
    """Eigene Paarliste: Spalten ticker_a;ticker_b (+ beliebige weitere, z. B. kategorie;begruendung).

    Zeilen mit # am Anfang sind Kommentare. Trennzeichen (; , oder Tab) wird aus der Kopfzeile erkannt.
    Enthaelt die letzte Spalte selbst Trennzeichen (z. B. ein Semikolon im Begruendungstext),
    bleiben sie Teil des Textes und verschieben keine Spalten.
    """
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise SystemExit(f"{path.name} ist leer")
    sep = max(";,\t", key=lines[0].count)
    header = [h.strip().strip('"').lower() for h in lines[0].split(sep)]
    if not {"ticker_a", "ticker_b"} <= set(header):
        raise SystemExit(f"{path.name} braucht die Spalten ticker_a und ticker_b (gefunden: {header})")
    rows = []
    for ln in lines[1:]:
        parts = [p.strip().strip('"') for p in ln.split(sep, len(header) - 1)]
        rows.append((parts + [""] * len(header))[:len(header)])
    pl = pd.DataFrame(rows, columns=header)
    if MARKET == "us":                                      # Yahoo-Schreibweise: BRK.B -> BRK-B
        valid = r"[A-Z]{1,5}(-[A-Z])?"
        for c in ("ticker_a", "ticker_b"):
            pl[c] = pl[c].str.upper().str.replace(".", "-", regex=False)
    else:
        valid = r"[A-Z0-9]{1,6}\.[A-Z]{1,3}"
        for c in ("ticker_a", "ticker_b"):
            pl[c] = pl[c].str.upper()
            pl[c] = np.where(pl[c].str.contains(r"\."), pl[c], pl[c] + ".DE")   # "EXV1" -> "EXV1.DE"
    bad = pl[~pl["ticker_a"].str.fullmatch(valid) | ~pl["ticker_b"].str.fullmatch(valid)]
    if len(bad):
        log(f"  {len(bad)} Zeile(n) der Paarliste haben kein gueltiges Tickerkuerzel und werden uebersprungen: "
            + "; ".join(f"{a} / {b}" for a, b in zip(bad["ticker_a"], bad["ticker_b"]))[:300])
        pl = pl.drop(bad.index)
    return pl.drop_duplicates(["ticker_a", "ticker_b"]).reset_index(drop=True)


def select_candidates(stage1: pd.DataFrame) -> pd.DataFrame:
    """Kandidaten fuer Stufe 2: beste Paare aus Stufe 1 (nach p-Wert), beta > 0, begrenzt je ETF."""
    ok = stage1[stage1["p_value"].notna() & (stage1["p_value"] <= STAGE1_MAX_P) & (stage1["beta"] > 0)]
    ok = ok.sort_values("p_value")
    used: dict[str, int] = {}
    rows = []
    for idx, r in ok.iterrows():
        if used.get(r["etf_a"], 0) >= STAGE1_MAX_PER_ETF or used.get(r["etf_b"], 0) >= STAGE1_MAX_PER_ETF:
            continue
        rows.append(idx)
        used[r["etf_a"]] = used.get(r["etf_a"], 0) + 1
        used[r["etf_b"]] = used.get(r["etf_b"], 0) + 1
        if len(rows) == STAGE1_TOP_N:
            break
    cand = ok.loc[rows]
    extra = [c for c in ("kategorie", "begruendung") if c in cand.columns]
    out = pd.DataFrame({"ticker_a": cand["etf_a"].to_numpy(), "ticker_b": cand["etf_b"].to_numpy()})
    for c in extra:
        out[c] = cand[c].to_numpy()
    out["rang_stufe1"] = cand["rang"].to_numpy()
    out["p_stufe1"] = cand["p_value"].to_numpy()
    out["beta_stufe1"] = cand["beta"].to_numpy()
    return out


def run_two_stage(close, volume, aum, names, index_key, pair_list, mark_small, mode, f0, last,
                  near_same: bool = False, groups: pd.Series | None = None) -> None:
    """Stufe 1: Kandidaten im frueheren Zeitraum auswaehlen. Stufe 2: nur diese im spaeteren Zeitraum testen.

    Die Zeitraeume ueberlappen nicht. Die Korrektur fuer viele Tests bezieht sich in Stufe 2 nur auf
    die wenigen Kandidaten, die Daten der Stufe 2 wurden fuer die Auswahl nicht verwendet.
    """
    s0 = f0 - pd.DateOffset(years=STAGE1_YEARS)
    w1 = (close.index > s0) & (close.index <= f0)
    w2 = close.index > f0
    if w1.sum() < 252 * STAGE1_YEARS * 0.9:
        raise SystemExit(f"Zu wenig Kurshistorie fuer Stufe 1 ({s0:%d.%m.%Y} - {f0:%d.%m.%Y}): "
                         f"nur {int(w1.sum())} Handelstage. HISTORY_YEARS erhoehen und mit --refresh neu laden.")

    sfx = "_fast_gleich" if near_same else ""
    if near_same:
        mode = "fast gleiche Indizes"
    log(f"STUFE 1 - Auswahl ({mode}): {s0:%d.%m.%Y} - {f0:%d.%m.%Y}")
    spread_filter = pair_list is None and not near_same    # eigene Paarliste / fast gleiche: kein Spread-Filter
    stage1 = mark_small(screen(close[w1], volume[w1], aum, names, index_key, pair_list, tag="_stufe1",
                               spread_filter=spread_filter, near_same=near_same, groups=groups))
    save_csv(stage1, f"ranking{sfx}_stufe1.csv")
    cand = select_candidates(stage1)
    n1 = int(stage1["p_value"].notna().sum())
    log(f"    {len(cand)} Kandidaten fuer Stufe 2 (p <= {STAGE1_MAX_P} in Stufe 1, beta > 0, "
        f"max. {STAGE1_MAX_PER_ETF} Paare je ETF, max. {STAGE1_TOP_N})")

    print_header = lambda: (print(), print("=" * 78), print(f"ZUSAMMENFASSUNG (zweistufig, {mode})"),
                            print("=" * 78))
    if len(cand) == 0:
        save_csv(cand, f"ranking{sfx}_zweistufig.csv")
        print_header()
        print(f"Stufe 1 ({s0:%d.%m.%Y} - {f0:%d.%m.%Y}): {n1} Paare getestet, kein Kandidat mit "
              f"p <= {STAGE1_MAX_P}. Stufe 2 entfaellt.")
        print(f"\nErgebnisse: {OUT_DIR}")
        return

    log(f"STUFE 2 - Bestaetigung: {f0:%d.%m.%Y} - {last:%d.%m.%Y}")
    stage2 = screen(close[w2], volume[w2], aum, names, index_key, cand, tag="_stufe2",
                    spread_filter=spread_filter, near_same=near_same, groups=groups)
    stage2 = stage2.rename(columns={"p_value": "p_stufe2", "p_adj": "p_adj_stufe2", "beta": "beta_stufe2",
                                    "rang": "rang_stufe2", "status": "status_stufe2"})
    front = ["rang_stufe2", "etf_a", "etf_b", "name_a", "name_b"]
    front += [c for c in ("kategorie", "begruendung") if c in stage2.columns]
    front += ["rang_stufe1", "p_stufe1", "p_stufe2", "p_adj_stufe2", "signifikant", "status_stufe2",
              "beta_stufe1", "beta_stufe2"]
    stage2 = stage2[front + [c for c in stage2.columns if c not in front]]
    save_csv(stage2, f"ranking{sfx}_zweistufig.csv")

    tested = stage2[stage2["p_stufe2"].notna()]
    n_sig = int(tested["signifikant"].astype(bool).sum())
    print_header()
    print(f"Stufe 1 ({s0:%d.%m.%Y} - {f0:%d.%m.%Y}): {n1} Paare getestet -> {len(cand)} Kandidaten")
    print(f"Stufe 2 ({f0:%d.%m.%Y} - {last:%d.%m.%Y}): {len(tested)} Kandidaten getestet, "
          f"{n_sig} bestaetigt (BH-p <= {FDR_ALPHA} ueber {len(tested)} Tests, beta > 0)")
    if len(tested):
        need = FDR_ALPHA / len(tested)
        print(f"  noetiger p-Wert fuer das beste Paar in Stufe 2: {need:.2g} "
              f"(ohne Vorauswahl, bei {n1} Tests, waeren es ca. {FDR_ALPHA / max(n1, 1):.1g})")
    view = stage2[["etf_a", "etf_b", "name_a", "name_b", "rang_stufe1", "p_stufe1", "p_stufe2",
                   "p_adj_stufe2", "signifikant"]]
    print()
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    not_tested = stage2[stage2["p_stufe2"].isna()]
    if len(not_tested):
        print(f"\nIn Stufe 2 nicht testbar: {len(not_tested)} Kandidaten (Grund in Spalte 'status_stufe2')")
    print(f"\nErgebnisse: {OUT_DIR}")


def run_screen(universe: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
               aum: pd.Series, aum_meta: pd.DataFrame | None = None,
               pair_list: pd.DataFrame | None = None, two_stage: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    names = universe.set_index("ticker")["name"]
    typ = universe.set_index("ticker")["typ"] if "typ" in universe else pd.Series(dtype=object)
    index_key = aum_meta["index_key"] if aum_meta is not None and "index_key" in aum_meta else None
    stocks = MODE == "aktien"
    groups = None
    if stocks and aum_meta is not None and "gruppe" in aum_meta and STOCK_GROUP_BY:
        groups = aum_meta["gruppe"].where(aum_meta["gruppe"].notna(), "")
    size_name = "Marktkapitalisierung" if stocks else "Fondsvolumen"
    min_size = MIN_MARKET_CAP_USD if stocks else MIN_AUM_USD
    close = clean_prices(close)

    # --- Groessenfilter: Marktkapitalisierung (Aktien) bzw. Fondsvolumen (ETFs; ETCs optional ausgenommen)
    min_rows = int(252 * FORMATION_YEARS * MIN_COVERAGE)
    has_hist = close.columns[close.notna().sum() >= min_rows]
    aum = aum.reindex(close.columns)
    missing = [t for t in has_hist if pd.isna(aum.get(t))]
    save_csv(universe[universe["ticker"].isin(missing)], "missing_marktkap.csv" if stocks else "missing_aum.csv")
    exempt = typ.reindex(close.columns).eq("ETC") & (not AUM_FILTER_ETC)
    aum_ok = (aum >= min_size) | exempt
    if KEEP_MISSING_AUM or pair_list is not None:          # eigene Paare: fehlendes AUM nicht ausschliessen
        aum_ok = aum_ok | aum.isna()
    tradable = [t for t in has_hist if aum_ok.get(t, False)]
    log(f"Universum: {close.shape[1]} Instrumente mit Kursen, {len(has_hist)} mit >= {FORMATION_YEARS} J. "
        f"Historie, {len(missing)} davon ohne Angabe: {size_name}, {len(tradable)} nach Groessenfilter "
        f"({size_name} > {min_size / 1e6:.0f} Mio. USD"
        f"{'; ETCs ausgenommen' if not stocks and MARKET != 'us' and not AUM_FILTER_ETC else ''})")

    uni_out = pd.DataFrame({"ticker": tradable, "name": names.reindex(tradable).to_numpy(),
                            "typ": typ.reindex(tradable).to_numpy(),
                            ("marktkap_musd" if stocks else "aum_musd"): (aum.reindex(tradable) / 1e6).to_numpy()})
    if stocks and aum_meta is not None:
        for c in ("sector", "industry", "gruppe", "long_name"):
            if c in aum_meta:
                uni_out[c] = aum_meta[c].reindex(tradable).to_numpy()
        check = aum_meta.join(names.rename("name")).sort_values("aum_usd", ascending=False)
        check["marktkap_musd"] = check.pop("aum_usd") / 1e6
        save_csv(check.reset_index(names="ticker"), "aktien_details.csv")
    elif aum_meta is not None:
        uni_out["fund_ccy"] = aum_meta["fund_ccy"].reindex(tradable).to_numpy()
        uni_out["aum_quelle"] = aum_meta["aum_quelle"].reindex(tradable).to_numpy()
        check = aum_meta.join(names.rename("name")).sort_values("aum_usd", ascending=False)
        check["aum_musd"] = check.pop("aum_usd") / 1e6
        save_csv(check.reset_index(names="ticker"), "etf_details.csv")
    save_csv(uni_out, "universe_used.csv")

    if pair_list is not None:                              # Paare mit zu kleinem Fonds markieren
        small = set(has_hist) - set(tradable)
        pair_list = pair_list.copy()
        pair_list["_klein"] = [", ".join(t for t in (a, b) if t in small)
                               for a, b in zip(pair_list["ticker_a"], pair_list["ticker_b"])]
    close = close[tradable]
    volume = volume.reindex(index=close.index, columns=close.columns)

    # --- Suche: letzte FORMATION_YEARS Jahre bis zum letzten Kursdatum
    last = close.index[-1]
    f0 = last - pd.DateOffset(years=FORMATION_YEARS)
    mode = "Paarliste" if pair_list is not None else (
        f"alle Paare je {'Branche' if STOCK_GROUP_BY == 'industry' else 'Sektor'}"
        if groups is not None else "alle Kombinationen")
    mode = f"{'US-' if MARKET == 'us' else ''}{'Aktien' if stocks else 'ETFs'}, {mode}"
    log(f"Suche cointegrierte Paare ({mode}): {f0:%d.%m.%Y} - {last:%d.%m.%Y}")
    in_form = close.index > f0
    pl = pair_list.drop(columns="_klein") if pair_list is not None else None

    def mark_small(rk: pd.DataFrame) -> pd.DataFrame:
        if pair_list is None:
            return rk
        klein = dict(zip(zip(pair_list["ticker_a"], pair_list["ticker_b"]), pair_list["_klein"]))
        for k, r in rk.iterrows():
            kl = klein.get((r["etf_a"], r["etf_b"]), "")
            if kl and r["status"] != "getestet":
                rk.at[k, "status"] = f"{size_name} < {min_size / 1e6:.0f} Mio. USD: {kl}"
        return rk

    if two_stage:
        run_two_stage(close, volume, aum, names, index_key, pl, mark_small, mode, f0, last, groups=groups)
        if pair_list is None and not stocks:               # zusaetzlich: fast gleiche Indizes, zweistufig
            run_two_stage(close, volume, aum, names, index_key, None, mark_small, mode, f0, last,
                          near_same=True)
        return

    ranking = screen(close[in_form], volume[in_form], aum, names, index_key, pl,
                     spread_filter=pl is None, groups=groups)

    if pair_list is not None:
        ranking = mark_small(ranking)
        save_csv(ranking, "ranking_paarliste.csv")
    else:
        save_csv(ranking, "ranking.csv")
        save_csv(ranking[ranking["signifikant"]], "cointegrated_pairs.csv")

    # --- Zusammenfassung
    tested = ranking[ranking["p_value"].notna()]
    print()
    print("=" * 78)
    print(f"ZUSAMMENFASSUNG ({mode})")
    print("=" * 78)
    print(f"Zeitraum: {f0:%d.%m.%Y} - {last:%d.%m.%Y}")
    print(f"Getestete Paare: {len(tested)} | signifikant nach Korrektur (BH-p <= {FDR_ALPHA}): "
          f"{int(tested['signifikant'].astype(bool).sum())}")
    if len(tested):
        for thr in (0.001, 0.01, 0.05):
            print(f"  p < {thr:<5}: {int((tested['p_value'] < thr).sum()):6d}   "
                  f"(allein durch Zufall erwartet: {thr * len(tested):8.1f})")
        view = tested[["rang", "etf_a", "etf_b", "name_a", "name_b", "p_value", "p_adj", "beta"]].head(15)
        print()
        print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if pair_list is not None and (ranking["status"] != "getestet").any():
        print(f"\nNicht testbar: {int((ranking['status'] != 'getestet').sum())} Paare "
              f"(Grund in ranking_paarliste.csv, Spalte 'status')")

    # --- Zusatzauswertung (nur ETFs): fast gleiche Indizes mit breitem Spread (eigene Rangliste)
    if pair_list is None and not stocks:
        near = screen(close[in_form], volume[in_form], aum, names, index_key, None, near_same=True)
        save_csv(near, "ranking_fast_gleich.csv")
        print()
        print("-" * 78)
        print(f"FAST GLEICHE INDIZES mit Spread >= {NEAR_SAME_MIN_SPREAD:.1%} (rel. Tracking Error < {MIN_REL_TE})")
        print("-" * 78)
        print(f"Getestete Paare: {len(near)} | signifikant nach Korrektur (BH-p <= {FDR_ALPHA}): "
              f"{int(near['signifikant'].astype(bool).sum()) if len(near) else 0}")
        if len(near):
            for thr in (0.001, 0.01, 0.05):
                print(f"  p < {thr:<5}: {int((near['p_value'] < thr).sum()):6d}   "
                      f"(allein durch Zufall erwartet: {thr * len(near):8.1f})")
            view = near[["rang", "etf_a", "etf_b", "name_a", "name_b", "p_value", "p_adj", "beta",
                         "spread_std", "rel_te"]].head(15)
            print()
            print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nErgebnisse: {OUT_DIR}")


# ------------------------------------------------------------------ Zusatzauswertung: Mean Reversion
def mr_pair_series(la: np.ndarray, lb: np.ndarray) -> dict:
    """Handelsregel fuer ein Paar ueber die ganze Historie. la, lb: Log-Kurse (NaN erlaubt).

    Spread = log(A) - log(B), also gleicher Euro-Betrag long und short. z misst den Abstand vom
    gleitenden Mittel der letzten MR_LOOKBACK Tage in Standardabweichungen. Die Position von Tag t
    wird mit dem Schlusskurs von t festgelegt und verdient erst die Rendite von t+1 (kein Look-Ahead).
    Rendite je Tag bezogen auf das eingesetzte Kapital (je die Haelfte long und short).
    """
    s = pd.Series(la - lb)
    m = s.rolling(MR_LOOKBACK, min_periods=MR_LOOKBACK).mean()
    sd = s.rolling(MR_LOOKBACK, min_periods=MR_LOOKBACK).std()
    z = ((s - m) / sd).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    cur, held = 0, 0
    for t in range(n):
        zt = z[t]
        if zt != zt:                                        # NaN: kein Kurs -> glattstellen
            cur, held = 0, 0
        elif cur == 0:
            if zt > MR_ENTRY_Z:
                cur, held = -1, 0
            elif zt < -MR_ENTRY_Z:
                cur, held = 1, 0
        else:
            held += 1
            if (cur == 1 and zt >= -MR_EXIT_Z) or (cur == -1 and zt <= MR_EXIT_Z) or held >= MR_MAX_HOLD:
                cur, held = 0, 0
        pos[t] = cur
    ra = np.concatenate([[0.0], np.expm1(np.diff(la))])
    rb = np.concatenate([[0.0], np.expm1(np.diff(lb))])
    spread_ret = 0.5 * (np.nan_to_num(ra) - np.nan_to_num(rb))
    prev = np.concatenate([[0.0], pos[:-1]])
    gross = prev * spread_ret
    costs = np.abs(np.diff(np.concatenate([[0.0], pos]))) * MR_COST_BPS / 1e4
    borrow = np.abs(prev) * 0.5 * MR_BORROW_FEE / 252
    return {"pos": pos, "gross": gross, "net": gross - costs - borrow}


def mr_stats(net: np.ndarray, pos: np.ndarray, gross: np.ndarray | None = None) -> dict:
    """Kennzahlen eines Zeitraums: Sharpe (annualisiert), Rendite p. a., Trades, Trefferquote, Haltedauer."""
    out = {"sharpe": np.nan, "rendite_pa": np.nan, "trades": 0, "trefferquote": np.nan, "haltedauer": np.nan}
    if len(net) < 20:
        return out
    sd = net.std()
    out["sharpe"] = float(np.sqrt(252) * net.mean() / sd) if sd > 0 else np.nan
    out["rendite_pa"] = float(net.mean() * 252)
    if gross is not None:
        gs = gross.std()
        out["sharpe_vor_kosten"] = float(np.sqrt(252) * gross.mean() / gs) if gs > 0 else np.nan
    # einzelne Trades: Bloecke mit Position != 0; die Rendite faellt jeweils am Folgetag an
    prev = np.concatenate([[0.0], pos[:-1]])
    start = np.where((pos != 0) & (prev != pos))[0]
    pnl, hold = [], []
    for st in start:
        end = st
        while end + 1 < len(pos) and pos[end + 1] == pos[st]:
            end += 1
        pnl.append(net[st:min(end + 2, len(net))].sum())
        hold.append(end - st + 1)
    if pnl:
        out["trades"] = len(pnl)
        out["trefferquote"] = float(np.mean(np.array(pnl) > 0))
        out["haltedauer"] = float(np.mean(hold))
    return out


def mr_candidates(px: pd.DataFrame, vol: pd.DataFrame, names: pd.Series, groups: pd.Series | None,
                  pair_list: pd.DataFrame | None) -> list[tuple[str, str]]:
    """Zulaessige Paare eines Auswahlzeitraums: liquide Titel, gleiche Gruppe bzw. genug Korrelation."""
    lp, _, _ = prepare_window(px, vol, names, quiet=True)
    cols = list(lp.columns)
    ok = set(cols)
    if pair_list is not None:
        return [(a, b) for a, b in zip(pair_list["ticker_a"], pair_list["ticker_b"]) if a in ok and b in ok]
    if len(cols) < 2:
        return []
    corr = np.corrcoef(lp.diff().fillna(0.0).to_numpy(), rowvar=False)
    iu, ju = np.triu_indices(len(cols), k=1)
    c = corr[iu, ju]
    min_corr = STOCK_MIN_RET_CORR if MODE == "aktien" else MIN_RET_CORR
    keep = (c >= min_corr) & (c <= MAX_RET_CORR)
    if groups is not None:
        g = groups.reindex(cols).to_numpy(dtype=object)
        keep &= np.array([isinstance(x, str) and x != "" and x == y for x, y in zip(g[iu], g[ju])], dtype=bool)
    if MODE == "etf":                                       # (fast) gleiche Indizes: nur Kursrauschen
        _, rel = monthly_te_stats(lp)
        keep &= ~(rel[iu, ju] < MIN_REL_TE)
    order = np.argsort(-c[keep])[:MR_MAX_PAIRS]
    return [(cols[i], cols[j]) for i, j in zip(iu[keep][order], ju[keep][order])]


def mr_top_development(pairs: pd.DataFrame, series: dict, idx: pd.DatetimeIndex, windows: list,
                       r_all: pd.Series) -> None:
    """Die besten Paare (nach Sharpe im Gesamtzeitraum) Jahr fuer Jahr: Rendite, Sharpe, Trend.

    Die Paare sind im Rueckblick ausgewaehlt. Zum Vergleich stehen darunter der Durchschnitt dieser
    Paare und der Durchschnitt aller zulaessigen Paare.
    """
    top = pairs[pairs["sharpe_nach_kosten"].notna()].head(MR_SHOW_TOP)
    if top.empty:
        return
    labels = [f"{t0:%Y}/{t1:%y}" for _, t0, t1 in windows]
    masks = [np.asarray((idx > t0) & (idx <= t1)) for _, t0, t1 in windows]
    in_all = np.asarray(idx > windows[0][1])
    n_recent = min(MR_TREND_YEARS, len(windows) - 1)
    recent = np.asarray(idx > windows[-n_recent][1]) if n_recent else np.zeros(len(idx), dtype=bool)
    early = in_all & ~recent
    half = np.asarray(idx > idx[-1] - pd.DateOffset(months=6))

    def sharpe(x: np.ndarray) -> float:
        sd = x.std() if len(x) > 20 else 0.0
        return float(np.sqrt(252) * x.mean() / sd) if sd > 0 else np.nan

    def describe(net: np.ndarray) -> dict:
        row = {f"rendite_{lab}": float(net[m].sum()) for lab, m in zip(labels, masks)}
        row.update({f"sharpe_{lab}": sharpe(net[m]) for lab, m in zip(labels, masks)})
        row["rendite_letzte_6_monate"] = float(net[half].sum())
        if n_recent:
            a, b = sharpe(net[early]), sharpe(net[recent])
            row["sharpe_frueher"], row["sharpe_zuletzt"] = a, b
            d = b - a
            row["entwicklung"] = ("" if d != d else "zunehmend" if d > MR_TREND_MIN_DIFF
                                  else "abnehmend" if d < -MR_TREND_MIN_DIFF else "gleichbleibend")
        return row

    nets, rows = {}, []
    for rank, r in enumerate(top.itertuples(index=False), 1):
        net = series[(r.etf_a, r.etf_b)]["net"]
        nets[f"{r.etf_a}/{r.etf_b}"] = net
        rows.append({"rang": rank, "etf_a": r.etf_a, "etf_b": r.etf_b, "name_a": r.name_a, "name_b": r.name_b,
                     "gruppe": r.gruppe, "sharpe_gesamt": r.sharpe_nach_kosten, **describe(net)})
    top_net = np.mean(list(nets.values()), axis=0)
    all_net = r_all.reindex(idx).fillna(0.0).to_numpy()
    for label, net in ((f"Top {len(top)} zusammen", top_net), ("alle Paare (Durchschnitt)", all_net)):
        rows.append({"rang": "", "etf_a": label, "etf_b": "", "name_a": "", "name_b": "", "gruppe": "",
                     "sharpe_gesamt": sharpe(net[in_all]), **describe(net)})
    out = pd.DataFrame(rows)
    save_csv(out, "mr_top_paare.csv")
    curves = pd.DataFrame({k: np.cumsum(np.where(in_all, v, 0.0)) for k, v in nets.items()}, index=idx)
    curves[f"Top {len(top)} zusammen"] = np.cumsum(np.where(in_all, top_net, 0.0))
    curves["alle Paare"] = np.cumsum(np.where(in_all, all_net, 0.0))
    curves = curves[in_all]
    save_csv(curves.rename_axis("datum"), "mr_top_paare_verlauf.csv", index=True)

    name = [f"{r['etf_a']}/{r['etf_b']}" if r["etf_b"] else r["etf_a"] for r in rows]
    width = max(len(x) for x in name) + 2
    print()
    print("-" * 78)
    print(f"ENTWICKLUNG DER {len(top)} BESTEN PAARE (nach Sharpe im Gesamtzeitraum, im Rueckblick ausgewaehlt)")
    print("-" * 78)
    for title, key, fmt in (("Rendite je Testjahr nach Kosten (in %)", "rendite_", lambda v: f"{v * 100:+.1f}"),
                            ("Sharpe Ratio je Testjahr nach Kosten", "sharpe_", lambda v: f"{v:.2f}")):
        cols = [f"{key}{lab}" for lab in labels]
        extra = (["rendite_letzte_6_monate"] if key == "rendite_"
                 else ["sharpe_frueher", "sharpe_zuletzt", "entwicklung"] if n_recent else [])
        head = labels + (["6 Mon."] if key == "rendite_" else ["frueher", "zuletzt", "Entwicklung"] if n_recent else [])
        print(f"\n{title}")
        print(" " * width + "".join(f"{h:>9}" for h in head[:len(cols) + 2])
              + (f"  {head[-1]}" if len(head) > len(cols) + 2 else ""))
        for i, r in enumerate(rows):
            if i == len(top):
                print(" " * width + "-" * (9 * (len(cols) + 2)))
            vals = [r[c] for c in cols + extra]
            cells = "".join(f"{(fmt(v) if v == v else '-'):>9}" for v in vals if not isinstance(v, str))
            print(f"{name[i]:<{width}}" + cells + "".join(f"  {v}" for v in vals if isinstance(v, str)))
    if n_recent:
        cnt = out["entwicklung"].iloc[:len(top)].value_counts()
        print(f"\n'zuletzt' = die letzten {n_recent} Testjahre (ab {windows[-n_recent][1]:%d.%m.%Y}), 'frueher' = die Jahre "
              f"davor. Von den {len(top)} Paaren: "
              + ", ".join(f"{cnt.get(k, 0)} {k}" for k in ("zunehmend", "gleichbleibend", "abnehmend")) + ".")
    print("Hinweis: Diese Paare wurden ausgesucht, WEIL sie im Gesamtzeitraum am besten liefen. Unter tausenden\n"
          "Paaren sehen die besten auch dann gut aus, wenn nur Zufall im Spiel ist. Ein Paar, das zuletzt\n"
          "nachlaesst, ist ein Warnsignal; ein Paar, das sich haelt, ist noch kein Beweis.")
    mr_top_chart(curves, out, n_recent, windows)

    # Gegenprobe ohne Rueckblick: die besten Paare der frueheren Jahre - wie liefen sie danach?
    if n_recent and early.sum() > 250:
        cand = sorted(((sharpe(v["net"][early]), k) for k, v in series.items()
                       if np.count_nonzero(v["pos"][early]) >= 20), key=lambda x: -(x[0] if x[0] == x[0] else -9))
        cand = [(sh, k) for sh, k in cand if sh == sh][:MR_SHOW_TOP]
        if cand:
            yrs = recent.sum() / 252
            chk = pd.DataFrame([{"etf_a": k[0], "etf_b": k[1], "sharpe_frueher": sh,
                                 "sharpe_zuletzt": sharpe(series[k]["net"][recent]),
                                 "rendite_pa_frueher": float(series[k]["net"][early].mean() * 252),
                                 "rendite_pa_zuletzt": float(series[k]["net"][recent].sum() / yrs)} for sh, k in cand])
            both = np.mean([series[k]["net"] for _, k in cand], axis=0)
            save_csv(chk, "mr_top_paare_gegenprobe.csv")
            print()
            print("-" * 78)
            print(f"GEGENPROBE: die {len(chk)} besten Paare bis {windows[-n_recent][1]:%d.%m.%Y} - und wie sie danach liefen")
            print("-" * 78)
            print(f"{'':<{width}}{'Sharpe':>9}{'Sharpe':>9}{'Rendite':>10}{'Rendite':>10}")
            print(f"{'':<{width}}{'frueher':>9}{'zuletzt':>9}{'frueher':>10}{'zuletzt':>10}   (Rendite p. a. nach Kosten)")
            for r in chk.itertuples(index=False):
                z = f"{r.sharpe_zuletzt:.2f}" if r.sharpe_zuletzt == r.sharpe_zuletzt else "-"
                print(f"{r.etf_a + '/' + r.etf_b:<{width}}{r.sharpe_frueher:>9.2f}{z:>9}"
                      f"{r.rendite_pa_frueher * 100:>+9.1f}%{r.rendite_pa_zuletzt * 100:>+9.1f}%")
            print(" " * width + "-" * 38)
            print(f"{'zusammen':<{width}}{sharpe(both[early]):>9.2f}{sharpe(both[recent]):>9.2f}"
                  f"{both[early].mean() * 252 * 100:>+9.1f}%{both[recent].sum() / yrs * 100:>+9.1f}%")
            print(f"{'alle Paare (Durchschnitt)':<{width}}{sharpe(all_net[early]):>9.2f}{sharpe(all_net[recent]):>9.2f}"
                  f"{all_net[early].mean() * 252 * 100:>+9.1f}%{all_net[recent].sum() / yrs * 100:>+9.1f}%")
            print("Diese Paare wurden nur mit den frueheren Jahren ausgesucht; 'zuletzt' sind fuer sie neue Daten.\n"
                  "Bleibt dort wenig uebrig, war das fruehere Ergebnis ueberwiegend Zufall.")


def mr_top_chart(curves: pd.DataFrame, table: pd.DataFrame, n_recent: int, windows: list) -> None:
    """Aufgelaufene Rendite der besten Paare als kleine Einzeldiagramme -> mr_top_paare.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        log("Diagramm mr_top_paare.png uebersprungen (matplotlib fehlt: pip install matplotlib)")
        return
    pair_cols = [c for c in curves.columns if c != "alle Paare"]
    n = len(pair_cols)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    surface, ink, muted, blue, gray = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#a3a29c"
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow + 0.9), sharex=True, sharey=True,
                             squeeze=False, facecolor=surface)
    for k, ax in enumerate(axes.ravel()):
        ax.set_facecolor(surface)
        if k >= n:
            ax.set_visible(False)
            continue
        col = pair_cols[k]
        if n_recent:
            ax.axvspan(windows[-n_recent][1], curves.index[-1], color="#f0efec", lw=0)
        ax.axhline(0, color=muted, lw=0.8)
        ax.plot(curves.index, curves["alle Paare"], color=gray, lw=1.2, label="alle Paare (Durchschnitt)")
        ax.plot(curves.index, curves[col], color=blue, lw=2.0,
                label="Durchschnitt dieser Paare" if col.startswith("Top ") else "Paar")
        row = table.iloc[k]
        sub = ""
        if n_recent and row["sharpe_frueher"] == row["sharpe_frueher"] and row["sharpe_zuletzt"] == row["sharpe_zuletzt"]:
            sub = f"Sharpe {row['sharpe_frueher']:.1f} -> {row['sharpe_zuletzt']:.1f}  ({row['entwicklung']})"
        ax.set_title((col if col.startswith("Top ") else f"{k + 1}. {col.replace('/', ' / ')}") + ("\n" + sub if sub else ""),
                     fontsize=10, color=ink, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f} %"))
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    if n_recent:
        from matplotlib.patches import Patch
        handles.append(Patch(color="#f0efec"))
        labels_.append(f"letzte {n_recent} Testjahre ('zuletzt')")
    fig.legend(handles, labels_, loc="upper right", ncol=len(handles), frameon=False, fontsize=9, labelcolor=muted)
    fig.suptitle("Aufgelaufene Rendite nach Kosten: die besten Paare im Gesamtzeitraum (im Rueckblick ausgewaehlt)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / "mr_top_paare.png"
    try:
        fig.savefig(path, dpi=130, facecolor=surface)
    except PermissionError:
        path = OUT_DIR / f"mr_top_paare_{time.strftime('%H%M%S')}.png"
        fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    log(f"Diagramm gespeichert: {path.name}")


def run_mean_reversion(universe: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, aum: pd.Series,
                       aum_meta: pd.DataFrame | None = None, pair_list: pd.DataFrame | None = None) -> None:
    """Rollierender Test: Paare anhand der Vorjahre auswaehlen, im Folgejahr handeln (nach Kosten)."""
    OUT_DIR.mkdir(exist_ok=True)
    stocks = MODE == "aktien"
    names = universe.set_index("ticker")["name"]
    typ = universe.set_index("ticker")["typ"] if "typ" in universe else pd.Series(dtype=object)
    groups = None
    if stocks and aum_meta is not None and "gruppe" in aum_meta and STOCK_GROUP_BY:
        groups = aum_meta["gruppe"].where(aum_meta["gruppe"].notna(), "")
    close = clean_prices(close)
    aum = aum.reindex(close.columns)
    min_size = MIN_MARKET_CAP_USD if stocks else MIN_AUM_USD
    size_ok = (aum >= min_size) | (typ.reindex(close.columns).eq("ETC") & (not AUM_FILTER_ETC))
    if KEEP_MISSING_AUM or pair_list is not None:
        size_ok = size_ok | aum.isna()
    close = close.loc[:, size_ok.reindex(close.columns, fill_value=False)]
    # Brueche in den Kursdaten: Yahoo verkettet z. B. alte und neue Aktien nach einer Insolvenz. Ein solcher
    # Sprung waere im Test ein riesiger Scheingewinn, deshalb zaehlt nur die Historie ab dem letzten Bruch.
    step = np.log(close.ffill()).diff().abs() > np.log(MR_MAX_DAILY_MOVE)
    breaks = []
    close = close.copy()
    for t in step.columns[step.any()]:
        d = step.index[step[t]][-1]
        breaks.append({"ticker": t, "name": names.get(t, ""), "letzter_bruch": d.date(),
                       "anzahl_brueche": int(step[t].sum())})
        close.loc[close.index < d, t] = np.nan
    if breaks:
        save_csv(pd.DataFrame(breaks), "mr_kursbrueche.csv")
        log(f"{len(breaks)} Titel mit Bruch in den Kursdaten (Tagesbewegung ueber Faktor {MR_MAX_DAILY_MOVE:g}): "
            f"Historie davor nicht verwendet -> mr_kursbrueche.csv")
    volume = volume.reindex(index=close.index, columns=close.columns)
    logp = np.log(close)
    idx = close.index
    first, last = idx[0], idx[-1]

    # Testjahre vom letzten Datum rueckwaerts; davor jeweils MR_FORMATION_YEARS Auswahlzeitraum
    windows = []
    t1 = last
    while True:
        t0 = t1 - pd.DateOffset(years=MR_TEST_YEARS)
        f0 = t0 - pd.DateOffset(years=MR_FORMATION_YEARS)
        if f0 < first:
            break
        windows.append((f0, t0, t1))
        t1 = t0
    windows = windows[::-1]
    if not windows:
        raise SystemExit("Zu wenig Kurshistorie fuer die Mean-Reversion-Auswertung.")
    log(f"Mean-Reversion-Auswertung: {len(windows)} Testjahre, {windows[0][1]:%d.%m.%Y} - {last:%d.%m.%Y} "
        f"(Auswahl jeweils aus den {MR_FORMATION_YEARS} Jahren davor)")

    series: dict[tuple[str, str], dict] = {}

    def pair_series(a: str, b: str) -> dict:
        if (a, b) not in series:
            series[(a, b)] = mr_pair_series(logp[a].to_numpy(), logp[b].to_numpy())
        return series[(a, b)]

    year_rows, pick_rows = [], []
    port_all, port_top, port_all_gross, port_top_gross = [], [], [], []
    for f0, t0, t1 in windows:
        in_form = (idx > f0) & (idx <= t0)
        in_test = (idx > t0) & (idx <= t1)
        cands = mr_candidates(close[in_form], volume[in_form], names, groups, pair_list)
        form = []
        for a, b in cands:
            ps_ = pair_series(a, b)
            st = mr_stats(ps_["net"][in_form], ps_["pos"][in_form])
            form.append((a, b, st["sharpe"], st["trades"]))
        form = pd.DataFrame(form, columns=["a", "b", "sharpe_vorjahre", "trades_vorjahre"])
        good = form[(form["trades_vorjahre"] >= MR_MIN_TRADES) & (form["sharpe_vorjahre"] > 0)]
        good = good.sort_values("sharpe_vorjahre", ascending=False)
        used: dict[str, int] = {}
        top = []
        for _, r in good.iterrows():
            if used.get(r.a, 0) >= MR_MAX_PER_STOCK or used.get(r.b, 0) >= MR_MAX_PER_STOCK:
                continue
            top.append((r.a, r.b, r.sharpe_vorjahre))
            used[r.a] = used.get(r.a, 0) + 1
            used[r.b] = used.get(r.b, 0) + 1
            if len(top) == MR_TOP_N:
                break
        test_idx = idx[in_test]
        all_ret = (np.mean([pair_series(a, b)["net"][in_test] for a, b in cands], axis=0)
                   if cands else np.zeros(in_test.sum()))
        top_ret = (np.mean([pair_series(a, b)["net"][in_test] for a, b, _ in top], axis=0)
                   if top else np.zeros(in_test.sum()))
        port_all.append(pd.Series(all_ret, index=test_idx))
        port_top.append(pd.Series(top_ret, index=test_idx))
        port_all_gross.append(pd.Series(np.mean([pair_series(a, b)["gross"][in_test] for a, b in cands], axis=0)
                                        if cands else np.zeros(in_test.sum()), index=test_idx))
        port_top_gross.append(pd.Series(np.mean([pair_series(a, b)["gross"][in_test] for a, b, _ in top], axis=0)
                                        if top else np.zeros(in_test.sum()), index=test_idx))
        for a, b, shf in top:
            ps_ = pair_series(a, b)
            st = mr_stats(ps_["net"][in_test], ps_["pos"][in_test], ps_["gross"][in_test])
            pick_rows.append({"testjahr_ab": t0.date(), "etf_a": a, "etf_b": b, "name_a": names.get(a, ""),
                              "name_b": names.get(b, ""), "sharpe_vorjahre": shf,
                              "sharpe_testjahr": st["sharpe"], "rendite_testjahr": float(ps_["net"][in_test].sum()),
                              "trades_testjahr": st["trades"], "trefferquote_testjahr": st["trefferquote"]})
        sa, st_ = mr_stats(all_ret, np.zeros(len(all_ret))), mr_stats(top_ret, np.zeros(len(top_ret)))
        year_rows.append({"testjahr_ab": t0.date(), "testjahr_bis": t1.date(), "paare_zulaessig": len(cands),
                          "paare_ausgewaehlt": len(top),
                          "alle_rendite": float(all_ret.sum()), "alle_sharpe": sa["sharpe"],
                          "auswahl_rendite": float(top_ret.sum()), "auswahl_sharpe": st_["sharpe"]})
        log(f"  Testjahr ab {t0:%d.%m.%Y}: {len(cands)} Paare zulaessig, {len(top)} ausgewaehlt | "
            f"alle Paare {all_ret.sum():+.1%}, Auswahl {top_ret.sum():+.1%} (nach Kosten)")

    if not series:
        log("Kein Paar war in irgendeinem Testjahr zulaessig (Kurse, Historie, Groesse oder Umsatz fehlen) - "
            "keine Auswertung moeglich.")
        return
    years = pd.DataFrame(year_rows)
    picks = pd.DataFrame(pick_rows)
    r_all, r_top = pd.concat(port_all), pd.concat(port_top)
    g_all, g_top = pd.concat(port_all_gross), pd.concat(port_top_gross)
    save_csv(years, "mr_jahre.csv")
    save_csv(picks, "mr_auswahl.csv")
    save_csv(pd.DataFrame({"alle_paare": r_all, "auswahl": r_top}).rename_axis("datum"),
             "mr_portfolio_renditen.csv", index=True)

    # Kennzahlen je Paar ueber den gesamten Testzeitraum (alle Paare, die je zulaessig waren)
    in_all = idx > windows[0][1]
    n_years = max(in_all.sum() / 252, 1e-9)
    rows = []
    for (a, b), ps_ in series.items():
        st = mr_stats(ps_["net"][in_all], ps_["pos"][in_all], ps_["gross"][in_all])
        yearly = [ps_["net"][(idx > t0) & (idx <= t1)].sum() for _, t0, t1 in windows]
        rows.append({"etf_a": a, "etf_b": b, "name_a": names.get(a, ""), "name_b": names.get(b, ""),
                     "gruppe": groups.get(a, "") if groups is not None else "",
                     "sharpe_nach_kosten": st["sharpe"], "sharpe_vor_kosten": st.get("sharpe_vor_kosten"),
                     "rendite_pa": st["rendite_pa"], "trades_pro_jahr": st["trades"] / n_years,
                     "trefferquote": st["trefferquote"], "haltedauer_tage": st["haltedauer"],
                     "anteil_positive_jahre": float(np.mean(np.array(yearly) > 0)),
                     "wie_oft_ausgewaehlt": int(((picks["etf_a"] == a) & (picks["etf_b"] == b)).sum())
                     if len(picks) else 0})
    pairs = pd.DataFrame(rows).sort_values("sharpe_nach_kosten", ascending=False)
    save_csv(pairs, "mr_paare.csv")

    def total(r: pd.Series) -> tuple[float, float, float, float]:
        sd = r.std()
        sh = np.sqrt(252) * r.mean() / sd if sd > 0 else np.nan
        eq = (1 + r).cumprod()
        return sh, r.mean() * 252, (eq / eq.cummax() - 1).min(), sh * np.sqrt(len(r) / 252)

    sh_a, ret_a, dd_a, t_a = total(r_all)
    sh_t, ret_t, dd_t, t_t = total(r_top)
    print()
    print("=" * 78)
    print(f"MEAN-REVERSION-AUSWERTUNG ({'US-' if MARKET == 'us' else ''}{'Aktien' if stocks else 'ETFs'}"
          f"{', Paarliste' if pair_list is not None else ''})")
    print("=" * 78)
    print(f"Regel: Spread = log(A) - log(B); Einstieg bei |z| > {MR_ENTRY_Z} gegenueber dem {MR_LOOKBACK}-Tage-Mittel, "
          f"Ausstieg am Mittel\n       oder nach {MR_MAX_HOLD} Tagen. Kosten {MR_COST_BPS} Basispunkte je Kauf/Verkauf, "
          f"Leihgebuehr {MR_BORROW_FEE:.0%} p. a.")
    print(f"Auswahl: je Testjahr die {MR_TOP_N} Paare mit dem besten Ergebnis der {MR_FORMATION_YEARS} Vorjahre.\n")
    view = years.copy()
    for c in ("alle_rendite", "auswahl_rendite"):
        view[c] = (view[c] * 100).map(lambda v: f"{v:+.1f} %")
    print(view.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nGesamt ueber {len(windows)} Testjahre (nach Kosten, auf neuen Daten):")
    print(f"  alle zulaessigen Paare : Sharpe {sh_a:5.2f} | Rendite p. a. {ret_a:+.1%} | max. Verlust {dd_a:.1%} | "
          f"t-Wert {t_a:.1f} | vor Kosten {g_all.mean() * 252:+.1%} p. a.")
    print(f"  Auswahl nach Vorjahren : Sharpe {sh_t:5.2f} | Rendite p. a. {ret_t:+.1%} | max. Verlust {dd_t:.1%} | "
          f"t-Wert {t_t:.1f} | vor Kosten {g_top.mean() * 252:+.1%} p. a.")
    print("  (t-Wert ueber etwa 2 = kaum noch mit Zufall erklaerbar; Renditen bezogen auf das eingesetzte Kapital)")
    if len(pairs):
        pos_share = float((pairs["sharpe_nach_kosten"] > 0).mean())
        gross_share = float((pairs["sharpe_vor_kosten"] > 0).mean())
        print(f"\nEinzelne Paare ({len(pairs)}): {gross_share:.0%} vor Kosten im Plus, {pos_share:.0%} nach Kosten; "
              f"mittlere Trefferquote {pairs['trefferquote'].mean():.0%}, "
              f"Haltedauer {pairs['haltedauer_tage'].mean():.0f} Tage")
    mr_top_development(pairs, series, idx, windows, r_all)
    print(f"\nErgebnisse: {OUT_DIR}  (mr_jahre.csv, mr_auswahl.csv, mr_paare.csv, mr_portfolio_renditen.csv,\n"
          f"            mr_top_paare.csv, mr_top_paare_verlauf.csv, mr_top_paare.png, mr_top_paare_gegenprobe.csv)")


# ------------------------------------------------------------------ Demo-Daten
def make_demo_data(seed: int = 7):
    """Synthetische 'ETFs' zum Testen der Pipeline:
    CA/CB  6 echte cointegrierte Paare (sollen gefunden werden)
    RW     30 nur korrelierte Titel (sollen NICHT gefunden werden)
    SA/SB  4 cointegrierte Paare mit zu kleinem Fondsvolumen (AUM-Filter)
    NQ     3 ETFs auf denselben Index mit leichtem Kursrauschen (Tracking-Error-Filter)
    ST     1 selten gehandelter Zwilling von CB0 mit veralteten Kursen (Liquiditaetsfilter)
    JX/JY  gleicher Referenzindex laut Metadaten, aber verrauscht (Index-Filter)
    DE0    Kopie von CA1 mit fehlerhaftem Kursblock (Datenfehler-Erkennung)
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * HISTORY_YEARS)
    n = len(idx)
    market = np.cumsum(rng.normal(0.0003, 0.010, n))
    series, names, aum, keys = {}, {}, {}, {}

    def add(t, logp, name, a, key=None):
        a = a * (1 + 0.0007 * len(series))                  # eindeutiges Fondsvolumen je Demo-ETF
        series[t], names[t], aum[t], keys[t] = np.exp(logp), name, a, key

    def ou(half_life, sigma):
        x = np.zeros(n)
        theta = np.log(2) / half_life
        for t in range(1, n):
            x[t] = x[t - 1] * (1 - theta) + rng.normal(0, sigma)
        return x

    for g in range(6):                                   # cointegrierte Paare
        base = market + np.cumsum(rng.normal(0, 0.007, n))
        beta = rng.uniform(0.7, 1.4)
        add(f"CB{g}.DE", base + rng.normal(0, 0.0005, n), f"DEMO COINT B{g}", 5e8)
        add(f"CA{g}.DE", 0.3 + beta * base + ou(rng.uniform(8, 30), 0.004), f"DEMO COINT A{g}", 5e8)
    for g in range(30):                                   # nur korreliert, nicht cointegriert
        add(f"RW{g}.DE", market + np.cumsum(rng.normal(0, 0.006, n)), f"DEMO RANDOM {g}", 3e8)
    for g in range(4):                                    # cointegriert, aber zu klein (AUM)
        base = market + np.cumsum(rng.normal(0, 0.007, n))
        add(f"SB{g}.DE", base, f"DEMO SMALL B{g}", 4e7)
        add(f"SA{g}.DE", base + ou(14, 0.004), f"DEMO SMALL A{g}", 4e7)
    nq = market + np.cumsum(rng.normal(0.0002, 0.008, n))  # gleicher Index, 3 Anbieter
    for g in range(3):
        add(f"NQ{g}.DE", 0.5 * g + nq + ou(6, 0.0015), f"DEMO NASDAQ ETF {g}", 2e9)
    stale = np.log(series["CB0.DE"]) + ou(10, 0.003)       # selten gehandelt: Kurs nur an 40 % der Tage
    stale[rng.random(n) > 0.4] = np.nan
    add("ST0.DE", pd.Series(stale).ffill().bfill().to_numpy(), "DEMO ILLIQUIDE", 3e8)
    bad = np.log(series["CA1.DE"]).copy()                   # Datenfehler: Kursblock +13 % fuer 20 Tage
    bad[n - 120:n - 100] += 0.13
    add("DE0.DE", bad, "DEMO DATENFEHLER", 5e8)
    jp = market + np.cumsum(rng.normal(0, 0.008, n))       # gleicher Index laut Yahoo, verrauscht
    add("JX0.DE", jp + ou(12, 0.006), "DEMO JAPAN A", 1e9, "MSCIJAPAN")
    add("JY0.DE", jp + ou(12, 0.006), "DEMO JAPAN B", 1e9, "MSCIJAPAN")

    close = pd.DataFrame(series, index=idx)
    close.iloc[: 252 * 4, close.columns.get_loc("RW29.DE")] = np.nan   # spaeter aufgelegter ETF
    volume = (1.2 * max(min_turnover(), 250_000) / close).round()      # Umsatz knapp ueber der Schwelle
    unchanged = close["ST0.DE"].diff() == 0
    volume.loc[unchanged, "ST0.DE"] = 0
    volume["ST0.DE"] = volume["ST0.DE"] * 0.2
    universe = pd.DataFrame({"ticker": list(names), "name": list(names.values()), "isin": "", "typ": "ETF",
                             "segment": ""})
    aum_s = pd.Series(aum, dtype=float)
    meta = pd.DataFrame({"total_assets": aum_s, "fund_ccy": "EUR", "aum_quelle": "demo",
                         "aum_usd": aum_s, "benchmark": pd.Series(keys), "index_key": pd.Series(keys),
                         "index_quelle": "demo"})
    if MODE == "aktien":                                    # Demo-Branchen: Paar-Partner in derselben Branche
        tick = list(names)

        def demo_group(t: str) -> str:
            digits = "".join(ch for ch in t.split(".")[0] if ch.isdigit()) or "0"
            if t[:2] in ("CA", "CB", "SA", "SB"):
                return f"Demo-Branche {int(digits) % 3 + 1}"
            return f"Demo-Branche {int(digits) % 5 + 1}"

        meta = pd.DataFrame({"market_cap": aum_s * 3, "currency": "EUR", "sector": "Demo-Sektor",
                             "industry": [demo_group(t) for t in tick], "long_name": list(names.values()),
                             "aum_usd": aum_s * 3}, index=tick)
        meta["gruppe"] = meta["industry"] if STOCK_GROUP_BY else ""
        universe["typ"] = "Aktie"
        aum_s = meta["aum_usd"]
    return universe, close, volume, aum_s, meta


# ------------------------------------------------------------------ main
def main() -> None:
    global OUT_DIR, MODE, MARKET, MR_COST_BPS
    ap = argparse.ArgumentParser(description="Screener fuer cointegrierte Paare (Aktien oder ETFs, Xetra oder USA)")
    ap.add_argument("--etf", action="store_true", help="ETF/ETC-Modus statt Aktien")
    ap.add_argument("--us", action="store_true", help="US-Markt (Nasdaq, NYSE u. a.) statt Xetra")
    ap.add_argument("--demo", action="store_true", help="Probelauf mit synthetischen Daten")
    ap.add_argument("--refresh", action="store_true", help="Cache ignorieren und alles neu laden")
    ap.add_argument("--pairs", nargs="?", const="", default=None, metavar="DATEI",
                    help="nur die Paare aus DATEI testen (Standard: pairs_aktien.csv, mit --etf pairs.csv)")
    ap.add_argument("--two-stage", "--zweistufig", dest="two_stage", action="store_true",
                    help="Kandidaten im frueheren Zeitraum auswaehlen, im spaeteren bestaetigen")
    ap.add_argument("--mean-reversion", "--mr", dest="mean_reversion", action="store_true",
                    help="Zusatzauswertung: Rueckkehr zum gleitenden Mittel, mit Kosten, rollierend getestet")
    args = ap.parse_args()

    MODE = "etf" if args.etf else "aktien"
    MARKET = "us"
    stocks, us = MODE == "aktien", MARKET == "us"
    OUT_DIR = BASE_DIR / ((f"output_us_{MODE}") if us else ("output_aktien" if stocks else "output"))
    if us:
        MR_COST_BPS = US_MR_COST_BPS
    log(f"Modus: {'US-' if us else 'Xetra-'}{'Aktien' if stocks else ('ETFs' if us else 'ETFs/ETCs')}")

    t_start = time.time()
    pair_list = None
    if args.pairs is not None:
        fname = args.pairs or ((f"pairs_us_{MODE}.csv") if us
                               else ("pairs_aktien.csv" if stocks else "pairs.csv"))
        path = Path(fname) if Path(fname).is_absolute() else BASE_DIR / fname
        if not path.exists():
            raise SystemExit(f"Paarliste {path} nicht gefunden")
        pair_list = load_pair_list(path)
        log(f"Paarliste {path.name}: {len(pair_list)} Paare")

    if args.demo:
        OUT_DIR = BASE_DIR / ((f"output_demo_us_{MODE}") if us
                              else ("output_demo_aktien" if stocks else "output_demo"))
        log(f"DEMO-Modus mit synthetischen Daten (Ergebnisse in {OUT_DIR.name})")
        universe, close, volume, aum, aum_meta = make_demo_data()
    else:
        universe = load_universe(args.refresh)
        tickers = universe["ticker"].tolist()
        if pair_list is not None:                          # Ticker der Liste, die nicht im Universum sind
            extra = sorted((set(pair_list["ticker_a"]) | set(pair_list["ticker_b"])) - set(tickers))
            if extra:
                log(f"{len(extra)} Ticker aus der Paarliste stehen nicht in der {'US' if us else 'Xetra'}-Liste: "
                    f"{', '.join(extra)} (werden trotzdem versucht)")
                universe = pd.concat([universe, pd.DataFrame({"ticker": extra, "name": "", "isin": "",
                                                              "typ": "Aktie" if stocks else "ETF",
                                                              "segment": ""})], ignore_index=True)
                tickers += extra
        if us and pair_list is not None:                   # Paarliste: nur deren Titel laden (schnell)
            tickers = sorted(set(pair_list["ticker_a"]) | set(pair_list["ticker_b"]))
        elif us:                                           # lange Historie nur fuer ausreichend liquide Titel
            tickers = us_prescreen(tickers, args.refresh)
        close, volume = load_prices(tickers, args.refresh)
        min_rows = int(252 * FORMATION_YEARS * MIN_COVERAGE)
        candidates = close.columns[close.notna().sum() >= min_rows].tolist()
        if stocks:
            aum, aum_meta = load_stock_info(candidates, args.refresh)
        else:
            aum, aum_meta = load_aum_usd(candidates, args.refresh)

    if args.mean_reversion:
        run_mean_reversion(universe, close, volume, aum, aum_meta, pair_list)
    else:
        run_screen(universe, close, volume, aum, aum_meta, pair_list, args.two_stage)
    log(f"Fertig nach {(time.time() - t_start) / 60:.1f} Minuten")


if __name__ == "__main__":
    main()
