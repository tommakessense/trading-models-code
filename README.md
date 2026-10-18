# Trading-Modelle

Zwei regelbasierte Handelsmodelle in Python, inspiriert durch Ernest P. Chan,
*Quantitative Trading*. Beide Modelle haben eine Rückrechnung (Backtest) und einen Protokollmodus fürs
Paper Trading.

> **Keine Anlageberatung.** Der Code dient der Forschung und dem Lernen. Rückrechnungen sind keine
> Garantie für künftige Ergebnisse. Wer damit handelt, tut das auf eigenes Risiko.

## Modelle

| Ordner | Modell | Markt |
|---|---|---|
| `hgv_mgm_modell/` | **Mean Reversion mit Aktienpaaren.** Handelt den Spread von 15 Paaren nach seinem z-Score. Konto mit ganzen Stücken, Gebühren von Interactive Brokers, Slippage, Leihgebühr und drei Hebel-Profilen (ohne Hebel, Margin-Konto, Viertel-Kelly mit individueller Beschränkung). | US-Aktien |
| `trend_modell/` | **Trendfolge.** Zwei robuste Grundstrategien:<br>Robust 1: neues 200-Tage-Hoch, 20 Handelstage halten.<br>Robust 2: neues 50-Tage-Hoch bei doppeltem Volumen, 60 Tage halten, nachgezogener Stopp 15 %.<br>Je höchstens 10 Positionen, Auswahl nach Momentum, Handel in der Xetra-Schlussauktion, Gebühren von Scalable Capital. | Xetra-Aktien |
| `pair_screener/` | **Screener.** Sucht cointegrierte Paare mit dem Engle-Granger-Test, wobei die erwartbaren, zufällig auftretenden Treffer mit der Benjamini-Hochberg-Prozedur erkannt werden sollen. Lädt außerdem die Xetra-Kurse, die das Trendmodell für den Backtest braucht. | Xetra, USA |

Zusätzliche Prüfungen im Trendmodell:

| Datei | Inhalt |
|---|---|
| `trend_modell.py` | Rückrechnung vieler Einstiegs- und Ausstiegsvarianten, optional mit Medienecho (GDELT). Dazu Placebo-Prüfung und Test des Survivorship-Bias mit Pleitefällen. |
| `trend_cpo.py` | Conditional Parameter Optimization (machine learning zur Optimierung der einzelnen Parameter des Modells vor jedem potenziellen Trade): Walk-forward mit Gradient Boosting, das je Signal den Ausstieg wählt. |
| `cpo_stresstest.py` | CPO gegen feste Regeln in simulierten Crashs und Volatilitätsschüben um den nur eingeschränkten Zeitraum des Backtests auf mehr Eventualitäten auszubreiten. |
| `trend_bewertung.py` | Bewertung der Varianten: Gewinnquellen, Plausibilität, probabilistische und deflationierte Sharpe Ratio, White's Reality Check, Abhängigkeit vom Startdatum. |
| `trend_positionsgroesse.py` | Feste Positionsgröße (1/10) gegen eine Größe je nach Signalqualität (CPO und einfache Regeln). |
| `trend_ausfuehrung.py` | Ausführungszeitpunkt im Vergleich: Schlussauktion oder Eröffnung des Folgetags. |
| `trend_cpo_lang.py` | CPO-Test mit langer Kurshistorie ab 2000 und CPO als laufendem Ausstiegssignal. |
| `trend_protokoll.py` | Protokollmodus (Paper Trading) für Robust 1 und Robust 2. |

## Einrichten

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
```

Die Kurse kommen kostenlos von Yahoo Finance (`yfinance`). Kursdaten liegen bewusst nicht im Repository.
Jeder lädt sie beim ersten Lauf selbst.

## Aufruf

```bash
# Mean Reversion (Kurse werden automatisch geladen)
cd hgv_mgm_modell
python pair_modell.py                 # Rückrechnung: Training, Test, Korb und einzelne Paare
python pair_modell.py --protokoll     # Paper Trading: Tagesstand, Orders für den nächsten Handelstag

# Trendfolge: zuerst die Xetra-Kurse über den Screener laden
cd pair_screener
python pair_screener.py               # legt pair_screener/cache/prices_aktien.pkl an
cd ../trend_modell
python trend_modell.py --nur-kurse    # Rückrechnung ohne Medienecho
python trend_protokoll.py --protokoll # Paper Trading Robust 1 und Robust 2
```

Jedes Skript beschreibt am Anfang der Datei, was es rechnet, welche Schalter es hat und welche Dateien es schreibt.
Ergebnisse sind CSV-Dateien mit `;` und Dezimalkomma (für Excel).

## Protokollmodus

Beim ersten Aufruf von `--protokoll` werden der Starttag und alle Einstellungen in `protokoll/start.json`
festgehalten. Ändert man danach eine Einstellung, bricht das Protokoll mit einer Meldung ab, damit das
Ergebnis nicht unbemerkt verfälscht wird. `protokoll.csv` wird nur fortgeschrieben, bestehende Zeilen bleiben
unverändert. In `ausfuehrungen.csv` kann man die real erzielten Kurse eintragen. Das Skript misst daraus die
tatsächliche Slippage.

## Bekannte Grenzen

- Die Kurse stammen aus Yahoo Finance: rückwirkend um Dividenden bereinigt und gelegentlich nachträglich korrigiert.
- Das Universum besteht nur aus heute noch gelisteten Titeln (Survivorship Bias). Das Trendmodell kann das mit `--mit-pleiten` abschätzen.
- Gerechnet wird mit Schlusskursen statt mit Geld- und Briefkursen. Slippage ist pauschal angesetzt.

## Lizenz

MIT, siehe [LICENSE](LICENSE).
