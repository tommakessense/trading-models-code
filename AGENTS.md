# AGENTS.md

Guidance for coding agents working in this repository. Human-facing documentation is in [README.md](README.md);
every script also explains its logic, switches and output files in its module docstring. Read that docstring
before changing a script.

## Project

Research code for two rule-based trading models plus a pair screener, following the methods in
Ernest P. Chan, *Quantitative Trading*. Plain Python scripts, no package, no web service, no database.

- `pair_screener/pair_screener.py` – finds cointegrated / mean-reverting pairs; also builds the Xetra price cache.
- `hgv_mgm_model/pair_model.py` – mean-reversion model for a basket of US stock pairs (backtest + protocol mode).
- `trend_model/trend_model.py` – trend-following backtest on Xetra stocks; shared functions used by all trend scripts.
- `trend_model/trend_cpo.py` – conditional parameter optimization (walk-forward, gradient boosting).
- `trend_model/cpo_stress_test.py`, `trend_evaluation.py`, `trend_position_size.py`, `trend_execution.py`,
  `trend_cpo_long.py` – studies that build on `trend_cpo.py` (run `trend_cpo.py` first).
- `trend_model/trend_protocol.py` – paper-trading protocol for the strategies Robust 1 and Robust 2.

Dependencies between scripts: `trend_cpo.py` imports `trend_model` as `tm`; the other trend scripts import
`trend_model as tm` and `trend_cpo as tc`. `trend_model.py` reads its prices from `pair_screener/cache/prices_stocks.pkl`,
which the screener creates. Renaming a function, constant or column in one file usually affects the others.

## Environment

- Python 3.11 or newer (pandas 3 requires it). Install: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- Scripts are run from their own folder (`cd trend_model && python trend_cpo.py`); paths are built from `__file__`.
- Data comes from the internet at runtime: Yahoo Finance (yfinance), GDELT, nasdaqtrader.com, deutsche-boerse.com.
  No API keys are needed. Without network access the scripts fall back to cached data where it exists.
- Price data, caches and results are not in the repository (see `.gitignore`) and must never be committed.

## Commands

Fast checks (seconds) – use these after every change; each line is run from the repository root:

```bash
python -m py_compile */*.py                                 # syntax of all scripts
python -m pyflakes */*.py                                   # undefined names (pip install pyflakes)
cd pair_screener && python pair_screener.py --demo          # offline end-to-end run on synthetic data
cd pair_screener && python pair_screener.py --demo --two-stage --mean-reversion
```

Real runs (need network or existing caches; minutes to hours – only when asked; also from the repository root):

```bash
cd hgv_mgm_model && python pair_model.py [--grid]           # pair model backtest (loads prices itself)
cd pair_screener && python pair_screener.py                 # Xetra stocks, creates cache/prices_stocks.pkl
cd trend_model && python trend_model.py --prices-only       # trend backtest without media echo (minutes)
cd trend_model && python trend_cpo.py                       # walk-forward CPO (long-running)
```

`--cutoff YYYY-MM-DD` (hidden switch of `pair_model.py --protocol` and `trend_protocol.py --protocol`) runs the
protocol with prices up to a given day; use it in a copy of the folder to test protocol changes.

There is no test suite. To verify a change that must not alter results, run the affected script before and after
the change on the same cached data and compare the numbers in the console output and in the result CSV files.

## Rules that protect the results

- No look-ahead: a signal uses the close of day t and is executed later (next open or next close). Features,
  ranks and model training may only use data up to the signal day; walk-forward models train only on prior years.
- Costs are part of every result (broker fees, slippage, borrowing fee, financing). Do not remove them to make
  results look better, and do not change cost assumptions without saying so.
- Protocol mode is append-only: `protocol/start.json` freezes the settings (`frozen_settings()` in both protocol
  scripts), `protocol.csv` never has existing rows rewritten, `executions.csv` keeps entered real prices.
  Changing a frozen setting deliberately aborts the protocol – do not weaken that check.
- `PAIRS` in `pair_model.py` is fixed on purpose during paper trading; do not "improve" it from new screener runs.
- Keep randomness reproducible (fixed seeds such as `TEST_SEED` / `SEED`).

## Code style

- Settings are module-level constants in UPPER_CASE near the top of each script, each with a short comment.
  Add new options there, not as hard-coded values inside functions.
- Comments, docstrings, output text, file names and column names are in English. Exceptions on purpose:
  the GDELT query terms (`GDELT_CONTEXT`, `search_terms.csv`) search German media; the legal-suffix regex removes
  German company forms; `parse_price_file()` recognizes German headers of exported price files. Do not translate these.
- CSV output uses `;` as separator and a decimal comma (`save_csv()`), for German Excel. Keep this format and
  keep column names stable – other scripts and users read these files.
- Console output: short status lines via `log()`, results as aligned plain-text tables.
- Existing line length is up to about 120 characters; match the surrounding code. No formatter is configured,
  so do not reformat untouched code.
- Prefer vectorized numpy/pandas; the loops over trading days in the account simulations are intentional.

## Permissions

- Allowed without asking: reading files, running the fast checks above, running scripts in `--demo` mode,
  editing code and docs within the requested scope.
- Ask first: installing or upgrading packages, changing `requirements.txt`, long downloads or long runs
  (full screener, GDELT media echo, `trend_cpo.py`), changing default settings, costs or strategy rules,
  deleting files, committing or pushing.
- Never: commit data, caches, results or credentials; rewrite git history.

## Commits

- Short imperative subject in English; one topic per commit.
- State in the message if a change alters results (numbers in backtests or protocols).

## When unsure

If a requested change would alter results, the protocol logic or a frozen setting, describe the effect and ask
before doing it. After two failed attempts at the same fix, stop and report what was tried.
