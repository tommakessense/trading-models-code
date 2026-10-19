# Trading Models

Two rule-based trading models in Python, inspired by Ernest P. Chan,
*Quantitative Trading*. Both models have a backtest and a protocol mode for
paper trading.

> **No investment advice.** The code serves research and learning. Backtests are no
> guarantee of future results. Anyone who trades with it does so at their own risk.

## Models

| Folder | Model | Market |
|---|---|---|
| `hgv_mgm_model/` | **Mean reversion with stock pairs.** Trades the spread of 15 pairs according to its z-score. Account with whole shares, fees of Interactive Brokers, slippage, borrowing fee and three leverage profiles (without leverage, margin account, quarter Kelly with individual limit). | US stocks |
| `trend_model/` | **Trend following.** Two robust basic strategies:<br>Robust 1: new 200-day high, hold for 20 trading days.<br>Robust 2: new 50-day high with double volume, hold for 60 days, trailing stop 15 %.<br>At most 10 positions each, selection by momentum, trading in the Xetra closing auction, fees of Scalable Capital. | Xetra stocks |
| `pair_screener/` | **Screener.** Searches for cointegrated pairs with the Engle-Granger test, where the expected hits that occur by chance are to be detected with the Benjamini-Hochberg procedure. Also loads the Xetra prices that the trend model needs for the backtest. | Xetra, USA |

Additional tests in the trend model:

| File | Content |
|---|---|
| `trend_model.py` | Backtest of many entry and exit variants, optionally with media echo (GDELT). In addition a placebo test and a test of the survivorship bias with bankruptcy cases. |
| `trend_cpo.py` | Conditional Parameter Optimization (machine learning for optimizing the individual parameters of the model before each potential trade): walk-forward with gradient boosting, which chooses the exit for each signal. |
| `cpo_stresstest.py` | CPO against fixed rules in simulated crashes and volatility surges, in order to extend the only limited period of the backtest to more eventualities. |
| `trend_bewertung.py` | Evaluation of the variants: sources of profit, plausibility, probabilistic and deflated Sharpe ratio, White's Reality Check, dependence on the start date. |
| `trend_positionsgroesse.py` | Fixed position size (1/10) against a size depending on signal quality (CPO and simple rules). |
| `trend_ausfuehrung.py` | Execution time in comparison: closing auction or opening of the following day. |
| `trend_cpo_lang.py` | CPO test with a long price history from 2000 and CPO as an ongoing exit signal. |
| `trend_protokoll.py` | Protocol mode (paper trading) for Robust 1 and Robust 2. |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
```

The prices come free of charge from Yahoo Finance (`yfinance`). Price data is deliberately not in the repository.
Everyone loads it themselves on the first run.

## Usage

```bash
# Mean reversion (prices are loaded automatically)
cd hgv_mgm_model
python pair_model.py                  # backtest: training, test, basket and individual pairs
python pair_model.py --protokoll      # paper trading: daily status, orders for the next trading day

# Trend following: first load the Xetra prices via the screener
cd ../pair_screener
python pair_screener.py               # creates pair_screener/cache/prices_aktien.pkl
cd ../trend_model
python trend_model.py --nur-kurse     # backtest without media echo
python trend_protokoll.py --protokoll # paper trading Robust 1 and Robust 2
```

Each script describes at the beginning of the file what it calculates, which switches it has and which files it writes.
Results are CSV files with `;` and decimal comma (for Excel).

## Protocol mode

On the first call of `--protokoll`, the start day and all settings are recorded in `protokoll/start.json`.
If a setting is changed afterwards, the protocol aborts with a message so that the
result is not falsified unnoticed. `protokoll.csv` is only continued, existing rows remain
unchanged. In `ausfuehrungen.csv` you can enter the prices actually achieved. From this the script measures the
actual slippage.

## Known limitations

- The prices come from Yahoo Finance: adjusted retroactively for dividends and occasionally corrected afterwards.
- The universe consists only of securities still listed today (survivorship bias). The trend model can estimate this with `--mit-pleiten`.
- Calculations use closing prices instead of bid and ask prices. Slippage is set as a flat rate.

## License

MIT, see [LICENSE](LICENSE).
