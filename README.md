# Quant Finance Testing Center

A menu-driven Python toolkit with four quant finance tools, all running on **live market data** via `yfinance`:

1. **EMA Crossover Backtest** — dual exponential moving average trend-following strategy, backtested against real price history, benchmarked against buy-and-hold (CAGR, Sharpe, max drawdown, win rate, and summary metrics).
2. **Portfolio Optimizer** — Markowitz mean-variance optimization across any basket of tickers; finds the max-Sharpe and min-volatility portfolios and plots the efficient frontier.
3. **Options Greeks Dashboard** — Black-Scholes pricing and Greeks (Delta, Gamma, Vega, Theta, Rho) using the live spot price and historical (or manual) volatility.
4. **Monte Carlo Simulation** — Geometric Brownian Motion price path simulation calibrated on real historical drift/volatility, with percentile outcomes and Value-at-Risk.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running

### Terminal app

```bash
python main.py
```

This launches an interactive menu. Pick a tool, follow the prompts (ticker symbols, lookback periods, parameters), and each tool will:

- Pull live data from Yahoo Finance via `yfinance`
- Print a results summary to the console
- Save a chart PNG to `outputs/`
- Display the chart in a matplotlib window

You can also run any tool directly, bypassing the menu, e.g.:

```bash
python -m modules.ema_crossover
python -m modules.portfolio_optimizer
python -m modules.options_greeks
python -m modules.monte_carlo
```

## Project Structure

```
quant_finance_center/
├── main.py                     # CLI menu / entry point
├── requirements.txt
├── modules/
│   ├── utils.py                 # shared data-fetching & helper functions
│   ├── ema_crossover.py
│   ├── portfolio_optimizer.py
│   ├── options_greeks.py
│   └── monte_carlo.py
└── outputs/                     # generated charts land here
```

## Notes

- All tools require an internet connection (they call Yahoo Finance live).
- Ticker symbols must be valid Yahoo Finance tickers (e.g. `AAPL`, `MSFT`, `^GSPC`).
- The Portfolio Optimizer uses a **long-only, no-leverage** constraint (weights between 0 and 1, summing to 1). Edit the `bounds` in `portfolio_optimizer.py` if you want to allow short positions.
- Historical volatility, drift, and covariance are all computed from real daily log/percentage returns — no data is simulated except the Monte Carlo *future* price paths, which are, by design, simulated.
- This is an educational/research tool, not investment advice. Backtested performance does not guarantee future results.

## Extending it

Each tool lives in its own module with a `run()` function, so adding a fifth tool is just:
1. Create `modules/your_tool.py` with a `run()` function following the same pattern.
2. Import it in `main.py` and add it to the `MENU` dict.
