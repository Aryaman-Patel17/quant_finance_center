"""
Shared helpers: data fetching, input parsing, console formatting.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf


def hr(title=""):
    """Print a horizontal rule with an optional title, for readable console output."""
    width = 70
    if title:
        print(f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}")
    else:
        print("-" * width)


def prompt(text, default=None, cast=str):
    """Prompt user for input with an optional default and type cast."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{text}{suffix}: ").strip()
    if raw == "" and default is not None:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"Could not parse '{raw}' as {cast.__name__}, using default: {default}")
        return default


def fetch_price_history(tickers, period="2y", interval="1d"):
    """
    Download adjusted close price history for one or more tickers via yfinance.
    Returns a DataFrame (columns = tickers) even for a single ticker.
    """
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    print(f"Fetching data for {', '.join(tickers)} ({period}, {interval})...")
    data = yf.download(tickers, period=period, interval=interval,
                        auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError("No data returned. Check ticker symbols and try again.")

    # yfinance returns a MultiIndex column structure for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty:
        raise ValueError("Downloaded data was empty after cleaning. Try a different period.")
    return prices


def annualization_factor(interval="1d"):
    """Rough trading-periods-per-year factor for the given interval."""
    mapping = {"1d": 252, "1wk": 52, "1mo": 12}
    return mapping.get(interval, 252)


def max_drawdown(cumulative_returns):
    """Compute max drawdown from a series of cumulative return multiples (e.g. 1.0, 1.05, 0.98...)."""
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    return drawdown.min()


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Annualized Sharpe ratio from a series of periodic returns."""
    excess = returns - (risk_free_rate / periods_per_year)
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def pause():
    input("\nPress Enter to return to the menu...")
