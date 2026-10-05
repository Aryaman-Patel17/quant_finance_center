"""
EMA Crossover Backtest
-----------------------
Classic dual-EMA trend-following strategy: go long when the fast EMA crosses
above the slow EMA, exit (go flat) when it crosses back below. Backtested
against real historical prices pulled live from yfinance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.utils import (hr, prompt, fetch_price_history, annualization_factor,
                            max_drawdown, sharpe_ratio, pause)


def run():
    hr("EMA CROSSOVER BACKTEST")

    ticker = prompt("Ticker symbol", default="AAPL")
    period = prompt("History period (e.g. 6mo, 1y, 2y, 5y, max)", default="2y")
    fast_span = prompt("Fast EMA span", default=12, cast=int)
    slow_span = prompt("Slow EMA span", default=26, cast=int)
    starting_capital = prompt("Starting capital ($)", default=10000, cast=float)

    if fast_span >= slow_span:
        print("Fast EMA span must be shorter than slow EMA span. Aborting.")
        pause()
        return

    try:
        prices = fetch_price_history(ticker, period=period)
    except ValueError as e:
        print(f"Error: {e}")
        pause()
        return

    df = prices.copy()
    df.columns = ["price"]
    df["ema_fast"] = df["price"].ewm(span=fast_span, adjust=False).mean()
    df["ema_slow"] = df["price"].ewm(span=slow_span, adjust=False).mean()

    # Signal: 1 = long, 0 = flat. Shifted by 1 day to avoid lookahead bias
    # (you trade on the day AFTER the crossover is confirmed).
    df["signal"] = np.where(df["ema_fast"] > df["ema_slow"], 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)

    df["daily_return"] = df["price"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["daily_return"]

    df["buy_hold_equity"] = starting_capital * (1 + df["daily_return"]).cumprod()
    df["strategy_equity"] = starting_capital * (1 + df["strategy_return"]).cumprod()

    # Identify actual crossover event points for plotting/reporting
    df["cross_up"] = (df["signal"] == 1) & (df["signal"].shift(1) == 0)
    df["cross_down"] = (df["signal"] == 0) & (df["signal"].shift(1) == 1)
    n_trades = int(df["cross_up"].sum())

    periods_per_year = annualization_factor()
    years = len(df) / periods_per_year

    strat_total_return = df["strategy_equity"].iloc[-1] / starting_capital - 1
    bh_total_return = df["buy_hold_equity"].iloc[-1] / starting_capital - 1
    strat_cagr = (1 + strat_total_return) ** (1 / years) - 1 if years > 0 else np.nan
    bh_cagr = (1 + bh_total_return) ** (1 / years) - 1 if years > 0 else np.nan

    strat_sharpe = sharpe_ratio(df["strategy_return"])
    bh_sharpe = sharpe_ratio(df["daily_return"])

    strat_mdd = max_drawdown(df["strategy_equity"] / starting_capital)
    bh_mdd = max_drawdown(df["buy_hold_equity"] / starting_capital)

    win_days = df.loc[df["position"] == 1, "strategy_return"]
    win_rate = (win_days > 0).mean() if len(win_days) else np.nan

    hr("RESULTS")
    print(f"{'Metric':<28}{'Strategy':>18}{'Buy & Hold':>18}")
    print(f"{'Total return':<28}{strat_total_return:>17.2%}{bh_total_return:>18.2%}")
    print(f"{'CAGR':<28}{strat_cagr:>17.2%}{bh_cagr:>18.2%}")
    print(f"{'Sharpe ratio':<28}{strat_sharpe:>18.2f}{bh_sharpe:>18.2f}")
    print(f"{'Max drawdown':<28}{strat_mdd:>17.2%}{bh_mdd:>18.2%}")
    print(f"{'Ending equity ($)':<28}{df['strategy_equity'].iloc[-1]:>18,.2f}{df['buy_hold_equity'].iloc[-1]:>18,.2f}")
    print(f"\nNumber of long entries: {n_trades}")
    print(f"Win rate on days in market: {win_rate:.2%}" if not np.isnan(win_rate) else "No trades taken.")

    _plot(df, ticker, fast_span, slow_span)
    pause()


def _plot(df, ticker, fast_span, slow_span):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["price"], label=f"{ticker} price", color="black", linewidth=1)
    ax1.plot(df.index, df["ema_fast"], label=f"EMA {fast_span}", color="tab:blue", linewidth=1.2)
    ax1.plot(df.index, df["ema_slow"], label=f"EMA {slow_span}", color="tab:orange", linewidth=1.2)
    ax1.scatter(df.index[df["cross_up"]], df["price"][df["cross_up"]],
                marker="^", color="green", s=80, label="Buy signal", zorder=5)
    ax1.scatter(df.index[df["cross_down"]], df["price"][df["cross_down"]],
                marker="v", color="red", s=80, label="Sell signal", zorder=5)
    ax1.set_title(f"{ticker} — EMA({fast_span}/{slow_span}) Crossover Backtest")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["strategy_equity"], label="Strategy equity", color="tab:green")
    ax2.plot(df.index, df["buy_hold_equity"], label="Buy & hold equity", color="gray", linestyle="--")
    ax2.set_ylabel("Equity ($)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "outputs/ema_crossover_result.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nChart saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    run()
