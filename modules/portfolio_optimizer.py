"""
Portfolio Optimizer (Markowitz Mean-Variance)
----------------------------------------------
Pulls live historical prices for a basket of tickers, computes the efficient
frontier, and finds the max-Sharpe and min-volatility portfolios via
constrained optimization (scipy.optimize.minimize, SLSQP).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from modules.utils import hr, prompt, fetch_price_history, annualization_factor, pause


def _portfolio_stats(weights, mean_returns, cov_matrix, periods_per_year, risk_free_rate):
    ret = np.dot(weights, mean_returns) * periods_per_year
    vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(periods_per_year)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
    return ret, vol, sharpe


def _neg_sharpe(weights, mean_returns, cov_matrix, periods_per_year, risk_free_rate):
    _, _, sharpe = _portfolio_stats(weights, mean_returns, cov_matrix, periods_per_year, risk_free_rate)
    return -sharpe


def _volatility(weights, mean_returns, cov_matrix, periods_per_year, risk_free_rate):
    _, vol, _ = _portfolio_stats(weights, mean_returns, cov_matrix, periods_per_year, risk_free_rate)
    return vol


def _optimize(objective, n_assets, args):
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # long-only, no leverage
    init_guess = np.repeat(1 / n_assets, n_assets)
    result = minimize(objective, init_guess, args=args, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    return result.x


def run():
    hr("PORTFOLIO OPTIMIZER (MEAN-VARIANCE)")

    tickers_raw = prompt("Tickers, comma-separated (e.g. AAPL,MSFT,GOOGL,AMZN)", default="AAPL,MSFT,GOOGL,AMZN")
    period = prompt("History period (e.g. 1y, 2y, 5y)", default="3y")
    risk_free_rate = prompt("Risk-free rate (annual, decimal, e.g. 0.04)", default=0.04, cast=float)

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if len(tickers) < 2:
        print("Need at least 2 tickers for portfolio optimization. Aborting.")
        pause()
        return

    try:
        prices = fetch_price_history(tickers, period=period)
    except ValueError as e:
        print(f"Error: {e}")
        pause()
        return

    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    periods_per_year = annualization_factor()
    n_assets = len(tickers)

    max_sharpe_w = _optimize(_neg_sharpe, n_assets,
                              args=(mean_returns, cov_matrix, periods_per_year, risk_free_rate))
    min_vol_w = _optimize(_volatility, n_assets,
                           args=(mean_returns, cov_matrix, periods_per_year, risk_free_rate))
    equal_w = np.repeat(1 / n_assets, n_assets)

    hr("OPTIMAL WEIGHTS")
    print(f"{'Ticker':<10}{'Max Sharpe':>14}{'Min Volatility':>16}{'Equal Weight':>16}")
    for i, t in enumerate(tickers):
        print(f"{t:<10}{max_sharpe_w[i]:>13.1%} {min_vol_w[i]:>15.1%} {equal_w[i]:>15.1%}")

    hr("PORTFOLIO STATS (ANNUALIZED)")
    for label, w in [("Max Sharpe", max_sharpe_w), ("Min Volatility", min_vol_w), ("Equal Weight", equal_w)]:
        ret, vol, sharpe = _portfolio_stats(w, mean_returns, cov_matrix, periods_per_year, risk_free_rate)
        print(f"{label:<16} return={ret:>7.2%}   volatility={vol:>7.2%}   sharpe={sharpe:>5.2f}")

    _plot_frontier(tickers, mean_returns, cov_matrix, periods_per_year, risk_free_rate,
                    max_sharpe_w, min_vol_w)
    pause()


def _plot_frontier(tickers, mean_returns, cov_matrix, periods_per_year, risk_free_rate,
                    max_sharpe_w, min_vol_w, n_portfolios=4000):
    n_assets = len(tickers)
    results = np.zeros((3, n_portfolios))
    rng = np.random.default_rng(42)

    for i in range(n_portfolios):
        w = rng.random(n_assets)
        w /= w.sum()
        ret, vol, sharpe = _portfolio_stats(w, mean_returns, cov_matrix, periods_per_year, risk_free_rate)
        results[:, i] = [ret, vol, sharpe]

    max_sharpe_stats = _portfolio_stats(max_sharpe_w, mean_returns, cov_matrix, periods_per_year, risk_free_rate)
    min_vol_stats = _portfolio_stats(min_vol_w, mean_returns, cov_matrix, periods_per_year, risk_free_rate)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(results[1], results[0], c=results[2], cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, label="Sharpe ratio")
    plt.scatter(max_sharpe_stats[1], max_sharpe_stats[0], marker="*", color="red", s=400,
                label="Max Sharpe", edgecolors="black")
    plt.scatter(min_vol_stats[1], min_vol_stats[0], marker="*", color="blue", s=400,
                label="Min Volatility", edgecolors="black")
    plt.title(f"Efficient Frontier — {', '.join(tickers)}")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = "outputs/portfolio_optimizer_result.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nChart saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    run()
