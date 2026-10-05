"""
Monte Carlo Simulation
------------------------
Estimates drift and volatility from real historical returns (via yfinance),
then simulates thousands of future price paths using Geometric Brownian
Motion. Reports percentile outcomes and Value-at-Risk.
"""
import numpy as np
import matplotlib.pyplot as plt

from modules.utils import hr, prompt, fetch_price_history, pause


def run():
    hr("MONTE CARLO PRICE SIMULATION")

    ticker = prompt("Ticker symbol", default="AAPL")
    lookback = prompt("Lookback period for drift/volatility estimation (e.g. 1y, 2y)", default="2y")
    horizon_days = prompt("Simulation horizon (trading days ahead)", default=252, cast=int)
    n_sims = prompt("Number of simulated paths", default=2000, cast=int)
    confidence = prompt("VaR confidence level (e.g. 0.95)", default=0.95, cast=float)

    try:
        prices = fetch_price_history(ticker, period=lookback)
    except ValueError as e:
        print(f"Error: {e}")
        pause()
        return

    series = prices.iloc[:, 0]
    log_returns = np.log(series / series.shift(1)).dropna()

    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()
    S0 = float(series.iloc[-1])

    print(f"\nLast close: ${S0:.2f}")
    print(f"Estimated daily drift: {mu_daily:.5f}   Estimated daily volatility: {sigma_daily:.5f}")
    print(f"Annualized: drift={mu_daily*252:.2%}   volatility={sigma_daily*np.sqrt(252):.2%}")

    rng = np.random.default_rng(7)
    # GBM: S_t = S_0 * exp( (mu - 0.5*sigma^2)*t + sigma*W_t )
    daily_shocks = rng.normal(
        loc=(mu_daily - 0.5 * sigma_daily ** 2),
        scale=sigma_daily,
        size=(horizon_days, n_sims),
    )
    log_paths = np.vstack([np.zeros(n_sims), np.cumsum(daily_shocks, axis=0)])
    price_paths = S0 * np.exp(log_paths)

    ending_prices = price_paths[-1, :]
    pct_5, pct_25, pct_50, pct_75, pct_95 = np.percentile(ending_prices, [5, 25, 50, 75, 95])

    returns_dist = ending_prices / S0 - 1
    var_cutoff = np.percentile(returns_dist, (1 - confidence) * 100)
    cvar = returns_dist[returns_dist <= var_cutoff].mean()

    hr("SIMULATION RESULTS")
    print(f"Ticker: {ticker}   Horizon: {horizon_days} trading days   Paths simulated: {n_sims}")
    print(f"\n{'Percentile':<20}{'Ending Price':>16}")
    for label, val in [("5th", pct_5), ("25th", pct_25), ("50th (median)", pct_50),
                        ("75th", pct_75), ("95th", pct_95)]:
        print(f"{label:<20}${val:>15.2f}")

    print(f"\nExpected return (median path): {(pct_50 / S0 - 1):.2%}")
    print(f"{confidence:.0%} Value-at-Risk (over {horizon_days}d): {var_cutoff:.2%} "
          f"(i.e. {100*(1-confidence):.0f}% chance of losing at least this much)")
    print(f"Conditional VaR / Expected Shortfall: {cvar:.2%}")

    _plot(price_paths, ending_prices, S0, ticker, horizon_days)
    pause()


def _plot(price_paths, ending_prices, S0, ticker, horizon_days):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})

    n_display = min(300, price_paths.shape[1])
    ax1.plot(price_paths[:, :n_display], linewidth=0.5, alpha=0.25, color="tab:blue")
    percentiles = np.percentile(price_paths, [5, 50, 95], axis=1)
    ax1.plot(percentiles[1], color="black", linewidth=2, label="Median path")
    ax1.plot(percentiles[0], color="red", linewidth=1.5, linestyle="--", label="5th percentile")
    ax1.plot(percentiles[2], color="green", linewidth=1.5, linestyle="--", label="95th percentile")
    ax1.axhline(S0, color="gray", linestyle=":", label="Starting price")
    ax1.set_title(f"{ticker} — {price_paths.shape[1]} Simulated Paths ({horizon_days}d)")
    ax1.set_xlabel("Trading days ahead")
    ax1.set_ylabel("Price ($)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(ending_prices, bins=50, orientation="horizontal", color="tab:blue", alpha=0.7)
    ax2.axhline(S0, color="gray", linestyle=":", label="Starting price")
    ax2.axhline(np.median(ending_prices), color="black", label="Median ending price")
    ax2.set_title("Distribution of Ending Prices")
    ax2.set_xlabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "outputs/monte_carlo_result.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nChart saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    run()
