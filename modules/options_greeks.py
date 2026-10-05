"""
Options Greeks Dashboard (Black-Scholes)
------------------------------------------
Pulls the live spot price for a ticker via yfinance, computes historical
volatility from recent returns (or accepts a manual override), then computes
and plots Black-Scholes Greeks (Delta, Gamma, Vega, Theta, Rho) across a
range of spot prices.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.utils import hr, prompt, fetch_price_history, pause


def _d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Returns a dict of Delta, Gamma, Vega, Theta (per day), Rho (per 1% rate move)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100  # per 1 vol point (1%)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def _historical_volatility(ticker, lookback="1y"):
    prices = fetch_price_history(ticker, period=lookback)
    log_returns = np.log(prices.iloc[:, 0] / prices.iloc[:, 0].shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def run():
    hr("OPTIONS GREEKS DASHBOARD (BLACK-SCHOLES)")

    ticker = prompt("Underlying ticker", default="AAPL")

    try:
        prices = fetch_price_history(ticker, period="5d")
        spot = float(prices.iloc[-1, 0])
        print(f"Live spot price for {ticker}: ${spot:.2f}")
    except ValueError as e:
        print(f"Error fetching spot price: {e}")
        pause()
        return

    use_hist_vol = prompt("Use historical volatility from price data? (y/n)", default="y")
    if use_hist_vol.lower().startswith("y"):
        sigma = _historical_volatility(ticker)
        print(f"Historical (1y) annualized volatility: {sigma:.2%}")
    else:
        sigma = prompt("Manual volatility (annualized, decimal, e.g. 0.30)", default=0.30, cast=float)

    strike = prompt("Strike price", default=round(spot), cast=float)
    days_to_expiry = prompt("Days to expiration", default=30, cast=int)
    r = prompt("Risk-free rate (annual, decimal)", default=0.04, cast=float)
    option_type = prompt("Option type (call/put)", default="call")
    option_type = "put" if option_type.lower().startswith("p") else "call"

    T = days_to_expiry / 365
    price = bs_price(spot, strike, T, r, sigma, option_type)
    greeks = bs_greeks(spot, strike, T, r, sigma, option_type)

    hr("RESULTS")
    print(f"Underlying: {ticker}   Spot: ${spot:.2f}   Strike: ${strike:.2f}   "
          f"{days_to_expiry}d to expiry   Type: {option_type.upper()}")
    print(f"Volatility used: {sigma:.2%}   Risk-free rate: {r:.2%}\n")
    print(f"{'Theoretical price':<20}${price:>10.2f}")
    print(f"{'Delta':<20}{greeks['delta']:>11.4f}   (change in option price per $1 move in underlying)")
    print(f"{'Gamma':<20}{greeks['gamma']:>11.4f}   (change in delta per $1 move in underlying)")
    print(f"{'Vega':<20}{greeks['vega']:>11.4f}   (change in price per 1pt change in IV)")
    print(f"{'Theta':<20}{greeks['theta']:>11.4f}   (change in price per day, time decay)")
    print(f"{'Rho':<20}{greeks['rho']:>11.4f}   (change in price per 1pt change in rates)")

    _plot(spot, strike, T, r, sigma, option_type)
    pause()


def _plot(spot, strike, T, r, sigma, option_type):
    spot_range = np.linspace(spot * 0.6, spot * 1.4, 200)
    delta_vals, gamma_vals, vega_vals, theta_vals = [], [], [], []

    for S in spot_range:
        g = bs_greeks(S, strike, T, r, sigma, option_type)
        delta_vals.append(g["delta"])
        gamma_vals.append(g["gamma"])
        vega_vals.append(g["vega"])
        theta_vals.append(g["theta"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_specs = [
        (axes[0, 0], delta_vals, "Delta", "tab:blue"),
        (axes[0, 1], gamma_vals, "Gamma", "tab:orange"),
        (axes[1, 0], vega_vals, "Vega", "tab:green"),
        (axes[1, 1], theta_vals, "Theta (per day)", "tab:red"),
    ]
    for ax, vals, name, color in plot_specs:
        ax.plot(spot_range, vals, color=color)
        ax.axvline(spot, color="gray", linestyle="--", linewidth=1, label="Current spot")
        ax.axvline(strike, color="black", linestyle=":", linewidth=1, label="Strike")
        ax.set_title(name)
        ax.set_xlabel("Underlying Price ($)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"Option Greeks vs Underlying Price ({option_type.upper()}, "
                 f"K={strike}, T={T*365:.0f}d, σ={sigma:.1%})")
    plt.tight_layout()

    out_path = "outputs/options_greeks_result.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nChart saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    run()
