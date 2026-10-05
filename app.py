"""Browser-based Streamlit interface for the quant finance tools."""
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import options_greeks, portfolio_optimizer
from modules.utils import (
    annualization_factor,
    fetch_price_history,
    max_drawdown,
    sharpe_ratio,
)


st.set_page_config(
    page_title="Quant Finance Center",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers, period):
    return fetch_price_history(list(tickers), period=period)


def show_data_error(error):
    st.error(str(error))
    st.info("Check the ticker symbols and lookback period, then try again.")


def render_ema():
    st.subheader("EMA Crossover Backtest")
    st.caption("Compare a dual-EMA strategy with buy-and-hold using historical prices.")

    with st.form("ema_form"):
        ticker = st.text_input("Ticker symbol", "AAPL").strip().upper()
        period = st.selectbox("History period", ["6mo", "1y", "2y", "5y", "max"], index=2)
        left, right = st.columns(2)
        fast_span = left.number_input("Fast EMA span", min_value=2, max_value=250, value=12)
        slow_span = right.number_input("Slow EMA span", min_value=3, max_value=500, value=26)
        starting_capital = st.number_input(
            "Starting capital ($)", min_value=1.0, value=10000.0, step=1000.0
        )
        submitted = st.form_submit_button("Run backtest", type="primary")

    if not submitted:
        return
    if not ticker:
        st.error("Enter a ticker symbol.")
        return
    if fast_span >= slow_span:
        st.error("Fast EMA span must be shorter than slow EMA span.")
        return

    try:
        prices = load_prices((ticker,), period)
    except ValueError as error:
        show_data_error(error)
        return

    df = prices.copy()
    df.columns = ["price"]
    df["ema_fast"] = df["price"].ewm(span=fast_span, adjust=False).mean()
    df["ema_slow"] = df["price"].ewm(span=slow_span, adjust=False).mean()
    df["signal"] = np.where(df["ema_fast"] > df["ema_slow"], 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)
    df["daily_return"] = df["price"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["daily_return"]
    df["buy_hold_equity"] = starting_capital * (1 + df["daily_return"]).cumprod()
    df["strategy_equity"] = starting_capital * (1 + df["strategy_return"]).cumprod()
    df["cross_up"] = (df["signal"] == 1) & (df["signal"].shift(1) == 0)
    df["cross_down"] = (df["signal"] == 0) & (df["signal"].shift(1) == 1)

    periods_per_year = annualization_factor()
    years = len(df) / periods_per_year
    strategy_return = df["strategy_equity"].iloc[-1] / starting_capital - 1
    buy_hold_return = df["buy_hold_equity"].iloc[-1] / starting_capital - 1
    strategy_cagr = (1 + strategy_return) ** (1 / years) - 1 if years else np.nan
    buy_hold_cagr = (1 + buy_hold_return) ** (1 / years) - 1 if years else np.nan
    in_market_returns = df.loc[df["position"] == 1, "strategy_return"]
    win_rate = (in_market_returns > 0).mean() if len(in_market_returns) else np.nan

    st.markdown("#### Results")
    metrics = st.columns(4)
    metrics[0].metric("Strategy return", f"{strategy_return:.2%}")
    metrics[1].metric("Buy & hold return", f"{buy_hold_return:.2%}")
    metrics[2].metric("Strategy Sharpe", f"{sharpe_ratio(df['strategy_return']):.2f}")
    metrics[3].metric("Strategy max drawdown", f"{max_drawdown(df['strategy_equity'] / starting_capital):.2%}")
    st.dataframe(
        pd.DataFrame(
            {
                "Strategy": [
                    strategy_return,
                    strategy_cagr,
                    sharpe_ratio(df["strategy_return"]),
                    max_drawdown(df["strategy_equity"] / starting_capital),
                    df["strategy_equity"].iloc[-1],
                ],
                "Buy & Hold": [
                    buy_hold_return,
                    buy_hold_cagr,
                    sharpe_ratio(df["daily_return"]),
                    max_drawdown(df["buy_hold_equity"] / starting_capital),
                    df["buy_hold_equity"].iloc[-1],
                ],
            },
            index=["Total return", "CAGR", "Sharpe ratio", "Max drawdown", "Ending equity ($)"],
        ).style.format(
            "{:.2%}",
            subset=pd.IndexSlice[["Total return", "CAGR", "Max drawdown"], :],
        ).format(
            "{:.2f}",
            subset=pd.IndexSlice[["Sharpe ratio", "Ending equity ($)"], :],
        ),
        use_container_width=True,
    )
    st.caption(
        f"Long entries: {int(df['cross_up'].sum())} · "
        f"Win rate while invested: {win_rate:.2%}" if not np.isnan(win_rate)
        else f"Long entries: {int(df['cross_up'].sum())} · No days invested."
    )

    fig, (price_ax, equity_ax) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    price_ax.plot(df.index, df["price"], label=f"{ticker} price", color="black")
    price_ax.plot(df.index, df["ema_fast"], label=f"EMA {fast_span}", color="tab:blue")
    price_ax.plot(df.index, df["ema_slow"], label=f"EMA {slow_span}", color="tab:orange")
    price_ax.scatter(df.index[df["cross_up"]], df.loc[df["cross_up"], "price"],
                     marker="^", color="green", label="Buy", zorder=5)
    price_ax.scatter(df.index[df["cross_down"]], df.loc[df["cross_down"], "price"],
                     marker="v", color="red", label="Sell", zorder=5)
    price_ax.set_ylabel("Price ($)")
    price_ax.legend(loc="upper left", fontsize=8)
    price_ax.grid(alpha=0.3)
    equity_ax.plot(df.index, df["strategy_equity"], label="Strategy", color="tab:green")
    equity_ax.plot(df.index, df["buy_hold_equity"], label="Buy & hold", color="gray", linestyle="--")
    equity_ax.set_ylabel("Equity ($)")
    equity_ax.legend(loc="upper left", fontsize=8)
    equity_ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_portfolio():
    st.subheader("Portfolio Optimizer")
    st.caption("Find long-only maximum-Sharpe and minimum-volatility allocations.")

    with st.form("portfolio_form"):
        tickers_raw = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN")
        period = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=2)
        risk_free_rate = st.number_input(
            "Annual risk-free rate", min_value=0.0, max_value=1.0, value=0.04, step=0.01
        )
        submitted = st.form_submit_button("Optimize portfolio", type="primary")

    if not submitted:
        return
    tickers = list(dict.fromkeys(
        ticker.strip().upper() for ticker in tickers_raw.split(",") if ticker.strip()
    ))
    if len(tickers) < 2:
        st.error("Enter at least two distinct ticker symbols.")
        return

    try:
        prices = load_prices(tuple(tickers), period)
    except ValueError as error:
        show_data_error(error)
        return
    if prices.shape[1] < 2:
        st.error("Price data was available for fewer than two tickers.")
        return

    tickers = list(prices.columns)
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    periods_per_year = annualization_factor()
    n_assets = len(tickers)
    max_sharpe_weights = portfolio_optimizer._optimize(
        portfolio_optimizer._neg_sharpe,
        n_assets,
        args=(mean_returns, cov_matrix, periods_per_year, risk_free_rate),
    )
    min_vol_weights = portfolio_optimizer._optimize(
        portfolio_optimizer._volatility,
        n_assets,
        args=(mean_returns, cov_matrix, periods_per_year, risk_free_rate),
    )
    equal_weights = np.repeat(1 / n_assets, n_assets)
    strategies = {
        "Max Sharpe": max_sharpe_weights,
        "Min Volatility": min_vol_weights,
        "Equal Weight": equal_weights,
    }

    weights = pd.DataFrame(strategies, index=tickers)
    st.markdown("#### Allocation weights")
    st.dataframe(weights.style.format("{:.1%}"), use_container_width=True)
    stats = {
        name: portfolio_optimizer._portfolio_stats(
            allocation, mean_returns, cov_matrix, periods_per_year, risk_free_rate
        )
        for name, allocation in strategies.items()
    }
    st.markdown("#### Annualized portfolio statistics")
    st.dataframe(
        pd.DataFrame(
            {name: {"Return": values[0], "Volatility": values[1], "Sharpe": values[2]}
             for name, values in stats.items()}
        ).style.format({"Max Sharpe": "{:.2%}", "Min Volatility": "{:.2%}",
                        "Equal Weight": "{:.2%}"}, subset=pd.IndexSlice[["Return", "Volatility"], :])
        .format("{:.2f}", subset=pd.IndexSlice[["Sharpe"], :]),
        use_container_width=True,
    )

    rng = np.random.default_rng(42)
    results = np.empty((3, 2000))
    for index in range(results.shape[1]):
        allocation = rng.random(n_assets)
        allocation /= allocation.sum()
        results[:, index] = portfolio_optimizer._portfolio_stats(
            allocation, mean_returns, cov_matrix, periods_per_year, risk_free_rate
        )
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results[1], results[0], c=results[2], cmap="viridis", s=10, alpha=0.6)
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    for name, color in [("Max Sharpe", "red"), ("Min Volatility", "blue")]:
        ret, volatility, _ = stats[name]
        ax.scatter(volatility, ret, marker="*", color=color, s=300,
                   label=name, edgecolors="black")
    ax.set_title(f"Efficient Frontier — {', '.join(tickers)}")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_options():
    st.subheader("Options Greeks")
    st.caption("Black–Scholes theoretical pricing and sensitivities using live market data.")

    with st.form("options_form"):
        ticker = st.text_input("Underlying ticker", "AAPL").strip().upper()
        option_type = st.selectbox("Option type", ["call", "put"])
        left, right = st.columns(2)
        strike = left.number_input("Strike price ($)", min_value=0.01, value=100.0, step=1.0)
        days_to_expiry = right.number_input("Days to expiration", min_value=1, max_value=3650, value=30)
        use_historical_vol = st.checkbox("Estimate volatility from one year of historical prices", True)
        manual_volatility = st.number_input(
            "Annualized volatility (%)", min_value=0.1, max_value=500.0, value=30.0, step=1.0,
            disabled=use_historical_vol,
        )
        risk_free_rate = st.number_input(
            "Annual risk-free rate (%)", min_value=-20.0, max_value=100.0, value=4.0, step=0.25
        )
        submitted = st.form_submit_button("Calculate option", type="primary")

    if not submitted:
        return
    if not ticker:
        st.error("Enter an underlying ticker.")
        return
    try:
        spot_prices = load_prices((ticker,), "5d")
        spot = float(spot_prices.iloc[-1, 0])
        if use_historical_vol:
            historical_prices = load_prices((ticker,), "1y")
            log_returns = np.log(
                historical_prices.iloc[:, 0] / historical_prices.iloc[:, 0].shift(1)
            ).dropna()
            volatility = float(log_returns.std() * np.sqrt(252))
        else:
            volatility = manual_volatility / 100
    except ValueError as error:
        show_data_error(error)
        return
    if volatility <= 0 or not np.isfinite(volatility):
        st.error("Volatility must be a positive finite value.")
        return

    years_to_expiry = days_to_expiry / 365
    rate = risk_free_rate / 100
    price = options_greeks.bs_price(
        spot, strike, years_to_expiry, rate, volatility, option_type
    )
    greeks = options_greeks.bs_greeks(
        spot, strike, years_to_expiry, rate, volatility, option_type
    )
    st.caption(
        f"{ticker} spot: ${spot:,.2f} · Strike: ${strike:,.2f} · "
        f"Volatility: {volatility:.2%} · {days_to_expiry} days to expiry"
    )
    cards = st.columns(3)
    cards[0].metric("Theoretical option price", f"${price:,.2f}")
    cards[1].metric("Delta", f"{greeks['delta']:.4f}")
    cards[2].metric("Gamma", f"{greeks['gamma']:.4f}")
    greek_frame = pd.DataFrame(
        {"Value": [greeks["vega"], greeks["theta"], greeks["rho"]]},
        index=["Vega (per 1 vol point)", "Theta (per day)", "Rho (per 1 rate point)"],
    )
    st.dataframe(greek_frame.style.format("{:.4f}"), use_container_width=True)

    spot_range = np.linspace(spot * 0.6, spot * 1.4, 200)
    greek_values = [
        [options_greeks.bs_greeks(value, strike, years_to_expiry, rate, volatility, option_type)[key]
         for value in spot_range]
        for key in ("delta", "gamma", "vega", "theta")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, values, label in zip(
        axes.flat, greek_values, ("Delta", "Gamma", "Vega", "Theta (per day)")
    ):
        ax.plot(spot_range, values)
        ax.axvline(spot, color="gray", linestyle="--", label="Current spot")
        ax.axvline(strike, color="black", linestyle=":", label="Strike")
        ax.set_title(label)
        ax.set_xlabel("Underlying price ($)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{option_type.title()} Greeks vs. underlying price")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_monte_carlo():
    st.subheader("Monte Carlo Simulation")
    st.caption("Simulate possible future prices with Geometric Brownian Motion.")

    with st.form("monte_carlo_form"):
        ticker = st.text_input("Ticker symbol", "AAPL").strip().upper()
        lookback = st.selectbox("Historical lookback", ["1y", "2y", "3y", "5y"], index=1)
        left, right = st.columns(2)
        horizon_days = left.number_input(
            "Horizon (trading days)", min_value=1, max_value=2520, value=252
        )
        simulations = right.number_input(
            "Number of paths", min_value=100, max_value=10000, value=2000, step=100
        )
        confidence = st.slider("VaR confidence level", min_value=0.80, max_value=0.99, value=0.95)
        submitted = st.form_submit_button("Run simulation", type="primary")

    if not submitted:
        return
    if not ticker:
        st.error("Enter a ticker symbol.")
        return
    try:
        prices = load_prices((ticker,), lookback)
    except ValueError as error:
        show_data_error(error)
        return

    series = prices.iloc[:, 0]
    log_returns = np.log(series / series.shift(1)).dropna()
    daily_drift = log_returns.mean()
    daily_volatility = log_returns.std()
    starting_price = float(series.iloc[-1])
    rng = np.random.default_rng(7)
    daily_shocks = rng.normal(
        loc=daily_drift - 0.5 * daily_volatility**2,
        scale=daily_volatility,
        size=(int(horizon_days), int(simulations)),
    )
    price_paths = starting_price * np.exp(
        np.vstack([np.zeros(int(simulations)), np.cumsum(daily_shocks, axis=0)])
    )
    ending_prices = price_paths[-1]
    percentiles = np.percentile(ending_prices, [5, 25, 50, 75, 95])
    returns_distribution = ending_prices / starting_price - 1
    var_cutoff = np.percentile(returns_distribution, (1 - confidence) * 100)
    conditional_var = returns_distribution[returns_distribution <= var_cutoff].mean()

    st.caption(
        f"Last close: ${starting_price:,.2f} · Estimated annualized drift: "
        f"{daily_drift * 252:.2%} · Annualized volatility: {daily_volatility * np.sqrt(252):.2%}"
    )
    results = st.columns(4)
    results[0].metric("Median ending price", f"${percentiles[2]:,.2f}")
    results[1].metric(f"{confidence:.0%} VaR", f"{var_cutoff:.2%}")
    results[2].metric("Conditional VaR", f"{conditional_var:.2%}")
    results[3].metric("Median return", f"{percentiles[2] / starting_price - 1:.2%}")
    st.dataframe(
        pd.DataFrame(
            {"Ending price ($)": percentiles},
            index=["5th percentile", "25th percentile", "Median", "75th percentile", "95th percentile"],
        ).style.format("${:,.2f}"),
        use_container_width=True,
    )

    fig, (paths_ax, distribution_ax) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    n_display = min(300, price_paths.shape[1])
    paths_ax.plot(price_paths[:, :n_display], linewidth=0.5, alpha=0.2)
    path_percentiles = np.percentile(price_paths, [5, 50, 95], axis=1)
    paths_ax.plot(path_percentiles[1], color="black", linewidth=2, label="Median path")
    paths_ax.plot(path_percentiles[0], color="red", linestyle="--", label="5th percentile")
    paths_ax.plot(path_percentiles[2], color="green", linestyle="--", label="95th percentile")
    paths_ax.axhline(starting_price, color="gray", linestyle=":", label="Starting price")
    paths_ax.set_title(f"{ticker} simulated paths ({horizon_days} days)")
    paths_ax.set_xlabel("Trading days ahead")
    paths_ax.set_ylabel("Price ($)")
    paths_ax.legend(fontsize=8)
    paths_ax.grid(alpha=0.3)
    distribution_ax.hist(ending_prices, bins=50, orientation="horizontal", alpha=0.7)
    distribution_ax.axhline(starting_price, color="gray", linestyle=":", label="Starting price")
    distribution_ax.axhline(np.median(ending_prices), color="black", label="Median")
    distribution_ax.set_title("Ending price distribution")
    distribution_ax.set_xlabel("Frequency")
    distribution_ax.legend(fontsize=8)
    distribution_ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.title("Quant Finance Center")
st.markdown(
    "Explore backtests, portfolio allocations, options Greeks, and simulated market scenarios."
)
st.sidebar.title("Finance tools")
tool = st.sidebar.radio(
    "Choose a tool",
    [
        "EMA Crossover Backtest",
        "Portfolio Optimizer",
        "Options Greeks",
        "Monte Carlo Simulation",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Market history is fetched from Yahoo Finance. For education only—not investment advice.")

{
    "EMA Crossover Backtest": render_ema,
    "Portfolio Optimizer": render_portfolio,
    "Options Greeks": render_options,
    "Monte Carlo Simulation": render_monte_carlo,
}[tool]()
