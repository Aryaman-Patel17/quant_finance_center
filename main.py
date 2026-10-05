"""
Quant Finance Testing Center
------------------------------
A menu-driven hub for four quantitative finance tools, all running on
live market data pulled via yfinance:

  1) EMA Crossover Backtest
  2) Portfolio Optimizer (Mean-Variance)
  3) Options Greeks Dashboard (Black-Scholes)
  4) Monte Carlo Price Simulation

Run with:  python main.py
"""
import os
import sys

from modules import ema_crossover, portfolio_optimizer, options_greeks, monte_carlo
from modules.utils import hr

MENU = {
    "1": ("EMA Crossover Backtest", ema_crossover.run),
    "2": ("Portfolio Optimizer", portfolio_optimizer.run),
    "3": ("Options Greeks Dashboard", options_greeks.run),
    "4": ("Monte Carlo Simulation", monte_carlo.run),
}



def main():
    os.makedirs("outputs", exist_ok=True)

    while True:
        hr("QUANT FINANCE TESTING CENTER")
        for key, (label, _) in MENU.items():
            print(f"  [{key}] {label}")
        print("  [5] Exit")

        choice = input("\nSelect a tool: ").strip()

        if choice == "5" or choice.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            sys.exit(0)

        entry = MENU.get(choice)
        if not entry:
            print("Invalid selection, try again.")
            continue

        _, func = entry
        try:
            func()
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            input("Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
