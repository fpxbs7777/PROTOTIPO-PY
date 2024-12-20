import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import random
import scipy.optimize as op

# Import our own files and reload
import market_data
import options
importlib.reload(market_data)
importlib.reload(options)

# Set up the inputs for the option pricing
inputs = options.inputs()
inputs.price = 36882  # S
inputs.time = 0  # t
inputs.maturity = 6 / 12  # 6 months
inputs.strike = 1000  # K
inputs.interest_rate = 0.0435  # r
inputs.volatility = 0.556  # sigma
inputs.type = 'call'
inputs.monte_carlo_size = 10**6

# Initialize the options manager with the inputs
option_mgr = options.manager(inputs)

# Compute option prices using Black-Scholes and Monte Carlo methods
option_mgr.compute_black_scholes_price()
option_mgr.compute_monte_carlo_price()

# Perform Monte Carlo simulations for underlying asset prices
N = np.random.standard_normal(inputs.monte_carlo_size)
price_underlying = inputs.price * np.exp(
    (inputs.interest_rate - 0.5 * (inputs.volatility**2)) * inputs.maturity +
    inputs.volatility * np.sqrt(inputs.maturity) * N)
monte_carlo_simulations = np.array(
    [max(0, s - inputs.strike) for s in price_underlying])
monte_carlo_price = np.mean(monte_carlo_simulations)

# Compute the Monte Carlo confidence interval
monte_carlo_confidence_interval = monte_carlo_price + np.array([-1, +1]) * 1.96 * np.std(monte_carlo_simulations) / np.sqrt(inputs.monte_carlo_size)

# Output the standard deviation of the Monte Carlo simulations
std_monte_carlo_price = np.std(monte_carlo_simulations) / np.sqrt(inputs.monte_carlo_size)

# Plot the histogram of the Monte Carlo simulations
option_mgr.plot_histogram()

# Print the results
print(f"Monte Carlo estimated price: {monte_carlo_price}")
print(f"Standard deviation of Monte Carlo price: {std_monte_carlo_price}")
print(f"Monte Carlo confidence interval: {monte_carlo_confidence_interval}")
