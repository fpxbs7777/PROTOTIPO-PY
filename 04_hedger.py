import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import scipy.optimize as op

# import our own files and reload
import capm
importlib.reload(capm)

# inputs
position_security = 'AAPL'
position_delta_usd = 10 # in mn USD
benchmark = 'IWM'
hedge_universe = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX','SPY','XLK','XLF']
hedge_universe = ['BRK-B','JPM','V','MA','BAC','MS','GS','BLK','SPY','XLF']
hedge_universe = ['QQQ','AAPL','^SPX','XLK','XLF', 'IWM']
regularisation = 0.1

# compute correlations
directory = 'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master//' # You need to set this to the correct directory path
df = capm.dataframe_correlation_beta(benchmark, position_security, hedge_universe, directory)
print("Correlation and Beta DataFrame:")
print(df)

# computations
hedge_securities = ['ETH-USD','SOL-USD']
hedger = capm.Hedger(position_security, position_delta_usd, benchmark, hedge_securities, directory)
hedger.compute_betas()
hedger.compute_hedge_weights(regularisation)

# variables
position_beta_usd = hedger.position_beta_usd
hedge_weights = hedger.hedge_weights
hedge_delta_usd = hedger.hedge_delta_usd
hedge_beta_usd = hedger.hedge_beta_usd
hedge_cost_usd = hedger.hedge_cost_usd

print(f"Position beta in USD: {position_beta_usd}")
print(f"Hedge weights: {hedge_weights}")
print(f"Hedge delta in USD: {hedge_delta_usd}")
print(f"Hedge beta in USD: {hedge_beta_usd}")
print(f"Hedge cost in USD: {hedge_cost_usd}")