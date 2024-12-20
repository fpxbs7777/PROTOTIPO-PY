import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import random
import scipy.optimize as op

# import our own files and reload
import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)
import portfolio
importlib.reload(portfolio)

# inputs
notional = 1 # in mn USD
universe = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI','^FCHI','^VIX',
            'XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU',
            'SPY','EWW',
            'IVW','IVE','QUAL','MTUM','SIZE','USMV',\
            'AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
            'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
            'LLY','JNJ','PG','MRK','ABBV','PFE',\
            'BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD',\
            'EURUSD=X','GBPUSD=X','CHFUSD=X','SEKUSD=X','NOKUSD=X','JPYUSD=X','MXNUSD=X',"GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA"]
rics = random.sample(universe, 50)
#◘rics = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI']
#rics = ['XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU']
#rics = ['IVW','IVE','QUAL','MTUM','SIZE','USMV']
#rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX']
#rics = ['BRK-B','JPM','V','MA','BAC','MS','GS','BLK']
#rics = ['LLY','JNJ','PG','MRK','PFE']
#rics = ['BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD']
#rics = ['BTC-USD','ETH-USD','SOL-USD']
#rics = ['USDC-USD','USDT-USD','DAI-USD']
#rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',
        #'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',         'LLY','JNJ','PG','MRK','ABBV','PFE']
#rics = ['XLC', 'XLU', 'QUAL', 'BTC-USD', 'XLY', 'NOKUSD=X', 'XLB', '^GDAXI', 'GOOG', 'CHFUSD=X']
#rics = universe
rics= ["GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA"]

# efficient frontier
target_return = None
include_min_variance = True
dict_portfolios = portfolio.compute_efficient_frontier(rics, notional, target_return, include_min_variance)
print(rics)
dict_portfolios['markowitz-target'].plot_histogram()


# Obtener los pesos del portafolio óptimo
optimal_weights = dict_portfolios['markowitz-target'].weights

# Imprimir las claves disponibles
print("\nClaves disponibles en dict_portfolios:")
print(dict_portfolios.keys())

# Mostrar pesos no nulos (> 0.1%)
print("\nPesos óptimos del portafolio (posiciones > 0.1%):")
print("-" * 50)
df_alloc = dict_portfolios['markowitz-target'].dataframe_allocation
for _, row in df_alloc[df_alloc['weights'] > 0.01].iterrows():
    print(f"{row['rics']:10} : {row['weights']:8.2%}")
print("-" * 50)
print(f"Suma total: {df_alloc['weights'].sum():8.2%}")

# Graficar
plt.figure(figsize=(10, 6))

# Usar volatilidades y retornos del dataframe_allocation
#plt.scatter(df_alloc['volatilities'], 
        #   df_alloc['returns'],
        #   alpha=0.5,
         #  s=50,
        #   label='Activos Individuales')

# Marcar activos seleccionados (peso > 0.1%)
#selected = df_alloc[df_alloc['weights'] > 10.0]
#plt.scatter(selected['volatilities'],
          # selected['returns'],
         #  color='red',
          # s=100,
         #  label='Activos Seleccionados')

#plt.xlabel('Riesgo (Volatilidad)')
#plt.ylabel('Rendimiento Esperado')
#plt.title('Selección de Activos en el Portafolio Óptimo')
#plt.legend()
#plt.grid(True)
#plt.show()





