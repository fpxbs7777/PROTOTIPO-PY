import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# import our own files and reload
import capm
importlib.reload(capm)

# inputs
security = 'LOMA'
factors = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI','^FCHI','^VIX',
            'XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU',
            'SPY','EWW',
            'IVW','IVE','QUAL','MTUM','SIZE','USMV']
directory = 'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master//'

# compute factors
df = capm.dataframe_factors(security, factors, directory)

# Imprimir resultados
print("Factors DataFrame:")
print(df)