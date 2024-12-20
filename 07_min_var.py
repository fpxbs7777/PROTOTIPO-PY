# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:00:46 2023

@author: Meva
"""

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

# create offline instances of pca_model
notional = 1 # in mn USD
universe = ["GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA"]
rics = random.sample(universe, 15)

# define the directory
directory = 'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master//'

# sincronizar todas las series temporales de rendimientos
df = market_data.synchronise_returns(rics, directory)

# Agregar diagnósticos para verificar el contenido del DataFrame
print("Contenido del DataFrame después de la sincronización:")
print(df.head())

# Verificar si el DataFrame está vacío
if df.empty:
    raise ValueError("El DataFrame de rendimientos está vacío después de la sincronización.")

mtx = df.drop(columns=['date'])
if mtx.shape[0] == 0:
    raise ValueError("El DataFrame de rendimientos no tiene filas después de eliminar la columna de fechas.")

mtx_var_covar = np.cov(mtx, rowvar=False) * 252
mtx_correl = np.corrcoef(mtx, rowvar=False)

# min-var with eigenvectors
try:
    eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
    min_var_vector = eigenvectors[:,0]
except np.linalg.LinAlgError as e:
    raise ValueError("Los autovalores no convergieron. Verifica la matriz de varianza-covarianza.") from e

# unit test for variance function
variance_1 = np.matmul(np.transpose(min_var_vector), np.matmul(mtx_var_covar, min_var_vector))

######################################
# min-var with scipy optimize minimize
######################################

# function to minimize
def portfolio_variance(x, mtx_var_covar):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_covar, x))
    return variance

# compute optimisation
x0 = [1 / np.sqrt(len(rics))] * len(rics)
l2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
l1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
optimal_result = op.minimize(fun=portfolio_variance, x0=x0, args=(mtx_var_covar), constraints=(l2_norm))
optimize_vector = optimal_result.x
variance_2 = optimal_result.fun

df_weights = pd.DataFrame()
df_weights['rics'] = rics
df_weights['min_var_vector'] = min_var_vector
df_weights['optimize_vector'] = optimize_vector

print("Resultados de los pesos mínimos de varianza:")
print(df_weights)
# %%

