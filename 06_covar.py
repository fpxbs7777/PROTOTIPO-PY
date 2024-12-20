import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# Importar nuestros propios archivos y recargar
import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)

# Lista de RICs
rics = ["GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA"]

# Definir el directorio donde se encuentran tus datos
directory = 'C:/Users/Outlet VL/Desktop/PRUEBA SIMULADOR/data-master/'

# Sincronizar todas las series de tiempo de retornos
df = market_data.synchronise_returns(rics, directory)

# Calcular las matrices de varianza-covarianza y correlación
mtx = df.drop(columns=['date'])
mtx_var_covar = np.cov(mtx, rowvar=False) * 252
mtx_correl = np.corrcoef(mtx, rowvar=False)

# Calcular autovalores y autovectores
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
variance_explained = eigenvalues / np.sum(eigenvalues)
prod = np.matmul(eigenvectors, np.transpose(eigenvectors))

##########################
# PCA para visualización 2D
##########################

# Calcular volatilidades mínima y máxima
volatility_min = np.sqrt(eigenvalues[0])
volatility_max = np.sqrt(eigenvalues[-1])

# Calcular la base PCA para visualización 2D
pca_vector_1 = eigenvectors[:, -1]
pca_vector_2 = eigenvectors[:, -2]
pca_eigenvalue_1 = eigenvalues[-1]
pca_eigenvalue_2 = eigenvalues[-2]
pca_variance_explained = variance_explained[-2:].sum()

# Calcular el portafolio de mínima varianza
min_var_vector = eigenvectors[:, 0]
min_var_eigenvalue = eigenvalues[0]
min_var_variance_explained = variance_explained[0]