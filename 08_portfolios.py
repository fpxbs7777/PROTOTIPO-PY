# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:18:34 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import random
import scipy.optimize as op

# importar nuestros archivos y recargarlos
import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)
import portfolio
importlib.reload(portfolio)

# inputs
notional = 15 # in mn USD
universe =  ["GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA"]
#["GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA"]
#['XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU',"GGAL.BA","YPFD.BA","BMA.BA","PAMP.BA","TECO2.BA","SUPV.BA", "CEPU.BA", "COME.BA","TXAR.BA", "ALUA.BA","EDN.BA","TGSU2.BA","MIRG.BA", "TRAN.BA","CRES.BA", "HARG.BA","IRSA.BA","LOMA.BA","VALO.BA","AGRO.BA",'BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD']
# Ajustar el tamaño de la muestra para que no exceda el tamaño de la lista universe
sample_size = min(30, len(universe))
rics = random.sample(universe, sample_size)

directory = 'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master//'

print(rics)

# inicializar la instancia de la clase
port_mgr = portfolio.manager(rics, notional, directory)

# computar correlación y matriz de varianza-covarianza
port_mgr.compute_covariance()

# computar los portafolios deseados: clase de salida = portfolio.output
port_min_variance_l1 = port_mgr.compute_portfolio('min-variance-l1')
port_min_variance_l2 = port_mgr.compute_portfolio('min-variance-l2')
port_long_only = port_mgr.compute_portfolio('long-only')
port_equi_weight = port_mgr.compute_portfolio('equi-weight')
port_markowitz = port_mgr.compute_portfolio('markowitz', target_return=None)

# graficar los histogramas de retornos para el portafolio deseado
port_min_variance_l1.plot_histogram()
port_min_variance_l2.plot_histogram()
port_long_only.plot_histogram()
port_equi_weight.plot_histogram()
port_markowitz.plot_histogram()