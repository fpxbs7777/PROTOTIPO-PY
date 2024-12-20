import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as op
import importlib

import market_data
importlib.reload(market_data)

def compute_beta(benchmark, security, directory):
    m = model(benchmark, security, directory)
    m.synchronise_timeseries()
    m.compute_linear_regression()
    return m.beta

def compute_correlation(security_1, security_2, directory):
    m = model(security_1, security_2, directory)
    m.synchronise_timeseries()
    m.compute_linear_regression()
    return m.correlation

def dataframe_correlation_beta(benchmark, position_security, hedge_universe, directory):
    decimals = 5
    df = pd.DataFrame()
    correlations = []
    betas = []
    for hedge_security in hedge_universe:
        try:
            correlation = compute_correlation(position_security, hedge_security, directory)
            beta = compute_beta(benchmark, hedge_security, directory)
            correlations.append(np.round(correlation, decimals))
            betas.append(np.round(beta, decimals))
        except ValueError as e:
            correlations.append(np.nan)
            betas.append(np.nan)
    df['hedge_security'] = hedge_universe
    df['correlation'] = correlations
    df['beta'] = betas
    df = df.dropna()
    df = df.sort_values(by='correlation', ascending=False)
    return df

def dataframe_factors(security, factors, directory):
    decimals = 5
    df = pd.DataFrame()
    correlations = []
    betas = []
    for factor in factors:
        try:
            correlation = compute_correlation(factor, security, directory)
            beta = compute_beta(factor, security, directory)
            correlations.append(np.round(correlation, decimals))
            betas.append(np.round(beta, decimals))
        except ValueError as e:
            correlations.append(np.nan)
            betas.append(np.nan)
    df['factor'] = factors
    df['correlation'] = correlations
    df['beta'] = betas
    df = df.dropna()
    df = df.sort_values(by='correlation', ascending=False)
    return df

# Definir la función de costo para la optimización del hedge
def cost_function_capm(x, betas, target_delta, target_beta, regularisation):
    f_delta = (np.sum(x) + target_delta) ** 2
    f_beta = (np.dot(betas, x) + target_beta) ** 2
    f_penalty = regularisation * np.sum(x ** 2)
    f = f_delta + f_beta + f_penalty
    return f

class model:
    def __init__(self, benchmark, security, directory, decimals=6):
        self.benchmark = benchmark
        self.security = security
        self.directory = directory
        self.decimals = decimals
        self.timeseries = None
        self.x = None
        self.y = None
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = None
        self.correlation = None
        self.r_squared = None
        self.predictor_linreg = None

    def synchronise_timeseries(self):
        self.timeseries = market_data.synchronise_timeseries(self.benchmark, self.security, self.directory)
        if self.timeseries.empty:
            raise ValueError(f"Timeseries for {self.benchmark} and {self.security} is empty.")

    def plot_timeseries(self):
        plt.figure(figsize=(12, 5))
        plt.title('Series de tiempo de precios de cierre')
        plt.xlabel('Tiempo')
        plt.ylabel('Precios')
        ax = plt.gca()
        self.timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True, color='blue', label=self.benchmark)
        self.timeseries.plot(kind='line', x='date', y='close_y', ax=ax, grid=True, color='red', secondary_y=True, label=self.security)
        plt.legend()
        plt.show()

    def compute_linear_regression(self):
        self.x = self.timeseries['return_x'].values
        self.y = self.timeseries['return_y'].values
        if len(self.x) == 0 or len(self.y) == 0:
            raise ValueError("Los vectores para regresión lineal no deben estar vacíos.")
        slope, intercept, r_value, p_value, std_err = st.linregress(self.x, self.y)
        self.alpha = np.round(intercept, self.decimals)
        self.beta = np.round(slope, self.decimals)
        self.p_value = np.round(p_value, self.decimals)
        self.null_hypothesis = p_value > 0.05
        self.correlation = np.round(r_value, self.decimals)
        self.r_squared = np.round(r_value ** 2, self.decimals)
        self.predictor_linreg = intercept + slope * self.x

    def plot_linear_regression(self):
        str_self = f'Regresión lineal | security {self.security} | benchmark {self.benchmark}\n' \
                   f'alpha (intercept): {self.alpha} | beta (slope): {self.beta}\n' \
                   f'p-value: {self.p_value} | hipótesis nula: {self.null_hypothesis}\n' \
                   f'correlación (r): {self.correlation} | r-cuadrado: {self.r_squared}'
        plt.figure()
        plt.title(f'Dispersión de retornos\n{str_self}')
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.predictor_linreg, color='green')
        plt.xlabel(self.benchmark)
        plt.ylabel(self.security)
        plt.grid()
        plt.show()

class Hedger:
    def __init__(self, position_security, position_delta_usd, benchmark, hedge_securities, directory):
        self.position_security = position_security
        self.position_delta_usd = position_delta_usd
        self.position_beta = None
        self.position_beta_usd = None
        self.benchmark = benchmark
        self.hedge_securities = hedge_securities
        self.hedge_betas = []
        self.hedge_weights = None
        self.hedge_delta_usd = None
        self.hedge_cost_usd = None
        self.hedge_beta_usd = None
        self.directory = directory

    def compute_betas(self):
        self.position_beta = compute_beta(self.benchmark, self.position_security, self.directory)
        self.position_beta_usd = self.position_beta * self.position_delta_usd
        for security in self.hedge_securities:
            beta = compute_beta(self.benchmark, security, self.directory)
            self.hedge_betas.append(beta)

    def compute_hedge_weights(self, regularisation=0):
        x0 = [-self.position_delta_usd / len(self.hedge_betas)] * len(self.hedge_betas)
        optimal_result = op.minimize(
            fun=cost_function_capm,
            x0=x0,
            args=(self.hedge_betas, self.position_delta_usd, self.position_beta_usd, regularisation)
        )
        self.hedge_weights = optimal_result.x
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        self.hedge_cost_usd = np.sum(np.abs(self.hedge_weights))
        self.hedge_beta_usd = np.dot(self.hedge_betas, self.hedge_weights)

    def compute_hedge_weights_exact(self):
        dimensions = len(self.hedge_securities)
        if dimensions != 2:
            print('No se puede calcular la solución exacta porque las dimensiones son:', dimensions, '≠ 2')
            return
        betas = np.array(self.hedge_betas)
        A = np.vstack([np.ones(dimensions), betas]).T
        b = np.array([-self.position_delta_usd, -self.position_beta_usd])
        self.hedge_weights = np.linalg.solve(A, b)
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        self.hedge_beta_usd = np.dot(betas, self.hedge_weights)