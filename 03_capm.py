# -*- coding: utf-8 -*-
"""
Herramienta de análisis y comparación: Modelo CAPM con Benchmark
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# Importar y recargar módulos personalizados (si existen)
import capm
import market_data
importlib.reload(capm)
importlib.reload(market_data)


class CAPMModel:
    def __init__(self, security, benchmarks, directory):
        """
        Inicializa el modelo CAPM.

        Args:
            security (str): Ticker del activo de interés.
            benchmarks (list): Lista de tickers para los factores (benchmark).
            directory (str): Ruta al directorio que contiene los datos CSV.
        """
        self.security = security
        self.benchmarks = benchmarks
        self.directory = directory
        self.security_data = None
        self.benchmark_data = {}
        self.merged_data = None
        self.regression_results = None

    def load_timeseries(self, ticker):
        """
        Carga datos de la serie temporal de un archivo CSV.

        Args:
            ticker (str): Ticker del activo.

        Returns:
            pd.DataFrame: DataFrame con las columnas ['date', 'close', 'return'].
        """
        path = os.path.join(self.directory, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        # Cargar datos y calcular retornos
        data = pd.read_csv(path)
        if 'Date' not in data.columns or 'Close' not in data.columns:
            raise ValueError(f"El archivo {path} no contiene las columnas 'Date' y 'Close'.")

        data['date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)  # Convertir las fechas a UTC
        data = data.dropna(subset=['date', 'Close']).reset_index(drop=True)
        data['close'] = data['Close'].astype(float)
        data['return'] = data['close'].pct_change()  # Calcular retornos
        data['date'] = data['date'].dt.date  # Eliminar horas y minutos para solo quedarnos con la fecha
        
        return data[['date', 'close', 'return']].dropna().reset_index(drop=True)

    def synchronise_timeseries(self):
        """
        Sincroniza las series temporales del security y los benchmarks.
        """
        self.security_data = self.load_timeseries(self.security)

        for benchmark in self.benchmarks:
            self.benchmark_data[benchmark] = self.load_timeseries(benchmark)

        # Intersección de fechas comunes
        common_dates = set(self.security_data['date'])
        for benchmark, data in self.benchmark_data.items():
            common_dates = common_dates.intersection(set(data['date']))

        if not common_dates:
            raise ValueError("No hay fechas comunes entre el security y los benchmarks.")

        self.security_data = self.security_data[self.security_data['date'].isin(common_dates)].sort_values(by='date')
        for benchmark in self.benchmarks:
            self.benchmark_data[benchmark] = self.benchmark_data[benchmark][
                self.benchmark_data[benchmark]['date'].isin(common_dates)
            ].sort_values(by='date')

        self.merged_data = self.security_data.rename(columns={'return': f'return_{self.security}', 'close': f'close_{self.security}'})
        for benchmark in self.benchmarks:
            self.merged_data = self.merged_data.merge(
                self.benchmark_data[benchmark][['date', 'return', 'close']].rename(columns={'return': f'return_{benchmark}', 'close': f'close_{benchmark}'}),
                on='date',
            )

    def compute_linear_regression(self):
        """
        Realiza una regresión lineal entre los retornos del security y los benchmarks.
        """
        if self.merged_data is None or self.merged_data.empty:
            raise ValueError("Las series de tiempo no han sido sincronizadas. Llama primero a `synchronise_timeseries`.")

        # Tomar los retornos del primer benchmark
        x = self.merged_data[f'return_{self.benchmarks[0]}']
        y = self.merged_data[f'return_{self.security}']
        
        # Realizar la regresión lineal
        slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
        nb_decimals = 4
        alpha = np.round(intercept, nb_decimals)
        beta = np.round(slope, nb_decimals)
        r_value = np.round(r_value, nb_decimals)
        null_hypothesis = p_value > 0.05  # p_value < 0.05 --> reject null hypothesis
        correlation = np.round(r_value, nb_decimals)  # correlation coefficient
        r_squared = np.round(r_value**2, nb_decimals)  # pct of variance of y explained by x
        predictor_linreg = intercept + slope * x

        # Guardar resultados de la regresión
        self.regression_results = {
            'beta': beta,
            'alpha': alpha,
            'r_value': r_value,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'null_hypothesis': null_hypothesis,
            'correlation': correlation
        }

        # Generar el gráfico de regresión lineal
        str_self = 'Linear regression | security ' + self.security + \
            ' | benchmark ' + self.benchmarks[0] + '\n' + \
            'alpha (intercept) ' + str(alpha) + '\n' + \
            'beta (slope) ' + str(beta) + '\n' + \
            'p-value ' + str(p_value) + '\n' + \
            'null hypothesis ' + str(null_hypothesis) + '\n' + \
            'correl (r-value) ' + str(correlation) + '\n' + \
            'r-squared ' + str(r_squared)
        str_title = 'Scatterplot of returns' + '\n' + str_self
        plt.figure()
        plt.title(str_title)
        plt.scatter(x, y)
        plt.plot(x, predictor_linreg, color='green')
        plt.xlabel(self.benchmarks[0])
        plt.ylabel(self.security)
        plt.grid()
        plt.show()

    def plot_timeseries(self):
        """
        Grafica las series temporales de precios de cierre.
        """
        timeseries = pd.DataFrame()
        timeseries['date'] = self.merged_data['date']
        timeseries['close_x'] = self.merged_data[f'close_{self.benchmarks[0]}']
        timeseries['close_y'] = self.merged_data[f'close_{self.security}']
        
        # Graficar series temporales
        plt.figure(figsize=(12,5))
        plt.title('Time series of close prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        ax1 = timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True, color='blue', label=self.benchmarks[0])
        ax2 = timeseries.plot(kind='line', x='date', y='close_y', ax=ax, grid=True, color='red', secondary_y=True, label=self.security)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()

    def plot_conclusions(self):
        """
        Grafica las conclusiones del análisis CAPM.
        """
        plt.figure(figsize=(12, 5))
        plt.title('Conclusiones del Análisis CAPM')
        plt.axis('off')
        text = f"""
        Benchmark: {self.benchmarks[0]}
        Beta: {self.regression_results['beta']}
        Alpha: {self.regression_results['alpha']}
        R^2: {self.regression_results['r_squared']}
        P-Value: {self.regression_results['p_value']}
        Std Err: {self.regression_results['std_err']}

        Conclusión: {"El activo es más volátil que el benchmark." if self.regression_results['beta'] > 1 else "El activo es menos volátil que el benchmark." if self.regression_results['beta'] < 1 else "El activo tiene la misma volatilidad que el benchmark."}
        Conclusión: {"El activo ha superado al benchmark." if self.regression_results['alpha'] > 0 else "El activo no ha superado al benchmark."}
        Conclusión: {"El modelo explica bien la relación entre el activo y el benchmark." if self.regression_results['r_squared'] > 0.7 else "El modelo no explica bien la relación entre el activo y el benchmark."}
        """
        plt.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12)
        plt.show()


# Inputs
benchmark = 'AAPL'
security = 'IWM'
directory = 'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master//'

# Ejecutar el modelo CAPM
try:
    model = CAPMModel(security, [benchmark], directory)
    model.synchronise_timeseries()
    model.compute_linear_regression()  # Asegúrate de llamar a este método
    model.plot_timeseries()  # Graficar las series temporales
    model.plot_conclusions()  # Graficar las conclusiones

    print("Resultados de la regresión lineal:")
    
    # Imprimir los resultados de la regresión con sus interpretaciones
    for key, value in model.regression_results.items():
        print(f"{key}: {value}")

    # Interpretaciones de los resultados:
    print("\nInterpretaciones:")
    print(f"\n1. Alpha: {model.regression_results['alpha']} - Este es el rendimiento del activo que no está explicado por el benchmark. Si es positivo, el activo ha superado al benchmark.")
    print(f"2. Beta: {model.regression_results['beta']} - Mide la sensibilidad del activo respecto al benchmark. Un beta mayor a 1 significa que el activo es más volátil que el benchmark.")
    print(f"3. R-squared: {model.regression_results['r_squared']} - Indica qué porcentaje de la variabilidad en los retornos del activo está explicada por los movimientos del benchmark. Un valor cercano a 1 es ideal.")
    print(f"4. P-value: {model.regression_results['p_value']} - Un p-value menor a 0.05 indica que la relación entre el activo y el benchmark es estadísticamente significativa.")
    print(f"5. Correlation (r-value): {model.regression_results['correlation']} - Mide la fuerza y dirección de la relación entre los retornos del activo y del benchmark.")
    print(f"6. Null hypothesis (p-value > 0.05): {model.regression_results['null_hypothesis']} - Si el p-value es mayor a 0.05, no podemos rechazar la hipótesis nula de que no existe relación entre el activo y el benchmark.")
except FileNotFoundError as fnf_error:
    print(fnf_error)
except Exception as e:
    print(f"Error ejecutando el modelo CAPM: {e}")