# -*- coding: utf-8 -*-
"""
Cálculos y análisis de distribuciones financieras.
Created on Thu Sep  7 09:11:35 2023
@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import sys

# Ruta del módulo personalizado
module_path = 'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master/'
if module_path not in sys.path:
    sys.path.append(module_path)

# Función para cargar datos
def load_timeseries(ric, directory):
    """
    Carga los datos de la serie temporal desde un archivo CSV.
    """
    path = os.path.join(directory, f"{ric}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    raw_data = pd.read_csv(path)
    if raw_data.empty:
        raise ValueError(f"El archivo {path} está vacío.")
    
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], utc=True, errors='coerce')
    t['close'] = raw_data['Close']
    t = t.sort_values(by='date').dropna().reset_index(drop=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close'] / t['close_previous'] - 1
    t = t.dropna().reset_index(drop=True)
    return t

# Clase para análisis de distribuciones
class Distribution:
    def __init__(self, ric, directory, investment_amount, decimals=5, factor=252):
        self.ric = ric
        self.directory = directory
        self.investment_amount = investment_amount
        self.decimals = decimals
        self.factor = factor
        self.timeseries = None
        self.vector = None
        self.mean_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.var_95 = None
        self.skewness = None
        self.kurtosis = None
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None
        self.max_loss = None
        self.expected_loss = None
        self.expected_gain = None
        self.max_gain = None
        self.most_probable = None
        self.current_price = None

    def load_timeseries(self):
        """
        Carga la serie temporal y calcula los retornos.
        """
        try:
            self.timeseries = load_timeseries(self.ric, self.directory)
            self.vector = self.timeseries['return_close'].values
            self.current_price = self.timeseries['close'].iloc[-1]
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        except ValueError as e:
            print(e)
            sys.exit(1)

    def compute_stats(self):
        """
        Calcula estadísticas clave:
        - Media anualizada
        - Volatilidad anualizada
        - Ratio de Sharpe
        - Percentil al 5% (VaR)
        - Test de normalidad (Jarque-Bera)
        """
        self.mean_annual = np.mean(self.vector) * self.factor
        self.volatility_annual = np.std(self.vector) * np.sqrt(self.factor)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual if self.volatility_annual > 0 else 0.0
        self.var_95 = np.percentile(self.vector, 5)
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = len(self.vector) / 6 * (self.skewness**2 + (1 / 4) * self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = self.p_value > 0.05

        # Calcular escenarios de inversión
        self.max_loss = self.current_price * self.var_95
        self.expected_loss = self.current_price * np.mean(self.vector[self.vector < 0])  # Media de los retornos negativos
        self.expected_gain = self.current_price * np.mean(self.vector[self.vector > 0])  # Media de los retornos positivos
        self.max_gain = self.current_price * np.max(self.vector)
        self.most_probable = self.current_price * np.median(self.vector)  # Retorno más probable

        # Diagnóstico
        print(f"Datos para {self.ric}:")
        print(f"  Precio Actual: {self.current_price:.2f}")
        print(f"  Media Anualizada: {self.mean_annual:.5f}")
        print(f"  Volatilidad Anualizada: {self.volatility_annual:.5f}")
        print(f"  Ratio de Sharpe: {self.sharpe_ratio:.5f}")
        print(f"  JB Stat: {self.jb_stat:.5f}, p-value: {self.p_value:.5f}")
        print(f"  ¿Distribución Normal?: {self.is_normal}")

        # Interpretación
        print("\nInterpretaciones:")
        print(f"- Media Anualizada: La media anualizada de los retornos es {self.mean_annual:.5f}. Esto indica el retorno esperado de la inversión durante un año.")
        print(f"- Volatilidad Anualizada: La volatilidad anualizada es {self.volatility_annual:.5f}, lo que nos dice cuánta variabilidad o riesgo tiene el activo en su rendimiento anual.")
        print(f"- Ratio de Sharpe: El ratio de Sharpe es {self.sharpe_ratio:.5f}. Un valor más alto indica un mejor rendimiento ajustado por riesgo. En este caso, el activo tiene un ratio de Sharpe de {self.sharpe_ratio:.5f}, lo cual es {('bueno' if self.sharpe_ratio > 1 else 'bajo')} para este tipo de análisis.")
        print(f"- VaR al 95%: El percentil al 5% (VaR) es {self.var_95:.5f}. Esto significa que hay un 5% de probabilidad de que los retornos caigan por debajo de este valor, lo que indica el riesgo de pérdida en condiciones extremas.")
        print(f"- Skewness: La asimetría (skewness) es {self.skewness:.5f}. Si es positiva, el activo tiene una cola hacia los rendimientos altos; si es negativa, hacia los bajos.")
        print(f"- Kurtosis: La curtosis es {self.kurtosis:.5f}. Esto mide la \"altitud\" de la distribución. Una curtosis mayor a 3 indica colas más gruesas (mayor riesgo de eventos extremos).")
        print(f"- Jarque-Bera (JB Stat): El valor del estadístico de Jarque-Bera es {self.jb_stat:.5f}. Un valor cercano a 0 sugiere que los datos siguen una distribución normal. Un p-value mayor que 0.05 (p-value={self.p_value:.5f}) sugiere que no se puede rechazar la hipótesis de normalidad.")
        print(f"- Pérdida Máxima: La pérdida máxima esperada es {self.max_loss:.2f} unidades monetarias por unidad del activo.")
        print(f"- Pérdida Esperada: La pérdida esperada es {self.expected_loss:.2f} unidades monetarias por unidad del activo.")
        print(f"- Ganancia Esperada: La ganancia esperada es {self.expected_gain:.2f} unidades monetarias por unidad del activo.")
        print(f"- Ganancia Máxima: La ganancia máxima esperada es {self.max_gain:.2f} unidades monetarias por unidad del activo.")
        print(f"- Escenario Más Probable: El retorno más probable es {self.most_probable:.2f} unidades monetarias por unidad del activo.")

    def plot_histogram(self):
        """
        Grafica el histograma de los retornos.
        """
        title = (
            f"{self.ric} | mean_annual={self.mean_annual:.{self.decimals}f} "
            f"| volatility_annual={self.volatility_annual:.{self.decimals}f}\n"
            f"sharpe_ratio={self.sharpe_ratio:.{self.decimals}f} | var_95={self.var_95:.{self.decimals}f}\n"
            f"skewness={self.skewness:.{self.decimals}f} | kurtosis={self.kurtosis:.{self.decimals}f}\n"
            f"JB_stat={self.jb_stat:.{self.decimals}f} | p-value={self.p_value:.{self.decimals}f}\n"
            f"is_normal={self.is_normal}"
        )
        plt.figure()
        plt.hist(self.vector, bins=100, alpha=0.75, edgecolor='black')
        plt.title(title)
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.show()

    def plot_timeseries(self):
        """
        Grafica la serie temporal de precios.
        """
        plt.figure()
        plt.plot(self.timeseries['date'], self.timeseries['close'], color='blue', label='Close Price')
        plt.title(f'Timeseries of close prices for {self.ric}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_conclusions(self):
        """
        Grafica las conclusiones del análisis.
        """
        plt.figure(figsize=(12, 5))
        plt.title(f'Conclusiones del Análisis para {self.ric}')
        plt.axis('off')
        text = f"""
        Conclusiones para la toma de decisiones de inversión:
        - Precio Actual: {self.current_price:.2f}
        - Media Anualizada: {self.mean_annual:.5f}
        - Volatilidad Anualizada: {self.volatility_annual:.5f}
        - Ratio de Sharpe: {self.sharpe_ratio:.5f}
        - VaR al 95%: {self.var_95:.5f}
        - Pérdida Máxima: {self.max_loss:.2f} unidades monetarias por unidad del activo.
        - Pérdida Esperada: {self.expected_loss:.2f} unidades monetarias por unidad del activo.
        - Ganancia Esperada: {self.expected_gain:.2f} unidades monetarias por unidad del activo.
        - Ganancia Máxima: {self.max_gain:.2f} unidades monetarias por unidad del activo.
        - Escenario Más Probable: {self.most_probable:.2f} unidades monetarias por unidad del activo.

        Basado en estos resultados:
        - Si la media anualizada es alta y el ratio de Sharpe es mayor a 1, es favorable invertir en el activo.
        - Si la volatilidad anualizada y el VaR al 95% son bajos, el riesgo es menor.
        - La pérdida máxima y la ganancia máxima indican los extremos posibles, mientras que el escenario más probable da una idea del retorno esperado.
        - Para un periodo de un año, con una inversión mínima de {self.investment_amount} unidades monetarias, se puede esperar un retorno de {self.most_probable:.2f} unidades monetarias en el escenario más probable.

        Diagnóstico:
        - {self.diagnose_investment()}

        Fórmulas utilizadas:
        - Pérdida Máxima: current_price * var_95
        - Pérdida Esperada: current_price * np.mean(vector[vector < 0])
        - Ganancia Esperada: current_price * np.mean(vector[vector > 0])
        - Ganancia Máxima: current_price * np.max(vector)
        - Retorno Más Probable: current_price * np.median(vector)
        """
        plt.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12)
        plt.show()

    def diagnose_investment(self):
        """
        Diagnostica si es favorable invertir en el activo.
        """
        if self.mean_annual > 0 and self.sharpe_ratio > 1 and self.volatility_annual < 0.2 and self.var_95 > -0.2:
            return "Los resultados indican que es favorable invertir en el activo."
        else:
            return "Los resultados indican que no es favorable invertir en el activo."

# Parámetros
ric = 'ALUA.BA'  # Cambiar por cualquier activo deseado
directory = r'C://Users//Outlet VL//Desktop//PRUEBA SIMULADOR//data-master/'
investment_amount = 10000  # Cantidad de dinero a invertir

# Ejecución
dist = Distribution(ric, directory, investment_amount)
dist.load_timeseries()
dist.plot_timeseries()
dist.compute_stats()
dist.plot_histogram()
dist.plot_conclusions()