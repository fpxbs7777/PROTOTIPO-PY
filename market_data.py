import os
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def load_timeseries(ric, directory):
    path = os.path.join(directory, f"{ric}.csv")
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], utc=True, errors='coerce').dt.normalize()
    t['close'] = raw_data['Close']
    t = t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return'] = t['close'] / t['close_previous'] - 1
    t = t.dropna(subset=['date', 'close', 'close_previous', 'return'])
    t = t.reset_index(drop=True)
    return t

def synchronise_timeseries(benchmark, security, directory):
    timeseries_x = load_timeseries(benchmark, directory)
    timeseries_y = load_timeseries(security, directory)
    
    if timeseries_x.empty or timeseries_y.empty:
        return pd.DataFrame()
    
    # Obtener las fechas comunes
    common_dates = pd.to_datetime(timeseries_x['date']).isin(pd.to_datetime(timeseries_y['date']))
    timeseries_x = timeseries_x[common_dates].sort_values(by='date').reset_index(drop=True)
    timeseries_y = timeseries_y[timeseries_y['date'].isin(timeseries_x['date'])].sort_values(by='date').reset_index(drop=True)
    
    timeseries = pd.DataFrame()
    timeseries['date'] = timeseries_x['date']
    timeseries['close_x'] = timeseries_x['close']
    timeseries['close_y'] = timeseries_y['close']
    timeseries['return_x'] = timeseries_x['return']
    timeseries['return_y'] = timeseries_y['return']
    
    return timeseries

def synchronise_returns(rics, directory):
    dic_timeseries = {}
    for ric in rics:
        t = load_timeseries(ric, directory)
        dic_timeseries[ric] = t
    
    # Obtener las fechas comunes a todos los RICs
    common_dates = set(dic_timeseries[rics[0]]['date'])
    for ric in rics[1:]:
        common_dates = common_dates.intersection(set(dic_timeseries[ric]['date']))
    common_dates = sorted(common_dates)
    
    df = pd.DataFrame({'date': common_dates})
    for ric in rics:
        t = dic_timeseries[ric]
        t = t[t['date'].isin(common_dates)].sort_values(by='date').reset_index(drop=True)
        df[ric] = t['return'].values
    return df

class distribution:
    def __init__(self, ric, directory, decimals=5):  
        self.ric = ric
        self.directory = directory
        self.decimals = decimals
        self.str_title = None
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
        
    def load_timeseries(self):
        self.timeseries = load_timeseries(self.ric, self.directory)
        self.vector = self.timeseries['return'].values
        self.size = len(self.vector)
        self.str_title = self.ric + " | datos reales"
        
    def plot_timeseries(self):
        plt.figure()
        self.timeseries.plot(kind='line', x='date', y='close', grid=True, color='blue',
                             title='Serie de precios de cierre para ' + self.ric)
        plt.show()
                
    def compute_stats(self, factor=252):
        self.mean_annual = st.tmean(self.vector) * factor
        self.volatility_annual = st.tstd(self.vector) * np.sqrt(factor)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual if self.volatility_annual > 0 else 0.0
        self.var_95 = np.percentile(self.vector, 5)
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.size / 6 * (self.skewness**2 + (self.kurtosis**2) / 4)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05)  # Equivalente a jb_stat < 6
                
    def plot_histogram(self):
        self.str_title += '\n' + 'mean_annual=' + str(np.round(self.mean_annual, self.decimals)) \
            + ' | ' + 'volatility_annual=' + str(np.round(self.volatility_annual, self.decimals)) \
            + '\n' + 'sharpe_ratio=' + str(np.round(self.sharpe_ratio, self.decimals)) \
            + ' | ' + 'var_95=' + str(np.round(self.var_95, self.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness, self.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis, self.decimals)) \
            + '\n' + 'JB stat=' + str(np.round(self.jb_stat, self.decimals)) \
            + ' | ' + 'p-value=' + str(np.round(self.p_value, self.decimals)) \
            + '\n' + 'is_normal=' + str(self.is_normal)
        plt.figure()
        plt.hist(self.vector, bins=100)
        plt.title(self.str_title)
        plt.show()