# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
#import numpy as np
#import tensorflow as tf
#from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
#from sklearn.model_selection import train_test_split  
import requests
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.stattools import adfuller
sns.set_style("darkgrid")

class XML2DataFrame:

    def __init__(self, xml_data):
        self.root = ET.XML(xml_data)

    def parse_root(self, root):
        return [self.parse_element(child) for child in iter(root)]

    def parse_element(self, element, parsed=None):
        if parsed is None:
            parsed = dict()
        for key in element.keys():
            parsed[key] = element.attrib.get(key)
        if element.text:
            parsed[element.tag] = element.text
        for child in list(element):
            self.parse_element(child, parsed)
        return parsed

    def process_data(self):
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)

def get_wx(sid, Y, m, d):
    url = 'https://www.wunderground.com/weatherstation/WXDailyHistory.asp?'
    url += 'ID='+sid
    url += '&day='+d
    url += '&month='+m
    url += '&year='+Y
    url += '&graphspan=day&format=XML'
    print(url)
    r = requests.get(url).content
    xml2df = XML2DataFrame(r)
    df = xml2df.process_data()
    df.index = pd.to_datetime(df.observation_time_rfc822)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df

def get_data():
    data = pd.DataFrame()
    for i in range(31):
        df = get_wx(sid, Y, m, str(i))
        data = data.append(df)     
    data.to_csv('data.csv', index=False)
    
def read_data():
    data = pd.read_csv('data.csv')
    data.index = pd.to_datetime(data.observation_time_rfc822)
    data = data.resample('240T').mean().bfill()
    data['solar_radiation'] = data['solar_radiation']/10
    #data['dat'] = data.index
    return data

def test_stationarity(ts):
    # Determing rolling statistics
    rolmean = ts.rolling(window=24).mean()
    
    # Plot rolling statistics:
    plt.plot(ts, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show(block=False)
    
    # Perform Augmented Dickey-Fuller test:
    print('Results of Augmented Dickey-Fuller test:')
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    plot_acf(ts, lags=50)
    plot_pacf(ts, lags=50)
    plt.xlabel('lags')
    plt.show()

def arma_pred(data, prop):
    
    p = d = range(0, 6)
    pdq = itertools.product(p, d)
    aic = 1e100
    for param in pdq:
        try:
            mod = ARMA(data[prop].dropna(), order=param)
            results = mod.fit()
            #print('ARMA{} - AIC:{}'.format(param, results.aic))
            if results.aic < aic:
                aic = results.aic
                param_use = param
        except:
            continue
    #print(param_use)
    ts_datemin = data[prop].dropna().index.min()
    ts_datemax = data[prop].dropna().index.max()
    ts_datefcst = ts_datemax + (24)
    model = ARMA(data[prop].dropna(), order=param_use)  
    results_MA = model.fit()
    res = pd.DataFrame(results_MA.predict(ts_datemin, ts_datefcst), columns=[prop+'_fcst'])
    plt.plot(res, color='blue')
    plt.plot(results_MA.fittedvalues, color='red')
    plt.plot(data[prop], color='black')
    plt.title('Fitting data _ MSE: %.2f'% (((results_MA.fittedvalues-data[prop])**2).mean()))
    plt.show()
    data = data.join(res, how='outer')
    return data

sid = 'KTXHOUST151'
d = '7'
m = '8'
Y = '2018'
prop = 'temp_f'
props = ['temp_f','dewpoint_f','pressure_in','relative_humidity','solar_radiation','UV','wind_mph']
props_fcst = ['temp_f_fcst','dewpoint_f_fcst','pressure_in_fcst','relative_humidity_fcst','solar_radiation_fcst','UV_fcst','wind_mph_fcst']

#sid = 'KCABAKER38'
#d = '24'
#m = '6'
#Y = '2017'

#get_data()
data = read_data()
#test_stationarity(data[prop])
for prop in props:
    print(prop)
    data = arma_pred(data, prop)

plt.figure(figsize=(20,10))
plt.plot(data[props_fcst])
plt.plot(data[props], marker='x', linewidth=0)
plt.savefig('fig.png', dpi=200)
