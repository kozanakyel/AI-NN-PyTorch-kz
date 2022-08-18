import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def SMA(data, ndays):
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name='SMA')
    data = data.join(SMA)
    return data

data = yf.download('EREGL.IS', start='2020-01-01', end='2022-08-18')
close = data['Close']

n = 50
SMA = SMA(data, n)
SMA = SMA.dropna()
SMA = SMA['SMA']

plt.figure(figsize=(10,7))
plt.title('Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(data['Close'], lw=1, label='Close Price')
plt.plot(SMA, 'g', lw=1, label='50-day SMA')
plt.legend()
plt.show()