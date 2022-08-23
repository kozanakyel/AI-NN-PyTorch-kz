import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
import yfinance as yf

data = yf.download('ASELS.IS', start="2022-08-01", end="2022-08-22", interval="30m")

df = pd.DataFrame(data)
df.columns= df.columns.str.lower()
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)

aselsan = StockDataFrame(df)

aselsan[['close_50_sma','close_10_sma', 'close_20_ema', 'close_5_ema', 'close_10_ema', 'rsi_3', 'rsi_14','macd', 'dma']]

buy_signals1 = aselsan['close_10_ema_xd_close_5_ema']

for i in range(len(buy_signals1)):
    if aselsan['close_5_ema'].iloc[i] >= aselsan['close_10_ema'].iloc[i] \
            and aselsan['close_20_ema'].iloc[i] >= aselsan['close_50_sma'].iloc[i] \
        and aselsan['close_10_sma'].iloc[i] <= aselsan.close[i] \
            and aselsan['rsi_3'].iloc[i] > aselsan['rsi_14'].iloc[i] \
                and aselsan['macdh'].iloc[i] > aselsan['macds'].iloc[i]:
        buy_signals1.iloc[i] = aselsan.close[i]
    else:
        buy_signals1.iloc[i] = np.nan

plt.plot(aselsan['close'], linewidth=2.5, label='ASELSAN')
plt.plot(aselsan['close_10_sma'], linewidth=2.5, alpha=0.6, label='SMA 10')
plt.plot(aselsan['close_50_sma'], linewidth=2.5, alpha=0.6, label='SMA 50')
plt.plot(aselsan['close_20_ema'], linewidth=2.5, alpha=0.6, label='EMA 20')
plt.plot(aselsan['close_10_ema'], linewidth=2.5, alpha=0.6, label='EMA 10')
plt.plot(aselsan['close_5_ema'], linewidth=2.5, alpha=0.6, label='EMA 5')
plt.plot(aselsan.index, buy_signals1, marker='^', markersize=15, color='green', linewidth=0, label='BUY SIGNAL')
plt.legend(loc='upper left')
plt.title('ASELSAN Trading View Strategy')
plt.show()