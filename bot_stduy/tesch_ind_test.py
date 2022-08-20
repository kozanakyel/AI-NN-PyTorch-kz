import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
import yfinance as yf

data = yf.download('ASELS.IS', '2020-01-01')

df = pd.DataFrame(data)
df.columns= df.columns.str.lower()
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)

aselsan = StockDataFrame(df)

# print(aselsan)

aselsan[['close_10_sma', 'close_20_sma', 'close_50_sma', 'rsi_11', 'rsi_14', 'rsi_21','macd', 'macdh', 'macds']]
rsi = aselsan.get('rsi')
macd = aselsan.get('macd')
print(aselsan['macd'].tail())

# 1. BOLLINGER BANDS
# aselsan[['boll', 'boll_ub', 'boll_lb']]
# 2. RSI
#aselsan[['rsi_11', 'rsi_14', 'rsi_21']]
# 3. WILLIAMS %R
# aselsan[['wr_11', 'wr_14', 'wr_21']]
# 4. MACD
# aselsan[['macd', 'macdh', 'macds']]

# 5. COMMODITY CHANNEL INDEX
# aselsan[['cci_11', 'cci_14', 'cci_21']]

buy_signals = aselsan['close_50_sma_xd_close_20_sma']
sell_signals = aselsan['close_20_sma_xd_close_50_sma']

print(aselsan.shape)

for i in range(len(buy_signals)):
    if aselsan['close_20_sma'].iloc[i] < aselsan['close_50_sma'].iloc[i] \
            and aselsan['rsi_14'].iloc[i] < 30:
        buy_signals.iloc[i] = aselsan.close[i]
    else:
        buy_signals.iloc[i] = np.nan
for i in range(len(sell_signals)):
    if sell_signals.iloc[i]:
        sell_signals.iloc[i] = aselsan.close[i]
    else:
        sell_signals.iloc[i] = np.nan

plt.plot(aselsan['close'], linewidth=2.5, label='ASELSAN')
plt.plot(aselsan['close_20_sma'], linewidth=2.5, alpha=0.6, label='SMA 20')
plt.plot(aselsan['close_50_sma'], linewidth=2.5, alpha=0.6, label='SMA 50')
plt.plot(aselsan['rsi_11'], linewidth=2.5, alpha=0.6, label='RSI 11')
plt.plot(aselsan['rsi_14'], linewidth=2.5, alpha=0.6, label='RSI 14')
plt.plot(aselsan['rsi_21'], linewidth=2.5, alpha=0.6, label='RSI 21')
plt.plot(aselsan['macd']*10, linewidth=2.5, alpha=0.6, label='MACD')
plt.plot(aselsan['macdh']*10, linewidth=2.5, alpha=0.6, label='MACD H')
plt.plot(aselsan['macds'], linewidth=2.5, alpha=0.6, label='MACD S')
plt.plot(aselsan.index, buy_signals, marker='^', markersize=15, color='green', linewidth=0, label='BUY SIGNAL')
plt.plot(aselsan.index, sell_signals, marker='v', markersize=15, color='r', linewidth=0, label='SELL SIGNAL')
plt.legend(loc='upper left')
plt.title('ASELSAN SMA 20,50 CROSSOVER STRATEGY SIGNALS \n And RSI 11, 14, 21 values')
plt.show()

