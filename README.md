# Pairs Finding
1. use Hierarchical Clustering
2. use fundemental approach to find industry neutural pairs
3. use IC information

```python
from polygon.rest import RESTClient

client = RESTClient("5MKBfYvCSCPekWl824p5l2aSHIIjqh_p")
main_exchanges = ["XNYS", "XNAS", "XASE"] 
tickers = []
for ex in main_exchanges:
	for t in client.list_tickers(
		market="stocks",
		exchange=ex,
		active="true",
		order="asc",
		limit="100",
		sort="ticker",
	):
		tickers.append(t.ticker)

tickers = list(set(tickers))
print(tickers)

```

```python
aggs = []
for a in client.list_aggs(
        "FHB",
        1,
        "day",
        "2020-04-01",
        "2025-04-01",
        adjusted="true",
        sort="asc",
        limit=50000,
    ):
        aggs.append(a.close)
print(len(aggs))
```

```python

import pandas as pd
import numpy as np

df = pd.DataFrame(
    columns=tickers
)

for t in tickers:
    aggs = []
    for a in client.list_aggs(
        t,
        1,
        "day",
        "2020-04-01",
        "2025-04-01",
        adjusted="true",
        sort="asc",
        limit=50000,
    ):
        
        aggs.append(a.close)
    if len(aggs) != 1242:
        continue
    df[t] = aggs

```

```python
df = df.dropna(axis=1).copy()
train, test = df.iloc[:int(1243*0.8)].copy(), df.iloc[int(1243*0.8):].copy()
train = train.apply(lambda x: np.log(x))
train_standard = train.apply(lambda x: (x - x.mean())/ x.std())
X = train_standard.T 
```

```python
test = test.apply(lambda x: np.log(x))
```

```python
X
```

```python
from scipy.cluster.hierarchy import linkage
Z = linkage(X, method='single')
first_merges = Z[:3735//2]
first_merges
```

```python
tickers[3107], tickers[3108]
```

```python
from scipy.cluster.hierarchy import linkage
import heapq

def find_pairs_with_single_linkage(data):
    tickers = data.index.tolist()
    n = data.shape[0]
    Z = linkage(data, method='single')
    first_merges = Z[:n//2]

    pairs = []
    result = []
    for row in first_merges:
        idx1, idx2 = int(row[0]), int(row[1])
        if idx1 < n and idx2 < n:
            ticker1 = tickers[idx1]
            ticker2 = tickers[idx2]
            distance = row[2]
            heapq.heappush(pairs, (distance, ticker1, ticker2))
            if len(pairs) > 50:
                _, t1, t2 = heapq.heappop(pairs)
    
    while pairs:
        distance, t1, t2 = heapq.heappop(pairs)
        result.append((t1, t2, distance))

    result.reverse()
    return result

potential_pairs = find_pairs_with_single_linkage(X)
potential_pairs
    
```

## Cointegration (Pairs validation)
We already find some pairs through clustering, now we need to validate these pairs from cointegration

```python
from statsmodels.tsa.stattools import coint

def validate_pairs(df, pairs):
    result = []
    for t1, t2, _ in pairs:
        score, pvalue, _ = coint(df[t1], df[t2])

        result.append({
                'ticker1': t1,
                'ticker2': t2,
                'coint_score': score,
                'p_value': pvalue,
                'is_cointegrated': pvalue < 0.06  
            })
        
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values('p_value')
    
    return result_df

validate_pairs(train, potential_pairs)

```

As we can see some of them are actually same company or same type of assets deliveried by different companies, after excluding such pairs, we still have about 5 pairs can be potential pairs targets, now let's pick one pair first (SUSC, VTC) to build some trading strategy

```python
import matplotlib.pyplot as plt

def plot_price(a, b, train):

    stock_a = train[a]
    stock_b = train[b]
    spread = stock_a - stock_b

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(stock_a, label=stock_a.name, color='blue', marker='o')
    ax1.plot(stock_b, label=stock_b.name, color='red', marker='o')
    ax1.set_title('Train Price history')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    data_length = len(stock_a)
    tick_positions = np.linspace(0, data_length-1, 5, dtype=int)
    tick_labels = ['2020', '2021', '2022', '2023', '2024']

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([]) 

    ax2.plot(spread, label='spread', color='green', marker='o')
    ax2.set_xlabel('time')
    ax2.set_ylabel('spread')
    ax2.axhline(y=0, color='gray', linestyle='--') 
    ax2.grid(True)
    ax2.legend()

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.show()

plot_price("LEVI", "MHK", train)
```

As we can see spread is jumping up and down around -2, which indicate good opportunity for pairs trading, let plot another one

```python
plot_price("SHO","XHR", train)
```

this above pair has some period with going up and down together period, which indicate we need to do some trend following strategy transition from pairs trading to increase profit

# Trading and Backtesting and forward testing

if the z score is too large, it means spread is too positive, so we long the l

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hybrid_trading_strategy(test_data, ticker1, ticker2, train_mean, train_std):
    data = test_data.copy()
    
    data['spread'] = data[ticker1] - data[ticker2]
    
    data['zscore'] = (data['spread'] - train_mean) / train_std
    
    data[f'{ticker1}_ret'] = data[ticker1].pct_change(5)
    data[f'{ticker2}_ret'] = data[ticker2].pct_change(5)
    
    data[f'{ticker1}_trend'] = data[ticker1].rolling(10).mean() > data[ticker1].rolling(25).mean()
    data[f'{ticker2}_trend'] = data[ticker2].rolling(10).mean() > data[ticker2].rolling(25).mean()
    
    data['same_trend'] = data[f'{ticker1}_trend'] == data[f'{ticker2}_trend']
    data['strong_trend'] = (abs(data[f'{ticker1}_ret']) > 0.05) | (abs(data[f'{ticker2}_ret']) > 0.05)
    
    data['position'] = 0
    data['strategy'] = 'none'
    
    for i in range(1, len(data)):
        if data['position'].iloc[i-1] == 0:
            if abs(data['zscore'].iloc[i]) > 1 and data['same_trend'].iloc[i] and data['strong_trend'].iloc[i]:
                if data[f'{ticker1}_trend'].iloc[i]:
                    data.loc[data.index[i], 'position'] = 1 if data[f'{ticker1}_ret'].iloc[i] > data[f'{ticker2}_ret'].iloc[i] else -1
                    data.loc[data.index[i], 'strategy'] = 'trend_following'
                else:
                    data.loc[data.index[i], 'position'] = -1 if data[f'{ticker1}_ret'].iloc[i] > data[f'{ticker2}_ret'].iloc[i] else 1
                    data.loc[data.index[i], 'strategy'] = 'trend_following'
            elif data['zscore'].iloc[i] > 0.75:
                data.loc[data.index[i], 'position'] = -1
                data.loc[data.index[i], 'strategy'] = 'mean_reversion'
            elif data['zscore'].iloc[i] < -0.75:
                data.loc[data.index[i], 'position'] = 1
                data.loc[data.index[i], 'strategy'] = 'mean_reversion'
        else:
            if data['strategy'].iloc[i-1] == 'mean_reversion':
                if (data['position'].iloc[i-1] == 1 and data['zscore'].iloc[i] > -0.5) or \
                   (data['position'].iloc[i-1] == -1 and data['zscore'].iloc[i] < 0.5) or \
                   abs(data['zscore'].iloc[i]) > 2.0:
                    data.loc[data.index[i], 'position'] = 0
                    data.loc[data.index[i], 'strategy'] = 'none'
                else:
                    data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
                    data.loc[data.index[i], 'strategy'] = data['strategy'].iloc[i-1]
            elif data['strategy'].iloc[i-1] == 'trend_following':
                if (data['position'].iloc[i-1] == 1 and data[f'{ticker1}_ret'].iloc[i] < 0) or \
                   (data['position'].iloc[i-1] == -1 and data[f'{ticker1}_ret'].iloc[i] > 0) or \
                   data['same_trend'].iloc[i] == False:
                    data.loc[data.index[i], 'position'] = 0
                    data.loc[data.index[i], 'strategy'] = 'none'
                else:
                    data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
                    data.loc[data.index[i], 'strategy'] = data['strategy'].iloc[i-1]
    
    data['returns'] = data['position'].shift(1) * (data[ticker1] - data[ticker1].shift(1)) - \
                     data['position'].shift(1) * (data[ticker2] - data[ticker2].shift(1))
    
    data['cumulative_returns'] = data['returns'].cumsum()
    
    return data

def visualize_strategy(results, ticker1, ticker2):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    ax1.plot(results[ticker1], label=ticker1)
    ax1.plot(results[ticker2], label=ticker2)
    ax1.set_title('Price Movement')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(results['zscore'], color='red')
    ax2.axhline(y=1.0, color='green', linestyle='--')
    ax2.axhline(y=-1.0, color='green', linestyle='--')
    ax2.axhline(y=0.4, color='black', linestyle='-.')
    ax2.axhline(y=-0.4, color='black', linestyle='-.')
    ax2.axhline(y=0, color='gray', linestyle='-')
    ax2.set_title('Z-Score')
    ax2.grid(True)
    
    mean_rev_signals = results[results['strategy'] == 'mean_reversion']
    trend_signals = results[results['strategy'] == 'trend_following']
    
    for idx, row in mean_rev_signals.iterrows():
        if row['position'] == 1:
            ax2.scatter(idx, row['zscore'], color='green', marker='^', s=100)
        elif row['position'] == -1:
            ax2.scatter(idx, row['zscore'], color='red', marker='v', s=100)
    
    for idx, row in trend_signals.iterrows():
        if row['position'] == 1:
            ax2.scatter(idx, row['zscore'], color='blue', marker='^', s=120, edgecolors='k')
        elif row['position'] == -1:
            ax2.scatter(idx, row['zscore'], color='purple', marker='v', s=120, edgecolors='k')
    
    ax3.plot(results['cumulative_returns'], color='blue')
    ax3.set_title('Cumulative Returns')
    ax3.grid(True)
    ax3.set_ylabel('time')
    
    plt.tight_layout()
    plt.show()
    
    print(f"total return: {results['cumulative_returns'].iloc[-1]:.2f}")
    print(f"MDD: {(results['cumulative_returns'] - results['cumulative_returns'].cummax()).min():.2f}")
    print(f"pairs trading times: {len(mean_rev_signals[mean_rev_signals['position'] != 0])}")
    print(f"trend following time: {len(trend_signals[trend_signals['position'] != 0])}")

def prepare_data_and_run(train_data, test_data, ticker1, ticker2):
    
    spread_train = train_data[ticker1] - train_data[ticker2]
    train_mean = spread_train.mean()
    train_std = spread_train.std()
    
    results = hybrid_trading_strategy(test_data, ticker1, ticker2, train_mean, train_std)
    visualize_strategy(results, ticker1, ticker2)
    
    return results
```

first let's see how it perform on training data

```python
train_trading = train[["LEVI", "MHK"]]
train_trading_data = prepare_data_and_run(train_trading, train_trading, "LEVI", "MHK")
```

it perform not bad and meet with out expectation, now let do test data

```python
train_trading = train[["LEVI", "MHK"]]
test_trading = test[["LEVI", "MHK"]]
test_trading_data = prepare_data_and_run(train_trading, test_trading, "LEVI", "MHK")
```

```python
test_trading_data
```

# Blotter and Lotter

```python


def create_blotter(data, ticker1, ticker2, commission_rate=0.0005):
    blotter = []
    
    for i in range(1, len(data)):
        prev_pos = data['position'].iloc[i-1]
        curr_pos = data['position'].iloc[i]
        
        if prev_pos != curr_pos:
            trade_date = data.index[i]
            ticker1_price = data[ticker1].iloc[i]
            ticker2_price = data[ticker2].iloc[i]
            
            if prev_pos == 1:
                blotter.append([trade_date, ticker1, 'SELL', ticker1_price, 1, ticker1_price * 1 * commission_rate, data['strategy'].iloc[i-1], data['zscore'].iloc[i]])
                blotter.append([trade_date, ticker2, 'BUY', ticker2_price, 1, ticker2_price * 1 * commission_rate, data['strategy'].iloc[i-1], data['zscore'].iloc[i]])
            elif prev_pos == -1:
                blotter.append([trade_date, ticker1, 'BUY', ticker1_price, 1, ticker1_price * 1 * commission_rate, data['strategy'].iloc[i-1], data['zscore'].iloc[i]])
                blotter.append([trade_date, ticker2, 'SELL', ticker2_price, 1, ticker2_price * 1 * commission_rate, data['strategy'].iloc[i-1], data['zscore'].iloc[i]])
            
            if curr_pos == 1:
                blotter.append([trade_date, ticker1, 'BUY', ticker1_price, 1, ticker1_price * 1 * commission_rate, data['strategy'].iloc[i], data['zscore'].iloc[i]])
                blotter.append([trade_date, ticker2, 'SELL', ticker2_price, 1, ticker2_price * 1 * commission_rate, data['strategy'].iloc[i], data['zscore'].iloc[i]])
            elif curr_pos == -1:
                blotter.append([trade_date, ticker1, 'SELL', ticker1_price, 1, ticker1_price * 1 * commission_rate, data['strategy'].iloc[i], data['zscore'].iloc[i]])
                blotter.append([trade_date, ticker2, 'BUY', ticker2_price, 1, ticker2_price * 1 * commission_rate, data['strategy'].iloc[i], data['zscore'].iloc[i]])
    
    blotter_df = pd.DataFrame(blotter, columns=['date', 'ticker', 'action', 'price', 'quantity', 'commission', 'strategy', 'zscore'])
    return blotter_df

def create_lotter(blotter, initial_capital=100000):
    lotter = []
    
    capital = initial_capital
    positions = {}
    
    for i, row in blotter.iterrows():
        date = row['date']
        ticker = row['ticker']
        action = row['action']
        price = row['price']
        quantity = row['quantity']
        commission = row['commission']
        
        if action == 'BUY':
            capital -= price * quantity + commission
            positions[ticker] = positions.get(ticker, 0) + quantity
        else:
            capital += price * quantity - commission
            positions[ticker] = positions.get(ticker, 0) - quantity
        
        portfolio_value = capital
        for pos_ticker, pos_qty in positions.items():
            if pos_ticker in blotter['ticker'].unique():
                last_price = blotter[blotter['ticker'] == pos_ticker].iloc[-1]['price']
                portfolio_value += pos_qty * last_price
        
        lotter.append([date, ticker, action, price, quantity, commission, capital, portfolio_value])
    
    lotter_df = pd.DataFrame(lotter, columns=['date', 'ticker', 'action', 'price', 'quantity', 'commission', 'cash', 'portfolio_value'])
    return lotter_df
```

```python
blotter = create_blotter(test_trading_data, "LEVI", "MHK")
blotter
```

```python
lotter = create_lotter(blotter)
lotter
```

let's test if we have trend following for another pair

```python
train_trading = train[["SHO","XHR"]]
test_trading = test[["SHO","XHR"]]
test_trading_data = prepare_data_and_run(train_trading, test_trading, "SHO","XHR")
```

as we can see there is no trend following here, which indicate future improvement, in the future I will make the long and short at the same time and based on every pair's different maginitude consider different entry time and exit time



