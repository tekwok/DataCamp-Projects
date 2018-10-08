# coding: utf-8

# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce nice plots
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading in the data
stock_data = pd.read_csv('datasets/stock_data.csv', parse_dates=['Date'], index_col='Date').dropna()
benchmark_data = pd.read_csv('datasets/benchmark_data.csv', parse_dates=['Date'], index_col='Date').dropna() 


get_ipython().run_cell_magic('nose', '', '\ndef test_benchmark_data():\n    assert isinstance(benchmark_data, pd.core.frame.DataFrame), \\\n        \'Did you import the benchmark_data as a DataFrame?\'\n\ndef test_stock_data():\n    assert isinstance(stock_data, pd.core.frame.DataFrame), \\\n        \'Did you import the stock_data as a DataFrame?\'\n\ndef test_benchmark_index():\n    assert isinstance(benchmark_data.index, pd.core.indexes.datetimes.DatetimeIndex), \\\n        "Did you set the \'Date\' column as Index for the benchmark_data?"\n\ndef test_stock_index():\n    assert isinstance(stock_data.index, pd.core.indexes.datetimes.DatetimeIndex), \\\n        "Did you set the \'Date\' column as Index for the stock_data?"\n\ndef test_stock_data_shape():\n    assert stock_data.shape == (252, 2), \\\n        "Did you use .dropna() on the stock_data?"\n\ndef test_stock_benchmark_shape():\n    assert benchmark_data.shape == (252, 1), \\\n        "Did you use .dropna() on the benchmark_data?"\n    ')


# Display summary for stock_data
print('Stocks\n')
stock_data.info()
print(stock_data.head())

# Display summary for benchmark_data
print('\nBenchmarks\n')
benchmark_data.info()
print(benchmark_data.head())



get_ipython().run_cell_magic('nose', '', '\ndef test_nothing():\n    pass')



# visualize the stock_data
stock_data.plot(subplots=True, title='Stock Data');


# summarize the stock_data
stock_data.describe()


get_ipython().run_cell_magic('nose', '', '\ndef test_nothing():\n    pass')


# plot the benchmark_data
benchmark_data.plot(title='S&P 500')


# summarize the benchmark_data
benchmark_data.describe()


get_ipython().run_cell_magic('nose', '', '\ndef test_nothing():\n    pass')



# calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# plot the daily returns
stock_returns.plot()


# summarize the daily returns
stock_returns.describe()


get_ipython().run_cell_magic('nose', '', "\ndef test_stock_returns():\n    assert stock_returns.equals(stock_data.pct_change()), \\\n    'Did you use pct_change()?'")




# calculate daily benchmark_data returns
sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns
sp_returns.plot()

# summarize the daily returns
sp_returns.describe()



get_ipython().run_cell_magic('nose', '', "\ndef test_sp_returns():\n    assert sp_returns.equals(benchmark_data['S&P 500'].pct_change()), \\\n    'Did you use pct_change()?'")



# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plot the excess_returns
excess_returns.plot()

# summarize the excess_returns
excess_returns.describe()



get_ipython().run_cell_magic('nose', '', "\ndef test_excess_returns():\n    assert excess_returns.equals(stock_returns.sub(sp_returns, axis=0)), \\\n    'Did you use .sub()?'")



# calculate the mean of excess_returns 
avg_excess_return = excess_returns.mean()

# plot avg_excess_returns
avg_excess_return.plot.bar(title='Mean of the Return Difference')



get_ipython().run_cell_magic('nose', '', "\ndef test_avg_excess_return():\n    assert avg_excess_return.equals(excess_returns.mean()), \\\n    'Did you use .mean()?'")




# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations
sd_excess_return.plot(kind='bar', title='Standard Deviation of the Return Difference')




get_ipython().run_cell_magic('nose', '', "\ndef test_sd_excess():\n    assert sd_excess_return.equals(excess_returns.std()), \\\n    'Did you use .std() on excess_returns?'")



# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio
annual_sharpe_ratio.plot(kind='bar', title='Annualized Sharpe Ratio: Stocks vs S&P 500')




get_ipython().run_cell_magic('nose', '', "\ndef test_daily_sharpe():\n    assert daily_sharpe_ratio.equals(avg_excess_return.div(sd_excess_return)), \\\n    'Did you use .div() avg_excess_return and sd_excess_return?'\n    \ndef test_annual_factor():\n    assert annual_factor == np.sqrt(252), 'Did you apply np.sqrt() to, number_of_trading_days?'\n    \ndef test_annual_sharpe():\n    assert annual_sharpe_ratio.equals(daily_sharpe_ratio.mul(annual_factor)), 'Did you use .mul() with daily_sharpe_ratio and annual_factor?'")



# Uncomment your choice.
buy_amazon = True
# buy_facebook = True



get_ipython().run_cell_magic('nose', '', "\ndef test_decision():\n    assert 'buy_amazon' in globals() and buy_amazon == True, \\\n    'Which stock has the higher Sharpe Ratio'")

