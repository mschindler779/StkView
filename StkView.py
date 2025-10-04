#"/usr/bin/python
# -*- coding: utf-8 -*-

"""StkView.py: Python script for data analysis of stock prices"""

__author__ = "Markus Schindler"
__copyright__ = "Copyright 2025"

__license__ = "Unlicense"
__version__ = "0.1.0"
__maintainer__ = "Markus Schindler"
__email__ = "schindlerdrmarkus@gmail.com"
__status__ = "Education"

# Built-in / Generic Imports
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import seaborn as sns

#################
# Load the Data #
#################

df_original = pd.read_csv('data/Stocks_08-21-2025_5y.csv', sep = ';', low_memory = False)

####################
# Data Preparation #
####################

# Create a copy
df = df_original.copy()

# Change Date Format
df['Date'] = pd.to_datetime(df['Date'], dayfirst = 'True', yearfirst = 'False')

# Format numbers and convert strings to float values
df.replace({r'\.': r''}, regex = True, inplace = True)
df.replace({r'\,': r'.'}, regex = True, inplace = True)
df.set_index('Date', inplace = True)
df = df.astype(float)

# Some numbers are not correclty imported and need to be adjusted
df[df > 2000] = df[df > 2000] / 1000

# check for missing values
missing_values = df.isnull().sum()

# check if the data has sufficient rows for time-series analysis
sufficient_rows = df.shape[0] >= 20  # Minimum rows needed for rolling / moving averages

# preparing a summary of the checks
data_preparation_status = {
    "Missing Values in Columns": missing_values[missing_values > 0].to_dict(),
    "Sufficient Rows for Time-Series Analysis": sufficient_rows
}
print(data_preparation_status)

# drop the several column since it contains not all required values
df = df.drop(columns=['Novartis (EUR)', 'Palantir (EUR)', 'Porsche (Vz) (EUR)', 'Total Energy (EUR)'])

# Drop rows with NaN in the columns
df.dropna(subset = ['Evonik Industries (EUR)'], inplace = True)

# sort the dataset by date to ensure proper time-series order
df = df.sort_values(by='Date')

#########################
# Some Basic Statistics #
#########################

# calculate descriptive statistics
descriptive_stats = df.describe().T  # Transpose for better readability
descriptive_stats = descriptive_stats[['mean', 'std', 'min', 'max']]
descriptive_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
print(descriptive_stats)

######################
# Portfolio Creation #
######################

# Stock selection
data_portfolio = df[['Alphabet A (EUR)', 'Advanced Micro Devices (EUR)', 'Meta (EUR)', 'Amazon (EUR)', 'Cisco (EUR)']]
weights_portfolio = [0.3, 0.1, 0.3, 0.2, 0.1]

# Create function to calculate portfolio returns
def portfolio_return(data_portfolio, weights_portfolio):
    if data_portfolio.shape[1] != len(weights_portfolio):
        print('Please check correct number of inputs!')
    else:
        daily_changes = data_portfolio.pct_change().dropna()
        return (daily_changes * weights_portfolio).sum(axis = 1)

portfolio = portfolio_return(data_portfolio, weights_portfolio)

###############################################
# Calculation of Volatility and Value of Risk #
###############################################

# Calculate standard deviation (volatility)
daily_return = data_portfolio.pct_change().dropna()
volatility = daily_return.std()

# Calculate Value at Risk (95% confidence level)
confidence_level = 0.95
alpha = 1 - confidence_level
VaR = daily_return.quantile(alpha)

# Display risk metrics
risk_metrics = pd.DataFrame({'Volatility (Std Dev)': volatility, 'Value at Risk (VaR)': VaR})
print(risk_metrics)

####################################
# Creation of a Correlation Matrix #
####################################

# Generate the correlation matrix for daily_return
corr_daily_return = daily_return.corr()

sns.set_theme(style = 'dark')
mask = np.triu(np.ones_like(corr_daily_return, dtype = bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(170, 130, as_cmap = True)

sns.heatmap(corr_daily_return, mask = mask, cmap = cmap, vmax = 1, center = 0,
            square=True, linewidths = 0.5, cbar_kws = {"shrink": .5})

# Save the heatmap figure to a PNG file
f.savefig('Heatmap.png', dpi=300, bbox_inches='tight')

# Calculation of moving averages for one stock
df['Advanced Micro Devices (EUR)_5d_MA'] = df['Advanced Micro Devices (EUR)'].rolling(window = 5).mean()
df['Advanced Micro Devices (EUR)_20d_MA'] = df['Advanced Micro Devices (EUR)'].rolling(window = 20).mean()

# Plotting the generated data
plt.figure(figsize = (8, 6), dpi = 100)
plt.title('Share prices for AMD in recent years', weight = 'bold')
plt.plot(df.index, df['Advanced Micro Devices (EUR)'], linewidth = 1, color = 'maroon', label = 'AMD Price')
plt.plot(df.index, df['Advanced Micro Devices (EUR)_5d_MA'], linewidth = 1, color = 'seagreen', label = 'AMD Price 5 days MA')
plt.plot(df.index, df['Advanced Micro Devices (EUR)_20d_MA'], linewidth = 1, color = 'dodgerblue', label = 'AMD Price 20 days MA')
plt.xlabel('Date')
plt.ylabel('Stock Price in (EUR)')
plt.grid()
plt.legend(loc = 'upper right')
plt.xlim(20100, 20300) 
plt.savefig('AMD-MA.png')

##########################################
# Calculation of Relative Strength Index #
##########################################

# Definition of the RSI function
def calculation_rsi(price, window = 14):
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window = window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
    relative_strength = gain / loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi

df['Advanced Micro Devices (EUR)_rsi'] = calculation_rsi(df['Advanced Micro Devices (EUR)'])
plt.figure(figsize = (8, 6), dpi = 100)
plt.title('RSI for AMD in recent years', weight = 'bold')
plt.plot(df.index, df['Advanced Micro Devices (EUR)_rsi'], linewidth = 1, color = 'maroon', label = 'AMD RSI')
plt.axhline(y = 30, linestyle = 'dashed', linewidth = 1, color = 'seagreen', label = 'Oversold')
plt.axhline(y = 70, linestyle = 'dashed', linewidth = 1, color = 'dodgerblue', label = 'Overbought')
plt.grid()
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend(loc = 'lower right')
plt.xlim(20100, 20300) 
plt.ylim(0, 100)
plt.savefig('AMD-RSI.png')

################################
# Calculation of Sharpe Ratios #
################################

mean_return = daily_return.mean()
volatility = daily_return.std()

# Assumption of a risk-free rate
risk_free_rate = 0.04 / 250

# Calculate Sharpe ratios
sharpe_ratios = (mean_return - risk_free_rate) / volatility

table_data = pd.DataFrame({
    'Stock': sharpe_ratios.index,
    'Sharpe Ratio': sharpe_ratios.values
})
# Bar chart of Sharpe ratios
plt.figure(figsize=(8, 6))
bars = plt.bar(sharpe_ratios.index, sharpe_ratios.values, color='mediumseagreen')
plt.title('Sharpe Ratios for Selected Stocks', fontsize=14, weight = 'bold')
plt.xlabel('Stock', fontsize=12)
plt.ylabel('Sharpe Ratio', fontsize=12)
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('Sharpe-Ratios.png')

##############################
# Utilization of Monte Carlo #
##############################

#Prediction for AMD share price using Monte Carlo 
number_runs = 1000
number_days = 250
recent_price = df['Advanced Micro Devices (EUR)'].iloc[-1]
simulated_prices = np.zeros((number_runs, number_days))
volatility = df['Advanced Micro Devices (EUR)'].pct_change().std()

for i in range(number_runs):
    simulated_prices[i, 0] = recent_price
    for j in range(1, number_days):
        simulated_prices[i, j] = simulated_prices[i, j-1] * np.exp(
            np.random.normal(0, volatility)
        )

# Create list for processed days
x = np.arange(0, number_days, 1)

plt.figure(figsize = (8, 6), dpi = 100)
plt.title('Monte Carlo Simulation for AMD', weight = 'bold')
plt.xlabel('Days')
plt.ylabel('Stock Price AMD (EUR)')
plt.grid()
plt.xlim(0, 250) 
plt.ylim(0, 700)
for itervar in range(number_runs):
    plt.plot(x, simulated_prices[itervar], color = 'dodgerblue', linewidth = 0.1, alpha = 0.3)
plt.savefig('Monte-Carlo.png')

############################################
# Implementation of Long Short-Term Memory #
############################################

timeseries = df[['Advanced Micro Devices (EUR)']].values.astype('float32')

# train-test split for time series
train_size = int(len(timeseries) * 0.80)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

# Transform a time series into predictive dataset
# A NumPy array of time series with lookback (Size of windows prediction)
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)

lookback = 4
X_train, y_train = create_dataset(train, lookback = lookback)
X_test, y_test = create_dataset(test, lookback = lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
if torch.cuda.is_available():
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    X_test = X_test.cuda()
    y_test = y_test.cuda()

# LSTM Model
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 50, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = LSTM_Model()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle = True, batch_size = 8)

n_epochs = 1700
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        if torch.cuda.is_available():
            y_pred = y_pred.cuda()
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = sqrt(loss_fn(y_pred, y_test))
    print('Epoch %d: train RMSE %4f, test RMSE %.4f' % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # Shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred_train = model(X_train)
    y_pred_train = y_pred_train[:, -1, :]
    for itervar in range(lookback, train_size):
        train_plot[itervar] = float(y_pred_train[itervar - lookback])
    test_plot = np.ones_like(timeseries) * np.nan
    y_pred_test = model(X_test)
    y_pred_test = y_pred_test[:, -1, :]
    for itervar in range(lookback, test_size):
        test_plot[itervar + train_size] = float(y_pred_test[itervar - lookback])

# Plot
plt.figure(figsize = (8, 6), dpi = 100)
plt.title('LSTM Model for AMD', weight = 'bold')
plt.plot(timeseries, linewidth = 0, marker = 'o', markersize = 3, color = 'dodgerblue', label = 'Data')
plt.plot(train_plot, linewidth = 1, color = 'maroon', label = 'Trained Model')
plt.plot(test_plot, linewidth = 1, color = 'seagreen', label = 'Prediction')
plt.grid()
plt.xlabel('Days')
plt.ylabel('AMD Stock Price (EUR)')
plt.legend(loc = 'upper left')
plt.xlim(0, 1400)
plt.ylim(0, 200)
plt.savefig('LSTM.png')
