import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Load preprocessed dataset
df = pd.read_json('aggregated_hourlyGridActivity_with_features.json', orient='records', lines=True)

# Convert startTime to datetime & set index
df['startTime'] = pd.to_datetime(df['startTime'])
df.set_index('startTime', inplace=True)

# Select the target variable (e.g., 'internet')
target_col = 'internet'

# Plot ACF & PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sm.graphics.tsa.plot_acf(df[target_col], lags=50, ax=axes[0])
sm.graphics.tsa.plot_pacf(df[target_col], lags=50, ax=axes[1])
axes[0].set_title(f'ACF of {target_col}')
axes[1].set_title(f'PACF of {target_col}')
plt.show()


results = adfuller(df[target_col])
print(f'ADF Statistic: {results[0]}')


import itertools
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Define the parameter grid
p = q = P = Q = range(0, 3)  # Testing values from 0 to 2
d = D = [1]  # Differencing
s = [24]  # Assuming daily seasonality for hourly data

# Generate all parameter combinations
param_grid = list(itertools.product(p, d, q, P, D, Q, s))

best_aic = float("inf")
best_params = None
best_model = None

# Iterate through each parameter combination
for params in param_grid:
    try:
        model = SARIMAX(df[target_col],
                        order=(params[0], params[1], params[2]),
                        seasonal_order=(params[3], params[4], params[5], params[6]),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        
        # Check if this model has the lowest AIC
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = params
            best_model = results
    except Exception as e:
        continue  # Skip failed models

print(f"Best SARIMAX Model: {best_params} with AIC: {best_aic}")


