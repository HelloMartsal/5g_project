import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load preprocessed dataset
df = pd.read_json('aggregated_hourlyGridActivity_with_features.json', orient='records', lines=True)

# Convert startTime to datetime & set index
df['startTime'] = pd.to_datetime(df['startTime'])
df.set_index('startTime', inplace=True)

# Select the target variable (e.g., 'internet')
target_col = 'internet'  # Change this to your target variable
y_raw = df[target_col]

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#adfuller test for stationarity
result = adfuller(y_raw.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is non-stationary.")

    

# 1. **Non-seasonal differencing (for ACF/PACF analysis)**
y_diff = y_raw.diff().dropna()  # Apply first-order differencing

# 2. **Seasonal differencing (for seasonal ACF/PACF analysis)**
seasonal_period = 24  # Assuming daily seasonality for hourly data (adjust as needed)
y_seasonal_diff = y_raw.diff(seasonal_period).dropna()

# Create the plots for both non-seasonal and seasonal ACF/PACF

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Non-seasonal ACF/PACF
plot_acf(y_diff, lags=40, ax=axes[0, 0])
axes[0, 0].set_title('Non-Seasonal ACF')

plot_pacf(y_diff, lags=40, ax=axes[0, 1])
axes[0, 1].set_title('Non-Seasonal PACF')

# Seasonal ACF/PACF
plot_acf(y_seasonal_diff, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('Seasonal ACF')

plot_pacf(y_seasonal_diff, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('Seasonal PACF')

plt.tight_layout()
plt.show()



