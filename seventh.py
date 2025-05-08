import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load your data (replace this with your actual data loading method)
data  = pd.read_json('aggregated_hourlyGridActivity_with_features.json', lines=True) # Replace with the path to your CSV file

# Assuming you want to forecast the 'internet' column, so we extract it
y = data['internet']

# Step 1: Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Step 2: Split the data into training and test sets (e.g., 80% for training)
train_size = int(len(y_scaled) * 0.8)
train, test = y_scaled[:train_size], y_scaled[train_size:]

# If you have external variables (X), you can also normalize them
# X_scaled = scaler.fit_transform(data[['smsIn', 'callIn', 'smsOut', 'callOut']])

# Step 3: Apply auto_arima to find the best parameters for the ARIMA model
seasonal_period = 24  # Seasonal period for daily seasonality (adjust as needed)

# Running the auto_arima to automatically search for the best (p, d, q) and (P, D, Q, s)
stepwise_model = auto_arima(train, 
                            seasonal=True,  # Enable seasonal ARIMA
                            m=seasonal_period,  # Seasonal period (e.g., 24 for daily seasonality)
                            start_p=0, start_q=0, 
                            max_p=5, max_q=5, 
                            d=1,  # If you want auto-differencing, or manually differenced data
                            trace=True,  # Show progress during fitting
                            error_action='ignore',  # Ignore errors in case of convergence issues
                            suppress_warnings=True, 
                            stepwise=True)

# Print the summary of the best model found by auto_arima
print(stepwise_model.summary())

# Step 4: Forecast the next n steps (e.g., forecast the next 10 steps)
n_periods = len(test)  # Forecast the same length as the test set
forecast, conf_int = stepwise_model.predict(n_periods=n_periods, 
                                            return_conf_int=True)

# Step 5: Reverse the scaling to get the forecast in the original scale
forecast_original = scaler.inverse_transform(forecast.reshape(-1, 1))
test_original = scaler.inverse_transform(test.reshape(-1, 1))

# Step 6: Evaluate the model's performance (RMSE, MAPE)
rmse = np.sqrt(mean_squared_error(test_original, forecast_original))
mape = np.mean(np.abs((test_original - forecast_original) / test_original)) * 100

print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')

# Step 7: Plot the results - actual vs forecasted
plt.figure(figsize=(10, 6))
plt.plot(data.index[train_size:], test_original, color='blue', label='Actual')
plt.plot(data.index[train_size:], forecast_original, color='red', linestyle='dashed', label='Forecasted')
plt.fill_between(data.index[train_size:], conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.title('Forecast vs Actuals')
plt.legend()
plt.show()

