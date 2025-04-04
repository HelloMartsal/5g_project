import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import json

# === Load data ===
file_path = 'aggregated_hourlyGridActivity_with_features.json'
data = []
with open(file_path, 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print("⚠️ Skipping malformed line:", line.strip())
df = pd.DataFrame(data)

# === Setup ===
target_col = 'internet'
total_rows = len(df)
train_end = int(total_rows * 0.7)

train_df = df.iloc[:train_end]
test_df = df.iloc[train_end:]

rolling_window = 24 * 7  # 7 days
forecast_horizon = 24  # Forecast 7 days ahead

rolling_forecasts = []
actuals = []

# === Rolling forecast loop ===
for start in range(0, len(test_df) - forecast_horizon + 1, forecast_horizon):
    train_data = pd.concat([train_df, test_df.iloc[:start]])  # up to current test window
    test_data = test_df.iloc[start:start + forecast_horizon]

    model = SARIMAX(train_data[target_col],
                    order=(0, 1, 2),
                    seasonal_order=(2, 1, 2, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    try:
        results = model.fit(disp=False)
        forecast = results.predict(start=len(train_data), end=len(train_data) + forecast_horizon - 1)
        rolling_forecasts.extend(forecast)
        actuals.extend(test_data[target_col].values)
    except Exception as e:
        print(f"Model failed at window starting {start}: {e}")
        continue

# === Evaluate ===
rmse = np.sqrt(mean_squared_error(actuals, rolling_forecasts))
mape = mean_absolute_percentage_error(actuals, rolling_forecasts)

print(f'✅ Rolling RMSE: {rmse:.2f}')
print(f'✅ Rolling MAPE: {mape * 100:.2f}%')

# === Plot results ===
plt.figure(figsize=(14, 6))
plt.plot(range(len(actuals)), actuals, label='Actual', color='black')
plt.plot(range(len(rolling_forecasts)), rolling_forecasts, label='Forecast', color='red')
plt.title('Rolling Forecast (7-day Window)')
plt.xlabel('Hour Index')
plt.ylabel('Internet Usage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
