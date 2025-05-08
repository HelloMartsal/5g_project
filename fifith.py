import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import json
import pandas as pd

file_path = 'aggregated_hourlyGridActivity_with_features.json'

# Load JSON lines safely
data = []
with open(file_path, 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print("⚠️ Skipping malformed line:", line.strip())

df = pd.DataFrame(data)



# === 1. Load and preprocess data ===
df = pd.read_json('aggregated_hourlyGridActivity_with_features.json', orient='records', lines=True)
df['startTime'] = pd.to_datetime(df['startTime'])
df.set_index('startTime', inplace=True)

# Set frequency and fill gaps
df = df.asfreq('h')
df = df.ffill()

print(f"Total rows in dataset: {len(df)}")


# === 2. Define target ===
target_col = 'internet'

# === 3. Train/test split (last 7 days for testing) ===
train = df.iloc[:-24*7]
test = df.iloc[-24*7:]

# === 4. Fit SARIMA model (no exogenous features) ===
model = SARIMAX(train[target_col],
                order=(0, 1, 2),
                seasonal_order=(2, 1, 2, 24),
                enforce_stationarity=False,
                enforce_invertibility=False)

try:
    results = model.fit(disp=False)
except Exception as e:
    print("Model fitting failed:", e)
    exit()

# === 5. Forecast and score ===
forecast = results.predict(start=test.index[0], end=test.index[-1])

# Clip forecast to test set length
forecast = forecast[:len(test)]

rmse = np.sqrt(mean_squared_error(test[target_col], forecast))
mape = mean_absolute_percentage_error(test[target_col], forecast)

print(f'✅ RMSE: {rmse:.2f}')
print(f'✅ MAPE: {mape*100:.2f}%')

# === 6. Plot results ===
plt.figure(figsize=(14, 6))
plt.plot(train.index[-48:], train[target_col].iloc[-48:], label='Train (last 2 days)', alpha=0.6)
plt.plot(test.index, test[target_col], label='Actual', color='black')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Actual vs Forecast (No Exogenous Features)')
plt.xlabel('Time')
plt.ylabel('Internet Usage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
