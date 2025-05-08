import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_json('aggregated_hourlyGridActivity_with_features.json', lines=True)
df['startTime'] = pd.to_datetime(df['startTime'])
df.set_index('startTime', inplace=True)

# 2. Define target and exogenous variables
# Set frequency and fill gaps
df = df.asfreq('H')
target = 'callTraffic'  # Change this to your target variable
df['callTraffic_lag24'] = df['callTraffic'].shift(24)
df['smsTraffic_lag24'] = df['smsTraffic'].shift(24)
df['smsTraffic_lag168'] = df['smsTraffic'].shift(168)
df['internet_lag24'] = df['internet'].shift(24)
df['internet_lag168'] = df['internet'].shift(168)
df['callTraffic_lag168'] = df['callTraffic'].shift(168)
exog_vars = ['fourier_sin_1_24', 'fourier_cos_1_24',
            'fourier_sin_2_24', 'fourier_cos_2_24',
            'fourier_sin_1_168', 'fourier_cos_1_168',
            'fourier_sin_2_168', 'fourier_cos_2_168',
            'isWeekend', 'callTraffic_lag24', 'callTraffic_lag168']

from statsmodels.tsa.stattools import adfuller
result = adfuller(df[target])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

y_raw = np.log1p(df[target].copy()) 
X_raw = df[exog_vars].copy()

# 3. Scale the data
scaler_y = StandardScaler()
scaler_X = StandardScaler()

y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).flatten()
X_scaled = pd.DataFrame(scaler_X.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)

# 4. Rolling forecast parameters
w = 96
h = 1
start_index = len(df) // 2
end_index = len(df) - h - 100

predictions = []
true_values = []

for i in tqdm(range(start_index, end_index), desc="Rolling Forecast"):
    # Rolling train/test split
    y_train = y_scaled[i - w:i]
    X_train = X_scaled.iloc[i - w:i]
    
    y_test = y_scaled[i + h - 1]   # actual future value (still scaled)
    X_test = X_scaled.iloc[i + h - 1].values.reshape(1, -1)

    # Fit SARIMAX model
    model = SARIMAX(y_train, exog=X_train,
                    order=(2, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    try:
        model_fit = model.fit(disp=False)
        y_pred = model_fit.forecast(steps=1, exog=X_test)
        predictions.append(y_pred.item())
        true_values.append(y_test)
    except:
        predictions.append(np.nan)
        true_values.append(np.nan)

# First, inverse scale
predictions_log = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
true_values_log = scaler_y.inverse_transform(np.array(true_values).reshape(-1, 1)).flatten()

# Then, reverse log1p → expm1
predictions = np.expm1(predictions_log)
true_values = np.expm1(true_values_log)

# 6. Evaluation
rmse = mean_squared_error(true_values, predictions, squared=False)
mape = mean_absolute_percentage_error(true_values, predictions)
#symmetric mean absolute percentage error (SMAPE)
smape = np.mean(np.abs((true_values - predictions) / ((np.abs(true_values) + np.abs(predictions)) / 2))) * 100


print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"SMAPE: {smape:.2f}")

# 6.5: Bucket-based MAPE
import pandas as pd
import numpy as np

def bucket_mape_analysis(y_true, y_pred, buckets=[0, 100, 1000, np.inf]):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['bucket'] = pd.cut(df['true'], bins=buckets, labels=['low', 'medium', 'high'])

    results = {}
    for bucket in df['bucket'].unique():
        subset = df[df['bucket'] == bucket]
        if len(subset) == 0:
            results[str(bucket)] = 'No data'
            continue
        mape = np.mean(np.abs((subset['true'] - subset['pred']) / subset['true'])) * 100
        results[str(bucket)] = round(mape, 2)

    return results

bucket_results = bucket_mape_analysis(true_values, predictions)
print("\nMAPE by Traffic Bucket:")
for bucket, score in bucket_results.items():
    print(f"  {bucket.capitalize()} Traffic: {score}%")

    

# 7. Plot results
plt.figure(figsize=(12, 6))
plt.plot(true_values, label='True callTraffic Traffic')
plt.plot(predictions, label='Predicted Traffic', linestyle='--')
plt.title("Rolling Forecast: callTraffic Traffic")
plt.xlabel("Time Steps")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
