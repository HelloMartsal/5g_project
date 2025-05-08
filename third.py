import pandas as pd
import numpy as np

# Load aggregated JSON
input_file = 'aggregated_hourlyGridActivity_by_time.json'
df = pd.read_json(input_file, orient='records', lines=True)

# Convert startTime to datetime
df['startTime'] = pd.to_datetime(df['startTime'])
df.sort_values('startTime', inplace=True)

# === Feature Engineering ===

# Total traffic
df['totalTraffic'] = df[['smsIn', 'smsOut', 'callIn', 'callOut', 'internet']].sum(axis=1)

# Ratios
df['smsRatio'] = df['smsIn'] / (df['smsOut'] + 1)
df['callRatio'] = df['callIn'] / (df['callOut'] + 1)
df['smsTraffic'] = df['smsIn'] + df['smsOut']
df['callTraffic'] = df['callIn'] + df['callOut']
df['dataPerCall'] = df['internet'] / (df['callIn'] + df['callOut'] + 1)

# Time-based features
df['hour'] = df['startTime'].dt.hour
df['dayofweek'] = df['startTime'].dt.dayofweek
df['isWeekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Lag features (example)
df['internet_lag1'] = df['internet'].shift(1)
df['callIn_lag1'] = df['callIn'].shift(1)

# Rolling window features (example)
df['internet_roll_mean_3'] = df['internet'].shift(1).rolling(window=3).mean()
df['callOut_roll_std_6'] = df['callOut'].rolling(window=6).std()

# === Fourier Terms Generator ===
def add_fourier_terms(df, period=168, n_terms=4):
    t = np.arange(len(df))
    for i in range(1, n_terms + 1):
        df[f'fourier_sin_{i}_{period}'] = np.sin(2 * np.pi * i * t / period)
        df[f'fourier_cos_{i}_{period}'] = np.cos(2 * np.pi * i * t / period)
    return df

df = add_fourier_terms(df, period=24, n_terms=4)
df = add_fourier_terms(df, period=168, n_terms=4)

# Drop initial rows with NaNs from lag/rolling
df.dropna(inplace=True)

#reset index to include 'startTime' as a column
df.reset_index(drop=True, inplace=True)
# Convert to JSON format
# Save to JSON

# Save the engineered dataset
output_file = 'aggregated_hourlyGridActivity_with_features.json'
df.to_json(output_file, orient='records', lines=True, date_format='iso')
print(f"Feature-engineered data saved to {output_file}")
