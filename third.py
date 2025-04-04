import pandas as pd

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
df['dataPerCall'] = df['internet'] / (df['callIn'] + df['callOut'] + 1)

# Time-based features
df['hour'] = df['startTime'].dt.hour
df['dayofweek'] = df['startTime'].dt.dayofweek
df['isWeekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Lag features (example)
df['internet_lag1'] = df['internet'].shift(1)
df['callIn_lag1'] = df['callIn'].shift(1)

# Rolling window features (example)
df['internet_roll_mean_3'] = df['internet'].rolling(window=3).mean()
df['callOut_roll_std_6'] = df['callOut'].rolling(window=6).std()

# Drop initial rows with NaNs from lag/rolling
df.dropna(inplace=True)

# Save the engineered dataset
output_file = 'aggregated_hourlyGridActivity_with_features.json'
df.to_json(output_file, orient='records', lines=True)
print(f"Feature-engineered data saved to {output_file}")
