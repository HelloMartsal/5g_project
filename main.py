import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for the progress bar

# Set style for the notebook
sns.set()

# Initialize an empty dataframe to append daily and hourly resampled data
dailyGridActivity = pd.DataFrame()
hourlyGridActivity = pd.DataFrame()

# Create a list of 62 data file names placed under directory "./Data/" with extension .txt
filenames = glob.glob("../data/*.txt")

# Set the column names for the data read
col_list = ['gridID', 'timeInterval', 'countryCode', 'smsIn', 'smsOut', 'callIn', 'callOut', 'internet']

def process_file(file):
    chunks = pd.read_csv(file, sep='\t', header=None, names=col_list, parse_dates=True, chunksize=100000)
    daily_chunks = []
    hourly_chunks = []
    
    for chunk in chunks:
        # Convert timeInterval column to UTC and then to Milan's local timezone
        chunk['startTime'] = pd.to_datetime(chunk.timeInterval, unit='ms', utc=True).dt.tz_convert('CET').dt.tz_localize(None)
        
        # Drop timeInterval & countryCode columns
        chunk.drop(columns=['timeInterval', 'countryCode'], inplace=True)
        
        # Resample to daily and hourly aggregation
        daily_chunk = chunk.groupby(['gridID', pd.Grouper(key='startTime', freq='D')]).sum()
        hourly_chunk = chunk.groupby(['gridID', pd.Grouper(key='startTime', freq='h')]).sum()
        
        daily_chunks.append(daily_chunk)
        hourly_chunks.append(hourly_chunk)
    
    # Combine all chunks
    read_data_daily = pd.concat(daily_chunks)
    read_data_hourly = pd.concat(hourly_chunks)
    
    return read_data_daily, read_data_hourly

# Process files sequentially with a progress bar
results = []
for file in tqdm(filenames, desc="Processing files"):
    results.append(process_file(file))

# Combine results from all files
for daily, hourly in results:
    dailyGridActivity = pd.concat([dailyGridActivity, daily]).groupby(['gridID', 'startTime']).sum()
    hourlyGridActivity = pd.concat([hourlyGridActivity, hourly]).groupby(['gridID', 'startTime']).sum()

# Save the activities into separate JSON files
# Reset the index to include 'startTime' as a column
dailyGridActivity.reset_index().to_json('dailyGridActivity.json', orient='records', lines=True, date_format='iso')

# Save hourlyGridActivity in chunks
chunk_size = 10000  # Number of rows per chunk
with open('hourlyGridActivity.json', 'w') as f:
    for i in range(0, len(hourlyGridActivity), chunk_size):
        chunk = hourlyGridActivity.iloc[i:i + chunk_size].reset_index()  # Reset index to include 'startTime'
        f.write(chunk.to_json(orient='records', lines=True, date_format='iso'))
        f.write('\n')  # Add a newline between chunks

# Reset the index for totalGridActivity and save it
totalGridActivity = dailyGridActivity.groupby('gridID').sum()
totalGridActivity.reset_index().to_json('totalGridActivity.json', orient='records', lines=True, date_format='iso')

print("Activities saved to JSON files:")
print("dailyGridActivity.json")
print("hourlyGridActivity.json")
print("totalGridActivity.json")

import ijson

def read_json_in_chunks(file_path, num_rows=20):
    rows = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            rows.append(pd.read_json(line, orient='records', lines=True))
    return pd.concat(rows, ignore_index=True)


# Access the JSON files and print the first 20 entries
print("\nFirst 20 entries from dailyGridActivity.json:")
daily_data = read_json_in_chunks('dailyGridActivity.json')
print(daily_data)

print("\nFirst 20 entries from hourlyGridActivity.json:")
hourly_data = read_json_in_chunks('hourlyGridActivity.json')
print(hourly_data)

print("\nFirst 20 entries from totalGridActivity.json:")
total_data = read_json_in_chunks('totalGridActivity.json')
print(total_data)

