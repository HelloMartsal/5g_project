import pandas as pd
import orjson
from tqdm import tqdm


def read_and_filter_json_aggregated(file_path, grid_ids):
    results = []
    line_count = 0

    # Count lines for progress bar
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, 'r') as f:
        progress_bar = tqdm(f, total=total_lines, desc="Processing lines", unit="line")

        for line in progress_bar:
            line_count += 1
            try:
                record = orjson.loads(line)

                if isinstance(record, dict):
                    if record.get('gridID') in grid_ids:
                        results.append(record)
                elif isinstance(record, list):
                    filtered = [r for r in record if r.get('gridID') in grid_ids]
                    results.extend(filtered)

            except orjson.JSONDecodeError as e:
                progress_bar.set_postfix_str(f"JSON error: {str(e)[:20]}...")

    df = pd.DataFrame(results)
    print(f"\nProcessing complete. Scanned {line_count:,} lines.")
    print(f"Found {len(df):,} matching records.")

    return df


# === CONFIGURATION ===
file_path = 'hourlyGridActivity.json'
grid_ids_to_filter = [4459, 4456, 5060, 5646]
output_file = 'aggregated_hourlyGridActivity_by_time.json'
time_column = 'startTime'

# === PROCESS & AGGREGATE ===
filtered_data = read_and_filter_json_aggregated(file_path, grid_ids_to_filter)

if not filtered_data.empty:
    if time_column not in filtered_data.columns:
        raise ValueError(f"Time column '{time_column}' not found in data.")

    # Convert numeric columns to float (in case any are string)
    numeric_cols = ['smsIn', 'smsOut', 'callIn', 'callOut', 'internet']
    filtered_data[numeric_cols] = filtered_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Group by startTime and sum values
    aggregated = filtered_data.groupby(time_column)[numeric_cols].sum().reset_index()

    # Save to JSON
    aggregated.to_json(output_file, orient='records', lines=True)
    print(f"\nAggregated data saved to {output_file}")
    print(aggregated.head())
else:
    print("No matching records found.")
