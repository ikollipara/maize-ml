"""
process_weather_data.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

Script for processing weather data for Maize-ML.
"""

import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python process_weather_data.py <input_file> <output_dir>")
    exit(1)

if len(sys.argv) == 2 and sys.argv[1] in ("--help", "-h"):
    print("Usage: python process_weather_data.py <input_file> <output_dir>")
    print("Processes the weather data for Maize-ML.")
    print("Arguments:")
    print("\t<input_file>: The file containing the weather data.")
    print("\t<output_dir>: The directory to save the processed data to.")
    exit(0)

weather_df = pd.read_csv(sys.argv[1])

# Drop rows with missing data
weather_df = weather_df.dropna()

# Add period column
weather_df['period'] = weather_df.apply(
    lambda x: pd.Period(freq='D', year=x['year'], day=x['yday']),
    axis=1
)

# Save the processed data
for env in weather_df['sourceEnvironment'].unique():
    env_df = weather_df[weather_df['sourceEnvironment'] == env].drop(columns=['sourceEnvironment']).sort_values(by=['period'])
    env_df.to_csv(f'{sys.argv[2]}/{env}_weather.csv', index=False)
    print(f'Saved {env}_weather.csv')
