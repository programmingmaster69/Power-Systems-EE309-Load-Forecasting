import pandas as pd

# =========================
# LOAD FILES
# =========================
load_df = pd.read_csv('data/Complete_datasheet.csv')
weather_df = pd.read_csv('data/open-meteo-51.51N0.13W16m.csv')

# =========================
# DATETIME PARSING
# =========================
load_df['Datetime'] = pd.to_datetime(load_df['Date Time'])
weather_df['Datetime'] = pd.to_datetime(weather_df['time'])

# =========================
# CLEAN + SELECT
# =========================
load_df = load_df[['Datetime', 'MW']].rename(columns={'MW': 'Load'})

weather_df = weather_df[['Datetime',
                         'temperature_2m',
                         'relative_humidity_2m',
                         'windspeed_10m']]

# =========================
# SORT
# =========================
load_df = load_df.sort_values('Datetime')
weather_df = weather_df.sort_values('Datetime')

# =========================
# ALIGN WEATHER → LOAD TIMESTAMPS
# =========================
weather_df = weather_df.set_index('Datetime')

# Reindex weather to match load timestamps
weather_aligned = weather_df.reindex(load_df['Datetime'], method='nearest')

# Fill gaps
weather_aligned = weather_aligned.interpolate(method='time').ffill().bfill()

# =========================
# MERGE
# =========================
df = pd.concat([load_df.reset_index(drop=True),
                weather_aligned.reset_index(drop=True)], axis=1)

# =========================
# ADD TIME FEATURES
# =========================
df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month

# =========================
# SAVE
# =========================
df.to_csv('data/final_lstm_dataset.csv', index=False)

# =========================
# VERIFY
# =========================
print("Final rows:", len(df))
print("Expected rows (load):", len(load_df))
print("Range:", df['Datetime'].min(), "→", df['Datetime'].max())
print(df.head())