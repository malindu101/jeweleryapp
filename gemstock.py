import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import snowflake.connector

# --- Snowflake connection parameters ---
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",         # Same DB
        schema="PUBLIC"

# --- Connect to Snowflake ---
conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    warehouse=warehouse,
    database=database,
    schema=schema
)

# --- Query your Snowflake table ---
query = "SELECT GEM_COLOR, CREATED_AT FROM STOCK"
df = pd.read_sql(query, conn)
conn.close()

# --- Process datetime ---
df['CREATED_AT'] = pd.to_datetime(df['CREATED_AT'], errors='coerce')
df['year_month'] = df['CREATED_AT'].dt.to_period('M')

# --- Group and reshape data ---
monthly_counts = df.groupby(['year_month', 'GEM_COLOR']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.to_timestamp()
monthly_counts = monthly_counts.asfreq('MS')

# --- Prepare training data (up to 2025) ---
train_data = monthly_counts[monthly_counts.index.year <= 2025]
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year

X = train_data[['year', 'month']]
y = train_data.drop(columns=['year', 'month'])

# --- Train the model ---
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
model.fit(X, y)

# --- Predict for 2026 ---
future_dates = pd.date_range(start="2026-01-01", end="2026-12-01", freq='MS')
future_df = pd.DataFrame({
    'year': future_dates.year,
    'month': future_dates.month
})
predictions = model.predict(future_df)
prediction_df = pd.DataFrame(predictions, index=future_dates, columns=y.columns)

# --- Plot predictions ---
prediction_df.plot(figsize=(12, 6), title="Predicted Monthly Gem Usage in 2026")
plt.xlabel("Month")
plt.ylabel("Predicted Gem Count")
plt.grid(True)
plt.tight_layout()
plt.show()
