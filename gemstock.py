import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector

st.set_page_config(layout="wide")
st.title("🔮 2026 Gemstone Usage Forecast")

# ✅ Fetch data from Snowflake
@st.cache_data
def get_gem_data():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT * FROM GEM_USAGE"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

# ✅ Load data
try:
    df = get_gem_data()
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# ✅ Preprocess
df['year_month'] = df['created_at'].dt.to_period('M')
monthly_counts = df.groupby(['year_month', 'gem_color']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.to_timestamp()
monthly_counts = monthly_counts.asfreq('MS')

train_data = monthly_counts[monthly_counts.index.year <= 2025].copy()
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year

X = train_data[['year', 'month']]
y = train_data.drop(columns=['year', 'month'], errors='ignore')

# ✅ Train model
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
model.fit(X, y)

# ✅ Forecast for 2026
future_dates = pd.date_range(start="2026-01-01", end="2026-12-01", freq='MS')
future_df = pd.DataFrame({
    'year': future_dates.year,
    'month': future_dates.month
})
predictions = model.predict(future_df)
prediction_df = pd.DataFrame(predictions, index=future_dates, columns=y.columns)

# -------------------------------
# 🔷 1. Smoothed Line Chart
# -------------------------------
st.subheader("📈 Smoothed Line Chart - 2026 Gemstone Usage Forecast")
plt.figure(figsize=(12, 6))
x_vals = np.arange(len(prediction_df.index))  # 0 to 11

for column in prediction_df.columns:
    y_vals = prediction_df[column].values
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    y_smooth = make_interp_spline(x_vals, y_vals, k=3)(x_smooth)
    plt.plot(x_smooth, y_smooth, label=column)

plt.xticks(x_vals, prediction_df.index.strftime('%b'), rotation=45)
plt.title("Predicted Monthly Gem Usage in 2026 (Smoothed Line Chart)")
plt.xlabel("Month")
plt.ylabel("Predicted Gem Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# -------------------------------
# 🔷 2. Grouped Bar Chart
# -------------------------------
st.subheader("📊 Grouped Bar Chart - Monthly Gemstone Usage")
fig, ax = plt.subplots(figsize=(14, 6))
bar_width = 0.1
x = np.arange(len(prediction_df.index))  # 12 months

for i, column in enumerate(prediction_df.columns):
    ax.bar(x + i * bar_width, prediction_df[column], width=bar_width, label=column)

ax.set_xticks(x + bar_width * (len(prediction_df.columns) - 1) / 2)
ax.set_xticklabels(prediction_df.index.strftime('%b'), rotation=45)

ax.set_title("Predicted Gemstone Usage per Month in 2026 (Grouped Bar Chart)")
ax.set_xlabel("Month")
ax.set_ylabel("Predicted Gem Count")
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig)
