import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector
from datetime import datetime

st.set_page_config(page_title="Gemstone Usage Prediction", layout="wide")
st.title("💎 Predicted Gemstone Usage in 2026")

# -------------------------------
# Fetch Data from Snowflake
# -------------------------------
@st.cache_data
def load_data():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="YOUR_PASSWORD",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT * FROM SEQUENTIAL_CUSTOM_ORDERS"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df

df = load_data()

# -------------------------------
# Preprocess Data
# -------------------------------
df['year_month'] = df['created_at'].dt.to_period('M')
monthly_counts = df.groupby(['year_month', 'gem_color']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.to_timestamp()
monthly_counts = monthly_counts.asfreq('MS')

train_data = monthly_counts[monthly_counts.index.year <= 2025]
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year

X = train_data[['year', 'month']]
y = train_data.drop(columns=['year', 'month'])

# -------------------------------
# Train the Model
# -------------------------------
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
model.fit(X, y)

# -------------------------------
# Predict for 2026
# -------------------------------
future_dates = pd.date_range(start="2026-01-01", end="2026-12-01", freq='MS')
future_df = pd.DataFrame({'year': future_dates.year, 'month': future_dates.month})
predictions = model.predict(future_df)
prediction_df = pd.DataFrame(predictions, index=future_dates, columns=y.columns)

# -------------------------------
# Smoothed Line Chart
# -------------------------------
st.subheader("📈 Smoothed Line Chart")

fig1, ax1 = plt.subplots(figsize=(12, 6))
x_vals = np.arange(len(prediction_df.index))

for column in prediction_df.columns:
    y_vals = prediction_df[column].values
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    y_smooth = make_interp_spline(x_vals, y_vals, k=3)(x_smooth)
    ax1.plot(x_smooth, y_smooth, label=column)

ax1.set_xticks(x_vals)
ax1.set_xticklabels(prediction_df.index.strftime('%b'), rotation=45)
ax1.set_title("Predicted Monthly Gem Usage in 2026 (Smoothed)")
ax1.set_xlabel("Month")
ax1.set_ylabel("Predicted Gem Count")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# -------------------------------
# Grouped Bar Chart
# -------------------------------
st.subheader("📊 Grouped Bar Chart")

fig2, ax2 = plt.subplots(figsize=(14, 6))
bar_width = 0.1
x = np.arange(len(prediction_df.index))

for i, column in enumerate(prediction_df.columns):
    ax2.bar(x + i * bar_width, prediction_df[column], width=bar_width, label=column)

ax2.set_xticks(x + bar_width * (len(prediction_df.columns) - 1) / 2)
ax2.set_xticklabels(prediction_df.index.strftime('%b'), rotation=45)
ax2.set_title("Predicted Gemstone Usage per Month in 2026 (Bar Chart)")
ax2.set_xlabel("Month")
ax2.set_ylabel("Predicted Gem Count")
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig2)

# -------------------------------
# CSV Download Button
# -------------------------------
csv = prediction_df.reset_index().to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Prediction Data as CSV", csv, "predicted_gemstone_usage_2026.csv", "text/csv")
