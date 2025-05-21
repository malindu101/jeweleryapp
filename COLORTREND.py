import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import snowflake.connector

st.set_page_config(layout="wide")
st.title("💎 Gem Color Usage Forecasting (2026–2028)")

# Load data from Snowflake
@st.cache_data
def load_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="YOUR_USERNAME",
        password="YOUR_PASSWORD",
        account="YOUR_ACCOUNT",
        warehouse="YOUR_WAREHOUSE",
        database="YOUR_DATABASE",
        schema="YOUR_SCHEMA"
    )
    query = "SELECT * FROM COLORTREND"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    valid_colors = ['Ruby', 'Sapphire', 'Emerald', 'Diamond']
    for color in valid_colors:
        df[color] = df[['top_color_1', 'top_color_2', 'top_color_3']].apply(lambda x: sum(x == color), axis=1)

    monthly_usage = df.groupby(['year', 'month'])[['Ruby', 'Sapphire', 'Emerald', 'Diamond']].sum().reset_index()
    return monthly_usage, valid_colors

monthly_usage, valid_colors = load_data_from_snowflake()

# Sidebar selections
selected_year = st.sidebar.selectbox("Select Year", [2026, 2027, 2028])
selected_month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# Train model and forecast
@st.cache_data
def train_and_predict():
    X = monthly_usage[['year', 'month']]
    future = pd.concat([pd.DataFrame({'year': [yr]*12, 'month': list(range(1, 13))}) for yr in [2026, 2027, 2028]], ignore_index=True)

    predictions = {}
    for gem in valid_colors:
        y = monthly_usage[gem]
        model = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1, verbosity=0)
        model.fit(X, y)
        predictions[gem] = np.round(model.predict(future)).astype(int)

    predicted_df = future.copy()
    for gem in valid_colors:
        predicted_df[gem] = predictions[gem]

    return predicted_df

predicted_df = train_and_predict()

# Filter selected month-year data
selected_row = predicted_df[(predicted_df['year'] == selected_year) & (predicted_df['month'] == selected_month)]

st.subheader(f"📊 Predicted Gem Usage for {selected_month}/{selected_year}")
if not selected_row.empty:
    usage = selected_row[valid_colors].values.flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(valid_colors, usage, color=['red', 'blue', 'green', 'purple'])
    ax.set_ylabel("Predicted Usage Count")
    ax.set_title(f"Predicted Gem Usage - {selected_month}/{selected_year}")
    ax.grid(axis='y')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, int(bar.get_height()), ha='center', va='bottom')
    st.pyplot(fig)
else:
    st.warning("No prediction available for selected month and year.")

# Display full year data
st.subheader(f"📅 Monthly Forecast for {selected_year}")
yearly_data = predicted_df[predicted_df['year'] == selected_year].reset_index(drop=True)
st.dataframe(yearly_data)

# Optional: Smoothed line chart for 2026
st.subheader("📈 Smoothed Forecast for 2026")
months = np.array(range(1, 13))
fig, ax = plt.subplots(figsize=(10, 6))
for gem in valid_colors:
    y = predicted_df[predicted_df['year'] == 2026][gem].values
    spline = make_interp_spline(months, y, k=3)
    xnew = np.linspace(months.min(), months.max(), 300)
    y_smooth = spline(xnew)
    ax.plot(xnew, y_smooth, label=gem)

ax.set_title("Smoothed Gem Usage Forecast (2026)")
ax.set_xlabel("Month")
ax.set_ylabel("Usage Count")
ax.set_xticks(months)
ax.legend()
ax.grid(True)
st.pyplot(fig)
