import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector

st.set_page_config(layout="wide")
st.title("📈 Sapphire Price Forecasting (Live from Snowflake)")

# ✅ Snowflake connection
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT timestamp, weight_range, price FROM SAPPHIRE_PRICE"
    df = pd.read_sql(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

try:
    df = get_data_from_snowflake()
    st.success("✅ Connected to Snowflake and retrieved data successfully!")
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# Group monthly
monthly = df.groupby([pd.Grouper(key='timestamp', freq='MS'), 'weight_range'])['price'].mean().reset_index()

# Filter to last 1 year
latest_date = monthly['timestamp'].max()
one_year_ago = latest_date - pd.DateOffset(years=1)
monthly_filtered = monthly[monthly['timestamp'] >= one_year_ago]

# Forecasting function
def predict_xgb(data, range_type, future_months=12):
    sub = data[data['weight_range'] == range_type].copy()
    sub['Year'] = sub['timestamp'].dt.year
    sub['Month'] = sub['timestamp'].dt.month
    X = sub[['Year', 'Month']]
    y = sub['price']
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)

    last_date = sub['timestamp'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='MS')
    future_X = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month})
    preds = model.predict(future_X)
    return sub['timestamp'], y, future_dates, preds

# Forecast for all types
t1_x, t1_y, f1_x, f1_y = predict_xgb(monthly_filtered, 1)
t2_x, t2_y, f2_x, f2_y = predict_xgb(monthly_filtered, 2)
t3_x, t3_y, f3_x, f3_y = predict_xgb(monthly_filtered, 3)

# Plotting
def smooth_plot(x, y, label, color, linestyle='-'):
    if len(x) < 4:
        plt.plot(x, y, label=label, color=color, linestyle=linestyle)
        return
    x_numeric = pd.to_datetime(x).astype(np.int64) // 10**9 // 86400
    spline = make_interp_spline(x_numeric, y, k=3)
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 300)
    y_smooth = spline(x_smooth)
    x_smooth_dt = pd.to_datetime(x_smooth * 86400, unit='s', origin='unix')
    plt.plot(x_smooth_dt, y_smooth, label=label, color=color, linestyle=linestyle)

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'axes.edgecolor': 'gray',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'axes.titlepad': 15,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'figure.facecolor': 'white'
})

# Final Plot
fig = plt.figure(figsize=(14, 6))
smooth_plot(t1_x, t1_y, 'Type 1 - Historical', 'blue')
smooth_plot(f1_x, f1_y, 'Type 1 - Forecast', 'blue', linestyle='--')
smooth_plot(t2_x, t2_y, 'Type 2 - Historical', 'green')
smooth_plot(f2_x, f2_y, 'Type 2 - Forecast', 'green', linestyle='--')
smooth_plot(t3_x, t3_y, 'Type 3 - Historical', 'red')
smooth_plot(f3_x, f3_y, 'Type 3 - Forecast', 'red', linestyle='--')
plt.title('Sapphire Price Forecast')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
