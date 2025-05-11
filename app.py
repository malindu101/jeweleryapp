import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📈 Sapphire Price Forecasting (Live from Snowflake)")

# ✅ Connect to Snowflake
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT * FROM SAPPHIRE_PRICE"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ✅ Load data
try:
    df = get_data_from_snowflake()
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

#  Sidebar inputs
st.sidebar.header("🔧 Select Forecast Options")
year = st.sidebar.selectbox("Select Year", list(range(datetime.now().year, 2031)))
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))
weight_option = st.sidebar.selectbox("Select Weight Range", ["0.5–2", "2–4", "5–6"])
weight_map = {"0.5–2": 1, "2–4": 2, "5–6": 3}
selected_range = weight_map[weight_option]

#  Confirm Button
if st.sidebar.button(" Confirm Selection"):
    #  Forecasting function (trained on last 1 year of data)
    def forecast_price(data, range_type, target_year, target_month):
        sub = data[data['weight_range'] == range_type].copy()
        sub = sub.sort_values("timestamp")

        latest_date = sub['timestamp'].max()
        one_year_ago = latest_date - pd.DateOffset(years=1)
        sub = sub[sub['timestamp'] >= one_year_ago]

        sub['Year'] = sub['timestamp'].dt.year
        sub['Month'] = sub['timestamp'].dt.month

        X = sub[['Year', 'Month']]
        y = sub['price']
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)

        input_df = pd.DataFrame([[target_year, target_month]], columns=['Year', 'Month'])
        prediction = model.predict(input_df)
        return prediction[0], sub['timestamp'], y, model

    #  Run forecast
    predicted_price, hist_x, hist_y, trained_model = forecast_price(df, selected_range, year, month)
    st.subheader(f"📊 Predicted Price for {weight_option} in {month}/{year}: **${predicted_price:.2f}**")

    #  Future forecast
    last_date = df['timestamp'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    future_X = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month})
    future_preds = trained_model.predict(future_X)

    #  Smooth plot
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

    # Plotting

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(12, 5))
smooth_plot(hist_x, hist_y, 'Historical (1 Year)', 'blue')
smooth_plot(future_dates, future_preds, 'Forecast (Next 12 Months)', 'blue', linestyle='--')
plt.axvline(datetime(year, month, 1), color='red', linestyle=':', label='Selected Forecast Month')
plt.title(f"Price Trend for Weight Range: {weight_option}")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)  
st.pyplot(fig)

