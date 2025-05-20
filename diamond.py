import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🪙 Gold Price Forecasting in LKR")

# Connect to Snowflake and load data
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT * FROM GOLD_PRICE"
    df = pd.read_sql(query, conn)
    conn.close()

    # Clean and format
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'gold price / lkr': 'price'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    df.dropna(subset=['date', 'price'], inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    return df

# Sidebar controls
st.sidebar.header("🔧 Forecast Options")
year = st.sidebar.selectbox("Select Year", [2026, 2027, 2028])
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# Load data
try:
    df = get_data_from_snowflake()
except Exception as e:
    st.error(f"❌ Failed to load data from Snowflake: {e}")
    st.stop()

if st.sidebar.button("Confirm Selection"):

    def forecast_price(data, target_year, target_month):
        data = data.sort_values("date")
        data['Year'] = data['date'].dt.year
        data['Month'] = data['date'].dt.month

        # Use last 1 year of data
        one_year_ago = data['date'].max() - pd.DateOffset(years=1)
        recent = data[data['date'] >= one_year_ago]

        X = recent[['Year', 'Month']]
        y = recent['price']

        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)

        # Forecast input
        input_df = pd.DataFrame([[target_year, target_month]], columns=['Year', 'Month'])
        prediction = model.predict(input_df)
        return prediction[0], recent['date'], y, model

    predicted_price, hist_x, hist_y, trained_model = forecast_price(df, year, month)

    st.subheader(f"📊 Forecasted Gold Price for {month}/{year}: **LKR {predicted_price:,.2f}**")

    # Generate future forecast
    future_dates = pd.date_range(start=df['date'].max() + pd.DateOffset(months=1), periods=36, freq='MS')
    future_X = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month})
    future_preds = trained_model.predict(future_X)

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

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 5))
    smooth_plot(hist_x, hist_y, 'Historical (1 Year)', 'goldenrod')
    smooth_plot(future_dates, future_preds, 'Forecast (Next 3 Years)', 'gold', linestyle='--')
    plt.axvline(datetime(year, month, 1), color='red', linestyle=':', label='Forecasted Month')
    plt.title("Gold Price Forecast Trend")
    plt.xlabel("Date")
    plt.ylabel("Gold Price (LKR)")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    st.pyplot(fig)

else:
    st.info("Please select a month and year, then click 'Confirm Selection' to forecast gold price.")
