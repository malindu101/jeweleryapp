import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🪙 Gold Price Forecasting")

# Snowflake data fetch function
def get_gold_data():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    df = pd.read_sql("SELECT * FROM GOLD_PRICE", conn)
    conn.close()

    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(subset=['date', 'gold price /lkr'], inplace=True)
    df['price'] = df['gold price /lkr'].astype(float)
    return df

# Sidebar inputs
st.sidebar.header("🔧 Select Forecast Options")
year = st.sidebar.selectbox("Select Year", [2026, 2027, 2028])
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# Try loading data
try:
    df = get_gold_data()
except Exception as e:
    st.error(f"❌ Error loading data from Snowflake: {e}")
    st.stop()

# Forecast and plot after confirmation
if st.sidebar.button("Confirm Selection"):

    def forecast_gold(data, target_year, target_month):
        sub = data.copy()
        sub = sub.sort_values("date")
        latest_date = sub['date'].max()
        one_year_ago = latest_date - pd.DateOffset(years=1)
        sub = sub[sub['date'] >= one_year_ago]
        sub['Year'] = sub['date'].dt.year
        sub['Month'] = sub['date'].dt.month
        X = sub[['Year', 'Month']]
        y = sub['price']
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        input_df = pd.DataFrame([[target_year, target_month]], columns=['Year', 'Month'])
        prediction = model.predict(input_df)
        return prediction[0], sub['date'], y, model

    # Perform forecast
    predicted_price, hist_x, hist_y, model = forecast_gold(df, year, month)

    st.subheader(f"📊 Predicted Gold Price for {month}/{year}: **LKR {predicted_price:,.2f}**")

    # Prepare future forecast (next 3 years)
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=36, freq='MS')
    future_X = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month})
    future_preds = model.predict(future_X)

    # Plotting with smoothing
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

    # Plot section
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 5))
    smooth_plot(hist_x, hist_y, 'Historical (1 Year)', 'goldenrod')
    smooth_plot(future_dates, future_preds, 'Forecast (Next 3 Years)', 'gold', linestyle='--')
    plt.axvline(datetime(year, month, 1), color='darkorange', linestyle=':', label='Selected Forecast Month')
    plt.title(f"Gold Price Trend Forecast")
    plt.xlabel("Month")
    plt.ylabel("Price (LKR)")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    st.pyplot(fig)

else:
    st.info("Please select forecast options and click 'Confirm Selection'.")
