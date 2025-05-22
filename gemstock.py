import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🪙 Gold Live Price Forecasting")

# ✅ Fetch gold price data from Snowflake
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

    # ✅ Rename and preprocess
    df.rename(columns={
        'DATE': 'timestamp',
        'Gold Price /LKR': 'price'
    }, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price'] = df['price'].astype(float)
    return df

# ✅ Sidebar options
st.sidebar.header("🔧 Select Forecast Options")
year = st.sidebar.selectbox("Select Year", list(range(datetime.now().year, 2029)))
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# ✅ Load data
try:
    df = get_data_from_snowflake()
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# ✅ Forecast logic
if st.sidebar.button("Confirm Selection"):
    def forecast_price(data, target_year, target_month):
        sub = data.sort_values("timestamp")
        one_year_ago = sub['timestamp'].max() - pd.DateOffset(years=1)
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

    # ✅ Run prediction
    predicted_price, hist_x, hist_y, trained_model = forecast_price(df, year, month)
    st.subheader(f"📊 Predicted Gold Price for {month}/{year}: **LKR {predicted_price:,.2f}**")

    # ✅ Forecast future prices
    last_date = df['timestamp'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=36, freq='MS')
    future_X = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month})
    future_preds = trained_model.predict(future_X)

    # ✅ Smoothed plotting function
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

    # ✅ Plot results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 5))
    smooth_plot(hist_x, hist_y, 'Historical (1 Year)', 'goldenrod')
    smooth_plot(future_dates, future_preds, 'Forecast (Next 3 Years)', 'goldenrod', linestyle='--')
    plt.axvline(datetime(year, month, 1), color='red', linestyle=':', label='Selected Forecast Month')
    plt.title("📈 Gold Price Trend in LKR")
    plt.xlabel("Month")
    plt.ylabel("Price (LKR)")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    st.pyplot(fig)

else:
    st.info("ℹ️ Please select options and click 'Confirm Selection' to view prediction.")
