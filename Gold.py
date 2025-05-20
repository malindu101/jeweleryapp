import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from prophet import Prophet
import pmdarima as pm
import snowflake.connector

# Load from Snowflake
@st.cache_data
def load_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",         # Same DB
        schema="PUBLIC"
    )
    df = pd.read_sql("SELECT * FROM GOLD_PRICE", conn)
    conn.close()
    df.columns = [c.strip() for c in df.columns]
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df.dropna(subset=['DATE', 'Gold Price /LKR'], inplace=True)
    df['Gold_Price_LKR'] = df['Gold Price /LKR'].astype(float)
    df['Month'] = df['DATE'].dt.month
    df['Year'] = df['DATE'].dt.year
    return df[['DATE', 'Year', 'Month', 'Gold_Price_LKR']]

# Train XGBoost
def train_xgboost(df):
    X = df[['Year', 'Month']]
    y = df['Gold_Price_LKR']
    model = XGBRegressor()
    model.fit(X, y)
    return model

# Forecast with Prophet
def forecast_with_prophet(df, selected_dates):
    df_prophet = df[['DATE', 'Gold_Price_LKR']].rename(columns={"DATE": "ds", "Gold_Price_LKR": "y"})
    model = Prophet()
    model.fit(df_prophet)

    future_df = pd.DataFrame({'ds': selected_dates})
    forecast = model.predict(future_df)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Prophet'})

# Forecast with ARIMA
def forecast_with_arima(df, selected_dates):
    ts = df.set_index('DATE')['Gold_Price_LKR']
    model = pm.auto_arima(ts, seasonal=False, suppress_warnings=True)
    steps = len(selected_dates)
    forecast = model.predict(n_periods=steps)
    return pd.DataFrame({'Date': selected_dates, 'ARIMA': forecast})

# Forecast with XGBoost
def forecast_with_xgboost(model, selected_dates):
    df_future = pd.DataFrame({
        'Year': selected_dates.dt.year,
        'Month': selected_dates.dt.month
    })
    preds = model.predict(df_future)
    return pd.DataFrame({'Date': selected_dates, 'XGBoost': preds})

# UI
st.title("🪙 Gold Price Prediction (2026–2028) — Multi-Model Comparison")
df = load_data_from_snowflake()
xgb_model = train_xgboost(df)

# Select years and months
years = st.multiselect("📅 Select Years", [2026, 2027, 2028], default=[2026])
months = st.multiselect("📆 Select Months", list(range(1, 13)), default=[1, 6, 12])

if st.button("✅ Compare Models"):
    if not years or not months:
        st.warning("Please select at least one year and one month.")
    else:
        # Build date list for selected month/year pairs
        selected_dates = pd.to_datetime(
            [f"{y}-{m:02d}-01" for y in years for m in months]
        ).sort_values()

        prophet_df = forecast_with_prophet(df, selected_dates)
        arima_df = forecast_with_arima(df, selected_dates)
        xgb_df = forecast_with_xgboost(xgb_model, selected_dates)

        # Merge all
        result = prophet_df.merge(arima_df, on="Date").merge(xgb_df, on="Date")
        result = result.sort_values("Date")

        # Plot
        st.subheader("📊 Comparison of Forecasted Prices")
        fig, ax = plt.subplots()
        ax.plot(result['Date'], result['XGBoost'], label="XGBoost", marker='o')
        ax.plot(result['Date'], result['Prophet'], label="Prophet", marker='^')
        ax.plot(result['Date'], result['ARIMA'], label="ARIMA", marker='s')
        ax.set_xlabel("Date")
        ax.set_ylabel("LKR")
        ax.set_title("Gold Price Forecast by Model")
        ax.legend()
        st.pyplot(fig)

        # Table
        st.dataframe(result.set_index("Date"))
