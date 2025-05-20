import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import snowflake.connector

# Load data from Snowflake
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

# Train XGBoost model
def train_xgboost(df):
    X = df[['Year', 'Month']]
    y = df['Gold_Price_LKR']
    model = XGBRegressor()
    model.fit(X, y)
    return model

# Forecast using XGBoost
def forecast_with_xgboost(model, selected_dates):
    df_future = pd.DataFrame({
        'Year': selected_dates.dt.year,
        'Month': selected_dates.dt.month
    })
    preds = model.predict(df_future)
    return pd.DataFrame({'Date': selected_dates, 'Predicted_Price_LKR': preds})

# Streamlit App UI
st.title("🪙 Gold Price Forecast (2026–2028) — XGBoost Only")
st.markdown("Select future years and months to predict gold prices in **LKR** using XGBoost.")

# Load and train
df = load_data_from_snowflake()
xgb_model = train_xgboost(df)

# Selection inputs
years = st.multiselect("📅 Select Years", [2026, 2027, 2028], default=[2026])
months = st.multiselect("📆 Select Months", list(range(1, 13)), default=[1, 6, 12])

# Prediction button
if st.button("✅ Predict with XGBoost"):
    if not years or not months:
        st.warning("Please select at least one year and one month.")
    else:
        # Build future dates
        selected_dates = pd.to_datetime(
            [f"{y}-{m:02d}-01" for y in years for m in months]
        ).sort_values()

        forecast_df = forecast_with_xgboost(xgb_model, selected_dates)

        # Line chart
        st.subheader("📈 Forecasted Gold Prices (LKR)")
        fig, ax = plt.subplots()
        ax.plot(forecast_df['Date'], forecast_df['Predicted_Price_LKR'], marker='o', color='gold', label="XGBoost")
        ax.set_title("Gold Price Forecast (XGBoost)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price (LKR)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Table
        st.dataframe(forecast_df.set_index("Date"))
