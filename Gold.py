import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import snowflake.connector
from datetime import datetime

# Load data from Snowflake
@st.cache_data
def load_data_from_snowflake():
    conn = snowflake.connector.connect(
        user=st.secrets["user"],
        password=st.secrets["password"],
        account=st.secrets["account"],
        warehouse=st.secrets["warehouse"],
        database=st.secrets["database"],
        schema=st.secrets["schema"]
    )
    query = "SELECT * FROM GOLD_PRICE"
    df = pd.read_sql(query, conn)
    conn.close()

    # Clean and transform
    df.columns = [col.strip() for col in df.columns]
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df.dropna(subset=['DATE', 'Gold Price /LKR'], inplace=True)
    df['Gold_Price_LKR'] = df['Gold Price /LKR'].astype(float)
    df['Month'] = df['DATE'].dt.month
    df['Year'] = df['DATE'].dt.year
    return df[['DATE', 'Year', 'Month', 'Gold_Price_LKR']]

# Train XGBoost model
def train_model(df):
    X = df[['Year', 'Month']]
    y = df['Gold_Price_LKR']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

# Forecast next 24 months
def forecast_24_months(model, start_year):
    months = []
    years = []

    for i in range(24):
        month = (i % 12) + 1
        year = start_year + (i // 12)
        months.append(month)
        years.append(year)

    X_future = pd.DataFrame({'Year': years, 'Month': months})
    predictions = model.predict(X_future)
    forecast_df = pd.DataFrame({
        'Year': years,
        'Month': months,
        'Predicted_Price_LKR': predictions
    })
    forecast_df['Date'] = pd.to_datetime(forecast_df[['Year', 'Month']].assign(DAY=1))
    return forecast_df

# Streamlit UI
st.title("🪙 2-Year Monthly Gold Price Forecast (Snowflake + XGBoost)")
st.markdown("This tool predicts monthly gold prices in **LKR** for 2 years ahead based on selected start year.")

df = load_data_from_snowflake()
model = train_model(df)

# User input
st.subheader("📅 Select Starting Year")
available_years = list(range(datetime.now().year, datetime.now().year + 6))
start_year = st.selectbox("Start Prediction From", available_years, index=1)

if st.button("✅ Confirm and Predict"):
    forecast_df = forecast_24_months(model, start_year)

    st.subheader("📈 Forecasted Monthly Prices")
    fig, ax = plt.subplots()
    ax.plot(forecast_df['Date'], forecast_df['Predicted_Price_LKR'], marker='o', linestyle='-')
    ax.set_title(f"Gold Price Forecast from {start_year} for 24 Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Price (LKR)")
    ax.grid(True)
    st.pyplot(fig)

    # Optional: show table
    st.dataframe(forecast_df[['Date', 'Predicted_Price_LKR']].set_index('Date'))
