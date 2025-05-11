import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import snowflake.connector
import matplotlib.pyplot as plt
from datetime import datetime

# Load models
models = joblib.load("sapphire_xgb_models.pkl")

# Snowflake connection
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user='YOUR_USERNAME',
        password='YOUR_PASSWORD',
        account='YOUR_ACCOUNT_URL',
        warehouse='YOUR_WAREHOUSE',
        database='YOUR_DATABASE',
        schema='YOUR_SCHEMA'
    )
    query = "SELECT WEIGHT, PRICE, TIMESTAMP, WEIGHT_RANGE FROM YOUR_TABLE_NAME"
    df = pd.read_sql(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Forecasting function
def forecast_price(model, year, month, data):
    model_data = data.copy()
    model_data['Year'] = model_data['timestamp'].dt.year
    model_data['Month'] = model_data['timestamp'].dt.month
    X = model_data[['Year', 'Month']]
    
    # Predict for selected future date
    prediction = model.predict(pd.DataFrame([[year, month]], columns=['Year', 'Month']))
    return prediction[0]

# UI
st.title("Gemstone Price Forecasting")

# Sidebar inputs
year = st.selectbox("Select Year", list(range(datetime.now().year + 1, 2031)))
month = st.selectbox("Select Month", list(range(1, 13)))
weight_range = st.selectbox("Select Weight Range", ["0.5-2", "2-4", "5-6"])

# Load data and map to model
df = get_data_from_snowflake()

range_map = {
    "0.5-2": 1,
    "2-4": 2,
    "5-6": 3
}
model_key = range_map[weight_range]
model = models[model_key]

# Filter for historical chart
historical = df[df['WEIGHT_RANGE'] == model_key]
historical = historical.groupby(pd.Grouper(key='timestamp', freq='M'))['PRICE'].mean().reset_index()

# Predict
if st.button("Forecast Price"):
    price = forecast_price(model, year, month, historical)
    st.success(f"Predicted Price for {month}/{year}: ${price:.2f}")

    # Plot historical data
    plt.figure(figsize=(10, 4))
    plt.plot(historical['timestamp'], historical['PRICE'], marker='o')
    plt.axvline(datetime(year, month, 1), color='red', linestyle='--', label='Forecast Point')
    plt.title(f"Historical Prices for {weight_range}")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend()
    st.pyplot(plt)
