import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import snowflake.connector

# 1. Load data from Snowflake
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

# 2. Train model
def train_model(data):
    X = data[['Year', 'Month']]
    y = data['Gold_Price_LKR']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

# 3. Streamlit interface
st.title("🪙 Gold Price Forecasting (via Snowflake + XGBoost)")
st.markdown("Enter a future month and year to forecast the **gold price in LKR**.")

# Load and display data
df = load_data_from_snowflake()

st.subheader("📊 Summary Statistics")
st.write(df.describe())

# Line chart
st.subheader("📈 Historical Gold Price Trend")
fig, ax = plt.subplots()
ax.plot(df['DATE'], df['Gold_Price_LKR'], color='gold', label='Gold Price')
ax.set_xlabel("Date")
ax.set_ylabel("LKR")
ax.set_title("Gold Price Over Time")
ax.legend()
st.pyplot(fig)

# Boxplot
st.subheader("📦 Monthly Distribution")
fig2, ax2 = plt.subplots()
df.boxplot(column='Gold_Price_LKR', by='Month', ax=ax2)
plt.suptitle("")
ax2.set_title("Gold Price by Month")
ax2.set_xlabel("Month")
ax2.set_ylabel("LKR")
st.pyplot(fig2)

# Prediction form
st.subheader("🔢 Prediction Input")
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Select Year", min_value=2024, max_value=2030, value=2025)
with col2:
    month = st.selectbox("Select Month", list(range(1, 13)))

# Train model and predict
model = train_model(df)
predicted_price = model.predict(np.array([[year, month]]))[0]

# Display prediction
st.subheader("🔮 Predicted Gold Price")
st.success(f"Estimated price for {month}/{year}: **LKR {predicted_price:,.2f}**")

