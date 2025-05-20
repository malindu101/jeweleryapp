import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import snowflake.connector

# 1. Snowflake credentials (secure these in secrets.toml or Streamlit cloud secrets)
@st.cache_data
def load_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT Date, `Gold Price  / LKR` AS Gold_Price_LKR FROM GOLD_TABLE_NAME"
    df = pd.read_sql(query, conn)
    conn.close()

    # Clean data
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(inplace=True)
    df['Gold_Price_LKR'] = df['Gold_Price_LKR'].astype(float)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df

# 2. Model training
def train_model(data):
    X = data[['Year', 'Month']]
    y = data['Gold_Price_LKR']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

# 3. UI starts here
st.title("Gold Price Prediction (via Snowflake + XGBoost)")
st.markdown("Predict Sri Lankan gold prices by selecting month and year.")

# Load and cache Snowflake data
df = load_data_from_snowflake()

# Summary
st.subheader("📊 Summary Statistics")
st.write(df.describe())

# Line Chart
st.subheader("📈 Gold Price Trend")
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Gold_Price_LKR'], color='gold', label='Gold Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price (LKR)")
ax.set_title("Gold Price Over Time")
ax.legend()
st.pyplot(fig)

# Boxplot
st.subheader("📦 Monthly Price Distribution")
fig2, ax2 = plt.subplots()
df.boxplot(column='Gold_Price_LKR', by='Month', ax=ax2)
plt.suptitle("")
ax2.set_title("Monthly Boxplot")
ax2.set_xlabel("Month")
ax2.set_ylabel("Price (LKR)")
st.pyplot(fig2)

# Prediction Input
st.subheader("📅 Predict Future Gold Price")
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Select Year", min_value=2024, max_value=2030, value=2025)
with col2:
    month = st.selectbox("Select Month", list(range(1, 13)))

# Train model and predict
model = train_model(df)
predicted_price = model.predict(np.array([[year, month]]))[0]

# Display result
st.subheader("🔮 Predicted Gold Price")
st.success(f"Estimated Gold Price for {month}/{year}: **LKR {predicted_price:,.2f}**")
