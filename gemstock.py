import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import snowflake.connector
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🔮 Gemstone Usage Forecasting")

# --- Connect to Snowflake ---
@st.cache_data
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT * FROM STOCK"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df

df = get_data_from_snowflake()

# --- Preprocess data ---
df['year_month'] = df['created_at'].dt.to_period('M')
monthly_counts = df.groupby(['year_month', 'gem_color']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.to_timestamp()
monthly_counts = monthly_counts.asfreq('MS')
train_data = monthly_counts[monthly_counts.index.year <= 2025]
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year

X = train_data[['year', 'month']]
y = train_data.drop(columns=['year', 'month'])

# --- Model Training ---
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
model.fit(X, y)

# --- User Selection ---
st.sidebar.header("📅 Forecast Parameters")
selected_year = st.sidebar.selectbox("Select Year to Predict", list(range(2026, 2029)))
months = st.sidebar.multiselect("Select Months", list(range(1, 13)), default=list(range(1, 13)))

if months:
    # --- Predict for Selected Year and Months ---
    future_dates = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-01", freq='MS')
    future_dates = future_dates[future_dates.month.isin(months)]
    future_df = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month
    })

    predictions = model.predict(future_df)
    prediction_df = pd.DataFrame(predictions, index=future_dates, columns=y.columns)

    # --- Plot ---
    st.subheader(f"📊 Predicted Gemstone Usage for {selected_year}")
    st.line_chart(prediction_df)

    st.dataframe(prediction_df.style.format(precision=0))
else:
    st.warning("Please select at least one month to see predictions.")
