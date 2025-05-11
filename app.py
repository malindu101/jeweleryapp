import streamlit as st
import pandas as pd
import snowflake.connector
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Fetch data from Snowflake
def fetch_data():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )
    query = "SELECT * FROM SAPPHIRE_PRICE"
    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    return df

# Step 2: Train XGBoost models for each weight range
def train_models(df):
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df['month'] = df['TIMESTAMP'].dt.month
    df['year'] = df['TIMESTAMP'].dt.year

    models = {}
    trend_data = {}

    for weight_range in df['WEIGHT_RANGE'].unique():
        sub_df = df[df['WEIGHT_RANGE'] == weight_range]
        X = sub_df[['year', 'month']]
        y = sub_df['PRICE']
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        
        models[weight_range] = model
        trend_data[weight_range] = sub_df.groupby(['year', 'month'])['PRICE'].mean().reset_index()

    return models, trend_data

# Step 3: Streamlit App
st.set_page_config(page_title="Gemstone Price Predictor", layout="wide")
st.title("💎 XGBoost Gemstone Price Forecast (from Snowflake)")

selected_year = st.selectbox("Select Year", list(range(2026, 2031)))
selected_month = st.selectbox("Select Month", list(range(1, 13)))
selected_range = st.selectbox("Select Weight Range", ["0.5-2", "2-4", "5-6"])

if st.button("Predict Price"):
    try:
        df = fetch_data()
        models, trend_data = train_models(df)

        input_df = pd.DataFrame({
            'year': [selected_year],
            'month': [selected_month]
        })

        # Predict using the appropriate model
        model = models[selected_range]
        predicted_price = model.predict(input_df)[0]

        st.success(f"📈 Predicted Price for {selected_range} in {selected_year}-{selected_month:02d}: ${predicted_price:.2f}")

        # Show historical trend
        trend_df = trend_data[selected_range]
        trend_df['date'] = pd.to_datetime(trend_df[['year', 'month']].assign(day=1))

        fig, ax = plt.subplots()
        ax.plot(trend_df['date'], trend_df['PRICE'], label='Historical Price')
        ax.axvline(pd.to_datetime(f"{selected_year}-{selected_month:02d}-01"), color='red', linestyle='--', label='Prediction Month')
        ax.set_title(f"Price Trend for Weight Range {selected_range}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Price")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
