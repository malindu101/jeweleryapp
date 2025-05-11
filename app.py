import streamlit as st
import joblib
import matplotlib.pyplot as plt
from utils import get_data_from_snowflake

st.set_page_config(page_title="Gem Price Predictor", layout="centered")
st.title("💎 Gemstone Price Prediction – Model 1")

# User input
month = st.selectbox("Select Month", list(range(1, 13)))
year = st.selectbox("Select Year", list(range(2015, 2026)))

if st.button("Predict"):
    df = get_data_from_snowflake(month, year)
    
    if df.empty:
        st.warning("No data found for the selected date.")
    else:
        model = joblib.load("models/model1.pkl")
        X = df[['month', 'year', 'weight']]  # adjust to your model features
        df['predicted_price'] = model.predict(X)

        # Show results
        st.subheader("📋 Prediction Table")
        st.dataframe(df[['weight', 'predicted_price']])

        st.subheader("📊 Predicted Price Chart")
        fig, ax = plt.subplots()
        ax.bar(df['weight'].astype(str), df['predicted_price'])
        ax.set_xlabel("Weight Range")
        ax.set_ylabel("Price")
        ax.set_title("Predicted Gemstone Prices")
        st.pyplot(fig)
