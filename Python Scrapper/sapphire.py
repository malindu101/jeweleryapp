import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('gem_model.pkl')

# UI
st.title("Gemstone Price Prediction")
st.write("Select a month and year to predict prices by weight range.")

# Dropdowns
year = st.selectbox("Year", list(range(2015, 2026)))
month = st.selectbox("Month", list(range(1, 13)))

# Button to predict
if st.button("Predict"):
    # Example: Predict for all weight ranges
    weight_ranges = ['0.5-2', '2-4', '5-6']
    predictions = []

    for weight in weight_ranges:
        # You need to encode weight and pass correct input format
        weight_encoded = int(weight.split('-')[0])  # dummy transformation
        input_df = pd.DataFrame([[month, year, weight_encoded]], columns=['month', 'year', 'weight'])
        price = model.predict(input_df)[0]
        predictions.append((weight, price))

    # Display as chart
    weights, prices = zip(*predictions)
    fig, ax = plt.subplots()
    ax.bar(weights, prices)
    ax.set_xlabel("Weight Range")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Predicted Gem Prices for {month}/{year}")
    st.pyplot(fig)
