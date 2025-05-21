import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

st.set_page_config(layout="wide")
st.title("🔧 Material Usage Forecasting (2026–2028)")

# Load from uploaded CSV
@st.cache_data
def load_material_data():
    df = pd.read_csv("materialtrend.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    valid_materials = ['Gold', 'Silver', 'Platinum', 'Rose Gold']  # Adjust based on your dataset
    for material in valid_materials:
        df[material] = df[['top_material_1', 'top_material_2', 'top_material_3']].apply(lambda x: sum(x == material), axis=1)

    monthly_usage = df.groupby(['year', 'month'])[valid_materials].sum().reset_index()
    return monthly_usage, valid_materials

monthly_usage, valid_materials = load_material_data()

# Sidebar for input
selected_year = st.sidebar.selectbox("Select Year", [2026, 2027, 2028])
selected_month = st.sidebar.selectbox("Select Month", list(range(1, 13)))
confirm = st.sidebar.button("✅ Confirm Selection")

# Forecasting with XGBoost
@st.cache_data
def train_and_predict():
    X = monthly_usage[['year', 'month']]
    future = pd.concat(
        [pd.DataFrame({'year': [yr] * 12, 'month': list(range(1, 13))}) for yr in [2026, 2027, 2028]],
        ignore_index=True
    )

    predictions = {}
    for material in valid_materials:
        y = monthly_usage[material]
        model = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1, verbosity=0)
        model.fit(X, y)
        predictions[material] = np.round(model.predict(future))  # Round predictions

    predicted_df = future.copy()
    for material in valid_materials:
        predicted_df[material] = predictions[material].astype(int)

    return predicted_df

if confirm:
    predicted_df = train_and_predict()

    # Line Chart for full year
    st.subheader(f"📈 XGBoost Predicted Trends for All Materials in {selected_year}")
    monthly_data = predicted_df[predicted_df['year'] == selected_year]

    if not monthly_data.empty:
        months = monthly_data['month'].values
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        for material in valid_materials:
            y = monthly_data[material].values
            spline = make_interp_spline(months, y, k=3)
            xnew = np.linspace(months.min(), months.max(), 300)
            y_smooth = spline(xnew)
            ax2.plot(xnew, y_smooth, label=material)

        ax2.axvline(x=selected_month, color='black', linestyle='--', linewidth=1.5, label=f'Selected Month ({selected_month})')
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Predicted Usage Count")
        ax2.set_title(f"Predicted Monthly Trends for {selected_year}")
        ax2.set_xticks(months)
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
    else:
        st.warning("No prediction data available for this year.")

    # Bar Chart for selected month
    st.subheader(f"📊 Predicted Material Usage for {selected_month}/{selected_year}")
    selected_row = predicted_df[(predicted_df['year'] == selected_year) & (predicted_df['month'] == selected_month)]

    if not selected_row.empty:
        usage = selected_row[valid_materials].values.flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(valid_materials, usage, color=['gold', 'silver', 'gray', 'pink'])
        ax.set_ylabel("Predicted Usage Count")
        ax.set_title(f"Material Usage Forecast - {selected_month}/{selected_year}")
        ax.grid(axis='y')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, int(bar.get_height()), ha='center', va='bottom')
        st.pyplot(fig)
    else:
        st.warning("No prediction available for selected month and year.")
else:
    st.info("Please select a year and month, then click **Confirm Selection** to view forecasts.")
