import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector

st.set_page_config(layout="wide")
st.title("🔮 Gemstone Usage Forecast")

# ✅ Fetch data from Snowflake
@st.cache_data
def get_gem_data():
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
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

# ✅ Load data
try:
    df = get_gem_data()
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# ✅ Preprocess
df['year_month'] = df['created_at'].dt.to_period('M')
monthly_counts = df.groupby(['year_month', 'gem_color']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.to_timestamp()
monthly_counts = monthly_counts.asfreq('MS')

# ✅ Sidebar user input
st.sidebar.header("🔧 Select Forecast Period")
selected_year = st.sidebar.selectbox("Select Forecast Year", [2026, 2027, 2028])
selected_month = st.sidebar.selectbox("Select Forecast Month", list(range(1, 13)))
confirm = st.sidebar.button("✅ Confirm Selection")

# ✅ Only proceed on confirm
if confirm:
    # Prepare training data (up to and including 2025 only)
    train_data = monthly_counts[monthly_counts.index.year <= 2025].copy()
    train_data['month'] = train_data.index.month
    train_data['year'] = train_data.index.year
    X = train_data[['year', 'month']]
    y = train_data.drop(columns=['year', 'month'], errors='ignore')

    # Train model
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    model.fit(X, y)

    # Create forecast index for selected year
    future_dates = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-01", freq='MS')
    future_df = pd.DataFrame({'year': future_dates.year, 'month': future_dates.month})
    predictions = model.predict(future_df)
    prediction_df = pd.DataFrame(predictions, index=future_dates, columns=y.columns)

    # Extract past year (2025) actual data
    past_year_actual = monthly_counts[monthly_counts.index.year == 2025].copy()
    predicted_selected_year = prediction_df

    # -------------------------------
    # 🔷 1. Smoothed Line Chart
    # -------------------------------
    st.subheader(f"📈  {selected_year} Forecast - Gem Usage Trend")

    fig, ax = plt.subplots(figsize=(16, 7))
    x_actual = np.arange(len(past_year_actual))
    x_pred = np.arange(len(predicted_selected_year))
    months = predicted_selected_year.index.strftime('%b')

    for column in y.columns:
        # Actual past year
        if column in past_year_actual.columns:
            y_actual = past_year_actual[column].values
            x_smooth_actual = np.linspace(x_actual.min(), x_actual.max(), 300)
            y_smooth_actual = make_interp_spline(x_actual, y_actual, k=3)(x_smooth_actual)
            ax.plot(x_smooth_actual, y_smooth_actual, label=f"{column} (2025)", linestyle="-")

        # Forecast
        y_pred = predicted_selected_year[column].values
        x_smooth_pred = np.linspace(x_pred.min(), x_pred.max(), 300)
        y_smooth_pred = make_interp_spline(x_pred, y_pred, k=3)(x_smooth_pred)
        ax.plot(x_smooth_pred, y_smooth_pred, label=f"{column} ({selected_year})", linestyle="--")

    ax.set_xticks(x_pred)
    ax.set_xticklabels(months, rotation=45)

    # Highlight selected forecast month
    if 1 <= selected_month <= 12:
        ax.axvline(x=selected_month - 1, color='red', linestyle=':', linewidth=2, label='Selected Month')

    ax.set_title(f"📊 Gemstone Usage: Actual 2025 vs Forecast {selected_year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Gem Count")
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    st.pyplot(fig)

    # -------------------------------
    # 🔷 2. Bar Chart for Selected Month
    # -------------------------------
    st.subheader(f"📊 Gemstone Forecast for {selected_month:02}/{selected_year}")
    selected_date = pd.to_datetime(f"{selected_year}-{selected_month:02}-01")

    if selected_date in prediction_df.index:
        month_data = prediction_df.loc[selected_date]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(month_data.index, month_data.values, color='teal')
        ax2.set_title(f"Predicted Gemstone Usage for {selected_date.strftime('%B %Y')}")
        ax2.set_xlabel("Gemstone Color")
        ax2.set_ylabel("Predicted Count")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Forecast data not available for selected date.")
else:
    st.info("ℹ️ Please select a forecast year and month, then click '✅ Confirm Selection' to view predictions.")
