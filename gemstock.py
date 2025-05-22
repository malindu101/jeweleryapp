import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import make_interp_spline
import snowflake.connector

st.set_page_config(layout="wide")
st.title("🔮 Gemstone Usage Forecast (2025 vs Selected Year)")

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

# ✅ Extract 2025 actual data
actual_2025 = monthly_counts[monthly_counts.index.year == 2025].copy()

# ✅ Train on data up to 2025
train_data = monthly_counts[monthly_counts.index.year <= 2025].copy()
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year

X = train_data[['year', 'month']]
y = train_data.drop(columns=['year', 'month'], errors='ignore')

# ✅ Train model
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
model.fit(X, y)

# ✅ User selects year/month
st.sidebar.header("🔧 Select Forecast Period")
selected_year = st.sidebar.selectbox("Select Forecast Year", [2026, 2027, 2028])
selected_month = st.sidebar.selectbox("Select Forecast Month", list(range(1, 13)))

# ✅ Forecast for 2026–2028
future_dates = pd.date_range(start="2026-01-01", end="2028-12-01", freq='MS')
future_df = pd.DataFrame({'year': future_dates.year, 'month': future_dates.month})
predictions = model.predict(future_df)
prediction_df = pd.DataFrame(predictions, index=future_dates, columns=y.columns)

# ✅ Extract selected year from prediction
predicted_selected_year = prediction_df[prediction_df.index.year == selected_year]

# -------------------------------
# 🔷 1. Smoothed Line Chart
# -------------------------------
st.subheader("📈 Smoothed Line Chart - Actual 2025 vs Predicted " + str(selected_year))
plt.figure(figsize=(12, 6))
x_2025 = np.arange(len(actual_2025))
x_sel = np.arange(len(predicted_selected_year))

for column in y.columns:
    # Actual 2025
    y_2025_vals = actual_2025[column].values
    x_2025_smooth = np.linspace(x_2025.min(), x_2025.max(), 300)
    y_2025_smooth = make_interp_spline(x_2025, y_2025_vals, k=3)(x_2025_smooth)
    plt.plot(x_2025_smooth, y_2025_smooth, label=f"{column} (2025)", linestyle="-")

    # Forecast selected year
    y_sel_vals = predicted_selected_year[column].values
    x_sel_smooth = np.linspace(x_sel.min(), x_sel.max(), 300)
    y_sel_smooth = make_interp_spline(x_sel, y_sel_vals, k=3)(x_sel_smooth)
    plt.plot(x_sel_smooth, y_sel_smooth, label=f"{column} ({selected_year})", linestyle="--")

plt.xticks(x_2025, actual_2025.index.strftime('%b'), rotation=45)
plt.title(f"Predicted Monthly Gem Usage: 2025 (Actual) vs {selected_year} (Forecast)")
plt.xlabel("Month")
plt.ylabel("Gem Count")
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0.15, 1, 1])
st.pyplot(plt)

# -------------------------------
# 🔷 2. Bar Chart for Selected Month
# -------------------------------
st.subheader(f"📊 Grouped Bar Chart - {selected_month:02}/{selected_year} Gemstone Usage Forecast")
selected_date = pd.to_datetime(f"{selected_year}-{selected_month:02}-01")

if selected_date in prediction_df.index:
    month_data = prediction_df.loc[selected_date]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(month_data.index, month_data.values, color='teal')
    ax.set_title(f"Predicted Gemstone Usage for {selected_date.strftime('%B %Y')}")
    ax.set_xlabel("Gemstone Color")
    ax.set_ylabel("Predicted Count")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Selected date not found in forecast.")
