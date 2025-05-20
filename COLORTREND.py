import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from datetime import datetime
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.title("📈 Gem Color Usage Trend & Forecast")

# Connect to Snowflake
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",         
        schema="PUBLIC"
    )
    query = "SELECT * FROM COLOR_TREND"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    return df

# Sidebar
st.sidebar.header("🔧 Filter Options")
year = st.sidebar.selectbox("Select Year", list(range(2022, 2029)))
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# Fetch data
try:
    df = get_data_from_snowflake()
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# Preprocessing
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['date'].dt.to_period('M').astype(str)

# Filter 4-year range (2 past + 2 future)
today = datetime.now()
cutoff_start = (today.replace(day=1) - pd.DateOffset(years=2)).to_period('M').to_timestamp()
cutoff_end = (today.replace(day=1) + pd.DateOffset(years=2)).to_period('M').to_timestamp()

# Aggregate monthly counts by color
df = df[df['date'] >= cutoff_start]
monthly_color_usage = df.groupby(['year', 'month', 'color']).size().reset_index(name='count')

# Prepare chart data with predictions
st.subheader("🎨 Color Trend Forecast (Past & Next 2 Years)")

chart_df = pd.DataFrame()
all_colors = df['color'].unique()
future_dates = pd.date_range(start=today.replace(day=1), periods=24, freq='MS')  # 2 years future

for color in all_colors:
    color_data = monthly_color_usage[monthly_color_usage['color'] == color].copy()
    color_data = color_data.sort_values(['year', 'month'])
    color_data['target'] = color_data['count']

    # Training data
    X_train = color_data[['year', 'month']]
    y_train = color_data['target']

    # Train XGBoost model
    model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Create prediction input
    future_input = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month
    })

    preds = model.predict(future_input).clip(0)  # No negative predictions
    future_input['count'] = preds
    future_input['color'] = color
    future_input['date'] = future_dates
    future_input['type'] = 'Forecast'

    # Historical for past 2 years only
    hist = color_data.copy()
    hist['date'] = pd.to_datetime(hist[['year', 'month']].assign(day=1))
    hist = hist[hist['date'] >= cutoff_start]
    hist['type'] = 'Actual'
    hist = hist[['date', 'count', 'color', 'type']]

    # Combine
    combined = pd.concat([
        hist[['date', 'count', 'color', 'type']],
        future_input[['date', 'count', 'color', 'type']]
    ])
    chart_df = pd.concat([chart_df, combined])

# Charting
selected_colors = st.multiselect("Select Colors to View", list(all_colors), default=list(all_colors))

if not selected_colors:
    st.warning("Please select at least one color.")
else:
    chart_df = chart_df[chart_df['color'].isin(selected_colors)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for color in selected_colors:
        sub_actual = chart_df[(chart_df['color'] == color) & (chart_df['type'] == 'Actual')]
        sub_forecast = chart_df[(chart_df['color'] == color) & (chart_df['type'] == 'Forecast')]
        ax.plot(sub_actual['date'], sub_actual['count'], label=f"{color} - Actual", marker='o')
        ax.plot(sub_forecast['date'], sub_forecast['count'], label=f"{color} - Forecast", linestyle='--')

    ax.set_title("Monthly Gem Color Usage: 2 Years Back & Forecast 2 Years Ahead")
    ax.set_xlabel("Month")
    ax.set_ylabel("Usage Count")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
