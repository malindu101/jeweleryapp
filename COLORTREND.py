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
    query = "SELECT * FROM COLORTREND"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    df.rename(columns={'timestamp': 'date'}, inplace=True)
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

# Melt top color columns
df_melted = pd.melt(
    df,
    id_vars=['date', 'year', 'month'],
    value_vars=['top_color_1', 'top_color_2', 'top_color_3'],
    var_name='rank',
    value_name='color'
)
df_melted.dropna(subset=['color'], inplace=True)

# Limit to past 2 years
today = datetime.now()
cutoff_start = (today.replace(day=1) - pd.DateOffset(years=2)).to_period('M').to_timestamp()
df_melted = df_melted[df_melted['date'] >= cutoff_start]

# Aggregate monthly color counts
monthly_color_usage = df_melted.groupby(['year', 'month', 'color']).size().reset_index(name='count')

# Forecast section
st.subheader("🎨 Color Forecast for Month {}".format(month))
chart_df = pd.DataFrame()
all_colors = monthly_color_usage['color'].unique()
future_dates = pd.date_range(start=today.replace(day=1), periods=24, freq='MS')

for color in all_colors:
    color_data = monthly_color_usage[monthly_color_usage['color'] == color].copy()
    color_data = color_data.sort_values(['year', 'month'])
    color_data['target'] = color_data['count']

    X_train = color_data[['year', 'month']]
    y_train = color_data['target']

    model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Predict future months
    future_input = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month
    })
    preds = model.predict(future_input).clip(0)
    future_input['count'] = preds
    future_input['color'] = color
    future_input['date'] = future_dates
    future_input['type'] = 'Forecast'

    # Historical
    hist = color_data.copy()
    hist['date'] = pd.to_datetime(hist[['year', 'month']].assign(day=1))
    hist = hist[hist['date'] >= cutoff_start]
    hist['type'] = 'Actual'
    hist = hist[['date', 'count', 'color', 'type']]

    combined = pd.concat([
        hist[['date', 'count', 'color', 'type']],
        future_input[['date', 'count', 'color', 'type']]
    ])
    chart_df = pd.concat([chart_df, combined])

# Filter chart data to selected month and year ± 2
selected_years = [year - 2, year - 1, year]
filtered_df = chart_df[
    (chart_df['date'].dt.month == month) &
    (chart_df['date'].dt.year.isin(selected_years))
]

if filtered_df.empty:
    st.warning("No data found for the selected month and years.")
else:
    # 📈 Line Chart
    st.markdown("### 📉 Line Chart for Month {}".format(month))
    fig, ax = plt.subplots(figsize=(10, 5))
    for color in filtered_df['color'].unique():
        actual = filtered_df[(filtered_df['color'] == color) & (filtered_df['type'] == 'Actual')]
        forecast = filtered_df[(filtered_df['color'] == color) & (filtered_df['type'] == 'Forecast')]
        ax.plot(actual['date'], actual['count'], label=f"{color} - Actual", marker='o')
        ax.plot(forecast['date'], forecast['count'], label=f"{color} - Forecast", linestyle='--')
    ax.set_title(f"Gem Color Usage Trend - {month} over {selected_years}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Usage Count")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 📊 Bar Chart
    st.markdown("### 📊 Bar Chart for Month {}".format(month))
    bar_df = filtered_df.copy()
    bar_df['year'] = bar_df['date'].dt.year
    bar_pivot = bar_df.pivot_table(index='year', columns='color', values='count', fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title(f"Gem Color Forecast (Bar Chart) - Month {month}")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Usage Count")
    ax2.legend(title="Color")
    st.pyplot(fig2)
