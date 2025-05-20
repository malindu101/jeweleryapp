import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from datetime import datetime

st.set_page_config(layout="wide")
st.title("💎 Gem Color Usage Trend")

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
    df['date'] = pd.to_datetime(df['date'])
    return df

# Sidebar for input
st.sidebar.header("🔧 Select Filter Options")
year = st.sidebar.selectbox("Select Year", list(range(2022, 2029)))
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# Fetch and filter data
try:
    df = get_data_from_snowflake()
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

# Extract month and year
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Filter by selected month and year
filtered = df[(df['year'] == year) & (df['month'] == month)]

# Check and visualize
if filtered.empty:
    st.warning("No color data found for the selected month and year.")
else:
    color_counts = filtered['color'].value_counts().reset_index()
    color_counts.columns = ['Color', 'Count']

    st.subheader(f"🎨 Most Used Colors in {month}/{year}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(color_counts['Color'], color_counts['Count'], color='skyblue')
    ax.set_xlabel("Color")
    ax.set_ylabel("Usage Count")
    ax.set_title("Most Used Gem Colors")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Optional: Show as pie chart
    with st.expander("📊 View as Pie Chart"):
        fig2, ax2 = plt.subplots()
        ax2.pie(color_counts['Count'], labels=color_counts['Color'], autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)
