import streamlit as st
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# ----- Snowflake Connection -----
@st.cache_data
def load_data():
    conn = snowflake.connector.connect(
        user='YOUR_USER',
        password='YOUR_PASSWORD',
        account='YOUR_ACCOUNT',
        warehouse='YOUR_WAREHOUSE',
        database='YOUR_DATABASE',
        schema='YOUR_SCHEMA'
    )
    query = "SELECT * FROM user_behavior_1000_rows_final"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df = load_data()

# ----- UI Header -----
st.title("Material Usage Analytics Dashboard")
st.markdown("Live data from Snowflake")

# ----- Summary Section -----
st.header("📈 Summary Statistics")
st.write("Total rows:", len(df))
st.write("Null values per column:")
st.write(df.isnull().sum())

# ----- Visualization Section -----
st.header("📊 Visualizations")

# Plot material usage
st.subheader("Top Materials (All Time)")
material_cols = ['top_material_1', 'top_material_2', 'top_material_3']
all_materials = pd.Series(df[material_cols].values.ravel()).dropna()
material_counts = all_materials.value_counts()
st.bar_chart(material_counts)

# Plot material usage over time
st.subheader("Monthly Material Usage Trend")
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
material_long = df.melt(id_vars='year_month', value_vars=material_cols, value_name='material').dropna()
monthly_trend = material_long.groupby(['year_month', 'material']).size().unstack().fillna(0)
st.line_chart(monthly_trend)

# ----- Prediction Section -----
st.header("🔮 Predict Future Material Usage")

month_choice = st.selectbox("Select Month", list(range(1, 13)))
year_choice = st.selectbox("Select Year", [2025, 2026])

# Prepare for prediction
filtered = df.dropna(subset=['timestamp'])
filtered['month'] = filtered['timestamp'].dt.month
filtered['year'] = filtered['timestamp'].dt.year

# Use past year's data for prediction
relevant = filtered[(filtered['month'] == month_choice) & (filtered['year'] == year_choice - 1)]
past_materials = pd.Series(relevant[material_cols].values.ravel()).dropna()
predicted = Counter(past_materials).most_common(3)

st.subheader("📌 Predicted Top Materials")
if predicted:
    for material, count in predicted:
        st.write(f"🔹 {material} ({count} selections)")
else:
    st.write("⚠️ Not enough historical data for this selection.")

# ----- Footer -----
st.markdown("---")
st.caption("Streamlit app powered by Snowflake and frequency-based material predictions.")
