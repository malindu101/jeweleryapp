import streamlit as st
import snowflake.connector
import pandas as pd

# Snowflake connection + query
def get_data_from_snowflake(month, year):
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )

    query = f"""
        SELECT * FROM gem_prices
        WHERE EXTRACT(MONTH FROM TIMESTAMP) = {month}
          AND EXTRACT(YEAR FROM TIMESTAMP) = {year}
    """

    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    return df

# Streamlit UI
st.set_page_config(page_title="Gemstone Data Viewer")
st.title("💠 View Gem Prices from Snowflake")

month = st.selectbox("Select Month", list(range(1, 13)))
year = st.selectbox("Select Year", list(range(2015, 2026)))

if st.button("Fetch Data"):
    try:
        df = get_data_from_snowflake(month, year)
        if df.empty:
            st.warning("No data found for this selection.")
        else:
            st.success("Data loaded successfully from Snowflake.")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
