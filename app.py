import streamlit as st
import snowflake.connector
import pandas as pd

# Hardcoded Snowflake credentials
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",
        schema="PUBLIC"
    )

    query = "SELECT * FROM SAPPHIRE_PRICE"
    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    return df

# Streamlit UI
st.set_page_config(page_title="Gemstone Data Viewer", layout="wide")
st.title("💎 View All Gemstone Prices from Snowflake")

if st.button("Load Data"):
    try:
        df = get_data_from_snowflake()
        if df.empty:
            st.warning("No data found in table.")
        else:
            st.success(f"Loaded {len(df)} rows from Snowflake.")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
