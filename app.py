import streamlit as st
import pandas as pd
import snowflake.connector

# Snowflake connection
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
    df = pd.read_sql(query, conn)
    conn.close()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    return df

# Streamlit UI
st.title("Sapphire Price Data Viewer")

# Load and show data
try:
    df = get_data_from_snowflake()
    st.success("Successfully fetched data from Snowflake!")
    st.dataframe(df)
except Exception as e:
    st.error(f"Error fetching data: {e}")
