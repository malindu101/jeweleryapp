import streamlit as st
import snowflake.connector
import pandas as pd

def get_data_from_snowflake(month, year):
    creds = st.secrets["snowflake"]
    conn = snowflake.connector.connect(
        user=creds["MOW101"],
        password=creds["Killme@20021128123123"],
        account=creds["KWLEACZ-DX82931"],
        warehouse=creds["COMPUTE_WH"],
        database=creds["SAPPHIRE"],
        schema=creds["PUBLIC"]
    )

    query = f"""
        SELECT * FROM GEM_DATA
        WHERE MONTH = {month} AND YEAR = {year};
    """
    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    return df
