import streamlit as st
import pandas as pd
import snowflake.connector
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Streamlit page setup
st.set_page_config(page_title="Gem Color Predictor", layout="centered")
st.title("💎 Gem Color Prediction App")

# Function to load data from Snowflake
@st.cache_data
def get_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="MOW101",
        password="Killme@20021128123123",
        account="KWLEACZ-DX82931",
        warehouse="COMPUTE_WH",
        database="SAPPHIRE",         # Same DB
        schema="PUBLIC"
    )
    df = pd.read_sql("SELECT * FROM STOCK", conn)
    conn.close()
    return df

# Function to train the model
@st.cache_resource
def train_model():
    df = get_data_from_snowflake()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df.dropna(subset=['created_at', 'gem_color'], inplace=True)
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['hour'] = df['created_at'].dt.hour
    df['weekday'] = df['created_at'].dt.weekday

    # Encode categorical variables
    material_le = LabelEncoder()
    color_le = LabelEncoder()
    df['material_encoded'] = material_le.fit_transform(df['material'].astype(str))
    df['gem_color_encoded'] = color_le.fit_transform(df['gem_color'])

    X = df[['year', 'month', 'hour', 'weekday', 'material_encoded']]
    y = df['gem_color_encoded']

    model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.2,
                          use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    return model, material_le, color_le

# Load trained model and encoders
model, material_le, color_le = train_model()

# Form for user input
with st.form("predict_form"):
    st.subheader("📅 Enter Order Details to Predict Gem Color")

    date_input = st.date_input("Order Date")
    time_input = st.time_input("Order Time")
    material_input = st.selectbox("Material", material_le.classes_)

    submitted = st.form_submit_button("Predict Gem Color")

# Prediction
if submitted:
    year = date_input.year
    month = date_input.month
    weekday = date_input.weekday()
    hour = time_input.hour
    material_encoded = material_le.transform([material_input])[0]

    input_data = pd.DataFrame([[year, month, hour, weekday, material_encoded]],
                              columns=['year', 'month', 'hour', 'weekday', 'material_encoded'])

    prediction = model.predict(input_data)[0]
    predicted_color = color_le.inverse_transform([prediction])[0]

    st.success(f"🎯 Predicted Gem Color: **{predicted_color}**")
