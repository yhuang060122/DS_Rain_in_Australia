# ==========================================
# STREAMLIT APP - Rain Prediction Australia
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Rain Prediction App", layout="wide")

# ------------------------------------------
# LOAD DATA
# ------------------------------------------  
@st.cache_data
def load_data():
    if not os.path.exists("streamlit/weatherAUS.csv"):
        st.error("Dataset not found. Please check file path.")
        return None
    df = pd.read_csv("streamlit/weatherAUS.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    return df

df = load_data()

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Project Overview",
    "Data Exploration",
    "Prediction"
])

# ------------------------------------------
# PAGE 1 - OVERVIEW
# ------------------------------------------
if page == "Project Overview":
    st.title("🌧 Rain Prediction in Australia")
    
    st.markdown("""
    ### 🎯 Objective
    Predict whether it will rain tomorrow based on weather data.
    
    ### 📊 Dataset
    - 10 years of daily weather observations
    - Multiple Australian locations
    - Target: RainTomorrow (Yes/No)
    
    ### 💡 Business Value
    - Agriculture planning
    - Transport optimization
    - Weather risk management
    """)

# ------------------------------------------
# PAGE 2 - EDA
# ------------------------------------------
if page == "Data Exploration":
    st.title("📊 Data Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='RainTomorrow', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Numerical Distribution")
    num_col = st.selectbox("Select a numerical feature", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[num_col], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), ax=ax)
    st.pyplot(fig)

# ------------------------------------------
# PAGE 3 - PREDICTION
# ------------------------------------------
if page == "Prediction":
    st.title("🤖 Rain Prediction")

    st.markdown("Enter weather conditions:")

    humidity = st.slider("Humidity (3pm)", 0, 100, 50)
    pressure = st.slider("Pressure (3pm)", 980, 1050, 1010)
    temp = st.slider("Temperature (3pm)", -5, 50, 25)
    wind = st.slider("Wind Speed", 0, 100, 20)

    # Dummy prediction (replace with real model)
    if st.button("Predict"):
        score = humidity * 0.4 - pressure * 0.01 + wind * 0.2
        if score > 20:
            st.error("🌧 It will likely rain tomorrow")
        else:
            st.success("☀️ No rain expected")

    st.warning("⚠️ Demo prediction (replace with trained model)")

# ==========================================
# END APP
# ==========================================
