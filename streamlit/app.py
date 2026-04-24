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

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from model import build_model
from preprocessing import feature_engineering 

# ------------------------------------------
# CONFIG
# ------------------------------------------
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

@st.cache_resource
def load_model(df):
    return build_model(df)

model, feature_names = load_model(df)


# ------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📊 Data Exploration",
    "⚙️ Feature Engineering",
    "🎯 Feature Selection",
    "🤖 Model",
    "🌧️ Prediction"
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

    st.dataframe(df.head())

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


# ==========================================
# PAGE 2 - PREPROCESSING
# ==========================================
elif page == "🧹 Preprocessing":
    st.title("🧹 Data Preprocessing")

    st.subheader("Missing Values")
    st.bar_chart(df.isnull().sum())

    st.markdown("""
    ✔ Numerical → Median  
    ✔ Categorical → Mode  
    ✔ Encoding → OneHot  
    ✔ Scaling → StandardScaler  
    """)

# ==========================================
# PAGE 3 - FEATURE ENGINEERING
# ==========================================
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")

    st.code("""
TempRange = MaxTemp - MinTemp
HumidityDiff = Humidity3pm - Humidity9am
""")

    df_fe = feature_engineering(df)
    st.dataframe(df_fe.head())

# ==========================================
# PAGE 4 - FEATURE SELECTION
# ==========================================
elif page == "🎯 Feature Selection":
    st.title("🎯 Feature Importance")

    df_fe = feature_engineering(df).dropna()

    X = df_fe.drop(columns=['RainTomorrow'])
    y = df_fe['RainTomorrow']

    rf = RandomForestClassifier()
    rf.fit(X.select_dtypes(include=np.number), y)

    importances = pd.Series(
        rf.feature_importances_,
        index=X.select_dtypes(include=np.number).columns
    ).sort_values(ascending=False)

    st.bar_chart(importances)

# ==========================================
# PAGE 5 - MODEL
# ==========================================
elif page == "🤖 Model":
    st.title("🤖 Model Pipeline")

    st.code("""
Preprocessing:
- Imputation
- Encoding
- Scaling

Feature Engineering:
- TempRange
- HumidityDiff

Model:
- Random Forest (balanced)
""")

# ==========================================
# PAGE 6 - PREDICTION
# ==========================================
elif page == "🌧️ Prediction":
    st.title("🌧️ Will it rain tomorrow?")

    col1, col2 = st.columns(2)

    with col1:
        MinTemp = st.number_input("MinTemp")
        MaxTemp = st.number_input("MaxTemp")
        Humidity9am = st.slider("Humidity 9am", 0, 100)
        Humidity3pm = st.slider("Humidity 3pm", 0, 100)

    with col2:
        Pressure9am = st.number_input("Pressure 9am")
        Pressure3pm = st.number_input("Pressure 3pm")
        RainToday = st.selectbox("Rain Today", ["No", "Yes"])

    input_df = pd.DataFrame({
        'MinTemp':[MinTemp],
        'MaxTemp':[MaxTemp],
        'Humidity9am':[Humidity9am],
        'Humidity3pm':[Humidity3pm],
        'Pressure9am':[Pressure9am],
        'Pressure3pm':[Pressure3pm],
        'RainToday':[1 if RainToday=="Yes" else 0]
    })

    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"🌧️ Rain Tomorrow (probability {proba:.2f})")
            else:
                st.success(f"☀️ No Rain (probability {proba:.2f})")

        except Exception as e:
            st.warning("⚠️ Please fill all inputs correctly")




# ==========================================
# END APP
# ==========================================
