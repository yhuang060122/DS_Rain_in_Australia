import streamlit as st
import pandas as pd
from utils.preprocessing import feature_engineering

def show(df):
    st.title("⚙️ Feature Engineering")

    st.subheader("➕ Created Features")

    st.code("""
        TempRange = MaxTemp - MinTemp
        HumidityDiff = Humidity3pm - Humidity9am
        Season from Date
    """)


    if st.checkbox("Apply Feature Engineering"):

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month

        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['HumidityDiff'] = df['Humidity3pm'] - df['Humidity9am']

        st.success("Features added 🚀")
        st.write(df.head())




