import streamlit as st
import pandas as pd

def show(df):
    st.title("🧹 Data Preprocessing")

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # ---- Missing values ----
    st.subheader("❗ Missing Values")

    missing = df.isnull().sum().sort_values(ascending=False)
    st.bar_chart(missing)

    st.markdown("""
    ✔ Numerical → Median  
    ✔ Categorical → Mode  
    ✔ Delete rows with too many missing values  
    ✔ Encoding → OneHot  (TODO: feature engineering ? )
    ✔ Scaling → StandardScaler  (TODO: feature engineering ? ) 
    """)

    # ---- Apply preprocessing ----
    if st.checkbox("Apply preprocessing"):

        df_clean = df.copy()

        # 示例处理
        for col in df_clean.select_dtypes(include=['float64', 'int64']):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        for col in df_clean.select_dtypes(include=['object']):
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        st.success("Preprocessing applied ✅")

        st.subheader("📊 Cleaned Data")
        st.write(df_clean.head())



