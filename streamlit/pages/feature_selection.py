import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def show(df):
    st.title("🎯 Feature Selection")

    st.subheader("📊 Correlation Heatmap")
    num_df = df.select_dtypes(include=['float64', 'int64'])

    corr = num_df.corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🌳 Feature Importance")

    df.head()

    X = df.select_dtypes(include=['float64', 'int64']).drop(
        columns=['RainTomorrow'])
    y = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

    model = RandomForestClassifier()
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    st.bar_chart(importances)

    st.subheader("🔄 ML Pipeline")

    st.code("""
        Preprocessing:
        - Missing values handling
        - Encoding
        - Scaling

        Feature Engineering:
        - TempRange
        - HumidityDiff

        Feature Selection:
        - Correlation filtering
        - Random Forest importance

        Model:
        - Random Forest Classifier
    """)
