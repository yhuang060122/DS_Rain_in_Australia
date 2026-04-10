import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Airbnb Pricing Dashboard", layout="wide")

# Title
st.title("🏠 Airbnb Pricing Analysis Dashboard")
st.markdown("Analyse exploratoire et prédiction des prix Airbnb à Bordeaux")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("airbnb_bordeaux.csv")
    df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filtres")

room_type = st.sidebar.multiselect(
    "Type de logement",
    options=df["room_type"].dropna().unique(),
    default=df["room_type"].dropna().unique()
)

max_price = st.sidebar.slider(
    "Prix maximum",
    int(df["price"].min()),
    int(df["price"].max()),
    int(df["price"].quantile(0.95))
)

# Apply filters
filtered_df = df[
    (df["room_type"].isin(room_type)) &
    (df["price"] <= max_price)
]

# KPI Section
st.subheader("📊 KPI clés")
col1, col2, col3 = st.columns(3)

col1.metric("Prix moyen", f"{filtered_df['price'].mean():.0f} €")
col2.metric("Prix médian", f"{filtered_df['price'].median():.0f} €")
col3.metric("Nb logements", filtered_df.shape[0])

# Distribution
st.subheader("💰 Distribution des prix")
fig, ax = plt.subplots()
sns.histplot(filtered_df["price"], bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Price by room type
st.subheader("🛏️ Prix par type de logement")
fig, ax = plt.subplots()
sns.boxplot(x="room_type", y="price", data=filtered_df, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

# Scatter
st.subheader("👥 Capacité vs Prix")
fig, ax = plt.subplots()
sns.scatterplot(x="accommodates", y="price", data=filtered_df, ax=ax)
st.pyplot(fig)

# Correlation
st.subheader("🔗 Corrélations")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Simple prediction
st.subheader("🤖 Simulation de prix")

accommodates = st.slider("Capacité", 1, 10, 2)
bedrooms = st.slider("Chambres", 0, 5, 1)
bathrooms = st.slider("Salles de bain", 0, 3, 1)

# simple heuristic model
pred_price = 30 + accommodates * 20 + bedrooms * 15 + bathrooms * 10

st.success(f"💡 Prix estimé : {pred_price} €")

# Raw data
st.subheader("📄 Données brutes")
st.dataframe(filtered_df.head(100))

st.markdown("---")
st.markdown("✅ Dashboard prêt pour démonstration en soutenance")
