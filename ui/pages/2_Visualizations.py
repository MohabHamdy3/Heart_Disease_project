# ui/pages/2_Visualizations.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Dataset Visualizations",
    page_icon="📊",
    layout="wide"
)
st.title("Dataset Visualizations")

@st.cache_data
def load_data():
    return pd.read_csv("data/heart_disease_cleaned.csv")

df = load_data()

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

st.subheader("Feature Distributions")
selected_feature = st.selectbox("Select feature", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax)
st.pyplot(fig)
