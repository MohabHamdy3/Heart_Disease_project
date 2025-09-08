# ui/pages/3_Model_Insights.py
import streamlit as st
import joblib
import pandas as pd
import shap
    

# Page config
st.set_page_config(
    page_title="Model Insights",
    page_icon="🔍",
    layout="wide"
)
st.title("Model Insights")

@st.cache_resource
def load_model():
    return joblib.load("models/final_model.pkl")

model = load_model()

st.subheader("Global Feature Importance")
try:
    importances = model.named_steps["clf"].feature_importances_
    st.bar_chart(importances)
except:
    st.warning("Feature importance not available.")

st.subheader("Explain Individual Prediction")
uploaded = st.file_uploader("Upload patient data (CSV)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(df)
    st.write("SHAP summary plot:")
    st.pyplot(shap.summary_plot(shap_values, df, plot_type="bar"))
