import streamlit as st

# Page config
st.set_page_config(
    page_title="About the Project",
    page_icon="ℹ️",
    layout="wide"
)
st.title("About the Project")

st.markdown(
    """
    <div style="background-color:#ffffff; padding:20px; border-radius:12px; border:1px solid #ddd;">
        <h3 style="color:#2c3e50;">Heart Disease Prediction</h3>
        <p style="color:#34495e; font-size:16px;">
            This project applies machine learning to predict the likelihood of heart disease
            based on patient data such as age, cholesterol, blood pressure, and other
            clinical features.
        </p>
        <p style="color:#34495e; font-size:16px;">
            It demonstrates:
        </p>
        <ul style="color:#2c3e50; font-size:15px;">
            <li>Data preprocessing & feature engineering</li>
            <li>Model training and evaluation</li>
            <li>Deployment with Streamlit</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="background-color:#ffffff; padding:20px; border-radius:12px; border:1px solid #ddd; margin-top:20px;">
        <h3 style="color:#2c3e50;">Technologies Used</h3>
        <ul style="color:#2c3e50; font-size:15px;">
            <li>Python, Pandas, NumPy for data manipulation</li>
            <li>Scikit-learn for machine learning</li>
            <li>Matplotlib, Seaborn for visualizations</li>
            <li>Streamlit for web app deployment</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="background-color:#ffffff; padding:20px; border-radius:12px; border:1px solid #ddd; margin-top:20px;">
        <h3 style="color:#2c3e50;">Data Source</h3>
        <p style="color:#34495e; font-size:16px;">
            The dataset used is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.
            It contains clinical data for patients, including whether they have heart disease.
        </p>
        <a href="https://archive.ics.uci.edu/ml/datasets/heart+Disease" target="_blank" style="color:#2980b9; font-size:16px; text-decoration:none; font-weight:bold; margin-top:10px; display:inline-block;">
            View Dataset
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
