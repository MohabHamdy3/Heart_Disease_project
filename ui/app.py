import streamlit as st

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    """
    <div style="background-color:#262730; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:20px;">
        <h2 style="text-align:center;">Heart Disease Prediction App</h2>
        <p style="text-align:center; font-size:16px;">
            Use the sidebar to explore predictions, visualizations, model insights, and project information.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
