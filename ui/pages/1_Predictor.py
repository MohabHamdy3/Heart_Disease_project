# ui/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("Heart Disease Risk Predictor")
st.write("Enter patient information in the sidebar and press **Predict**. The app uses a pre-trained Random Forest pipeline.")

# ---- Load model and feature schema ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/final_model.pkl")
        schema = joblib.load("models/feature_schema.pkl")
        return model, schema
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model, schema = load_model()
NUM_FEATURES = schema["num"]
CAT_FEATURES = schema["cat"]

# ---- Sidebar inputs ----
st.sidebar.header("Patient Information")

def user_input_form():
    data = {}

    # ----------------- Personal Info -----------------
    st.sidebar.subheader("Personal Information")
    data["age"] = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=55)
    sex = st.sidebar.radio("Sex", ["Female", "Male"], index=1)
    data["sex"] = 1 if sex == "Male" else 0

    # ----------------- Medical Test Results -----------------
    st.sidebar.subheader("Medical Test Results")
    data["trestbps"] = st.sidebar.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=130)
    data["chol"] = st.sidebar.number_input("Serum cholesterol (mg/dl)", min_value=50, max_value=600, value=246)
    data["thalach"] = st.sidebar.number_input("Max heart rate achieved", min_value=50, max_value=250, value=150)
    data["oldpeak"] = st.sidebar.number_input("ST depression (exercise-induced)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # ----------------- Symptoms & ECG -----------------
    st.sidebar.subheader("Symptoms & ECG")

    # Chest pain type → one-hot encoding
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical angina (0)", "Atypical angina (1)", "Non-anginal pain (2)", "Asymptomatic (3)"],
        index=1
    )
    cp_map = {"Typical angina (0)": 0, "Atypical angina (1)": 1, "Non-anginal pain (2)": 2, "Asymptomatic (3)": 3}
    cp_val = cp_map[cp]
    data["cp_2.0"] = 1 if cp_val == 2 else 0
    data["cp_3.0"] = 1 if cp_val == 3 else 0
    data["cp_4.0"] = 1 if cp_val == 4 else 0   # include if schema has cp_4.0

    # Fasting blood sugar
    fbs = st.sidebar.radio("Fasting blood sugar > 120 mg/dl", ["No", "Yes"], index=0)
    data["fbs"] = 1 if fbs == "Yes" else 0

    # Resting ECG → one-hot encoding
    restecg = st.sidebar.selectbox(
        "Resting ECG results",
        ["Normal (0)", "ST-T abnormality (1)", "Left ventricular hypertrophy (2)"],
        index=0
    )
    restecg_map = {"Normal (0)": 0, "ST-T abnormality (1)": 1, "Left ventricular hypertrophy (2)": 2}
    rest_val = restecg_map[restecg]
    data["restecg_1.0"] = 1 if rest_val == 1 else 0
    data["restecg_2.0"] = 1 if rest_val == 2 else 0

    # Exercise induced angina
    exang = st.sidebar.radio("Exercise induced angina", ["No", "Yes"], index=0)
    data["exang"] = 1 if exang == "Yes" else 0

    # Slope of ST segment → one-hot encoding
    slope = st.sidebar.selectbox(
        "Slope of peak exercise ST segment",
        ["Upsloping (0)", "Flat (1)", "Downsloping (2)"],
        index=1
    )
    slope_map = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
    slope_val = slope_map[slope]
    data["slope_2.0"] = 1 if slope_val == 2 else 0
    data["slope_3.0"] = 1 if slope_val == 3 else 0

    # Thalium stress test → one-hot encoding
    thal = st.sidebar.selectbox(
        "Thalium stress test",
        ["Normal (3)", "Fixed defect (6)", "Reversible defect (7)"],
        index=0
    )
    thal_map = {"Normal (3)": 3, "Fixed defect (6)": 6, "Reversible defect (7)": 7}
    thal_val = thal_map[thal]
    data["thal_6.0"] = 1 if thal_val == 6 else 0
    data["thal_7.0"] = 1 if thal_val == 7 else 0

    return data

input_data = user_input_form()

# Convert to dataframe in required feature order
def make_input_df(input_data, num_features, cat_features):
    df_in = pd.DataFrame([input_data])
    cols = num_features + cat_features
    # fill missing columns with 0 (e.g. if cp=0, all cp_2.0, cp_3.0, cp_4.0 are 0)
    for col in cols:
        if col not in df_in:
            df_in[col] = 0
    df_in = df_in[cols]
    return df_in

input_df = make_input_df(input_data, NUM_FEATURES, CAT_FEATURES)

st.subheader("Input summary")
st.table(input_df.T.rename(columns={0:"value"}))

# ---- Predict button ----
if st.button("Predict risk"):
    with st.spinner("Predicting..."):
        proba = model.predict_proba(input_df)[:,1][0]  # probability of class 1
        pred = int(proba >= 0.5)
        st.metric("Predicted class", "Disease" if pred==1 else "No disease", delta=f"{proba*100:.1f}% prob")
        st.write(f"**Probability of heart disease:** {proba:.3f}")

        # Show short guidance
        if proba >= 0.7:
            st.warning("High predicted risk — recommend clinical evaluation.")
        elif proba >= 0.4:
            st.info("Moderate risk — consider follow-up tests and physician review.")
        else:
            st.success("Low predicted risk.")

        # Feature importances (only works if underlying model is tree-based)
        try:
            # get feature names after preprocessing
            pre = model.named_steps["preprocessor"]
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
            feature_names = NUM_FEATURES + cat_names

            importances = model.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=True)

            st.subheader("Feature importance (model)")
            fig, ax = plt.subplots(figsize=(6, max(4, len(imp_df)*0.25)))
            ax.barh(imp_df["feature"], imp_df["importance"])
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        except Exception as e:
            st.info("Feature importance not available for this model or preprocessing.")

# ---- Predict button in the sidebar ----
st.sidebar.markdown("---")
if st.sidebar.button("Predict risk"):
    with st.spinner("Predicting..."):
        proba = model.predict_proba(input_df)[:, 1][0]  # probability of class 1
        pred = int(proba >= 0.5)

        # Show prediction in main area
        st.subheader("Prediction Result")
        st.metric("Predicted class", "Disease" if pred == 1 else "No disease", delta=f"{proba*100:.1f}% prob")
        st.write(f"**Probability of heart disease:** {proba:.3f}")

        # Show short guidance
        if proba >= 0.7:
            st.warning("High predicted risk — recommend clinical evaluation.")
        elif proba >= 0.4:
            st.info("Moderate risk — consider follow-up tests and physician review.")
        else:
            st.success("Low predicted risk.")

        # Feature importances (only works if underlying model is tree-based)
        try:
            pre = model.named_steps["preprocessor"]
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
            feature_names = NUM_FEATURES + cat_names

            importances = model.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=True)

            st.subheader("Feature importance (model)")
            fig, ax = plt.subplots(figsize=(6, max(4, len(imp_df)*0.25)))
            ax.barh(imp_df["feature"], imp_df["importance"])
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        except Exception:
            st.info("Feature importance not available for this model or preprocessing.")
# ---- Quick visualization (distribution) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Quick visualizations")
if st.sidebar.button("Show dataset distribution (sample)"):


    # load original cleaned df to show distributions
    try:
        df_clean = pd.read_csv("data/heart_disease_cleaned.csv")
    except:
        df_clean = None
    if df_clean is None:
        st.error("Cleaned dataset not found in data/heart_disease_cleaned.csv")
    else:
        st.write("Histogram of numeric features (sample):")
        st.write(df_clean[NUM_FEATURES].hist(bins=20, figsize=(10,6)))
        st.write(df_clean[NUM_FEATURES].describe())

        st.write("Histogram of categorical features (sample):")
        st.write(df_clean[CAT_FEATURES].hist(bins=10, figsize=(10,6)))
        st.write(df_clean[CAT_FEATURES].describe())