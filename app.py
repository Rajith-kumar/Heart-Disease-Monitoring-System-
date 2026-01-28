# app.py
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "This app uses a trained machine learning model to estimate the probability "
    "of heart disease given basic clinical features. "
    "It is **for educational use only** and not medical advice."
)

# Load trained pipeline (preprocessor is inside the saved model pipeline)
MODEL_FILE = "model.joblib"

@st.cache_resource
def load_model():
    try:
        model = load(MODEL_FILE)
    except Exception as e:
        st.error(f"Failed to load '{MODEL_FILE}'. Did you run `python train.py`? Error: {e}")
        st.stop()
    return model

model = load_model()

with st.form("input_form"):
    st.subheader("Enter Patient Data")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=54)
        sex = st.selectbox("Sex", options=[0,1], index=1, help="0 = female, 1 = male")
        cp = st.selectbox("Chest Pain Type (cp)", options=[0,1,2,3], index=2)
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=70, max_value=250, value=130)
        chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=700, value=245)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0,1], index=0)

    with col2:
        restecg = st.selectbox("Resting ECG (restecg)", options=[0,1,2], index=0)
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0,1], index=0)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
        slope = st.selectbox("Slope of Peak Exercise ST (slope)", options=[0,1,2], index=1)
        ca = st.selectbox("Number of Major Vessels (ca)", options=[0,1,2,3,4], index=0)
        thal = st.selectbox("Thal", options=[0,1,2,3], index=2)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    proba = model.predict_proba(input_df)[:, 1][0]
    pred = int(proba >= 0.3)

    st.markdown("### Results")
    st.write(f"**Estimated Probability of Heart Disease:** `{proba:.3f}`")
    st.write(f"**Predicted Class @ 0.30 threshold:** `{pred}` (1 = disease, 0 = no disease)")

    st.info(
        "Thresholds can be adjusted in clinical practice to balance sensitivity vs. specificity. "
        "This demo uses 0.30 by default."
    )

st.markdown("---")
st.caption("Educational demo • Not a medical device • Trained with scikit-learn")
