import streamlit as st
import pandas as pd
import joblib, json

# Load model and feature names
rf = joblib.load("/content/phish-login-detector/models/phishing_model.pkl")
with open("/content/phish-login-detector/models/feature_names.json", "r") as f:
    features = json.load(f)

st.set_page_config(page_title="Phishing Email Detector", layout="wide")

st.title(" AI-Powered Phishing Email & Login Anomaly Detector")
st.write("Enter the email characteristics below to analyze whether it looks phishing or legitimate:")

# Create input fields dynamically
user_input = {}
cols = st.columns(5)
for i, feature in enumerate(features):
    col = cols[i % 5]
    user_input[feature] = col.number_input(f"{feature.replace('_', ' ').title()}", min_value=0, value=1)

# Convert to DataFrame in correct order
sample_df = pd.DataFrame([user_input])[features]

if st.button(" Analyze Email"):
    try:
        pred = rf.predict(sample_df)[0]
        prob = rf.predict_proba(sample_df)[0][pred]
        if pred == 1:
            st.error(f"⚠ The email looks PHISHING! (probability: {prob:.2f})")
        else:
            st.success(f" The email looks LEGITIMATE! (probability: {prob:.2f})")
    except Exception as e:
        st.exception(e)
