import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether the transaction is genuine or fraudulent.")

# Number of features in your dataset
feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

# Create input fields
input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(
        feature,
        value=0.0
    )

# Prediction button
if st.button("🔍 Detect Fraud"):

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale Amount
    input_df["Amount"] = scaler.transform(
        input_df[["Amount"]]
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")

    st.write(
        f"Fraud Probability: *{probability:.2%}*"
    )