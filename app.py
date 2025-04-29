# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Title
st.title("💳 Credit Card Fraud Detection System")

st.markdown("Please enter the transaction details below:")

# Function to load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Input collection
st.subheader("Transaction Details")

transaction_id = st.text_input('🆔 Transaction ID')
customer_id = st.text_input('👤 Customer ID')
transaction_amount = st.number_input('💵 Transaction Amount', min_value=0.0, format="%.2f")
geographical_distance = st.number_input('🌍 Geographical Distance (in km)', min_value=0.0, format="%.2f")
transaction_hour = st.slider('⏰ Transaction Hour (0-23)', 0, 23, 12)
fail_attempts = st.selectbox('❌ Number of Failed Attempts', options=[0, 1, 2, 3], index=0)

# Prepare input as a dictionary (for display)
input_data = {
    'Transaction ID': transaction_id,
    'Customer ID': customer_id,
    'Transaction Amount': transaction_amount,
    'Geographical Distance': geographical_distance,
    'Transaction Hour': transaction_hour,
    'Failed Attempts': fail_attempts
}

# Convert to DataFrame for nice display
input_df = pd.DataFrame([input_data])

# Show the input data
st.subheader("🔎 Your Input Data")
st.dataframe(input_df)

# Prediction button
if st.button("🚀 Predict Fraud"):

    # Prepare the model input exactly in the order your model expects
    model_input = [
        transaction_amount,
        customer_id,
        transaction_id,
        transaction_hour,
        geographical_distance,
        fail_attempts
    ]

    # Important: customer_id and transaction_id are strings, 
    # you might need to encode them if your model was trained on encoded IDs!

    model = load_model()

    # Predict
    prediction = model.predict([model_input])

    # Show result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("🚨 Fraud Detected! This transaction is flagged.")
    else:
        st.success("✅ Transaction appears legitimate.")
