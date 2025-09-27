import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load model and scaler
# -------------------------------
with open("notebooks/gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("notebooks/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Crypto Price Prediction")

# -------------------------------
# Input fields
# -------------------------------
one_h = st.number_input("1h Change (%)", min_value=-100.0, max_value=100.0, step=0.01)
twentyfour_h = st.number_input("24h Change (%)", min_value=-100.0, max_value=100.0, step=0.01)
seven_d = st.number_input("7d Change (%)", min_value=-100.0, max_value=100.0, step=0.01)
vol_24h = st.number_input("24h Volume", min_value=0.0, step=1.0)
market_cap = st.number_input("Market Cap", min_value=0.0, step=1.0)

# -------------------------------
# Make prediction when button is clicked
# -------------------------------
if st.button("Predict"):
    
    X_input = np.array([[one_h, twentyfour_h, seven_d, vol_24h, market_cap]])
   

    price_pred = model.predict(X_scaled)
    try:
        price_pred = np.expm1(price_pred)  
    except:
        pass

    st.success(f"Predicted Price: {price_pred[0]:,.2f}")
