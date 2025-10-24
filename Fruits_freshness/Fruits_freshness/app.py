import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('fruit_freshness.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍎 Fruit vs Vegetable Freshness AI")
st.write("Predict whether your item is **Fresh or Rotten**!")

# Input features
weight = st.number_input("Weight (grams)", 50, 500, 150)
color_score = st.slider("Color Score", 1.0, 10.0, 5.0)
firmness = st.slider("Firmness", 1.0, 10.0, 5.0)
sweetness = st.slider("Sweetness", 1.0, 10.0, 5.0)
temperature_storage = st.slider("Storage Temperature (°C)", 0.0, 30.0, 10.0)
humidity = st.slider("Humidity (%)", 30.0, 90.0, 60.0)
days_since_harvest = st.slider("Days Since Harvest", 1, 30, 10)
item_type = st.selectbox("Type", ["Fruit", "Vegetable"])

# Convert type to numeric
type_encoded = 0 if item_type == "Fruit" else 1

# Prepare features
features = np.array([[weight, color_score, firmness, sweetness,
                      type_encoded, temperature_storage, humidity, days_since_harvest]])

# Scale and predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]

# Output
result = "Fresh 🍏" if prediction == 1 else "Rotten 🍂"
st.subheader(f"Result: **{result}**")
