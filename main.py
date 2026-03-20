import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")

# Sidebar
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.3)

# Inputs
st.subheader("Customer Details")

tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Encode (simple demo)
contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
internet_map = {"DSL":0, "Fiber optic":1, "No":2}

# Prepare input
input_data = np.zeros(len(columns))

feature_map = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

for i, col in enumerate(columns):
    if col in feature_map:
        input_data[i] = feature_map[col]

# Predict
if st.button("Predict Churn"):
    prob = model.predict_proba([input_data])[0][1]
    pred = int(prob > threshold)

    st.subheader("📈 Result")

    st.write(f"Churn Probability: **{prob:.2f}**")

    if pred == 1:
        st.error("⚠️ High Risk Customer → Take retention action!")
    else:
        st.success("✅ Safe Customer")

    # Business insight
    if prob > 0.7:
        st.warning("Offer discount or call customer immediately!")


import pickle
import os

if os.path.exists("columns.pkl") and os.path.getsize("columns.pkl") > 0:
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
else:
    raise ValueError("❌ columns.pkl file empty ya missing hai")

