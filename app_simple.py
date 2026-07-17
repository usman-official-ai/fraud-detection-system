import streamlit as st
import requests
import json

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("🛡️ Credit Card Fraud Detection System")

api_url = st.text_input("API Endpoint", "http://127.0.0.1:8000/predict")

if st.button("🔌 Test API"):
    try:
        response = requests.get(api_url.replace("/predict", "/health"), timeout=5)
        if response.status_code == 200:
            st.success("✅ API is running!")
        else:
            st.error(f"❌ Error: {response.status_code}")
    except:
        st.error("❌ Cannot connect")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input("Time", value=0.0)
    v1 = st.number_input("V1", value=-1.359807, format="%.6f")
    v2 = st.number_input("V2", value=-0.072781, format="%.6f")
    v3 = st.number_input("V3", value=2.536347, format="%.6f")
    v4 = st.number_input("V4", value=1.378155, format="%.6f")
    v5 = st.number_input("V5", value=-0.338321, format="%.6f")
    v6 = st.number_input("V6", value=0.462388, format="%.6f")
    v7 = st.number_input("V7", value=0.239599, format="%.6f")
    v8 = st.number_input("V8", value=0.098698, format="%.6f")
    v9 = st.number_input("V9", value=0.363787, format="%.6f")
    v10 = st.number_input("V10", value=0.090794, format="%.6f")

with col2:
    v11 = st.number_input("V11", value=-0.551600, format="%.6f")
    v12 = st.number_input("V12", value=-0.617801, format="%.6f")
    v13 = st.number_input("V13", value=-0.991390, format="%.6f")
    v14 = st.number_input("V14", value=-0.311169, format="%.6f")
    v15 = st.number_input("V15", value=1.468177, format="%.6f")
    v16 = st.number_input("V16", value=-0.470401, format="%.6f")
    v17 = st.number_input("V17", value=0.207971, format="%.6f")
    v18 = st.number_input("V18", value=0.025791, format="%.6f")
    v19 = st.number_input("V19", value=0.403993, format="%.6f")
    v20 = st.number_input("V20", value=0.251412, format="%.6f")

with col3:
    v21 = st.number_input("V21", value=-0.018307, format="%.6f")
    v22 = st.number_input("V22", value=0.277838, format="%.6f")
    v23 = st.number_input("V23", value=-0.110474, format="%.6f")
    v24 = st.number_input("V24", value=0.066928, format="%.6f")
    v25 = st.number_input("V25", value=0.128539, format="%.6f")
    v26 = st.number_input("V26", value=-0.189115, format="%.6f")
    v27 = st.number_input("V27", value=0.133558, format="%.6f")
    v28 = st.number_input("V28", value=-0.021053, format="%.6f")
    amount = st.number_input("Amount", value=149.62, format="%.2f")

if st.button("🔍 Predict", type="primary"):
    # Create features list in correct order
    features_list = [
        float(time), float(v1), float(v2), float(v3), float(v4),
        float(v5), float(v6), float(v7), float(v8), float(v9),
        float(v10), float(v11), float(v12), float(v13), float(v14),
        float(v15), float(v16), float(v17), float(v18), float(v19),
        float(v20), float(v21), float(v22), float(v23), float(v24),
        float(v25), float(v26), float(v27), float(v28), float(amount)
    ]
    
    data = {"features": features_list}
    
    try:
        response = requests.post(api_url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Result: {'FRAUD' if result['is_fraud'] else 'SAFE'}")
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
            st.metric("Risk Score", f"{result['risk_score']:.2f}")
            st.metric("Processing Time", f"{result['processing_time_ms']:.2f} ms")
            with st.expander("📋 Full Response"):
                st.json(result)
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"❌ Error: {e}")