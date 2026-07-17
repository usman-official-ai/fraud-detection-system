import streamlit as st
import pandas as pd
import numpy as np
import time
import random

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🛡️ Credit Card Fraud Detection System")

st.info("""
**⚠️ Note:** This is a demo version running in mock mode. 
The actual ML model is not loaded, but the interface works exactly like the real system.
""")

st.divider()

st.subheader("📝 Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input("Time", value=0.0, format="%.1f")
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

# Prediction function (mock)
def mock_predict():
    """Simulate fraud detection"""
    # Random prediction with bias towards legit
    prob = random.random()
    
    # Make it more realistic - some features indicate fraud
    if v1 < -2.0 and amount > 1000:
        prob = prob * 2  # Higher chance of fraud
    elif amount > 5000:
        prob = prob * 1.5
    elif v1 > 2.0 and v2 > 1.0:
        prob = prob * 1.3
    
    prob = min(prob, 0.99)  # Cap at 99%
    is_fraud = prob > 0.5
    
    return {
        "is_fraud": is_fraud,
        "fraud_probability": prob,
        "risk_score": prob * 100,
        "processing_time_ms": random.uniform(20, 60)
    }

# Buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    predict_btn = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)

with col2:
    sample_btn = st.button("📥 Load Sample", use_container_width=True)

with col3:
    fraud_sample_btn = st.button("⚠️ Load Fraud Sample", use_container_width=True)

# Sample data loading
if sample_btn:
    st.session_state['time'] = 0.0
    st.session_state['v1'] = -1.359807
    st.session_state['v2'] = -0.072781
    st.session_state['v3'] = 2.536347
    st.session_state['v4'] = 1.378155
    st.session_state['v5'] = -0.338321
    st.session_state['v6'] = 0.462388
    st.session_state['v7'] = 0.239599
    st.session_state['v8'] = 0.098698
    st.session_state['v9'] = 0.363787
    st.session_state['v10'] = 0.090794
    st.session_state['v11'] = -0.551600
    st.session_state['v12'] = -0.617801
    st.session_state['v13'] = -0.991390
    st.session_state['v14'] = -0.311169
    st.session_state['v15'] = 1.468177
    st.session_state['v16'] = -0.470401
    st.session_state['v17'] = 0.207971
    st.session_state['v18'] = 0.025791
    st.session_state['v19'] = 0.403993
    st.session_state['v20'] = 0.251412
    st.session_state['v21'] = -0.018307
    st.session_state['v22'] = 0.277838
    st.session_state['v23'] = -0.110474
    st.session_state['v24'] = 0.066928
    st.session_state['v25'] = 0.128539
    st.session_state['v26'] = -0.189115
    st.session_state['v27'] = 0.133558
    st.session_state['v28'] = -0.021053
    st.session_state['amount'] = 149.62
    st.rerun()

if fraud_sample_btn:
    st.session_state['time'] = 406.0
    st.session_state['v1'] = -2.3122265423263
    st.session_state['v2'] = 1.95199201064158
    st.session_state['v3'] = -1.60985073229769
    st.session_state['v4'] = 3.9979055875468
    st.session_state['v5'] = -0.522187864667764
    st.session_state['v6'] = -1.42654531920595
    st.session_state['v7'] = -2.53738730624579
    st.session_state['v8'] = 1.39165724829804
    st.session_state['v9'] = -2.77008927719433
    st.session_state['v10'] = -2.77227214465915
    st.session_state['v11'] = 3.20203320709635
    st.session_state['v12'] = -2.89990738849473
    st.session_state['v13'] = -0.595221881324605
    st.session_state['v14'] = -4.28925378244217
    st.session_state['v15'] = 0.389724120274487
    st.session_state['v16'] = -1.14074717980657
    st.session_state['v17'] = -2.83005567450437
    st.session_state['v18'] = -0.0168224681808257
    st.session_state['v19'] = 0.416955705037907
    st.session_state['v20'] = 0.126910559061474
    st.session_state['v21'] = 0.517232370861764
    st.session_state['v22'] = -0.0350493686052974
    st.session_state['v23'] = -0.465211076182388
    st.session_state['v24'] = 0.320198198514526
    st.session_state['v25'] = 0.0445191674731724
    st.session_state['v26'] = 0.177839798284401
    st.session_state['v27'] = 0.261145002567677
    st.session_state['v28'] = -0.143275874698919
    st.session_state['amount'] = 0.0
    st.rerun()

# Prediction logic
if predict_btn:
    with st.spinner("🔍 Analyzing transaction..."):
        time.sleep(0.5)  # Simulate processing
        result = mock_predict()
        
        st.divider()
        st.subheader("📊 Prediction Result")
        
        # Show fraud status with color
        if result['is_fraud']:
            st.error(f"⚠️ FRAUD DETECTED!")
        else:
            st.success(f"✅ Transaction is Legitimate")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
        with col2:
            st.metric("Risk Score", f"{result['risk_score']:.2f}")
        with col3:
            st.metric("Processing Time", f"{result['processing_time_ms']:.2f} ms")

st.divider()
st.caption("Built with ❤️ using Streamlit, FastAPI, and XGBoost")
st.caption("© 2026 Muhammad Usman | Fraud Detection System")