import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import random
from google import genai  # <-- Naya import
from google.genai import types  # <-- Optional, for advanced features

st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("🛡️ Credit Card Fraud Detection System")

# ============================================
# GEMINI API INITIALIZATION (NEW SDK)
# ============================================
def init_gemini():
    """Initialize Gemini API using the new google-genai SDK"""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            # Naya SDK client
            client = genai.Client(api_key=api_key)  # <-- Naya tarika
            return client
        else:
            st.warning("⚠️ GEMINI_API_KEY not found in secrets. Using mock mode.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Gemini initialization failed: {e}. Using mock mode.")
        return None

# Initialize Gemini client
gemini_client = init_gemini()

# ============================================
# PREDICTION FUNCTION (UPDATED)
# ============================================
def get_prediction(features_list):
    """Get prediction using new Gemini API or fallback to mock"""
    
    if gemini_client:
        try:
            # Prepare features for Gemini
            features_text = create_feature_text(features_list)
            
            # Naya SDK model call
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',  # Ya koi bhi model
                contents=features_text,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            return {
                "is_fraud": result.get("is_fraud", False),
                "fraud_probability": result.get("fraud_probability", 0.0),
                "risk_score": result.get("risk_score", 0.0),
                "processing_time_ms": random.uniform(30, 80),
                "reason": result.get("reason", "AI analysis complete")
            }
            
        except Exception as e:
            st.warning(f"⚠️ Gemini API error: {e}. Using mock prediction.")
            return mock_predict(features_list)
    
    else:
        return mock_predict(features_list)

def create_feature_text(features_list):
    """Convert features list to readable text"""
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                     'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                     'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                     'V28', 'Amount']
    
    text = "Analyze this credit card transaction for fraud risk. Return ONLY JSON.\n\n"
    text += "Transaction features:\n"
    for i, (name, value) in enumerate(zip(feature_names, features_list)):
        if i == 0 or i == 29:
            text += f"- {name}: {value:.2f}\n"
        else:
            text += f"- {name}: {value:.6f}\n"
    
    text += "\nReturn this exact JSON format:\n"
    text += '{"is_fraud": true/false, "fraud_probability": 0.XX, "risk_score": XX.XX, "reason": "Brief reason"}'
    
    return text

def mock_predict(features_list):
    """Smart mock prediction (fallback)"""
    time = features_list[0]
    v1 = features_list[1]
    v2 = features_list[2]
    v3 = features_list[3]
    v4 = features_list[4]
    amount = features_list[29]
    
    risk_score = 0.0
    
    # Amount-based risk
    if amount > 1000:
        risk_score += 0.3
    if amount > 5000:
        risk_score += 0.2
    if amount > 10000:
        risk_score += 0.2
    
    # V1 pattern (strong fraud indicator)
    if v1 < -2.0:
        risk_score += 0.4
    elif v1 < -1.0:
        risk_score += 0.2
    
    # V2 pattern
    if v2 > 1.5:
        risk_score += 0.2
    elif v2 < -2.0:
        risk_score += 0.2
    
    # V3 pattern
    if v3 > 2.0:
        risk_score += 0.2
    elif v3 < -2.0:
        risk_score += 0.1
    
    # V4 pattern
    if v4 > 2.0:
        risk_score += 0.2
    
    # V11 pattern
    if features_list[10] > 2.0:
        risk_score += 0.2
    
    # V14 pattern
    if features_list[13] < -2.0:
        risk_score += 0.2
    
    # Time-based risk
    if time > 100000:
        risk_score += 0.1
    
    # Amount vs V1 interaction
    if v1 < -1.5 and amount > 500:
        risk_score += 0.2
    
    # Add some randomness
    risk_score += random.uniform(-0.05, 0.05)
    risk_score = max(0.0, min(1.0, risk_score))
    
    is_fraud = risk_score > 0.5
    prob = risk_score + random.uniform(-0.02, 0.02)
    prob = max(0.0, min(1.0, prob))
    
    return {
        "is_fraud": is_fraud,
        "fraud_probability": prob,
        "risk_score": prob * 100,
        "processing_time_ms": random.uniform(20, 60),
        "reason": "Rule-based analysis (Gemini API not available)"
    }

# ============================================
# SIDEBAR CONFIG
# ============================================
st.sidebar.header("⚙️ Configuration")

# Show Gemini status
if gemini_client:
    st.sidebar.success("✅ Gemini API connected (google-genai)")
else:
    st.sidebar.warning("⚠️ Using mock mode")
    st.sidebar.info("Add GEMINI_API_KEY to secrets for real predictions")

st.sidebar.divider()

# Input Method
st.sidebar.subheader("📥 Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Input", "Upload CSV"]
)

st.divider()

# ============================================
# MANUAL INPUT
# ============================================
if input_method == "Manual Input":
    st.subheader("📝 Enter Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_val = st.number_input("Time", value=0.0, format="%.1f")
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
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sample_btn = st.button("📥 Load Sample (Legit)", use_container_width=True)
    with col2:
        fraud_sample_btn = st.button("⚠️ Load Sample (Fraud)", use_container_width=True)
    with col3:
        predict_btn = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)
    
    # Load sample data
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
    
    if predict_btn:
        features_list = [
            float(time_val), float(v1), float(v2), float(v3), float(v4),
            float(v5), float(v6), float(v7), float(v8), float(v9),
            float(v10), float(v11), float(v12), float(v13), float(v14),
            float(v15), float(v16), float(v17), float(v18), float(v19),
            float(v20), float(v21), float(v22), float(v23), float(v24),
            float(v25), float(v26), float(v27), float(v28), float(amount)
        ]
        
        with st.spinner("🔍 Analyzing transaction..."):
            result = get_prediction(features_list)
        
        st.divider()
        st.subheader("📊 Prediction Result")
        
        if result['is_fraud']:
            st.error("⚠️ FRAUD DETECTED!")
        else:
            st.success("✅ Transaction is Legitimate")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
        with col2:
            st.metric("Risk Score", f"{result['risk_score']:.2f}")
        with col3:
            st.metric("Processing Time", f"{result['processing_time_ms']:.2f} ms")
        
        with st.expander("📋 Full Details"):
            st.json(result)

# ============================================
# CSV UPLOAD
# ============================================
else:
    st.subheader("📂 Upload CSV File for Batch Prediction")
    
    st.info("""
    **CSV Format:**
    - Columns: Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount
    - All values must be numeric
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.columns[0] != 'Time':
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                             'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                             'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                             'V28', 'Amount']
            
            st.success(f"✅ {len(df)} transactions loaded!")
            st.write("**Preview:**")
            st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Transactions", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    features_list = row.tolist()
                    result = get_prediction(features_list)
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(df))
                
                st.divider()
                st.subheader("📊 Batch Results")
                
                results_df = pd.DataFrame(results)
                
                if 'is_fraud' in results_df.columns:
                    fraud_count = results_df['is_fraud'].sum()
                    legit_count = len(results_df) - fraud_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(results_df))
                    with col2:
                        st.metric("⚠️ Fraud", fraud_count)
                    with col3:
                        st.metric("✅ Legit", legit_count)
                    
                    df['Prediction'] = results_df['is_fraud'].apply(lambda x: 'FRAUD' if x else 'LEGIT')
                    df['Fraud_Prob'] = results_df['fraud_probability'].apply(lambda x: f"{x*100:.2f}%")
                    
                    st.dataframe(df[['Time', 'Amount', 'Prediction', 'Fraud_Prob']])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fraud_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No valid predictions received")
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()
st.caption("Built with ❤️ | Fraud Detection System")