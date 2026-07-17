import streamlit as st
import requests
import pandas as pd
import json
import time

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("🛡️ Credit Card Fraud Detection System")

# Sidebar
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input(
    "API Endpoint",
    value="http://127.0.0.1:8000/predict",
    help="URL of the FastAPI backend"
)

# Input Method
st.sidebar.subheader("📥 Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Input", "Upload CSV", "Use Sample"]
)

# Test API Connection
if st.sidebar.button("🔌 Test API"):
    try:
        health_url = api_url.replace("/predict", "/health")
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ API is running!")
        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")
    except:
        st.sidebar.error("❌ Cannot connect to API")

st.divider()

# ============================================
# MANUAL INPUT
# ============================================
if input_method == "Manual Input":
    st.subheader("📝 Enter Transaction Details")
    
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
    
    if st.button("🔍 Predict Fraud", type="primary"):
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
                with st.expander("📋 Full Response"):
                    st.json(result)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================
# CSV UPLOAD
# ============================================
elif input_method == "Upload CSV":
    st.subheader("📂 Upload CSV File")
    
    st.info("""
    **CSV Format Requirements:**
    - Columns: Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount
    - All values must be numeric
    - No header row? Check the box below
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if header exists
            if df.columns[0] != 'Time':
                # Try without header
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                             'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                             'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                             'V28', 'Amount']
            
            st.success(f"✅ {len(df)} transactions loaded successfully!")
            st.write("**Data Preview:**")
            st.dataframe(df.head())
            
            # Batch prediction button
            if st.button("🔍 Analyze All Transactions", type="primary"):
                with st.spinner(f"Analyzing {len(df)} transactions..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        features_list = row.tolist()
                        data = {"features": features_list}
                        
                        try:
                            response = requests.post(api_url, json=data, timeout=10)
                            if response.status_code == 200:
                                result = response.json()
                                results.append(result)
                            else:
                                results.append({"error": response.text})
                        except:
                            results.append({"error": "Request failed"})
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    st.divider()
                    st.subheader("📊 Batch Analysis Results")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    if 'is_fraud' in results_df.columns:
                        fraud_count = results_df['is_fraud'].sum()
                        legit_count = len(results_df) - fraud_count
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(results_df))
                        with col2:
                            st.metric("⚠️ Fraud Detected", fraud_count)
                        with col3:
                            st.metric("✅ Legitimate", legit_count)
                        
                        # Add predictions to original df
                        df['Prediction'] = results_df['is_fraud'].apply(lambda x: 'FRAUD' if x else 'LEGIT')
                        df['Fraud_Probability'] = results_df['fraud_probability'].apply(lambda x: f"{x*100:.2f}%")
                        df['Risk_Score'] = results_df['risk_score'].apply(lambda x: f"{x:.2f}")
                        
                        st.write("**Results:**")
                        st.dataframe(df[['Time', 'Amount', 'Prediction', 'Fraud_Probability', 'Risk_Score']])
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No valid predictions received. Check API connection.")
                        
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.info("Make sure the CSV format is correct.")

# ============================================
# SAMPLE DATA
# ============================================
else:
    st.subheader("📋 Sample Transactions")
    
    st.info("Click the button below to analyze a sample fraud transaction.")
    
    if st.button("🔍 Analyze Sample Fraud Transaction", type="primary"):
        features_list = [
            406.0, -2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468,
            -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804,
            -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473,
            -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657,
            -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474,
            0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526,
            0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919, 0.0
        ]
        data = {"features": features_list}
        
        try:
            response = requests.post(api_url, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
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
                with st.expander("📋 Full Response"):
                    st.json(result)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()
st.caption("Built with ❤️ using Streamlit, FastAPI, and XGBoost")
st.caption("© 2026 Muhammad Usman | Fraud Detection System")