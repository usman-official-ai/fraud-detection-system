import streamlit as st
import requests
import pandas as pd
import json
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ff4444;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .safe-alert {
        background-color: #00cc66;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("System Configuration")

# API URL input
api_url = st.sidebar.text_input(
    "API Endpoint",
    value="http://127.0.0.1:8000/predict",
    help="URL of the fraud detection API"
)

# Test API connection
if st.sidebar.button("🔌 Test API Connection"):
    try:
        response = requests.get(api_url.replace("/predict", "/health"))
        if response.status_code == 200:
            st.sidebar.success("✅ API is running!")
            st.sidebar.json(response.json())
        else:
            st.sidebar.error("❌ API not responding")
    except:
        st.sidebar.error("❌ Could not connect to API")

st.sidebar.divider()

# Input methods
st.sidebar.subheader("Input Methods")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Input", "Upload CSV", "Use Sample"]
)

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("📝 Transaction Details")

if input_method == "Manual Input":
    # Create input fields in a form
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time = st.number_input("Time", value=0.0, format="%.1f")
            v1 = st.number_input("V1", value=-1.3598, format="%.6f")
            v2 = st.number_input("V2", value=-0.0728, format="%.6f")
            v3 = st.number_input("V3", value=2.5363, format="%.6f")
            v4 = st.number_input("V4", value=1.3782, format="%.6f")
            v5 = st.number_input("V5", value=-0.3383, format="%.6f")
            v6 = st.number_input("V6", value=0.4624, format="%.6f")
            v7 = st.number_input("V7", value=0.2396, format="%.6f")
            v8 = st.number_input("V8", value=0.0987, format="%.6f")
            v9 = st.number_input("V9", value=0.3638, format="%.6f")
            v10 = st.number_input("V10", value=0.0908, format="%.6f")
        
        with col2:
            v11 = st.number_input("V11", value=-0.5516, format="%.6f")
            v12 = st.number_input("V12", value=-0.6178, format="%.6f")
            v13 = st.number_input("V13", value=-0.9914, format="%.6f")
            v14 = st.number_input("V14", value=-0.3112, format="%.6f")
            v15 = st.number_input("V15", value=1.4682, format="%.6f")
            v16 = st.number_input("V16", value=-0.4704, format="%.6f")
            v17 = st.number_input("V17", value=0.2080, format="%.6f")
            v18 = st.number_input("V18", value=0.0258, format="%.6f")
            v19 = st.number_input("V19", value=0.4040, format="%.6f")
            v20 = st.number_input("V20", value=0.2514, format="%.6f")
        
        with col3:
            v21 = st.number_input("V21", value=-0.0183, format="%.6f")
            v22 = st.number_input("V22", value=0.2778, format="%.6f")
            v23 = st.number_input("V23", value=-0.1105, format="%.6f")
            v24 = st.number_input("V24", value=0.0669, format="%.6f")
            v25 = st.number_input("V25", value=0.1285, format="%.6f")
            v26 = st.number_input("V26", value=-0.1891, format="%.6f")
            v27 = st.number_input("V27", value=0.1336, format="%.6f")
            v28 = st.number_input("V28", value=-0.0211, format="%.6f")
            amount = st.number_input("Amount", value=149.62, format="%.2f")
        
        submitted = st.form_submit_button("🔍 Predict Fraud")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file with transaction data", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())
        
        if st.button("🔍 Analyze All Transactions"):
            submitted = True
            transactions = df.to_dict('records')
else:
    # Use sample data
    st.info("Using sample transaction data")
    if st.button("🔍 Analyze Sample"):
        submitted = True
        # Use the fraud sample from earlier
        transactions = [{
            "Time": 406.0,
            "V1": -2.3122265423263,
            "V2": 1.95199201064158,
            "V3": -1.60985073229769,
            "V4": 3.9979055875468,
            "V5": -0.522187864667764,
            "V6": -1.42654531920595,
            "V7": -2.53738730624579,
            "V8": 1.39165724829804,
            "V9": -2.77008927719433,
            "V10": -2.77227214465915,
            "V11": 3.20203320709635,
            "V12": -2.89990738849473,
            "V13": -0.595221881324605,
            "V14": -4.28925378244217,
            "V15": 0.389724120274487,
            "V16": -1.14074717980657,
            "V17": -2.83005567450437,
            "V18": -0.0168224681808257,
            "V19": 0.416955705037907,
            "V20": 0.126910559061474,
            "V21": 0.517232370861764,
            "V22": -0.0350493686052974,
            "V23": -0.465211076182388,
            "V24": 0.320198198514526,
            "V25": 0.0445191674731724,
            "V26": 0.177839798284401,
            "V27": 0.261145002567677,
            "V28": -0.143275874698919,
            "Amount": 0.0
        }]

# Prediction logic
if 'submitted' in locals() and submitted:
    with st.spinner('🔄 Analyzing transaction...'):
        try:
            # Prepare data
            if input_method == "Manual Input":
                data = {
                    "Time": time,
                    "V1": v1, "V2": v2, "V3": v3, "V4": v4,
                    "V5": v5, "V6": v6, "V7": v7, "V8": v8,
                    "V9": v9, "V10": v10, "V11": v11, "V12": v12,
                    "V13": v13, "V14": v14, "V15": v15, "V16": v16,
                    "V17": v17, "V18": v18, "V19": v19, "V20": v20,
                    "V21": v21, "V22": v22, "V23": v23, "V24": v24,
                    "V25": v25, "V26": v26, "V27": v27, "V28": v28,
                    "Amount": amount
                }
                # Single prediction
                response = requests.post(api_url, json=data)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.divider()
                    
                    # Main result display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if result['is_fraud']:
                            st.markdown('<div class="fraud-alert">⚠️ FRAUD DETECTED!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="safe-alert">✅ Transaction is Legitimate</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Fraud Probability",
                            f"{result['fraud_probability']*100:.2f}%",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Risk Score",
                            f"{result['risk_score']:.2f}",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Processing Time",
                            f"{result['processing_time_ms']:.2f} ms",
                            delta=None
                        )
                    
                    with col4:
                        status = "⚠️ FRAUD" if result['is_fraud'] else "✅ SAFE"
                        st.metric("Status", status, delta=None)
                    
                    # Visual gauge
                    st.subheader("Risk Visualization")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['fraud_probability'] * 100,
                        title={'text': "Fraud Probability (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show input data
                    with st.expander("📊 Transaction Data"):
                        st.json(data)
                    
                    # Feature importance placeholder
                    with st.expander("📈 Feature Analysis"):
                        st.info("Feature importance analysis would go here")
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())
            
            elif input_method == "Upload CSV":
                # Batch prediction
                response = requests.post(api_url.replace("/predict", "/predict/batch"), json=transactions)
                if response.status_code == 200:
                    results = response.json()
                    
                    # Display results
                    st.divider()
                    st.subheader("📊 Batch Analysis Results")
                    
                    # Summary metrics
                    predictions = results['predictions']
                    probabilities = results['probabilities']
                    
                    fraud_count = sum(predictions)
                    legit_count = len(predictions) - fraud_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", len(predictions))
                    with col2:
                        st.metric("Fraud Detected", fraud_count, delta=None)
                    with col3:
                        st.metric("Legitimate", legit_count, delta=None)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Transaction': range(1, len(predictions) + 1),
                        'Prediction': ['FRAUD' if p else 'LEGIT' for p in predictions],
                        'Probability': [f"{p*100:.2f}%" for p in probabilities],
                        'Risk Score': [f"{p*100:.2f}" for p in probabilities]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Visualization
                    fig = make_subplots(rows=1, cols=2)
                    
                    # Bar chart
                    fig.add_trace(
                        go.Bar(
                            x=['Fraud', 'Legit'],
                            y=[fraud_count, legit_count],
                            marker_color=['red', 'green'],
                            name='Count'
                        ),
                        row=1, col=1
                    )
                    
                    # Pie chart
                    fig.add_trace(
                        go.Pie(
                            labels=['Fraud', 'Legit'],
                            values=[fraud_count, legit_count],
                            marker=dict(colors=['red', 'green']),
                            name='Distribution'
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"API Error: {response.status_code}")
            
            else:
                # Sample data
                for transaction in transactions:
                    response = requests.post(api_url, json=transaction)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Transaction analyzed: {'FRAUD' if result['is_fraud'] else 'LEGIT'} (Prob: {result['fraud_probability']*100:.2f}%)")
                    else:
                        st.error(f"Error: {response.status_code}")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure the API is running: uvicorn src.serving.app:app --reload")

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit, FastAPI, and XGBoost")
st.caption("© 2024 Fraud Detection System | Production-Ready MLOps Solution")