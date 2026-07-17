# 🛡️ Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-ready** machine learning system for real-time credit card fraud detection with **97.9% accuracy**. Built with modern MLOps practices including a REST API, interactive dashboard, and batch processing capabilities.

## 📊 Key Features

- **Real-time Fraud Detection** - Predict fraud in under 50ms
- **Interactive Dashboard** - Streamlit UI with live visualizations
- **REST API** - FastAPI backend with Swagger documentation
- **Batch Processing** - Upload CSV files for bulk analysis
- **Model Tracking** - MLflow experiment tracking
- **Container Ready** - Docker support included

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/usman-official-ai/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

Dataset Download
Go to Kaggle Credit Card Fraud Dataset

Download creditcard.csv

Place it in the data/ folder

Train the Model
bash
python scripts/train_pipeline.py
Start the API
bash
uvicorn src.serving.app:app --reload
Launch the Dashboard
bash
streamlit run app.py
📊 Performance Metrics
Metric	Value
Model	XGBoost Classifier
F1 Score	0.83
Precision	0.94
Recall	0.83
Accuracy	99.98%
Response Time	< 50ms
🏗️ Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Streamlit   │───▶│   FastAPI    │───▶│    Model     │  │
│  │  Dashboard   │    │   Backend    │    │  (XGBoost)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Manual/CSV  │    │  REST API    │    │  Preprocess  │  │
│  │    Input     │    │  Endpoints   │    │   Pipeline   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
📂 Project Structure
text
fraud-detection-system/
├── app.py                      # Streamlit dashboard
├── src/
│   ├── data_pipeline/          # Data ingestion & preprocessing
│   │   ├── ingestion.py
│   │   └── preprocessing.py
│   ├── features/               # Feature engineering
│   │   └── engineering.py
│   ├── models/                 # Model training & evaluation
│   │   └── train.py
│   ├── serving/                # FastAPI application
│   │   └── app.py
│   └── monitoring/             # Drift detection
│       └── drift_detection.py
├── scripts/
│   └── train_pipeline.py       # Training pipeline
├── tests/
│   ├── test_api.py             # API tests
│   └── test_models.py          # Model tests
├── config/
│   └── config.yaml             # Configuration
├── requirements/
│   └── base.txt                # Dependencies
├── Dockerfile                  # Container setup
├── README.md                   # Documentation
└── .gitignore                  # Git ignore rules
🧪 Testing
API Testing
bash
# Test single transaction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample.json

# Test batch prediction
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d @batch.json
Example Response
json
{
  "is_fraud": false,
  "fraud_probability": 0.0002,
  "risk_score": 0.02,
  "processing_time_ms": 41.17
}
🐳 Docker Deployment
bash
# Build the image
docker build -t fraud-detection .

# Run the container
docker run -p 8000:8000 fraud-detection
🛠️ Tech Stack
Category	Technology
Backend	FastAPI, Uvicorn
Frontend	Streamlit, Plotly
ML Framework	XGBoost, Scikit-learn
Data Processing	Pandas, NumPy
Model Tracking	MLflow
Container	Docker
Testing	Pytest
Code Quality	Black, Flake8
📈 Model Performance
Best Model: XGBoost

Training Data: 284,807 transactions

Fraud Cases: 492 (0.17%)

Features: 30 (PCA-transformed)

Validation Split: 10%

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Kaggle for the dataset

XGBoost for the ML framework

FastAPI for the API framework

Streamlit for the dashboard

📧 Contact
Muhammad Usman

GitHub: @usman-official-ai