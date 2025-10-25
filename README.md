# AgroX Machine Learning Models

A comprehensive machine learning suite for agricultural customer intelligence, featuring customer segmentation, retention prediction, and product recommendation systems. This repository contains trained models, training notebooks, and supporting data for the AgroX platform.

## 🎯 Overview

The AgroX ML models provide advanced analytics capabilities for agricultural e-commerce platforms, enabling data-driven decision making through customer behavior analysis, segmentation, and personalized recommendations.

## 🚀 Features

### 📊 **Customer Segmentation**
- **Dynamic K-Means Clustering**: Automatically determines optimal number of customer segments
- **RFM Analysis**: Recency, Frequency, Monetary value analysis
- **Feature Engineering**: Advanced preprocessing and feature selection
- **Scalable Architecture**: Handles large customer datasets efficiently

### 🔄 **Customer Retention Prediction**
- **Random Forest Classifier**: High-accuracy retention prediction
- **Multi-feature Analysis**: Comprehensive customer behavior modeling
- **Risk Assessment**: Identifies customers at risk of churning
- **Confidence Scoring**: Provides prediction confidence levels

### 🛍️ **Product Recommendation System**
- **Collaborative Filtering**: Advanced recommendation algorithms
- **Personalized Suggestions**: Customer-specific product recommendations
- **Likelihood Scoring**: Confidence scores for recommendation accuracy
- **Scalable Processing**: Handles large product catalogs

## 📁 Repository Structure

```
model/
├── models/                           # Trained model files
│   ├── customer_retention.pkl         # Retention prediction model
│   ├── customer_segmentation_kmeans.pkl # Segmentation clustering model
│   ├── customer_segmetation_scaler.pkl # Feature scaler for segmentation
│   └── recomendation_model.pkl       # Product recommendation model
├── train_test_code/                  # Jupyter notebooks for training
│   ├── customer_retention.ipynb      # Retention model training
│   ├── customer_segmentation.ipynb   # Segmentation model training
│   ├── Product_Recommendations.ipynb # Recommendation model training
│   └── Test_save_prediction.ipynb   # Model testing and validation
├── Data/                            # Training and test data
│   └── top_n_recommendations.csv    # Recommendation dataset
├── docs/                            # Documentation
│   └── Technical Documentation.docx # Technical specifications
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies
├── start.sh                        # Startup script
└── README.md                       # This file
```

## 🛠️ Technology Stack

- **Python**: 3.8+
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment
- **Joblib**: Model serialization and persistence

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Jupyter Notebook**: For interactive development
- **Git**: Version control

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd model

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Using Pre-trained Models**
```python
import joblib
import pandas as pd
import numpy as np

# Load customer segmentation model
scaler = joblib.load('models/customer_segmetation_scaler.pkl')
kmeans_model = joblib.load('models/customer_segmentation_kmeans.pkl')

# Load retention prediction model
retention_model = joblib.load('models/customer_retention.pkl')

# Load recommendation model
recommendation_model = joblib.load('models/recomendation_model.pkl')

# Prepare customer data
customer_data = pd.DataFrame({
    'recency': [30],
    'avg_order_value': [150.0],
    'customer_lifetime_days': [365],
    'purchase_rate': [0.05],
    'total_items_sold': [25]
})

# Customer segmentation prediction
scaled_data = scaler.transform(customer_data)
segment = kmeans_model.predict(scaled_data)
print(f"Customer segment: {segment[0]}")

# Retention prediction
retention_features = customer_data[['recency', 'avg_order_value', 'customer_lifetime_days', 'purchase_rate', 'total_items_sold']]
retention_prob = retention_model.predict_proba(retention_features)
print(f"Retention probability: {retention_prob[0][1]:.3f}")
```

### 3. **Running Training Notebooks**
```bash
# Start Jupyter Notebook
jupyter notebook

# Open the desired notebook:
# - customer_segmentation.ipynb
# - customer_retention.ipynb
# - Product_Recommendations.ipynb
```

## 📊 Model Details

### **Customer Segmentation Model**

**Algorithm**: K-Means Clustering with Dynamic Cluster Selection
**Features**:
- Recency (days since last purchase)
- Average Order Value
- Customer Lifetime Days
- Purchase Rate
- Total Items Sold

**Segments**:
- **Dormant/Churned**: Inactive customers requiring re-engagement
- **Loyal/Engaged**: High-value repeat customers
- **New/Recent but Inactive**: Recent customers with low engagement
- **High-Engagement/Recent High-Value**: Premium active customers

**Performance**:
- Optimal clusters determined dynamically (typically 3-6)
- Silhouette score optimization for cluster quality
- Elbow method for cluster number selection

### **Customer Retention Model**

**Algorithm**: Random Forest Classifier
**Features**:
- Recency_y (days since last purchase)
- Frequency (number of transactions)
- Monetary (total monetary value)
- Customer Lifetime Days
- Purchase Rate
- Total Items Sold
- Unique Products Count
- Average Items Per Order
- Average Revenue Per Order
- Average Net Sales Per Order

**Output**:
- Binary classification: 1 (Returning) or 0 (Not Returning)
- Confidence scores for predictions
- Feature importance analysis

**Performance**:
- High accuracy on validation data
- Robust to overfitting
- Feature importance ranking available

### **Product Recommendation Model**

**Algorithm**: Collaborative Filtering with Matrix Factorization
**Features**:
- Customer purchase history
- Product characteristics
- Customer behavior patterns
- Purchase frequency and recency

**Output**:
- Product recommendations with likelihood scores
- Personalized suggestions per customer
- Confidence levels for recommendations

**Performance**:
- Handles large product catalogs efficiently
- Provides top-N recommendations
- Scalable to thousands of customers and products

## 📈 Training Process

### **Data Preparation**
1. **Data Loading**: Import customer and transaction data
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Create RFM metrics and derived features
4. **Data Splitting**: Train/validation/test splits
5. **Feature Scaling**: Normalize features for optimal performance

### **Model Training**
1. **Hyperparameter Tuning**: Grid search for optimal parameters
2. **Cross-Validation**: K-fold validation for robust evaluation
3. **Model Selection**: Choose best performing algorithm
4. **Performance Evaluation**: Metrics calculation and analysis
5. **Model Persistence**: Save trained models and scalers

### **Validation Process**
1. **Holdout Testing**: Test on unseen data
2. **Performance Metrics**: Accuracy, precision, recall, F1-score
3. **Business Metrics**: Customer lifetime value, retention rate
4. **Model Comparison**: Compare different algorithms
5. **Feature Importance**: Analyze most predictive features

## 🔧 Model Configuration

### **Segmentation Model Parameters**
```python
# K-Means parameters
n_clusters = 'auto'  # Dynamic cluster selection
random_state = 42
n_init = 10
max_iter = 300

# Feature scaling
scaler = StandardScaler()
```

### **Retention Model Parameters**
```python
# Random Forest parameters
n_estimators = 100
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
random_state = 42
```

### **Recommendation Model Parameters**
```python
# Collaborative filtering parameters
n_factors = 50
n_epochs = 20
learning_rate = 0.01
regularization = 0.02
```

## 📊 Data Requirements

### **Customer Data Schema**
```python
customer_data = {
    'customer_id': str,           # Unique customer identifier
    'recency': int,               # Days since last purchase
    'avg_order_value': float,     # Average order value
    'customer_lifetime_days': int, # Customer account age
    'purchase_rate': float,       # Purchase frequency
    'total_items_sold': int,      # Total items purchased
    'gender': str,                # Customer gender
    'age': int,                   # Customer age
    'region': str                 # Customer region
}
```

### **Transaction Data Schema**
```python
transaction_data = {
    'customer_id': str,           # Customer identifier
    'product_id': str,            # Product identifier
    'purchase_date': datetime,    # Purchase timestamp
    'quantity': int,              # Quantity purchased
    'price': float,               # Product price
    'total_amount': float         # Total transaction amount
}
```

## 🧪 Testing and Validation

### **Model Testing Script**
```python
# Test model loading and prediction
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load models
models = {
    'segmentation': joblib.load('models/customer_segmentation_kmeans.pkl'),
    'retention': joblib.load('models/customer_retention.pkl'),
    'scaler': joblib.load('models/customer_segmetation_scaler.pkl')
}

# Test predictions
def test_models(test_data):
    # Segmentation test
    scaled_data = models['scaler'].transform(test_data)
    segments = models['segmentation'].predict(scaled_data)
    
    # Retention test
    retention_pred = models['retention'].predict(test_data)
    retention_prob = models['retention'].predict_proba(test_data)
    
    return segments, retention_pred, retention_prob
```

### **Performance Metrics**
- **Segmentation**: Silhouette score, inertia, cluster separation
- **Retention**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Recommendations**: Precision@K, Recall@K, NDCG@K

## 🚀 Deployment

### **Docker Deployment**
```bash
# Build Docker image
docker build -t agrox-models .

# Run container
docker run -p 8888:8888 agrox-models
```

### **Model Serving**
```python
# Flask API for model serving
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models at startup
models = {
    'segmentation': joblib.load('models/customer_segmentation_kmeans.pkl'),
    'retention': joblib.load('models/customer_retention.pkl'),
    'scaler': joblib.load('models/customer_segmetation_scaler.pkl')
}

@app.route('/predict/segmentation', methods=['POST'])
def predict_segmentation():
    data = request.json
    scaled_data = models['scaler'].transform([data['features']])
    prediction = models['segmentation'].predict(scaled_data)
    return jsonify({'segment': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📈 Performance Monitoring

### **Model Performance Tracking**
- **Accuracy Monitoring**: Track prediction accuracy over time
- **Drift Detection**: Monitor for data and concept drift
- **Performance Metrics**: Regular evaluation of model metrics
- **A/B Testing**: Compare model versions in production

### **Retraining Pipeline**
- **Automated Retraining**: Schedule regular model updates
- **Data Validation**: Ensure data quality before retraining
- **Model Versioning**: Track model versions and performance
- **Rollback Capability**: Ability to revert to previous models

## 🔍 Troubleshooting

### **Common Issues**

1. **Model Loading Errors**
   - Verify model files exist and are not corrupted
   - Check Python version compatibility
   - Ensure all dependencies are installed

2. **Prediction Errors**
   - Validate input data format and types
   - Check feature scaling consistency
   - Verify feature order matches training data

3. **Performance Degradation**
   - Monitor for data drift
   - Check model accuracy over time
   - Consider retraining with recent data

## 📚 Documentation

### **Technical Documentation**
- **Model Architecture**: Detailed model specifications
- **Training Process**: Step-by-step training procedures
- **API Reference**: Model serving API documentation
- **Performance Benchmarks**: Model performance metrics

### **Business Documentation**
- **Use Cases**: Business applications and scenarios
- **ROI Analysis**: Business impact and value proposition
- **Implementation Guide**: Deployment and integration steps

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Development Guidelines**
- Follow Python best practices (PEP 8)
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the technical documentation
- Review the troubleshooting section
- Contact the ML team

---

**AgroX Machine Learning Models** - Advanced AI-powered analytics for agricultural customer intelligence and business optimization.