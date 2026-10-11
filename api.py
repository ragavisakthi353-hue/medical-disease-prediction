"""
REST API Microservice for Disease Prediction System (Flask)
Provides endpoints for Heart Disease, Diabetes, Breast Cancer, and Symptom assessment.
"""

import os
import sys
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predictor import DiseasePredictor, load_all_metrics

app = Flask(__name__)

# Preload domain predictors
predictors = {
    'heart': DiseasePredictor('heart'),
    'diabetes': DiseasePredictor('diabetes'),
    'breast_cancer': DiseasePredictor('breast_cancer'),
    'symptoms': DiseasePredictor('symptoms')
}

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "system": "MediPredict AI Disease Prediction API",
        "status": "online",
        "supported_domains": ["heart", "diabetes", "breast_cancer", "symptoms"],
        "algorithms": ["SVM", "Logistic Regression", "Random Forest", "XGBoost", "Ensemble Consensus"],
        "endpoints": {
            "heart": "/predict/heart",
            "diabetes": "/predict/diabetes",
            "breast_cancer": "/predict/breast-cancer",
            "symptoms": "/predict/symptoms",
            "metrics": "/metrics"
        }
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns cross-algorithm performance benchmarks."""
    return jsonify(load_all_metrics())

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    data = request.get_json(force=True)
    algorithm = data.pop('algorithm', 'Random Forest')
    result = predictors['heart'].predict(data, algorithm)
    return jsonify(result)

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.get_json(force=True)
    algorithm = data.pop('algorithm', 'XGBoost')
    result = predictors['diabetes'].predict(data, algorithm)
    return jsonify(result)

@app.route('/predict/breast-cancer', methods=['POST'])
def predict_breast_cancer():
    data = request.get_json(force=True)
    algorithm = data.pop('algorithm', 'SVM')
    result = predictors['breast_cancer'].predict(data, algorithm)
    return jsonify(result)

@app.route('/predict/symptoms', methods=['POST'])
def predict_symptoms():
    data = request.get_json(force=True)
    algorithm = data.pop('algorithm', 'Random Forest')
    result = predictors['symptoms'].predict(data, algorithm)
    return jsonify(result)

if __name__ == '__main__':
    print("Starting MediPredict AI REST API on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
