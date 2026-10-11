"""
Model Training and Evaluation Engine
Trains and benchmarks SVM, Logistic Regression, Random Forest, and XGBoost
across Heart Disease, Diabetes, Breast Cancer, and Symptom-based datasets.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_heart_disease_data,
    load_diabetes_data,
    load_breast_cancer_data,
    create_symptom_disease_data
)

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

def get_models(is_multiclass=False, num_classes=2):
    """Returns instances of the 4 requested algorithms."""
    if is_multiclass:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=num_classes
        )
    else:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

    return {
        'SVM': SVC(kernel='rbf', C=1.5, probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'XGBoost': xgb
    }

def train_and_evaluate_domain(domain_name, df, feature_cols, target_col, is_multiclass=False):
    """
    Trains all 4 algorithms on a given dataset domain, computes rich evaluation metrics,
    and serializes the models and scalers.
    """
    domain_dir = os.path.join(SAVED_MODELS_DIR, domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    label_encoder = None
    target_names = ['Healthy / Negative', 'Disease Detected / Positive']
    if is_multiclass:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        target_names = list(label_encoder.classes_)
        joblib.dump(label_encoder, os.path.join(domain_dir, 'label_encoder.joblib'))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(domain_dir, 'scaler.joblib'))
    
    # Save feature names and test sample for easy testing/demo in UI
    meta = {
        'domain': domain_name,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'target_names': target_names,
        'is_multiclass': is_multiclass,
        'sample_ranges': {
            col: {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'type': 'int' if np.issubdtype(df[col].dtype, np.integer) else 'float'
            }
            for col in feature_cols
        }
    }
    with open(os.path.join(domain_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    num_classes = len(np.unique(y))
    models = get_models(is_multiclass, num_classes=num_classes)
    domain_results = {}
    
    print(f"\n=======================================================")
    print(f"Training models for domain: {domain_name.upper()} ({len(df)} samples, {len(feature_cols)} features)")
    print(f"=======================================================")
    
    for model_name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC
        if is_multiclass:
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0
        else:
            try:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            except Exception:
                auc = 0.0
                
        # 5-fold cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature importance extraction where available
        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = {f: float(round(imp, 4)) for f, imp in zip(feature_cols, importances)}
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim == 1:
                coefs = np.abs(model.coef_)
            else:
                coefs = np.mean(np.abs(model.coef_), axis=0)
            norm_coefs = coefs / (np.sum(coefs) + 1e-9)
            feature_importances = {f: float(round(imp, 4)) for f, imp in zip(feature_cols, norm_coefs)}
            
        domain_results[model_name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'cv_mean': round(float(cv_scores.mean()), 4),
            'cv_std': round(float(cv_scores.std()), 4),
            'confusion_matrix': cm,
            'feature_importances': feature_importances
        }
        
        # Save model
        clean_model_key = model_name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(domain_dir, f"{clean_model_key}.joblib"))
        
        print(f"[{model_name.ljust(19)}] Acc: {acc*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | 5-Fold CV: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
        
    return domain_results

def run_all_trainings():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    all_metrics = {}
    
    # 1. Heart Disease
    df_heart = load_heart_disease_data()
    heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    all_metrics['heart'] = train_and_evaluate_domain(
        'heart', df_heart, heart_features, 'target', is_multiclass=False
    )
    
    # 2. Diabetes
    df_diabetes = load_diabetes_data()
    diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    all_metrics['diabetes'] = train_and_evaluate_domain(
        'diabetes', df_diabetes, diabetes_features, 'Outcome', is_multiclass=False
    )
    
    # 3. Breast Cancer (Wisconsin FNA Biopsy diagnostic)
    df_cancer = load_breast_cancer_data()
    # 10 core clinical mean morphological features
    cancer_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    all_metrics['breast_cancer'] = train_and_evaluate_domain(
        'breast_cancer', df_cancer, cancer_features, 'target', is_multiclass=False
    )
    
    # 4. Multi-Disease Symptom Checker
    df_symptoms = create_symptom_disease_data()
    symptom_features = [c for c in df_symptoms.columns if c != 'disease']
    all_metrics['symptoms'] = train_and_evaluate_domain(
        'symptoms', df_symptoms, symptom_features, 'disease', is_multiclass=True
    )
    
    # Save consolidated metrics
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
        
    print("\n=======================================================")
    print("ALL MODELS TRAINED AND METRICS SAVED TO:", metrics_path)
    print("=======================================================")
    return all_metrics

if __name__ == '__main__':
    run_all_trainings()
