"""
Prediction and Clinical Risk Analysis Engine
Loads trained models, performs inference, generates multi-model consensus,
and provides clinical interpretability and actionable health advice.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

# Clinical reference baselines and normal ranges
CLINICAL_RANGES = {
    'heart': {
        'trestbps': {'normal_max': 120, 'unit': 'mm Hg', 'name': 'Resting Blood Pressure', 'high_desc': 'Stage 1/2 Hypertension'},
        'chol': {'normal_max': 200, 'unit': 'mg/dl', 'name': 'Serum Cholesterol', 'high_desc': 'Hypercholesterolemia'},
        'thalach': {'normal_min': 100, 'unit': 'bpm', 'name': 'Max Heart Rate', 'low_desc': 'Reduced Cardiac Reserve'},
        'oldpeak': {'normal_max': 1.0, 'unit': 'mm', 'name': 'ST Depression', 'high_desc': 'Myocardial Ischemia sign'},
        'fbs': {'normal_max': 0, 'unit': 'binary', 'name': 'Fasting Blood Sugar > 120', 'high_desc': 'Elevated Blood Glucose'},
        'ca': {'normal_max': 0, 'unit': 'vessels', 'name': 'Major Vessels Blocked', 'high_desc': 'Coronary Vessel Calcification'}
    },
    'diabetes': {
        'Glucose': {'normal_max': 140, 'normal_min': 70, 'unit': 'mg/dL', 'name': 'Plasma Glucose', 'high_desc': 'Hyperglycemia / Impaired Fasting Glucose'},
        'BMI': {'normal_max': 25.0, 'unit': 'kg/m²', 'name': 'Body Mass Index', 'high_desc': 'Overweight / Class I-III Obesity'},
        'BloodPressure': {'normal_max': 80, 'unit': 'mm Hg', 'name': 'Diastolic BP', 'high_desc': 'Elevated Diastolic BP'},
        'Insulin': {'normal_max': 166, 'unit': 'µU/mL', 'name': '2-Hr Serum Insulin', 'high_desc': 'Hyperinsulinemia / Insulin Resistance'},
        'Age': {'normal_max': 45, 'unit': 'years', 'name': 'Age Factor', 'high_desc': 'Age-associated diabetes risk'}
    },
    'breast_cancer': {
        'mean radius': {'normal_max': 14.0, 'unit': 'mm', 'name': 'Mean Cell Radius', 'high_desc': 'Enlarged Nuclear Dimension'},
        'mean texture': {'normal_max': 19.0, 'unit': 'gray scale', 'name': 'Texture Roughness', 'high_desc': 'Structural Heterogeneity'},
        'mean perimeter': {'normal_max': 90.0, 'unit': 'mm', 'name': 'Perimeter Size', 'high_desc': 'Mass Perimeter Expansion'},
        'mean area': {'normal_max': 650.0, 'unit': 'mm²', 'name': 'Mass Area', 'high_desc': 'Elevated Tumor Cross-section'},
        'mean concavity': {'normal_max': 0.08, 'unit': 'severity', 'name': 'Concavity Depth', 'high_desc': 'Irregular Contour Severity'},
        'mean concave points': {'normal_max': 0.05, 'unit': 'count', 'name': 'Concave Points', 'high_desc': 'Malignant Infiltration Markers'}
    }
}

class DiseasePredictor:
    def __init__(self, domain='heart'):
        self.domain = domain
        self.domain_dir = os.path.join(SAVED_MODELS_DIR, domain)
        self.models = {}
        self.scaler = None
        self.meta = {}
        self.label_encoder = None
        self._load_artifacts()

    def _load_artifacts(self):
        # Load meta
        meta_path = os.path.join(self.domain_dir, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.meta = json.load(f)
                
        # Load scaler
        scaler_path = os.path.join(self.domain_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        # Load label encoder if multiclass
        le_path = os.path.join(self.domain_dir, 'label_encoder.joblib')
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            
        # Load algorithms
        model_names = ['svm', 'logistic_regression', 'random_forest', 'xgboost']
        for m in model_names:
            model_file = os.path.join(self.domain_dir, f"{m}.joblib")
            if os.path.exists(model_file):
                display_name = m.replace('_', ' ').title()
                if display_name == 'Svm':
                    display_name = 'SVM'
                elif display_name == 'Xgboost':
                    display_name = 'XGBoost'
                self.models[display_name] = joblib.load(model_file)

    def predict(self, input_dict, selected_model_name='Random Forest'):
        """
        Runs prediction for given input dictionary.
        Returns:
            - primary_prediction (class label)
            - probability (0.0 - 1.0)
            - risk_level ('Low', 'Moderate', 'High')
            - all_model_probs (dict of probabilities from all 4 models)
            - top_risk_factors (list of identified risk indicators)
            - recommendations (actionable health steps)
        """
        feature_cols = self.meta.get('feature_cols', [])
        # Format input vector
        features_vector = [float(input_dict.get(col, 0.0)) for col in feature_cols]
        df_input = pd.DataFrame([features_vector], columns=feature_cols)
        X_scaled = self.scaler.transform(df_input) if self.scaler else df_input.values
        
        is_multiclass = self.meta.get('is_multiclass', False)
        
        all_model_predictions = {}
        all_model_probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                
                if is_multiclass:
                    if self.label_encoder:
                        class_label = self.label_encoder.inverse_transform([pred])[0]
                    else:
                        class_label = str(pred)
                    top_prob = float(np.max(proba))
                    all_model_predictions[name] = class_label
                    all_model_probabilities[name] = {
                        cls: round(float(p), 4)
                        for cls, p in zip(self.label_encoder.classes_, proba)
                    }
                else:
                    pos_prob = float(proba[1]) if len(proba) > 1 else float(pred)
                    all_model_predictions[name] = int(pred)
                    all_model_probabilities[name] = round(pos_prob, 4)
            except Exception as e:
                all_model_predictions[name] = 0
                all_model_probabilities[name] = 0.0

        # Selected model resolution
        if selected_model_name == 'Ensemble Consensus':
            if is_multiclass:
                # Majority vote
                votes = list(all_model_predictions.values())
                final_pred = max(set(votes), key=votes.count)
                # Average probability for that voted disease
                avg_prob = np.mean([
                    all_model_probabilities[m].get(final_pred, 0.5)
                    for m in self.models.keys()
                ])
                final_prob = float(avg_prob)
            else:
                avg_prob = np.mean(list(all_model_probabilities.values()))
                final_pred = 1 if avg_prob >= 0.50 else 0
                final_prob = float(avg_prob)
        else:
            model = self.models.get(selected_model_name, list(self.models.values())[0])
            raw_pred = model.predict(X_scaled)[0]
            raw_proba = model.predict_proba(X_scaled)[0]
            if is_multiclass:
                final_pred = self.label_encoder.inverse_transform([raw_pred])[0]
                final_prob = float(np.max(raw_proba))
            else:
                final_pred = int(raw_pred)
                final_prob = float(raw_proba[1]) if len(raw_proba) > 1 else float(raw_pred)

        # Risk level determination
        if is_multiclass:
            risk_level = "High Probability" if final_prob > 0.65 else ("Moderate Probability" if final_prob > 0.40 else "Inconclusive")
        else:
            if final_prob < 0.35:
                risk_level = "Low Risk"
            elif final_prob < 0.65:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"

        # Risk factor explainability
        risk_factors = self._analyze_risk_factors(input_dict)
        recommendations = self._generate_recommendations(final_pred, final_prob, risk_level, risk_factors)
        
        return {
            'prediction': final_pred,
            'probability': round(final_prob, 4),
            'risk_level': risk_level,
            'all_model_probabilities': all_model_probabilities,
            'all_model_predictions': all_model_predictions,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }

    def _analyze_risk_factors(self, input_dict):
        factors = []
        domain_rules = CLINICAL_RANGES.get(self.domain, {})
        for feature, rule in domain_rules.items():
            if feature in input_dict:
                val = float(input_dict[feature])
                if 'normal_max' in rule and val > rule['normal_max']:
                    factors.append({
                        'feature': rule['name'],
                        'value': val,
                        'unit': rule['unit'],
                        'threshold': f"> {rule['normal_max']} {rule['unit']}",
                        'clinical_meaning': rule.get('high_desc', 'Elevated above standard range'),
                        'status': 'Elevated'
                    })
                elif 'normal_min' in rule and val < rule['normal_min']:
                    factors.append({
                        'feature': rule['name'],
                        'value': val,
                        'unit': rule['unit'],
                        'threshold': f"< {rule['normal_min']} {rule['unit']}",
                        'clinical_meaning': rule.get('low_desc', 'Lower than standard range'),
                        'status': 'Reduced'
                    })
                    
        # For symptoms domain
        if self.domain == 'symptoms':
            temp = float(input_dict.get('temperature', 98.6))
            if temp > 100.4:
                factors.append({
                    'feature': 'Body Temperature',
                    'value': temp,
                    'unit': '°F',
                    'threshold': '> 100.4 °F',
                    'clinical_meaning': 'Febrile State / Fever',
                    'status': 'Elevated'
                })
            for k, v in input_dict.items():
                if k not in ['age', 'temperature'] and float(v) == 1:
                    factors.append({
                        'feature': k.replace('_', ' ').title(),
                        'value': 'Present',
                        'unit': '',
                        'threshold': 'Positive',
                        'clinical_meaning': 'Active Symptom Reported',
                        'status': 'Active'
                    })
        return factors

    def _generate_recommendations(self, pred, prob, risk_level, factors):
        recs = []
        if self.domain == 'heart':
            if risk_level == "High Risk":
                recs.append("Urgent Consultation: Schedule an appointment with a Cardiologist for 12-lead ECG and Echocardiogram evaluation.")
                recs.append("Diagnostic Tests: Recommend Stress Echo / Coronary Angiogram and complete Lipid Panel monitoring.")
                recs.append("Lifestyle Intervention: Restrict sodium intake (<2,000 mg/day) and monitor blood pressure twice daily.")
            elif risk_level == "Moderate Risk":
                recs.append("Preventative Care: Undergo an annual cardiac risk profiling test and routine ECG.")
                recs.append("Aerobic Exercise: Aim for 150 minutes of moderate aerobic activity weekly after doctor clearance.")
                recs.append("Dietary Adjustment: Adopt Mediterranean or DASH dietary pattern rich in omega-3 and dietary fiber.")
            else:
                recs.append("Maintain Health: Continue regular routine check-ups and maintain optimal BMI and balanced nutrition.")
                recs.append("Cardiovascular Wellness: Keep active with at least 30 minutes of daily physical exercise.")
                
        elif self.domain == 'diabetes':
            if risk_level == "High Risk":
                recs.append("Urgent Diagnostic: Order HbA1c test and Fasting Plasma Glucose (FPG) test immediately.")
                recs.append("Endocrine Evaluation: Consult an endocrinologist for personalized glycemic control protocol.")
                recs.append("Dietary Protocol: Reduce refined carbohydrates and sugars; establish low-glycemic index dietary regimen.")
            elif risk_level == "Moderate Risk":
                recs.append("Monitoring: Schedule an oral glucose tolerance test (OGTT) or HbA1c screening.")
                recs.append("Weight Management: Target 5-7% weight reduction to significantly decrease insulin resistance.")
                recs.append("Activity: Engage in brisk walking 30 minutes daily after main meals.")
            else:
                recs.append("Optimal Health: Maintain balanced metabolic wellness with whole grains, lean protein, and hydration.")
                recs.append("Routine Screen: Annual metabolic blood panel is recommended.")
                
        elif self.domain == 'breast_cancer':
            if risk_level == "High Risk" or pred == 1:
                recs.append("Immediate Action: Schedule an appointment with a breast surgical specialist / surgical oncologist.")
                recs.append("Confirmatory Imaging: Diagnostic mammography, ultrasound, or core needle biopsy is strongly indicated.")
                recs.append("Clinical Assessment: Do not delay physical clinical examination and tissue pathology review.")
            else:
                recs.append("Screening Protocol: Follow age-appropriate screening mammography guidelines (every 1-2 years).")
                recs.append("Self-Examination: Continue monthly breast self-examinations and report any palpable changes.")
                
        elif self.domain == 'symptoms':
            recs.append(f"Predicted Condition: '{pred}' with {prob*100:.1f}% model confidence.")
            recs.append("Medical Advice: Please consult a licensed physician or general practitioner for proper clinical confirmation.")
            recs.append("Rest and Hydration: Maintain adequate hydration, isolate if contagious symptoms are present, and seek emergency care if experiencing breathing difficulty.")
            
        return recs

def load_all_metrics():
    """Loads consolidated evaluation metrics across all domains and algorithms."""
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}
