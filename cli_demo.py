"""
Command-Line Interface (CLI) & Demonstration Script
Run instant predictions and benchmark evaluations from the terminal.
"""

import sys
import os
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import DiseasePredictor, load_all_metrics

def print_banner():
    print("""
========================================================================
     MEDIPREDICT AI - MULTI-DISEASE PREDICTION & BENCHMARK ENGINE
========================================================================
Algorithms: SVM | Logistic Regression | Random Forest | XGBoost
Datasets:   UCI Heart Disease | Diabetes | Breast Cancer | Symptoms
========================================================================
    """)

def run_benchmarks():
    print_banner()
    metrics = load_all_metrics()
    if not metrics:
        print("No trained models found. Running training now...")
        from src.model_trainer import run_all_trainings
        metrics = run_all_trainings()
        
    for domain, models in metrics.items():
        print(f"\n--- DOMAIN: {domain.upper()} ---")
        rows = []
        for model_name, m in models.items():
            rows.append({
                'Algorithm': model_name,
                'Accuracy': f"{m['accuracy']*100:.2f}%",
                'F1-Score': f"{m['f1_score']:.4f}",
                'ROC-AUC': f"{m['roc_auc']:.4f}",
                '5-Fold CV': f"{m['cv_mean']*100:.2f}% (±{m['cv_std']*100:.2f}%)"
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

def test_heart_prediction():
    print("\n--- SAMPLE PREDICTION: HEART DISEASE ---")
    predictor = DiseasePredictor('heart')
    sample = {
        'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
        'fbs': 0, 'restecg': 2, 'thalach': 108, 'exang': 1,
        'oldpeak': 2.6, 'slope': 2, 'ca': 3, 'thal': 7
    }
    print("Patient Vitals:", sample)
    for model_name in ['Random Forest', 'XGBoost', 'SVM', 'Logistic Regression', 'Ensemble Consensus']:
        res = predictor.predict(sample, model_name)
        call = "Heart Disease Present" if res['prediction'] == 1 else "Healthy / No Disease"
        print(f"[{model_name.ljust(18)}] Result: {call.ljust(22)} | Prob: {res['probability']*100:.1f}% | Risk: {res['risk_level']}")
        
    res_final = predictor.predict(sample, 'Ensemble Consensus')
    print("\nPrimary Risk Drivers:")
    for f in res_final['risk_factors']:
        print(f" - {f['feature']}: {f['value']} {f['unit']} ({f['clinical_meaning']})")
    print("\nClinical Recommendations:")
    for r in res_final['recommendations']:
        print(f" • {r}")

def main():
    parser = argparse.ArgumentParser(description="Disease Prediction CLI Utility")
    parser.add_argument('--benchmark', action='store_true', help="Print benchmark comparison table")
    parser.add_argument('--demo', action='store_true', help="Run demonstration inference")
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmarks()
    elif args.demo:
        test_heart_prediction()
    else:
        run_benchmarks()
        test_heart_prediction()

if __name__ == '__main__':
    main()
