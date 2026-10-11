"""
Data Loader and Preprocessing Module
Fetches, cleans, and structures datasets for:
1. Heart Disease (UCI Cleveland)
2. Diabetes (Pima Indians)
3. Breast Cancer (UCI Wisconsin Diagnostic)
4. General Symptoms to Disease
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def load_heart_disease_data():
    """
    Loads UCI Heart Disease dataset (Cleveland Clinic Foundation).
    Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    Target: target (1 = Heart Disease Present [angiographic disease > 50% diameter narrowing], 0 = No Heart Disease [< 50% narrowing])
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, 'heart_disease.csv')
    
    # Check if existing file has canonical target correlation or needs refresh
    needs_refresh = True
    if os.path.exists(filepath):
        try:
            df_test = pd.read_csv(filepath)
            # If thal or ca correlates positively with target, it is the canonical dataset
            if 'target' in df_test.columns and df_test.corr()['target'].get('thal', 0) > 0.2:
                needs_refresh = False
                df = df_test
        except Exception:
            needs_refresh = True

    if needs_refresh:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        try:
            df = pd.read_csv(url, header=None, names=cols, na_values='?')
        except Exception:
            # Fallback if UCI server throttles: read local backup or invert inverted file
            alt_url = 'https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv'
            df_alt = pd.read_csv(alt_url)
            # Invert 0 and 1 so 1 = disease
            df_alt['target'] = 1 - df_alt['target']
            df_alt.to_csv(filepath, index=False)
            return df_alt

        df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
        df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
        df['target'] = (df['num'] > 0).astype(int)
        df = df.drop(columns=['num'])
        df.to_csv(filepath, index=False)
        print("Canonical UCI Cleveland Heart Disease dataset saved successfully!")
        
    return df

def load_diabetes_data():
    """
    Loads Pima Indians Diabetes Dataset.
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    Target: Outcome (1 = Diabetic, 0 = Non-diabetic)
    Imputes invalid 0 values in Glucose, BloodPressure, SkinThickness, Insulin, BMI with medians.
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, 'diabetes.csv')
    
    if not os.path.exists(filepath):
        url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
        print(f"Downloading Diabetes dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        
    df = pd.read_csv(filepath)
    
    # Preprocessing: Replace 0s with NaN for columns where 0 is physiologically impossible, then impute with median
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    return df

def load_breast_cancer_data():
    """
    Loads Breast Cancer Wisconsin (Diagnostic) Dataset from scikit-learn (UCI).
    Target: 1 = Malignant (Disease present), 0 = Benign (No malignancy).
    Note: sklearn default is 0=malignant, 1=benign, so we invert target for consistency.
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, 'breast_cancer.csv')
    
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    
    # Sklearn target: 0 = malignant, 1 = benign. Invert so 1 = Malignant (Cancer detected)
    df['target'] = 1 - df['target']
    
    # Save local copy for offline and reproducible inspection
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
        
    return df

def create_symptom_disease_data():
    """
    Generates a structured medical symptom-disease dataset based on clinical symptom profiles.
    Predicts: Common Cold, Influenza (Flu), COVID-19, Malaria, Dengue Fever, Typhoid, Pneumonia, Migraine, Gastroenteritis.
    Features: 20 clinical symptoms (binary 0/1) plus Age and Fever Temperature (°F).
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, 'symptoms_disease.csv')
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
        
    np.random.seed(42)
    symptoms_list = [
        'fever', 'high_fever', 'cough', 'fatigue', 'headache', 'chills', 
        'muscle_pain', 'sore_throat', 'shortness_of_breath', 'nausea',
        'vomiting', 'diarrhea', 'loss_of_taste_smell', 'skin_rash',
        'joint_pain', 'runny_nose', 'chest_pain', 'dizziness', 
        'sweating', 'loss_of_appetite'
    ]
    
    diseases = {
        'Common Cold': {
            'core': ['cough', 'sore_throat', 'runny_nose', 'fatigue'],
            'prob': {'fever': 0.3, 'headache': 0.4, 'muscle_pain': 0.2},
            'temp_mean': 99.2, 'temp_std': 0.6
        },
        'Influenza (Flu)': {
            'core': ['fever', 'high_fever', 'chills', 'muscle_pain', 'fatigue', 'headache', 'cough'],
            'prob': {'sore_throat': 0.6, 'sweating': 0.7, 'loss_of_appetite': 0.8},
            'temp_mean': 102.1, 'temp_std': 0.9
        },
        'COVID-19': {
            'core': ['fever', 'cough', 'fatigue', 'loss_of_taste_smell', 'shortness_of_breath'],
            'prob': {'headache': 0.7, 'sore_throat': 0.5, 'muscle_pain': 0.6, 'chest_pain': 0.4},
            'temp_mean': 101.3, 'temp_std': 1.0
        },
        'Malaria': {
            'core': ['high_fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting', 'muscle_pain'],
            'prob': {'fatigue': 0.9, 'diarrhea': 0.3, 'loss_of_appetite': 0.7},
            'temp_mean': 103.5, 'temp_std': 1.1
        },
        'Dengue Fever': {
            'core': ['high_fever', 'joint_pain', 'muscle_pain', 'skin_rash', 'headache', 'fatigue'],
            'prob': {'nausea': 0.8, 'vomiting': 0.6, 'chills': 0.7, 'loss_of_appetite': 0.8},
            'temp_mean': 103.8, 'temp_std': 0.9
        },
        'Typhoid': {
            'core': ['high_fever', 'headache', 'fatigue', 'diarrhea', 'nausea', 'loss_of_appetite'],
            'prob': {'vomiting': 0.6, 'sweating': 0.8, 'cough': 0.3, 'skin_rash': 0.4},
            'temp_mean': 102.8, 'temp_std': 0.8
        },
        'Pneumonia': {
            'core': ['cough', 'shortness_of_breath', 'chest_pain', 'fever', 'fatigue', 'chills'],
            'prob': {'sweating': 0.8, 'muscle_pain': 0.5, 'headache': 0.5, 'nausea': 0.4},
            'temp_mean': 102.0, 'temp_std': 1.0
        },
        'Gastroenteritis': {
            'core': ['nausea', 'vomiting', 'diarrhea', 'fatigue', 'loss_of_appetite'],
            'prob': {'fever': 0.5, 'headache': 0.4, 'muscle_pain': 0.3, 'dizziness': 0.6},
            'temp_mean': 100.1, 'temp_std': 0.8
        }
    }
    
    rows = []
    # Generate 150 realistic cases per disease = 1200 patients
    for disease_name, profile in diseases.items():
        for _ in range(150):
            row = {}
            age = int(np.clip(np.random.normal(38, 18), 5, 85))
            temp = round(float(np.clip(np.random.normal(profile['temp_mean'], profile['temp_std']), 97.5, 105.5)), 1)
            row['age'] = age
            row['temperature'] = temp
            
            for s in symptoms_list:
                if s in profile['core']:
                    row[s] = 1 if np.random.rand() < 0.92 else 0
                elif s in profile.get('prob', {}):
                    row[s] = 1 if np.random.rand() < profile['prob'][s] else 0
                else:
                    # Rare symptom
                    row[s] = 1 if np.random.rand() < 0.06 else 0
                    
            row['disease'] = disease_name
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Generated symptom dataset with {len(df)} samples across {len(diseases)} conditions.")
    return df

if __name__ == '__main__':
    print("Pre-fetching all datasets...")
    h = load_heart_disease_data()
    print("Heart:", h.shape)
    d = load_diabetes_data()
    print("Diabetes:", d.shape)
    b = load_breast_cancer_data()
    print("Breast Cancer:", b.shape)
    s = create_symptom_disease_data()
    print("Symptoms:", s.shape)
    print("All datasets loaded successfully!")
