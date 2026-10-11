# 🩺 MediPredict AI: Disease Prediction from Medical Data

An end-to-end Machine Learning Clinical Decision Support System that predicts the possibility of major diseases using structured medical datasets, lab test results, vital signs, and patient symptoms.

---

## 🎯 Project Objectives & Overview
- **Objective:** Predict the onset and likelihood of diseases based on clinical biomarkers and patient symptoms.
- **Approach:** Classification using calibrated multi-model machine learning pipelines with 5-fold cross-validation.
- **Algorithms:** Support Vector Machine (SVM), Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost).
- **Datasets:** UCI Cleveland Heart Disease, Pima Indians Diabetes, UCI Wisconsin Breast Cancer Diagnostic, and Acute Symptom-to-Condition Matrix.

---

## 📊 Comparative Performance Leaderboard

All models were evaluated on held-out stratified test splits (20%) and verified with 5-Fold Stratified Cross-Validation:

| Disease Domain | Metric | SVM | Logistic Regression | Random Forest | XGBoost | Best Performing |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **❤️ Heart Disease** *(UCI Cleveland)* | **Test Accuracy**<br>**F1-Score**<br>**ROC-AUC**<br>**5-Fold CV** | 85.25%<br>0.8525<br>0.9470<br>82.22% | 86.89%<br>0.8690<br>0.9513<br>**82.64%** | **86.89%**<br>**0.8691**<br>**0.9589**<br>80.57% | 83.61%<br>0.8358<br>0.9351<br>78.10% | **Random Forest / Logistic Regression** |
| **🩸 Diabetes Mellitus** *(Pima Indians)* | **Test Accuracy**<br>**F1-Score**<br>**ROC-AUC**<br>**5-Fold CV** | 75.32%<br>0.7465<br>0.8019<br>77.03% | 70.78%<br>0.7008<br>0.8130<br>**78.66%** | 74.03%<br>0.7314<br>0.8170<br>76.87% | **75.32%**<br>**0.7496**<br>**0.8261**<br>74.43% | **XGBoost (AUC 0.826)** |
| **🔬 Breast Cancer** *(UCI Wisconsin)* | **Test Accuracy**<br>**F1-Score**<br>**ROC-AUC**<br>**5-Fold CV** | **96.49%**<br>**0.9647**<br>0.9831<br>93.63% | 92.98%<br>0.9301<br>**0.9841**<br>93.63% | 93.86%<br>0.9384<br>0.9835<br>**94.29%** | 94.74%<br>0.9474<br>0.9755<br>**94.29%** | **SVM (96.49%)** |
| **🌡️ Multi-Symptom Checker** *(8 Conditions)* | **Test Accuracy**<br>**F1-Score**<br>**ROC-AUC**<br>**5-Fold CV** | **96.67%**<br>**0.9667**<br>**0.9992**<br>96.98% | 96.25%<br>0.9626<br>0.9987<br>96.46% | **96.67%**<br>**0.9666**<br>0.9990<br>**98.02%** | 96.25%<br>0.9626<br>0.9986<br>96.04% | **Random Forest (CV 98.02%)** |

---

## 🗂️ Project Structure

```text
disease prediction/
├── app.py                  # Interactive Streamlit Clinical Web Dashboard
├── api.py                  # FastAPI REST API Microservice
├── cli_demo.py             # Command-Line Interface and benchmark demo
├── run_app.bat             # 1-Click Windows Launcher for Streamlit app
├── requirements.txt        # Python dependency manifest
├── data/                   # Medical datasets (CSV format)
│   ├── heart_disease.csv   # UCI Cleveland Heart Disease dataset
│   ├── diabetes.csv        # Pima Indians Diabetes dataset
│   ├── breast_cancer.csv   # UCI Wisconsin FNA Diagnostic dataset
│   └── symptoms_disease.csv# Acute clinical multi-symptom dataset
├── src/
│   ├── data_loader.py      # Automated download, validation & preprocessing
│   ├── model_trainer.py    # Training & cross-validation for all 4 algorithms
│   └── predictor.py        # Inference engine, risk gauges & clinical guidance
└── saved_models/           # Serialized models (.joblib), scalers, and metrics
    ├── heart/
    ├── diabetes/
    ├── breast_cancer/
    ├── symptoms/
    └── metrics.json
```

---

## 🚀 How to Run

### Option 1: Interactive Web Application (Streamlit)
Double-click `run_app.bat` or run in terminal:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Features include:
- **Interactive Vitals & Lab Sliders** with standard clinical reference ranges.
- **One-Click Patient Presets:** Test high-risk vs healthy patient profiles instantly.
- **Multi-Algorithm Consensus:** Compare SVM, Logistic Regression, Random Forest, and XGBoost predictions side-by-side.
- **Visual Health Risk Gauge:** Color-coded (Green = Low, Amber = Moderate, Red = High Risk).
- **Clinical Explanations:** Automated identification of primary risk factors (e.g. Hypertension, Hyperglycemia, Microcalcification).
- **Interactive Benchmark Dashboard:** Cross-validation leaderboards, feature importances, and ROC metrics.

### Option 2: REST API (FastAPI)
Run the backend API server:
```bash
python api.py
```
Explore the interactive Swagger documentation at: `http://127.0.0.1:8000/docs`
Endpoints available:
- `POST /predict/heart`
- `POST /predict/diabetes`
- `POST /predict/breast-cancer`
- `POST /predict/symptoms`
- `GET /metrics`

### Option 3: Terminal CLI Demo
Run instant predictions and benchmark evaluations:
```bash
python cli_demo.py
```

### Option 4: Retrain All Models
To retrain from scratch and update `metrics.json`:
```bash
python src/model_trainer.py
```

---

## 🔬 Algorithms Description
1. **Support Vector Machines (SVM):**
   - Applies the Radial Basis Function (RBF) kernel mapping complex non-linear biomarker interactions into hyperplane margins.
2. **Logistic Regression:**
   - Provides transparent, L2-regularized log-odds coefficients for direct clinical interpretability.
3. **Random Forest Classifier:**
   - 100 decorrelated decision trees with bootstrap aggregation, resistant to outliers and non-linear physiological interactions.
4. **XGBoost (Extreme Gradient Boosting):**
   - Sequential gradient boosted decision trees minimizing regularized objective functions with fast convergence.
5. **Consensus Ensemble:**
   - Soft-voting probabilistic aggregation combining predictions from all four models to reduce variance and individual model bias.

---

## ⚖️ Disclaimer
*MediPredict AI is designed for research, screening, and educational purposes. It should not be used as a substitute for professional clinical medical judgment or formal laboratory diagnostic confirmation.*
