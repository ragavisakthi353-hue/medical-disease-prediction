"""
Disease Prediction from Medical Data - Clinical Intelligence System
Interactive Medical Web Application powered by Streamlit, Scikit-Learn, and XGBoost.
Supported Datasets:
1. Heart Disease (UCI Cleveland)
2. Diabetes (Pima Indians)
3. Breast Cancer (UCI Wisconsin FNA Diagnostic)
4. Multi-Condition Clinical Symptom Checker
Algorithms: SVM, Logistic Regression, Random Forest, XGBoost, and Consensus Ensemble.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import DiseasePredictor, load_all_metrics
from src.data_loader import (
    load_heart_disease_data,
    load_diabetes_data,
    load_breast_cancer_data,
    create_symptom_disease_data
)

# Page configuration
st.set_page_config(
    page_title="MediPredict AI | Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        color: #ffffff;
    }
    
    .main-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    
    .badge-svm { background: #e0e7ff; color: #3730a3; }
    .badge-lr { background: #fef3c7; color: #92400e; }
    .badge-rf { background: #dcfce7; color: #166534; }
    .badge-xgb { background: #ffedd5; color: #9a3412; }
    
    .risk-card {
        padding: 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
        color: #991b1b;
    }
    
    .risk-mod {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 6px solid #f59e0b;
        color: #92400e;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 6px solid #10b981;
        color: #166534;
    }
    
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
    }
    
    .info-card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to plot risk gauge
def create_risk_gauge(probability, risk_label):
    color = "#10b981" if probability < 0.35 else ("#f59e0b" if probability < 0.65 else "#ef4444")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Probability: {risk_label}", 'font': {'size': 18, 'color': '#1e293b'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 35], 'color': '#dcfce7'},
                {'range': [35, 65], 'color': '#fef3c7'},
                {'range': [65, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
    return fig

# Helper to plot model comparison bars for current prediction
def create_model_comparison_chart(model_probs):
    names = list(model_probs.keys())
    values = [val * 100 for val in model_probs.values()]
    colors = ['#6366f1', '#0ea5e9', '#10b981', '#f97316']
    
    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.1)', width=1))
    ))
    fig.update_layout(
        title="Cross-Algorithm Disease Probability Comparison",
        yaxis=dict(title="Disease Probability (%)", range=[0, 105]),
        xaxis=dict(title="Algorithm"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.8rem 0 1.2rem 0;">
    <h2 style="margin: 0; color: #0f766e; font-weight: 800;">🩺 MediPredict AI</h2>
    <p style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">Clinical Diagnostic Support System</p>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Select Medical Diagnostic Module:",
    [
        "❤️ Heart Disease Prediction",
        "🩸 Diabetes Risk Prediction",
        "🔬 Breast Cancer Diagnostic",
        "🌡️ Symptom & Fever Checker",
        "📊 Model Benchmarks & Comparison",
        "📚 Clinical Data & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.82rem; color: #64748b; line-height: 1.4;">
    <strong>Algorithms Trained:</strong><br>
    • Support Vector Machine (SVM)<br>
    • Logistic Regression<br>
    • Random Forest (Ensemble)<br>
    • XGBoost (Gradient Boosted Trees)<br><br>
    <strong>Benchmark Datasets:</strong><br>
    • UCI Heart Disease (Cleveland)<br>
    • Pima Indians Diabetes<br>
    • UCI Breast Cancer (Wisconsin FNA)<br>
    • Clinical Multi-Symptom Matrix
</div>
""", unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 class="main-title">Medical Disease Prediction System</h1>
            <p class="main-sub">Multi-Model AI Diagnosis from Structured Clinical Parameters & Lab Diagnostics</p>
        </div>
        <div style="text-align: right; display: none; @media(min-width: 768px){display: block;}">
            <span class="metric-badge badge-svm">SVM</span>
            <span class="metric-badge badge-lr">Logistic Regression</span>
            <span class="metric-badge badge-rf">Random Forest</span>
            <span class="metric-badge badge-xgb">XGBoost</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. HEART DISEASE MODULE
# ==========================================
if nav_choice == "❤️ Heart Disease Prediction":
    st.subheader("❤️ Cardiovascular Heart Disease Risk Assessment")
    st.markdown("Predict the presence of heart disease using 13 clinical biomarkers from the **UCI Cleveland Heart Disease** dataset.")
    
    col_settings, col_preset = st.columns([2, 1])
    with col_settings:
        selected_model = st.selectbox(
            "Select Inference Algorithm:",
            ["Random Forest", "XGBoost", "SVM", "Logistic Regression", "Ensemble Consensus"],
            index=0,
            key="heart_model_select"
        )
    with col_preset:
        preset = st.radio("Quick Patient Presets:", ["Custom Input", "Sample High-Risk Patient", "Sample Low-Risk Patient"], horizontal=True)

    # Defaults
    # Defaults aligned with official UCI Cleveland clinical values
    if preset == "Sample High-Risk Patient":
        def_age, def_sex, def_cp, def_trestbps, def_chol = 67, 1, 4, 160, 286
        def_fbs, def_restecg, def_thalach, def_exang = 0, 2, 108, 1
        def_oldpeak, def_slope, def_ca, def_thal = 2.6, 2, 3, 7
    elif preset == "Sample Low-Risk Patient":
        def_age, def_sex, def_cp, def_trestbps, def_chol = 41, 0, 2, 115, 172
        def_fbs, def_restecg, def_thalach, def_exang = 0, 0, 172, 0
        def_oldpeak, def_slope, def_ca, def_thal = 0.0, 1, 0, 3
    else:
        def_age, def_sex, def_cp, def_trestbps, def_chol = 54, 1, 3, 130, 240
        def_fbs, def_restecg, def_thalach, def_exang = 0, 0, 150, 0
        def_oldpeak, def_slope, def_ca, def_thal = 1.0, 1, 0, 3

    with st.expander("📝 Patient Clinical Vitals & Lab Measurements", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Patient Age (Years)", 25, 80, def_age)
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", index=0 if def_sex==1 else 1)
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical Angina (1)",
                    2: "Atypical Angina (2)",
                    3: "Non-anginal Pain (3)",
                    4: "Asymptomatic / Silent Ischemia (4)"
                }[x],
                index=[1, 2, 3, 4].index(def_cp)
            )
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, def_trestbps, help="Normal: <120 mm Hg. Stage 1 Hypertension: 130-139 mm Hg.")
            chol = st.slider("Serum Cholesterol (mg/dl)", 120, 564, def_chol, help="Desirable: <200 mg/dl. Borderline: 200-239 mg/dl. High: >=240 mg/dl.")
        
        with c2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "True (>120 mg/dl)" if x == 1 else "False (<=120 mg/dl)", index=def_fbs)
            restecg = st.selectbox(
                "Resting ECG (restecg)",
                options=[0, 1, 2],
                format_func=lambda x: {0: "Normal (0)", 1: "ST-T Wave Abnormality (1)", 2: "Left Ventricular Hypertrophy (2)"}[x],
                index=def_restecg
            )
            thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, def_thalach, help="Peak heart rate during graded exercise test.")
            exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=def_exang)
            
        with c3:
            oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, float(def_oldpeak), step=0.1, help="ST depression in mm induced by exercise relative to rest.")
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"}[x],
                index=[1, 2, 3].index(def_slope)
            )
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3], index=def_ca)
            thal = st.selectbox(
                "Thalassemia Blood Flow (thal)",
                options=[3, 6, 7],
                format_func=lambda x: {3: "Normal Flow (3)", 6: "Fixed Defect (6)", 7: "Reversable Defect (7)"}[x],
                index=[3, 6, 7].index(def_thal)
            )

    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    if st.button("🔍 Run Heart Disease Risk Assessment", type="primary", use_container_width=True):
        predictor = DiseasePredictor('heart')
        res = predictor.predict(input_data, selected_model)
        
        st.markdown("---")
        st.subheader("📋 Diagnostic Evaluation Results")
        
        col_res1, col_res2 = st.columns([1, 1.4])
        with col_res1:
            st.plotly_chart(create_risk_gauge(res['probability'], res['risk_level']), use_container_width=True)
            
            risk_class = "risk-high" if res['risk_level'] == "High Risk" else ("risk-mod" if res['risk_level'] == "Moderate Risk" else "risk-low")
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h3 style="margin: 0; font-size: 1.3rem;">Assessment: {res['risk_level']}</h3>
                <p style="margin: 6px 0 0 0; font-size: 0.95rem;">
                    <strong>Decision:</strong> {'Cardiovascular Disease Detected' if res['prediction'] == 1 else 'Negative / No Significant Disease'}<br>
                    <strong>Selected Algorithm:</strong> {selected_model}<br>
                    <strong>Estimated Probability:</strong> {res['probability']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.plotly_chart(create_model_comparison_chart(res['all_model_probabilities']), use_container_width=True)
            
        # Risk factors breakdown and recommendations
        c_rf, c_rec = st.columns(2)
        with c_rf:
            st.markdown("#### ⚠️ Key Clinical Risk Indicators")
            if res['risk_factors']:
                df_factors = pd.DataFrame(res['risk_factors'])[['feature', 'value', 'threshold', 'clinical_meaning']]
                st.dataframe(df_factors, use_container_width=True, hide_index=True)
            else:
                st.success("✅ All monitored laboratory biomarkers are within expected standard ranges.")
                
        with c_rec:
            st.markdown("#### 🩺 Clinical Action Plan & Guidance")
            for rec in res['recommendations']:
                st.info(f"• {rec}")

# ==========================================
# 2. DIABETES RISK MODULE
# ==========================================
elif nav_choice == "🩸 Diabetes Risk Prediction":
    st.subheader("🩸 Type 2 Diabetes Mellitus Risk Evaluation")
    st.markdown("Predict diabetes onset and insulin resistance from clinical biomarkers using the **Pima Indians Diabetes** dataset.")
    
    col_settings, col_preset = st.columns([2, 1])
    with col_settings:
        selected_model = st.selectbox(
            "Select Inference Algorithm:",
            ["XGBoost", "Random Forest", "SVM", "Logistic Regression", "Ensemble Consensus"],
            index=0,
            key="diab_model_select"
        )
    with col_preset:
        preset = st.radio("Quick Patient Presets:", ["Custom Input", "Sample High-Risk Patient", "Sample Low-Risk Patient"], horizontal=True, key="diab_preset")

    if preset == "Sample High-Risk Patient":
        def_preg, def_glu, def_bp, def_skin = 6, 175, 88, 38
        def_ins, def_bmi, def_ped, def_age = 220, 36.8, 0.85, 52
    elif preset == "Sample Low-Risk Patient":
        def_preg, def_glu, def_bp, def_skin = 1, 95, 70, 20
        def_ins, def_bmi, def_ped, def_age = 79, 22.4, 0.28, 26
    else:
        def_preg, def_glu, def_bp, def_skin = 2, 130, 76, 28
        def_ins, def_bmi, def_ped, def_age = 110, 28.5, 0.45, 38

    with st.expander("📝 Metabolic & Glycemic Laboratory Parameters", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.slider("Number of Pregnancies", 0, 17, def_preg)
            glucose = st.slider("Fasting / 2-hr Plasma Glucose (mg/dL)", 40, 200, def_glu, help="Normal fasting: 70-99 mg/dL. Pre-diabetes: 100-125 mg/dL. Diabetes: >=126 mg/dL.")
            blood_pressure = st.slider("Diastolic Blood Pressure (mm Hg)", 40, 130, def_bp, help="Normal diastolic: <80 mm Hg.")
            skin_thickness = st.slider("Triceps Skin Fold Thickness (mm)", 5, 99, def_skin, help="Marker of subcutaneous adipose tissue.")
            
        with c2:
            insulin = st.slider("2-Hour Serum Insulin (µU/mL)", 10, 846, def_ins, help="Normal fasting: 16-166 µU/mL.")
            bmi = st.slider("Body Mass Index (BMI kg/m²)", 15.0, 67.0, float(def_bmi), step=0.1, help="Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >=30.")
            diabetes_pedigree = st.slider("Diabetes Pedigree Function (Genetic Index)", 0.05, 2.50, float(def_ped), step=0.01, help="Score reflecting genetic risk based on family history.")
            age = st.slider("Patient Age (Years)", 18, 90, def_age)

    input_data = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree, 'Age': age
    }

    if st.button("🔍 Run Diabetes Risk Analysis", type="primary", use_container_width=True):
        predictor = DiseasePredictor('diabetes')
        res = predictor.predict(input_data, selected_model)
        
        st.markdown("---")
        st.subheader("📋 Diagnostic Evaluation Results")
        
        col_res1, col_res2 = st.columns([1, 1.4])
        with col_res1:
            st.plotly_chart(create_risk_gauge(res['probability'], res['risk_level']), use_container_width=True)
            risk_class = "risk-high" if res['risk_level'] == "High Risk" else ("risk-mod" if res['risk_level'] == "Moderate Risk" else "risk-low")
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h3 style="margin: 0; font-size: 1.3rem;">Assessment: {res['risk_level']}</h3>
                <p style="margin: 6px 0 0 0; font-size: 0.95rem;">
                    <strong>Diagnosis:</strong> {'Diabetes Mellitus Indicated' if res['prediction'] == 1 else 'Negative / Non-Diabetic'}<br>
                    <strong>Selected Algorithm:</strong> {selected_model}<br>
                    <strong>Estimated Probability:</strong> {res['probability']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.plotly_chart(create_model_comparison_chart(res['all_model_probabilities']), use_container_width=True)

        c_rf, c_rec = st.columns(2)
        with c_rf:
            st.markdown("#### ⚠️ Metabolic Risk Drivers")
            if res['risk_factors']:
                df_factors = pd.DataFrame(res['risk_factors'])[['feature', 'value', 'threshold', 'clinical_meaning']]
                st.dataframe(df_factors, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Glycemic indices and metabolic biomarkers are in the normal healthy range.")
        with c_rec:
            st.markdown("#### 🩺 Clinical Action Plan & Guidance")
            for rec in res['recommendations']:
                st.info(f"• {rec}")

# ==========================================
# 3. BREAST CANCER MODULE
# ==========================================
elif nav_choice == "🔬 Breast Cancer Diagnostic":
    st.subheader("🔬 Breast Cancer FNA Biopsy Cytological Diagnostic")
    st.markdown("Classify breast tumors as **Benign** or **Malignant** using nuclear morphological features from fine needle aspirate (FNA) digitized images (**UCI Wisconsin Diagnostic**).")
    
    col_settings, col_preset = st.columns([2, 1])
    with col_settings:
        selected_model = st.selectbox(
            "Select Inference Algorithm:",
            ["SVM", "XGBoost", "Random Forest", "Logistic Regression", "Ensemble Consensus"],
            index=0,
            key="cancer_model_select"
        )
    with col_preset:
        preset = st.radio("Biopsy Sample Presets:", ["Custom Biopsy", "Malignant Case Sample", "Benign Case Sample"], horizontal=True, key="cancer_preset")

    if preset == "Malignant Case Sample":
        def_r, def_tex, def_p, def_a = 18.25, 23.4, 122.8, 1040.0
        def_s, def_c, def_con, def_cp = 0.115, 0.223, 0.245, 0.128
        def_sym, def_fd = 0.225, 0.072
    elif preset == "Benign Case Sample":
        def_r, def_tex, def_p, def_a = 11.54, 14.2, 74.0, 412.0
        def_s, def_c, def_con, def_cp = 0.082, 0.048, 0.021, 0.015
        def_sym, def_fd = 0.165, 0.058
    else:
        def_r, def_tex, def_p, def_a = 14.12, 19.2, 91.9, 654.8
        def_s, def_c, def_con, def_cp = 0.096, 0.104, 0.088, 0.048
        def_sym, def_fd = 0.181, 0.062

    with st.expander("🔬 Core FNA Nuclear Morphometry Features", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            m_radius = st.slider("Mean Radius (mm)", 6.0, 28.0, float(def_r), step=0.1)
            m_texture = st.slider("Mean Texture (gray-scale)", 9.0, 40.0, float(def_tex), step=0.1)
            m_perimeter = st.slider("Mean Perimeter (mm)", 43.0, 190.0, float(def_p), step=0.1)
            m_area = st.slider("Mean Area (mm²)", 140.0, 2500.0, float(def_a), step=1.0)
            m_smooth = st.slider("Mean Smoothness (local variation)", 0.05, 0.17, float(def_s), step=0.001)
        with c2:
            m_compact = st.slider("Mean Compactness (perimeter²/area - 1.0)", 0.01, 0.35, float(def_c), step=0.001)
            m_concavity = st.slider("Mean Concavity (severity of concave portions)", 0.0, 0.43, float(def_con), step=0.001)
            m_concave_pts = st.slider("Mean Concave Points (number of concave portions)", 0.0, 0.21, float(def_cp), step=0.001)
            m_symmetry = st.slider("Mean Symmetry", 0.10, 0.31, float(def_sym), step=0.001)
            m_fractal = st.slider("Mean Fractal Dimension ('coastline approx' - 1)", 0.04, 0.10, float(def_fd), step=0.001)

    input_data = {
        'mean radius': m_radius, 'mean texture': m_texture, 'mean perimeter': m_perimeter,
        'mean area': m_area, 'mean smoothness': m_smooth, 'mean compactness': m_compact,
        'mean concavity': m_concavity, 'mean concave points': m_concave_pts,
        'mean symmetry': m_symmetry, 'mean fractal dimension': m_fractal
    }

    if st.button("🔬 Analyze Biopsy Cytology", type="primary", use_container_width=True):
        predictor = DiseasePredictor('breast_cancer')
        res = predictor.predict(input_data, selected_model)
        
        st.markdown("---")
        st.subheader("📋 Biopsy Pathology Assessment")
        
        col_res1, col_res2 = st.columns([1, 1.4])
        with col_res1:
            st.plotly_chart(create_risk_gauge(res['probability'], res['risk_level']), use_container_width=True)
            risk_class = "risk-high" if res['prediction'] == 1 else "risk-low"
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h3 style="margin: 0; font-size: 1.3rem;">Histological Call: {'Malignant Mass Detected' if res['prediction'] == 1 else 'Benign Lesion'}</h3>
                <p style="margin: 6px 0 0 0; font-size: 0.95rem;">
                    <strong>Status:</strong> {res['risk_level']}<br>
                    <strong>Selected Algorithm:</strong> {selected_model}<br>
                    <strong>Malignancy Probability:</strong> {res['probability']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.plotly_chart(create_model_comparison_chart(res['all_model_probabilities']), use_container_width=True)

        c_rf, c_rec = st.columns(2)
        with c_rf:
            st.markdown("#### ⚠️ Nuclear Morphometry Risk Contributors")
            if res['risk_factors']:
                df_factors = pd.DataFrame(res['risk_factors'])[['feature', 'value', 'threshold', 'clinical_meaning']]
                st.dataframe(df_factors, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Nuclear contours, borders, and area conform to non-malignant benign cytology.")
        with c_rec:
            st.markdown("#### 🩺 Clinical Action Plan & Guidance")
            for rec in res['recommendations']:
                st.info(f"• {rec}")

# ==========================================
# 4. MULTI-SYMPTOM CHECKER
# ==========================================
elif nav_choice == "🌡️ Symptom & Fever Checker":
    st.subheader("🌡️ Multi-Disease Symptom & Vitals Assessment")
    st.markdown("Evaluate acute patient symptoms and vitals across conditions including **Influenza, COVID-19, Malaria, Dengue Fever, Typhoid, Pneumonia, and Gastroenteritis**.")
    
    col_settings, _ = st.columns([2, 1])
    with col_settings:
        selected_model = st.selectbox(
            "Select Inference Algorithm:",
            ["Random Forest", "SVM", "Logistic Regression", "XGBoost", "Ensemble Consensus"],
            index=0,
            key="sym_model_select"
        )

    st.markdown("#### 1. Patient Demographics & Vitals")
    c_vit1, c_vit2 = st.columns(2)
    with c_vit1:
        age_sym = st.slider("Patient Age (Years)", 2, 95, 34, key="sym_age")
    with c_vit2:
        temp_sym = st.slider("Oral Temperature (°F)", 96.8, 105.0, 101.4, step=0.1, key="sym_temp")

    st.markdown("#### 2. Select Active Patient Symptoms")
    symptoms_dict = {
        "fever": "Fever",
        "high_fever": "High Grade Fever (>102°F)",
        "chills": "Chills / Rigors",
        "cough": "Persistent Cough",
        "sore_throat": "Sore Throat",
        "runny_nose": "Runny Nose / Congestion",
        "shortness_of_breath": "Shortness of Breath / Dyspnea",
        "chest_pain": "Chest Tightness / Pain",
        "fatigue": "General Fatigue / Lethargy",
        "headache": "Severe Headache",
        "muscle_pain": "Muscle Aches (Myalgia)",
        "joint_pain": "Severe Joint Pain (Arthralgia)",
        "skin_rash": "Skin Rash / Petechiae",
        "nausea": "Nausea",
        "vomiting": "Vomiting",
        "diarrhea": "Diarrhea",
        "loss_of_appetite": "Loss of Appetite",
        "loss_of_taste_smell": "Anosmia / Loss of Smell & Taste",
        "sweating": "Profuse Sweating",
        "dizziness": "Dizziness / Lightheadedness"
    }

    cols = st.columns(4)
    selected_symptoms = {}
    for i, (sym_key, sym_name) in enumerate(symptoms_dict.items()):
        col = cols[i % 4]
        with col:
            selected_symptoms[sym_key] = 1 if st.checkbox(sym_name, key=f"chk_{sym_key}") else 0

    input_data = {'age': age_sym, 'temperature': temp_sym, **selected_symptoms}

    if st.button("🌡️ Predict Disease from Symptoms", type="primary", use_container_width=True):
        predictor = DiseasePredictor('symptoms')
        res = predictor.predict(input_data, selected_model)
        
        st.markdown("---")
        st.subheader("📋 Clinical Diagnosis Inference")
        
        c_diag1, c_diag2 = st.columns([1, 1.3])
        with c_diag1:
            st.markdown(f"""
            <div class="risk-card risk-high">
                <h3 style="margin: 0; font-size: 1.4rem;">Top Match: {res['prediction']}</h3>
                <p style="margin: 8px 0 0 0; font-size: 0.95rem;">
                    <strong>Confidence:</strong> {res['probability']*100:.1f}%<br>
                    <strong>Reliability:</strong> {res['risk_level']}<br>
                    <strong>Classifier:</strong> {selected_model}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🩺 Clinical Advisory")
            for rec in res['recommendations']:
                st.info(rec)
                
        with c_diag2:
            # Distribution of disease probabilities for selected model
            probs_dict = res['all_model_probabilities'].get(selected_model, {})
            if isinstance(probs_dict, dict):
                sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                labels = [x[0] for x in sorted_probs]
                values = [x[1]*100 for x in sorted_probs]
                
                fig = px.bar(
                    x=values, y=labels, orientation='h',
                    labels={'x': 'Diagnostic Probability (%)', 'y': 'Condition'},
                    title=f"Differential Diagnosis Breakdown ({selected_model})",
                    color=values,
                    color_continuous_scale='Teal'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. BENCHMARKS & MODEL COMPARISON
# ==========================================
elif nav_choice == "📊 Model Benchmarks & Comparison":
    st.subheader("📊 Comparative Algorithm Benchmark Dashboard")
    st.markdown("Performance comparison across **SVM**, **Logistic Regression**, **Random Forest**, and **XGBoost** on medical classification tasks.")
    
    metrics = load_all_metrics()
    
    tab_heart, tab_diab, tab_cancer, tab_sym = st.tabs([
        "❤️ Heart Disease", "🩸 Diabetes Mellitus", "🔬 Breast Cancer", "🌡️ Symptom Classifier"
    ])
    
    def display_domain_benchmark(domain_key, domain_name):
        dom_metrics = metrics.get(domain_key, {})
        if not dom_metrics:
            st.warning("No metrics found for this domain.")
            return
            
        rows = []
        for model_name, m in dom_metrics.items():
            rows.append({
                'Algorithm': model_name,
                'Accuracy': f"{m['accuracy']*100:.2f}%",
                'Precision': f"{m['precision']*100:.2f}%",
                'Recall': f"{m['recall']*100:.2f}%",
                'F1-Score': f"{m['f1_score']:.4f}",
                'ROC-AUC': f"{m['roc_auc']:.4f}",
                '5-Fold CV Mean': f"{m['cv_mean']*100:.2f}%",
                'CV Std': f"±{m['cv_std']*100:.2f}%"
            })
        df_table = pd.DataFrame(rows)
        st.markdown(f"### Performance Leaderboard: {domain_name}")
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        # Plot comparative bar charts
        models = list(dom_metrics.keys())
        accs = [dom_metrics[m]['accuracy']*100 for m in models]
        f1s = [dom_metrics[m]['f1_score']*100 for m in models]
        aucs = [dom_metrics[m]['roc_auc']*100 for m in models]
        cvs = [dom_metrics[m]['cv_mean']*100 for m in models]
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy (%)', x=models, y=accs, marker_color='#3b82f6'),
            go.Bar(name='F1-Score (x100)', x=models, y=f1s, marker_color='#10b981'),
            go.Bar(name='ROC-AUC (x100)', x=models, y=aucs, marker_color='#f59e0b'),
            go.Bar(name='5-Fold CV (%)', x=models, y=cvs, marker_color='#8b5cf6')
        ])
        fig.update_layout(
            barmode='group',
            title=f"Multi-Metric Algorithm Comparison ({domain_name})",
            yaxis=dict(title="Score (%)", range=[50, 105]),
            height=360,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importances where available
        rf_importances = dom_metrics.get('Random Forest', {}).get('feature_importances', {})
        xgb_importances = dom_metrics.get('XGBoost', {}).get('feature_importances', {})
        if rf_importances:
            st.markdown("#### 🔍 Feature Importance Analysis (Random Forest vs XGBoost)")
            c1, c2 = st.columns(2)
            with c1:
                df_rf = pd.DataFrame(list(rf_importances.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig_rf = px.bar(df_rf, x='Importance', y='Feature', orientation='h', title="Random Forest Gini Importance", color='Importance', color_continuous_scale='Blues')
                fig_rf.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_rf, use_container_width=True)
            with c2:
                df_xgb = pd.DataFrame(list(xgb_importances.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig_xgb = px.bar(df_xgb, x='Importance', y='Feature', orientation='h', title="XGBoost Feature Weight", color='Importance', color_continuous_scale='Oranges')
                fig_xgb.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_xgb, use_container_width=True)

    with tab_heart:
        display_domain_benchmark('heart', 'Heart Disease')
    with tab_diab:
        display_domain_benchmark('diabetes', 'Diabetes Mellitus')
    with tab_cancer:
        display_domain_benchmark('breast_cancer', 'Breast Cancer Diagnostic')
    with tab_sym:
        display_domain_benchmark('symptoms', 'Symptom-to-Disease Multi-Class')

# ==========================================
# 6. METHODOLOGY & DATA
# ==========================================
elif nav_choice == "📚 Clinical Data & Methodology":
    st.subheader("📚 Datasets, Preprocessing, and Algorithmic Methodology")
    
    st.markdown("""
    ### 🔬 Machine Learning Architecture
    This clinical diagnostic system employs four complementary machine learning paradigms:
    
    1. **Support Vector Machines (SVM)**:
       - Uses the Radial Basis Function (RBF) kernel to map complex, non-linear clinical biomarker boundaries into high-dimensional space.
       - Calibrated probability estimates enable granular risk probability assessment.
       
    2. **Logistic Regression (L2 Regularized)**:
       - Baseline linear probabilistic classifier providing direct log-odds interpretability and clinical weight transparency.
       
    3. **Random Forest Classifier**:
       - Non-parametric ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging).
       - Robust against outliers, physiological non-linearities, and feature interactions.
       
    4. **Extreme Gradient Boosting (XGBoost)**:
       - State-of-the-art gradient boosted trees optimizing second-order Taylor expansion loss functions.
       - Outstanding predictive power across tabular medical datasets.
       
    5. **Consensus Ensemble**:
       - Unifies predictions across all four classifiers using soft-voting probability aggregation, minimizing individual model bias.
       
    ---
    ### 📁 Dataset Specifications
    | Dataset | Source | Samples | Key Clinical Features | Target Variable |
    | :--- | :--- | :--- | :--- | :--- |
    | **Heart Disease** | UCI Cleveland Clinic | 303 | Age, Chest Pain Type, Resting BP, Cholesterol, Max HR, ST Depression, Fluoroscopy Vessels | Angiographic Disease Status (0 or 1) |
    | **Diabetes** | Pima Indians (NIDDK) | 768 | Glucose, Insulin, BMI, Blood Pressure, Pedigree, Age | Diabetic Status (0 or 1) |
    | **Breast Cancer** | UCI Wisconsin (WDBC) | 569 | Radius, Perimeter, Area, Concavity, Compactness, Texture | Malignant vs. Benign |
    | **Acute Symptoms** | Clinical Symptom Matrix | 1,200 | 20 Symptoms + Body Temperature + Age | 8 Acute Conditions |
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem 0;">
    MediPredict AI Diagnostic Research Tool • Built for Medical Pair-Programming & Clinical Decision Support • 
    <em>For educational and screening purposes only. Always consult a qualified medical professional for diagnosis.</em>
</div>
""", unsafe_allow_html=True)
