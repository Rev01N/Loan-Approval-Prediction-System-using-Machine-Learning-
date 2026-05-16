import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ensure src is in the path so pickle can find the 'preprocessing' module
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing import LoanPreprocessor

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-top: 20px;
    }
    .approved {
        background-color: #28a745;
    }
    .rejected {
        background-color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'loan_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model_data = load_model()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Loan Approval System")
    st.markdown("---")
    st.info("""
    **Objective:**
    Automate the loan eligibility process based on applicant details.
    
    **Tech Stack:**
    - Random Forest / Logistic Regression
    - Scikit-learn
    - Streamlit
    """)
    
    if model_data:
        st.success("✅ Model Loaded Successfully")
    else:
        st.warning("⚠️ Model not found. Please run the training script.")

# Main Page
st.title("💰 Loan Approval Prediction System")
st.write("Assess the eligibility of a loan applicant in real-time.")

if not model_data:
    st.error("Model file missing! Please run `python src/train.py` first.")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Applicant Details")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                
            with c2:
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                credit_history = st.selectbox("Credit History", ["Clear (1.0)", "Unclear (0.0)"])
                credit_val = 1.0 if "Clear" in credit_history else 0.0
                
            st.markdown("---")
            
            c3, c4 = st.columns(2)
            with c3:
                applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
                coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=500)
            
            with c4:
                loan_amount = st.number_input("Loan Amount ($k)", min_value=0, value=150, step=10)
                loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=480, value=360, step=12)
            
            submit = st.form_submit_button("Analyze Application")

    if submit:
        # Prepare Input Data
        input_dict = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_val,
            'Property_Area': property_area
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess
        preprocessor = model_data['preprocessor']
        processed_input = preprocessor.transform(input_df)
        
        # Match feature order
        features = model_data['features']
        processed_input = processed_input[features]
        
        # Predict
        prediction = model_data['model'].predict(processed_input)[0]
        probability = model_data['model'].predict_proba(processed_input)[0][1]
        
        with col2:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card approved">
                    <h2>APPLICATION APPROVED</h2>
                    <p>Probability: {probability:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-card rejected">
                    <h2>APPLICATION REJECTED</h2>
                    <p>Approval Probability: {probability:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.write("**Key Factors:**")
            st.progress(probability)
            st.caption("Approval Confidence Level")
            
            if credit_val == 0:
                st.error("Major Risk: Unclear Credit History detected.")
            if applicant_income < 3000:
                st.warning("Low Applicant Income observed.")

# Footer
st.markdown("---")
st.caption("Loan Approval Prediction System")
