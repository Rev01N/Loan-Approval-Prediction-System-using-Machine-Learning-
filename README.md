# Loan Approval Prediction System

A professional Machine Learning system designed to automate loan eligibility assessments using applicant profiles.

## Features
- **Data-Driven Predictions**: Uses a trained Logistic Regression model for high-accuracy classification.
- **Real-Time Analysis**: Interactive Streamlit dashboard for instant loan status results.
- **Risk Assessment**: Identifies key risk factors (e.g., poor credit history, low income).
- **Rich Visualizations**: Includes a detailed EDA notebook for business insights.

## Technology Stack
- **Language**: Python 3.x
- **Libraries**: Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib
- **Web App**: Streamlit
- **Serialization**: Joblib

## Project Structure
- `data/`: Contains the raw `train.csv` dataset.
- `models/`: Stores the trained `.pkl` model and preprocessor.
- `notebooks/`: EDA and model experimentation.
- `src/`: Modular Python code for preprocessing and training.
- `app.py`: The main interactive dashboard.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model** (Optional, if `.pkl` already exists):
   ```bash
   python src/train.py
   ```
3. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Dataset Features
- `Loan_ID`: Unique Identifier
- `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
- `ApplicantIncome`, `CoapplicantIncome`
- `LoanAmount`, `Loan_Amount_Term`
- `Credit_History` (0/1)
- `Property_Area` (Urban, Semiurban, Rural)
- `Loan_Status` (Target: Y/N)
