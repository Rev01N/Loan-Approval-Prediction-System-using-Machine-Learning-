import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class LoanPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        self.categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        self.target_col = 'Loan_Status'

    def fit_transform(self, df):
        df = df.copy()
        
        # 1. Handle Missing Values
        for col in self.categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        
        for col in ['LoanAmount', 'Loan_Amount_Term']:
            df[col] = df[col].fillna(df[col].median())

        # 2. Encoding Categorical Variables
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        if self.target_col in df.columns:
            le_target = LabelEncoder()
            df[self.target_col] = le_target.fit_transform(df[self.target_col].astype(str))
            self.label_encoders[self.target_col] = le_target

        # 3. Scaling Numerical Data
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        
        return df

    def transform(self, df):
        df = df.copy()
        
        # Handle Missing Values (using training set modes/medians would be better, but for simplicity here we fill with common defaults if not provided)
        # In a real app, the input form ensures no NaNs.
        for col in self.categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        df['Credit_History'] = df['Credit_History'].fillna(1.0) # Assume good credit if missing in prediction
        
        for col in ['LoanAmount', 'Loan_Amount_Term']:
            df[col] = df[col].fillna(0)

        # Encoding
        for col in self.categorical_cols:
            if col in self.label_encoders:
                # Handle unseen labels by assigning a default or the first class
                le = self.label_encoders[col]
                df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

        # Scaling
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        
        return df
