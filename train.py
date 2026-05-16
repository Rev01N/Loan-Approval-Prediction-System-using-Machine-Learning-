import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from preprocessing import LoanPreprocessor

def train_model():
    # Load data
    data_path = os.path.join('data', 'train.csv')
    df = pd.read_csv(data_path)
    
    # Preprocess
    preprocessor = LoanPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    # Drop Loan_ID as it's not a feature
    X = df_processed.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df_processed['Loan_Status']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True)
    }
    
    best_model = None
    best_accuracy = 0
    
    print("Model Evaluation:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # Final Evaluation Report
    y_pred_final = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final))
    
    # Save the model and preprocessor
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_data = {
        'model': best_model,
        'preprocessor': preprocessor,
        'features': X.columns.tolist()
    }
    
    joblib.dump(model_data, os.path.join(model_dir, 'loan_model.pkl'))
    print(f"Model saved to {os.path.join(model_dir, 'loan_model.pkl')}")

if __name__ == "__main__":
    train_model()
