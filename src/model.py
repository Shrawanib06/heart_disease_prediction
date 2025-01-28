import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load models and scaler
ensemble_model = joblib.load('app/ensemble_model.pkl')
scaler = joblib.load('app/scaler.pkl')

# Preprocess input data
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    df_scaled = scaler.transform(df_imputed)
    return df_scaled

# Predict function
def predict_heart_disease(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = ensemble_model.predict(preprocessed_data)
    probabilities = ensemble_model.predict_proba(preprocessed_data)
    return int(prediction[0]), probabilities

