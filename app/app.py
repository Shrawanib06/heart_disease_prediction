import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from flask_cors import CORS
import gdown
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Google Drive file links
model_url = "https://drive.google.com/file/d/1yX7jbL70N8Bc28ZK5hrPNnVmnuuQW1SR/view?usp=sharing"
scaler_url = "https://drive.google.com/uc?id=1Ky6u3zBPJaysOnSM8OIDOCxpNboDN_vx"  # Updated link for scaler.pkl

# Paths for downloaded files
model_path = "best_random_forest_model.pkl"
scaler_path = "scaler.pkl"

# Function to download files if they don't exist
def download_files():
    if not os.path.exists(model_path):
        print("Downloading model...")
        gdown.download(model_url, model_path, quiet=False)
    if not os.path.exists(scaler_path):
        print("Downloading scaler...")
        gdown.download(scaler_url, scaler_path, quiet=False)

# Download files before loading the model
download_files()

# Load the trained model and scaler
model = load(model_path)
scaler = load(scaler_path)

# Home route that serves the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Make sure the file is in the 'templates' folder

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the frontend
        data = request.get_json()

        # Define the features expected by the model
        expected_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'slope', 'ca',
                             'cp_atypical_angina', 'cp_non_anginal', 'cp_typical_angina', 'restecg_normal', 'restecg_st-t_abnormality',
                             'thal_normal', 'thal_reversable_defect']

        # Create a feature vector for prediction
        feature_vector = []
        for feature in expected_features:
            feature_vector.append(data.get(feature, 0))  # Default to 0 if feature is missing

        # Convert the feature vector into a dataframe and scale it
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)

        # Predict using the model
        prediction = model.predict(feature_vector_scaled)
        probabilities = model.predict_proba(feature_vector_scaled)

        # Return the prediction and probabilities
        return jsonify({
            'prediction': prediction[0],
            'probabilities': probabilities.tolist()  # Convert probabilities to list for JSON response
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
