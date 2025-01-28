# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from joblib import load
# from flask_cors import CORS

# # Initialize Flask app and enable CORS
# app = Flask(__name__)
# CORS(app)

# # Load the trained model and scaler
# model = load('app/best_random_forest_model.pkl')
# scaler = load('app/scaler.pkl')

# # Home route that serves the HTML form
# @app.route('/')
# def home():
#     return render_template('index.html')  # Make sure the file is in the 'templates' folder

# # Route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get data from the frontend
#         data = request.get_json()

#         # Define the features expected by the model
#         expected_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'slope', 'ca',
#                              'cp_atypical_angina', 'cp_non_anginal', 'cp_typical_angina', 'restecg_normal', 'restecg_st-t_abnormality', 
#                              'thal_normal', 'thal_reversable_defect']

#         # Create a feature vector for prediction
#         feature_vector = []
#         for feature in expected_features:
#             feature_vector.append(data.get(feature, 0))  # Default to 0 if feature is missing
        
#         # Convert the feature vector into a dataframe and scale it
#         feature_vector = np.array(feature_vector).reshape(1, -1)
#         feature_vector_scaled = scaler.transform(feature_vector)
        
#         # Predict using the model
#         prediction = model.predict(feature_vector_scaled)
#         probabilities = model.predict_proba(feature_vector_scaled)

#         # Return the prediction and probabilities
#         return jsonify({
#             'prediction': prediction[0],
#             'probabilities': probabilities.tolist()  # Convert probabilities to list for JSON response
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from flask_cors import CORS

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = load('app/best_random_forest_model.pkl')
scaler = load('app/scaler.pkl')

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

