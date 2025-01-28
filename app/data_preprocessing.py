import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Function to preprocess the data
def preprocess_data(df):
    label_encoder = LabelEncoder()

    # Encode categorical variables if they exist
    if 'sex' in df.columns:
        df['sex'] = label_encoder.fit_transform(df['sex'])

    # One-hot encoding for categorical variables
    if 'cp' in df.columns:
        cp_encoded = pd.get_dummies(df['cp'], prefix='cp')
        df = pd.concat([df, cp_encoded], axis=1)
        df.drop('cp', axis=1, inplace=True)

    if 'restecg' in df.columns:
        restecg_encoded = pd.get_dummies(df['restecg'], prefix='restecg')
        df = pd.concat([df, restecg_encoded], axis=1)
        df.drop('restecg', axis=1, inplace=True)

    if 'thal' in df.columns:
        thal_encoded = pd.get_dummies(df['thal'], prefix='thal')
        df = pd.concat([df, thal_encoded], axis=1)
        df.drop('thal', axis=1, inplace=True)

    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    return df

# Load the dataset
df = pd.read_csv('notebooks/cleaned_heart_disease.csv')

# Preprocess data
df = preprocess_data(df)

# Drop irrelevant columns that were not used during training
irrelevant_columns = ['dataset_Hungary', 'dataset_Switzerland', 'dataset_VA Long Beach']
df.drop(irrelevant_columns, axis=1, inplace=True)

# List of features expected by the model
expected_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'slope', 'ca',
                      'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina', 
                      'restecg_normal', 'restecg_st-t abnormality', 'thal_normal', 'thal_reversable defect']

# Ensure that the columns in the test set match those the model expects
X = df[expected_features]

# Load the pre-trained model
model = joblib.load('app/best_random_forest_model.pkl')

# Make predictions without passing feature names
predictions = model.predict(X.values)  # Use .values to pass data without feature names

# Output predictions
print(predictions)
