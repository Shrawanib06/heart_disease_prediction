import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, cohen_kappa_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

# Step 1: Load your dataset
df = pd.read_csv('notebooks/cleaned_heart_disease.csv')

# Step 2: Drop irrelevant columns
df = df.drop(['dataset_Hungary', 'dataset_Switzerland', 'dataset_VA Long Beach'], axis=1)

# Step 3: Preprocess data
def preprocess_data(df):
    df = df.dropna(axis=1, how='all')  # Drop columns with all missing values
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')  
    transformed_values = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = transformed_values
    
    return df

# Preprocess the data
df = preprocess_data(df)

# Step 4: Define features and target variable
X = df.drop('num', axis=1)  # 'num' is your target column
y = df['num']

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 8: RandomizedSearchCV for Random Forest
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)

# Step 9: Evaluate the best Random Forest model
best_rf_model = random_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Best Parameters for RF: {random_search.best_params_}")
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print("\nClassification Report (RF):\n", classification_report(y_test, y_pred_rf, zero_division=1))

# Step 10: Train and evaluate Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb * 100:.2f}%")

# Step 11: Ensemble Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('gb', gb_model),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
], voting='soft')

voting_clf.fit(X_train, y_train)
y_pred_ensemble = voting_clf.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble * 100:.2f}%")
print("\nClassification Report (Ensemble):\n", classification_report(y_test, y_pred_ensemble, zero_division=1))

# AUC-ROC Score
y_proba_ensemble = voting_clf.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba_ensemble, multi_class='ovr')
print(f"AUC-ROC (Ensemble): {roc_auc:.2f}")

# Step 12: Feature Importance (Random Forest)
importances = best_rf_model.feature_importances_
features = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), features[sorted_indices], rotation=90)
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()

# Step 13: Save models and scaler
joblib.dump(best_rf_model, 'app/best_random_forest_model.pkl')
joblib.dump(voting_clf, 'app/ensemble_model.pkl')
joblib.dump(scaler, 'app/scaler.pkl')


