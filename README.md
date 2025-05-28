# ❤️ Heart Disease Prediction Web Application

A web-based application that predicts the likelihood of heart disease using machine learning models. Built with Python, Flask, and trained using a heart disease dataset. This project includes model training, preprocessing, frontend UI, and backend integration.

---

## 🚀 Features

- Predicts heart disease risk based on user inputs
- Pre-trained machine learning model (Random Forest & Ensemble)
- Clean and responsive web interface
- Data preprocessing and feature scaling
- Deployment-ready (includes `wsgi.py` and `vercel.json`)

---

## 🧠 Technologies Used

- Python
- Flask
- HTML, CSS, JavaScript
- Scikit-learn, Pandas, NumPy
- Jupyter Notebook (for EDA)
- Git & GitHub
- Vercel (for deployment)

---

## 📂 Project Structure

heart-disease-prediction/

├── app/

│ ├── app.py # Flask backend

│ ├── data_preprocessing.py # Data preprocessing functions

│ ├── scaler.pkl # Pre-trained feature scaler

│ ├── best_random_forest_model.pkl

│ ├── ensemble_model.pkl

│ ├── static/ # CSS, JS, Images

│ └── templates/

│ └── index.html # Main web page

│

├── src/

│ ├── train_model.py # ML training script

│ └── model.py # Model structure

│

├── notebooks/

│ ├── EDA_notebook.ipynb # Data exploration

│ └── cleaned_heart_disease.csv

│

├── vercel.json # Vercel deployment config

├── requirements.txt # Required Python packages

└── README.md


---

## 🧪 How It Works

1. **User inputs** data on the web interface.
2. Data is sent to the Flask backend (`app.py`).
3. **Data Preprocessing**:
   - Categorical variables are converted to numeric
   - Features are scaled using `scaler.pkl`
   - Structured to match model format
4. The **trained model** (`ensemble_model.pkl`) makes a prediction.
5. Prediction is sent back and displayed on the web page.

---

## 📊 Sample Inputs

- Age
- Gender
- Chest Pain Type
- Cholesterol
- Resting BP
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST depression)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Shrawanib06/heart-disease-prediction.git
cd heart-disease-prediction

### 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Application

python app/app.py

## Deployment

This app is ready to deploy with:

Vercel using vercel.json

## 📌 Credits
Dataset: UCI Heart Disease Dataset

Created by: Shrawani Bhambare
