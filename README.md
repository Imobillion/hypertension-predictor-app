# 🩺 Hypertension Risk Prediction App

This is a machine learning web application built with **Streamlit** that predicts the likelihood of an individual having hypertension based on various health and lifestyle inputs.

## 💡 Features

- Built with a **Neural Network** using scikit-learn (MLPClassifier)
- Accepts input like:
  - Age
  - Salt Intake
  - Stress Score
  - Blood Pressure History
  - Sleep Duration
  - BMI
  - Smoking Status
  - Exercise Level
  - Family History
  - Medication
- Displays prediction result with probability and health message
- Clean, modern, and responsive UI using Streamlit

## 📁 Files

| File                   | Purpose                            |
|------------------------|------------------------------------|
| `app.py`               | Main Streamlit app                 |
| `hypertension_model.pkl` | Trained ML model (MLPClassifier)  |
| `scaler.pkl`           | Scaler for feature normalization   |
| `features.pkl`         | Model input features (column list) |
| `hypertension_dataset.csv` | Original dataset (optional)    |

## 🚀 How to Run the App

1. Clone or download this repository
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```
    streamlit run app.py
    ```

## 📊 Model Performance

- Accuracy: ~89%
- Precision, Recall, F1-score reported in `train_model.py` script

## 📌 About the Dataset

This dataset includes lifestyle and medical features to train a classification model to predict hypertension risk. It includes numeric and categorical features and was preprocessed using one-hot encoding and standard scaling.

## 🤖 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Joblib

---

## 👤 Author

Created by [Your Name Here]  
Feel free to fork, improve, and share!
