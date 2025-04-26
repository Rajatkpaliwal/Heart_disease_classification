import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

X_test, y_test = joblib.load('test_data.pkl')


# Load the trained model
model = joblib.load('gs_log_reg_model.pkl')

y_pred = model.predict(X_test)

# Create a classification report
report = classification_report(y_test, y_pred, output_dict=True)

st.title("❤️ Heart Disease Prediction App")
st.write("Provide the following details to predict the chance of heart disease.")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], 
                  format_func=lambda x: ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'][x])
trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=50, max_value=300, value=120)
chol = st.number_input('Serum Cholestoral (in mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'False' if x == 0 else 'True')
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], 
                       format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=6.0, step=0.1, value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2], 
                     format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia', [1, 3, 6, 7], 
                    format_func=lambda x: {1: 'Normal (1)', 3: 'Normal (3)', 6: 'Fixed Defect (6)', 7: 'Reversible Defect (7)'}[x])

# Predict button
if st.button('Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of heart disease! (Probability: {pred_proba[0][1]:.2f})")
    else:
        st.success(f"✅ Low risk of heart disease! (Probability: {pred_proba[0][0]:.2f})")

with st.expander("📊 Show Model Performance on Test Data"):
    st.subheader("Model Evaluation Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    }))