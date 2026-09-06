import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Prediction",page_icon="❤️",layout="centered")

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's information below to make a prediction.")

age = st.number_input(
"Age",
min_value=1,
max_value=120,
value=50
)

sex = st.selectbox(
"Sex",
["Female", "Male"]
)

chest_pain = st.selectbox(
"Chest Pain Type",
[
"Asymptomatic (ASY)",
"Atypical Angina (ATA)",
"Non-Anginal Pain (NAP)",
"Typical Angina (TA)"
]
)

resting_bp = st.number_input(
"Resting Blood Pressure",
min_value=0,
max_value=300,
value=120
)

cholesterol = st.number_input(
"Cholesterol",
min_value=0,
max_value=1000,
value=200
)

fasting_bs = st.selectbox(
"Fasting Blood Sugar",
[0, 1],
format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
)

resting_ecg = st.selectbox(
"Resting ECG",
[
"Left Ventricular Hypertrophy (LVH)",
"Normal",
"ST"
]
)

max_hr = st.number_input(
"Maximum Heart Rate",
min_value=50,
max_value=300,
value=150
)

exercise_angina = st.selectbox(
"Exercise-Induced Angina",
["No", "Yes"]
)

oldpeak = st.number_input(
"Oldpeak",
min_value=-10.0,
max_value=20.0,
value=0.0,
step=0.1
)

st_slope = st.selectbox(
"ST Slope",
["Down", "Flat", "Up"]
)

sex_value = 0 if sex == "Female" else 1

chest_pain_mapping = {
"Asymptomatic (ASY)": 0,
"Atypical Angina (ATA)": 1,
"Non-Anginal Pain (NAP)": 2,
"Typical Angina (TA)": 3
}

resting_ecg_mapping = {
"Left Ventricular Hypertrophy (LVH)": 0,
"Normal": 1,
"ST": 2
}

exercise_angina_value = 0 if exercise_angina == "No" else 1

st_slope_mapping = {
"Down": 0,
"Flat": 1,
"Up": 2
}

chest_pain_value = chest_pain_mapping[chest_pain]
resting_ecg_value = resting_ecg_mapping[resting_ecg]
st_slope_value = st_slope_mapping[st_slope]

if st.button("Predict", use_container_width=True):

    input_data = pd.DataFrame(
    [[
        age,
        sex_value,
        chest_pain_value,
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg_value,
        max_hr,
        exercise_angina_value,
        oldpeak,
        st_slope_value
    ]],
    columns=[
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope"
    ]
)

prediction = model.predict(input_data)[0]

if prediction == 1:
    st.error("Prediction: Heart Disease")
else:
    st.success("Prediction: Normal")

# Prediction probabilities
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(input_data)[0]

    st.write(
        f"Probability of Heart Disease: "
        f"{probabilities[1] * 100:.2f}%"
    )

    st.write(
        f"Probability of Normal: "
        f"{probabilities[0] * 100:.2f}%"
    )
