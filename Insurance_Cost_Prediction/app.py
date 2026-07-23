
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Insurance Cost Prediction")

st.write("Enter the details below")

age = st.number_input("Age", 18, 100, 25)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    10.0,
    60.0,
    25.0
)

children = st.number_input(
    "Children",
    0,
    10,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

if st.button("Predict"):

    data = pd.DataFrame({

        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]

    })

    prediction = model.predict(data)

    st.success(
        f"Predicted Insurance Cost : ${prediction[0]:,.2f}"
    )
    