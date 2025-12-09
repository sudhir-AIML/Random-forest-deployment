import streamlit as st
import requests

st.title("Iris Flower Prediction")

st.write("Please enter the dimensions of the Iris flower to predict its species.")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width", 0.1, 3.0, 0.2)

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            st.success(
                f"The predicted species is: {species_map.get(prediction, 'Unknown')}"
            )
        else:
            st.error("Error connecting to API")
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
