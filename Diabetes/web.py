import streamlit as st
import pandas as pd
import pickle

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    # Page title
    st.title("Diabetes Prediction")

    # Sidebar inputs
    st.sidebar.title("Input Features")
    age = st.sidebar.number_input("Age", min_value=0, max_value=150, value=30)
    urea = st.sidebar.number_input("Urea", min_value=0.0, value=20.0)
    cr = st.sidebar.number_input("Creatinine (Cr)", min_value=0.0, value=0.7)
    hba1c = st.sidebar.number_input("HbA1c", min_value=0.0, value=5.0)
    chol = st.sidebar.number_input("Cholesterol", min_value=0.0, value=150.0)
    tg = st.sidebar.number_input("Triglycerides (TG)", min_value=0.0, value=100.0)
    hdl = st.sidebar.number_input("HDL Cholesterol", min_value=0.0, value=40.0)
    ldl = st.sidebar.number_input("LDL Cholesterol", min_value=0.0, value=100.0)
    vldl = st.sidebar.number_input("VLDL Cholesterol", min_value=0.0, value=20.0)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, value=25.0)

    input_data = [[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]]

    # Load the trained model
    model_path = 'diabetes.pkl'
    model = load_model(model_path)

    # Make predictions
    prediction = predict(model, input_data)

    # Display prediction result
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The patient is predicted to have diabetes.")
    else:
        st.write("The patient is predicted to be diabetes-free.")

if __name__ == "__main__":
    main()
