import streamlit as st
import joblib 
import numpy as np

model=joblib.load('model.pkl')
scalar=joblib.load('scalar.pkl')
encoders=joblib.load('encoders.pkl')

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

Partner = st.selectbox("Partner", ["Yes", "No"])

Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure")

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges")

TotalCharges = st.number_input("Total Charges")


if st.button("Predict"):

    gender = encoders['gender'].transform([gender])[0]

    Partner = encoders['Partner'].transform([Partner])[0]

    Dependents = encoders['Dependents'].transform([Dependents])[0]

    PhoneService = encoders['PhoneService'].transform([PhoneService])[0]

    MultipleLines = encoders['MultipleLines'].transform([MultipleLines])[0]

    InternetService = encoders['InternetService'].transform([InternetService])[0]

    OnlineSecurity = encoders['OnlineSecurity'].transform([OnlineSecurity])[0]

    OnlineBackup = encoders['OnlineBackup'].transform([OnlineBackup])[0]

    DeviceProtection = encoders['DeviceProtection'].transform([DeviceProtection])[0]

    TechSupport = encoders['TechSupport'].transform([TechSupport])[0]

    StreamingTV = encoders['StreamingTV'].transform([StreamingTV])[0]

    StreamingMovies = encoders['StreamingMovies'].transform([StreamingMovies])[0]

    Contract = encoders['Contract'].transform([Contract])[0]

    PaperlessBilling = encoders['PaperlessBilling'].transform([PaperlessBilling])[0]

    PaymentMethod = encoders['PaymentMethod'].transform([PaymentMethod])[0]

    # Final input array

    input_data = np.array([[

        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges

    ]])

    input_data_scaled = scalar.transform(input_data)

    
    prediction=model.predict(input_data_scaled)

    if prediction[0]==1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")


